from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import tyro

from friendly_splat.viewer.viewer_runtime import ViewerRuntime


@dataclass(frozen=True)
class ViewerScriptConfig:
    # Path to checkpoint file (.pt). If omitted, load from `result_dir/ckpts`.
    ckpt_path: Optional[str] = None
    # Training result directory that contains `ckpts/`.
    result_dir: str = "results"
    # Optional 1-based checkpoint step to load, e.g. 30000 -> ckpt_step030000.pt.
    step: Optional[int] = None
    # Device for rendering.
    device: str = "cuda"
    # Viewer server port.
    port: int = 8080


def _infer_result_dir_from_ckpt(ckpt_path: Path) -> Path:
    if ckpt_path.parent.name == "ckpts":
        return ckpt_path.parent.parent
    return ckpt_path.parent


def _find_checkpoint_from_result_dir(result_dir: Path, step: Optional[int]) -> Path:
    ckpt_dir = result_dir / "ckpts"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    if step is not None:
        step_i = int(step)
        if step_i <= 0:
            raise ValueError(f"`step` must be > 0, got {step_i}")
        ckpt_path = ckpt_dir / f"ckpt_step{step_i:06d}.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    candidates = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if len(candidates) == 0:
        raise FileNotFoundError(f"No checkpoints found under: {ckpt_dir}")
    return candidates[-1]


def _resolve_checkpoint(cfg: ViewerScriptConfig) -> tuple[Path, Path]:
    if cfg.ckpt_path is not None and str(cfg.ckpt_path).strip() != "":
        ckpt_path = Path(cfg.ckpt_path).expanduser().resolve()
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        result_dir = _infer_result_dir_from_ckpt(ckpt_path)
        return ckpt_path, result_dir

    result_dir = Path(cfg.result_dir).expanduser().resolve()
    ckpt_path = _find_checkpoint_from_result_dir(result_dir, cfg.step)
    return ckpt_path, result_dir


def _build_splats_from_state_dict(
    splat_state: dict[str, Any], device: torch.device
) -> torch.nn.ParameterDict:
    required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    missing = required - set(splat_state.keys())
    if len(missing) > 0:
        missing_keys = ", ".join(sorted(missing))
        raise KeyError(f"Checkpoint splats are missing keys: {missing_keys}")

    params: dict[str, torch.nn.Parameter] = {}
    for name, value in splat_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"splats[{name!r}] is not a tensor: {type(value)!r}")
        params[name] = torch.nn.Parameter(
            value.to(device=device),
            requires_grad=False,
        )
    return torch.nn.ParameterDict(params)


def _extract_render_flags(ckpt_obj: dict[str, Any]) -> tuple[bool, bool, bool]:
    packed = False
    sparse_grad = False
    absgrad = False

    cfg = ckpt_obj.get("cfg")
    if not isinstance(cfg, dict):
        return packed, sparse_grad, absgrad

    optim = cfg.get("optim")
    if isinstance(optim, dict):
        packed = bool(optim.get("packed", False))
        sparse_grad = bool(optim.get("sparse_grad", False))

    strategy = cfg.get("strategy")
    if isinstance(strategy, dict):
        absgrad = bool(strategy.get("absgrad", False))

    return packed, sparse_grad, absgrad


def main(cfg: ViewerScriptConfig) -> None:
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    ckpt_path, result_dir = _resolve_checkpoint(cfg)
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt_obj, dict):
        raise TypeError(f"Checkpoint content must be a dict, got {type(ckpt_obj)!r}")
    splat_state = ckpt_obj.get("splats")
    if not isinstance(splat_state, dict):
        raise KeyError("Checkpoint does not contain `splats` state dict.")

    splats = _build_splats_from_state_dict(splat_state, device)
    packed, sparse_grad, absgrad = _extract_render_flags(ckpt_obj)

    print(f"Loaded checkpoint: {ckpt_path}", flush=True)
    print(f"Loaded gaussians: {int(splats['means'].shape[0])}", flush=True)
    print(
        f"Render flags: packed={packed}, sparse_grad={sparse_grad}, absgrad={absgrad}",
        flush=True,
    )

    viewer_runtime = ViewerRuntime(
        disable_viewer=False,
        port=int(cfg.port),
        device=device,
        splats=splats,
        output_dir=result_dir,
        packed=packed,
        sparse_grad=sparse_grad,
        absgrad=absgrad,
    )
    viewer_runtime.keep_alive()


if __name__ == "__main__":
    main(tyro.cli(ViewerScriptConfig))
