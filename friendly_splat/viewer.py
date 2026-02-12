from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import tyro

from friendly_splat.data.colmap_dataparser import ColmapDataParser
from friendly_splat.data.dataset import InputDataset
from friendly_splat.modules.gaussian import GaussianModel
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


@dataclass(frozen=True)
class ViewerDatasetConfig:
    data_dir: str
    data_factor: int = 1
    normalize_world_space: bool = True
    align_world_axes: bool = True
    test_every: int = 8
    benchmark_train_split: bool = False
    depth_dir_name: Optional[str] = None
    normal_dir_name: Optional[str] = None
    dynamic_mask_dir_name: Optional[str] = None
    sky_mask_dir_name: Optional[str] = None


@dataclass(frozen=True)
class ViewerCkptSettings:
    packed: bool = False
    sparse_grad: bool = False
    absgrad: bool = False
    dataset_cfg: Optional[ViewerDatasetConfig] = None


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)
    if value is None:
        return bool(default)
    return bool(value)


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return None


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


def _build_gaussian_model_from_state_dict(
    splat_state: dict[str, Any], device: torch.device
) -> GaussianModel:
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
    return GaussianModel(params=params).to(device)


def _parse_ckpt_settings(ckpt_obj: dict[str, Any]) -> ViewerCkptSettings:
    cfg = ckpt_obj.get("cfg")
    if not isinstance(cfg, dict):
        return ViewerCkptSettings()

    optim = cfg.get("optim")
    packed = (
        _coerce_bool(optim.get("packed", False), False)
        if isinstance(optim, dict)
        else False
    )
    sparse_grad = (
        _coerce_bool(optim.get("sparse_grad", False), False)
        if isinstance(optim, dict)
        else False
    )
    strategy = cfg.get("strategy")
    absgrad = (
        _coerce_bool(strategy.get("absgrad", False), False)
        if isinstance(strategy, dict)
        else False
    )

    io_raw = cfg.get("io")
    if not isinstance(io_raw, dict):
        return ViewerCkptSettings(
            packed=packed,
            sparse_grad=sparse_grad,
            absgrad=absgrad,
            dataset_cfg=None,
        )
    data_dir = io_raw.get("data_dir")
    if not isinstance(data_dir, str) or len(data_dir.strip()) == 0:
        return ViewerCkptSettings(
            packed=packed,
            sparse_grad=sparse_grad,
            absgrad=absgrad,
            dataset_cfg=None,
        )

    data_raw = cfg.get("data")
    if not isinstance(data_raw, dict):
        data_raw = {}

    dataset_cfg = ViewerDatasetConfig(
        data_dir=data_dir,
        data_factor=_coerce_int(data_raw.get("data_factor", 1), 1),
        normalize_world_space=_coerce_bool(
            data_raw.get("normalize_world_space", True),
            True,
        ),
        # Backward compatibility: older configs did rotate normalization by default.
        align_world_axes=_coerce_bool(
            data_raw.get(
                "align_world_axes",
                data_raw.get("normalize_world_space_rotate", True),
            ),
            True,
        ),
        test_every=_coerce_int(data_raw.get("test_every", 8), 8),
        benchmark_train_split=_coerce_bool(
            data_raw.get("benchmark_train_split", False),
            False,
        ),
        depth_dir_name=_coerce_optional_str(data_raw.get("depth_dir_name")),
        normal_dir_name=_coerce_optional_str(data_raw.get("normal_dir_name")),
        dynamic_mask_dir_name=_coerce_optional_str(
            data_raw.get("dynamic_mask_dir_name")
        ),
        sky_mask_dir_name=_coerce_optional_str(data_raw.get("sky_mask_dir_name")),
    )
    return ViewerCkptSettings(
        packed=packed,
        sparse_grad=sparse_grad,
        absgrad=absgrad,
        dataset_cfg=dataset_cfg,
    )


def _load_checkpoint_safely(ckpt_path: Path) -> dict[str, Any]:
    # Prefer `weights_only=True` to avoid the FutureWarning and reduce pickle risk.
    try:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Backward compatibility for older torch versions without `weights_only`.
        ckpt_obj = torch.load(ckpt_path, map_location="cpu")

    if not isinstance(ckpt_obj, dict):
        raise TypeError(f"Checkpoint content must be a dict, got {type(ckpt_obj)!r}")
    return ckpt_obj


def _build_train_dataset_from_ckpt_settings(
    settings: ViewerCkptSettings,
) -> Optional[InputDataset]:
    dataset_cfg = settings.dataset_cfg
    if dataset_cfg is None:
        return None

    try:
        dataparser = ColmapDataParser(
            data_dir=dataset_cfg.data_dir,
            factor=int(dataset_cfg.data_factor),
            normalize_world_space=bool(dataset_cfg.normalize_world_space),
            align_world_axes=bool(dataset_cfg.align_world_axes),
            test_every=int(dataset_cfg.test_every),
            benchmark_train_split=bool(dataset_cfg.benchmark_train_split),
            depth_dir_name=dataset_cfg.depth_dir_name,
            normal_dir_name=dataset_cfg.normal_dir_name,
            dynamic_mask_dir_name=dataset_cfg.dynamic_mask_dir_name,
            sky_mask_dir_name=dataset_cfg.sky_mask_dir_name,
        )
        parsed_scene = dataparser.get_dataparser_outputs(split="train")
        return InputDataset(parsed_scene)
    except Exception as e:
        print(
            f"Skip train camera frustums: failed to build dataset from checkpoint cfg ({e}).",
            flush=True,
        )
        return None


def main(cfg: ViewerScriptConfig) -> None:
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    ckpt_path, result_dir = _resolve_checkpoint(cfg)
    ckpt_obj = _load_checkpoint_safely(ckpt_path)
    splat_state = ckpt_obj.get("splats")
    if not isinstance(splat_state, dict):
        raise KeyError("Checkpoint does not contain `splats` state dict.")

    gaussian_model = _build_gaussian_model_from_state_dict(splat_state, device)
    settings = _parse_ckpt_settings(ckpt_obj)
    train_dataset = _build_train_dataset_from_ckpt_settings(settings)

    print(f"Loaded checkpoint: {ckpt_path}", flush=True)
    print(f"Loaded gaussians: {int(gaussian_model.num_gaussians)}", flush=True)
    print(
        "Render flags: "
        f"packed={settings.packed}, "
        f"sparse_grad={settings.sparse_grad}, "
        f"absgrad={settings.absgrad}",
        flush=True,
    )
    if train_dataset is None:
        print("Train camera frustums: disabled (dataset unavailable).", flush=True)
    else:
        print(
            f"Train camera frustums: enabled ({len(train_dataset)} train images).",
            flush=True,
        )

    viewer_runtime = ViewerRuntime(
        disable_viewer=False,
        port=cfg.port,
        device=device,
        gaussian_model=gaussian_model,
        output_dir=result_dir,
        packed=settings.packed,
        sparse_grad=settings.sparse_grad,
        absgrad=settings.absgrad,
        train_dataset=train_dataset,
    )
    viewer_runtime.keep_alive()


if __name__ == "__main__":
    main(tyro.cli(ViewerScriptConfig))
