#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

import torch

# Allow running as a standalone script from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from friendly_splat.data.colmap_dataparser import ColmapDataParser
from friendly_splat.data.dataloader import DataLoader
from friendly_splat.data.dataset import InputDataset
from friendly_splat.modules.bilateral_grid import BilateralGridPostProcessor
from friendly_splat.modules.gaussian import GaussianModel
from friendly_splat.trainer.configs import EvalConfig, OptimConfig, StrategyConfig
from friendly_splat.trainer.eval_runtime import build_eval_summary, run_evaluation


@dataclass(frozen=True)
class _EvalCfg:
    eval: EvalConfig
    optim: OptimConfig
    strategy: StrategyConfig


def _load_ckpt(ckpt_path: Path) -> dict[str, Any]:
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Checkpoint content must be a dict, got {type(obj)!r}")
    return obj


def _find_ckpt(result_dir: Path, step: Optional[int]) -> Path:
    ckpt_dir = result_dir / "ckpts"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    if step is not None:
        step_i = int(step)
        if step_i <= 0:
            raise ValueError(f"--step must be > 0, got {step_i}")
        ckpt = ckpt_dir / f"ckpt_step{step_i:06d}.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt
    cands = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if len(cands) == 0:
        raise FileNotFoundError(f"No checkpoints found under: {ckpt_dir}")
    return cands[-1]


def _build_gaussian_model(splat_state: dict[str, Any], device: torch.device) -> GaussianModel:
    required = {"means", "scales", "quats", "opacities", "sh0", "shN"}
    missing = required - set(splat_state.keys())
    if len(missing) > 0:
        raise KeyError(f"Checkpoint splats missing keys: {sorted(missing)}")
    params: dict[str, torch.nn.Parameter] = {}
    for name, value in splat_state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"splats[{name!r}] is not a tensor: {type(value)!r}")
        params[name] = torch.nn.Parameter(value.to(device=device), requires_grad=False)
    model = GaussianModel(params=params).to(device)
    model.eval()
    return model


def _as_bool(x: Any, default: bool) -> bool:
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    return bool(x)


def _as_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="eval_single_scene_ckpt.py",
        description=(
            "Evaluate a single FriendlySplat result directory from checkpoint, "
            "and optionally compute color-corrected metrics (cc_*)."
        ),
    )
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--step", type=int, default=None, help="Checkpoint train step (1-based).")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--metrics-backend", type=str, default="inria", choices=("gsplat", "inria"))
    parser.add_argument("--lpips-net", type=str, default="vgg", choices=("alex", "vgg"))
    parser.add_argument("--compute-cc-metrics", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)

    result_dir = Path(str(args.result_dir)).expanduser().resolve()
    ckpt_path = _find_ckpt(result_dir=result_dir, step=args.step)
    ckpt = _load_ckpt(ckpt_path)
    cfg = ckpt.get("cfg") or {}
    if not isinstance(cfg, dict):
        cfg = {}

    io_cfg = cfg.get("io") if isinstance(cfg.get("io"), dict) else {}
    data_cfg = cfg.get("data") if isinstance(cfg.get("data"), dict) else {}
    optim_raw = cfg.get("optim") if isinstance(cfg.get("optim"), dict) else {}
    strategy_raw = cfg.get("strategy") if isinstance(cfg.get("strategy"), dict) else {}
    post_cfg = cfg.get("postprocess") if isinstance(cfg.get("postprocess"), dict) else {}

    data_dir_raw = args.data_dir if args.data_dir is not None else io_cfg.get("data_dir")
    if not isinstance(data_dir_raw, str) or len(data_dir_raw.strip()) == 0:
        raise ValueError("Missing data_dir. Provide --data-dir or ensure checkpoint cfg.io.data_dir exists.")

    data_dir = str(data_dir_raw)
    data_factor = _as_float(data_cfg.get("data_factor"), 1.0)
    normalize_world_space = _as_bool(data_cfg.get("normalize_world_space"), True)
    align_world_axes = _as_bool(
        data_cfg.get("align_world_axes", data_cfg.get("normalize_world_space_rotate")),
        True,
    )
    test_every = _as_int(data_cfg.get("test_every"), 8)
    benchmark_train_split = _as_bool(data_cfg.get("benchmark_train_split"), False)

    dataparser = ColmapDataParser(
        data_dir=data_dir,
        factor=float(data_factor),
        normalize_world_space=bool(normalize_world_space),
        align_world_axes=bool(align_world_axes),
        test_every=int(test_every),
        benchmark_train_split=bool(benchmark_train_split),
        depth_dir_name=None,
        normal_dir_name=None,
        dynamic_mask_dir_name=None,
        sky_mask_dir_name=None,
    )
    parsed_scene = dataparser.get_dataparser_outputs(split=str(args.split))
    eval_dataset = InputDataset(parsed_scene)

    device = torch.device(str(args.device))
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=0,
        device=device,
        infinite_sampler=False,
        prefetch_to_gpu=False,
        preload="cuda",  # type: ignore[arg-type]
        seed=42,
    )

    splat_state = ckpt.get("splats")
    if not isinstance(splat_state, dict):
        raise KeyError("Checkpoint does not contain a valid `splats` state dict.")
    gaussian_model = _build_gaussian_model(splat_state=splat_state, device=device)

    bilateral_grid: Optional[BilateralGridPostProcessor] = None
    if bool(args.compute_cc_metrics):
        bilagrid_state = ckpt.get("bilagrid")
        use_bilateral_grid = _as_bool(post_cfg.get("use_bilateral_grid"), False)
        if use_bilateral_grid and isinstance(bilagrid_state, dict):
            grid_shape_raw = post_cfg.get("bilateral_grid_shape", (16, 16, 8))
            if not isinstance(grid_shape_raw, (list, tuple)) or len(grid_shape_raw) != 3:
                raise ValueError(f"Invalid bilateral_grid_shape in checkpoint cfg: {grid_shape_raw!r}")
            grid_shape = (int(grid_shape_raw[0]), int(grid_shape_raw[1]), int(grid_shape_raw[2]))
            bilateral_grid = BilateralGridPostProcessor.create(
                num_frames=int(len(parsed_scene.image_names)),
                grid_shape=grid_shape,
                device=device,
            )
            bilateral_grid.bil_grids.load_state_dict(bilagrid_state, strict=True)
            bilateral_grid.eval()
        else:
            print(
                "[warn] compute_cc_metrics requested, but checkpoint has no usable bilateral grid state; "
                "running plain metrics only.",
                flush=True,
            )

    eval_cfg = replace(
        EvalConfig(),
        enable=True,
        split=str(args.split),
        eval_every_n=1,
        max_images=args.max_images,
        lpips_net=str(args.lpips_net),
        metrics_backend=str(args.metrics_backend),
        compute_cc_metrics=bool(args.compute_cc_metrics),
    )
    optim_cfg = replace(
        OptimConfig(),
        sh_degree=_as_int(optim_raw.get("sh_degree"), OptimConfig().sh_degree),
        sh_degree_interval=_as_int(
            optim_raw.get("sh_degree_interval"),
            OptimConfig().sh_degree_interval,
        ),
        packed=_as_bool(optim_raw.get("packed"), OptimConfig().packed),
        sparse_grad=_as_bool(optim_raw.get("sparse_grad"), OptimConfig().sparse_grad),
        antialiased=_as_bool(optim_raw.get("antialiased"), OptimConfig().antialiased),
    )
    strategy_cfg = replace(
        StrategyConfig(),
        absgrad=_as_bool(strategy_raw.get("absgrad"), StrategyConfig().absgrad),
    )

    step_index = _as_int(ckpt.get("step"), 0)
    eval_out = run_evaluation(
        cfg=_EvalCfg(eval=eval_cfg, optim=optim_cfg, strategy=strategy_cfg),  # type: ignore[arg-type]
        step=int(step_index),
        eval_loader=eval_loader,
        gaussian_model=gaussian_model,
        bilateral_grid=bilateral_grid,
    )
    stats = dict(eval_out.stats)
    print(build_eval_summary(eval_step=int(step_index), stats=stats), flush=True)

    out_dir = result_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_step = int(stats.get("train_step", int(step_index) + 1))
    out_path = out_dir / f"metrics_step{train_step:06d}.json"
    stats["checkpoint_path"] = str(ckpt_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[write] {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
