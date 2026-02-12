#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class DatasetDef:
    key: str
    dir_name: str
    scenes: tuple[str, ...]


_DATASETS: dict[str, DatasetDef] = {
    "mipnerf360": DatasetDef(
        key="mipnerf360",
        dir_name="Mip-NeRF360",
        scenes=(
            "bicycle",
            "bonsai",
            "counter",
            "flowers",
            "garden",
            "kitchen",
            "room",
            "stump",
            "treehill",
        ),
    ),
    "tanksandtemples_vis": DatasetDef(
        key="tanksandtemples_vis",
        dir_name="Tanks&Temples-Vis",
        scenes=("train", "truck"),
    ),
    "deepblending": DatasetDef(
        key="deepblending",
        dir_name="DeepBlending",
        scenes=("drjohnson", "playroom"),
    ),
}

_DATASET_ALIASES: dict[str, str] = {
    "mip": "mipnerf360",
    "mipnerf360": "mipnerf360",
    "tnt": "tanksandtemples_vis",
    "tanks": "tanksandtemples_vis",
    "tanksandtemples": "tanksandtemples_vis",
    "tankstemples": "tanksandtemples_vis",
    "tanksandtemplesvis": "tanksandtemples_vis",
    "tankstemplesvis": "tanksandtemples_vis",
    "deep": "deepblending",
    "deepblending": "deepblending",
}

# Improved-GS scene budgets (see /home/joker/learning/Imporved-GS/budget.txt).
_IMPROVED_GS_BUDGETS_NORMAL: dict[str, int] = {
    "bicycle": 3_000_000,
    "flowers": 1_500_000,
    "garden": 3_000_000,
    "stump": 3_000_000,
    "treehill": 1_500_000,
    "bonsai": 1_000_000,
    "counter": 1_000_000,
    "kitchen": 1_000_000,
    "room": 1_000_000,
    "drjohnson": 1_500_000,
    "playroom": 1_000_000,
    "train": 1_000_000,
    "truck": 1_500_000,
}

_IMPROVED_GS_BUDGETS_SMALL: dict[str, int] = {
    "bicycle": 1_200_000,
    "flowers": 300_000,
    "garden": 1_200_000,
    "stump": 1_200_000,
    "treehill": 300_000,
    "bonsai": 300_000,
    "counter": 300_000,
    "kitchen": 300_000,
    "room": 300_000,
    "drjohnson": 600_000,
    "playroom": 300_000,
    "train": 300_000,
    "truck": 600_000,
}


def _split_csv(value: Optional[str]) -> list[str]:
    if value is None:
        return []
    items: list[str] = []
    for part in value.split(","):
        s = part.strip()
        if not s:
            continue
        items.append(s)
    return items


def _parse_scene_int_map(value: Optional[str]) -> dict[str, int]:
    """Parse `scene=int` CSV pairs (e.g. `garden=4,bicycle=2`)."""
    if value is None:
        return {}
    out: dict[str, int] = {}
    for part in _split_csv(value):
        if "=" not in part:
            raise ValueError(f"Invalid scene mapping entry {part!r}. Expected 'scene=int'.")
        scene, raw = part.split("=", 1)
        scene = scene.strip()
        if not scene:
            raise ValueError(f"Invalid scene mapping entry {part!r}: empty scene name.")
        out[scene] = int(raw)
    return out


def _normalize_dataset_key(name: str) -> str:
    raw = str(name).strip().lower().replace(" ", "")
    raw = raw.replace("_", "").replace("-", "").replace("&", "")
    if raw in _DATASET_ALIASES:
        return _DATASET_ALIASES[raw]
    raise KeyError(f"Unknown dataset {name!r}. Use --list to see available datasets.")


def _iter_selected_scenes(
    *,
    dataset: DatasetDef,
    include_scenes: set[str],
    exclude_scenes: set[str],
) -> Iterable[str]:
    for scene in dataset.scenes:
        if scene in exclude_scenes:
            continue
        if include_scenes and scene not in include_scenes:
            continue
        yield scene


def _ply_done_path(*, result_dir: Path, max_steps: int) -> Path:
    return result_dir / "ply" / f"splats_step{int(max_steps):06d}.ply"


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _infer_data_factor_from_scene_dir(scene_data_dir: Path) -> Optional[int]:
    """Infer `data_factor` from on-disk image folders.

    Preference order (highest resolution first):
    - images/ -> factor=1
    - images_2/ -> factor=2
    - images_4/ -> factor=4
    - images_8/ -> factor=8

    Returns None when no known image folder exists.
    """
    if (scene_data_dir / "images").exists():
        return 1
    for factor in (2, 4, 8):
        if (scene_data_dir / f"images_{factor}").exists():
            return factor
    return None


def _is_done(
    *,
    result_dir: Path,
    max_steps: int,
) -> bool:
    return _ply_done_path(result_dir=result_dir, max_steps=max_steps).exists()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_train_batch.py",
        description=(
            "Batch runner for FriendlySplat training (single GPU, sequential). "
            "Designed for benchmarking different densification strategies."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root directory (contains Mip-NeRF360/, Tanks&Temples-Vis/, DeepBlending/...).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device string to pass to trainer (e.g. cuda:0).",
    )
    parser.add_argument(
        "--data-preload",
        type=str,
        choices=("none", "cuda"),
        default="cuda",
        help=(
            "DataLoader preload mode for benchmarking. "
            "'cuda' preloads the entire dataset to GPU once (fast, uses VRAM). "
            "'none' loads on-demand on CPU."
        ),
    )
    parser.add_argument(
        "--scene-data-factor",
        type=str,
        action="append",
        default=None,
        help=(
            "Repeatable per-scene data_factor override: 'scene=int' or 'scene=int,scene=int'. "
            "Example: --scene-data-factor garden=4 --scene-data-factor bicycle=2"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30_000,
        help="Training steps for each scene (passed to --optim.max_steps).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=(
            "Comma-separated dataset keys to run: mipnerf360,tanksandtemples_vis,deepblending "
            "(or 'all'). Use --list to see details."
        ),
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Optional comma-separated scene names to include (filters within selected datasets).",
    )
    parser.add_argument(
        "--exclude-scenes",
        type=str,
        default=None,
        help="Optional comma-separated scene names to exclude.",
    )
    parser.add_argument(
        "--budget-profile",
        type=str,
        choices=("normal", "small", "none"),
        default="normal",
        help="Improved-GS per-scene budget table to use. 'none' disables budget overrides.",
    )
    parser.add_argument(
        "--strategy-impl",
        type=str,
        default="improved",
        choices=("improved", "default", "mcmc"),
        help="Strategy implementation to use for training.",
    )
    parser.add_argument(
        "--tb-every-n",
        type=int,
        default=100,
        help="TensorBoard logging cadence in training steps (1-based).",
    )
    parser.add_argument(
        "--tb-flush-every-n",
        type=int,
        default=500,
        help="TensorBoard flush cadence in logged training steps.",
    )
    parser.add_argument(
        "--no-mu",
        action="store_true",
        help="Disable MU schedule (only used when --strategy-impl improved).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available datasets/scenes and exit.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a scene if its output directory already exists (even if not done).",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to friendly_splat/trainer.py (use after '--').",
    )

    args = parser.parse_args(argv)

    if args.list:
        for key, d in _DATASETS.items():
            scenes = ", ".join(d.scenes)
            print(f"{key}: dir={d.dir_name} scenes=[{scenes}]")
        return 0

    include_datasets_raw = str(args.datasets).strip()
    if include_datasets_raw.lower() == "all":
        dataset_keys = list(_DATASETS.keys())
    else:
        dataset_keys = [_normalize_dataset_key(x) for x in _split_csv(include_datasets_raw)]

    include_scenes = set(_split_csv(args.scenes))
    exclude_scenes = set(_split_csv(args.exclude_scenes))
    scene_data_factors: dict[str, int] = {}
    for entry in args.scene_data_factor or []:
        scene_data_factors.update(_parse_scene_int_map(entry))

    budgets: dict[str, int] = {}
    if str(args.budget_profile) == "normal":
        budgets = dict(_IMPROVED_GS_BUDGETS_NORMAL)
    elif str(args.budget_profile) == "small":
        budgets = dict(_IMPROVED_GS_BUDGETS_SMALL)
    elif str(args.budget_profile) == "none":
        budgets = {}
    else:
        raise AssertionError(f"Unhandled budget_profile={args.budget_profile!r}")

    data_root = Path(str(args.data_root)).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[2]
    trainer_py = repo_root / "friendly_splat" / "trainer.py"
    if not trainer_py.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_py}")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    any_failed = False
    for dataset_key in dataset_keys:
        dataset = _DATASETS.get(dataset_key)
        if dataset is None:
            raise KeyError(f"Unknown dataset key {dataset_key!r}")

        for scene in _iter_selected_scenes(
            dataset=dataset, include_scenes=include_scenes, exclude_scenes=exclude_scenes
        ):
            dataset_dir = data_root / dataset.dir_name
            scene_data_dir = dataset_dir / scene
            if not scene_data_dir.exists():
                print(f"[skip] missing data_dir: {scene_data_dir}", flush=True)
                continue

            # Improved-GS-style layout: store outputs under the dataset directory.
            # Example: <data-root>/Mip-NeRF360/benchmark/improved/bicycle/
            scene_out_dir = dataset_dir / "benchmark" / str(args.strategy_impl) / scene

            log_path: Optional[Path] = None
            if not bool(args.dry_run):
                if _is_done(result_dir=scene_out_dir, max_steps=int(args.max_steps)):
                    print(f"[skip] done: {scene_out_dir}", flush=True)
                    continue
                if bool(args.skip_existing) and scene_out_dir.exists():
                    print(f"[skip] exists: {scene_out_dir}", flush=True)
                    continue

                scene_out_dir.mkdir(parents=True, exist_ok=True)
                log_dir = scene_out_dir / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                log_path = log_dir / "train.log"

            # Resolve data_factor with the following precedence:
            # 1) Explicit per-scene override (--scene-data-factor)
            # 2) Auto-infer from on-disk image folders
            if scene in scene_data_factors:
                data_factor = int(scene_data_factors[scene])
            else:
                inferred = _infer_data_factor_from_scene_dir(scene_data_dir)
                if inferred is None:
                    raise FileNotFoundError(
                        f"Could not infer data_factor from images/images_<factor> in {scene_data_dir}. "
                        "Use --scene-data-factor scene=int to set it explicitly."
                    )
                data_factor = int(inferred)
            if data_factor <= 0:
                raise ValueError(
                    f"data_factor must be > 0, got {data_factor} (scene={scene!r})."
                )

            cmd: list[str] = [
                sys.executable,
                str(trainer_py),
                "--io.data-dir",
                str(scene_data_dir),
                "--io.result-dir",
                str(scene_out_dir),
                "--io.device",
                str(args.device),
                "--data.preload",
                str(args.data_preload),
                "--data.data-factor",
                str(int(data_factor)),
                "--optim.max-steps",
                str(int(args.max_steps)),
                "--strategy.impl",
                str(args.strategy_impl),
                "--viewer.disable-viewer",
                "--data.benchmark-train-split",
            ]
            means_lr_final_override: Optional[str] = None

            # Enforce preload='cuda' constraints (see validate_train_config / DataLoader assertions).
            if str(args.data_preload) == "cuda":
                cmd += [
                    "--data.no-prefetch-to-gpu",
                    "--data.num-workers",
                    "0",
                ]

            if str(args.strategy_impl) == "improved":
                # Improved-GS alignment: match key benchmark switches + optimizer hyperparameters.
                cmd += ["--strategy.grow-grad2d", "0.0003"]
                cmd += ["--optim.no-random-bkgd"]
                if not bool(args.no_mu):
                    cmd += ["--optim.mu-enable"]
                cmd += [
                    "--optim.optimizers.means.optimizer.lr",
                    "4e-5",
                    "--optim.optimizers.opacities.optimizer.lr",
                    "0.025",
                    "--optim.optimizers.sh0.optimizer.lr",
                    "0.0025",
                    "--optim.optimizers.shN.optimizer.lr",
                    "0.00025",
                    "--optim.optimizers.scales.optimizer.lr",
                    "0.005",
                    "--optim.optimizers.quats.optimizer.lr",
                    "0.001",
                ]
                cmd += [
                    "--strategy.refine-scale2d-stop-iter",
                    "0",
                    "--strategy.prune-scale3d",
                    "999.0",
                ]
                means_lr_final_override = "2e-6"

            # Apply per-scene budget when requested and when the chosen strategy consumes it.
            if budgets:
                if scene not in budgets:
                    raise KeyError(
                        f"Budget missing for scene {scene!r} (budget_profile={args.budget_profile})."
                    )
                if str(args.strategy_impl) == "improved":
                    cmd += [
                        "--strategy.densification-budget",
                        str(int(budgets[scene])),
                    ]
                elif str(args.strategy_impl) == "mcmc":
                    cmd += [
                        "--strategy.mcmc-cap-max",
                        str(int(budgets[scene])),
                    ]

            cmd += [
                "--io.export-ply",
                "--io.ply-steps",
                str(int(args.max_steps)),
            ]

            cmd += ["--tb.enable"]
            cmd += ["--tb.every-n", str(int(args.tb_every_n))]
            cmd += ["--tb.flush-every-n", str(int(args.tb_flush_every_n))]

            if extra_args:
                cmd += extra_args

            if means_lr_final_override is not None:
                # Tyro models Optional[...] dataclasses as a subcommand. To override
                # `means.scheduler.lr-final`, we must select the scheduler branch and place the
                # scheduler override at the *very end* (otherwise later global flags would be
                # rejected as "misplaced").
                cmd += [
                    "optim.optimizers.means.scheduler:exponential-decay-scheduler-config",
                    "--optim.optimizers.means.scheduler.lr-final",
                    str(means_lr_final_override),
                ]

            tag = "dry-run" if bool(args.dry_run) else "run"
            print(f"[{tag}] {dataset.key}/{scene}", flush=True)
            print(f"[cmd] {_format_cmd(cmd)}", flush=True)
            if bool(args.dry_run):
                continue

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            if log_path is None:
                raise AssertionError("log_path is None in non-dry-run mode.")
            with open(log_path, "w", encoding="utf-8") as f:
                proc = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            if proc.returncode != 0:
                any_failed = True
                print(f"[fail] {dataset.key}/{scene} (see {log_path})", flush=True)
            else:
                print(f"[ok] {dataset.key}/{scene}", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
