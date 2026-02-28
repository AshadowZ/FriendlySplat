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

# Scene final budgets aligned with the upstream GNS repo's benchmark defaults.
# (See /home/joker/learning/GNS/test.py paramList comments.)
_GNS_FINAL_BUDGETS_NORMAL: dict[str, int] = {
    "bicycle": 600_000,
    "flowers": 600_000,
    "garden": 600_000,
    "stump": 600_000,
    "treehill": 600_000,
    "bonsai": 300_000,
    "counter": 300_000,
    "kitchen": 300_000,
    "room": 300_000,
    "drjohnson": 600_000,
    "playroom": 300_000,
    "train": 300_000,
    "truck": 600_000,
}

_PRUNER_ALIASES: dict[str, str] = {
    "all": "all",
    "pure_densify": "pure_densify",
    "pure-densify": "pure_densify",
    "dense": "pure_densify",
    "densify": "pure_densify",
    "densify_only": "pure_densify",
    "densifyonly": "pure_densify",
    "no_prune": "pure_densify",
    "noprune": "pure_densify",
    "baseline": "pure_densify",
    "gns": "gns",
    "natural_selection": "gns",
    "naturalselection": "gns",
    "speedy": "speedy",
    "hard_prune": "speedy",
    "hardprune": "speedy",
    "speedy_pruning": "speedy",
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
            raise ValueError(
                f"Invalid scene mapping entry {part!r}. Expected 'scene=int'."
            )
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
            "Batch runner for FriendlySplat pruning benchmarks (single GPU, sequential). "
            "Runs ImprovedStrategy densification to `densify_multiplier * final_budget`, "
            "then applies post-densification pruning (GNS or Speedy hard-prune)."
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
        "--pruners",
        type=str,
        default="all",
        help="Comma-separated methods to run: pure_densify,gns,speedy (or 'all').",
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
        "--final-budget",
        type=int,
        help=(
            "Optional final_budget override for all scenes. When omitted, uses the "
            "upstream GNS per-scene budget table (600k/300k defaults)."
        ),
        default=None,
    )
    parser.add_argument(
        "--scene-final-budget",
        type=str,
        action="append",
        default=None,
        help=(
            "Repeatable per-scene final_budget override: 'scene=int' or 'scene=int,scene=int'. "
            "Example: --scene-final-budget garden=1200000 --scene-final-budget bicycle=3000000"
        ),
    )
    parser.add_argument(
        "--densify-multiplier",
        type=int,
        default=3,
        help="Densify to `densify_multiplier * final_budget` before pruning.",
    )
    parser.add_argument(
        "--densify-stop-step",
        type=int,
        default=15_000,
        help=(
            "Last (1-based) training step with densification enabled. "
            "Passed to --strategy.refine-stop-iter (0-based boundary; numerically matches this step)."
        ),
    )
    parser.add_argument(
        "--gns-reg-end",
        type=int,
        default=23_000,
        help="GNS reg_end (1-based train step, inclusive).",
    )
    parser.add_argument(
        "--gns-opacity-reg-weight",
        type=float,
        default=2e-5,
        help="GNS base opacity regularization weight.",
    )
    parser.add_argument(
        "--speedy-every-n",
        type=int,
        default=2500,
        help="Speedy hard-prune cadence (every N 1-based steps).",
    )
    parser.add_argument(
        "--speedy-stop-step",
        type=int,
        default=25_000,
        help="Speedy hard-prune stop_step (1-based, inclusive).",
    )
    parser.add_argument(
        "--speedy-score-num-views",
        type=int,
        default=0,
        help=(
            "Speedy hard-prune scoring: use at most this many train views per pruning event. "
            "Use 0 to score using all train views (slow)."
        ),
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
        dataset_keys = [
            _normalize_dataset_key(x) for x in _split_csv(include_datasets_raw)
        ]

    include_scenes = set(_split_csv(args.scenes))
    exclude_scenes = set(_split_csv(args.exclude_scenes))
    scene_data_factors: dict[str, int] = {}
    for entry in args.scene_data_factor or []:
        scene_data_factors.update(_parse_scene_int_map(entry))

    pruners_raw = str(args.pruners).strip().lower()
    if pruners_raw == "all":
        pruners = ("pure_densify", "gns", "speedy")
    else:
        pruners_norm: list[str] = []
        for part in _split_csv(pruners_raw):
            k = part.strip().lower().replace("-", "_")
            k = _PRUNER_ALIASES.get(k)
            if k is None:
                raise KeyError(
                    f"Unknown pruner {part!r}. Expected one of: pure_densify,gns,speedy,all."
                )
            if k != "all" and k not in pruners_norm:
                pruners_norm.append(k)
        pruners = tuple(pruners_norm)

    scene_final_budgets: dict[str, int] = {}
    for entry in args.scene_final_budget or []:
        scene_final_budgets.update(_parse_scene_int_map(entry))

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
            dataset=dataset,
            include_scenes=include_scenes,
            exclude_scenes=exclude_scenes,
        ):
            dataset_dir = data_root / dataset.dir_name
            scene_data_dir = dataset_dir / scene
            if not scene_data_dir.exists():
                print(f"[skip] missing data_dir: {scene_data_dir}", flush=True)
                continue

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

            # Resolve final_budget with the following precedence:
            # 1) Explicit per-scene override (--scene-final-budget)
            # 2) Global override (--final-budget)
            # 3) Upstream GNS table (default)
            if scene in scene_final_budgets:
                final_budget = int(scene_final_budgets[scene])
            elif args.final_budget is not None:
                final_budget = int(args.final_budget)
            else:
                if scene not in _GNS_FINAL_BUDGETS_NORMAL:
                    raise KeyError(
                        f"GNS budget table missing for scene {scene!r}. "
                        "Use --final-budget or --scene-final-budget to override."
                    )
                final_budget = int(_GNS_FINAL_BUDGETS_NORMAL[scene])
            if final_budget <= 0:
                raise ValueError(f"final_budget must be > 0, got {final_budget}")

            densify_multiplier = int(args.densify_multiplier)
            if densify_multiplier <= 0:
                raise ValueError(
                    f"densify_multiplier must be > 0, got {densify_multiplier}"
                )
            densify_budget_for_pruning = int(final_budget) * int(densify_multiplier)
            densify_stop_step = int(args.densify_stop_step)
            if densify_stop_step <= 0:
                raise ValueError(
                    f"densify_stop_step must be > 0, got {densify_stop_step}"
                )

            # All outputs go under <data-root>/pruning_benchmark/...
            out_root = data_root / "pruning_benchmark"

            for pruner in pruners:
                scene_out_dir = out_root / dataset.key / scene / str(pruner)

                log_path: Optional[Path] = None
                if not bool(args.dry_run):
                    if _is_done(
                        result_dir=scene_out_dir, max_steps=int(args.max_steps)
                    ):
                        print(f"[skip] done: {scene_out_dir}", flush=True)
                        continue
                    if bool(args.skip_existing) and scene_out_dir.exists():
                        print(f"[skip] exists: {scene_out_dir}", flush=True)
                        continue

                    scene_out_dir.mkdir(parents=True, exist_ok=True)
                    log_dir = scene_out_dir / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    log_path = log_dir / "train.log"

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
                    "improved",
                    "--viewer.disable-viewer",
                    "--data.benchmark-train-split",
                ]

                # Enforce preload='cuda' constraints (see validate_train_config / DataLoader assertions).
                if str(args.data_preload) == "cuda":
                    cmd += [
                        "--data.no-prefetch-to-gpu",
                        "--data.num-workers",
                        "0",
                    ]

                # Improved-GS alignment (match common benchmark switches + optimizer hyperparameters).
                cmd += ["--strategy.grow-grad2d", "0.0002"]
                cmd += ["--optim.no-random-bkgd"]
                cmd += [
                    "--optim.optimizers.means.optimizer.lr",
                    "4e-5",
                    "--optim.optimizers.opacities.optimizer.lr",
                    "5e-2",
                    "--optim.optimizers.sh0.optimizer.lr",
                    "2.5e-3",
                    "--optim.optimizers.shN.optimizer.lr",
                    "1.25e-4",
                    "--optim.optimizers.scales.optimizer.lr",
                    "5e-3",
                    "--optim.optimizers.quats.optimizer.lr",
                    "1e-3",
                ]
                cmd += [
                    "--strategy.refine-scale2d-stop-iter",
                    "0",
                    "--strategy.prune-scale3d",
                    "999.0",
                ]

                # Densify to (multiplier * final_budget) and stop densification before pruning.
                cmd += [
                    "--strategy.refine-stop-iter",
                    str(int(densify_stop_step)),
                    "--strategy.densification-budget",
                    str(
                        int(final_budget)
                        if str(pruner) == "pure_densify"
                        else int(densify_budget_for_pruning)
                    ),
                ]

                # Post-densification pruning method.
                if str(pruner) == "pure_densify":
                    pass
                elif str(pruner) == "gns":
                    reg_start = int(densify_stop_step) + 1
                    reg_end = int(min(int(args.gns_reg_end), int(args.max_steps)))
                    if reg_end < reg_start:
                        reg_end = reg_start
                    cmd += [
                        "--gns.gns-enable",
                        "--gns.reg-start",
                        str(int(reg_start)),
                        "--gns.reg-end",
                        str(int(reg_end)),
                        "--gns.final-budget",
                        str(int(final_budget)),
                        "--gns.opacity-reg-weight",
                        str(float(args.gns_opacity_reg_weight)),
                    ]
                elif str(pruner) == "speedy":
                    start_step_1based = int(densify_stop_step) + 1
                    stop_step_1based = int(
                        min(int(args.speedy_stop_step), int(args.max_steps))
                    )
                    if stop_step_1based < start_step_1based:
                        stop_step_1based = int(start_step_1based)
                    cmd += [
                        "--hard-prune.enable",
                        "--hard-prune.start-step",
                        str(int(start_step_1based)),
                        "--hard-prune.stop-step",
                        str(int(stop_step_1based)),
                        "--hard-prune.every-n",
                        str(int(args.speedy_every_n)),
                        "--hard-prune.final-budget",
                        str(int(final_budget)),
                    ]
                    if int(args.speedy_score_num_views) > 0:
                        cmd += [
                            "--hard-prune.score-num-views",
                            str(int(args.speedy_score_num_views)),
                        ]
                else:
                    raise AssertionError(f"Unhandled pruner={pruner!r}")

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

                # Tyro models Optional[...] dataclasses as a subcommand. To override
                # `means.scheduler.lr-final`, we must select the scheduler branch and place the
                # scheduler override at the *very end* (otherwise later global flags would be rejected).
                cmd += [
                    "optim.optimizers.means.scheduler:exponential-decay-scheduler-config",
                    "--optim.optimizers.means.scheduler.lr-final",
                    "2e-6",
                ]

                tag = "dry-run" if bool(args.dry_run) else "run"
                print(
                    f"[{tag}] {dataset.key}/{scene} (pruner={pruner}) "
                    f"final_budget={final_budget} densify_budget="
                    f"{final_budget if str(pruner) == 'pure_densify' else densify_budget_for_pruning}",
                    flush=True,
                )
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
                    print(
                        f"[fail] {dataset.key}/{scene} (pruner={pruner}) (see {log_path})",
                        flush=True,
                    )
                else:
                    print(
                        f"[ok] {dataset.key}/{scene} (pruner={pruner})",
                        flush=True,
                    )

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
