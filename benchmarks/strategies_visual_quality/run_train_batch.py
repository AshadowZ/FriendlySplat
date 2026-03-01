#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional


_SCENES_360V2: tuple[str, ...] = (
    "bicycle",
    "bonsai",
    "counter",
    "flowers",
    "garden",
    "kitchen",
    "room",
    "stump",
    "treehill",
)

def _read_flag_value(*, extra_args: list[str], flag: str):
    value = None
    for i, a in enumerate(extra_args):
        if a.startswith(f"{flag}="):
            value = a.split("=", 1)[1]
            continue
        if a == flag and i + 1 < len(extra_args):
            value = extra_args[i + 1]
            continue
    return value


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_train_batch.py",
        description=(
            "Batch runner for FriendlySplat training (single GPU, sequential). "
            "Designed for benchmarking different densification strategies on 360v2."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Dataset root directory. Expected layout: 360v2/<scene>/...",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="all",
        help=(
            "all|csv of scene names (e.g. bicycle,garden). "
            "If 'all', uses the built-in 360v2 scene list."
        ),
    )
    parser.add_argument(
        "--strategy-impl",
        type=str,
        default="default",
        choices=("all", "improved", "default", "mcmc"),
        help="Strategy implementation(s) to use (default: default).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a scene if its final PLY already exists (default: on).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra args forwarded to friendly_splat/trainer.py (use after '--'). "
            "These are appended after the script's default trainer hyperparameters."
        ),
    )

    args = parser.parse_args(argv)

    # Baseline benchmark settings (edit here, or override via extra_args).
    device = "cuda:0"
    data_preload = "cuda"

    # 360v2 presets for this benchmark:
    # - bicycle/flowers/garden/stump/treehill: 1/4 resolution (factor=4)
    # - bonsai/counter/kitchen/room: 1/2 resolution (factor=2)
    # - others: full resolution (factor=1)
    scene_data_factors: dict[str, int] = {
        "bicycle": 4,
        "flowers": 4,
        "garden": 4,
        "stump": 4,
        "treehill": 4,
        "bonsai": 2,
        "counter": 2,
        "kitchen": 2,
        "room": 2,
    }

    # Improved-GS scene budgets (see Improved-GS/budget.txt).
    improved_gs_budgets_normal: dict[str, int] = {
        "bicycle": 3_000_000,
        "flowers": 1_500_000,
        "garden": 3_000_000,
        "stump": 3_000_000,
        "treehill": 1_500_000,
        "bonsai": 1_000_000,
        "counter": 1_000_000,
        "kitchen": 1_000_000,
        "room": 1_000_000,
    }

    data_root = Path(str(args.data_root)).expanduser().resolve()
    dataset_root_dir = data_root / "360v2"
    if not dataset_root_dir.exists():
        raise FileNotFoundError(
            f"Missing 360v2 directory: {dataset_root_dir}. "
            "Expected <data-root>/360v2/<scene>/..."
        )
    if bool(args.verbose):
        print(f"[auto] using dataset dir: {dataset_root_dir}", flush=True)
    repo_root = Path(__file__).resolve().parents[2]
    trainer_py = repo_root / "friendly_splat" / "trainer.py"
    if not trainer_py.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_py}")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    max_steps_override = _read_flag_value(extra_args=extra_args, flag="--optim.max-steps")
    max_steps = int(max_steps_override) if max_steps_override is not None else 30_000

    scenes_raw = str(args.scenes).strip()
    if scenes_raw.lower() == "all":
        scenes = list(_SCENES_360V2)
    else:
        scenes = [s.strip() for s in scenes_raw.split(",") if s.strip()]
    if not scenes:
        raise ValueError("No scenes selected.")

    any_failed = False
    for scene in scenes:
        scene_data_dir = dataset_root_dir / scene
        if not scene_data_dir.exists():
            print(f"[skip] missing data_dir: {scene_data_dir}", flush=True)
            any_failed = True
            continue

        data_factor = int(scene_data_factors.get(scene, 1))
        if data_factor <= 0:
            raise ValueError(
                f"data_factor must be > 0, got {data_factor} (scene={scene!r})."
            )

        strategy_impl_raw = str(args.strategy_impl)
        if strategy_impl_raw == "all":
            strategy_impls = ("improved", "default", "mcmc")
        else:
            strategy_impls = (strategy_impl_raw,)

        # All outputs go under:
        # <data-root>/benchmark/strategies_benchmark/<scene>/<strategy>/...
        out_root = data_root / "benchmark" / "strategies_benchmark"

        for strategy_impl in strategy_impls:
            scene_out_dir = out_root / scene / str(strategy_impl)

            final_ply = scene_out_dir / "ply" / f"splats_step{int(max_steps):06d}.ply"
            if bool(args.skip_existing) and final_ply.exists():
                print(f"[done] {scene} ({strategy_impl}): {scene_out_dir}", flush=True)
                continue

            means_lr_final_override: Optional[str] = None
            if str(strategy_impl) == "improved":
                strategy_args = [
                    "--strategy.grow-grad2d",
                    "0.0002",
                    "--optim.no-random-bkgd",
                    "--optim.mu-enable",
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
                    "--strategy.refine-scale2d-stop-iter",
                    "0",
                    "--strategy.prune-scale3d",
                    "999.0",
                ]
                budget_args = [
                    "--strategy.densification-budget",
                    str(int(improved_gs_budgets_normal[scene])),
                ]
                means_lr_final_override = "2e-6"
            elif str(strategy_impl) == "default":
                strategy_args = [
                    "--strategy.no-absgrad",
                    "--strategy.verbose",
                    "--strategy.refine-scale2d-stop-iter",
                    "0",
                    "--strategy.prune-scale3d",
                    "0.1",
                    "--optim.no-random-bkgd",
                    "--optim.optimizers.means.optimizer.lr",
                    "1.6e-4",
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
                budget_args = []
                means_lr_final_override = "1.6e-6"
            elif str(strategy_impl) == "mcmc":
                strategy_args = [
                    "--init.init-opacity",
                    "0.5",
                    "--init.init-scale",
                    "0.1",
                    "--reg.opacity-reg-weight",
                    "0.01",
                    "--reg.scale-l1-reg-weight",
                    "0.01",
                    "--strategy.refine-stop-iter",
                    "25000",
                    "--strategy.mcmc-noise-lr",
                    "5e5",
                    "--strategy.mcmc-noise-injection-stop-iter",
                    "-1",
                    "--strategy.mcmc-min-opacity",
                    "0.005",
                    "--strategy.verbose",
                    "--optim.no-random-bkgd",
                    "--optim.optimizers.means.optimizer.lr",
                    "1.6e-4",
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
                budget_args = [
                    "--strategy.mcmc-cap-max",
                    str(int(improved_gs_budgets_normal[scene])),
                ]
                means_lr_final_override = "1.6e-6"
            else:
                raise AssertionError(f"Unhandled strategy_impl={strategy_impl!r}")

            cmd: list[str] = [
                sys.executable,
                str(trainer_py),
                "--io.data-dir",
                str(scene_data_dir),
                "--io.result-dir",
                str(scene_out_dir),
                "--io.device",
                device,
                "--data.preload",
                data_preload,
                "--data.no-prefetch-to-gpu",
                "--data.num-workers",
                "0",
                "--data.data-factor",
                str(int(data_factor)),
                "--optim.max-steps",
                str(int(max_steps)),
                "--strategy.impl",
                str(strategy_impl),
                "--viewer.disable-viewer",
                "--data.benchmark-train-split",
                "--io.export-ply",
                "--io.ply-steps",
                str(int(max_steps)),
                "--tb.enable",
                "--tb.every-n",
                "100",
                "--tb.flush-every-n",
                "500",
                *strategy_args,
                *budget_args,
                *extra_args,
            ]
            if means_lr_final_override is not None:
                cmd += [
                    "optim.optimizers.means.scheduler:exponential-decay-scheduler-config",
                    "--optim.optimizers.means.scheduler.lr-final",
                    str(means_lr_final_override),
                ]

            tag = "dry-run" if bool(args.dry_run) else "run"
            print(f"[{tag}] {scene} ({strategy_impl})", flush=True)
            print(f"[cmd] {_format_cmd(cmd)}", flush=True)
            if bool(args.dry_run):
                continue

            scene_out_dir.mkdir(parents=True, exist_ok=True)
            log_dir = scene_out_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "train.log"

            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
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
                    f"[fail] {scene} ({strategy_impl}) (see {log_path})",
                    flush=True,
                )
            else:
                print(f"[ok] {scene} ({strategy_impl})", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
