#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


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
            "Batch runner for pruning visual quality benchmark on 360v2 "
            "(single GPU, sequential)."
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
        help="all|csv of scene names (e.g. bicycle,garden).",
    )
    parser.add_argument(
        "--pruners",
        type=str,
        default="all",
        help="all|csv of pruners: pure_densify,gns,speedy.",
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
            "Extra args forwarded to fs-train (use after '--'). "
            "These are appended after the script's default trainer hyperparameters."
        ),
    )
    args = parser.parse_args(argv)

    # Baseline benchmark settings (edit here, or override via extra_args).
    device = "cuda:0"
    data_preload = "cuda"
    max_steps_default = 30_000
    densify_multiplier = 3
    densify_stop_step = 15_000
    gns_reg_end = 23_000
    gns_opacity_reg_weight = 2e-5
    speedy_every_n = 2500
    speedy_stop_step = 25_000
    speedy_score_num_views = 0
    tb_every_n = 100
    tb_flush_every_n = 500

    scene_data_factors: dict[str, float] = {
        "bicycle": 4.0,
        "flowers": 4.0,
        "garden": 4.0,
        "stump": 4.0,
        "treehill": 4.0,
        "bonsai": 2.0,
        "counter": 2.0,
        "kitchen": 2.0,
        "room": 2.0,
    }
    scene_final_budgets: dict[str, int] = {
        "bicycle": 600_000,
        "flowers": 600_000,
        "garden": 600_000,
        "stump": 600_000,
        "treehill": 600_000,
        "bonsai": 300_000,
        "counter": 300_000,
        "kitchen": 300_000,
        "room": 300_000,
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
    max_steps = int(max_steps_override) if max_steps_override is not None else max_steps_default

    scenes_raw = str(args.scenes).strip()
    if scenes_raw.lower() in ("all", "default"):
        scenes = list(_SCENES_360V2)
    else:
        scenes = [s.strip() for s in scenes_raw.split(",") if s.strip()]
    if not scenes:
        raise ValueError("No scenes selected.")

    pruners_raw = str(args.pruners).strip().lower()
    if pruners_raw == "all":
        pruners = ("pure_densify", "gns", "speedy")
    else:
        aliases = {
            "pure_densify": "pure_densify",
            "pure-densify": "pure_densify",
            "dense": "pure_densify",
            "densify": "pure_densify",
            "no_prune": "pure_densify",
            "baseline": "pure_densify",
            "gns": "gns",
            "speedy": "speedy",
        }
        pruners_norm: list[str] = []
        for x in [v.strip() for v in pruners_raw.split(",") if v.strip()]:
            key = aliases.get(x.replace("-", "_"))
            if key is None:
                raise KeyError(
                    f"Unknown pruner {x!r}. Expected: pure_densify,gns,speedy,all."
                )
            if key not in pruners_norm:
                pruners_norm.append(key)
        pruners = tuple(pruners_norm)
    if not pruners:
        raise ValueError("No pruners selected.")

    any_failed = False
    out_root = data_root / "benchmark" / "pruning_benchmark"
    for scene in scenes:
        scene_data_dir = dataset_root_dir / scene
        if not scene_data_dir.exists():
            print(f"[skip] missing data_dir: {scene_data_dir}", flush=True)
            any_failed = True
            continue

        data_factor = float(scene_data_factors.get(scene, 1.0))
        if not (data_factor > 0.0):
            raise ValueError(
                f"data_factor must be > 0, got {data_factor} (scene={scene!r})."
            )
        if scene not in scene_final_budgets:
            raise KeyError(
                f"final_budget is not configured for scene {scene!r} in this script."
            )
        final_budget = int(scene_final_budgets[scene])
        densify_budget_for_pruning = int(final_budget) * int(densify_multiplier)

        for pruner in pruners:
            scene_out_dir = out_root / scene / str(pruner)
            final_ply = scene_out_dir / "ply" / f"splats_step{int(max_steps):06d}.ply"
            if bool(args.skip_existing) and final_ply.exists():
                print(f"[done] {scene} (pruner={pruner}): {scene_out_dir}", flush=True)
                continue

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
                str(float(data_factor)),
                "--optim.max-steps",
                str(int(max_steps)),
                "--strategy.impl",
                "improved",
                "--viewer.disable-viewer",
                "--data.benchmark-train-split",
                "--strategy.grow-grad2d",
                "0.0002",
                "--optim.no-random-bkgd",
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
                "--strategy.refine-stop-iter",
                str(int(densify_stop_step)),
                "--strategy.densification-budget",
                str(
                    int(final_budget)
                    if str(pruner) == "pure_densify"
                    else int(densify_budget_for_pruning)
                ),
            ]

            if str(pruner) == "pure_densify":
                pass
            elif str(pruner) == "gns":
                reg_start = int(densify_stop_step) + 1
                reg_end = int(min(int(gns_reg_end), int(max_steps)))
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
                    str(float(gns_opacity_reg_weight)),
                ]
            elif str(pruner) == "speedy":
                start_step_1based = int(densify_stop_step) + 1
                stop_step_1based = int(min(int(speedy_stop_step), int(max_steps)))
                if stop_step_1based < start_step_1based:
                    stop_step_1based = int(start_step_1based)
                cmd += [
                    "--hard-prune.enable",
                    "--hard-prune.start-step",
                    str(int(start_step_1based)),
                    "--hard-prune.stop-step",
                    str(int(stop_step_1based)),
                    "--hard-prune.every-n",
                    str(int(speedy_every_n)),
                    "--hard-prune.final-budget",
                    str(int(final_budget)),
                ]
                if int(speedy_score_num_views) > 0:
                    cmd += [
                        "--hard-prune.score-num-views",
                        str(int(speedy_score_num_views)),
                    ]
            else:
                raise AssertionError(f"Unhandled pruner={pruner!r}")

            cmd += [
                "--io.export-ply",
                "--io.ply-steps",
                str(int(max_steps)),
                "--tb.enable",
                "--tb.every-n",
                str(int(tb_every_n)),
                "--tb.flush-every-n",
                str(int(tb_flush_every_n)),
                *extra_args,
                "optim.optimizers.means.scheduler:exponential-decay-scheduler-config",
                "--optim.optimizers.means.scheduler.lr-final",
                "2e-6",
            ]

            tag = "dry-run" if bool(args.dry_run) else "run"
            print(
                f"[{tag}] {scene} (pruner={pruner}) "
                f"final_budget={final_budget} densify_budget="
                f"{final_budget if str(pruner) == 'pure_densify' else densify_budget_for_pruning}",
                flush=True,
            )
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
                    f"[fail] {scene} (pruner={pruner}) (see {log_path})",
                    flush=True,
                )
            else:
                print(f"[ok] {scene} (pruner={pruner})", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
