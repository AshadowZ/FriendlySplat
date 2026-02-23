#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional


_TNT_SCENES_DEFAULT: tuple[str, ...] = (
    "Barn",
    "Caterpillar",
    "Courthouse",
    "Ignatius",
    "Meetingroom",
    "Truck",
)


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


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


def _ply_done_path(*, result_dir: Path, max_steps: int) -> Path:
    return result_dir / "ply" / f"splats_step{int(max_steps):06d}.ply"


def _train_done(*, result_dir: Path, max_steps: int) -> bool:
    return _ply_done_path(result_dir=result_dir, max_steps=max_steps).exists()


def _auto_tnt_dir(*, data_root: Path) -> Path:
    candidates = [
        data_root / "Tanks&Temples-Geo",
        data_root / "TanksAndTemples-Geo",
        data_root / "TanksAndTemples",
        data_root / "TNT",
        data_root / "tnt",
    ]
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "TnT dir not found. Pass --tnt-dir explicitly. "
        f"Tried: {tried}"
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_train_tnt_batch.py",
        description="Batch runner for Tanks & Temples (TnT) training.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing TnT scenes (e.g. Tanks&Temples-Geo/).",
    )
    parser.add_argument(
        "--tnt-dir",
        type=str,
        default=None,
        help="Optional explicit TnT scene root dir. If omitted, auto-detect under --data-root.",
    )
    parser.add_argument(
        "--out-dir-name",
        type=str,
        default="geo_benchmark",
        help="Output root directory name under data-root (default: geo_benchmark).",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="tnt_default",
        help="Experiment name under each scene output folder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device string passed to trainer.",
    )
    parser.add_argument(
        "--data-preload",
        type=str,
        choices=("none", "cuda"),
        default="cuda",
        help="DataLoader preload mode (default: cuda).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30_000,
        help="Training steps for each scene.",
    )
    parser.add_argument(
        "--strategy-impl",
        type=str,
        default="improved",
        help="Strategy impl forwarded to trainer (default: improved).",
    )
    parser.add_argument(
        "--densification-budget",
        type=int,
        default=1_000_000,
        help="Per-scene densification budget (approx. max gaussians; default: 1000000).",
    )
    parser.add_argument(
        "--grow-grad2d",
        type=float,
        default=None,
        help="Optional: forward to trainer as --strategy.grow-grad2d (default: unchanged).",
    )
    parser.add_argument(
        "--prune-opa",
        type=float,
        default=0.05,
        help="Strategy prune_opa forwarded to trainer (default: 0.05).",
    )
    parser.add_argument(
        "--prune-scale3d",
        type=float,
        default=0.1,
        help="Strategy prune_scale3d forwarded to trainer (default: 0.1).",
    )
    parser.add_argument(
        "--flat-reg-weight",
        type=float,
        default=1.0,
        help="Forward to trainer as --reg.flat-reg-weight (default: 1.0).",
    )
    parser.add_argument(
        "--scale-ratio-reg-weight",
        type=float,
        default=1.0,
        help="Forward to trainer as --reg.scale-ratio-reg-weight (default: 1.0).",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="default",
        help="default|all|csv of scene names (e.g. Barn,Truck).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a scene if its final PLY already exists under the output dir (default: on).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging.",
    )
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to friendly_splat/trainer.py (use after '--').",
    )
    args = parser.parse_args(argv)

    data_root = Path(str(args.data_root)).expanduser().resolve()
    if args.tnt_dir is None:
        tnt_dir = _auto_tnt_dir(data_root=data_root)
        print(f"[auto] using tnt_dir={tnt_dir}", flush=True)
    else:
        tnt_dir = Path(str(args.tnt_dir)).expanduser().resolve()
    if not tnt_dir.exists():
        raise FileNotFoundError(f"TnT dir not found: {tnt_dir}")

    scenes_raw = str(args.scenes).strip()
    if scenes_raw.lower() == "default":
        scenes = list(_TNT_SCENES_DEFAULT)
    elif scenes_raw.lower() == "all":
        scenes = sorted({p.name for p in tnt_dir.iterdir() if p.is_dir() and not p.name.startswith(".")})
    else:
        scenes = _split_csv(scenes_raw)
    if not scenes:
        raise ValueError("No scenes selected.")

    repo_root = Path(__file__).resolve().parents[2]
    trainer_py = repo_root / "friendly_splat" / "trainer.py"
    if not trainer_py.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_py}")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    out_root = data_root / str(args.out_dir_name) / "TnT"
    if not bool(args.dry_run):
        out_root.mkdir(parents=True, exist_ok=True)

    any_failed = False
    for scene in scenes:
        scene_dir = tnt_dir / str(scene)
        if not scene_dir.exists():
            print(f"[skip] missing scene dir: {scene_dir}", flush=True)
            any_failed = True
            continue
        if not (scene_dir / "images").exists() or not (scene_dir / "sparse").exists():
            print(f"[skip] missing images/ or sparse/: {scene_dir}", flush=True)
            any_failed = True
            continue

        result_dir = out_root / str(scene) / str(args.exp_name)
        done = _train_done(result_dir=result_dir, max_steps=int(args.max_steps))
        if bool(args.skip_existing) and done:
            print(f"[done] {scene}: {result_dir}", flush=True)
            continue

        if not bool(args.dry_run):
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "logs").mkdir(parents=True, exist_ok=True)

        log_path = result_dir / "logs" / "train.log"

        cmd: list[str] = [
            sys.executable,
            str(trainer_py),
            "--io.data-dir",
            str(scene_dir),
            "--io.result-dir",
            str(result_dir),
            "--io.device",
            str(args.device),
            "--data.data-factor",
            "1",
            "--data.preload",
            str(args.data_preload),
            "--postprocess.use-bilateral-grid",
            "--optim.max-steps",
            str(int(args.max_steps)),
            "--strategy.impl",
            str(args.strategy_impl),
            "--strategy.densification-budget",
            str(int(args.densification_budget)),
            "--strategy.prune-opa",
            str(float(args.prune_opa)),
            "--strategy.prune-scale3d",
            str(float(args.prune_scale3d)),
            "--strategy.absgrad",
            "--reg.flat-reg-weight",
            str(float(args.flat_reg_weight)),
            "--reg.scale-ratio-reg-weight",
            str(float(args.scale_ratio_reg_weight)),
            "--viewer.disable-viewer",
            "--io.export-ply",
            "--io.ply-steps",
            str(int(args.max_steps)),
        ]
        if args.grow_grad2d is not None:
            cmd += ["--strategy.grow-grad2d", str(float(args.grow_grad2d))]
        if str(args.data_preload) == "cuda":
            cmd += [
                "--data.no-prefetch-to-gpu",
                "--data.num-workers",
                "0",
            ]
        if extra_args:
            cmd += extra_args

        tag = "dry-run" if bool(args.dry_run) else "run"
        print(f"[{tag}] {scene} -> {result_dir}", flush=True)
        print(f"[cmd] {_format_cmd(cmd)}", flush=True)
        if bool(args.dry_run):
            continue

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
            print(f"[fail] {scene} (see {log_path})", flush=True)
        else:
            print(f"[ok] {scene}", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

