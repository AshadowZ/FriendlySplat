#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


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
        prog="run_train_tnt_batch.py",
        description="Batch runner for Tanks & Temples (TnT) training.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "Root directory containing the TnT dataset folder "
            "(expects tnt_dataset/tnt/ under this root)."
        ),
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
        default="benchmark/geo_benchmark/tnt_benchmark",
        help=(
            "Output directory under data-root. Default: benchmark/geo_benchmark/tnt_benchmark"
        ),
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="tnt_default",
        help="Experiment name under each scene output folder.",
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
        help=(
            "Extra args forwarded to friendly_splat/trainer.py (use after '--'). "
            "These are appended after the script's default trainer hyperparameters."
        ),
    )
    args = parser.parse_args(argv)

    data_root = Path(str(args.data_root)).expanduser().resolve()
    tnt_dir = (
        (data_root / "tnt_dataset" / "tnt")
        if args.tnt_dir is None
        else Path(str(args.tnt_dir)).expanduser().resolve()
    )
    if args.tnt_dir is None:
        print(f"[auto] using tnt_dir={tnt_dir}", flush=True)
    if not tnt_dir.exists():
        raise FileNotFoundError(f"TnT dir not found: {tnt_dir}")

    scenes_raw = str(args.scenes).strip()
    if scenes_raw.lower() == "default":
        scenes = [
            "Barn",
            "Caterpillar",
            "Courthouse",
            "Ignatius",
            "Meetingroom",
            "Truck",
        ]
    elif scenes_raw.lower() == "all":
        scenes = sorted({p.name for p in tnt_dir.iterdir() if p.is_dir() and not p.name.startswith(".")})
    else:
        scenes = [s.strip() for s in scenes_raw.split(",") if s.strip()]
    if not scenes:
        raise ValueError("No scenes selected.")

    repo_root = Path(__file__).resolve().parents[2]
    trainer_py = repo_root / "friendly_splat" / "trainer.py"
    if not trainer_py.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_py}")

    preprocess_py = repo_root / "benchmarks" / "geo_quality" / "preprocess_tnt_batch.py"
    if not preprocess_py.exists():
        raise FileNotFoundError(f"Missing script: {preprocess_py}")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    max_steps_override = _read_flag_value(extra_args=extra_args, flag="--optim.max-steps")
    max_steps = int(max_steps_override) if max_steps_override is not None else 30_000

    tag = "dry-run" if bool(args.dry_run) else "run"
    exts_img = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    scenes_need_preprocess: list[str] = []
    scenes_preprocess_blocked: list[str] = []
    for scene in scenes:
        scene_dir = tnt_dir / str(scene)
        if not scene_dir.exists():
            continue

        out_root = data_root / str(args.out_dir_name)
        result_dir = out_root / str(scene) / str(args.exp_name)
        final_ply = result_dir / "ply" / f"splats_step{int(max_steps):06d}.ply"
        if bool(args.skip_existing) and final_ply.exists():
            continue

        image_dir = scene_dir / "images"
        if not image_dir.is_dir():
            scenes_preprocess_blocked.append(str(scene))
            continue

        images_2_dir = scene_dir / "images_2"
        moge_normal_dir = scene_dir / "moge_normal"
        moge_depth_dir = scene_dir / "moge_depth"
        invalid_mask_dir = scene_dir / "invalid_mask"

        have_images_2 = images_2_dir.is_dir() and any(
            p.is_file() and p.suffix.lower() in exts_img for p in images_2_dir.iterdir()
        )
        have_normals = moge_normal_dir.is_dir() and any(
            p.is_file() and p.suffix.lower() == ".png" for p in moge_normal_dir.iterdir()
        )
        have_depths = moge_depth_dir.is_dir() and any(
            p.is_file() and p.suffix.lower() == ".npy" for p in moge_depth_dir.iterdir()
        )
        have_invalid = invalid_mask_dir.is_dir() and any(
            p.is_file() and p.suffix.lower() == ".png" for p in invalid_mask_dir.iterdir()
        )

        if not (have_images_2 and have_normals and have_depths and have_invalid):
            scenes_need_preprocess.append(str(scene))

    if scenes_preprocess_blocked and bool(args.verbose):
        print(
            f"[warn] cannot preprocess (missing images/): {','.join(scenes_preprocess_blocked)}",
            flush=True,
        )

    if scenes_need_preprocess:
        preprocess_cmd = [
            sys.executable,
            str(preprocess_py),
            "--data-root",
            str(data_root),
            "--tnt-dir",
            str(tnt_dir),
            "--scenes",
            ",".join(scenes_need_preprocess),
            "--skip-existing",
            "--skip-existing-images-2",
            *(["--verbose"] if bool(args.verbose) else []),
            *(["--dry-run"] if bool(args.dry_run) else []),
        ]
        print(f"[{tag}] preprocess priors: {len(scenes_need_preprocess)} scenes", flush=True)
        print(f"[cmd] {_format_cmd(preprocess_cmd)}", flush=True)
        if not bool(args.dry_run):
            proc = subprocess.run(preprocess_cmd, check=False, cwd=str(repo_root))
            if proc.returncode != 0:
                print(f"[warn] preprocess returned {proc.returncode}.", flush=True)

    any_failed = False
    for scene in scenes:
        scene_dir = tnt_dir / str(scene)
        if not scene_dir.exists():
            print(f"[skip] missing scene dir: {scene_dir}", flush=True)
            any_failed = True
            continue
        if not ((scene_dir / "images").exists() or (scene_dir / "images_2").exists()):
            print(f"[skip] missing images/ (or images_2/): {scene_dir}", flush=True)
            any_failed = True
            continue
        if not (scene_dir / "sparse").exists():
            print(f"[skip] missing sparse/: {scene_dir}", flush=True)
            any_failed = True
            continue

        out_root = data_root / str(args.out_dir_name)
        result_dir = out_root / str(scene) / str(args.exp_name)
        final_ply = result_dir / "ply" / f"splats_step{int(max_steps):06d}.ply"
        if bool(args.skip_existing) and final_ply.exists():
            print(f"[done] {scene}: {result_dir}", flush=True)
            continue

        if not bool(args.dry_run):
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "logs").mkdir(parents=True, exist_ok=True)

        log_path = result_dir / "logs" / "train.log"

        cmd: list[str] = [
            sys.executable,
            str(trainer_py),
            # ----------------------------
            # I/O
            # ----------------------------
            "--io.data-dir", str(scene_dir),
            "--io.result-dir", str(result_dir),
            "--io.device", "cuda:0",
            "--io.export-ply",
            "--io.ply-steps", str(int(max_steps)),
            # ----------------------------
            # Data
            # ----------------------------
            "--data.data-factor", "2",
            "--data.preload", "cuda",
            "--data.normal-dir-name", "moge_normal",
            "--data.depth-dir-name", "moge_depth",
            "--data.sky-mask-dir-name",
            "invalid_mask",
            # ----------------------------
            # Postprocess
            # ----------------------------
            "--postprocess.use-bilateral-grid",
            # ----------------------------
            # Optim
            # ----------------------------
            "--optim.max-steps", str(int(max_steps)),
            # ----------------------------
            # Strategy
            # ----------------------------
            "--strategy.impl", "improved",
            "--strategy.absgrad",
            "--strategy.densification-budget", "2000000",
            "--strategy.grow-grad2d", "0.0001",
            "--strategy.prune-opa", "0.05",
            "--strategy.prune-scale3d", "0.1",
            # ----------------------------
            # Regularization
            # ----------------------------
            "--reg.flat-reg-weight", "1.0",
            "--reg.scale-ratio-reg-weight", "1.0",
            "--reg.prior-normal-reg-every-n", "2",
            "--reg.consistency-normal-loss-weight", "0.15",
            "--reg.consistency-normal-loss-activation-step", "15000",
            # ----------------------------
            # Viewer
            # ----------------------------
            "--viewer.disable-viewer",
            # ----------------------------
            # Extra overrides (appended last)
            # ----------------------------
            *extra_args,
        ]

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
