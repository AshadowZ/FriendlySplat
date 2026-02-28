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
        prog="run_train_dtu_batch.py",
        description="Batch runner for DTU training (half-res, priors).",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "Root directory containing the DTU dataset folders "
            "(expects dtu_dataset/dtu/ under this root)."
        ),
    )
    parser.add_argument(
        "--dtu-dir-name",
        type=str,
        default="dtu_dataset/dtu",
        help="DTU scene directory under data-root (default: dtu_dataset/dtu).",
    )
    parser.add_argument(
        "--out-dir-name",
        type=str,
        default="benchmark/geo_benchmark/dtu_benchmark",
        help="Output directory under data-root (default: benchmark/geo_benchmark/dtu_benchmark).",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="dtu_moge_priors",
        help="Experiment name under each scan output folder.",
    )
    parser.add_argument(
        "--scans",
        type=str,
        default="default",
        help="default|all|csv of scan ids/names (e.g. scan24,37).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a scan if its final PLY already exists under the output dir (default: on).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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
    dtu_dir = data_root / str(args.dtu_dir_name)
    if not dtu_dir.exists():
        raise FileNotFoundError(f"DTU dir not found: {dtu_dir}")

    scans_raw = str(args.scans).strip()
    if scans_raw.lower() == "default":
        scans = [
            "scan24",
            "scan37",
            "scan40",
            "scan55",
            "scan63",
            "scan65",
            "scan69",
            "scan83",
            "scan97",
            "scan105",
            "scan106",
            "scan110",
            "scan114",
            "scan118",
            "scan122",
        ]
    elif scans_raw.lower() == "all":
        scans = sorted({p.name for p in dtu_dir.iterdir() if p.is_dir() and p.name.startswith("scan")})
    else:
        scans = []
        for part in scans_raw.split(","):
            s = part.strip()
            if not s:
                continue
            s = s.lower()
            if s.startswith("scan"):
                scans.append(f"scan{int(s[4:]):d}")
            else:
                scans.append(f"scan{int(s):d}")
    if not scans:
        raise ValueError("No scans selected.")

    repo_root = Path(__file__).resolve().parents[2]
    trainer_py = repo_root / "friendly_splat" / "trainer.py"
    if not trainer_py.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_py}")

    preprocess_py = repo_root / "benchmarks" / "geo_quality" / "preprocess_dtu_batch.py"
    if not preprocess_py.exists():
        raise FileNotFoundError(f"Missing script: {preprocess_py}")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    max_steps_override = _read_flag_value(extra_args=extra_args, flag="--optim.max-steps")
    max_steps = int(max_steps_override) if max_steps_override is not None else 30_000

    out_root = data_root / str(args.out_dir_name)
    if not bool(args.dry_run):
        out_root.mkdir(parents=True, exist_ok=True)

    tag = "dry-run" if bool(args.dry_run) else "run"

    exts_img = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    scans_need_preprocess: list[str] = []
    for scan in scans:
        scene_dir = dtu_dir / str(scan)
        if not scene_dir.exists():
            continue

        result_dir = out_root / str(scan) / str(args.exp_name)
        final_ply = result_dir / "ply" / f"splats_step{int(max_steps):06d}.ply"
        if bool(args.skip_existing) and final_ply.exists():
            continue

        images_2_dir = scene_dir / "images_2"
        moge_normal_dir = scene_dir / "moge_normal"
        invalid_mask_dir = scene_dir / "invalid_mask"

        have_images_2 = images_2_dir.is_dir() and any(
            p.is_file() and p.suffix.lower() in exts_img for p in images_2_dir.iterdir()
        )
        have_normals = moge_normal_dir.is_dir() and any(
            p.is_file() and p.suffix.lower() == ".png" for p in moge_normal_dir.iterdir()
        )
        have_invalid = invalid_mask_dir.is_dir() and any(
            p.is_file() and p.suffix.lower() == ".png" for p in invalid_mask_dir.iterdir()
        )

        if not (have_images_2 and have_normals and have_invalid):
            scans_need_preprocess.append(str(scan))

    if scans_need_preprocess:
        preprocess_cmd = [
            sys.executable,
            str(preprocess_py),
            "--data-root",
            str(data_root),
            "--dtu-dir-name",
            str(args.dtu_dir_name),
            "--scans",
            ",".join(scans_need_preprocess),
            "--skip-existing",
            "--skip-existing-images-2",
            "--export-alpha-mask",
            *(["--verbose"] if bool(args.verbose) else []),
            *(["--dry-run"] if bool(args.dry_run) else []),
        ]
        print(f"[{tag}] preprocess priors: {len(scans_need_preprocess)} scans", flush=True)
        print(f"[cmd] {_format_cmd(preprocess_cmd)}", flush=True)
        if not bool(args.dry_run):
            proc = subprocess.run(preprocess_cmd, check=False, cwd=str(repo_root))
            if proc.returncode != 0:
                print(f"[warn] preprocess returned {proc.returncode}.", flush=True)

    any_failed = False
    for scan in scans:
        scene_dir = dtu_dir / str(scan)
        if not scene_dir.exists():
            print(f"[skip] missing scan dir: {scene_dir}", flush=True)
            any_failed = True
            continue
        if not (scene_dir / "images").exists():
            print(f"[skip] missing images/: {scene_dir}", flush=True)
            any_failed = True
            continue
        if not (scene_dir / "sparse").exists():
            print(f"[skip] missing sparse/: {scene_dir}", flush=True)
            any_failed = True
            continue

        result_dir = out_root / str(scan) / str(args.exp_name)
        final_ply = result_dir / "ply" / f"splats_step{int(max_steps):06d}.ply"
        if bool(args.skip_existing) and final_ply.exists():
            print(f"[done] {scan}: {result_dir}", flush=True)
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
            "--data.no-prefetch-to-gpu",
            "--data.num-workers", "0",
            "--data.normal-dir-name", "moge_normal",
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
            "--strategy.densification-budget", "1000000",
            "--strategy.grow-grad2d", "0.0003",
            "--strategy.prune-opa", "0.05",
            "--strategy.prune-scale3d", "0.1",
            # ----------------------------
            # Regularization
            # ----------------------------
            "--reg.surf_normal_loss_activation_step", "1000",
            "--reg.flat-reg-weight", "1.0",
            "--reg.scale-ratio-reg-weight", "1.0",
            # ----------------------------
            # Viewer
            # ----------------------------
            "--viewer.disable-viewer",
            # ----------------------------
            # Extra overrides (appended last)
            # ----------------------------
            *extra_args,
        ]

        print(f"[{tag}] {scan} -> {result_dir}", flush=True)
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
            print(f"[fail] {scan} (see {log_path})", flush=True)
        else:
            print(f"[ok] {scan}", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
