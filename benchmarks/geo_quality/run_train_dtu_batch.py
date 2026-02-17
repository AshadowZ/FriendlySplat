#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_DTU_SCANS_DEFAULT: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class _Counts:
    images: int
    normals: int
    depths: int
    invalid_masks: int


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


def _normalize_scan_name(name: str) -> str:
    s = str(name).strip()
    if not s:
        raise ValueError("Empty scan name.")
    s = s.lower()
    if s.startswith("scan"):
        return f"scan{int(s[4:]):d}"
    return f"scan{int(s):d}"


def _count_images(*, image_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not image_dir.exists():
        return 0
    return sum(
        1 for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts
    )


def _count_scene_priors(*, scene_dir: Path, invalid_mask_dir_name: str) -> _Counts:
    images = _count_images(image_dir=scene_dir / "images")
    normals = sum(1 for _ in (scene_dir / "moge_normal").rglob("*.png"))
    depths = sum(1 for _ in (scene_dir / "moge_depth").rglob("*.npy"))
    invalid_masks = sum(
        1 for _ in (scene_dir / str(invalid_mask_dir_name)).rglob("*.png")
    )
    return _Counts(
        images=int(images),
        normals=int(normals),
        depths=int(depths),
        invalid_masks=int(invalid_masks),
    )


def _priors_ready(*, scene_dir: Path, invalid_mask_dir_name: str) -> bool:
    c = _count_scene_priors(scene_dir=scene_dir, invalid_mask_dir_name=invalid_mask_dir_name)
    if c.images <= 0:
        return False
    return (
        c.normals == c.images
        and c.depths == c.images
        and c.invalid_masks == c.images
    )


def _ply_done_path(*, result_dir: Path, max_steps: int) -> Path:
    return result_dir / "ply" / f"splats_step{int(max_steps):06d}.ply"


def _train_done(*, result_dir: Path, max_steps: int) -> bool:
    return _ply_done_path(result_dir=result_dir, max_steps=max_steps).exists()


def _maybe_prepare_priors(
    *,
    repo_root: Path,
    data_root: Path,
    scans: list[str],
    python: str,
    invalid_mask_dir_name: str,
    model_id: str,
    verbose: bool,
    dry_run: bool,
) -> None:
    prep_script = repo_root / "benchmarks" / "geo_quality" / "run_moge_priors_dtu_batch.py"
    if not prep_script.exists():
        raise FileNotFoundError(f"Missing prep script: {prep_script}")

    cmd = [
        str(python),
        str(prep_script),
        "--data-root",
        str(data_root),
        "--scans",
        ",".join(scans),
        "--export-alpha-mask",
        "--alpha-mask-dir-name",
        str(invalid_mask_dir_name),
        "--model-id",
        str(model_id),
    ]
    if verbose:
        cmd.append("--verbose")
    if dry_run:
        cmd.append("--dry-run")

    print(f"[prepare] {_format_cmd(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_train_dtu_batch.py",
        description=(
            "Batch runner for DTU training with MoGe depth/normal priors. "
            "Uses invalid_mask as sky_mask to ignore alpha background pixels."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing DTU/ (e.g. /path/to/data).",
    )
    parser.add_argument(
        "--dtu-dir-name",
        type=str,
        default="DTU",
        help="DTU directory name under data-root (default: DTU).",
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
        default="dtu_moge_priors",
        help="Experiment name under each scan output folder.",
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
        default="none",
        help="DataLoader preload mode (default: none).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30_000,
        help="Training steps for each scan.",
    )
    parser.add_argument(
        "--scans",
        type=str,
        default="default",
        help=(
            "Comma-separated scan ids/names to run (e.g. '24,37' or 'scan24,scan37'). "
            "Use 'default' for the common 15-scan benchmark list, or 'all' to auto-discover."
        ),
    )
    parser.add_argument(
        "--exclude-scans",
        type=str,
        default=None,
        help="Optional comma-separated scans to exclude (same format as --scans).",
    )
    parser.add_argument(
        "--invalid-mask-dir-name",
        type=str,
        default="invalid_mask",
        help="Per-scan mask folder name used as sky_mask_dir_name (default: invalid_mask).",
    )
    parser.add_argument(
        "--prepare-priors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, auto-run the MoGe prior + invalid_mask generation script before training.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Ruicheng/moge-2-vitl-normal",
        help="MoGe pretrained model id used when --prepare-priors is enabled.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a scan if its final PLY already exists under the output dir (default: on).",
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
    dtu_dir = data_root / str(args.dtu_dir_name)
    if not dtu_dir.exists():
        raise FileNotFoundError(f"DTU dir not found: {dtu_dir}")

    scans_raw = str(args.scans).strip().lower()
    if scans_raw == "default":
        scans = list(_DTU_SCANS_DEFAULT)
    elif scans_raw == "all":
        scans = sorted({p.name for p in dtu_dir.iterdir() if p.is_dir() and p.name.startswith("scan")})
    else:
        scans = [_normalize_scan_name(s) for s in _split_csv(args.scans)]

    exclude = {_normalize_scan_name(s) for s in _split_csv(args.exclude_scans)}
    scans = [s for s in scans if s not in exclude]
    if not scans:
        raise ValueError("No scans selected.")

    repo_root = Path(__file__).resolve().parents[2]
    trainer_py = repo_root / "friendly_splat" / "trainer.py"
    if not trainer_py.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_py}")

    extra_args = list(args.extra_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    out_root = data_root / str(args.out_dir_name) / "DTU"
    if not bool(args.dry_run):
        out_root.mkdir(parents=True, exist_ok=True)

    if bool(args.prepare_priors):
        _maybe_prepare_priors(
            repo_root=repo_root,
            data_root=data_root,
            scans=scans,
            python=sys.executable,
            invalid_mask_dir_name=str(args.invalid_mask_dir_name),
            model_id=str(args.model_id),
            verbose=bool(args.verbose),
            dry_run=bool(args.dry_run),
        )

    any_failed = False
    for scan in scans:
        scene_dir = dtu_dir / scan
        if not scene_dir.exists():
            print(f"[skip] missing scan dir: {scene_dir}", flush=True)
            continue

        if not _priors_ready(scene_dir=scene_dir, invalid_mask_dir_name=str(args.invalid_mask_dir_name)):
            c = _count_scene_priors(scene_dir=scene_dir, invalid_mask_dir_name=str(args.invalid_mask_dir_name))
            print(
                f"[skip] priors missing/incomplete: {scan} "
                f"(images={c.images}, normals={c.normals}, depths={c.depths}, invalid_masks={c.invalid_masks})",
                flush=True,
            )
            any_failed = True
            continue

        result_dir = out_root / scan / str(args.exp_name)
        done = _train_done(result_dir=result_dir, max_steps=int(args.max_steps))
        if bool(args.skip_existing) and done:
            print(f"[done] {scan}: {result_dir}", flush=True)
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
            "--data.depth-dir-name",
            "moge_depth",
            "--data.normal-dir-name",
            "moge_normal",
            "--data.sky-mask-dir-name",
            str(args.invalid_mask_dir_name),
            "--optim.max-steps",
            str(int(args.max_steps)),
            "--strategy.impl",
            "improved",
            "--viewer.disable-viewer",
            "--io.export-ply",
            "--io.ply-steps",
            str(int(args.max_steps)),
        ]

        if str(args.data_preload) == "cuda":
            cmd += [
                "--data.no-prefetch-to-gpu",
                "--data.num-workers",
                "0",
            ]

        # Reasonable defaults for benchmarks.
        cmd += ["--optim.no-random-bkgd"]

        if extra_args:
            cmd += extra_args

        tag = "dry-run" if bool(args.dry_run) else "run"
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

