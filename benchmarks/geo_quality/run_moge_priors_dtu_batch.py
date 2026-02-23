#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
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

_DTU_DIR_NAME = "DTU"
_MOGE_MODEL_ID = "Ruicheng/moge-2-vitl-normal"
_ALPHA_MASK_DIR_NAME = "invalid_mask"


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


def _list_images_flat(*, image_dir: Path) -> list[Path]:
    """List images under image_dir (non-recursive)."""
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not image_dir.is_dir():
        return []
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def _all_pngs_exist(*, parent_dir: Path, stems: list[str]) -> bool:
    if not stems or not parent_dir.is_dir():
        return False
    return all((parent_dir / f"{s}.png").exists() for s in stems)


def _write_alpha_masks(
    *,
    scene_dir: Path,
    image_paths: list[Path],
    alpha_mask_dir_name: str,
    verbose: bool,
) -> None:
    """Write alpha-derived invalid masks (255=invalid/background, 0=valid/foreground)."""
    import cv2
    import numpy as np

    if not image_paths:
        raise FileNotFoundError("No images provided for alpha-mask export.")

    alpha_dir = scene_dir / str(alpha_mask_dir_name)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")

        mask_u8 = np.zeros(img.shape[:2], dtype=np.uint8)
        if img.ndim == 3 and int(img.shape[-1]) == 4:
            alpha = img[..., 3]
            if np.issubdtype(alpha.dtype, np.integer):
                denom = float(np.iinfo(alpha.dtype).max)
                alpha_f = alpha.astype(np.float32) / max(denom, 1.0)
            else:
                alpha_f = alpha.astype(np.float32)
            mask_u8 = (alpha_f < 0.5).astype(np.uint8) * 255

        out_path = alpha_dir / f"{img_path.stem}.png"
        if not cv2.imwrite(str(out_path), mask_u8):
            raise RuntimeError(f"cv2.imwrite failed: {out_path}")
        if verbose:
            print(f"[MASK] {out_path}", flush=True)


def _run_one_scene(
    *,
    scene_dir: Path,
    verbose: bool,
    dry_run: bool,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    moge_script = repo_root / "tools" / "depth_prior" / "moge_infer.py"
    if not moge_script.exists():
        raise FileNotFoundError(f"Missing script: {moge_script}")

    cmd = [
        sys.executable,
        str(moge_script),
        "--data-dir",
        str(scene_dir),
        "--factor",
        "1",
        "--model-id",
        _MOGE_MODEL_ID,
        "--out-normal-dir",
        "moge_normal",
        "--no-align-depth-with-colmap",
    ]
    if verbose:
        cmd.append("--verbose")

    print(_format_cmd(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_moge_priors_dtu_batch.py",
        description=(
            "Batch runner to generate MoGe normal priors for the DTU dataset. "
            "Outputs are written into each scan folder as moge_normal/. "
            "Assumes DTU's common flat image layout: scanXX/images/* (non-recursive)."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory containing DTU/ (e.g. /path/to/data).",
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
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip scans that already have complete moge_normal outputs (default: on).",
    )
    parser.add_argument(
        "--export-alpha-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Export alpha-derived background masks from images/ into a per-scan folder. "
            f"Outputs to '{_ALPHA_MASK_DIR_NAME}/' (white=invalid/background, black=valid/foreground)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    data_root = Path(str(args.data_root)).expanduser().resolve()
    dtu_dir = data_root / _DTU_DIR_NAME
    if not dtu_dir.exists():
        raise FileNotFoundError(f"DTU dir not found: {dtu_dir}")

    scans_raw = str(args.scans).strip().lower()
    if scans_raw == "default":
        scans = list(_DTU_SCANS_DEFAULT)
    elif scans_raw == "all":
        scans = sorted({p.name for p in dtu_dir.iterdir() if p.is_dir() and p.name.startswith("scan")})
    else:
        scans = [_normalize_scan_name(s) for s in _split_csv(args.scans)]

    if not scans:
        raise ValueError("No scans selected.")

    for scan in scans:
        scene_dir = dtu_dir / scan
        if not scene_dir.exists():
            print(f"[SKIP] Missing scan dir: {scene_dir}", flush=True)
            continue
        image_dir = scene_dir / "images"
        image_paths = _list_images_flat(image_dir=image_dir)
        stems = sorted(p.stem for p in image_paths)
        if not stems:
            print(f"[SKIP] No images found under: {image_dir}", flush=True)
            continue

        moge_done = _all_pngs_exist(parent_dir=scene_dir / "moge_normal", stems=stems)
        alpha_done = (
            (not bool(args.export_alpha_mask))
            or _all_pngs_exist(
                parent_dir=scene_dir / _ALPHA_MASK_DIR_NAME,
                stems=stems,
            )
        )
        if bool(args.skip_existing) and moge_done and alpha_done:
            print(
                f"[DONE] {scan} (images={len(stems)}, normals={len(stems)})",
                flush=True,
            )
            continue

        print(f"[RUN] {scan}", flush=True)
        if not moge_done:
            _run_one_scene(
                scene_dir=scene_dir,
                verbose=bool(args.verbose),
                dry_run=bool(args.dry_run),
            )

        if bool(args.export_alpha_mask) and not alpha_done:
            if bool(args.dry_run):
                print(
                    f"[DRY] export alpha masks -> {scene_dir / _ALPHA_MASK_DIR_NAME}",
                    flush=True,
                )
            else:
                _write_alpha_masks(
                    scene_dir=scene_dir,
                    image_paths=image_paths,
                    alpha_mask_dir_name=_ALPHA_MASK_DIR_NAME,
                    verbose=bool(args.verbose),
                )

        normals_ok = _all_pngs_exist(parent_dir=scene_dir / "moge_normal", stems=stems)
        n_normals = len(stems) if normals_ok else sum(
            1 for _ in (scene_dir / "moge_normal").glob("*.png")
        )
        print(f"[OUT] {scan} (images={len(stems)}, normals={int(n_normals)})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
