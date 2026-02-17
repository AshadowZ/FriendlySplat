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


def _iter_images(image_dir: Path) -> Iterable[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if not image_dir.exists():
        return []
    return (
        p
        for p in sorted(image_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in exts
    )

def _iter_image_relpaths(image_dir: Path) -> list[Path]:
    return [p.relative_to(image_dir) for p in _iter_images(image_dir)]


def _expected_alpha_mask_path(*, alpha_mask_dir: Path, image_relpath: Path) -> Path:
    # Always write masks as .png, even when the image extension differs.
    return alpha_mask_dir / image_relpath.with_suffix(".png")


def _alpha_masks_done(*, scene_dir: Path, alpha_mask_dir_name: str) -> bool:
    image_dir = scene_dir / "images"
    if not image_dir.exists():
        return False
    relpaths = _iter_image_relpaths(image_dir)
    if not relpaths:
        return False
    alpha_dir = scene_dir / str(alpha_mask_dir_name)
    for rel in relpaths:
        if not _expected_alpha_mask_path(alpha_mask_dir=alpha_dir, image_relpath=rel).exists():
            return False
    return True


def _write_alpha_masks(
    *,
    scene_dir: Path,
    alpha_mask_dir_name: str,
    verbose: bool,
) -> None:
    """Write alpha-based background masks for a scene.

    Mask convention:
    - White (255): non-existent background (alpha==0) => should be masked out.
    - Black (0): valid content (alpha>0) => keep.

    For non-RGBA images, emit an all-black mask.
    """
    import cv2
    import numpy as np

    image_dir = scene_dir / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {image_dir}")
    relpaths = _iter_image_relpaths(image_dir)
    if not relpaths:
        raise FileNotFoundError(f"No images found under: {image_dir}")

    alpha_dir = scene_dir / str(alpha_mask_dir_name)
    alpha_dir.mkdir(parents=True, exist_ok=True)

    for rel in relpaths:
        img_path = image_dir / rel
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {img_path}")

        if img.ndim == 2:
            h, w = int(img.shape[0]), int(img.shape[1])
            mask_u8 = np.zeros((h, w), dtype=np.uint8)
        else:
            h, w = int(img.shape[0]), int(img.shape[1])
            if int(img.shape[-1]) == 4:
                alpha = img[..., 3]
                mask_u8 = (alpha == 0).astype(np.uint8) * 255
            else:
                mask_u8 = np.zeros((h, w), dtype=np.uint8)

        out_path = _expected_alpha_mask_path(alpha_mask_dir=alpha_dir, image_relpath=rel)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), mask_u8):
            raise RuntimeError(f"cv2.imwrite failed: {out_path}")
        if verbose:
            print(f"[MASK] {out_path}", flush=True)


def _count_outputs(*, scene_dir: Path) -> _Counts:
    image_dir = scene_dir / "images"
    images = sum(1 for _ in _iter_images(image_dir))
    normals = sum(1 for _ in (scene_dir / "moge_normal").rglob("*.png"))
    depths = sum(1 for _ in (scene_dir / "moge_depth").rglob("*.npy"))
    return _Counts(images=images, normals=normals, depths=depths)


def _is_done(*, scene_dir: Path) -> bool:
    c = _count_outputs(scene_dir=scene_dir)
    if c.images <= 0:
        return False
    return c.normals == c.images and c.depths == c.images


def _run_one_scene(
    *,
    scene_dir: Path,
    python: str,
    cuda_visible_devices: Optional[str],
    model_id: str,
    read_threads: int,
    write_threads: int,
    queue_size: int,
    remove_depth_edge: bool,
    depth_edge_rtol: float,
    align_depth_with_colmap: bool,
    colmap_min_sparse: int,
    colmap_clip_low: float,
    colmap_clip_high: float,
    colmap_ransac_trials: int,
    verbose: bool,
    dry_run: bool,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    moge_script = repo_root / "tools" / "depth_prior" / "moge_infer.py"
    if not moge_script.exists():
        raise FileNotFoundError(f"Missing script: {moge_script}")

    cmd = [
        str(python),
        str(moge_script),
        "--data-dir",
        str(scene_dir),
        "--factor",
        "1",
        "--model-id",
        str(model_id),
        "--read-threads",
        str(int(read_threads)),
        "--write-threads",
        str(int(write_threads)),
        "--queue-size",
        str(int(queue_size)),
        "--save-depth",
        "--out-normal-dir",
        "moge_normal",
        "--out-depth-dir",
        "moge_depth",
        "--out-sky-mask-dir",
        "sky_mask",
        "--colmap-min-sparse",
        str(int(colmap_min_sparse)),
        "--colmap-clip-low",
        str(float(colmap_clip_low)),
        "--colmap-clip-high",
        str(float(colmap_clip_high)),
        "--colmap-ransac-trials",
        str(int(colmap_ransac_trials)),
    ]
    cmd.append("--remove-depth-edge" if bool(remove_depth_edge) else "--no-remove-depth-edge")
    cmd.extend(["--depth-edge-rtol", str(float(depth_edge_rtol))])
    cmd.append(
        "--align-depth-with-colmap"
        if bool(align_depth_with_colmap)
        else "--no-align-depth-with-colmap"
    )
    if verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    print(_format_cmd(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_moge_priors_dtu_batch.py",
        description=(
            "Batch runner to generate MoGe depth/normal priors for the DTU dataset. "
            "Outputs are written into each scan folder as moge_depth/ and moge_normal/."
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
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip scans that already have complete moge_depth + moge_normal outputs (default: on).",
    )
    parser.add_argument(
        "--export-alpha-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Export alpha-derived background masks from images/ into a per-scan folder. "
            "White means background (alpha==0), black means foreground (alpha>0)."
        ),
    )
    parser.add_argument(
        "--alpha-mask-dir-name",
        type=str,
        default="invalid_mask",
        help="Output folder name under each scan (default: invalid_mask).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable or "python3",
        help="Python executable to run moge_infer.py (default: current interpreter).",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override.",
    )

    # MoGe options (pass-through).
    parser.add_argument(
        "--model-id",
        type=str,
        default="Ruicheng/moge-2-vitl-normal",
        help="MoGe pretrained model id.",
    )
    parser.add_argument("--read-threads", type=int, default=2)
    parser.add_argument("--write-threads", type=int, default=4)
    parser.add_argument("--queue-size", type=int, default=8)

    parser.add_argument(
        "--remove-depth-edge",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--depth-edge-rtol", type=float, default=0.04)

    parser.add_argument(
        "--align-depth-with-colmap",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--colmap-min-sparse", type=int, default=30)
    parser.add_argument("--colmap-clip-low", type=float, default=5.0)
    parser.add_argument("--colmap-clip-high", type=float, default=95.0)
    parser.add_argument("--colmap-ransac-trials", type=int, default=2000)

    parser.add_argument("--verbose", action="store_true")
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

    for scan in scans:
        scene_dir = dtu_dir / scan
        if not scene_dir.exists():
            print(f"[SKIP] Missing scan dir: {scene_dir}", flush=True)
            continue
        moge_done = _is_done(scene_dir=scene_dir)
        alpha_done = (
            (not bool(args.export_alpha_mask))
            or _alpha_masks_done(
                scene_dir=scene_dir, alpha_mask_dir_name=str(args.alpha_mask_dir_name)
            )
        )
        if bool(args.skip_existing) and moge_done and alpha_done:
            c = _count_outputs(scene_dir=scene_dir)
            print(
                f"[DONE] {scan} (images={c.images}, normals={c.normals}, depths={c.depths})",
                flush=True,
            )
            continue

        print(f"[RUN] {scan}", flush=True)
        if not moge_done:
            _run_one_scene(
                scene_dir=scene_dir,
                python=str(args.python),
                cuda_visible_devices=args.cuda_visible_devices,
                model_id=str(args.model_id),
                read_threads=int(args.read_threads),
                write_threads=int(args.write_threads),
                queue_size=int(args.queue_size),
                remove_depth_edge=bool(args.remove_depth_edge),
                depth_edge_rtol=float(args.depth_edge_rtol),
                align_depth_with_colmap=bool(args.align_depth_with_colmap),
                colmap_min_sparse=int(args.colmap_min_sparse),
                colmap_clip_low=float(args.colmap_clip_low),
                colmap_clip_high=float(args.colmap_clip_high),
                colmap_ransac_trials=int(args.colmap_ransac_trials),
                verbose=bool(args.verbose),
                dry_run=bool(args.dry_run),
            )

        if bool(args.export_alpha_mask) and not alpha_done:
            if bool(args.dry_run):
                print(
                    f"[DRY] export alpha masks -> {scene_dir / str(args.alpha_mask_dir_name)}",
                    flush=True,
                )
            else:
                _write_alpha_masks(
                    scene_dir=scene_dir,
                    alpha_mask_dir_name=str(args.alpha_mask_dir_name),
                    verbose=bool(args.verbose),
                )

        c = _count_outputs(scene_dir=scene_dir)
        print(
            f"[OUT] {scan} (images={c.images}, normals={c.normals}, depths={c.depths})",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
