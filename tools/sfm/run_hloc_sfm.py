#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run Structure-from-Motion (SfM) using Hierarchical Localization (HLOC).

This tool is designed to be self-contained:
- it reads images from --input-image-dir;
- it writes all outputs under --output-dir;
- it exports a 3DGS-ready COLMAP scene to:
    <output-dir>/images/
    <output-dir>/sparse/0/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y", "t", "on"}:
        return True
    if s in {"false", "0", "no", "n", "f", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {v!r}")


def _format_dir(path: Path) -> str:
    return str(path.expanduser().resolve())


def _ensure_empty_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Path exists (use --overwrite): {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_images_fast(
    image_dir: Path,
    output_dir: Path,
    *,
    image_prefix: str = "frame_",
    overwrite: bool,
) -> Path:
    """Copy original images into output_dir with standardized names."""
    from tqdm import tqdm

    _ensure_empty_dir(output_dir, overwrite=overwrite)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {image_dir}")

    def copy_one(idx_path):
        idx, src_path = idx_path
        dst_path = output_dir / f"{image_prefix}{idx:05d}{src_path.suffix.lower()}"
        shutil.copy2(src_path, dst_path)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(
            tqdm(
                ex.map(copy_one, enumerate(image_paths, start=1)),
                total=len(image_paths),
                unit="img",
            )
        )
    return output_dir


def split_panoramas(
    pano_dir: Path,
    output_dir: Path,
    *,
    image_prefix: str = "frame_",
    downscale: float,
    overwrite: bool,
) -> Path:
    """Split each panorama into five perspective views and save under pano_camera{idx}/."""
    import numpy as np
    import cv2
    from scipy.spatial.transform import Rotation
    from tqdm import tqdm

    from tools.sfm.hloc_utils import PANO_CONFIG

    if downscale < 1.0:
        raise ValueError(f"--pano-downscale must be >= 1.0, got {downscale}")

    _ensure_empty_dir(output_dir, overwrite=overwrite)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    pano_paths = sorted([p for p in pano_dir.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
    if not pano_paths:
        raise RuntimeError(f"No images found in {pano_dir}")

    first_img = cv2.imread(str(pano_paths[0]))
    if first_img is None:
        raise RuntimeError(f"Cannot read image: {pano_paths[0]}")
    pano_h, pano_w = first_img.shape[:2]

    hfov_deg = float(PANO_CONFIG["fov"])
    vfov_deg = float(PANO_CONFIG["fov"])

    rots = []
    for yaw, pitch in PANO_CONFIG["views"]:
        rot = Rotation.from_euler("XY", [-pitch, -yaw], degrees=True).as_matrix()
        rots.append(rot)

    w_virt_raw = int(pano_w * hfov_deg / 360.0)
    h_virt_raw = int(pano_h * vfov_deg / 180.0)
    w_virt = max(1, int(w_virt_raw / float(downscale)))
    h_virt = max(1, int(h_virt_raw / float(downscale)))
    focal = w_virt / (2.0 * float(np.tan(np.deg2rad(hfov_deg) / 2.0)))

    cx, cy = w_virt / 2.0 - 0.5, h_virt / 2.0 - 0.5
    y_grid, x_grid = np.indices((h_virt, w_virt))
    rays = np.stack(
        [
            (x_grid - cx),
            (y_grid - cy),
            np.full_like(x_grid, focal, dtype=np.float32),
        ],
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    def process_one_pano(idx_path):
        idx, src_path = idx_path
        img = cv2.imread(str(src_path))
        if img is None:
            raise RuntimeError(f"Cannot read image: {src_path}")

        for cam_idx, rot in enumerate(rots):
            rays_rotated = rays @ rot
            x, y, z = (
                rays_rotated[..., 0],
                rays_rotated[..., 1],
                rays_rotated[..., 2],
            )
            yaw = np.arctan2(x, z)
            pitch = -np.arctan2(y, np.linalg.norm(rays_rotated[..., [0, 2]], axis=-1))

            u = (1.0 + yaw / np.pi) / 2.0 * pano_w
            v = (1.0 - pitch * 2.0 / np.pi) / 2.0 * pano_h

            perspective_img = cv2.remap(
                img,
                u.astype(np.float32),
                v.astype(np.float32),
                cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_WRAP,
            )

            sub_dir = output_dir / f"pano_camera{cam_idx}"
            sub_dir.mkdir(parents=True, exist_ok=True)
            save_path = sub_dir / f"{image_prefix}{idx:05d}{src_path.suffix.lower()}"
            cv2.imwrite(str(save_path), perspective_img)

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(
            tqdm(
                ex.map(process_one_pano, enumerate(pano_paths, start=1)),
                total=len(pano_paths),
                unit="pano",
            )
        )
    return output_dir


def main(argv: list[str]) -> int:
    # Allow running as a standalone script from the repo root without installation.
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from tools.sfm.hloc_utils import CameraModel, run_hloc

    parser = argparse.ArgumentParser(description="Run SfM using HLOC + pycolmap.")
    parser.add_argument("--input-image-dir", type=Path, required=True, help="Path to original images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory (exported COLMAP scene root).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs under --output-dir.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep intermediate files under <output-dir>/_sfm_work (default: delete).",
    )

    parser.add_argument(
        "--camera-model",
        type=str,
        required=True,
        choices=[m.name for m in CameraModel],
        help="Camera model (e.g., PINHOLE, OPENCV).",
    )
    parser.add_argument(
        "--matching-method",
        type=str,
        default="sequential",
        choices=["exhaustive", "sequential", "retrieval"],
        help="Method for image matching.",
    )
    parser.add_argument("--feature-type", type=str, default="superpoint_aachen")
    parser.add_argument("--matcher-type", type=str, default="superglue")
    parser.add_argument(
        "--retrieval-type",
        type=str,
        default="netvlad",
        choices=["netvlad", "megaloc", "dir", "openibl"],
        help="Global descriptor used for retrieval.",
    )
    parser.add_argument(
        "--num-matched",
        type=int,
        default=50,
        help="Number of matched images for retrieval-based pairing.",
    )
    parser.add_argument(
        "--refine-pixsfm",
        action="store_true",
        help="Refine reconstruction using pixel-perfect-sfm (requires pixsfm).",
    )
    parser.add_argument(
        "--use-single-camera-mode",
        type=_str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Use a single shared camera for all frames (default: True).",
    )
    parser.add_argument(
        "--gpu-ba",
        type=str.lower,
        choices=["on", "off"],
        default="off",
        help="Enable GPU bundle adjustment in pycolmap (default: off).",
    )

    parser.add_argument(
        "--is-panorama",
        action="store_true",
        help="Treat inputs as panoramas and split into a 5-view rig before SfM.",
    )
    parser.add_argument(
        "--pano-downscale",
        type=float,
        default=1.0,
        help="Downscale factor for panorama splitting (>=1).",
    )

    args = parser.parse_args(argv)
    input_image_dir: Path = args.input_image_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    overwrite: bool = bool(args.overwrite)
    keep_work_dir: bool = bool(args.keep_work_dir)

    if not input_image_dir.is_dir():
        raise FileNotFoundError(f"--input-image-dir is not a directory: {input_image_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / "_sfm_work"
    images_work_dir = work_dir / "images"
    hloc_dir = work_dir / "hloc"
    sfm_root = work_dir / "sparse"

    if args.is_panorama:
        working_images_dir = split_panoramas(
            input_image_dir,
            images_work_dir,
            downscale=max(1.0, float(args.pano_downscale)),
            overwrite=overwrite,
        )
    else:
        working_images_dir = copy_images_fast(
            input_image_dir,
            images_work_dir,
            overwrite=overwrite,
        )

    enable_gpu_ba = str(args.gpu_ba).lower() == "on"

    run_hloc(
        image_dir=working_images_dir,
        hloc_dir=hloc_dir,
        sfm_root=sfm_root,
        export_dir=output_dir,
        camera_model=CameraModel[args.camera_model],
        verbose=False,
        matching_method=args.matching_method,
        feature_type=args.feature_type,
        matcher_type=args.matcher_type,
        retrieval_type=args.retrieval_type,
        num_matched=int(args.num_matched),
        refine_pixsfm=bool(args.refine_pixsfm),
        use_single_camera_mode=bool(args.use_single_camera_mode),
        is_panorama=bool(args.is_panorama),
        enable_gpu_ba=enable_gpu_ba,
        overwrite=overwrite,
    )

    if not keep_work_dir and work_dir.exists():
        shutil.rmtree(work_dir)

    print("Done.", flush=True)
    print(f"Exported scene: {_format_dir(output_dir)}", flush=True)
    print(f"- images: {_format_dir(output_dir / 'images')}", flush=True)
    print(f"- sparse: {_format_dir(output_dir / 'sparse' / '0')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

