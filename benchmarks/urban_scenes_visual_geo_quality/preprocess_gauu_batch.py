#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _factor_image_dir_name(factor: float) -> str:
    """Return FriendlySplat naming: images_2 or images_2p5."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from friendly_splat.data.colmap_dataparser import (  # noqa: WPS433
        format_factor_dir_suffix,
    )

    return f"images_{format_factor_dir_suffix(float(factor))}"


def _ensure_images_downsampled(
    *,
    scene_dir: Path,
    factor: float,
    skip_existing: bool,
    dry_run: bool,
) -> list[str]:
    src_dir = scene_dir / "images"
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Missing source images dir: {src_dir}")

    if float(factor) <= 1.0:
        raise ValueError(f"factor must be > 1 for downsampling, got {factor}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(
        p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not images:
        raise FileNotFoundError(f"No images found under: {src_dir}")

    dst_dir = scene_dir / _factor_image_dir_name(float(factor))
    if not bool(dry_run):
        dst_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433
        from PIL import Image  # noqa: WPS433
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "OpenCV, numpy, and Pillow are required for downsampling. "
            "Install them (e.g. `pip install opencv-python numpy pillow`)."
        ) from exc

    pil_resampling = getattr(Image, "Resampling", Image)
    pil_lanczos = pil_resampling.LANCZOS

    wrote = 0
    stems: list[str] = []
    for p in images:
        stems.append(p.stem)
        out = dst_dir / p.name
        if bool(skip_existing) and out.exists():
            continue

        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {p}")

        # Resize in Pillow (Lanczos), but keep the original channel count when possible.
        if img.ndim == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            mode = "RGB"
        else:
            if int(img.shape[-1]) == 4:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                mode = "RGBA"
            elif int(img.shape[-1]) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mode = "RGB"
            else:
                img_rgb = img[..., :3]
                img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
                mode = "RGB"

        h, w = int(img_rgb.shape[0]), int(img_rgb.shape[1])
        new_w = max(1, int(round(float(w) / float(factor))))
        new_h = max(1, int(round(float(h) / float(factor))))
        resized = np.asarray(
            Image.fromarray(img_rgb, mode=mode).resize(
                (new_w, new_h), resample=pil_lanczos
            )
        )

        if mode == "RGBA":
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2BGRA)
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)

        ext = out.suffix.lower()
        params: list[int] = []
        if ext in {".jpg", ".jpeg"}:
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        elif ext == ".png":
            params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]

        if dry_run:
            wrote += 1
            continue
        ok = cv2.imwrite(str(out), resized, params)
        if not ok:
            raise RuntimeError(f"Failed to write downsampled image: {out}")
        wrote += 1

    print(
        f"[{dst_dir.name}] {scene_dir.name}: wrote={wrote} (skip_existing={bool(skip_existing)})",
        flush=True,
    )
    return stems


def _run_moge_normals(
    *,
    scene_dir: Path,
    factor: float,
    out_normal_dir: str,
    model_id: str,
    read_threads: int,
    write_threads: int,
    queue_size: int,
    skip_existing: bool,
    stems: list[str],
    verbose: bool,
    dry_run: bool,
) -> None:
    if bool(skip_existing) and stems:
        normal_dir = scene_dir / str(out_normal_dir)
        have_normals = all((normal_dir / f"{s}.png").exists() for s in stems)
        if have_normals:
            print(f"[skip] {scene_dir.name}: normals already exist under {normal_dir}")
            return

    repo_root = Path(__file__).resolve().parents[2]
    moge_script = repo_root / "tools" / "geometry_prior" / "moge_infer.py"
    if not moge_script.exists():
        raise FileNotFoundError(f"Missing script: {moge_script}")

    cmd: list[str] = [
        sys.executable,
        str(moge_script),
        "--data-dir",
        str(scene_dir),
        "--factor",
        str(float(factor)),
        "--model-id",
        str(model_id),
        "--out-normal-dir",
        str(out_normal_dir),
        "--read-threads",
        str(int(read_threads)),
        "--write-threads",
        str(int(write_threads)),
        "--queue-size",
        str(int(queue_size)),
        # Normals-only: avoid extra COLMAP reads and alignment logic.
        "--no-align-depth-with-colmap",
    ]
    if bool(verbose):
        cmd.append("--verbose")

    print(_format_cmd(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="preprocess_gauu_batch.py",
        description=(
            "Batch preprocessor for GauU-Scene:\n"
            "1) create downsampled images_<suffix>/ (FriendlySplat naming for float factors)\n"
            "2) run MoGe normals-only priors at the same factor.\n\n"
            "Writes into each scene folder:\n"
            "- images_<suffix>/* (downsampled)\n"
            "- moge_normal/*.png (normal prior)\n"
        ),
    )
    parser.add_argument(
        "--gauu-root",
        type=str,
        required=True,
        help="GauU-Scene root directory containing Modern_Building/Residence/Russian_Building.",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        default="default",
        help="default|all|csv of scene names (e.g. Modern_Building,Residence).",
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=3.4175,
        help="Downsample factor (float). Default matches the CityGaussian-style setup for GauU-Scene.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip MoGe when normals already exist for all images (default: on).",
    )
    parser.add_argument(
        "--skip-existing-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip generating downsampled images that already exist (default: on).",
    )
    parser.add_argument(
        "--out-normal-dir",
        type=str,
        default="moge_normal",
        help="Output dir name for normals.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Ruicheng/moge-2-vitl-normal",
        help="MoGe model id.",
    )
    parser.add_argument("--read-threads", type=int, default=8)
    parser.add_argument("--write-threads", type=int, default=8)
    parser.add_argument("--queue-size", type=int, default=32)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without executing.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    gauu_root = Path(str(args.gauu_root)).expanduser().resolve()
    if not gauu_root.exists():
        raise FileNotFoundError(f"gauu_root not found: {gauu_root}")

    scenes_raw = str(args.scenes).strip()
    if scenes_raw.lower() == "default":
        scenes = ["Modern_Building", "Residence", "Russian_Building"]
    elif scenes_raw.lower() == "all":
        scenes = sorted({p.name for p in gauu_root.iterdir() if p.is_dir()})
    else:
        scenes = [s.strip() for s in scenes_raw.split(",") if s.strip()]
    if not scenes:
        raise ValueError("No scenes selected.")

    factor = float(args.factor)
    if not (factor > 1.0):
        raise ValueError(f"--factor must be > 1.0, got {factor}")

    any_failed = False
    for scene in scenes:
        scene_dir = gauu_root / str(scene)
        if not scene_dir.exists():
            print(f"[skip] missing scene dir: {scene_dir}", flush=True)
            continue

        try:
            stems = _ensure_images_downsampled(
                scene_dir=scene_dir,
                factor=factor,
                skip_existing=bool(args.skip_existing_images),
                dry_run=bool(args.dry_run),
            )
            _run_moge_normals(
                scene_dir=scene_dir,
                factor=factor,
                out_normal_dir=str(args.out_normal_dir),
                model_id=str(args.model_id),
                read_threads=int(args.read_threads),
                write_threads=int(args.write_threads),
                queue_size=int(args.queue_size),
                skip_existing=bool(args.skip_existing),
                stems=stems,
                verbose=bool(args.verbose),
                dry_run=bool(args.dry_run),
            )
        except Exception as exc:
            any_failed = True
            print(f"[fail] {scene}: {exc}", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
