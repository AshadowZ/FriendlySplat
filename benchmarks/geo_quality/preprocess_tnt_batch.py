#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _ensure_images_2(
    *,
    scene_dir: Path,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    src_dir = scene_dir / "images"
    dst_dir = scene_dir / "images_2"
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Missing source images dir: {src_dir}")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(
        p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    )
    if not images:
        raise FileNotFoundError(f"No images found under: {src_dir}")

    if not bool(dry_run):
        dst_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433
        from PIL import Image  # noqa: WPS433
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "OpenCV and Pillow are required to generate images_2. "
            "Install them (e.g. `pip install opencv-python pillow`)."
        ) from exc
    pil_resampling = getattr(Image, "Resampling", Image)
    pil_lanczos = pil_resampling.LANCZOS

    wrote = 0
    for p in images:
        out = dst_dir / p.name
        if bool(skip_existing) and out.exists():
            continue
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        if img.ndim != 3 or int(img.shape[-1]) != 4:
            raise ValueError(
                f"TnT preprocess expects RGBA images (4 channels), got shape={img.shape} for {p}"
            )
        h, w = int(img.shape[0]), int(img.shape[1])
        new_w = max(1, int(round(float(w) / 2.0)))
        new_h = max(1, int(round(float(h) / 2.0)))
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        resized_rgba = np.asarray(
            Image.fromarray(img_rgba).resize((new_w, new_h), resample=pil_lanczos)
        )
        resized = cv2.cvtColor(resized_rgba, cv2.COLOR_RGBA2BGRA)

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
        f"[images_2] {scene_dir.name}: wrote={wrote} (skip_existing={bool(skip_existing)})",
        flush=True,
    )


def _run_one_scene(*, scene_dir: Path, verbose: bool, dry_run: bool) -> None:
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
        "2",
        "--model-id",
        "Ruicheng/moge-2-vitl-normal",
        "--out-normal-dir",
        "moge_normal",
        "--out-depth-dir",
        "moge_depth",
        "--out-sky-mask-dir",
        "invalid_mask",
        "--save-depth",
        "--save-sky-mask",
        *(["--verbose"] if bool(verbose) else []),
    ]

    print(_format_cmd(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="preprocess_tnt_batch.py",
        description=(
            "Batch preprocessor for Tanks & Temples (TnT):\n"
            "1) create half-resolution images_2/ from images/;\n"
            "2) run MoGe priors on images_2/ (factor=2).\n\n"
            "Writes into each scene folder:\n"
            "- moge_normal/*.png (normal prior)\n"
            "- moge_depth/*.npy (depth prior)\n"
            "- invalid_mask/*.png (255=invalid, 0=valid)\n"
            "Assumes TnT's common flat image layout: <Scene>/images/* (non-recursive)."
        ),
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
        "--scenes",
        type=str,
        default="default",
        help="default|all|csv of scene names (e.g. Barn,Truck).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip a scene if all outputs already exist (default: on).",
    )
    parser.add_argument(
        "--skip-existing-images-2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip generating images_2 files that already exist (default: on).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    data_root = Path(str(args.data_root)).expanduser().resolve()
    if args.tnt_dir is None:
        tnt_dir = data_root / "tnt_dataset" / "tnt"
        print(f"[auto] using tnt_dir={tnt_dir}", flush=True)
    else:
        tnt_dir = Path(str(args.tnt_dir)).expanduser().resolve()
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
        scenes = sorted(
            {
                p.name
                for p in tnt_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            }
        )
    else:
        scenes = [s.strip() for s in scenes_raw.split(",") if s.strip()]
    if not scenes:
        raise ValueError("No scenes selected.")

    any_failed = False
    for scene in scenes:
        scene_dir = tnt_dir / str(scene)
        if not scene_dir.exists():
            print(f"[skip] missing scene dir: {scene_dir}", flush=True)
            continue

        image_dir = scene_dir / "images"
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        stems = (
            [
                p.stem
                for p in sorted(image_dir.iterdir())
                if p.is_file() and p.suffix.lower() in exts
            ]
            if image_dir.is_dir()
            else []
        )
        if not stems:
            print(f"[skip] no images found: {scene_dir / 'images'}", flush=True)
            continue

        try:
            _ensure_images_2(
                scene_dir=scene_dir,
                skip_existing=bool(args.skip_existing_images_2),
                dry_run=bool(args.dry_run),
            )
            if bool(args.skip_existing):
                have_normals = all(
                    (scene_dir / "moge_normal" / f"{s}.png").exists() for s in stems
                )
                have_depths = all(
                    (scene_dir / "moge_depth" / f"{s}.npy").exists() for s in stems
                )
                have_invalid = all(
                    (scene_dir / "invalid_mask" / f"{s}.png").exists() for s in stems
                )
                if have_normals and have_depths and have_invalid:
                    print(f"[skip] {scene} (priors already exist)", flush=True)
                    continue
            _run_one_scene(
                scene_dir=scene_dir,
                verbose=bool(args.verbose),
                dry_run=bool(args.dry_run),
            )
            print(f"[ok] {scene}", flush=True)
        except Exception as exc:
            any_failed = True
            print(f"[fail] {scene}: {exc}", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
