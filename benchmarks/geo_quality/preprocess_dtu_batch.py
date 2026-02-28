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
    export_alpha_mask: bool,
    alpha_mask_dir_name: str,
    verbose: bool,
    dry_run: bool,
) -> list[str]:
    src_dir = scene_dir / "images"
    dst_dir = scene_dir / "images_2"
    if not src_dir.exists() or not src_dir.is_dir():
        raise FileNotFoundError(f"Missing source images dir: {src_dir}")

    try:
        import cv2  # noqa: WPS433
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError(
            "OpenCV is required to generate images_2. Install it (e.g. `pip install opencv-python`)."
        ) from exc

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = sorted(p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(f"No images found under: {src_dir}")

    if not bool(dry_run):
        dst_dir.mkdir(parents=True, exist_ok=True)
        if bool(export_alpha_mask):
            (scene_dir / str(alpha_mask_dir_name)).mkdir(parents=True, exist_ok=True)

    wrote_images = 0
    wrote_masks = 0
    stems: list[str] = []
    for p in images:
        stems.append(str(p.stem))
        out = dst_dir / p.name
        mask_out = scene_dir / str(alpha_mask_dir_name) / f"{p.stem}.png"
        need_image = not (bool(skip_existing) and out.exists())
        need_mask = bool(export_alpha_mask)
        if not (need_image or need_mask):
            continue

        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {p}")
        # Debug-friendly downsample path:
        # - drop alpha (RGBA -> RGB)
        # - use Lanczos resampling on 3 channels
        h, w = int(img.shape[0]), int(img.shape[1])
        alpha = img[..., 3] if (img.ndim == 3 and int(img.shape[-1]) == 4) else None

        if img.ndim == 2:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif int(img.shape[-1]) == 4:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        elif int(img.shape[-1]) >= 3:
            img_bgr = img[..., :3]
        else:
            img_bgr = img
        new_w = max(1, int(round(float(w) / 2.0)))
        new_h = max(1, int(round(float(h) / 2.0)))
        resized_bgr = None
        if need_image:
            resized_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        mask_u8 = None
        if need_mask:
            import numpy as np

            # Default: no alpha -> treat all pixels as valid (mask=0).
            mask_u8 = np.zeros((new_h, new_w), dtype=np.uint8)
            if alpha is not None:
                if np.issubdtype(alpha.dtype, np.integer):
                    denom = float(np.iinfo(alpha.dtype).max)
                    alpha_f = alpha.astype(np.float32) / max(denom, 1.0)
                else:
                    alpha_f = alpha.astype(np.float32, copy=False)
                    maxv = float(np.nanmax(alpha_f)) if alpha_f.size > 0 else 0.0
                    if maxv > 1.0:
                        alpha_f = alpha_f / max(maxv, 1e-6)
                alpha_ds = cv2.resize(alpha_f, (new_w, new_h), interpolation=cv2.INTER_AREA)
                mask_u8 = (alpha_ds < 0.5).astype(np.uint8) * 255

        if dry_run:
            if need_image:
                wrote_images += 1
            if need_mask:
                wrote_masks += 1
            continue

        if need_image:
            assert resized_bgr is not None
            ext = out.suffix.lower()
            params: list[int] = []
            if ext in {".jpg", ".jpeg"}:
                params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            elif ext == ".png":
                params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]
            ok = cv2.imwrite(str(out), resized_bgr, params)
            if not ok:
                raise RuntimeError(f"Failed to write downsampled image: {out}")
            wrote_images += 1

        if need_mask:
            assert mask_u8 is not None
            if not cv2.imwrite(str(mask_out), mask_u8):
                raise RuntimeError(f"cv2.imwrite failed: {mask_out}")
            wrote_masks += 1
            if verbose:
                print(f"[MASK] {mask_out}", flush=True)

    msg = f"[images_2] {scene_dir.name}: wrote={wrote_images} (skip_existing={bool(skip_existing)})"
    if export_alpha_mask:
        msg += f", invalid_mask={wrote_masks}"
    print(msg, flush=True)
    return stems


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
        "--no-align-depth-with-colmap",
        *(["--verbose"] if bool(verbose) else []),
    ]

    print(_format_cmd(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="preprocess_dtu_batch.py",
        description=(
            "DTU preprocessor (half resolution, factor=2):\n"
            "1) resize images/ -> images_2/;\n"
            "2) run MoGe inference on images_2/ to generate moge_normal/;\n"
            "3) optionally export alpha-derived invalid_mask/ from images/ (alpha), downsampled to images_2/.\n"
        ),
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
        help="Skip scans that already have complete outputs (default: on).",
    )
    parser.add_argument(
        "--skip-existing-images-2",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip generating images_2 files that already exist (default: on).",
    )
    parser.add_argument(
        "--export-alpha-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Export alpha-derived background masks from images/ into a per-scan folder. "
            "Outputs to 'invalid_mask/' (white=invalid/background, black=valid/foreground)."
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

    any_failed = False
    for scan in scans:
        scene_dir = dtu_dir / scan
        if not scene_dir.exists():
            print(f"[skip] missing scan dir: {scene_dir}", flush=True)
            any_failed = True
            continue
        try:
            stems = _ensure_images_2(
                scene_dir=scene_dir,
                skip_existing=bool(args.skip_existing_images_2),
                export_alpha_mask=bool(args.export_alpha_mask),
                alpha_mask_dir_name="invalid_mask",
                verbose=bool(args.verbose),
                dry_run=bool(args.dry_run),
            )

            marker_normals = scene_dir / "moge_normal" / ".factor2"
            marker_masks = scene_dir / "invalid_mask" / ".factor2"

            moge_done = bool(marker_normals.exists()) and all(
                (scene_dir / "moge_normal" / f"{s}.png").exists() for s in stems
            )
            mask_done = (
                (not bool(args.export_alpha_mask))
                or (
                    bool(marker_masks.exists())
                    and all((scene_dir / "invalid_mask" / f"{s}.png").exists() for s in stems)
                )
            )

            if bool(args.skip_existing) and moge_done and mask_done:
                print(f"[skip] {scan} (already done)", flush=True)
                continue

            print(f"[run] {scan}", flush=True)
            if not moge_done:
                _run_one_scene(scene_dir=scene_dir, verbose=bool(args.verbose), dry_run=bool(args.dry_run))
                if not bool(args.dry_run):
                    marker_normals.parent.mkdir(parents=True, exist_ok=True)
                    marker_normals.write_text("factor=2\n", encoding="utf-8")

            if bool(args.export_alpha_mask) and not mask_done and not bool(args.dry_run):
                if not bool(args.dry_run):
                    marker_masks.parent.mkdir(parents=True, exist_ok=True)
                    marker_masks.write_text("factor=2\n", encoding="utf-8")

            print(f"[ok] {scan}", flush=True)
        except Exception as exc:
            any_failed = True
            print(f"[fail] {scan}: {exc}", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
