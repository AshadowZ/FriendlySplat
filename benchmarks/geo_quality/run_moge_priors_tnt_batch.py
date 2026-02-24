#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

_MOGE_MODEL_ID = "Ruicheng/moge-2-vitl-normal"


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


def _list_images_flat(*, image_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    if not image_dir.is_dir():
        return []
    return sorted(
        p
        for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )


def _all_exist(*, parent_dir: Path, stems: list[str], suffix: str) -> bool:
    if not stems or not parent_dir.is_dir():
        return False
    return all((parent_dir / f"{s}{suffix}").exists() for s in stems)


def _run_one_scene(*, scene_dir: Path, verbose: bool, dry_run: bool) -> None:
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
        "--out-depth-dir",
        "moge_depth",
        "--out-sky-mask-dir",
        "invalid_mask",
        "--save-depth",
        "--save-sky-mask",
    ]
    if verbose:
        cmd.append("--verbose")

    print(_format_cmd(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_moge_priors_tnt_batch.py",
        description=(
            "Batch runner to generate MoGe priors for Tanks & Temples (TnT). "
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
        help="Root directory containing TnT scenes (e.g. Tanks&Temples-Geo/).",
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
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument("--verbose", action="store_true")
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

    any_failed = False
    for scene in scenes:
        scene_dir = tnt_dir / str(scene)
        if not scene_dir.exists():
            print(f"[skip] missing scene dir: {scene_dir}", flush=True)
            continue

        image_paths = _list_images_flat(image_dir=scene_dir / "images")
        stems = [p.stem for p in image_paths]
        if not stems:
            print(f"[skip] no images found: {scene_dir / 'images'}", flush=True)
            continue

        if bool(args.skip_existing):
            have_normals = _all_exist(parent_dir=scene_dir / "moge_normal", stems=stems, suffix=".png")
            have_depths = _all_exist(parent_dir=scene_dir / "moge_depth", stems=stems, suffix=".npy")
            have_invalid = _all_exist(parent_dir=scene_dir / "invalid_mask", stems=stems, suffix=".png")
            if have_normals and have_depths and have_invalid:
                print(f"[skip] {scene} (priors already exist)", flush=True)
                continue

        try:
            _run_one_scene(scene_dir=scene_dir, verbose=bool(args.verbose), dry_run=bool(args.dry_run))
            print(f"[ok] {scene}", flush=True)
        except Exception as exc:
            any_failed = True
            print(f"[fail] {scene}: {exc}", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

