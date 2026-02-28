#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _default_tsdf_params_2dgs(*, scene: str) -> tuple[float, float, float]:
    """Return (voxel_length, sdf_trunc, depth_trunc) default TSDF parameters."""
    # Align with 2DGS's TnT benchmark presets:
    # - scripts/tnt_eval.py (render stage) uses two fixed TSDF configs.
    s = str(scene)
    tnt_360 = {"Barn", "Caterpillar", "Ignatius", "Truck"}
    # "Church" is not used by 2DGS's TNT benchmark script, but we treat it as a
    # large-scale scene here for consistency with its tau and typical usage.
    tnt_large = {"Meetingroom", "Courthouse", "Church"}

    if s in tnt_large:
        voxel_length = 0.006
        sdf_trunc = 0.024
        depth_trunc = 4.5
        return voxel_length, sdf_trunc, depth_trunc
    if s in tnt_360:
        voxel_length = 0.004
        sdf_trunc = 0.016
        depth_trunc = 3.0
        return voxel_length, sdf_trunc, depth_trunc

    # Fallback for non-standard scenes: use the 360 preset.
    voxel_length = 0.004
    sdf_trunc = 0.016
    depth_trunc = 3.0
    return voxel_length, sdf_trunc, depth_trunc


@dataclass(frozen=True)
class _EvalRow:
    scene: str
    precision: Optional[float]
    recall: Optional[float]
    fscore: Optional[float]
    tau: Optional[float]
    result_dir: str


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _safe_rmtree_under(*, path: Path, root: Path) -> None:
    """Delete a directory tree only if it's under a given root (best-effort safety)."""
    p = Path(path).expanduser().resolve()
    r = Path(root).expanduser().resolve()
    try:
        p.relative_to(r)
    except Exception as e:
        raise ValueError(
            f"Refusing to delete path outside root (path={p}, root={r})."
        ) from e
    if p.exists() and p.is_dir():
        shutil.rmtree(p)


def _prepare_tsdf_mask_from_invalid_mask(
    *,
    scene_dir: Path,
    result_dir: Path,
    render_factor: int,
    invalid_mask_dir_name: str = "invalid_mask",
) -> Optional[Path]:
    """Invert invalid_mask into a TSDF object mask (255=keep/foreground, 0=discard)."""
    try:
        import cv2  # noqa: WPS433
        import numpy as np  # noqa: WPS433
    except ModuleNotFoundError:  # pragma: no cover
        print("[mask] disabled (missing dependency: cv2/numpy)", flush=True)
        return None

    invalid_dir = scene_dir / str(invalid_mask_dir_name)
    if not invalid_dir.exists() or not invalid_dir.is_dir():
        return None

    # Build masks at the TSDF integration resolution to avoid downstream resizing.
    factor = int(render_factor)
    image_dir = scene_dir / (f"images_{factor}" if factor > 1 else "images")
    if not image_dir.is_dir():
        image_dir = scene_dir / "images"
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = (
        sorted(
            p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
        )
        if image_dir.is_dir()
        else []
    )
    if not images:
        return None

    out_dir = result_dir / "mesh" / "tsdf_obj_mask"
    out_dir.mkdir(parents=True, exist_ok=True)

    missing = 0
    for img in images:
        stem = img.stem
        src = invalid_dir / f"{stem}.png"
        dst = out_dir / f"{stem}.png"
        im = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(f"Failed to read image to infer size: {img}")
        h, w = int(im.shape[0]), int(im.shape[1])
        if not src.exists():
            missing += 1
            # Keep all pixels if invalid mask is missing for this frame.
            out = np.full((h, w), 255, dtype=np.uint8)
        else:
            m = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if m is None:
                raise FileNotFoundError(f"Failed to read invalid mask: {src}")
            if m.ndim == 3:
                m = m[:, :, 0]
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            # invalid_mask convention: 255=invalid, 0=valid.
            valid = m.astype(np.uint8, copy=False) < 128
            out = np.where(valid, 255, 0).astype(np.uint8)
        if out.ndim == 3:
            out = out[:, :, 0]
        out = np.ascontiguousarray(out.astype(np.uint8, copy=False))
        if not cv2.imwrite(str(dst), out):
            raise RuntimeError(
                f"Failed to write TSDF mask: {dst} "
                f"(shape={tuple(out.shape)}, dtype={out.dtype}, contig={out.flags['C_CONTIGUOUS']})"
            )

    print(
        f"[mask] enabled (invalid_mask inverted -> {out_dir}, missing={missing}/{len(images)})",
        flush=True,
    )
    return out_dir


def _run_tsdf_mesh(
    *,
    repo_root: Path,
    ply_path: Path,
    scene_dir: Path,
    render_factor: int,
    device: str,
    interval: int,
    voxel_length: Optional[float],
    sdf_trunc: Optional[float],
    depth_trunc: Optional[float],
    post_process_clusters: int,
    tsdf_mask_dir: Optional[Path],
    dry_run: bool,
) -> None:
    mesh_script = repo_root / "tools" / "mesh" / "tsdf_mesh_from_ply.py"
    if not mesh_script.exists():
        raise FileNotFoundError(f"Missing mesh script: {mesh_script}")

    output_dir = ply_path.parent.parent / "mesh"
    cmd: list[str] = [
        sys.executable,
        str(mesh_script),
        "--ply_path",
        str(ply_path),
        "--data_dir",
        str(scene_dir),
        "--resolution",
        str(int(render_factor)),
        "--device",
        str(device),
        "--interval",
        str(int(interval)),
        "--post_process_clusters",
        str(int(post_process_clusters)),
        "--output_dir",
        str(output_dir),
        *(
            ["--voxel_length", str(float(voxel_length))]
            if voxel_length is not None
            else []
        ),
        *(["--sdf_trunc", str(float(sdf_trunc))] if sdf_trunc is not None else []),
        *(
            ["--depth_trunc", str(float(depth_trunc))]
            if depth_trunc is not None
            else []
        ),
        *(
            ["--mask_dir", str(tsdf_mask_dir), "--mask_dilate", "0"]
            if tsdf_mask_dir is not None
            else []
        ),
    ]

    print(f"[mesh] {_format_cmd(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def _run_tnt_eval(
    *,
    repo_root: Path,
    dataset_dir: Path,
    traj_path: Path,
    mesh_path: Path,
    out_dir: Path,
    save_eval_vis: bool,
    dry_run: bool,
) -> None:
    eval_py = repo_root / "benchmarks" / "geo_quality" / "tnt_eval" / "run.py"
    if not eval_py.exists():
        raise FileNotFoundError(f"Missing TnT eval script: {eval_py}")

    cmd: list[str] = [
        sys.executable,
        str(eval_py),
        "--dataset-dir",
        str(dataset_dir),
        "--traj-path",
        str(traj_path),
        "--ply-path",
        str(mesh_path),
        "--out-dir",
        str(out_dir),
        *(["--save-eval-vis"] if bool(save_eval_vis) else []),
    ]
    print(f"[tnt] {_format_cmd(cmd)}", flush=True)
    if dry_run:
        return
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "4"
    subprocess.run(cmd, check=True, cwd=str(eval_py.parent), env=env)


def _load_results_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_cell(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _write_summary_md(
    *, rows: list[_EvalRow], summary_path: Path, exp_name: str, max_steps: int
) -> None:
    rows_sorted = sorted(rows, key=lambda r: str(r.scene).lower())
    valid_f1 = [float(r.fscore) for r in rows_sorted if r.fscore is not None]
    valid_p = [float(r.precision) for r in rows_sorted if r.precision is not None]
    valid_r = [float(r.recall) for r in rows_sorted if r.recall is not None]

    mean_f1 = (sum(valid_f1) / len(valid_f1)) if valid_f1 else None
    mean_p = (sum(valid_p) / len(valid_p)) if valid_p else None
    mean_r = (sum(valid_r) / len(valid_r)) if valid_r else None

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# TnT geo benchmark summary: `{exp_name}`")
    lines.append("")
    lines.append(f"- PLY step: `{int(max_steps)}`")
    lines.append("")
    lines.append("| Scene | tau | precision | recall | fscore |")
    lines.append("|---|---:|---:|---:|---:|")
    for r in rows_sorted:
        lines.append(
            f"| {r.scene} | {_fmt_cell(r.tau)} | {_fmt_cell(r.precision)} | {_fmt_cell(r.recall)} | {_fmt_cell(r.fscore)} |"
        )
    lines.append(
        f"| **Mean** | N/A | **{_fmt_cell(mean_p)}** | **{_fmt_cell(mean_r)}** | **{_fmt_cell(mean_f1)}** |"
    )
    lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_eval_tnt_batch.py",
        description="Batch Tanks & Temples (TnT) mesh + F1 evaluation from FriendlySplat outputs.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help=(
            "Root directory containing the TnT dataset folder "
            "(expects tnt_dataset/tnt/ under this root) and the FriendlySplat outputs."
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
            "Output directory used by run_train_tnt_batch.py under data-root. "
            "Default: benchmark/geo_benchmark/tnt_benchmark"
        ),
    )
    parser.add_argument("--exp-name", type=str, default="tnt_default")
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--scenes", type=str, default="default")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument(
        "--tsdf-render-factor",
        type=int,
        default=2,
        help="Render downscale factor for TSDF meshing (forwarded to tsdf_mesh_from_ply.py as --resolution).",
    )
    parser.add_argument("--voxel-length", type=float, default=None)
    parser.add_argument("--sdf-trunc", type=float, default=None)
    parser.add_argument("--depth-trunc", type=float, default=None)
    parser.add_argument("--post-process-clusters", type=int, default=1)
    parser.add_argument(
        "--save-eval-vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save precision/recall colored point clouds under each eval_tnt/ dir (default: on).",
    )

    parser.add_argument(
        "--skip-existing-mesh",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip TSDF meshing if reconstructed mesh already exists (default: on).",
    )
    parser.add_argument(
        "--skip-existing-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip eval if results.json already exists (default: on).",
    )
    parser.add_argument(
        "--delete-mesh-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Delete <result_dir>/mesh/cache after each successfully evaluated scene (default: on).",
    )
    parser.add_argument("--dry-run", action="store_true")
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

    repo_root = Path(__file__).resolve().parents[2]
    rows: list[_EvalRow] = []
    any_failed = False
    results_root = data_root / str(args.out_dir_name)
    for scene in scenes:
        scene_dir = tnt_dir / str(scene)
        result_dir = results_root / str(scene) / str(args.exp_name)
        ply = result_dir / "ply" / f"splats_step{int(args.max_steps):06d}.ply"
        if not ply.exists():
            print(f"[skip] missing ply: {ply}", flush=True)
            any_failed = True
            continue

        mesh = result_dir / "mesh" / "tsdf_mesh_post.ply"
        if not (bool(args.skip_existing_mesh) and mesh.exists()):
            voxel_length = args.voxel_length
            sdf_trunc = args.sdf_trunc
            depth_trunc = args.depth_trunc
            if voxel_length is None or sdf_trunc is None or depth_trunc is None:
                v, s, d = _default_tsdf_params_2dgs(scene=str(scene))
                if voxel_length is None:
                    voxel_length = v
                if sdf_trunc is None:
                    sdf_trunc = s
                if depth_trunc is None:
                    depth_trunc = d

            tsdf_mask_dir = _prepare_tsdf_mask_from_invalid_mask(
                scene_dir=scene_dir,
                result_dir=result_dir,
                render_factor=int(args.tsdf_render_factor),
            )
            _run_tsdf_mesh(
                repo_root=repo_root,
                ply_path=ply,
                scene_dir=scene_dir,
                render_factor=int(args.tsdf_render_factor),
                device=str(args.device),
                interval=int(args.interval),
                voxel_length=float(voxel_length),
                sdf_trunc=float(sdf_trunc),
                depth_trunc=float(depth_trunc),
                post_process_clusters=int(args.post_process_clusters),
                tsdf_mask_dir=tsdf_mask_dir,
                dry_run=bool(args.dry_run),
            )
            if tsdf_mask_dir is not None and not bool(args.dry_run):
                _safe_rmtree_under(path=tsdf_mask_dir, root=result_dir)
                print(f"[mask] deleted {tsdf_mask_dir}", flush=True)

        eval_out_dir = result_dir / "eval_tnt"
        results_json = eval_out_dir / "results.json"
        if bool(args.skip_existing_eval) and results_json.exists():
            stats = _load_results_json(results_json)
            rows.append(
                _EvalRow(
                    scene=str(scene),
                    precision=float(stats.get("precision"))
                    if "precision" in stats
                    else None,
                    recall=float(stats.get("recall")) if "recall" in stats else None,
                    fscore=float(stats.get("fscore")) if "fscore" in stats else None,
                    tau=float(stats.get("tau")) if "tau" in stats else None,
                    result_dir=str(result_dir),
                )
            )
            print(f"[done] {scene}: {results_json}", flush=True)
            if bool(args.delete_mesh_cache) and not bool(args.dry_run):
                cache_dir = result_dir / "mesh" / "cache"
                if cache_dir.exists() and cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
                    print(f"[cache] deleted {cache_dir}", flush=True)
            continue

        traj_path = scene_dir / f"{scene}_COLMAP_SfM.log"
        if not traj_path.exists():
            print(f"[skip] missing traj: {traj_path}", flush=True)
            any_failed = True
            continue

        if not bool(args.dry_run):
            eval_out_dir.mkdir(parents=True, exist_ok=True)
        _run_tnt_eval(
            repo_root=repo_root,
            dataset_dir=scene_dir,
            traj_path=traj_path,
            mesh_path=mesh,
            out_dir=eval_out_dir,
            save_eval_vis=bool(args.save_eval_vis),
            dry_run=bool(args.dry_run),
        )

        if bool(args.dry_run):
            continue
        if not results_json.exists():
            print(f"[fail] missing results.json: {results_json}", flush=True)
            any_failed = True
            continue
        stats = _load_results_json(results_json)
        rows.append(
            _EvalRow(
                scene=str(scene),
                precision=float(stats.get("precision"))
                if "precision" in stats
                else None,
                recall=float(stats.get("recall")) if "recall" in stats else None,
                fscore=float(stats.get("fscore")) if "fscore" in stats else None,
                tau=float(stats.get("tau")) if "tau" in stats else None,
                result_dir=str(result_dir),
            )
        )
        print(f"[ok] {scene}", flush=True)
        if bool(args.delete_mesh_cache) and not bool(args.dry_run):
            cache_dir = result_dir / "mesh" / "cache"
            if cache_dir.exists() and cache_dir.is_dir():
                shutil.rmtree(cache_dir)
                print(f"[cache] deleted {cache_dir}", flush=True)

    summary_path = data_root / str(args.out_dir_name) / f"summary_{args.exp_name}.md"
    if rows and not bool(args.dry_run):
        _write_summary_md(
            rows=rows,
            summary_path=summary_path,
            exp_name=str(args.exp_name),
            max_steps=int(args.max_steps),
        )
        print(f"[summary] wrote {summary_path}", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
