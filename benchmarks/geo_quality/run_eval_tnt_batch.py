#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import shlex
import subprocess
import sys
from dataclasses import dataclass
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

_TNT_SCENES_360: frozenset[str] = frozenset({"Barn", "Caterpillar", "Ignatius", "Truck"})
_TNT_SCENES_LARGE: frozenset[str] = frozenset({"Meetingroom", "Courthouse"})


def _default_tsdf_params_2dgs(*, scene: str) -> tuple[float, float, float]:
    """Return (voxel_length, sdf_trunc, depth_trunc) aligned with 2DGS's TnT eval."""
    if str(scene) in _TNT_SCENES_LARGE:
        voxel_length = 0.006
        depth_trunc = 4.5
    else:
        voxel_length = 0.004
        depth_trunc = 3.0
    sdf_trunc = 4.0 * voxel_length
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


def _ply_path(*, result_dir: Path, max_steps: int) -> Path:
    return result_dir / "ply" / f"splats_step{int(max_steps):06d}.ply"


def _mesh_path(*, result_dir: Path) -> Path:
    return result_dir / "mesh" / "reconstructed_mesh.ply"


def _run_tsdf_mesh(
    *,
    repo_root: Path,
    ply_path: Path,
    scene_dir: Path,
    data_factor: int,
    device: str,
    interval: int,
    voxel_length: Optional[float],
    sdf_trunc: Optional[float],
    depth_trunc: Optional[float],
    post_process_clusters: int,
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
        "--data_factor",
        str(int(data_factor)),
        "--device",
        str(device),
        "--interval",
        str(int(interval)),
        "--post_process_clusters",
        str(int(post_process_clusters)),
        "--output_dir",
        str(output_dir),
    ]
    if voxel_length is not None:
        cmd += ["--voxel_length", str(float(voxel_length))]
    if sdf_trunc is not None:
        cmd += ["--sdf_trunc", str(float(sdf_trunc))]
    if depth_trunc is not None:
        cmd += ["--depth_trunc", str(float(depth_trunc))]

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
    dry_run: bool,
) -> None:
    eval_py = repo_root / "benchmarks" / "geo_quality" / "tnt_eval" / "run.py"
    if not eval_py.exists():
        raise FileNotFoundError(f"Missing TnT eval script: {eval_py}")

    cmd = [
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
    ]
    print(f"[tnt] {_format_cmd(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(eval_py.parent))


def _load_results_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_cell(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _write_summary_md(*, rows: list[_EvalRow], summary_path: Path, exp_name: str, max_steps: int) -> None:
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
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument(
        "--tnt-dir",
        type=str,
        default=None,
        help="Optional explicit TnT scene root dir. If omitted, auto-detect under --data-root.",
    )
    parser.add_argument("--out-dir-name", type=str, default="geo_benchmark")
    parser.add_argument("--exp-name", type=str, default="tnt_default")
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--scenes", type=str, default="default")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--voxel-length", type=float, default=None)
    parser.add_argument("--sdf-trunc", type=float, default=None)
    parser.add_argument("--depth-trunc", type=float, default=None)
    parser.add_argument("--post-process-clusters", type=int, default=1)

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

    repo_root = Path(__file__).resolve().parents[2]
    results_root = data_root / str(args.out_dir_name) / "TnT"

    rows: list[_EvalRow] = []
    any_failed = False
    for scene in scenes:
        scene_dir = tnt_dir / str(scene)
        result_dir = results_root / str(scene) / str(args.exp_name)
        ply = _ply_path(result_dir=result_dir, max_steps=int(args.max_steps))
        if not ply.exists():
            print(f"[skip] missing ply: {ply}", flush=True)
            any_failed = True
            continue

        mesh = _mesh_path(result_dir=result_dir)
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

            _run_tsdf_mesh(
                repo_root=repo_root,
                ply_path=ply,
                scene_dir=scene_dir,
                data_factor=1,
                device=str(args.device),
                interval=int(args.interval),
                voxel_length=voxel_length,
                sdf_trunc=sdf_trunc,
                depth_trunc=depth_trunc,
                post_process_clusters=int(args.post_process_clusters),
                dry_run=bool(args.dry_run),
            )

        eval_out_dir = result_dir / "eval_tnt"
        results_json = eval_out_dir / "results.json"
        if bool(args.skip_existing_eval) and results_json.exists():
            stats = _load_results_json(results_json)
            rows.append(
                _EvalRow(
                    scene=str(scene),
                    precision=float(stats.get("precision")) if "precision" in stats else None,
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
                precision=float(stats.get("precision")) if "precision" in stats else None,
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

    summary_path = data_root / str(args.out_dir_name) / f"summary_tnt_{args.exp_name}.md"
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
