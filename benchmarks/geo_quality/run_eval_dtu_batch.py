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
class _EvalRow:
    scan: str
    mean_d2s: Optional[float]
    mean_s2d: Optional[float]
    overall: Optional[float]
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


def _normalize_scan_name(name: str) -> str:
    s = str(name).strip()
    if not s:
        raise ValueError("Empty scan name.")
    s = s.lower()
    if s.startswith("scan"):
        return f"scan{int(s[4:]):d}"
    return f"scan{int(s):d}"


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
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
    post_process_clusters: int,
    tsdf_mask_dir_name: Optional[str],
    tsdf_mask_threshold: float,
    tsdf_mask_dilate: int,
    dry_run: bool,
) -> None:
    mesh_script = repo_root / "tools" / "mesh" / "tsdf_mesh_from_ply.py"
    if not mesh_script.exists():
        raise FileNotFoundError(f"Missing mesh script: {mesh_script}")

    output_dir = ply_path.parent.parent / "mesh"
    cmd = [
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
        "--voxel_length",
        str(float(voxel_length)),
        "--sdf_trunc",
        str(float(sdf_trunc)),
        "--depth_trunc",
        str(float(depth_trunc)),
        "--post_process_clusters",
        str(int(post_process_clusters)),
        "--output_dir",
        str(output_dir),
    ]
    if tsdf_mask_dir_name is not None:
        cmd.extend(
            [
                "--mask_dir",
                str(tsdf_mask_dir_name),
                "--mask_threshold",
                str(float(tsdf_mask_threshold)),
                "--mask_dilate",
                str(int(tsdf_mask_dilate)),
            ]
        )

    print(f"[mesh] {_format_cmd(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def _run_dtu_eval(
    *,
    eval_script: Path,
    dtu_official_dir: Path,
    dtu_mask_root: Path,
    scan: str,
    mesh_path: Path,
    out_dir: Path,
    dry_run: bool,
) -> None:
    scan_id = int(str(scan).replace("scan", ""))

    cmd = [
        sys.executable,
        str(eval_script),
        "--input_mesh",
        str(mesh_path),
        "--scan_id",
        str(int(scan_id)),
        "--output_dir",
        str(out_dir),
        "--mask_dir",
        str(dtu_mask_root),
        "--DTU",
        str(dtu_official_dir),
    ]
    print(f"[eval] {_format_cmd(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(eval_script.parent))


def _load_results_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_cell(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.4f}"


def _write_summary_md(
    *,
    rows: list[_EvalRow],
    summary_path: Path,
    exp_name: str,
    dtu_official_dir: Path,
    eval_script: Path,
    max_steps: int,
) -> None:
    rows_sorted = sorted(rows, key=lambda r: int(str(r.scan).replace("scan", "")))
    valid_overall = [float(r.overall) for r in rows_sorted if r.overall is not None]
    valid_d2s = [float(r.mean_d2s) for r in rows_sorted if r.mean_d2s is not None]
    valid_s2d = [float(r.mean_s2d) for r in rows_sorted if r.mean_s2d is not None]

    mean_overall = (sum(valid_overall) / len(valid_overall)) if valid_overall else None
    mean_d2s = (sum(valid_d2s) / len(valid_d2s)) if valid_d2s else None
    mean_s2d = (sum(valid_s2d) / len(valid_s2d)) if valid_s2d else None

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"# DTU geo benchmark summary: `{exp_name}`")
    lines.append("")
    lines.append(f"- Eval script: `{eval_script}`")
    lines.append(f"- DTU official dir: `{dtu_official_dir}`")
    lines.append(f"- PLY step: `{int(max_steps)}`")
    lines.append("")
    lines.append("| Scan | mean_d2s | mean_s2d | overall |")
    lines.append("|---:|---:|---:|---:|")
    for r in rows_sorted:
        lines.append(f"| {r.scan} | {_fmt_cell(r.mean_d2s)} | {_fmt_cell(r.mean_s2d)} | {_fmt_cell(r.overall)} |")
    lines.append(
        f"| **Mean** | **{_fmt_cell(mean_d2s)}** | **{_fmt_cell(mean_s2d)}** | **{_fmt_cell(mean_overall)}** |"
    )
    lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="run_eval_dtu_batch.py",
        description="Batch DTU geometry evaluation from FriendlySplat outputs.",
    )
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--dtu-dir-name", type=str, default="DTU")
    parser.add_argument("--out-dir-name", type=str, default="geo_benchmark")
    parser.add_argument("--exp-name", type=str, default="dtu_moge_priors")
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument(
        "--scans",
        type=str,
        default="default",
        help="default|all|csv of scan ids/names",
    )
    parser.add_argument("--exclude-scans", type=str, default=None)

    parser.add_argument(
        "--dtu-official-dir",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to the official DTU evaluation data (ObsMask/, Points/stl/, ...). "
            "If omitted, the script will try to auto-detect it from common locations "
            "under --data-root (e.g. <data-root>/DTU/eval_dtu)."
        ),
    )
    parser.add_argument(
        "--eval-script",
        type=str,
        default=None,
        help=(
            "Path to eval_dtu/evaluate_single_scene.py "
            "(default: <data-root>/DTU/eval_dtu/evaluate_single_scene.py)."
        ),
    )

    # Mesh extraction params.
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--voxel-length", type=float, default=0.004)
    parser.add_argument("--sdf-trunc", type=float, default=0.016)
    parser.add_argument("--depth-trunc", type=float, default=3.0)
    parser.add_argument("--post-process-clusters", type=int, default=1)
    parser.add_argument(
        "--tsdf-use-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply per-frame object mask during TSDF fusion (default: on).",
    )
    parser.add_argument(
        "--tsdf-mask-dir-name",
        type=str,
        default="mask",
        help="Mask dir name under each DTU scan scene dir (e.g. scan24/mask).",
    )
    parser.add_argument("--tsdf-mask-threshold", type=float, default=0.5)
    parser.add_argument("--tsdf-mask-dilate", type=int, default=0)
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
        help="Delete <result_dir>/mesh/cache after each successfully evaluated scan (default: on).",
    )
    parser.add_argument("--dry-run", action="store_true")
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

    dtu_official_dir: Optional[Path]
    if args.dtu_official_dir is None:
        candidates = [
            dtu_dir / "eval_dtu",
            data_root / "DTU_Official",
            data_root / "Official_DTU_Dataset",
            data_root / "Offical_DTU_Dataset",  # common misspelling in some repos
        ]
        dtu_official_dir = None
        for cand in candidates:
            if not cand.exists():
                continue
            if (cand / "ObsMask").exists() and (cand / "Points" / "stl").exists():
                dtu_official_dir = cand
                break
        if dtu_official_dir is None:
            tried = ", ".join(str(p) for p in candidates)
            raise FileNotFoundError(
                "DTU official eval data not found. Pass --dtu-official-dir explicitly. "
                f"Tried: {tried}"
            )
        print(f"[auto] using dtu_official_dir={dtu_official_dir}", flush=True)
    else:
        dtu_official_dir = Path(str(args.dtu_official_dir)).expanduser().resolve()
        if not dtu_official_dir.exists():
            raise FileNotFoundError(f"DTU official dir not found: {dtu_official_dir}")

    eval_script = (
        Path(str(args.eval_script)).expanduser().resolve()
        if args.eval_script is not None
        else (dtu_dir / "eval_dtu" / "evaluate_single_scene.py")
    )
    if not eval_script.exists():
        raise FileNotFoundError(f"Eval script not found: {eval_script}")

    repo_root = Path(__file__).resolve().parents[2]
    results_root = data_root / str(args.out_dir_name) / "DTU"

    rows: list[_EvalRow] = []
    any_failed = False
    for scan in scans:
        scene_dir = dtu_dir / scan
        result_dir = results_root / scan / str(args.exp_name)
        ply = _ply_path(result_dir=result_dir, max_steps=int(args.max_steps))
        if not ply.exists():
            print(f"[skip] missing ply: {ply}", flush=True)
            any_failed = True
            continue

        mesh = _mesh_path(result_dir=result_dir)
        if not (bool(args.skip_existing_mesh) and mesh.exists()):
            tsdf_mask_dir_name: Optional[str] = None
            if bool(args.tsdf_use_mask):
                cand = scene_dir / str(args.tsdf_mask_dir_name)
                if cand.exists() and cand.is_dir():
                    tsdf_mask_dir_name = str(args.tsdf_mask_dir_name)
                else:
                    print(f"[mask] disabled (missing dir): {cand}", flush=True)
            _run_tsdf_mesh(
                repo_root=repo_root,
                ply_path=ply,
                scene_dir=scene_dir,
                data_factor=1,
                device=str(args.device),
                interval=int(args.interval),
                voxel_length=float(args.voxel_length),
                sdf_trunc=float(args.sdf_trunc),
                depth_trunc=float(args.depth_trunc),
                post_process_clusters=int(args.post_process_clusters),
                tsdf_mask_dir_name=tsdf_mask_dir_name,
                tsdf_mask_threshold=float(args.tsdf_mask_threshold),
                tsdf_mask_dilate=int(args.tsdf_mask_dilate),
                dry_run=bool(args.dry_run),
            )

        eval_out_dir = result_dir / "eval_dtu"
        results_json = eval_out_dir / "results.json"
        if bool(args.skip_existing_eval) and results_json.exists():
            stats = _load_results_json(results_json)
            rows.append(
                _EvalRow(
                    scan=scan,
                    mean_d2s=float(stats.get("mean_d2s")) if "mean_d2s" in stats else None,
                    mean_s2d=float(stats.get("mean_s2d")) if "mean_s2d" in stats else None,
                    overall=float(stats.get("overall")) if "overall" in stats else None,
                    result_dir=str(result_dir),
                )
            )
            print(f"[done] {scan}: {results_json}", flush=True)
            if bool(args.delete_mesh_cache) and not bool(args.dry_run):
                cache_dir = result_dir / "mesh" / "cache"
                if cache_dir.exists() and cache_dir.is_dir():
                    shutil.rmtree(cache_dir)
                    print(f"[cache] deleted {cache_dir}", flush=True)
            continue

        _run_dtu_eval(
            eval_script=eval_script,
            dtu_official_dir=dtu_official_dir,
            dtu_mask_root=dtu_dir,
            scan=scan,
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
                scan=scan,
                mean_d2s=float(stats.get("mean_d2s")) if "mean_d2s" in stats else None,
                mean_s2d=float(stats.get("mean_s2d")) if "mean_s2d" in stats else None,
                overall=float(stats.get("overall")) if "overall" in stats else None,
                result_dir=str(result_dir),
            )
        )
        print(f"[ok] {scan}", flush=True)
        if bool(args.delete_mesh_cache) and not bool(args.dry_run):
            cache_dir = result_dir / "mesh" / "cache"
            if cache_dir.exists() and cache_dir.is_dir():
                shutil.rmtree(cache_dir)
                print(f"[cache] deleted {cache_dir}", flush=True)

    # Summary.
    summary_path = data_root / str(args.out_dir_name) / f"summary_{args.exp_name}.md"
    if rows and not bool(args.dry_run):
        _write_summary_md(
            rows=rows,
            summary_path=summary_path,
            exp_name=str(args.exp_name),
            dtu_official_dir=dtu_official_dir,
            eval_script=eval_script,
            max_steps=int(args.max_steps),
        )
        print(f"[summary] wrote {summary_path}", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
