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
from typing import Optional, Tuple


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
    return result_dir / "mesh" / "tsdf_mesh_post.ply"

def _load_K_Rt_from_P(*, P) -> Tuple["object", "object"]:
    """Decompose a 3x4 projection matrix to intrinsics (4x4) and pose (4x4).

    Matches the convention used by common DTU eval scripts (cv2.decomposeProjectionMatrix).
    Returns:
      intrinsics: 4x4 with K in the top-left.
      pose: 4x4 camera-to-world transform.
    """
    import numpy as np

    try:
        import cv2
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependency 'opencv-python' (cv2). Required for DTU mesh culling."
        ) from e

    P = np.asarray(P, dtype=np.float32)
    if P.shape != (3, 4):
        raise ValueError(f"Expected P shape (3,4), got {tuple(P.shape)}")

    K, R, t, *_rest = cv2.decomposeProjectionMatrix(P)
    K = K / max(float(K[2, 2]), 1e-8)

    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K.astype(np.float32)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose().astype(np.float32)
    tt = (t[:3] / t[3])[:, 0].astype(np.float32)
    pose[:3, 3] = tt
    return intrinsics, pose


def _list_images(*, image_dir: Path) -> list[Path]:
    image_paths: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        image_paths.extend(sorted(image_dir.glob(ext)))
    return sorted(image_paths)


def _build_mask_map(*, mask_dir: Path) -> dict[int, Path]:
    """Map integer frame id -> mask path.

    Filters out common junk like macOS resource forks (e.g. ._000.png) and other hidden files.
    """
    mask_by_id: dict[int, Path] = {}
    for p in sorted(mask_dir.glob("*.png")):
        if p.name.startswith("."):
            continue
        stem = p.stem
        if not stem.isdigit():
            continue
        mask_by_id[int(stem)] = p
    return mask_by_id


def _cull_dtu_mesh(
    *,
    scan_id: int,
    mesh_path: Path,
    result_mesh_path: Path,
    mask_root: Path,
    mask_dilate: int,
    prefer_cuda: bool,
) -> None:
    """Cull mesh using DTU preprocessed masks and camera parameters.

    Expected layout:
      <mask_root>/scanXX/
        images/
        mask/
        cameras.npz
    """
    try:
        import cv2
        import numpy as np
        import torch
        import torch.nn.functional as F
        import trimesh
        from skimage.morphology import binary_dilation, disk
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing dependencies for DTU mesh culling. "
            "Need cv2, numpy, torch, trimesh, scikit-image."
        ) from e

    instance_dir = mask_root / f"scan{int(scan_id):d}"
    if not instance_dir.exists():
        raise FileNotFoundError(f"Missing scan dir: {instance_dir}")

    image_dir = instance_dir / "images"
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {image_dir}")
    image_paths = _list_images(image_dir=image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {image_dir}")
    n_images = int(len(image_paths))

    cameras_npz = instance_dir / "cameras.npz"
    if not cameras_npz.exists():
        raise FileNotFoundError(f"Missing cameras.npz: {cameras_npz}")

    cam_dict = np.load(str(cameras_npz))
    scale_mats = [cam_dict[f"scale_mat_{idx}"].astype(np.float32) for idx in range(n_images)]
    world_mats = [cam_dict[f"world_mat_{idx}"].astype(np.float32) for idx in range(n_images)]
    intrinsics_all: list["torch.Tensor"] = []
    pose_all: list["torch.Tensor"] = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = (world_mat @ scale_mat)[:3, :4].astype(np.float32)
        intrinsics, pose = _load_K_Rt_from_P(P=P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())

    mask_dir = instance_dir / "mask"
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing mask dir: {mask_dir}")
    mask_by_id = _build_mask_map(mask_dir=mask_dir)

    image_ids: list[int] = []
    for p in image_paths:
        if not p.stem.isdigit():
            raise ValueError(f"Image filename is not an integer id: {p.name}")
        image_ids.append(int(p.stem))

    mask_paths: list[Path] = []
    missing: list[int] = []
    for img_id in image_ids:
        mp = mask_by_id.get(int(img_id))
        if mp is None:
            missing.append(int(img_id))
            continue
        mask_paths.append(mp)
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} masks in {mask_dir} for image ids: {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
        )

    masks: list[np.ndarray] = []
    for p in mask_paths:
        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise FileNotFoundError(f"Failed to read mask: {p}")
        masks.append(m)

    H, W = int(masks[0].shape[0]), int(masks[0].shape[1])

    mesh = trimesh.load(str(mesh_path))
    vertices = mesh.vertices
    if vertices is None or len(vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    use_cuda = bool(prefer_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    v = torch.from_numpy(vertices.astype(np.float32)).to(device=device)
    v = torch.cat((v, torch.ones_like(v[:, :1])), dim=-1)  # [N,4]
    v = v.permute(1, 0).contiguous()  # [4,N]

    dilate_radius = int(mask_dilate)
    if dilate_radius < 0:
        raise ValueError(f"mask_dilate must be >= 0, got {dilate_radius}")
    selem = disk(dilate_radius) if dilate_radius > 0 else None

    # Soft consistency across views: keep a vertex if at least `vote_threshold` fraction of
    # *in-view* cameras classify it as foreground.
    vote_threshold = 0.9
    vote_count = torch.zeros((v.shape[1],), device=device, dtype=torch.int32)
    view_count = torch.zeros((v.shape[1],), device=device, dtype=torch.int32)
    for i in range(n_images):
        pose = pose_all[i].to(device=device)
        w2c = torch.inverse(pose)
        intrinsic = intrinsics_all[i].to(device=device)

        cam_points = intrinsic @ w2c @ v
        pix = cam_points[:2, :] / (cam_points[2:3, :] + 1e-6)
        pix = pix.permute(1, 0)
        pix[..., 0] /= float(W - 1)
        pix[..., 1] /= float(H - 1)
        pix = (pix - 0.5) * 2.0
        in_bounds = ((pix > -1.0) & (pix < 1.0)).all(dim=-1)
        in_front = cam_points[2, :] > 1e-6
        in_view = in_bounds & in_front

        maski = masks[i]
        if maski.ndim == 3:
            maski = maski[:, :, 0]
        # Interpret mask intensity as alpha in [0,1] and keep pixels with alpha>threshold.
        if np.issubdtype(maski.dtype, np.integer):
            denom = float(np.iinfo(maski.dtype).max)
            alpha01 = (maski.astype(np.float32) / max(denom, 1.0)).clip(0.0, 1.0)
        else:
            mf = maski.astype(np.float32, copy=False)
            maxv = float(np.nanmax(mf)) if mf.size > 0 else 0.0
            if maxv <= 1.0:
                alpha01 = mf.clip(0.0, 1.0)
            elif maxv <= 255.0:
                alpha01 = (mf / 255.0).clip(0.0, 1.0)
            else:
                alpha01 = (mf / maxv).clip(0.0, 1.0)

        mask_bin = alpha01 > 0.5
        if selem is not None:
            mask_bin = binary_dilation(mask_bin, selem)
        mask_t = torch.from_numpy(mask_bin.astype(np.float32))[None, None].to(device=device)

        sampled = F.grid_sample(
            mask_t,
            pix[None, None],
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )[0, 0, 0]
        fg = sampled > 0.5
        if in_view.any():
            vote_count += (fg & in_view).to(torch.int32)
            view_count += in_view.to(torch.int32)

    # Only consider views where the vertex projects in-bounds and is in front of the camera.
    # Vertices seen by zero cameras are culled (keep=False).
    ratio = vote_count.to(torch.float32) / view_count.clamp(min=1).to(torch.float32)
    keep = ((view_count > 0) & (ratio >= float(vote_threshold))).detach().cpu().numpy()
    face_keep = keep[mesh.faces].all(axis=1)
    mesh.update_vertices(keep)
    mesh.update_faces(face_keep)

    scale_mat0 = cam_dict["scale_mat_0"].astype(np.float32)
    mesh.vertices = mesh.vertices * float(scale_mat0[0, 0]) + scale_mat0[:3, 3][None]

    result_mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(result_mesh_path))


def _run_tsdf_mesh(
    *,
    repo_root: Path,
    ply_path: Path,
    scene_dir: Path,
    device: str,
    interval: int,
    voxel_length: float,
    sdf_trunc: float,
    depth_trunc: float,
    post_process_clusters: int,
    tsdf_mask_dir_name: Optional[str],
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
                "--mask_dilate",
                str(int(tsdf_mask_dilate)),
            ]
        )

    print(f"[mesh] {_format_cmd(cmd)}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=str(repo_root))


def _run_dtu_eval_py(
    *,
    eval_py: Path,
    dtu_official_dir: Path,
    scan: str,
    mesh_path: Path,
    out_dir: Path,
    dry_run: bool,
) -> None:
    scan_id = int(str(scan).replace("scan", ""))
    cmd = [
        sys.executable,
        str(eval_py),
        "--data",
        str(mesh_path),
        "--scan",
        str(int(scan_id)),
        "--mode",
        "mesh",
        "--dataset_dir",
        str(dtu_official_dir),
        "--vis_out_dir",
        str(out_dir),
    ]
    print(f"[eval.py] {_format_cmd(cmd)}", flush=True)
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


def _write_summary_md(
    *,
    rows: list[_EvalRow],
    summary_path: Path,
    exp_name: str,
    dtu_official_dir: Path,
    eval_py: Path,
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
    lines.append(f"- Eval py: `{eval_py}`")
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

    parser.add_argument(
        "--dtu-official-dir",
        type=str,
        required=False,
        default=None,
        help=(
            "Path to the official DTU evaluation data (ObsMask/, Points/stl/, ...). "
            "If omitted, the script will try to auto-detect it from common locations "
            "under --data-root."
        ),
    )
    parser.add_argument(
        "--cull-mask-dilate",
        type=int,
        default=24,
        help="Mask dilation radius (pixels) for internal DTU mesh culling.",
    )
    parser.add_argument(
        "--cull-prefer-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer CUDA for internal DTU mesh culling when available.",
    )

    # Mesh extraction params.
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--interval", type=int, default=1)
    # Defaults aligned with PGSR DTU TSDF fusion:
    # - voxel_size=0.002, sdf_trunc=4*voxel_size, max_depth=5.0
    parser.add_argument("--voxel-length", type=float, default=0.002)
    parser.add_argument("--sdf-trunc", type=float, default=0.008)
    parser.add_argument("--depth-trunc", type=float, default=5.0)
    parser.add_argument("--post-process-clusters", type=int, default=2)
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

    if not scans:
        raise ValueError("No scans selected.")

    dtu_official_dir: Optional[Path] = None
    if args.dtu_official_dir is None:
        candidates = [
            dtu_dir / "eval_dtu",
            data_root / "DTU_Official",
            data_root / "Official_DTU_Dataset",
            data_root / "Offical_DTU_Dataset",  # common misspelling in some repos
        ]
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

    assert dtu_official_dir is not None

    repo_root = Path(__file__).resolve().parents[2]
    # Always use the vendored DTU eval.py under this repo.
    eval_py = repo_root / "benchmarks" / "geo_quality" / "dtu_eval" / "eval.py"
    if not eval_py.exists():
        raise FileNotFoundError(f"DTU eval.py not found: {eval_py}")
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
                device=str(args.device),
                interval=int(args.interval),
                voxel_length=float(args.voxel_length),
                sdf_trunc=float(args.sdf_trunc),
                depth_trunc=float(args.depth_trunc),
                post_process_clusters=int(args.post_process_clusters),
                tsdf_mask_dir_name=tsdf_mask_dir_name,
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

        scan_id = int(str(scan).replace("scan", ""))
        culled_mesh = eval_out_dir / "culled_mesh.ply"
        if not bool(args.dry_run):
            eval_out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[cull] {scan} -> {culled_mesh}", flush=True)
        if not bool(args.dry_run):
            _cull_dtu_mesh(
                scan_id=int(scan_id),
                mesh_path=mesh,
                result_mesh_path=culled_mesh,
                mask_root=dtu_dir,
                mask_dilate=int(args.cull_mask_dilate),
                prefer_cuda=bool(args.cull_prefer_cuda),
            )
        _run_dtu_eval_py(
            eval_py=eval_py,
            dtu_official_dir=dtu_official_dir,
            scan=scan,
            mesh_path=culled_mesh,
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
            eval_py=eval_py,
            max_steps=int(args.max_steps),
        )
        print(f"[summary] wrote {summary_path}", flush=True)

    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
