#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import sys
from pathlib import Path
from typing import Optional

import copy

from config import scenes_tau_dict


@dataclass(frozen=True)
class CameraPose:
    meta: Optional[tuple[int, int, int]]
    mat: "object"  # np.ndarray [4,4]


@dataclass(frozen=True)
class _EvalStats:
    precision: float
    recall: float
    fscore: float


def _registration_module(o3d):
    reg = getattr(getattr(o3d, "pipelines", None), "registration", None)
    if reg is not None:
        return reg
    reg = getattr(o3d, "registration", None)
    if reg is None:
        raise RuntimeError("Open3D registration module not found (unsupported Open3D version).")
    return reg


def read_trajectory(path: str) -> list[CameraPose]:
    p = Path(str(path)).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {p}")

    lines = [ln.strip() for ln in p.read_text(encoding="utf-8", errors="strict").splitlines()]
    lines = [ln for ln in lines if ln]
    out: list[CameraPose] = []
    i = 0
    while i < len(lines):
        header = lines[i].split()
        if len(header) < 3:
            raise ValueError(f"Invalid trajectory header line: {lines[i]!r}")
        meta = (int(header[0]), int(header[1]), int(header[2]))
        if i + 4 >= len(lines):
            raise ValueError("Unexpected EOF while reading trajectory matrix.")

        import numpy as np  # noqa: WPS433

        mat = np.zeros((4, 4), dtype=np.float64)
        for r in range(4):
            parts = lines[i + 1 + r].split()
            if len(parts) != 4:
                raise ValueError(f"Invalid trajectory matrix row: {lines[i + 1 + r]!r}")
            mat[r, :] = np.asarray([float(x) for x in parts], dtype=np.float64)
        out.append(CameraPose(meta=meta, mat=mat))
        i += 5
    return out


def _trajectory_to_pcd(*, traj: list[CameraPose], o3d):
    import numpy as np  # noqa: WPS433

    pts = np.stack([p.mat[:3, 3] for p in traj], axis=0).astype(np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _trajectory_alignment(
    *,
    traj_to_register: list[CameraPose],
    gt_traj_col: list[CameraPose],
    gt_trans,
    o3d,
) -> "object":
    import numpy as np  # noqa: WPS433

    reg = _registration_module(o3d)
    traj_pcd_col = _trajectory_to_pcd(traj=gt_traj_col, o3d=o3d)
    if gt_trans is not None:
        traj_pcd_col.transform(gt_trans)
    traj_to_register_pcd = _trajectory_to_pcd(traj=traj_to_register, o3d=o3d)

    corres = o3d.utility.Vector2iVector(
        np.asarray([[i, i] for i in range(len(gt_traj_col))], dtype=np.int32)
    )
    criteria = reg.RANSACConvergenceCriteria()
    # Open3D API differs across versions/builds:
    # - Older versions: max_iteration + max_validation
    # - Newer versions (e.g. 0.18): max_iteration + confidence
    if hasattr(criteria, "max_iteration"):
        criteria.max_iteration = 100000
    if hasattr(criteria, "max_validation"):
        criteria.max_validation = 100000
    elif hasattr(criteria, "confidence"):
        criteria.confidence = 0.999
    est = reg.TransformationEstimationPointToPoint(True)

    # Signature differs across Open3D versions; try keywords first.
    try:
        ransac = reg.registration_ransac_based_on_correspondence(
            source=traj_to_register_pcd,
            target=traj_pcd_col,
            corres=corres,
            max_correspondence_distance=0.2,
            estimation_method=est,
            ransac_n=6,
            checkers=[],
            criteria=criteria,
        )
    except TypeError:
        ransac = reg.registration_ransac_based_on_correspondence(
            traj_to_register_pcd,
            traj_pcd_col,
            corres,
            0.2,
            est,
            6,
            criteria,
        )
    return ransac.transformation


def _crop_and_downsample(
    *,
    pcd,
    crop_volume,
    down_sample_method: str,
    voxel_size: float,
    trans,
):
    pcd_copy = copy.deepcopy(pcd)
    pcd_copy.transform(trans)  # Open3D transforms in-place
    pcd_crop = crop_volume.crop_point_cloud(pcd_copy) if crop_volume is not None else pcd_copy

    if down_sample_method == "voxel":
        return pcd_crop.voxel_down_sample(float(voxel_size))
    if down_sample_method == "uniform":
        n_points = len(pcd_crop.points)
        if n_points > int(4e6):
            ds_rate = int(round(n_points / float(4e6)))
            return pcd_crop.uniform_down_sample(ds_rate)
        return pcd_crop
    raise ValueError(f"Unknown down_sample_method: {down_sample_method!r}")


def _registration_icp(*, source, target, threshold: float, init, max_itr: int, o3d):
    reg = _registration_module(o3d)
    est = reg.TransformationEstimationPointToPoint(True)
    try:
        criteria = reg.ICPConvergenceCriteria(
            relative_fitness=1e-6,
            relative_rmse=1e-6,
            max_iteration=int(max_itr),
        )
        return reg.registration_icp(
            source=source,
            target=target,
            max_correspondence_distance=float(threshold),
            init=init,
            estimation_method=est,
            criteria=criteria,
        )
    except TypeError:
        return reg.registration_icp(
            source,
            target,
            float(threshold),
            init,
            est,
            reg.ICPConvergenceCriteria(1e-6, int(max_itr)),
        )


def _registration_vol_ds(
    *,
    source,
    target,
    init_trans,
    crop_volume,
    voxel_size: float,
    threshold: float,
    max_itr: int,
    o3d,
):
    import numpy as np  # noqa: WPS433

    s = _crop_and_downsample(
        pcd=source,
        crop_volume=crop_volume,
        down_sample_method="voxel",
        voxel_size=float(voxel_size),
        trans=init_trans,
    )
    t = _crop_and_downsample(
        pcd=target,
        crop_volume=crop_volume,
        down_sample_method="voxel",
        voxel_size=float(voxel_size),
        trans=np.identity(4),
    )
    icp = _registration_icp(
        source=s,
        target=t,
        threshold=float(threshold),
        init=np.identity(4),
        max_itr=int(max_itr),
        o3d=o3d,
    )
    return np.matmul(icp.transformation, init_trans)


def _registration_unif(
    *,
    source,
    target,
    init_trans,
    crop_volume,
    threshold: float,
    max_itr: int,
    o3d,
):
    import numpy as np  # noqa: WPS433

    s = _crop_and_downsample(
        pcd=source,
        crop_volume=crop_volume,
        down_sample_method="uniform",
        voxel_size=0.0,
        trans=init_trans,
    )
    t = _crop_and_downsample(
        pcd=target,
        crop_volume=crop_volume,
        down_sample_method="uniform",
        voxel_size=0.0,
        trans=np.identity(4),
    )
    icp = _registration_icp(
        source=s,
        target=t,
        threshold=float(threshold),
        init=np.identity(4),
        max_itr=int(max_itr),
        o3d=o3d,
    )
    return np.matmul(icp.transformation, init_trans)


def _get_f1(*, threshold: float, dist_est_to_gt, dist_gt_to_est) -> _EvalStats:
    d1 = list(dist_est_to_gt)
    d2 = list(dist_gt_to_est)
    if len(d1) == 0 or len(d2) == 0:
        return _EvalStats(precision=0.0, recall=0.0, fscore=0.0)
    precision = float(sum(float(d) < float(threshold) for d in d1)) / float(len(d1))
    recall = float(sum(float(d) < float(threshold) for d in d2)) / float(len(d2))
    denom = precision + recall
    fscore = (2.0 * precision * recall / denom) if denom > 0 else 0.0
    return _EvalStats(precision=precision, recall=recall, fscore=fscore)


def _evaluate_fscore(
    *,
    est_pcd,
    gt_pcd,
    trans,
    crop_volume,
    voxel_size: float,
    threshold: float,
    save_eval_vis: bool,
    out_dir: "Path",
    o3d,
):
    s = copy.deepcopy(est_pcd)
    s.transform(trans)
    if crop_volume is not None:
        s = crop_volume.crop_point_cloud(s)
    s = s.voxel_down_sample(float(voxel_size))

    t = copy.deepcopy(gt_pcd)
    if crop_volume is not None:
        t = crop_volume.crop_point_cloud(t)
    t = t.voxel_down_sample(float(voxel_size))

    if len(s.points) == 0 or len(t.points) == 0:
        # Avoid Open3D errors on empty point clouds.
        if bool(save_eval_vis):
            out_dir.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(str(out_dir / "precision.ply"), s)
            o3d.io.write_point_cloud(str(out_dir / "recall.ply"), t)
        return _EvalStats(precision=0.0, recall=0.0, fscore=0.0)

    dist1 = s.compute_point_cloud_distance(t)
    dist2 = t.compute_point_cloud_distance(s)

    if bool(save_eval_vis):
        import numpy as np  # noqa: WPS433

        out_dir.mkdir(parents=True, exist_ok=True)

        d1 = np.asarray(list(dist1), dtype=np.float64).reshape(-1)
        d2 = np.asarray(list(dist2), dtype=np.float64).reshape(-1)

        # Precision visualization: est points close to GT are green.
        s_vis = copy.deepcopy(s)
        c1 = np.ones((len(s_vis.points), 3), dtype=np.float64)
        if d1.size == len(s_vis.points):
            c1[d1 < float(threshold)] = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        s_vis.colors = o3d.utility.Vector3dVector(c1)
        o3d.io.write_point_cloud(str(out_dir / "precision.ply"), s_vis)

        # Recall visualization: GT points close to est are green.
        t_vis = copy.deepcopy(t)
        c2 = np.ones((len(t_vis.points), 3), dtype=np.float64)
        if d2.size == len(t_vis.points):
            c2[d2 < float(threshold)] = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        t_vis.colors = o3d.utility.Vector3dVector(c2)
        o3d.io.write_point_cloud(str(out_dir / "recall.ply"), t_vis)

    return _get_f1(threshold=float(threshold), dist_est_to_gt=dist1, dist_gt_to_est=dist2)


def _mesh_to_pcd(*, mesh_path: Path, o3d):
    import numpy as np  # noqa: WPS433

    # Align with 2DGS's vendored Tanks&Temples evaluation:
    # evaluate mesh vertices + per-triangle centroid points (denser point set).
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.remove_unreferenced_vertices()
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    tris = np.asarray(mesh.triangles, dtype=np.int64)
    if tris.size > 0:
        centroids = verts[tris].mean(axis=1)
        points = np.concatenate([verts, centroids], axis=0)
    else:
        points = verts

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def run_evaluation(
    *,
    dataset_dir: Path,
    traj_path: Path,
    ply_path: Path,
    out_dir: Path,
    tau_override: Optional[float],
    save_eval_vis: bool,
) -> dict:
    import numpy as np  # noqa: WPS433
    import open3d as o3d  # noqa: WPS433

    scene = str(dataset_dir.name)
    if tau_override is None:
        if scene not in scenes_tau_dict:
            raise ValueError(
                f"Unknown scene {scene!r} (not in scenes_tau_dict). Pass --tau to override."
            )
        tau = float(scenes_tau_dict[scene])
    else:
        tau = float(tau_override)

    alignment = dataset_dir / f"{scene}_trans.txt"
    gt_ply = dataset_dir / f"{scene}.ply"
    cropfile = dataset_dir / f"{scene}.json"
    colmap_ref_logfile = dataset_dir / f"{scene}_COLMAP_SfM.log"

    if not alignment.is_file():
        raise FileNotFoundError(f"Missing alignment: {alignment}")
    if not gt_ply.is_file():
        raise FileNotFoundError(f"Missing GT ply: {gt_ply}")
    if not cropfile.is_file():
        raise FileNotFoundError(f"Missing cropfile: {cropfile}")
    if not colmap_ref_logfile.is_file():
        raise FileNotFoundError(f"Missing reference traj: {colmap_ref_logfile}")

    out_dir.mkdir(parents=True, exist_ok=True)

    est_pcd = _mesh_to_pcd(mesh_path=ply_path, o3d=o3d)
    gt_pcd = o3d.io.read_point_cloud(str(gt_ply))
    gt_trans = np.loadtxt(str(alignment)).astype(np.float64)

    traj_to_register = read_trajectory(str(traj_path))
    gt_traj_col = read_trajectory(str(colmap_ref_logfile))
    T0 = _trajectory_alignment(
        traj_to_register=traj_to_register,
        gt_traj_col=gt_traj_col,
        gt_trans=gt_trans,
        o3d=o3d,
    )

    vol = o3d.visualization.read_selection_polygon_volume(str(cropfile))
    T1 = _registration_vol_ds(
        source=est_pcd,
        target=gt_pcd,
        init_trans=T0,
        crop_volume=vol,
        voxel_size=tau,
        threshold=tau * 80.0,
        max_itr=20,
        o3d=o3d,
    )
    T2 = _registration_vol_ds(
        source=est_pcd,
        target=gt_pcd,
        init_trans=T1,
        crop_volume=vol,
        voxel_size=tau / 2.0,
        threshold=tau * 20.0,
        max_itr=20,
        o3d=o3d,
    )
    T = _registration_unif(
        source=est_pcd,
        target=gt_pcd,
        init_trans=T2,
        crop_volume=vol,
        threshold=2.0 * tau,
        max_itr=20,
        o3d=o3d,
    )

    stats = _evaluate_fscore(
        est_pcd=est_pcd,
        gt_pcd=gt_pcd,
        trans=T,
        crop_volume=vol,
        voxel_size=tau / 2.0,
        threshold=tau,
        save_eval_vis=bool(save_eval_vis),
        out_dir=out_dir,
        o3d=o3d,
    )

    results = {
        "scene": scene,
        "tau": float(tau),
        "precision": float(stats.precision),
        "recall": float(stats.recall),
        "fscore": float(stats.fscore),
        "mesh_path": str(ply_path),
        "dataset_dir": str(dataset_dir),
        "traj_path": str(traj_path),
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="tnt_eval/run.py")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--traj-path", type=str, required=True)
    parser.add_argument("--ply-path", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--tau", type=float, default=None, help="Override scene tau threshold.")
    parser.add_argument(
        "--save-eval-vis",
        action="store_true",
        help="Save precision/recall colored point clouds (precision.ply / recall.ply) under --out-dir.",
    )
    args = parser.parse_args(argv)

    dataset_dir = Path(str(args.dataset_dir)).expanduser().resolve()
    traj_path = Path(str(args.traj_path)).expanduser().resolve()
    ply_path = Path(str(args.ply_path)).expanduser().resolve()
    out_dir = (
        Path(str(args.out_dir)).expanduser().resolve()
        if args.out_dir is not None
        else (ply_path.parent / "evaluation")
    )

    results = run_evaluation(
        dataset_dir=dataset_dir,
        traj_path=traj_path,
        ply_path=ply_path,
        out_dir=out_dir,
        tau_override=args.tau,
        save_eval_vis=bool(args.save_eval_vis),
    )
    print(
        f"[tnt] scene={results['scene']} tau={results['tau']:.6g} "
        f"precision={results['precision']:.4f} recall={results['recall']:.4f} fscore={results['fscore']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
