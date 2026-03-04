#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from friendly_splat.data.colmap_dataparser import ColmapDataParser


# NOTE:
#   Debug artifact outputs (`partitions.png`, per-block camera assignment plots)
#   follow the CityGaussian partition script behavior:
#   `utils/partition_citygs.py`.
def _load_coarse_cfg(coarse_dir: Path) -> dict[str, Any]:
    cfg_path = coarse_dir / "cfg.yml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing coarse config: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid coarse config format: {cfg_path}")
    return cfg


def _find_latest_coarse_ckpt(coarse_dir: Path) -> Path:
    ckpt_dir = coarse_dir / "ckpts"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint dir: {ckpt_dir}")
    ckpts = sorted(ckpt_dir.glob("ckpt_step*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in: {ckpt_dir}")

    def _step(path: Path) -> int:
        digits = "".join(ch for ch in path.stem if ch.isdigit())
        return int(digits) if digits else -1

    ckpts.sort(key=_step)
    return ckpts[-1]


def _load_coarse_points(coarse_dir: Path) -> tuple[np.ndarray, Path]:
    ckpt_path = _find_latest_coarse_ckpt(coarse_dir)
    try:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt_obj, dict):
        raise ValueError(f"Invalid checkpoint format: {ckpt_path}")
    splats = ckpt_obj.get("splats")
    if not isinstance(splats, dict):
        raise ValueError(f"Checkpoint missing splats: {ckpt_path}")
    means = splats.get("means")
    if isinstance(means, torch.nn.Parameter):
        means = means.detach()
    if not torch.is_tensor(means):
        raise ValueError(f"Checkpoint splats missing tensor means: {ckpt_path}")
    points = means.cpu().numpy().astype(np.float64, copy=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Invalid coarse points shape: {points.shape}")
    return points, ckpt_path


def _resolve_dirs(data_dir: str, coarse_dir: str, out_dir: str | None) -> tuple[Path, Path, Path]:
    data_dir_path = Path(data_dir)
    coarse_dir_path = Path(coarse_dir)
    if out_dir is None:
        out_dir_path = coarse_dir_path.parent / "partition"
    else:
        out_dir_path = Path(out_dir)
    return data_dir_path, coarse_dir_path, out_dir_path


def _quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0:
        raise ValueError(f"bins must be > 0, got {bins}")
    if values.size == 0:
        raise ValueError("Cannot build quantile edges from empty values.")

    q = np.linspace(0.0, 1.0, bins + 1, dtype=np.float64)
    edges = np.quantile(values.astype(np.float64), q)
    edges[0] = float(np.min(values))
    edges[-1] = float(np.max(values))
    if edges[-1] <= edges[0]:
        edges[-1] = edges[0] + 1e-6
    for idx in range(1, edges.shape[0]):
        if not (edges[idx] > edges[idx - 1]):
            edges[idx] = edges[idx - 1] + 1e-6
    return edges


def _build_block_bounds(
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    block_dim_x: int,
    block_dim_y: int,
    enlarge_frac: float,
) -> tuple[list[str], np.ndarray]:
    block_ids: list[str] = []
    bounds = np.zeros((block_dim_x * block_dim_y, 4), dtype=np.float64)
    ptr = 0
    for bx in range(block_dim_x):
        for by in range(block_dim_y):
            x0 = float(x_edges[bx])
            x1 = float(x_edges[bx + 1])
            y0 = float(y_edges[by])
            y1 = float(y_edges[by + 1])
            ex = max(0.0, enlarge_frac) * (x1 - x0)
            ey = max(0.0, enlarge_frac) * (y1 - y0)
            bounds[ptr] = np.array([x0 - ex, x1 + ex, y0 - ey, y1 + ey], dtype=np.float64)
            block_ids.append(f"block_{bx:02d}_{by:02d}")
            ptr += 1
    return block_ids, bounds


def _build_location_assignment(cam_centers_xy: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    x = cam_centers_xy[:, 0][None, :]
    y = cam_centers_xy[:, 1][None, :]
    xmin = bounds[:, 0][:, None]
    xmax = bounds[:, 1][:, None]
    ymin = bounds[:, 2][:, None]
    ymax = bounds[:, 3][:, None]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def _points_in_bounds_mask(points_xy: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    x = points_xy[:, 0][None, :]
    y = points_xy[:, 1][None, :]
    xmin = bounds[:, 0][:, None]
    xmax = bounds[:, 1][:, None]
    ymin = bounds[:, 2][:, None]
    ymax = bounds[:, 3][:, None]
    return (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)


def _load_train_split_from_colmap(
    *,
    data_dir: Path,
    factor: float,
    normalize_world_space: bool,
    align_world_axes: bool,
    test_every: int,
    benchmark_train_split: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    dataparser = ColmapDataParser(
        data_dir=str(data_dir),
        factor=factor,
        normalize_world_space=normalize_world_space,
        align_world_axes=align_world_axes,
        test_every=test_every,
        benchmark_train_split=benchmark_train_split,
        depth_dir_name=None,
        normal_dir_name=None,
        dynamic_mask_dir_name=None,
        sky_mask_dir_name=None,
        train_image_list_file=None,
    )
    parsed = dataparser.get_dataparser_outputs(split="train")
    indices = parsed.indices.astype(np.int64)
    c2w = parsed.camtoworlds[indices].astype(np.float64, copy=False)
    Ks = parsed.Ks[indices].astype(np.float64, copy=False)
    all_sizes = np.asarray(parsed.metadata["image_sizes"], dtype=np.int64)
    image_sizes = all_sizes[indices]
    image_names = [parsed.image_names[int(i)] for i in indices.tolist()]
    return c2w, Ks, image_sizes, image_names


def _project_visibility_assignment(
    *,
    points_xyz: np.ndarray,
    points_xy_partition: np.ndarray,
    c2w: np.ndarray,
    Ks: np.ndarray,
    image_sizes: np.ndarray,
    vis_bounds: np.ndarray,
    location_assign: np.ndarray,
    content_threshold: float,
    max_points: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_points = int(points_xyz.shape[0])
    if n_points == 0:
        return np.zeros_like(location_assign, dtype=bool), {"points_used": 0}

    if max_points > 0 and n_points > max_points:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(n_points, size=max_points, replace=False)
        points_xyz = points_xyz[sample_idx]
        points_xy_partition = points_xy_partition[sample_idx]
    points_used = int(points_xyz.shape[0])

    block_point_mask = _points_in_bounds_mask(points_xy_partition, vis_bounds)
    n_blocks = int(block_point_mask.shape[0])
    n_cams = int(c2w.shape[0])
    visibility_assign = np.zeros((n_blocks, n_cams), dtype=bool)
    print(
        "[visibility] start "
        f"cameras={n_cams} blocks={n_blocks} points_used={points_used}/{n_points} "
        f"threshold={content_threshold:.4f}",
        flush=True,
    )

    w2c = np.linalg.inv(c2w)
    points = points_xyz.astype(np.float64, copy=False)
    for cam_idx in range(n_cams):
        R = w2c[cam_idx, :3, :3]
        t = w2c[cam_idx, :3, 3]
        points_cam = points @ R.T + t[None, :]

        z = points_cam[:, 2]
        positive_depth = z > 1e-6
        if not np.any(positive_depth):
            continue

        K = Ks[cam_idx].astype(np.float64, copy=False)
        proj = points_cam @ K.T
        u = proj[:, 0] / np.maximum(z, 1e-12)
        v = proj[:, 1] / np.maximum(z, 1e-12)
        width = float(image_sizes[cam_idx, 0])
        height = float(image_sizes[cam_idx, 1])
        valid = positive_depth & (u >= 0.0) & (u < width) & (v >= 0.0) & (v < height)
        total_vis = int(np.count_nonzero(valid))
        if total_vis == 0:
            continue

        counts = block_point_mask[:, valid].sum(axis=1).astype(np.float64)
        ratios = counts / float(total_vis)
        visible_blocks = ratios >= content_threshold
        visibility_assign[:, cam_idx] = visible_blocks & (~location_assign[:, cam_idx])
        if cam_idx == 0 or (cam_idx + 1) % 100 == 0 or (cam_idx + 1) == n_cams:
            print(
                f"[visibility] camera {cam_idx + 1}/{n_cams} total_vis={total_vis}",
                flush=True,
            )

    total_visibility_edges = int(np.count_nonzero(visibility_assign))
    print(
        f"[visibility] done assigned_pairs={total_visibility_edges}",
        flush=True,
    )

    return visibility_assign, {
        "points_used": points_used,
        "points_total": n_points,
        "content_threshold": content_threshold,
    }


def _write_outputs(
    *,
    out_dir: Path,
    manifest: dict[str, Any],
    assigned: np.ndarray,
    image_names: list[str],
) -> None:
    (out_dir / "blocks").mkdir(parents=True, exist_ok=True)
    for block_idx, block in enumerate(manifest["blocks"]):
        txt_path = out_dir / str(block["train_image_list_file"])
        txt_path.parent.mkdir(parents=True, exist_ok=True)
        selected_indices = np.where(assigned[block_idx])[0].tolist()
        selected_names = [image_names[i] for i in selected_indices]
        selected_names.sort()
        with open(txt_path, "w", encoding="utf-8") as f:
            for name in selected_names:
                f.write(f"{name}\n")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[ok] wrote manifest: {manifest_path}")


def _set_plot_limits(ax, points_xy: np.ndarray, cam_centers_xy: np.ndarray, margin_ratio: float = 0.05) -> None:
    all_x = np.concatenate([points_xy[:, 0], cam_centers_xy[:, 0]], axis=0)
    all_y = np.concatenate([points_xy[:, 1], cam_centers_xy[:, 1]], axis=0)
    x_min = float(np.min(all_x))
    x_max = float(np.max(all_x))
    y_min = float(np.min(all_y))
    y_max = float(np.max(all_y))
    dx = max(1e-6, x_max - x_min)
    dy = max(1e-6, y_max - y_min)
    mx = margin_ratio * dx
    my = margin_ratio * dy
    ax.set_xlim(x_min - mx, x_max + mx)
    ax.set_ylim(y_min - my, y_max + my)
    ax.set_aspect("equal", adjustable="box")


def _save_partition_debug_plots(
    *,
    out_dir: Path,
    block_ids: list[str],
    bounds: np.ndarray,
    points_xy: np.ndarray,
    cam_centers_xy: np.ndarray,
    location_assign: np.ndarray,
    visibility_assign: np.ndarray,
    max_plot_points: int = 51_200,
) -> None:
    # CityGaussian-style debug plots:
    #   - `partitions.png` for global block layout
    #   - `blocks/<block_id>.png` for location/visibility camera assignment
    if points_xy.shape[0] > int(max_plot_points):
        sample_idx = np.linspace(0, points_xy.shape[0] - 1, int(max_plot_points), dtype=np.int64)
        points_plot = points_xy[sample_idx]
    else:
        points_plot = points_xy

    out_dir.mkdir(parents=True, exist_ok=True)
    blocks_dir = out_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points_plot[:, 0], points_plot[:, 1], s=0.03, c="0.75")
    ax.scatter(cam_centers_xy[:, 0], cam_centers_xy[:, 1], s=0.2, c="black")
    for idx, block_id in enumerate(block_ids):
        xmin, xmax, ymin, ymax = bounds[idx]
        ax.add_patch(
            mpatches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="green",
                linewidth=0.8,
            )
        )
        ax.text(xmin, ymax, block_id, fontsize=4, color="green")
    _set_plot_limits(ax, points_plot, cam_centers_xy)
    overall_path = out_dir / "partitions.png"
    fig.savefig(overall_path, dpi=300)
    plt.close(fig)
    print(f"[debug] wrote overall partitions: {overall_path}", flush=True)

    for idx, block_id in enumerate(block_ids):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(points_plot[:, 0], points_plot[:, 1], s=0.03, c="0.82")
        ax.scatter(cam_centers_xy[:, 0], cam_centers_xy[:, 1], s=0.1, c="0.6")

        loc_mask = location_assign[idx]
        vis_mask = visibility_assign[idx]
        if np.any(loc_mask):
            ax.scatter(cam_centers_xy[loc_mask, 0], cam_centers_xy[loc_mask, 1], s=0.8, c="blue")
        if np.any(vis_mask):
            ax.scatter(cam_centers_xy[vis_mask, 0], cam_centers_xy[vis_mask, 1], s=0.8, c="red")

        xmin, xmax, ymin, ymax = bounds[idx]
        ax.add_patch(
            mpatches.Rectangle(
                (xmin, ymin),
                xmax - xmin,
                ymax - ymin,
                fill=False,
                edgecolor="green",
                linewidth=1.0,
            )
        )
        ax.set_title(block_id, fontsize=8)
        _set_plot_limits(ax, points_plot, cam_centers_xy)

        save_path = blocks_dir / f"{block_id}.png"
        fig.savefig(save_path, dpi=300)
        plt.close(fig)

    print(f"[debug] wrote per-block assignment plots: {blocks_dir}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Partition image lists from a coarse model.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--coarse-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--block-dim", type=int, nargs=2, default=[3, 3], metavar=("X", "Y"))
    parser.add_argument("--content-threshold", type=float, default=0.05)
    parser.add_argument("--location-enlarge-frac", type=float, default=0.0)
    parser.add_argument("--visibility-enlarge-frac", type=float, default=0.0)
    parser.add_argument("--no-visibility-assignment", action="store_true")
    parser.add_argument("--visibility-max-points", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    block_dim_x = int(args.block_dim[0])
    block_dim_y = int(args.block_dim[1])
    print("[1/5] Resolve input/output directories", flush=True)
    data_dir, coarse_dir, out_dir = _resolve_dirs(
        data_dir=args.data_dir,
        coarse_dir=args.coarse_dir,
        out_dir=args.out_dir,
    )
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data dir: {data_dir}")
    if not coarse_dir.exists():
        raise FileNotFoundError(f"Missing coarse dir: {coarse_dir}")

    print("[2/5] Load coarse config and COLMAP train split", flush=True)
    cfg = _load_coarse_cfg(coarse_dir)
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    factor = float(data_cfg.get("data_factor", 1.0))
    normalize_world_space = bool(data_cfg.get("normalize_world_space", True))
    align_world_axes = bool(data_cfg.get("align_world_axes", False))
    test_every = int(data_cfg.get("test_every", 8))
    benchmark_train_split = bool(data_cfg.get("benchmark_train_split", False))

    c2w, Ks, image_sizes, image_names = _load_train_split_from_colmap(
        data_dir=data_dir,
        factor=factor,
        normalize_world_space=normalize_world_space,
        align_world_axes=align_world_axes,
        test_every=test_every,
        benchmark_train_split=benchmark_train_split,
    )
    print("[3/5] Load coarse checkpoint points", flush=True)
    points_xyz, ckpt_path = _load_coarse_points(coarse_dir)
    cam_centers_xy = c2w[:, :2, 3].astype(np.float64, copy=False)
    points_xy = points_xyz[:, :2].astype(np.float64, copy=False)

    x_edges = _quantile_edges(points_xy[:, 0], block_dim_x)
    y_edges = _quantile_edges(points_xy[:, 1], block_dim_y)
    block_ids, loc_bounds = _build_block_bounds(
        x_edges=x_edges,
        y_edges=y_edges,
        block_dim_x=block_dim_x,
        block_dim_y=block_dim_y,
        enlarge_frac=float(args.location_enlarge_frac),
    )
    _, vis_bounds = _build_block_bounds(
        x_edges=x_edges,
        y_edges=y_edges,
        block_dim_x=block_dim_x,
        block_dim_y=block_dim_y,
        enlarge_frac=float(args.visibility_enlarge_frac),
    )

    print("[4/5] Build location/visibility assignments", flush=True)
    location_assign = _build_location_assignment(cam_centers_xy, loc_bounds)
    content_threshold = float(args.content_threshold)
    if bool(args.no_visibility_assignment):
        visibility_assign = np.zeros_like(location_assign, dtype=bool)
        vis_stats = {
            "points_used": 0,
            "points_total": int(points_xyz.shape[0]),
            "content_threshold": content_threshold,
        }
    else:
        visibility_assign, vis_stats = _project_visibility_assignment(
            points_xyz=points_xyz,
            points_xy_partition=points_xy,
            c2w=c2w,
            Ks=Ks,
            image_sizes=image_sizes,
            vis_bounds=vis_bounds,
            location_assign=location_assign,
            content_threshold=content_threshold,
            max_points=int(args.visibility_max_points),
            seed=int(args.seed),
        )

    assigned = location_assign | visibility_assign
    scene_name = data_dir.name
    manifest: dict[str, Any] = {
        "scene": scene_name,
        "data_dir": str(data_dir),
        "coarse_dir": str(coarse_dir),
        "coarse_ckpt": str(ckpt_path),
        "out_dir": str(out_dir),
        "data_factor": factor,
        "normalize_world_space": normalize_world_space,
        "align_world_axes": align_world_axes,
        "test_every": test_every,
        "benchmark_train_split": benchmark_train_split,
        "partition_mode": "quantile_xy",
        "block_dim": [block_dim_x, block_dim_y],
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "location_enlarge_frac": float(args.location_enlarge_frac),
        "visibility_enlarge_frac": float(args.visibility_enlarge_frac),
        "content_threshold": content_threshold,
        "num_train_images": int(len(image_names)),
        "num_coarse_points": int(points_xyz.shape[0]),
        "visibility_stats": vis_stats,
        "blocks": [],
    }

    print(f"[scene] {scene_name}")
    print(f"[data_dir] {data_dir}")
    print(f"[coarse_dir] {coarse_dir}")
    print(f"[coarse_ckpt] {ckpt_path}")
    print(f"[out_dir] {out_dir}")
    print(
        f"[partition] dim={block_dim_x}x{block_dim_y} "
        f"images={len(image_names)} points={points_xyz.shape[0]} "
        f"visibility={'off' if args.no_visibility_assignment else 'on'}"
    )

    for block_idx, block_id in enumerate(block_ids):
        loc_count = int(np.count_nonzero(location_assign[block_idx]))
        vis_count = int(np.count_nonzero(visibility_assign[block_idx]))
        total_count = int(np.count_nonzero(assigned[block_idx]))
        manifest["blocks"].append(
            {
                "block_id": block_id,
                "train_image_list_file": f"blocks/{block_id}_train_images.txt",
                "num_images": total_count,
                "num_location_images": loc_count,
                "num_visibility_images": vis_count,
            }
        )
        print(f"  - {block_id}: images={total_count} (location={loc_count}, visibility={vis_count})")

    print("[5/6] Save debug visualizations", flush=True)
    _save_partition_debug_plots(
        out_dir=out_dir,
        block_ids=block_ids,
        bounds=loc_bounds,
        points_xy=points_xy,
        cam_centers_xy=cam_centers_xy,
        location_assign=location_assign,
        visibility_assign=visibility_assign,
    )

    print("[6/6] Write partition files", flush=True)
    _write_outputs(
        out_dir=out_dir,
        manifest=manifest,
        assigned=assigned,
        image_names=image_names,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
