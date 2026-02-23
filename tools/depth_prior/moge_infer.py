from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Optional

import cv2
import numpy as np
import torch
import tqdm

# Allow running as a standalone script from the repo root without installation.
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from friendly_splat.data.colmap_dataparser import ColmapDataParser
from friendly_splat.data.colmap_io import get_extrinsic, read_model


def _to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr


def _alpha_to_exists_mask(alpha: np.ndarray) -> np.ndarray:
    """Convert an alpha channel into a boolean 'exists' mask using alpha>=0.5.

    This matches our invalid-mask convention: pixels with opacity < 0.5 are treated
    as invalid/background.
    """
    a = np.asarray(alpha)
    if np.issubdtype(a.dtype, np.integer):
        denom = float(np.iinfo(a.dtype).max)
        if denom <= 0:
            return np.zeros(a.shape[:2], dtype=bool)
        return (a.astype(np.float32) / denom) >= 0.5

    af = a.astype(np.float32, copy=False)
    maxv = float(np.nanmax(af)) if af.size > 0 else 0.0
    if maxv <= 1.0:
        return af >= 0.5
    if maxv <= 255.0:
        return (af / 255.0) >= 0.5
    return (af / maxv) >= 0.5


def _depth_edge_mask(
    depth: np.ndarray,
    *,
    rtol: float,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Return a boolean mask of depth edge pixels.

    Heuristic replacement for `utils3d.numpy.depth_edge`:
    mark pixels whose depth differs from a neighbor by a relative threshold.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be HxW, got {depth.shape}")

    valid = np.isfinite(depth) & (depth > 0)
    if mask is not None:
        valid = valid & mask.astype(bool)

    h, w = depth.shape
    edge = np.zeros((h, w), dtype=bool)
    eps = 1e-6

    def _mark(a: np.ndarray, b: np.ndarray, a_valid: np.ndarray, b_valid: np.ndarray):
        denom = np.maximum(np.maximum(a, b), eps)
        rel = np.abs(a - b) / denom
        bad = (rel > float(rtol)) & a_valid & b_valid
        return bad

    # Horizontal neighbors.
    a = depth[:, :-1]
    b = depth[:, 1:]
    a_valid = valid[:, :-1]
    b_valid = valid[:, 1:]
    bad = _mark(a, b, a_valid, b_valid)
    edge[:, :-1] |= bad
    edge[:, 1:] |= bad

    # Vertical neighbors.
    a = depth[:-1, :]
    b = depth[1:, :]
    a_valid = valid[:-1, :]
    b_valid = valid[1:, :]
    bad = _mark(a, b, a_valid, b_valid)
    edge[:-1, :] |= bad
    edge[1:, :] |= bad

    return edge


def _get_sparse_depth_map(
    *,
    points3d,
    w2c: np.ndarray,
    K: np.ndarray,
    point3d_ids: np.ndarray,
    height: int,
    width: int,
) -> Optional[np.ndarray]:
    if point3d_ids.size == 0 or bool(np.all(point3d_ids == -1)):
        return None

    valid_ids = [int(pid) for pid in point3d_ids.tolist() if int(pid) != -1 and int(pid) in points3d]
    if not valid_ids:
        return None

    points = np.asarray([points3d[pid].xyz for pid in valid_ids], dtype=np.float64)
    errs = np.asarray([points3d[pid].error for pid in valid_ids], dtype=np.float64)
    num_views = np.asarray([len(points3d[pid].image_ids) for pid in valid_ids], dtype=np.float64)

    R = np.asarray(w2c[:3, :3], dtype=np.float64)
    t = np.asarray(w2c[:3, 3], dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    cam = points @ R.T + t[None, :]
    proj = cam @ K.T
    z = proj[:, 2]
    posz = z > 0
    if not bool(np.any(posz)):
        return None
    proj = proj[posz]
    errs = errs[posz]
    num_views = num_views[posz]
    z = z[posz]

    xy = proj[:, :2] / proj[:, 2:3]

    sdpt = np.zeros((int(height), int(width), 3), dtype=np.float32)
    for (x, y), depth, err, views in zip(xy, z, errs, num_views):
        xi = int(np.clip(int(round(float(x))), 0, int(width) - 1))
        yi = int(np.clip(int(round(float(y))), 0, int(height) - 1))
        sdpt[yi, xi, 0] = float(depth)
        sdpt[yi, xi, 1] = float(err)
        sdpt[yi, xi, 2] = float(views)
    if not bool(np.any(sdpt[..., 0] > 0)):
        return None
    return sdpt


def _robust_affine_depth_fit(
    *,
    d_pred: np.ndarray,
    d_colmap: np.ndarray,
    clip_low: float,
    clip_high: float,
    ransac_trials: int,
    verbose: bool,
) -> tuple[float, float]:
    """Fit depth = a * pred + b mapping with optional sklearn RANSAC, else fallback."""
    d_pred = np.asarray(d_pred, dtype=np.float64).reshape(-1)
    d_colmap = np.asarray(d_colmap, dtype=np.float64).reshape(-1)
    if d_pred.size != d_colmap.size:
        raise ValueError("d_pred/d_colmap size mismatch")

    clip_mask = np.ones_like(d_colmap, dtype=bool)
    if 0.0 <= float(clip_low) < float(clip_high) <= 100.0 and d_colmap.size > 0:
        lo, hi = np.percentile(d_colmap, [float(clip_low), float(clip_high)])
        if float(hi) > float(lo):
            clip_mask = (d_colmap >= lo) & (d_colmap <= hi)

    x = d_pred[clip_mask]
    y = d_colmap[clip_mask]

    # Try sklearn RANSAC first (best behavior).
    try:
        from sklearn.linear_model import LinearRegression, RANSACRegressor  # type: ignore

        ransac = RANSACRegressor(
            LinearRegression(),
            max_trials=int(ransac_trials),
            random_state=0,
        )
        ransac.fit(x.reshape(-1, 1), y.reshape(-1, 1))
        a = float(ransac.estimator_.coef_.ravel()[0])
        b = float(ransac.estimator_.intercept_.ravel()[0])
        return a, b
    except Exception as exc:
        if verbose:
            print(f"[WARN] sklearn RANSAC unavailable/failed: {exc}. Falling back to robust scale.")

    # Fallback: robust scale only (no intercept), after filtering.
    ratio = y / np.maximum(x, 1e-12)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if ratio.size == 0:
        return 1.0, 0.0
    a = float(np.median(ratio))
    return a, 0.0


@dataclass(frozen=True)
class _SaveItem:
    image_name: str
    normal: Optional[torch.Tensor]
    raw_mask: Optional[torch.Tensor]
    depth: Optional[torch.Tensor]
    alpha_exists: Optional[np.ndarray]


def run_moge_infer(
    *,
    data_dir: str,
    factor: int,
    model_id: str,
    read_threads: int,
    write_threads: int,
    queue_size: int,
    save_depth: bool,
    save_depth_vis: bool,
    save_sky_mask: bool,
    remove_depth_edge: bool,
    depth_edge_rtol: float,
    align_depth_with_colmap: bool,
    colmap_min_sparse: int,
    colmap_clip_low: float,
    colmap_clip_high: float,
    colmap_ransac_trials: int,
    out_normal_dir_name: str,
    out_depth_dir_name: str,
    out_depth_vis_dir_name: str,
    out_sky_mask_dir_name: str,
    verbose: bool,
) -> None:
    data_dir = str(data_dir)
    factor = int(factor)
    if factor != 1:
        print(
            f"[WARN] factor={factor} is supported for this tool, but FriendlySplat training "
            "currently requires factor=1 when using auxiliary priors."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Resolve image list in the same order as COLMAP uses.
    dataparser = ColmapDataParser(
        data_dir=data_dir,
        factor=factor,
        normalize_world_space=False,
        align_world_axes=False,
        test_every=8,
        benchmark_train_split=False,
        depth_dir_name=None,
        normal_dir_name=None,
        dynamic_mask_dir_name=None,
        sky_mask_dir_name=None,
    )
    image_names = list(dataparser.image_names)
    image_paths = list(dataparser.image_paths)
    idx_by_name = {str(n): int(i) for i, n in enumerate(image_names)}
    if not image_names:
        raise ValueError("No images found from ColmapDataParser.")
    print(f"Found {len(image_names)} images.")

    # Optional COLMAP context for depth alignment.
    images_by_name = None
    points3d = None
    Ks_per_image = np.asarray(dataparser.Ks)  # [N, 3, 3]
    image_sizes = np.asarray(dataparser.image_sizes)  # [N, 2] (w, h)
    if bool(align_depth_with_colmap):
        colmap_dir = None
        for candidate in [
            Path(data_dir) / "sparse" / "0",
            Path(data_dir) / "sparse",
            Path(data_dir) / "colmap" / "sparse" / "0",
        ]:
            if candidate.exists():
                colmap_dir = candidate
                break
        if colmap_dir is None:
            print("[WARN] COLMAP model not found; disabling depth alignment.")
            align_depth_with_colmap = False
        else:
            cams, images, points3d = read_model(str(colmap_dir))
            images_by_name = {img.name: img for img in images.values()}
            print(f"Loaded COLMAP model from {colmap_dir} (images={len(images_by_name)}).")

    print("Loading MoGe model...")
    model_load_start = time.time()
    try:
        from moge.model.v2 import MoGeModel  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import `moge`. Install it first, e.g. `pip install moge`."
        ) from exc
    model = MoGeModel.from_pretrained(str(model_id)).to(device)
    model.eval()
    print(f"MoGe model loaded in {time.time() - model_load_start:.2f}s.")

    normal_output_dir = os.path.join(data_dir, str(out_normal_dir_name))
    os.makedirs(normal_output_dir, exist_ok=True)

    depth_output_dir = None
    depth_vis_output_dir = None
    sky_mask_output_dir = None
    if bool(save_depth):
        depth_output_dir = os.path.join(data_dir, str(out_depth_dir_name))
        os.makedirs(depth_output_dir, exist_ok=True)
    if bool(save_depth_vis):
        depth_vis_output_dir = os.path.join(data_dir, str(out_depth_vis_dir_name))
        os.makedirs(depth_vis_output_dir, exist_ok=True)
    if bool(save_sky_mask):
        sky_mask_output_dir = os.path.join(data_dir, str(out_sky_mask_dir_name))
        os.makedirs(sky_mask_output_dir, exist_ok=True)

    queue_size = max(1, int(queue_size))
    read_threads = max(1, int(read_threads))
    write_threads = max(1, int(write_threads))

    filename_queue: "Queue[Optional[tuple[int, str, str]]]" = Queue()
    data_queue: "Queue[tuple[int, str, Optional[np.ndarray], float]]" = Queue(maxsize=queue_size)
    save_queue: "Queue[Optional[_SaveItem]]" = Queue(maxsize=queue_size)

    for idx, (name, path) in enumerate(zip(image_names, image_paths)):
        filename_queue.put((int(idx), str(name), str(path)))
    for _ in range(read_threads):
        filename_queue.put(None)

    def reader_worker() -> None:
        while True:
            item = filename_queue.get()
            if item is None:
                filename_queue.task_done()
                break
            idx, name, path = item
            read_start = time.time()
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                data_queue.put((idx, name, None, None, time.time() - read_start))
                filename_queue.task_done()
                continue

            alpha_exists = None
            if img.ndim == 2:
                img_rgb_u8 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                if int(img.shape[-1]) == 4:
                    alpha = img[..., 3]
                    alpha_exists = _alpha_to_exists_mask(alpha)
                    img_rgb_u8 = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                elif int(img.shape[-1]) == 3:
                    img_rgb_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    img_rgb_u8 = img[..., :3]
                    img_rgb_u8 = cv2.cvtColor(img_rgb_u8, cv2.COLOR_BGR2RGB)

            img_rgb = img_rgb_u8.astype(np.float32) / 255.0
            img_rgb = np.ascontiguousarray(img_rgb)
            data_queue.put((idx, name, img_rgb, alpha_exists, time.time() - read_start))
            filename_queue.task_done()

    def writer_worker() -> None:
        while True:
            item = save_queue.get()
            if item is None:
                save_queue.task_done()
                break

            image_name = str(item.image_name)
            base_name = os.path.splitext(image_name)[0]
            idx = idx_by_name.get(image_name)
            if idx is None:
                # Should not happen, but keep the writer robust.
                save_queue.task_done()
                continue
            w_expected, h_expected = image_sizes[int(idx)]

            alpha_exists = item.alpha_exists
            if alpha_exists is not None and alpha_exists.shape[:2] != (int(h_expected), int(w_expected)):
                alpha_exists = cv2.resize(
                    alpha_exists.astype(np.uint8),
                    (int(w_expected), int(h_expected)),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            raw_mask_np = None
            if item.raw_mask is not None:
                raw_mask_np = _to_numpy(item.raw_mask)
                if raw_mask_np.dtype != np.bool_:
                    raw_mask_np = raw_mask_np > 0.5

            if item.normal is not None and item.normal.numel() > 0:
                normal_np = item.normal.detach().cpu().numpy()
                # Encode to match FriendlySplat decode: normal = 1 - (u8/255)*2.
                normal_vis = (0.5 - 0.5 * normal_np)  # (1 - normal)/2
                normal_u8 = np.clip(normal_vis * 255.0, 0, 255).astype(np.uint8)
                if normal_u8.shape[:2] != (int(h_expected), int(w_expected)):
                    normal_u8 = cv2.resize(
                        normal_u8,
                        (int(w_expected), int(h_expected)),
                        interpolation=cv2.INTER_NEAREST,
                    )
                if alpha_exists is not None:
                    normal_u8 = normal_u8.copy()
                    normal_u8[~alpha_exists] = 127
                normal_path = os.path.join(normal_output_dir, f"{base_name}.png")
                os.makedirs(os.path.dirname(normal_path), exist_ok=True)
                bgr = cv2.cvtColor(normal_u8, cv2.COLOR_RGB2BGR)
                if not cv2.imwrite(normal_path, bgr):
                    print(f"[WARN] Failed to write normal: {normal_path}")

            depth_np = item.depth.detach().cpu().numpy() if item.depth is not None else None
            depth_mask_np = raw_mask_np.copy() if raw_mask_np is not None else None

            if depth_np is not None:
                depth_np = depth_np.astype(np.float32, copy=False)
                finite = np.isfinite(depth_np)
                if not bool(np.all(finite)):
                    depth_np = np.where(finite, depth_np, 0.0).astype(np.float32, copy=False)
                    depth_mask_np = finite if depth_mask_np is None else (depth_mask_np & finite)
                if depth_np.shape[:2] != (int(h_expected), int(w_expected)):
                    depth_np = cv2.resize(
                        depth_np,
                        (int(w_expected), int(h_expected)),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    if depth_mask_np is not None:
                        depth_mask_np = cv2.resize(
                            depth_mask_np.astype(np.uint8),
                            (int(w_expected), int(h_expected)),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                if alpha_exists is not None:
                    depth_np = depth_np.copy()
                    depth_np[~alpha_exists] = 0.0
                    depth_mask_np = (
                        alpha_exists
                        if depth_mask_np is None
                        else (depth_mask_np & alpha_exists)
                    )

            # Export invalid-depth mask for training (255=invalid, 0=valid).
            if bool(save_sky_mask) and sky_mask_output_dir is not None and depth_np is not None:
                valid = np.isfinite(depth_np) & (depth_np > 0)
                if depth_mask_np is not None:
                    valid = valid & depth_mask_np.astype(bool)
                invalid_u8 = (~valid).astype(np.uint8) * 255
                mask_path = os.path.join(sky_mask_output_dir, f"{base_name}.png")
                os.makedirs(os.path.dirname(mask_path), exist_ok=True)
                if not cv2.imwrite(mask_path, invalid_u8):
                    print(f"[WARN] Failed to write sky mask: {mask_path}")

            # Align predicted depth to COLMAP sparse depth (per-image affine fit).
            if (
                bool(align_depth_with_colmap)
                and depth_np is not None
                and images_by_name is not None
                and points3d is not None
            ):
                img = images_by_name.get(image_name)
                if img is None:
                    if verbose:
                        print(f"[WARN] Image not found in COLMAP model: {image_name}")
                else:
                    # Get K from dataparser (already factor-adjusted and possibly rescaled).
                    K = Ks_per_image[int(idx)]
                    sdpt = _get_sparse_depth_map(
                        points3d=points3d,
                        w2c=get_extrinsic(img),
                        K=K,
                        point3d_ids=img.point3D_ids,
                        height=int(h_expected),
                        width=int(w_expected),
                    )
                    if sdpt is not None:
                        sparse_depth = sdpt[..., 0]
                        valid_sparse = sparse_depth > 0
                        if depth_mask_np is not None:
                            valid_sparse = valid_sparse & depth_mask_np
                        valid_sparse = valid_sparse & np.isfinite(depth_np) & (depth_np > 0)
                        if int(np.count_nonzero(valid_sparse)) >= int(colmap_min_sparse):
                            d_pred = depth_np[valid_sparse]
                            d_colmap = sparse_depth[valid_sparse]
                            a, b = _robust_affine_depth_fit(
                                d_pred=d_pred,
                                d_colmap=d_colmap,
                                clip_low=float(colmap_clip_low),
                                clip_high=float(colmap_clip_high),
                                ransac_trials=int(colmap_ransac_trials),
                                verbose=bool(verbose),
                            )
                            if verbose:
                                print(f"[COLMAP] {image_name}: depth = {a:.6f} * pred + {b:.6f}")
                            depth_np = depth_np * float(a) + float(b)

            # Remove depth edges by masking out discontinuities (optional).
            if depth_np is not None and bool(remove_depth_edge):
                edge = _depth_edge_mask(
                    depth_np,
                    rtol=float(depth_edge_rtol),
                    mask=depth_mask_np,
                )
                depth_mask_np = (~edge) if depth_mask_np is None else (depth_mask_np & (~edge))

            if bool(save_depth) and depth_np is not None and depth_output_dir is not None:
                depth_to_save = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0).astype(
                    np.float32, copy=False
                )
                if depth_mask_np is not None:
                    depth_to_save = depth_to_save.copy()
                    depth_to_save[~depth_mask_np] = 0.0
                depth_path = os.path.join(depth_output_dir, f"{base_name}.npy")
                os.makedirs(os.path.dirname(depth_path), exist_ok=True)
                np.save(depth_path, depth_to_save)

            if bool(save_depth_vis) and depth_np is not None and depth_vis_output_dir is not None:
                depth_vis = depth_np.astype(np.float32, copy=False)
                depth_vis = np.nan_to_num(depth_vis, nan=0.0, posinf=0.0, neginf=0.0)
                valid = (depth_vis > 0) & np.isfinite(depth_vis)
                if depth_mask_np is not None:
                    valid = valid & depth_mask_np
                if bool(np.any(valid)):
                    d = depth_vis[valid]
                    lo, hi = np.percentile(d, [2.0, 98.0])
                    if float(hi) > float(lo):
                        norm = np.clip((depth_vis - lo) / (hi - lo), 0.0, 1.0)
                    else:
                        norm = np.zeros_like(depth_vis)
                    depth_u8 = (norm * 255.0).astype(np.uint8)
                    if depth_mask_np is not None:
                        depth_u8 = depth_u8.copy()
                        depth_u8[~depth_mask_np] = 0
                    cmap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
                    depth_color = cv2.applyColorMap(depth_u8, cmap)
                    if depth_mask_np is not None:
                        depth_color = depth_color.copy()
                        depth_color[~depth_mask_np] = 255
                    vis_path = os.path.join(depth_vis_output_dir, f"{base_name}.png")
                    os.makedirs(os.path.dirname(vis_path), exist_ok=True)
                    if not cv2.imwrite(vis_path, depth_color):
                        print(f"[WARN] Failed to write depth vis: {vis_path}")

            save_queue.task_done()

    reader_threads = [
        threading.Thread(target=reader_worker, name=f"reader-{i}", daemon=True)
        for i in range(read_threads)
    ]
    writer_threads = [
        threading.Thread(target=writer_worker, name=f"writer-{i}", daemon=True)
        for i in range(write_threads)
    ]

    for t in reader_threads + writer_threads:
        t.start()

    pbar = tqdm.tqdm(total=len(image_names), desc="MoGe infer", unit="img")
    total_read_time = 0.0
    total_infer_time = 0.0

    processed = 0
    try:
        while processed < len(image_names):
            idx, name, img_rgb, alpha_exists, read_time = data_queue.get()
            total_read_time += float(read_time)
            if img_rgb is None:
                if verbose:
                    tqdm.tqdm.write(f"[WARN] Failed to read: {name}")
                processed += 1
                pbar.update(1)
                continue

            tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).to(device)
            infer_start = time.time()
            with torch.no_grad():
                output = model.infer(tensor)
            total_infer_time += float(time.time() - infer_start)

            mask_t = output.get("mask")
            normal_t = output.get("normal")
            depth_t = (
                output.get("depth")
                if (save_depth or save_depth_vis or save_sky_mask or align_depth_with_colmap)
                else None
            )

            item = _SaveItem(
                image_name=str(name),
                normal=normal_t.detach().cpu() if isinstance(normal_t, torch.Tensor) else None,
                raw_mask=mask_t.detach().cpu() if isinstance(mask_t, torch.Tensor) else None,
                depth=depth_t.detach().cpu() if isinstance(depth_t, torch.Tensor) else None,
                alpha_exists=alpha_exists,
            )
            save_queue.put(item)

            processed += 1
            pbar.update(1)
    finally:
        pbar.close()

    filename_queue.join()
    save_queue.join()

    for _ in writer_threads:
        save_queue.put(None)
    for t in writer_threads:
        t.join()
    for t in reader_threads:
        t.join()

    print("=" * 50)
    print("Done.")
    print(f"Total read time: {total_read_time:.2f}s")
    print(f"Total inference time: {total_infer_time:.2f}s")
    if len(image_names) > 0:
        print(f"Avg inference/img: {total_infer_time / len(image_names):.3f}s")
    print("=" * 50)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MoGe to export depth/normal priors for FriendlySplat.")
    p.add_argument("--data-dir", type=str, required=True, help="Scene root containing COLMAP + images.")
    p.add_argument(
        "--factor",
        type=int,
        default=1,
        help="Image downsample factor (uses images_{factor} if present). Priors are intended for factor=1.",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="Ruicheng/moge-2-vitl-normal",
        help="MoGe pretrained model id.",
    )

    p.add_argument("--read-threads", type=int, default=2, help="Parallel image read threads.")
    p.add_argument("--write-threads", type=int, default=4, help="Parallel write threads.")
    p.add_argument("--queue-size", type=int, default=8, help="Read/write queue size.")

    p.add_argument("--save-depth", action="store_true", help="Export dense depth .npy files.")
    p.add_argument("--save-depth-vis", action="store_true", help="Export depth visualization .png files.")
    p.add_argument("--save-sky-mask", action="store_true", help="Export invalid-depth mask as .png (255=invalid).")

    p.add_argument(
        "--remove-depth-edge",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask out depth edges (default: on).",
    )
    p.add_argument("--depth-edge-rtol", type=float, default=0.04, help="Relative threshold for depth edge mask.")

    p.add_argument(
        "--align-depth-with-colmap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align depth to COLMAP sparse depths when available (default: on).",
    )
    p.add_argument("--colmap-min-sparse", type=int, default=30, help="Min sparse points to run alignment.")
    p.add_argument("--colmap-clip-low", type=float, default=5.0, help="Sparse depth clip low percentile.")
    p.add_argument("--colmap-clip-high", type=float, default=95.0, help="Sparse depth clip high percentile.")
    p.add_argument("--colmap-ransac-trials", type=int, default=2000, help="RANSAC max trials.")

    p.add_argument("--out-normal-dir", type=str, default="moge_normal", help="Output dir name for normals.")
    p.add_argument("--out-depth-dir", type=str, default="moge_depth", help="Output dir name for depths.")
    p.add_argument("--out-depth-vis-dir", type=str, default="moge_depth_vis", help="Output dir name for depth vis.")
    p.add_argument("--out-sky-mask-dir", type=str, default="sky_mask", help="Output dir name for sky mask.")

    p.add_argument("--verbose", action="store_true", help="Verbose logging.")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    run_moge_infer(
        data_dir=args.data_dir,
        factor=args.factor,
        model_id=args.model_id,
        read_threads=args.read_threads,
        write_threads=args.write_threads,
        queue_size=args.queue_size,
        save_depth=bool(args.save_depth),
        save_depth_vis=bool(args.save_depth_vis),
        save_sky_mask=bool(args.save_sky_mask),
        remove_depth_edge=bool(args.remove_depth_edge),
        depth_edge_rtol=float(args.depth_edge_rtol),
        align_depth_with_colmap=bool(args.align_depth_with_colmap),
        colmap_min_sparse=int(args.colmap_min_sparse),
        colmap_clip_low=float(args.colmap_clip_low),
        colmap_clip_high=float(args.colmap_clip_high),
        colmap_ransac_trials=int(args.colmap_ransac_trials),
        out_normal_dir_name=str(args.out_normal_dir),
        out_depth_dir_name=str(args.out_depth_dir),
        out_depth_vis_dir_name=str(args.out_depth_vis_dir),
        out_sky_mask_dir_name=str(args.out_sky_mask_dir),
        verbose=bool(args.verbose),
    )


if __name__ == "__main__":
    main()
