from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
import zipfile
from io import BytesIO
import os
from typing import Any

import numpy as np
import torch

from .compression.sog_kmeans import cluster_kmeans
from .compression.sog_quantization import quantize_1d


def _log(message: str) -> None:
    print(f"[sog-export] {message}", flush=True)


def _part1by2_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.uint64) & np.uint64(0x000003FF)
    x = (x ^ (x << np.uint64(16))) & np.uint64(0xFF0000FF)
    x = (x ^ (x << np.uint64(8))) & np.uint64(0x0300F00F)
    x = (x ^ (x << np.uint64(4))) & np.uint64(0x030C30C3)
    x = (x ^ (x << np.uint64(2))) & np.uint64(0x09249249)
    return x


def _sort_centers_np(centers: np.ndarray) -> np.ndarray:
    min_vals = np.min(centers, axis=0)
    max_vals = np.max(centers, axis=0)
    lengths = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
    scaled = np.floor((centers - min_vals) / lengths * 1024.0).astype(np.int32)
    x = scaled[:, 0]
    y = scaled[:, 1]
    z = scaled[:, 2]
    morton = (_part1by2_np(z) << np.uint64(2)) + (_part1by2_np(y) << np.uint64(1)) + _part1by2_np(x)
    return np.argsort(morton, kind="stable")


def _log_transform_np(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def _encode_webp_rgba(data: np.ndarray) -> bytes:
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "SOG export requires Pillow with WebP support. Install `Pillow`."
        ) from e

    if data.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGBA image, got {data.dtype}")
    if data.ndim != 3 or data.shape[2] != 4:
        raise ValueError(f"Expected [H, W, 4] RGBA image, got {data.shape}")

    buffer = BytesIO()
    image = Image.fromarray(data, mode="RGBA")
    image.save(buffer, format="WEBP", lossless=True, quality=100, method=4)
    return buffer.getvalue()


def _encode_webp_rgba_many(images: list[np.ndarray]) -> list[bytes]:
    if not images:
        return []
    if len(images) == 1:
        return [_encode_webp_rgba(images[0])]

    max_workers = min(len(images), max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_encode_webp_rgba, images))


def _zip_bundle(files: dict[str, bytes]) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_STORED) as zf:
        for name, payload in files.items():
            zf.writestr(name, payload)
    return buffer.getvalue()


def _quantize_1d(
    values: np.ndarray,
    max_centroids: int = 256,
    alpha: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1).astype(np.float32, copy=False)
    n_clusters = int(min(max_centroids, max(flat.size, 1)))
    _log(
        "quantize_1d: "
        f"shape={values.shape}, flat={flat.size}, unique=skipped, "
        f"clusters={n_clusters}, alpha={alpha}"
    )
    return quantize_1d(values, k=n_clusters, alpha=alpha)


def _cluster_sh_features(
    features: torch.Tensor,
    iterations: int,
    cluster_device: str | torch.device | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if features.ndim != 2:
        raise ValueError(f"features must have shape (N, D), got {tuple(features.shape)}")

    if features.shape[0] == 0:
        return (
            np.zeros((0, features.shape[1]), dtype=np.float32),
            np.zeros((0,), dtype=np.uint16),
            0,
        )

    ratio = features.shape[0] / 1024.0
    requested_palette_size = int(min(64.0, 2 ** math.floor(math.log2(ratio))) * 1024)
    _log(
        "cluster_sh_features: "
        f"features={features.shape}, iterations={iterations}, "
        f"palette_size={requested_palette_size}, cluster_device={cluster_device}"
    )
    if requested_palette_size == 1:
        centroid = features.mean(dim=0, keepdim=True).detach().cpu().numpy().astype(np.float32)
        labels = np.zeros((features.shape[0],), dtype=np.uint16)
        return centroid, labels, requested_palette_size

    tic = time.perf_counter()
    _log("cluster_sh_features: torch k-means start")
    result = cluster_kmeans(
        features,
        num_clusters=requested_palette_size,
        iterations=iterations,
        cluster_device=cluster_device,
        seed=0,
    )
    toc = time.perf_counter()
    _log(
        "cluster_sh_features: torch k-means done "
        f"in {toc - tic:.2f}s "
        f"(backend={result.backend}, device={result.device}, batch_size={result.batch_size})"
    )
    return result.centroids, result.labels, requested_palette_size


def _sog_grid_shape(num_rows: int) -> tuple[int, int]:
    width = int(math.ceil(math.sqrt(num_rows) / 4.0) * 4)
    height = int(math.ceil(num_rows / max(width, 1) / 4.0) * 4)
    return max(width, 4), max(height, 4)


def _write_means(
    means: np.ndarray,
    width: int,
    height: int,
) -> tuple[bytes, bytes, dict[str, list[float]]]:
    _log(f"write_means: count={means.shape[0]}, grid={width}x{height}")
    means_l = np.zeros((height, width, 4), dtype=np.uint8)
    means_u = np.zeros((height, width, 4), dtype=np.uint8)

    transformed = _log_transform_np(means.astype(np.float32))
    mins = np.min(transformed, axis=0)
    maxs = np.max(transformed, axis=0)
    denom = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    quantized = np.clip(np.round((transformed - mins) / denom * 65535.0), 0, 65535).astype(np.uint16)

    num_rows = means.shape[0]
    flat_means_l = means_l.reshape(-1, 4)
    flat_means_u = means_u.reshape(-1, 4)
    flat_means_l[:num_rows, 0:3] = (quantized & 0xFF).astype(np.uint8)
    flat_means_l[:num_rows, 3] = 255
    flat_means_u[:num_rows, 0:3] = ((quantized >> 8) & 0xFF).astype(np.uint8)
    flat_means_u[:num_rows, 3] = 255

    meta = {"mins": mins.tolist(), "maxs": maxs.tolist()}
    means_l_webp, means_u_webp = _encode_webp_rgba_many([means_l, means_u])
    return means_l_webp, means_u_webp, meta


def _write_quats(quats: np.ndarray, width: int, height: int) -> bytes:
    _log(f"write_quats: count={quats.shape[0]}, grid={width}x{height}")
    quat_img = np.zeros((height, width, 4), dtype=np.uint8)

    q = quats.astype(np.float32).copy()
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    q = q / norms
    max_comp = np.argmax(np.abs(q), axis=1)

    largest_values = np.take_along_axis(q, max_comp[:, None], axis=1).reshape(-1)
    flip_mask = largest_values < 0.0
    q[flip_mask] *= -1.0
    q *= math.sqrt(2.0)

    gather_indices = np.array(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        dtype=np.int64,
    )[max_comp]
    packed = np.take_along_axis(q, gather_indices, axis=1)
    packed = np.clip(np.rint(255.0 * (packed * 0.5 + 0.5)), 0, 255).astype(np.uint8)

    flat_quat = quat_img.reshape(-1, 4)
    flat_quat[: q.shape[0], 0:3] = packed
    flat_quat[: q.shape[0], 3] = (252 + max_comp).astype(np.uint8)

    return _encode_webp_rgba(quat_img)


def _write_labeled_rgba(
    labels: np.ndarray,
    width: int,
    height: int,
    alpha: np.ndarray | None = None,
) -> bytes:
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    flat_rgba = rgba.reshape(-1, 4)
    num_rows = labels.shape[0]
    flat_rgba[:num_rows, 0 : labels.shape[1]] = labels
    if alpha is None:
        flat_rgba[:num_rows, 3] = 255
    else:
        flat_rgba[:num_rows, 3] = alpha
    return _encode_webp_rgba(rgba)


def _write_scales(
    scales: np.ndarray,
    width: int,
    height: int,
) -> tuple[bytes, list[float]]:
    _log(f"write_scales: count={scales.shape[0]}, grid={width}x{height}")
    codebook, labels = _quantize_1d(scales[:, 0:3])
    return _write_labeled_rgba(labels, width, height), codebook.astype(np.float32).tolist()


def _write_sh0(
    sh0: np.ndarray,
    opacities: np.ndarray,
    width: int,
    height: int,
) -> tuple[bytes, list[float]]:
    _log(f"write_sh0: count={sh0.shape[0]}, grid={width}x{height}")
    codebook, labels = _quantize_1d(sh0[:, 0:3])
    opacity_bytes = np.clip(
        np.round((1.0 / (1.0 + np.exp(-opacities.astype(np.float32)))) * 255.0),
        0,
        255,
    ).astype(np.uint8)
    return _write_labeled_rgba(labels, width, height, opacity_bytes), codebook.astype(np.float32).tolist()


def _write_shn(
    shn: torch.Tensor,
    width: int,
    height: int,
    iterations: int,
    cluster_device: str | torch.device | None,
) -> dict[str, Any] | None:
    coeffs = shn.shape[1] // 3
    if coeffs <= 0:
        _log("write_shn: skipped (no higher-order SH coefficients)")
        return None

    if coeffs <= 3:
        bands = 1
    elif coeffs <= 8:
        bands = 2
    else:
        bands = 3
        coeffs = min(coeffs, 15)
        shn = shn[:, : coeffs * 3]

    _log(
        "write_shn: "
        f"count={shn.shape[0]}, coeffs={coeffs}, bands={bands}, feature_dim={shn.shape[1]}"
    )

    centroids, labels, requested_palette_size = _cluster_sh_features(
        shn.to(dtype=torch.float32),
        iterations=iterations,
        cluster_device=cluster_device,
    )
    codebook, centroid_labels = _quantize_1d(centroids)

    centroid_w = 64 * coeffs
    centroid_h = int(math.ceil(max(1, centroids.shape[0]) / 64.0))
    centroid_img = np.zeros((centroid_h, centroid_w, 4), dtype=np.uint8)
    centroid_blocks = centroid_img.reshape(centroid_h, 64, coeffs, 4).reshape(-1, coeffs, 4)
    num_centroids = centroids.shape[0]
    centroid_blocks[:num_centroids, :, 0] = centroid_labels[:, 0:coeffs]
    centroid_blocks[:num_centroids, :, 1] = centroid_labels[:, coeffs : 2 * coeffs]
    centroid_blocks[:num_centroids, :, 2] = centroid_labels[:, 2 * coeffs : 3 * coeffs]
    centroid_blocks[:num_centroids, :, 3] = 255

    label_img = np.zeros((height, width, 4), dtype=np.uint8)
    flat_label = label_img.reshape(-1, 4)
    flat_label[: labels.shape[0], 0] = (labels & 0xFF).astype(np.uint8)
    flat_label[: labels.shape[0], 1] = ((labels >> 8) & 0xFF).astype(np.uint8)
    flat_label[: labels.shape[0], 3] = 255

    centroids_webp, labels_webp = _encode_webp_rgba_many([centroid_img, label_img])

    return {
        "count": int(requested_palette_size),
        "bands": int(bands),
        "codebook": codebook.astype(np.float32).tolist(),
        "files": ["shN_centroids.webp", "shN_labels.webp"],
        "_centroids_webp": centroids_webp,
        "_labels_webp": labels_webp,
    }


def splat2sog_bytes(
    means: torch.Tensor,
    scales: torch.Tensor,
    quats: torch.Tensor,
    opacities: torch.Tensor,
    sh0: torch.Tensor,
    shN: torch.Tensor,
    *,
    iterations: int = 10,
    cluster_device: str | torch.device | None = None,
) -> bytes:
    export_tic = time.perf_counter()
    _log(
        "start: "
        f"means={tuple(means.shape)}, scales={tuple(scales.shape)}, quats={tuple(quats.shape)}, "
        f"opacities={tuple(opacities.shape)}, sh0={tuple(sh0.shape)}, shN={tuple(shN.shape)}"
    )
    means = means.detach()
    scales = scales.detach()
    quats = quats.detach()
    opacities = opacities.detach()
    sh0 = sh0.detach()
    shN = shN.detach()

    means_np = means.cpu().numpy().astype(np.float32, copy=False)
    scales_np = scales.cpu().numpy().astype(np.float32, copy=False)
    quats_np = quats.cpu().numpy().astype(np.float32, copy=False)
    opacities_np = opacities.cpu().numpy().astype(np.float32, copy=False)
    sh0_np = sh0.cpu().numpy().astype(np.float32, copy=False)

    num_splats = means_np.shape[0]
    _log(f"count: num_splats={num_splats}")
    if num_splats == 0:
        raise ValueError("SOG export requires at least one valid splat.")

    tic = time.perf_counter()
    _log("sorting: morton order start")
    order = _sort_centers_np(means_np)
    _log(f"sorting: morton order done in {time.perf_counter() - tic:.2f}s")
    means_np = means_np[order]
    scales_np = scales_np[order]
    quats_np = quats_np[order]
    opacities_np = opacities_np[order]
    sh0_np = sh0_np[order]
    order_t = torch.from_numpy(order).to(device=shN.device, dtype=torch.long)
    shN = shN[order_t]

    width, height = _sog_grid_shape(num_splats)
    _log(f"grid: width={width}, height={height}, padded_pixels={width * height}")

    tic = time.perf_counter()
    means_l_webp, means_u_webp, means_meta = _write_means(means_np, width, height)
    _log(f"stage done: means in {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    quats_webp = _write_quats(quats_np, width, height)
    _log(f"stage done: quats in {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    scales_webp, scales_codebook = _write_scales(scales_np, width, height)
    _log(f"stage done: scales in {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    sh0_webp, sh0_codebook = _write_sh0(sh0_np, opacities_np, width, height)
    _log(f"stage done: sh0 in {time.perf_counter() - tic:.2f}s")

    tic = time.perf_counter()
    shn_meta = _write_shn(
        shN,
        width,
        height,
        iterations=iterations,
        cluster_device=cluster_device,
    )
    _log(f"stage done: shN in {time.perf_counter() - tic:.2f}s")

    meta: dict[str, Any] = {
        "version": 2,
        "asset": {"generator": "FriendlySplat gsplat.exporter"},
        "count": int(num_splats),
        "means": {
            "mins": means_meta["mins"],
            "maxs": means_meta["maxs"],
            "files": ["means_l.webp", "means_u.webp"],
        },
        "scales": {
            "codebook": scales_codebook,
            "files": ["scales.webp"],
        },
        "quats": {"files": ["quats.webp"]},
        "sh0": {
            "codebook": sh0_codebook,
            "files": ["sh0.webp"],
        },
    }

    files: dict[str, bytes] = {
        "meta.json": json.dumps(meta, separators=(",", ":")).encode("utf-8"),
        "means_l.webp": means_l_webp,
        "means_u.webp": means_u_webp,
        "quats.webp": quats_webp,
        "scales.webp": scales_webp,
        "sh0.webp": sh0_webp,
    }

    if shn_meta is not None:
        meta["shN"] = {
            "count": shn_meta["count"],
            "bands": shn_meta["bands"],
            "codebook": shn_meta["codebook"],
            "files": shn_meta["files"],
        }
        files["meta.json"] = json.dumps(meta, separators=(",", ":")).encode("utf-8")
        files["shN_centroids.webp"] = shn_meta["_centroids_webp"]
        files["shN_labels.webp"] = shn_meta["_labels_webp"]

    tic = time.perf_counter()
    _log(f"bundling: {len(files)} files into .sog zip")
    bundled = _zip_bundle(files)
    _log(
        f"done: bundled in {time.perf_counter() - tic:.2f}s, "
        f"size={len(bundled) / (1024 * 1024):.2f} MiB, total={time.perf_counter() - export_tic:.2f}s"
    )
    return bundled
