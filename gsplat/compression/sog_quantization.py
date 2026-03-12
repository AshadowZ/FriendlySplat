from __future__ import annotations

import numpy as np
import torch


def _to_numpy_f32(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)


def _build_cost_matrix(
    prefix_w: np.ndarray,
    prefix_wx: np.ndarray,
    prefix_wxx: np.ndarray,
) -> np.ndarray:
    h = prefix_w.shape[0] - 1
    a = np.arange(h, dtype=np.int32)[:, None]
    b = np.arange(h, dtype=np.int32)[None, :]
    valid = b >= a

    w = prefix_w[b + 1] - prefix_w[a]
    wx = prefix_wx[b + 1] - prefix_wx[a]
    wxx = prefix_wxx[b + 1] - prefix_wxx[a]

    cost = np.full((h, h), np.inf, dtype=np.float64)
    non_empty = valid & (w > 0.0)
    cost[non_empty] = wxx[non_empty] - (wx[non_empty] * wx[non_empty]) / w[non_empty]
    cost[valid & ~non_empty] = 0.0
    return cost


def quantize_1d(
    values: np.ndarray | torch.Tensor,
    k: int = 256,
    alpha: float = 0.5,
    histogram_bins: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize pooled values using the same histogram+DP strategy as splat-transform."""

    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > 256:
        raise ValueError(f"Only 8-bit labels are supported, got k={k}")
    if histogram_bins <= 1:
        raise ValueError(f"histogram_bins must be > 1, got {histogram_bins}")

    data = _to_numpy_f32(values)
    original_shape = data.shape
    flat = data.reshape(-1)

    if flat.size == 0:
        return np.zeros((k,), dtype=np.float32), np.zeros(original_shape, dtype=np.uint8)

    data_min = float(np.min(flat))
    data_max = float(np.max(flat))
    if data_max - data_min < 1e-20:
        centroids = np.full((k,), data_min, dtype=np.float32)
        labels = np.zeros(original_shape, dtype=np.uint8)
        return centroids, labels

    bin_width = (data_max - data_min) / float(histogram_bins)
    bins = np.floor((flat - data_min) / bin_width).astype(np.int32)
    bins = np.clip(bins, 0, histogram_bins - 1)

    counts = np.bincount(bins, minlength=histogram_bins).astype(np.float64)
    sums = np.bincount(bins, weights=flat, minlength=histogram_bins).astype(np.float64)

    fallback_centers = data_min + (np.arange(histogram_bins, dtype=np.float64) + 0.5) * bin_width
    centers = fallback_centers.copy()
    np.divide(sums, counts, out=centers, where=counts > 0.0)
    weights = np.where(counts > 0.0, np.power(counts, alpha), 0.0)

    prefix_w = np.concatenate(([0.0], np.cumsum(weights)))
    prefix_wx = np.concatenate(([0.0], np.cumsum(weights * centers)))
    prefix_wxx = np.concatenate(([0.0], np.cumsum(weights * centers * centers)))

    def range_mean(a: int, b: int) -> float:
        w = prefix_w[b + 1] - prefix_w[a]
        if w <= 0.0:
            return float((centers[a] + centers[b]) * 0.5)
        return float((prefix_wx[b + 1] - prefix_wx[a]) / w)

    cost = _build_cost_matrix(prefix_w, prefix_wx, prefix_wxx)
    non_empty_bins = int(np.count_nonzero(counts))
    effective_k = max(1, min(k, non_empty_bins))

    dp_prev = cost[0].copy()
    split_table: list[np.ndarray | None] = [None] * (effective_k + 1)
    split_table[1] = np.full((histogram_bins,), -1, dtype=np.int32)

    for m in range(2, effective_k + 1):
        dp_curr = np.full((histogram_bins,), np.inf, dtype=np.float64)
        split_m = np.full((histogram_bins,), m - 2, dtype=np.int32)

        for j in range(m - 1, histogram_bins):
            s = np.arange(m - 2, j, dtype=np.int32)
            candidates = dp_prev[s] + cost[s + 1, j]
            best_index = int(np.argmin(candidates))
            dp_curr[j] = float(candidates[best_index])
            split_m[j] = int(s[best_index])

        split_table[m] = split_m
        dp_prev = dp_curr

    centroid_values = np.empty((effective_k,), dtype=np.float32)
    j = histogram_bins - 1
    for m in range(effective_k, 0, -1):
        split = -1 if m == 1 else int(split_table[m][j])  # type: ignore[index]
        centroid_values[m - 1] = np.float32(range_mean(split + 1, j))
        j = split

    centroid_values.sort()
    centroids = np.empty((k,), dtype=np.float32)
    centroids[:effective_k] = centroid_values
    if effective_k < k:
        centroids[effective_k:] = centroid_values[effective_k - 1]

    thresholds = (centroids[:-1] + centroids[1:]) * 0.5
    labels = np.searchsorted(thresholds, flat, side="left").astype(np.uint8)
    return centroids, labels.reshape(original_shape)
