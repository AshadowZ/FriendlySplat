from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class KMeansResult:
    centroids: np.ndarray
    labels: np.ndarray
    backend: str
    device: str
    batch_size: int | None = None


def _resolve_device(
    features: torch.Tensor,
    cluster_device: str | torch.device | None,
) -> torch.device:
    if cluster_device is not None:
        device = torch.device(cluster_device)
        if device.type != "cuda":
            raise RuntimeError(f"SOG SH clustering requires a CUDA device, got: {device}")
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA clustering requested but CUDA is unavailable: {device}")
        return device

    if not torch.cuda.is_available():
        raise RuntimeError("SOG export requires CUDA for SH clustering, but CUDA is unavailable.")

    if features.device.type == "cuda":
        return features.device
    return torch.device("cuda")


def _initialize_centroids(
    points: torch.Tensor,
    num_clusters: int,
    rng: np.random.Generator,
) -> torch.Tensor:
    indices = rng.choice(points.shape[0], size=num_clusters, replace=False)
    return points[torch.from_numpy(indices).to(device=points.device, dtype=torch.long)].clone()


def _round_down(value: int, multiple: int) -> int:
    return max(multiple, (value // multiple) * multiple)


def _resolve_batch_size(num_clusters: int, device: torch.device) -> int:
    if device.type != "cuda":
        raise RuntimeError(f"SOG SH clustering batch sizing requires a CUDA device, got: {device}")

    try:
        free_bytes, _ = torch.cuda.mem_get_info(device=device)
        # Distance evaluation dominates memory: [batch, K] appears at least once,
        # and matmul / reduction typically need extra headroom on top of it.
        usable_bytes = max(int(free_bytes * 0.20), 1)
        bytes_per_point = max(num_clusters * 4 * 3, 1)
        inferred = usable_bytes // bytes_per_point
        return _round_down(max(512, int(inferred)), 128)
    except RuntimeError:
        pass

    target_distance_elements = 67_108_864
    inferred = target_distance_elements // max(num_clusters, 1)
    return _round_down(max(512, int(inferred)), 128)


def _assign_labels_torch(
    points: torch.Tensor,
    centroids: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    labels = torch.empty((points.shape[0],), device=points.device, dtype=torch.long)
    centroids_t = centroids.transpose(0, 1).contiguous()
    centroids_sq = torch.sum(centroids * centroids, dim=1)

    for start in range(0, points.shape[0], batch_size):
        end = min(start + batch_size, points.shape[0])
        batch = points[start:end]
        batch_sq = torch.sum(batch * batch, dim=1, keepdim=True)
        distances = batch_sq + centroids_sq.unsqueeze(0) - 2.0 * torch.matmul(batch, centroids_t)
        labels[start:end] = torch.argmin(distances, dim=1)
    return labels


def _update_centroids(
    points: torch.Tensor,
    labels: torch.Tensor,
    centroids: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    num_clusters, dims = centroids.shape
    sums = torch.zeros((num_clusters, dims), device=points.device, dtype=points.dtype)
    sums.index_add_(0, labels, points)

    counts = torch.bincount(labels, minlength=num_clusters)
    updated = sums / counts.clamp_min(1).unsqueeze(1).to(dtype=points.dtype)

    empty = counts == 0
    if torch.any(empty):
        reseed_indices = rng.integers(0, points.shape[0], size=int(empty.sum().item()))
        updated[empty] = points[
            torch.from_numpy(reseed_indices).to(device=points.device, dtype=torch.long)
        ]
    return updated


@torch.inference_mode()
def cluster_kmeans(
    features: torch.Tensor,
    num_clusters: int,
    iterations: int,
    *,
    cluster_device: str | torch.device | None = None,
    seed: int = 0,
) -> KMeansResult:
    if features.ndim != 2:
        raise ValueError(f"features must have shape (N, D), got {tuple(features.shape)}")
    if num_clusters <= 0:
        raise ValueError(f"num_clusters must be positive, got {num_clusters}")

    work_device = _resolve_device(features, cluster_device=cluster_device)
    points = features.detach().to(device=work_device, dtype=torch.float32, non_blocking=True)
    num_points = points.shape[0]

    if num_points == 0:
        return KMeansResult(
            centroids=np.zeros((0, points.shape[1]), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.uint16),
            backend="torch",
            device=str(work_device),
        )

    num_clusters = min(int(num_clusters), num_points)
    rng = np.random.default_rng(seed)
    batch_size = min(_resolve_batch_size(num_clusters, work_device), num_points)

    if num_clusters == num_points:
        labels = np.arange(num_points, dtype=np.uint16)
        return KMeansResult(
            centroids=points.cpu().numpy().astype(np.float32, copy=False),
            labels=labels,
            backend="torch",
            device=str(work_device),
            batch_size=batch_size,
        )

    centroids = _initialize_centroids(points, num_clusters=num_clusters, rng=rng)
    num_iterations = max(1, int(iterations))

    for _ in range(num_iterations):
        labels_t = _assign_labels_torch(points, centroids, batch_size=batch_size)
        centroids = _update_centroids(points, labels_t, centroids, rng=rng)

    labels_t = _assign_labels_torch(points, centroids, batch_size=batch_size)
    labels = labels_t.cpu().numpy().astype(np.uint16, copy=False)
    return KMeansResult(
        centroids=centroids.cpu().numpy().astype(np.float32, copy=False),
        labels=labels,
        backend="torch",
        device=str(work_device),
        batch_size=batch_size,
    )
