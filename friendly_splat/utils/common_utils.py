from __future__ import annotations

import random
from functools import lru_cache

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = x.clamp(eps, 1.0 - eps)
    return torch.log(x / (1.0 - x))


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    # Matches gsplat/exporter.py.
    c0 = 0.28209479177387814
    return (rgb - 0.5) / c0


def save_side_by_side_png(path: str, left: torch.Tensor, right: torch.Tensor) -> None:
    """Save two images [H,W,3] float in [0,1] side-by-side as PNG."""
    left_u8 = (left.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    right_u8 = (right.clamp(0, 1) * 255.0).to(torch.uint8).cpu().numpy()
    canvas = np.concatenate([left_u8, right_u8], axis=1)
    Image.fromarray(canvas).save(path)


def knn_distances(points: torch.Tensor, k: int = 4) -> torch.Tensor:
    """Return Euclidean KNN distances. Shape [N, k]."""
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "KNN-based scale initialization requires scikit-learn. "
            "Install it (e.g. `pip install scikit-learn` or `pip install -r friendly_splat/requirements.txt`)."
        ) from e

    x_np = points.detach().cpu().numpy()
    model = NearestNeighbors(n_neighbors=int(k), metric="euclidean").fit(x_np)
    distances, _indices = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(points)


@lru_cache(maxsize=16)
def _pixel_coords_3N(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Cached homogeneous pixel coordinates [3, H*W] with +0.5 center offset."""
    xx, yy = torch.meshgrid(
        torch.arange(width, device=device, dtype=dtype),
        torch.arange(height, device=device, dtype=dtype),
        indexing="xy",
    )
    ones = torch.ones_like(xx)
    return torch.stack((xx + 0.5, yy + 0.5, ones), dim=0).reshape(3, -1)  # [3, N]


def get_implied_normal_from_depth(
    depths_bhw1: torch.Tensor, Ks_b33: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Compute surface normals from depth maps and camera intrinsics.

    Returns camera-space normals in shape [B, H, W, 3]. Border pixels are zero-padded.
    """
    if depths_bhw1.dim() == 3:
        depths_bhw1 = depths_bhw1.unsqueeze(0)
    if depths_bhw1.dim() != 4 or depths_bhw1.shape[-1] != 1:
        raise ValueError(
            f"depths_bhw1 must be [H,W,1] or [B,H,W,1], got {tuple(depths_bhw1.shape)}"
        )

    B, H, W, _ = depths_bhw1.shape
    device = depths_bhw1.device
    dtype = depths_bhw1.dtype

    if Ks_b33.dim() == 2:
        Ks_b33 = Ks_b33.unsqueeze(0).expand(B, -1, -1)
    if Ks_b33.dim() != 3 or Ks_b33.shape[0] != B or Ks_b33.shape[-2:] != (3, 3):
        raise ValueError(
            f"Ks_b33 must be [3,3] or [B,3,3] with B={B}, got {tuple(Ks_b33.shape)}"
        )

    if H < 3 or W < 3:
        return torch.zeros((B, H, W, 3), device=device, dtype=dtype)

    depths_b1hw = depths_bhw1.permute(0, 3, 1, 2).contiguous()  # [B,1,H,W]

    invK = torch.linalg.inv(Ks_b33.to(dtype=dtype))  # [B,3,3]
    pix_3N = _pixel_coords_3N(H, W, device, dtype)  # [3, H*W]
    rays_b3N = torch.matmul(invK, pix_3N)  # [B,3,N]
    points_b3hw = (rays_b3N.view(B, 3, H, W) * depths_b1hw)  # [B,3,H,W]

    grad_x = points_b3hw[:, :, 1:-1, 2:] - points_b3hw[:, :, 1:-1, :-2]  # [B,3,H-2,W-2]
    grad_y = points_b3hw[:, :, 2:, 1:-1] - points_b3hw[:, :, :-2, 1:-1]  # [B,3,H-2,W-2]

    normals_b3hw = -torch.cross(grad_x, grad_y, dim=1)  # [B,3,H-2,W-2]
    normals_b3hw = torch.nn.functional.normalize(normals_b3hw, dim=1, eps=eps)

    normals_b3hw = torch.nn.functional.pad(normals_b3hw, (1, 1, 1, 1), value=0.0)
    return normals_b3hw.permute(0, 2, 3, 1).contiguous()


__all__ = [
    "get_implied_normal_from_depth",
    "knn_distances",
    "logit",
    "rgb_to_sh",
    "save_side_by_side_png",
    "set_seed",
]
