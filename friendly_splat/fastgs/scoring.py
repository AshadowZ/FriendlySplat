from __future__ import annotations

"""Multi-view consistency scoring (the core of FastGS).

For a handful of views we render the current model, find the badly reconstructed
pixels ("metric map"), and ask *which Gaussians are responsible for them*:

- ``importance``: per-Gaussian responsibility for bad pixels, averaged over the
  scored views. Densification is restricted to Gaussians above a threshold, so
  splats are only added where the error actually is.
- ``pruning``: the same responsibility weighted by each view's photometric loss
  and min-max normalized to ``[0, 1]``. Gaussians at the top of that range are
  the ones consistently sitting on top of unresolved error across views, and are
  the first to be removed.

Upstream FastGS gets the per-Gaussian counts from a patched CUDA rasterizer
(``diff_gaussian_rasterization_fastgs`` accumulates a counter per Gaussian while
compositing). FriendlySplat is built on gsplat, so we reconstruct the same
quantity through gsplat's public API; see ``FastGSConfig.score_backend``.
"""

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, List, Optional

import torch

from friendly_splat.fastgs.config import FastGSConfig
from friendly_splat.modules.gaussian import GaussianModel


@dataclass(frozen=True)
class CachedView:
    """One training view kept for scoring (image stored as uint8 to save memory)."""

    pixels_u8: torch.Tensor  # [H, W, 3] uint8
    camtoworld: torch.Tensor  # [4, 4] float32
    K: torch.Tensor  # [3, 3] float32
    # Frame id, needed by per-frame post-processing (e.g. the bilateral grid).
    image_id: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class MultiViewScores:
    """Per-Gaussian scores produced by one scoring event."""

    # Mean per-view responsibility for badly reconstructed pixels. [N]
    importance: Optional[torch.Tensor]
    # Loss-weighted responsibility, min-max normalized to [0, 1]. [N]
    pruning: torch.Tensor
    # Number of views that contributed.
    num_views: int


class ViewCache:
    """Ring buffer of the most recently *trained* views.

    FastGS samples cameras uniformly from the training set at every scoring
    event, which in FriendlySplat would mean re-reading images from disk in the
    middle of the training loop. Because the trainer already samples views
    uniformly at random, the last `capacity` batches are themselves a uniform
    sample, so we keep them (as uint8) and score those instead.
    """

    def __init__(self, *, capacity: int) -> None:
        if int(capacity) <= 0:
            raise ValueError(f"ViewCache capacity must be > 0, got {capacity}")
        self._views: Deque[CachedView] = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self._views)

    @property
    def capacity(self) -> int:
        return int(self._views.maxlen)

    @torch.no_grad()
    def observe(
        self,
        *,
        pixels: torch.Tensor,  # [B, H, W, 3] float in [0, 1]
        camtoworlds: torch.Tensor,  # [B, 4, 4]
        Ks: torch.Tensor,  # [B, 3, 3]
        image_ids: Optional[torch.Tensor] = None,  # [B]
    ) -> None:
        if pixels.dim() != 4 or int(pixels.shape[-1]) != 3:
            raise ValueError(
                f"pixels must have shape [B, H, W, 3], got {tuple(pixels.shape)}"
            )
        pixels_u8 = pixels.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
        for i in range(int(pixels.shape[0])):
            self._views.append(
                CachedView(
                    pixels_u8=pixels_u8[i].contiguous(),
                    camtoworld=camtoworlds[i].detach().clone(),
                    K=Ks[i].detach().clone(),
                    image_id=(
                        image_ids[i].detach().clone()
                        if isinstance(image_ids, torch.Tensor)
                        else None
                    ),
                )
            )

    def sample(self, *, num_views: int, generator: torch.Generator) -> List[CachedView]:
        """Sample without replacement from the cache (all of it if too small)."""
        available = len(self._views)
        n = min(int(num_views), int(available))
        if n <= 0:
            return []
        perm = torch.randperm(available, generator=generator)[:n].tolist()
        views = list(self._views)
        return [views[i] for i in perm]


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """SSIM between two [B, H, W, 3] images in [0, 1]."""
    from fused_ssim import fused_ssim

    x = pred.permute(0, 3, 1, 2).contiguous()
    y = gt.permute(0, 3, 1, 2).contiguous()
    return fused_ssim(x, y, padding="valid")


def _photometric_loss(
    *,
    pred: torch.Tensor,
    gt: torch.Tensor,
    ssim_lambda: float,
) -> torch.Tensor:
    """FastGS's per-view scalar loss: `(1 - l) * L1 + l * (1 - SSIM)`."""
    l1 = (pred - gt).abs().mean()
    if float(ssim_lambda) <= 0.0:
        return l1
    ssim_term = 1.0 - _ssim(pred, gt)
    return (1.0 - float(ssim_lambda)) * l1 + float(ssim_lambda) * ssim_term


def _metric_map(
    *,
    pred: torch.Tensor,  # [1, H, W, 3]
    gt: torch.Tensor,  # [1, H, W, 3]
    loss_thresh: float,
) -> torch.Tensor:
    """Boolean map of badly reconstructed pixels. [H, W]

    Mirrors FastGS's `get_loss`: channel-mean L1, min-max normalized per view,
    then thresholded.
    """
    l1 = (pred - gt).abs().mean(dim=-1)[0]  # [H, W]
    lo = l1.min()
    hi = l1.max()
    denom = (hi - lo).clamp_min(1e-12)
    l1_norm = (l1 - lo) / denom
    return l1_norm > float(loss_thresh)


def _tracked_max_per_pixel(tracked_vis_thresh: float) -> int:
    """Max Gaussians the tracker can record per pixel.

    Per-pixel contributions `alpha * T` sum to at most 1, so fewer than
    `1 / thresh` Gaussians can exceed the threshold; the slack is for rounding.
    """
    return int(1.0 / max(float(tracked_vis_thresh), 1e-3)) + 2


@torch.no_grad()
def _render_rgb(
    *,
    gaussian_model: GaussianModel,
    view: CachedView,
    width: int,
    height: int,
    sh_degree: int,
    rasterize_mode: str,
    track_pixel_gaussians: bool,
    tracked_vis_thresh: float,
) -> tuple[torch.Tensor, dict]:
    """Render one cached view with the current (detached) model."""
    from gsplat.rendering import rasterization

    tensors = gaussian_model.to_render_tensors(sh_degree=int(sh_degree))
    kwargs = {}
    if track_pixel_gaussians:
        # A pixel's contributions sum to at most 1, so at most
        # floor(1 / thresh) Gaussians can exceed the threshold; +2 is slack.
        kwargs = {
            "track_pixel_gaussians": True,
            "max_gaussians_per_pixel": _tracked_max_per_pixel(tracked_vis_thresh),
            "pixel_gaussian_threshold": float(tracked_vis_thresh),
        }
    renders, _alphas, meta = rasterization(
        means=tensors["means"].detach(),
        quats=tensors["quats"].detach(),
        scales=tensors["scales"].detach(),
        opacities=tensors["opacities"].detach(),
        colors=tensors["colors"].detach(),
        viewmats=torch.linalg.inv(view.camtoworld)[None, ...],
        Ks=view.K[None, ...],
        width=int(width),
        height=int(height),
        sh_degree=int(sh_degree),
        packed=False,
        rasterize_mode=str(rasterize_mode),
        render_mode="RGB",
        **kwargs,
    )
    return renders[..., 0:3].clamp(0.0, 1.0), meta


def _counts_weighted(
    *,
    gaussian_model: GaussianModel,
    view: CachedView,
    width: int,
    height: int,
    metric_map: torch.Tensor,  # [H, W] bool
    rasterize_mode: str,
) -> torch.Tensor:
    """Per-Gaussian `sum_{p in metric_map} alpha_p * T_p`. [N]

    A 1-channel feature render with a per-Gaussian feature `s` gives
    `out_p = sum_i s_i * w_ip` with `w_ip = alpha_ip * T_ip` independent of `s`,
    so `d(sum_p mask_p * out_p) / ds_i` is exactly the alpha-weighted number of
    bad pixels Gaussian `i` is responsible for. One extra forward/backward per
    view, and no allocation that scales with the number of intersections.
    """
    from gsplat.rendering import rasterization

    device = gaussian_model.device
    n = int(gaussian_model.num_gaussians)
    features = torch.ones(
        (n, 1), device=device, dtype=torch.float32, requires_grad=True
    )
    with torch.enable_grad():
        renders, _alphas, _meta = rasterization(
            means=gaussian_model.means.detach(),
            quats=gaussian_model.quats.detach(),
            scales=gaussian_model.scales.detach(),
            opacities=gaussian_model.opacities.detach(),
            colors=features,
            viewmats=torch.linalg.inv(view.camtoworld)[None, ...],
            Ks=view.K[None, ...],
            width=int(width),
            height=int(height),
            sh_degree=None,  # `colors` are raw per-Gaussian features, not SH
            packed=False,
            rasterize_mode=str(rasterize_mode),
            render_mode="RGB",
        )
        weighted = (renders[0, ..., 0] * metric_map.to(renders.dtype)).sum()
    (grad,) = torch.autograd.grad(weighted, features)
    return grad.detach().reshape(-1)


def _counts_tracked(
    *,
    num_gaussians: int,
    meta: dict,
    metric_map: torch.Tensor,  # [H, W] bool
    width: int,
    height: int,
    tracked_vis_thresh: float,
    verbose: bool,
) -> torch.Tensor:
    """Per-Gaussian hard count of bad pixels, from gsplat's pixel tracker. [N]"""
    pairs = meta.get("pixel_gaussians")
    if not isinstance(pairs, torch.Tensor) or int(pairs.numel()) == 0:
        return torch.zeros(
            (int(num_gaussians),), device=metric_map.device, dtype=torch.float32
        )
    gs_ids = pairs[:, 0].long()
    pixel_ids = pairs[:, 1].long()
    flat_mask = metric_map.reshape(-1)
    # `pixel_ids` are global (image-major) ids; a single camera is rendered here.
    num_pixels = int(width) * int(height)
    keep = (pixel_ids < num_pixels) & flat_mask[pixel_ids.clamp_max(num_pixels - 1)]
    counts = torch.bincount(gs_ids[keep], minlength=int(num_gaussians)).to(
        torch.float32
    )
    if verbose:
        # The tracker silently drops pairs once its buffer is full.
        capacity = num_pixels * _tracked_max_per_pixel(tracked_vis_thresh)
        if int(pairs.shape[0]) >= capacity:
            print(
                "[FastGS] warning: pixel-Gaussian tracker buffer saturated "
                f"({int(pairs.shape[0])} pairs); scores are truncated. "
                "Raise fastgs.tracked_vis_thresh.",
                flush=True,
            )
    return counts


def compute_multiview_scores(
    *,
    gaussian_model: GaussianModel,
    views: List[CachedView],
    fast_cfg: FastGSConfig,
    sh_degree: int,
    ssim_lambda: float,
    rasterize_mode: str,
    need_importance: bool,
    photometric_adapter: Any = None,
) -> Optional[MultiViewScores]:
    """Score every Gaussian against `views` (FastGS `compute_gaussian_score_fastgs`)."""
    if len(views) == 0:
        return None
    n = int(gaussian_model.num_gaussians)
    if n <= 0:
        return None

    device = gaussian_model.device
    backend = str(fast_cfg.score_backend)
    total_counts = torch.zeros((n,), device=device, dtype=torch.float32)
    total_score = torch.zeros((n,), device=device, dtype=torch.float32)

    for view in views:
        height = int(view.pixels_u8.shape[0])
        width = int(view.pixels_u8.shape[1])
        gt = (view.pixels_u8.to(torch.float32) / 255.0)[None, ...]
        pred, meta = _render_rgb(
            gaussian_model=gaussian_model,
            view=view,
            width=width,
            height=height,
            sh_degree=int(sh_degree),
            rasterize_mode=str(rasterize_mode),
            track_pixel_gaussians=(backend == "tracked"),
            tracked_vis_thresh=float(fast_cfg.tracked_vis_thresh),
        )
        with torch.no_grad():
            if photometric_adapter is not None and view.image_id is not None:
                # Match the training objective: score the post-processed render,
                # otherwise photometric drift shows up as reconstruction error.
                pred = photometric_adapter.apply(
                    rgb=pred, image_ids=view.image_id.reshape(1)
                ).clamp(0.0, 1.0)
            view_loss = _photometric_loss(pred=pred, gt=gt, ssim_lambda=ssim_lambda)
            metric_map = _metric_map(
                pred=pred, gt=gt, loss_thresh=float(fast_cfg.loss_thresh)
            )

        if backend == "weighted":
            counts = _counts_weighted(
                gaussian_model=gaussian_model,
                view=view,
                width=width,
                height=height,
                metric_map=metric_map,
                rasterize_mode=str(rasterize_mode),
            )
        elif backend == "tracked":
            counts = _counts_tracked(
                num_gaussians=n,
                meta=meta,
                metric_map=metric_map,
                width=width,
                height=height,
                tracked_vis_thresh=float(fast_cfg.tracked_vis_thresh),
                verbose=bool(fast_cfg.verbose),
            )
        else:
            raise ValueError(
                f"fastgs.score_backend must be 'weighted' or 'tracked', got {backend!r}"
            )

        with torch.no_grad():
            counts = torch.nan_to_num(counts, nan=0.0, posinf=0.0, neginf=0.0)
            if need_importance:
                total_counts += counts
            total_score += counts * view_loss

    with torch.no_grad():
        lo = total_score.min()
        hi = total_score.max()
        pruning = (total_score - lo) / (hi - lo).clamp_min(1e-12)
        importance = total_counts / float(len(views)) if need_importance else None
    return MultiViewScores(
        importance=importance,
        pruning=pruning,
        num_views=int(len(views)),
    )
