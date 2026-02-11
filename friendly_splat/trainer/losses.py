from __future__ import annotations

"""Loss utilities used by trainers under `friendly_splat/`.

This module centralizes loss computations so that `friendly_splat/trainer.py` can focus on
orchestration (data -> render -> loss -> optimize).

Conventions used here:
- Images are float tensors in **[0, 1]** with shape **[B, H, W, 3]** unless stated otherwise.
- Depth maps are float tensors with shape **[B, H, W, 1]** (or broadcastable to that).
- Masks are boolean tensors with shape **[B, H, W]** (or [H, W]) where True means "valid".
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from fused_ssim import fused_ssim

from friendly_splat.models.gaussian import GaussianModel
from friendly_splat.trainer.configs import RegConfig
from friendly_splat.utils.common_utils import get_implied_normal_from_depth


def ssim(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute SSIM between two RGB images.

    This uses the external `fused_ssim` package (CUDA-accelerated) and therefore expects
    NCHW layout, so we permute inputs from BHWC -> BCHW.

    Args:
        img1: Predicted image in [0, 1], shape [B, H, W, 3].
        img2: Target image in [0, 1], shape [B, H, W, 3].

    Returns:
        Scalar SSIM value (higher is better). Gradients propagate to `img1`.

    Notes:
        We call `fused_ssim(..., padding="valid")`, which computes SSIM only where the
        SSIM window fully fits inside the image. This avoids border effects but means
        very small images may not be supported by the underlying implementation.
    """
    if img1.shape != img2.shape:
        raise ValueError(f"SSIM shape mismatch: {img1.shape} vs {img2.shape}")
    if img1.dim() != 4 or img1.shape[-1] != 3:
        raise ValueError(f"SSIM expects [B,H,W,3], got {img1.shape}")

    # Convert from BHWC (trainer convention) to BCHW (fused_ssim convention).
    x = img1.permute(0, 3, 1, 2)
    y = img2.permute(0, 3, 1, 2)
    return fused_ssim(x, y, padding="valid")


def ssim_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """SSIM loss (lower is better): `1 - SSIM(img1, img2)`."""
    return 1.0 - ssim(img1, img2)


def expected_depth_l1_loss(
    expected_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """L1 loss between expected depth and a depth prior.

    Args:
        expected_depth: Expected depth prediction, shape [B, H, W, 1] (or [B, H, W]).
        gt_depth: Ground-truth/prior depth, shape [B, H, W, 1] (or [B, H, W]).
            Convention: gt_depth > 0 indicates valid pixels.
        valid_mask: Optional additional validity mask, shape [B, H, W] (or [H, W]).
            This is AND-ed with (gt_depth > 0).

    Returns:
        A scalar tensor depth loss. Returns 0 if no valid pixels.

    Implementation details:
        - Shapes are normalized to [B, H, W, 1] to avoid accidental broadcasting issues.
        - We compute errors with `torch.where(valid, ..., 0)` to avoid slow sparse indexing.
        - We normalize per-image by the number of valid pixels, then average over batch.
    """
    device = gt_depth.device
    if gt_depth.numel() == 0 or expected_depth.numel() == 0:
        return torch.tensor(0.0, device=device)

    ed = expected_depth
    gt = gt_depth
    if ed.dim() == 3:
        ed = ed.unsqueeze(-1)
    if gt.dim() == 3:
        gt = gt.unsqueeze(-1)

    valid = gt > 0.0
    if valid_mask is not None:
        m = valid_mask
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.dim() == 3:
            m = m.unsqueeze(-1)
        valid = valid & m.bool()

    if not valid.any():
        return torch.tensor(0.0, device=device)

    err = torch.where(valid, (ed - gt).abs(), 0.0)
    per_image = err.sum(dim=(1, 2, 3)) / valid.sum(dim=(1, 2, 3)).clamp(min=1)
    return per_image.mean()


def flatness_loss_from_log_scales(log_scales: torch.Tensor) -> torch.Tensor:
    """Flatness regularization for Gaussians (PhysGauss-style).

    This encourages each Gaussian to become "flat" by shrinking its smallest axis.
    Minimizing this loss tends to produce disk-/sheet-like Gaussians rather than spheres.

    Args:
        log_scales: Log-scales tensor of shape [N, 3] (or [..., N, 3]).
            This matches how we store scales in `friendly_splat/trainer.py` (log-parameterization).

    Returns:
        Scalar loss: mean of the minimum (linear) scale per Gaussian.
    """
    scales = torch.exp(log_scales)
    return scales.amin(dim=-1).mean()


def scale_ratio_regularization_from_log_scales(
    log_scales: torch.Tensor,
    *,
    max_gauss_ratio: float = 6.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Scale ratio regularization to suppress huge "spiky" Gaussians (PhysGauss-style).

    This implements the PhysGaussian scale regularization idea, but uses the ratio between
    the **maximum** and **median** scale (max/median) instead of max/min. This variant is
    commonly used in follow-up codebases and tends to encourage disk-shaped Gaussians
    without over-penalizing legitimate flatness (where min can be intentionally tiny).

    For each Gaussian:
        ratio = max(scale) / median(scale)
        loss_i = max(ratio - max_gauss_ratio, 0)

    Args:
        log_scales: Log-scales tensor of shape [N, 3] (or [..., N, 3]).
        max_gauss_ratio: Threshold above which the penalty activates.
        eps: Numerical stability epsilon for dividing by the median scale.

    Returns:
        Scalar loss: mean over Gaussians of the activated penalty.
    """
    scales = torch.exp(log_scales)
    max_s = scales.amax(dim=-1)
    med_s = scales.median(dim=-1).values.clamp(min=eps)
    ratio = max_s / med_s

    thr = ratio.new_tensor(float(max_gauss_ratio))
    return (ratio - thr).clamp(min=0.0).mean()


def cosine_normal_loss(
    pred_normals: torch.Tensor,
    gt_normals: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    eps: float = 5e-2,
) -> torch.Tensor:
    """Cosine similarity loss between predicted and target normals.

    Args:
        pred_normals: Predicted normals, shape [B, H, W, 3] (or [H, W, 3]).
        gt_normals: Target normals, shape [B, H, W, 3] (or [H, W, 3]).
            The target may be in any scale/range; we normalize both to unit vectors.
        valid_mask: Optional validity mask, shape [B, H, W] (or [H, W]) where True means valid.
        eps: Small epsilon for safe normalization and validity filtering. Increasing this
            value will treat near-zero GT normals as invalid (ignored by the loss).

    Returns:
        Scalar loss: `1 - mean(cos(theta))` over valid pixels.

    Validity rules:
        - We ignore NaNs/Infs.
        - We ignore pixels where the GT normal magnitude is ~0 (avoids undefined direction).
        - If `valid_mask` is provided, we AND it in.
    """
    if pred_normals.numel() == 0 or gt_normals.numel() == 0:
        return torch.tensor(0.0, device=pred_normals.device)

    # Normalize shapes to [B, H, W, 3].
    pred = pred_normals
    gt = gt_normals
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if gt.dim() == 3:
        gt = gt.unsqueeze(0)

    # Normalize to unit vectors before computing cosine similarity.
    pred_n = torch.nn.functional.normalize(pred, dim=-1, eps=eps)
    gt_n = torch.nn.functional.normalize(gt, dim=-1, eps=eps)
    dot = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)  # [B,H,W]

    # Filter invalid pixels.
    valid = torch.isfinite(dot)
    # Avoid sqrt: compare squared magnitudes instead of ||gt||.
    gt_mag2 = (gt * gt).sum(dim=-1)
    valid = valid & (gt_mag2 > (eps * eps))
    if valid_mask is not None:
        m = valid_mask
        if m.dim() == 2:
            m = m.unsqueeze(0)
        valid = valid & m.bool()

    if not valid.any():
        return torch.tensor(0.0, device=pred_normals.device)

    # Loss is 1 - mean(cos(theta)) so that lower is better.
    return 1.0 - dot[valid].mean()


@dataclass(frozen=True)
class LossOutput:
    total: torch.Tensor
    items: Dict[str, torch.Tensor]


def compute_losses(
    *,
    reg_cfg: RegConfig,
    do_depth_reg: bool,
    do_render_normal_reg: bool,
    do_surf_normal_reg: bool,
    do_consistency_normal_reg: bool,
    do_flat_reg: bool,
    do_scale_reg: bool,
    pixels: torch.Tensor,
    pred_rgb: torch.Tensor,
    alphas: torch.Tensor,
    expected_depth: Optional[torch.Tensor],
    render_normals: Optional[torch.Tensor],
    depth_prior: Optional[torch.Tensor],
    normal_prior: Optional[torch.Tensor],
    dynamic_mask: Optional[torch.Tensor],
    sky_mask: Optional[torch.Tensor],
    gaussian_model: GaussianModel,
    Ks: torch.Tensor,
) -> LossOutput:
    """Compute total loss and per-term metrics for one training step.

    The `do_*` switches are decided by train pipeline scheduling logic.
    """
    device = pixels.device
    batch_size, height, width = (
        int(pixels.shape[0]),
        int(pixels.shape[1]),
        int(pixels.shape[2]),
    )

    items: Dict[str, torch.Tensor] = {}

    # Shared validity mask for color/geometry terms.
    valid_color = torch.ones(
        (batch_size, height, width), dtype=torch.bool, device=device
    )
    if isinstance(sky_mask, torch.Tensor):
        valid_color = valid_color & (~sky_mask.bool())
    if isinstance(dynamic_mask, torch.Tensor):
        valid_color = valid_color & (~dynamic_mask.bool())

    # Photometric RGB loss: masked L1 + optional SSIM.
    if valid_color.any():
        diff = (pred_rgb - pixels).abs()
        l1 = diff[valid_color].mean()
        rgb_for_ssim = torch.where(valid_color.unsqueeze(-1), pred_rgb, pixels)
    else:
        l1 = torch.tensor(0.0, device=device)
        rgb_for_ssim = pixels

    ssim_term = (
        ssim_loss(rgb_for_ssim, pixels)
        if float(reg_cfg.ssim_lambda) > 0.0
        else torch.tensor(0.0, device=device)
    )
    rgb_loss = (1.0 - float(reg_cfg.ssim_lambda)) * l1 + float(
        reg_cfg.ssim_lambda
    ) * ssim_term

    total = rgb_loss
    items["rgb_l1"] = l1.detach()
    items["rgb_ssim"] = ssim_term.detach()
    items["rgb"] = rgb_loss.detach()

    # Sky supervision: encourage transparency on sky pixels.
    sky_loss = torch.tensor(0.0, device=device)
    if float(reg_cfg.sky_loss_weight) > 0.0 and isinstance(sky_mask, torch.Tensor):
        sky_pixels = sky_mask.bool()
        if isinstance(dynamic_mask, torch.Tensor):
            sky_pixels = sky_pixels & (~dynamic_mask.bool())
        if sky_pixels.any():
            acc = alphas[..., 0].clamp(min=1e-6, max=1.0 - 1e-6)  # [B,H,W]
            sky_loss = (-torch.log1p(-acc))[sky_pixels].mean()
            total = total + float(reg_cfg.sky_loss_weight) * sky_loss
        items["sky"] = sky_loss.detach()

    # Depth prior supervision.
    depth_loss = torch.tensor(0.0, device=device)
    if do_depth_reg:
        if not isinstance(depth_prior, torch.Tensor) or expected_depth is None:
            raise RuntimeError(
                "do_depth_reg=True requires both depth_prior and expected_depth."
            )
        valid_depth = valid_color & (alphas[..., 0] > 1e-6)
        depth_loss = expected_depth_l1_loss(
            expected_depth=expected_depth,
            gt_depth=depth_prior,
            valid_mask=valid_depth,
        )
        total = total + float(reg_cfg.depth_loss_weight) * depth_loss
        items["depth"] = depth_loss.detach()

    # Normal-related terms share the same visibility mask.
    valid_norm = valid_color & (alphas[..., 0] > 1e-6)

    # Rendered normal supervision.
    render_normal_loss = torch.tensor(0.0, device=device)
    if do_render_normal_reg:
        if not isinstance(normal_prior, torch.Tensor) or render_normals is None:
            raise RuntimeError(
                "do_render_normal_reg=True requires both normal_prior and render_normals."
            )
        render_normal_loss = cosine_normal_loss(
            pred_normals=render_normals,
            gt_normals=normal_prior,
            valid_mask=valid_norm,
        )
        total = total + float(reg_cfg.normal_loss_weight) * render_normal_loss
        items["render_normal"] = render_normal_loss.detach()

    # Compute depth-implied normals only when needed by downstream terms.
    surf_normals = None
    if do_surf_normal_reg or do_consistency_normal_reg:
        if expected_depth is None:
            raise RuntimeError(
                "do_surf_normal_reg/do_consistency_normal_reg=True requires expected_depth."
            )
        surf_normals = get_implied_normal_from_depth(expected_depth, Ks)

    # Depth-implied normal supervision.
    surf_normal_loss = torch.tensor(0.0, device=device)
    if do_surf_normal_reg:
        if not isinstance(normal_prior, torch.Tensor):
            raise RuntimeError("do_surf_normal_reg=True requires normal_prior.")
        if surf_normals is None:
            raise RuntimeError("do_surf_normal_reg=True requires surf_normals.")
        surf_normal_loss = cosine_normal_loss(
            pred_normals=surf_normals,
            gt_normals=normal_prior,
            valid_mask=valid_norm,
        )
        total = total + float(reg_cfg.surf_normal_loss_weight) * surf_normal_loss
        items["surf_normal"] = surf_normal_loss.detach()

    # Normal consistency: rendered normals should match depth-implied normals.
    consistency_normal_loss = torch.tensor(0.0, device=device)
    if do_consistency_normal_reg:
        if render_normals is None or surf_normals is None:
            raise RuntimeError(
                "do_consistency_normal_reg=True requires both render_normals and surf_normals."
            )
        consistency_normal_loss = cosine_normal_loss(
            pred_normals=render_normals,
            gt_normals=surf_normals,
            valid_mask=valid_norm,
        )
        total = (
            total
            + float(reg_cfg.consistency_normal_loss_weight) * consistency_normal_loss
        )
        items["consistency_normal"] = consistency_normal_loss.detach()

    # Scale regularizations.
    # `scale_reg` is used to prevent Gaussians from becoming overly elongated.
    flat_reg = torch.tensor(0.0, device=device)
    if do_flat_reg:
        flat_reg = flatness_loss_from_log_scales(gaussian_model.log_scales)
        total = total + float(reg_cfg.flat_reg_weight) * flat_reg
        items["flat_reg"] = flat_reg.detach()

    scale_reg = torch.tensor(0.0, device=device)
    if do_scale_reg:
        scale_reg = scale_ratio_regularization_from_log_scales(
            gaussian_model.log_scales,
            max_gauss_ratio=float(reg_cfg.max_gauss_ratio),
        )
        total = total + float(reg_cfg.scale_reg_weight) * scale_reg
        items["scale_reg"] = scale_reg.detach()

    items["total"] = total.detach()
    return LossOutput(total=total, items=items)
