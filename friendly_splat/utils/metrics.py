from __future__ import annotations

from math import exp

import torch
import torch.nn.functional as F


def _validate_bhwc_pair(
    pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, *, name: str
) -> None:
    if pred_rgb.shape != gt_rgb.shape:
        raise ValueError(f"{name} shape mismatch: {pred_rgb.shape} vs {gt_rgb.shape}")
    if pred_rgb.dim() != 4 or pred_rgb.shape[-1] != 3:
        raise ValueError(f"{name} expects [B,H,W,3], got {pred_rgb.shape}")


def _to_bchw(rgb: torch.Tensor) -> torch.Tensor:
    return rgb.permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def psnr(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    module: torch.nn.Module | None = None,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute PSNR for BHWC float images in [0, data_range].

    If `module` is provided, it is used as the implementation backend (expects BCHW).
    Otherwise, this falls back to the common closed-form formula.

    Returns:
        Scalar tensor: mean PSNR over the batch.
    """
    _validate_bhwc_pair(pred_rgb, gt_rgb, name="PSNR")
    if module is not None:
        return module(_to_bchw(pred_rgb), _to_bchw(gt_rgb))
    if int(pred_rgb.shape[0]) == 0:
        return torch.tensor(0.0, device=pred_rgb.device, dtype=pred_rgb.dtype)

    mse = (pred_rgb - gt_rgb).pow(2).mean(dim=(1, 2, 3))  # [B]
    mse = mse.clamp(min=1e-12)
    data_range_t = torch.tensor(float(data_range), device=mse.device, dtype=mse.dtype)
    psnr_per_image = 20.0 * torch.log10(data_range_t) - 10.0 * torch.log10(mse)  # [B]
    return psnr_per_image.mean()


@torch.no_grad()
def ssim(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    module: torch.nn.Module,
) -> torch.Tensor:
    """Compute SSIM using a provided module.

    Args:
        pred_rgb/gt_rgb: [B,H,W,3] float images in [0,1].
        module: A callable module taking BCHW tensors and returning a scalar.
    """
    _validate_bhwc_pair(pred_rgb, gt_rgb, name="SSIM")
    return module(_to_bchw(pred_rgb), _to_bchw(gt_rgb))


@torch.no_grad()
def lpips(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    module: torch.nn.Module,
) -> torch.Tensor:
    """Compute LPIPS using a provided module.

    Args:
        pred_rgb/gt_rgb: [B,H,W,3] float images in [0,1].
        module: A callable module taking BCHW tensors and returning a scalar.
    """
    _validate_bhwc_pair(pred_rgb, gt_rgb, name="LPIPS")
    return module(_to_bchw(pred_rgb), _to_bchw(gt_rgb))


@torch.no_grad()
def color_correct(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    num_iters: int = 5,
    eps: float = 0.5 / 255.0,
) -> torch.Tensor:
    """Apply iterative color correction from prediction to reference (BHWC, [0,1])."""
    _validate_bhwc_pair(pred_rgb, gt_rgb, name="color_correct")
    if int(pred_rgb.shape[0]) == 0:
        return pred_rgb

    def _is_unclipped(x: torch.Tensor) -> torch.Tensor:
        return (x >= float(eps)) & (x <= (1.0 - float(eps)))

    def _solve_single(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        channels = int(img.shape[-1])
        img_mat = img.reshape(-1, channels)
        ref_mat = ref.reshape(-1, channels)
        mask0 = _is_unclipped(img_mat)
        num_quad_terms = (channels * (channels + 1)) // 2

        for _ in range(int(num_iters)):
            a_terms = []
            for c in range(channels):
                a_terms.append(img_mat[:, c : (c + 1)] * img_mat[:, c:])  # quadratic
            a_terms.append(img_mat)  # linear
            a_terms.append(torch.ones_like(img_mat[:, :1]))  # bias
            a_mat = torch.cat(a_terms, dim=-1)

            warp_cols = []
            for c in range(channels):
                b = ref_mat[:, c]
                mask = mask0[:, c] & _is_unclipped(img_mat[:, c]) & _is_unclipped(b)
                if not bool(mask.any()):
                    w = torch.zeros(
                        (int(a_mat.shape[1]),),
                        device=img_mat.device,
                        dtype=img_mat.dtype,
                    )
                    w[num_quad_terms + c] = 1.0
                    warp_cols.append(w)
                    continue
                ma = torch.where(mask[:, None], a_mat, torch.zeros_like(a_mat))
                mb = torch.where(mask, b, torch.zeros_like(b))
                solve_out = torch.linalg.lstsq(ma, mb, rcond=-1)
                w = (
                    solve_out.solution
                    if hasattr(solve_out, "solution")
                    else solve_out[0]
                )
                if not bool(torch.isfinite(w).all()):
                    w = torch.zeros(
                        (int(a_mat.shape[1]),),
                        device=img_mat.device,
                        dtype=img_mat.dtype,
                    )
                    w[num_quad_terms + c] = 1.0
                warp_cols.append(w)

            warp = torch.stack(warp_cols, dim=-1)  # [M, 3]
            img_mat = torch.clamp(torch.matmul(a_mat, warp), 0.0, 1.0)

        return img_mat.reshape_as(img)

    corrected = [
        _solve_single(pred_rgb[i], gt_rgb[i]) for i in range(int(pred_rgb.shape[0]))
    ]
    return torch.stack(corrected, dim=0)


# ----------------------------
# Inria-style metrics (gaussian-splatting / Improved-GS reference implementation)
# ----------------------------


def _validate_bchw_pair(
    pred_rgb_bchw: torch.Tensor, gt_rgb_bchw: torch.Tensor, *, name: str
) -> None:
    if pred_rgb_bchw.shape != gt_rgb_bchw.shape:
        raise ValueError(
            f"{name} shape mismatch: {pred_rgb_bchw.shape} vs {gt_rgb_bchw.shape}"
        )
    if pred_rgb_bchw.dim() != 4:
        raise ValueError(f"{name} expects BCHW, got {pred_rgb_bchw.shape}")


@torch.no_grad()
def psnr_inria(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute Inria-style PSNR for BHWC float images in [0, data_range]."""
    _validate_bhwc_pair(pred_rgb, gt_rgb, name="PSNR(inria)")
    return psnr_inria_bchw(
        _to_bchw(pred_rgb), _to_bchw(gt_rgb), data_range=float(data_range)
    )


@torch.no_grad()
def psnr_inria_bchw(
    pred_rgb_bchw: torch.Tensor,
    gt_rgb_bchw: torch.Tensor,
    *,
    data_range: float = 1.0,
) -> torch.Tensor:
    """Compute Inria-style PSNR for BCHW float images in [0, data_range]."""
    _validate_bchw_pair(pred_rgb_bchw, gt_rgb_bchw, name="PSNR(inria)")
    if int(pred_rgb_bchw.shape[0]) == 0:
        return torch.tensor(0.0, device=pred_rgb_bchw.device, dtype=pred_rgb_bchw.dtype)

    # Match the gaussian-splatting reference implementation:
    # mse = (((img1 - img2)) ** 2).view(B, -1).mean(1, keepdim=True)
    mse = (
        (pred_rgb_bchw - gt_rgb_bchw)
        .pow(2)
        .view(int(pred_rgb_bchw.shape[0]), -1)
        .mean(1, keepdim=True)
    )
    mse = mse.clamp(min=1e-12)
    data_range_t = torch.tensor(float(data_range), device=mse.device, dtype=mse.dtype)
    psnr_per_image = 20.0 * torch.log10(data_range_t / torch.sqrt(mse))  # [B,1]
    return psnr_per_image.mean()


def _inria_gaussian_1d(window_size: int, sigma: float) -> torch.Tensor:
    values = [
        exp(-((x - window_size // 2) ** 2) / float(2.0 * sigma**2))
        for x in range(int(window_size))
    ]
    gauss = torch.tensor(values, dtype=torch.float32)
    return gauss / gauss.sum()


def _inria_create_window(*, window_size: int, channel: int) -> torch.Tensor:
    # Shape: [C, 1, ws, ws], ready for grouped conv2d.
    w1 = _inria_gaussian_1d(window_size, 1.5).unsqueeze(1)  # [ws,1]
    w2 = (w1 @ w1.t()).float().unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]
    return w2.expand(int(channel), 1, window_size, window_size).contiguous()


def create_inria_ssim_window(
    *, window_size: int = 11, channel: int = 3
) -> torch.Tensor:
    """Create the fixed gaussian window used by Inria-style SSIM (float32, CPU)."""
    return _inria_create_window(window_size=int(window_size), channel=int(channel))


@torch.no_grad()
def ssim_inria_bchw(
    pred_rgb_bchw: torch.Tensor,
    gt_rgb_bchw: torch.Tensor,
    *,
    window: torch.Tensor,
    window_size: int,
) -> torch.Tensor:
    """Compute Inria-style SSIM for BCHW float images in [0,1]."""
    _validate_bchw_pair(pred_rgb_bchw, gt_rgb_bchw, name="SSIM(inria)")
    channel = int(pred_rgb_bchw.shape[1])
    if channel != 3:
        raise ValueError(f"SSIM(inria) expects 3 channels (RGB), got C={channel}.")

    window_size_i = int(window_size)
    mu1 = F.conv2d(pred_rgb_bchw, window, padding=window_size_i // 2, groups=channel)
    mu2 = F.conv2d(gt_rgb_bchw, window, padding=window_size_i // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            pred_rgb_bchw * pred_rgb_bchw,
            window,
            padding=window_size_i // 2,
            groups=channel,
        )
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(
            gt_rgb_bchw * gt_rgb_bchw,
            window,
            padding=window_size_i // 2,
            groups=channel,
        )
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(
            pred_rgb_bchw * gt_rgb_bchw,
            window,
            padding=window_size_i // 2,
            groups=channel,
        )
        - mu1_mu2
    )

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    return ssim_map.mean()


@torch.no_grad()
def ssim_inria(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute Inria-style SSIM for BHWC float images in [0,1]."""
    _validate_bhwc_pair(pred_rgb, gt_rgb, name="SSIM(inria)")
    pred_rgb_bchw = _to_bchw(pred_rgb)
    gt_rgb_bchw = _to_bchw(gt_rgb)
    channel = int(pred_rgb_bchw.shape[1])
    window_size_i = int(window_size)
    window = create_inria_ssim_window(window_size=window_size_i, channel=channel).to(
        device=pred_rgb_bchw.device, dtype=pred_rgb_bchw.dtype
    )
    return ssim_inria_bchw(
        pred_rgb_bchw, gt_rgb_bchw, window=window, window_size=window_size_i
    )


@torch.no_grad()
def lpips_inria(
    pred_rgb: torch.Tensor,
    gt_rgb: torch.Tensor,
    *,
    module: torch.nn.Module,
) -> torch.Tensor:
    """Compute Inria-style LPIPS for BHWC float images in [0,1]."""
    _validate_bhwc_pair(pred_rgb, gt_rgb, name="LPIPS(inria)")
    return module(_to_bchw(pred_rgb), _to_bchw(gt_rgb)).mean()


__all__ = [
    "color_correct",
    "lpips",
    "lpips_inria",
    "psnr",
    "psnr_inria",
    "psnr_inria_bchw",
    "ssim",
    "ssim_inria",
    "ssim_inria_bchw",
    "create_inria_ssim_window",
]
