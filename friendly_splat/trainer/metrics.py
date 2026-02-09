from __future__ import annotations

import warnings
from typing import Literal

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


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
        # img/ref: [H, W, 3]
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
                    # Fallback to identity map for this channel.
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


class PSNRMetric:
    """Torchmetrics PSNR wrapper for BHWC float images in [0, 1]."""

    def __init__(self, *, device: torch.device, data_range: float = 1.0) -> None:
        self._module: torch.nn.Module = (
            PeakSignalNoiseRatio(data_range=float(data_range)).to(device).eval()
        )
        for param in self._module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
        _validate_bhwc_pair(pred_rgb, gt_rgb, name="PSNR")
        return self._module(_to_bchw(pred_rgb), _to_bchw(gt_rgb))


class SSIMMetric:
    """Torchmetrics SSIM wrapper for BHWC float images in [0, 1]."""

    def __init__(self, *, device: torch.device, data_range: float = 1.0) -> None:
        self._module: torch.nn.Module = (
            StructuralSimilarityIndexMeasure(data_range=float(data_range))
            .to(device)
            .eval()
        )
        for param in self._module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
        _validate_bhwc_pair(pred_rgb, gt_rgb, name="SSIM")
        return self._module(_to_bchw(pred_rgb), _to_bchw(gt_rgb))


class LPIPSMetric:
    """Torchmetrics LPIPS wrapper aligned with Extended-GS settings."""

    def __init__(
        self, *, device: torch.device, net: Literal["alex", "vgg"] = "alex"
    ) -> None:
        if str(net) not in ("alex", "vgg"):
            raise ValueError(f"LPIPS net must be 'alex' or 'vgg', got {net!r}")
        # Extended-GS uses normalize=True for alex and normalize=False for vgg.
        normalize = bool(net == "alex")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"The parameter 'pretrained' is deprecated since 0\.13.*",
                category=UserWarning,
                module=r"torchvision\.models\._utils",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"Arguments other than a weight enum or `None` for 'weights'.*",
                category=UserWarning,
                module=r"torchvision\.models\._utils",
            )
            warnings.filterwarnings(
                "ignore",
                message=r"You are using `torch\.load` with `weights_only=False`.*",
                category=FutureWarning,
                module=r"torchmetrics\.functional\.image\.lpips",
            )
            self._module: torch.nn.Module = (
                LearnedPerceptualImagePatchSimilarity(
                    net_type=str(net), normalize=normalize
                )
                .to(device)
                .eval()
            )
        for param in self._module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> torch.Tensor:
        _validate_bhwc_pair(pred_rgb, gt_rgb, name="LPIPS")
        return self._module(_to_bchw(pred_rgb), _to_bchw(gt_rgb))


__all__ = [
    "color_correct",
    "LPIPSMetric",
    "PSNRMetric",
    "SSIMMetric",
]
