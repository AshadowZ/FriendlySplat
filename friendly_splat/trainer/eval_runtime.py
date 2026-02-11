from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch

from friendly_splat.data.dataloader import DataLoader, PreparedBatch
from friendly_splat.modules.bilateral_grid import BilateralGridPostProcessor
from friendly_splat.modules.gaussian import GaussianModel
from friendly_splat.renderer.renderer import render_splats
from friendly_splat.trainer.configs import (
    EvalConfig,
    OptimConfig,
    TrainConfig,
)
from friendly_splat.trainer.metrics import (
    LPIPSMetric,
    PSNRMetric,
    SSIMMetric,
    color_correct,
)


@dataclass(frozen=True)
class EvalMetricBundle:
    psnr: PSNRMetric
    ssim: SSIMMetric
    lpips: LPIPSMetric


_EVAL_METRICS: Dict[Tuple[str, str], EvalMetricBundle] = {}


def _get_eval_metrics(*, device: torch.device, lpips_net: str) -> EvalMetricBundle:
    key = (str(device), str(lpips_net))
    metrics = _EVAL_METRICS.get(key)
    if metrics is None:
        metrics = EvalMetricBundle(
            psnr=PSNRMetric(device=device),
            ssim=SSIMMetric(device=device),
            lpips=LPIPSMetric(device=device, net=lpips_net),
        )
        _EVAL_METRICS[key] = metrics
    return metrics


@dataclass(frozen=True)
class EvalOutput:
    stats: Dict[str, float | int]


def build_eval_summary(*, eval_step: int, stats: Mapping[str, object]) -> str:
    lpips_suffix = f" lpips={float(stats['lpips']):.4f}"
    cc_suffix = ""
    cc_psnr = stats.get("cc_psnr")
    cc_ssim = stats.get("cc_ssim")
    cc_lpips = stats.get("cc_lpips")
    if cc_psnr is not None and cc_ssim is not None and cc_lpips is not None:
        cc_suffix = (
            f" cc_psnr={float(cc_psnr):.3f}"
            f" cc_ssim={float(cc_ssim):.4f}"
            f" cc_lpips={float(cc_lpips):.4f}"
        )
    return (
        "eval "
        f"step={int(eval_step) + 1} "
        f"psnr={float(stats['psnr']):.3f} "
        f"ssim={float(stats['ssim']):.4f}"
        f"{lpips_suffix} "
        f"{cc_suffix} "
        f"sec/img={float(stats['seconds_per_image']):.4f}"
    )


def should_run_evaluation(*, eval_cfg: EvalConfig, step: int) -> bool:
    if not bool(eval_cfg.enable):
        return False
    train_step = int(step) + 1
    # Eval cadence uses 1-based step numbers for user-facing logs/settings.
    return (int(train_step) % int(eval_cfg.eval_every_n)) == 0


def _active_sh_degree_for_step(*, step: int, optim_cfg: OptimConfig) -> int:
    max_sh_degree = int(optim_cfg.sh_degree)
    if int(optim_cfg.sh_degree_interval) > 0:
        # Progressive SH: start from degree 0 and grow until max_sh_degree.
        return min(max_sh_degree, int(step) // int(optim_cfg.sh_degree_interval))
    return max_sh_degree


def _slice_batch(batch: PreparedBatch, n: int) -> PreparedBatch:
    # Used only when eval.max_images cuts through a batch.
    return PreparedBatch(
        pixels=batch.pixels[:n],
        camtoworlds=batch.camtoworlds[:n],
        camtoworlds_input=batch.camtoworlds_input[:n],
        Ks=batch.Ks[:n],
        height=int(batch.height),
        width=int(batch.width),
        image_ids=batch.image_ids[:n]
        if isinstance(batch.image_ids, torch.Tensor)
        else None,
        depth_prior=batch.depth_prior[:n]
        if isinstance(batch.depth_prior, torch.Tensor)
        else None,
        normal_prior=batch.normal_prior[:n]
        if isinstance(batch.normal_prior, torch.Tensor)
        else None,
        dynamic_mask=batch.dynamic_mask[:n]
        if isinstance(batch.dynamic_mask, torch.Tensor)
        else None,
        sky_mask=batch.sky_mask[:n]
        if isinstance(batch.sky_mask, torch.Tensor)
        else None,
    )


@torch.inference_mode()
def run_evaluation(
    *,
    cfg: TrainConfig,
    step: int,
    eval_loader: DataLoader,
    gaussian_model: GaussianModel,
    bilateral_grid: Optional[BilateralGridPostProcessor] = None,
) -> EvalOutput:
    max_images = cfg.eval.max_images
    active_sh_degree = _active_sh_degree_for_step(step=int(step), optim_cfg=cfg.optim)
    eval_metrics = _get_eval_metrics(
        device=gaussian_model.device,
        lpips_net=str(cfg.eval.lpips_net),
    )

    # Keep accumulators on device to avoid per-batch `.item()` sync stalls.
    device = gaussian_model.device
    total_psnr = torch.zeros((), device=device)
    total_ssim = torch.zeros((), device=device)
    total_lpips = torch.zeros((), device=device)
    total_cc_psnr = torch.zeros((), device=device)
    total_cc_ssim = torch.zeros((), device=device)
    total_cc_lpips = torch.zeros((), device=device)
    total_images = 0
    compute_cc_metrics = bool(cfg.eval.compute_cc_metrics) and (
        bilateral_grid is not None
    )

    # Synchronize around the timer for accurate seconds/image measurement.
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    tic = time.time()

    for prepared_batch in eval_loader.iter_once():
        if max_images is not None and total_images >= int(max_images):
            break

        batch_size = int(prepared_batch.pixels.shape[0])
        if max_images is not None:
            remaining = int(max_images) - int(total_images)
            if remaining <= 0:
                break
            if batch_size > remaining:
                prepared_batch = _slice_batch(prepared_batch, remaining)
                batch_size = remaining

        out = render_splats(
            gaussian_model=gaussian_model,
            camtoworlds=prepared_batch.camtoworlds,
            Ks=prepared_batch.Ks,
            width=int(prepared_batch.width),
            height=int(prepared_batch.height),
            sh_degree=int(active_sh_degree),
            render_mode="RGB",
            absgrad=bool(cfg.strategy.absgrad),
            packed=bool(cfg.optim.packed),
            sparse_grad=bool(cfg.optim.sparse_grad),
            rasterize_mode="antialiased" if bool(cfg.optim.antialiased) else "classic",
        )
        pred_rgb = out.pred_rgb

        if bilateral_grid is not None:
            image_ids = prepared_batch.image_ids
            if image_ids is None:
                # Bilateral grid is per-frame; we need frame IDs to pick the right slice.
                raise KeyError("Bilateral grid requires `image_id` in the batch.")
            pred_rgb = bilateral_grid.apply(rgb=pred_rgb, image_ids=image_ids)

        # Bilateral grid can output out-of-range values.
        pred_rgb = pred_rgb.clamp(0.0, 1.0)
        target_rgb = prepared_batch.pixels

        weight = float(batch_size)
        total_psnr += eval_metrics.psnr(pred_rgb, target_rgb) * weight
        total_ssim += eval_metrics.ssim(pred_rgb, target_rgb) * weight
        total_lpips += eval_metrics.lpips(pred_rgb, target_rgb) * weight
        if compute_cc_metrics:
            # Color-corrected metrics: solve a per-image photometric warp (heavier).
            cc_pred_rgb = color_correct(pred_rgb, target_rgb)
            total_cc_psnr += eval_metrics.psnr(cc_pred_rgb, target_rgb) * weight
            total_cc_ssim += eval_metrics.ssim(cc_pred_rgb, target_rgb) * weight
            total_cc_lpips += eval_metrics.lpips(cc_pred_rgb, target_rgb) * weight
        total_images += int(batch_size)

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elapsed = max(time.time() - tic, 1e-10)

    if total_images <= 0:
        raise RuntimeError(
            "Evaluation produced zero images. Check eval split and eval.max_images."
        )

    stats: Dict[str, float | int] = {
        "step": int(step),
        "train_step": int(step) + 1,
        "psnr": float((total_psnr / float(total_images)).item()),
        "ssim": float((total_ssim / float(total_images)).item()),
        "lpips": float((total_lpips / float(total_images)).item()),
        "seconds_per_image": float(elapsed / float(total_images)),
        "num_eval_images": int(total_images),
        "num_gaussians": int(gaussian_model.num_gaussians),
        "active_sh_degree": int(active_sh_degree),
    }
    if compute_cc_metrics:
        stats["cc_psnr"] = float((total_cc_psnr / float(total_images)).item())
        stats["cc_ssim"] = float((total_cc_ssim / float(total_images)).item())
        stats["cc_lpips"] = float((total_cc_lpips / float(total_images)).item())
    return EvalOutput(stats=stats)
