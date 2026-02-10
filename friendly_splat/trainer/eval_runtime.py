from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch

from friendly_splat.data.dataloader import DataLoader, PreparedBatch
from friendly_splat.models.postprocess import PostProcessor, apply_postprocess
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
    return (int(train_step) % int(eval_cfg.eval_every_n)) == 0


def _active_sh_degree_for_step(*, step: int, optim_cfg: OptimConfig) -> int:
    max_sh_degree = int(optim_cfg.sh_degree)
    if int(optim_cfg.sh_degree_interval) > 0:
        return min(max_sh_degree, int(step) // int(optim_cfg.sh_degree_interval))
    return max_sh_degree


def _slice_batch(batch: PreparedBatch, n: int) -> PreparedBatch:
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


@torch.no_grad()
def run_evaluation(
    *,
    cfg: TrainConfig,
    step: int,
    eval_loader: DataLoader,
    splats: torch.nn.ParameterDict,
    postprocessor: Optional[PostProcessor] = None,
) -> EvalOutput:
    max_images = cfg.eval.max_images
    active_sh_degree = _active_sh_degree_for_step(step=int(step), optim_cfg=cfg.optim)
    eval_metrics = _get_eval_metrics(
        device=splats["means"].device,
        lpips_net=str(cfg.eval.lpips_net),
    )

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_cc_psnr = 0.0
    total_cc_ssim = 0.0
    total_cc_lpips = 0.0
    total_images = 0
    compute_cc_metrics = bool(cfg.eval.compute_cc_metrics) and (
        postprocessor is not None
    )

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
            splats=splats,
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

        pred_rgb = apply_postprocess(
            pred_rgb=pred_rgb,
            image_ids=prepared_batch.image_ids,
            postprocessor=postprocessor,
        )

        pred_rgb = pred_rgb.clamp(0.0, 1.0)
        target_rgb = prepared_batch.pixels

        batch_psnr = float(eval_metrics.psnr(pred_rgb, target_rgb).item())
        batch_ssim = float(eval_metrics.ssim(pred_rgb, target_rgb).item())
        batch_lpips = float(eval_metrics.lpips(pred_rgb, target_rgb).item())
        batch_cc_psnr = 0.0
        batch_cc_ssim = 0.0
        batch_cc_lpips = 0.0
        if compute_cc_metrics:
            cc_pred_rgb = color_correct(pred_rgb, target_rgb)
            batch_cc_psnr = float(eval_metrics.psnr(cc_pred_rgb, target_rgb).item())
            batch_cc_ssim = float(eval_metrics.ssim(cc_pred_rgb, target_rgb).item())
            batch_cc_lpips = float(eval_metrics.lpips(cc_pred_rgb, target_rgb).item())

        total_psnr += batch_psnr * float(batch_size)
        total_ssim += batch_ssim * float(batch_size)
        total_lpips += batch_lpips * float(batch_size)
        total_cc_psnr += batch_cc_psnr * float(batch_size)
        total_cc_ssim += batch_cc_ssim * float(batch_size)
        total_cc_lpips += batch_cc_lpips * float(batch_size)
        total_images += int(batch_size)

    elapsed = max(time.time() - tic, 1e-10)

    if total_images <= 0:
        raise RuntimeError(
            "Evaluation produced zero images. Check eval split and eval.max_images."
        )

    stats: Dict[str, float | int] = {
        "step": int(step),
        "train_step": int(step) + 1,
        "psnr": float(total_psnr / float(total_images)),
        "ssim": float(total_ssim / float(total_images)),
        "lpips": float(total_lpips / float(total_images)),
        "seconds_per_image": float(elapsed / float(total_images)),
        "num_eval_images": int(total_images),
        "num_gaussians": int(splats["means"].shape[0]),
        "active_sh_degree": int(active_sh_degree),
    }
    if compute_cc_metrics:
        stats["cc_psnr"] = float(total_cc_psnr / float(total_images))
        stats["cc_ssim"] = float(total_cc_ssim / float(total_images))
        stats["cc_lpips"] = float(total_cc_lpips / float(total_images))
    return EvalOutput(stats=stats)
