from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Mapping, Optional, Tuple

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
from friendly_splat.utils.metrics import (
    color_correct,
    create_inria_ssim_window,
    psnr_inria_bchw,
    ssim_inria_bchw,
)


@dataclass(frozen=True)
class EvalMetricBundle:
    # All callables expect RGB tensors in BCHW.
    psnr_bchw: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ssim_bchw: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    lpips_bchw: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


_EVAL_METRICS: Dict[Tuple[str, str, str], EvalMetricBundle] = {}


def _to_bchw(rgb_bhwc: torch.Tensor) -> torch.Tensor:
    return rgb_bhwc.permute(0, 3, 1, 2).contiguous()


def _get_eval_metrics(
    *, device: torch.device, lpips_net: str, backend: str
) -> EvalMetricBundle:
    key = (str(device), str(lpips_net), str(backend))
    metrics = _EVAL_METRICS.get(key)
    if metrics is None:
        backend_s = str(backend)
        if backend_s == "gsplat":
            from torchmetrics.image import (
                PeakSignalNoiseRatio,
                StructuralSimilarityIndexMeasure,
            )
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

            psnr_module: torch.nn.Module = (
                PeakSignalNoiseRatio(data_range=1.0).to(device).eval()
            )
            for param in psnr_module.parameters():
                param.requires_grad_(False)

            # SSIM: torchmetrics implementation.
            ssim_module: torch.nn.Module = (
                StructuralSimilarityIndexMeasure(data_range=1.0).to(device).eval()
            )
            for param in ssim_module.parameters():
                param.requires_grad_(False)

            # LPIPS: torchmetrics wrapper; keep common 3DGS convention.
            if str(lpips_net) not in ("alex", "vgg"):
                raise ValueError(
                    f"lpips_net must be 'alex' or 'vgg', got {lpips_net!r}"
                )
            normalize = bool(str(lpips_net) == "alex")
            import warnings

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
                lpips_module: torch.nn.Module = (
                    LearnedPerceptualImagePatchSimilarity(
                        net_type=str(lpips_net), normalize=normalize
                    )
                    .to(device)
                    .eval()
                )
            for param in lpips_module.parameters():
                param.requires_grad_(False)

            psnr_bchw = psnr_module
            ssim_bchw = ssim_module
            lpips_bchw = lpips_module
        elif backend_s == "inria":
            from friendly_splat.utils.lpipsPyTorch.modules.lpips import LPIPS

            # Inria evaluation scripts typically use vgg LPIPS.
            lpips_module = (
                LPIPS(net_type=str(lpips_net), version="0.1").to(device).eval()
            )
            for param in lpips_module.parameters():
                param.requires_grad_(False)

            window_size = 11
            base_window = create_inria_ssim_window(
                window_size=window_size, channel=3
            ).to(device=device)

            psnr_bchw = partial(psnr_inria_bchw, data_range=1.0)

            def ssim_bchw(
                pred_rgb_bchw: torch.Tensor, gt_rgb_bchw: torch.Tensor
            ) -> torch.Tensor:
                return ssim_inria_bchw(
                    pred_rgb_bchw,
                    gt_rgb_bchw,
                    window=base_window.to(dtype=pred_rgb_bchw.dtype),
                    window_size=window_size,
                )

            def lpips_bchw(
                pred_rgb_bchw: torch.Tensor, gt_rgb_bchw: torch.Tensor
            ) -> torch.Tensor:
                return lpips_module(pred_rgb_bchw, gt_rgb_bchw).mean()

        else:
            raise ValueError(
                f"Unknown metrics_backend {backend_s!r} (expected 'gsplat' or 'inria')."
            )

        metrics = EvalMetricBundle(
            psnr_bchw=psnr_bchw,
            ssim_bchw=ssim_bchw,
            lpips_bchw=lpips_bchw,
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
        backend=str(cfg.eval.metrics_backend),
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

        pred_rgb_bchw = _to_bchw(pred_rgb)
        target_rgb_bchw = _to_bchw(target_rgb)

        weight = float(batch_size)
        total_psnr += eval_metrics.psnr_bchw(pred_rgb_bchw, target_rgb_bchw) * weight
        total_ssim += eval_metrics.ssim_bchw(pred_rgb_bchw, target_rgb_bchw) * weight
        total_lpips += eval_metrics.lpips_bchw(pred_rgb_bchw, target_rgb_bchw) * weight
        if compute_cc_metrics:
            # Color-corrected metrics: solve a per-image photometric warp (heavier).
            cc_pred_rgb = color_correct(pred_rgb, target_rgb)
            cc_pred_rgb_bchw = _to_bchw(cc_pred_rgb)
            total_cc_psnr += (
                eval_metrics.psnr_bchw(cc_pred_rgb_bchw, target_rgb_bchw) * weight
            )
            total_cc_ssim += (
                eval_metrics.ssim_bchw(cc_pred_rgb_bchw, target_rgb_bchw) * weight
            )
            total_cc_lpips += (
                eval_metrics.lpips_bchw(cc_pred_rgb_bchw, target_rgb_bchw) * weight
            )
        total_images += int(batch_size)

    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elapsed = max(time.time() - tic, 1e-10)

    if total_images <= 0:
        raise RuntimeError(
            "Evaluation produced zero images. Check eval split and eval.max_images."
        )

    backend = str(cfg.eval.metrics_backend)
    psnr_v = float((total_psnr / float(total_images)).item())
    ssim_v = float((total_ssim / float(total_images)).item())
    lpips_v = float((total_lpips / float(total_images)).item())

    stats: Dict[str, float | int] = {
        "step": int(step),
        "train_step": int(step) + 1,
        "psnr": psnr_v,
        "ssim": ssim_v,
        "lpips": lpips_v,
        f"psnr_{backend}": psnr_v,
        f"ssim_{backend}": ssim_v,
        f"lpips_{backend}": lpips_v,
        "seconds_per_image": float(elapsed / float(total_images)),
        "num_eval_images": int(total_images),
        "num_gaussians": int(gaussian_model.num_gaussians),
        "active_sh_degree": int(active_sh_degree),
    }
    if compute_cc_metrics:
        cc_psnr_v = float((total_cc_psnr / float(total_images)).item())
        cc_ssim_v = float((total_cc_ssim / float(total_images)).item())
        cc_lpips_v = float((total_cc_lpips / float(total_images)).item())
        stats["cc_psnr"] = cc_psnr_v
        stats["cc_ssim"] = cc_ssim_v
        stats["cc_lpips"] = cc_lpips_v
        stats[f"cc_psnr_{backend}"] = cc_psnr_v
        stats[f"cc_ssim_{backend}"] = cc_ssim_v
        stats[f"cc_lpips_{backend}"] = cc_lpips_v
    return EvalOutput(stats=stats)
