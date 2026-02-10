from __future__ import annotations

from typing import Any, Callable, Collection, Dict, Mapping, Optional

import torch

from friendly_splat.data.dataloader import DataLoader, PreparedBatch
from friendly_splat.models.bilateral_grid import BilateralGridPostProcessor
from friendly_splat.models.camera_opt import CameraOptModule, apply_pose_adjust
from friendly_splat.models.ppisp import PPISPPostProcessor
from friendly_splat.renderer.renderer import RenderOutput, render_splats
from friendly_splat.trainer.configs import (
    OptimConfig,
    PostprocessConfig,
    RegConfig,
    TrainConfig,
)
from friendly_splat.trainer.eval_runtime import run_evaluation, should_run_evaluation
from friendly_splat.trainer.io_utils import save_eval_stats
from friendly_splat.trainer.losses import LossOutput, compute_losses
from friendly_splat.trainer.step_schedule import StepSchedule, compute_step_schedule
from gsplat.strategy.natural_selection import NaturalSelectionPolicy


def postprocess_enabled(postprocess_cfg: PostprocessConfig) -> bool:
    return bool(postprocess_cfg.use_bilateral_grid) or bool(postprocess_cfg.use_ppisp)


def apply_postprocess(
    *,
    pred_rgb: torch.Tensor,
    image_ids: Optional[torch.Tensor],
    postprocess_cfg: PostprocessConfig,
    bilagrid: Optional[BilateralGridPostProcessor] = None,
    ppisp: Optional[PPISPPostProcessor] = None,
) -> torch.Tensor:
    if image_ids is None:
        raise KeyError("Postprocess requires `image_id` in the batch.")

    out_rgb = pred_rgb
    if postprocess_cfg.use_bilateral_grid:
        if bilagrid is None:
            raise RuntimeError(
                "use_bilateral_grid=True but bilagrid is not initialized."
            )
        out_rgb = bilagrid.apply(rgb=out_rgb, image_ids=image_ids)
    elif postprocess_cfg.use_ppisp:
        if ppisp is None:
            raise RuntimeError("use_ppisp=True but ppisp is not initialized.")
        out_rgb = ppisp.apply(rgb=out_rgb, image_ids=image_ids)
    return out_rgb


def build_step_schedule_from_prepared_batch(
    *,
    step: int,
    optim_cfg: OptimConfig,
    reg_cfg: RegConfig,
    prepared_batch: PreparedBatch,
) -> StepSchedule:
    has_depth_prior = (
        isinstance(prepared_batch.depth_prior, torch.Tensor)
        and prepared_batch.depth_prior.numel() > 0
    )
    has_normal_prior = (
        isinstance(prepared_batch.normal_prior, torch.Tensor)
        and prepared_batch.normal_prior.numel() > 0
    )
    return compute_step_schedule(
        step=step,
        optim_cfg=optim_cfg,
        reg_cfg=reg_cfg,
        has_depth_prior=has_depth_prior,
        has_normal_prior=has_normal_prior,
    )


def prepare_training_batch(
    *,
    prepared_batch: PreparedBatch,
    pose_opt: bool,
    pose_adjust: Optional[CameraOptModule],
) -> PreparedBatch:
    camtoworlds, camtoworlds_input = apply_pose_adjust(
        camtoworlds=prepared_batch.camtoworlds,
        image_ids=prepared_batch.image_ids,
        pose_opt=bool(pose_opt),
        pose_adjust=pose_adjust,
    )
    return PreparedBatch(
        pixels=prepared_batch.pixels,
        camtoworlds=camtoworlds,
        camtoworlds_input=camtoworlds_input,
        Ks=prepared_batch.Ks,
        height=prepared_batch.height,
        width=prepared_batch.width,
        image_ids=prepared_batch.image_ids,
        depth_prior=prepared_batch.depth_prior,
        normal_prior=prepared_batch.normal_prior,
        dynamic_mask=prepared_batch.dynamic_mask,
        sky_mask=prepared_batch.sky_mask,
    )


def render_from_prepared_batch(
    *,
    prepared_batch: PreparedBatch,
    splats: torch.nn.ParameterDict,
    optim_cfg: OptimConfig,
    postprocess_cfg: PostprocessConfig,
    schedule: StepSchedule,
    absgrad: bool = False,
    bilagrid: Optional[BilateralGridPostProcessor] = None,
    ppisp: Optional[PPISPPostProcessor] = None,
) -> RenderOutput:
    out = render_splats(
        splats=splats,
        camtoworlds=prepared_batch.camtoworlds,
        Ks=prepared_batch.Ks,
        width=int(prepared_batch.width),
        height=int(prepared_batch.height),
        sh_degree=int(schedule.active_sh_degree),
        render_mode=schedule.render_mode,
        absgrad=absgrad,
        packed=bool(optim_cfg.packed),
        sparse_grad=bool(optim_cfg.sparse_grad),
        rasterize_mode="antialiased" if bool(optim_cfg.antialiased) else "classic",
    )
    pred_rgb = out.pred_rgb
    alphas = out.alphas
    image_ids = prepared_batch.image_ids

    if postprocess_enabled(postprocess_cfg):
        pred_rgb = apply_postprocess(
            pred_rgb=pred_rgb,
            image_ids=image_ids,
            postprocess_cfg=postprocess_cfg,
            bilagrid=bilagrid,
            ppisp=ppisp,
        )

    if optim_cfg.random_bkgd:
        bkgd = torch.rand((pred_rgb.shape[0], 3), device=pred_rgb.device)
        pred_rgb = pred_rgb + bkgd[:, None, None, :] * (1.0 - alphas)

    return RenderOutput(
        pred_rgb=pred_rgb,
        alphas=out.alphas,
        meta=out.meta,
        expected_depth=out.expected_depth,
        render_normals=out.render_normals,
        active_sh_degree=out.active_sh_degree,
    )


def compute_losses_from_prepared_batch_and_render(
    *,
    reg_cfg: RegConfig,
    postprocess_cfg: PostprocessConfig,
    schedule: StepSchedule,
    step: int,
    prepared_batch: PreparedBatch,
    render_out: RenderOutput,
    splats: torch.nn.ParameterDict,
    bilagrid: Optional[BilateralGridPostProcessor] = None,
    ppisp: Optional[PPISPPostProcessor] = None,
    gns: Optional[NaturalSelectionPolicy] = None,
) -> LossOutput:
    do_depth_reg = bool(schedule.do_depth_reg)
    do_render_normal_reg = bool(schedule.do_render_normal_reg)
    do_surf_normal_reg = bool(schedule.do_surf_normal_reg)
    do_consistency_normal_reg = bool(schedule.do_consistency_normal_reg)
    do_flat_reg = bool(schedule.do_flat_reg)
    do_scale_reg = bool(schedule.do_scale_reg)

    base = compute_losses(
        reg_cfg=reg_cfg,
        do_depth_reg=do_depth_reg,
        do_render_normal_reg=do_render_normal_reg,
        do_surf_normal_reg=do_surf_normal_reg,
        do_consistency_normal_reg=do_consistency_normal_reg,
        do_flat_reg=do_flat_reg,
        do_scale_reg=do_scale_reg,
        pixels=prepared_batch.pixels,
        pred_rgb=render_out.pred_rgb,
        alphas=render_out.alphas,
        expected_depth=render_out.expected_depth,
        render_normals=render_out.render_normals,
        depth_prior=prepared_batch.depth_prior,
        normal_prior=prepared_batch.normal_prior,
        dynamic_mask=prepared_batch.dynamic_mask,
        sky_mask=prepared_batch.sky_mask,
        splats=splats,
        Ks=prepared_batch.Ks,
    )

    device = base.total.device
    total = base.total
    items = dict(base.items)

    # Postprocess-specific regularizers.
    bilagrid_tv = torch.tensor(0.0, device=device)
    if postprocess_cfg.use_bilateral_grid:
        if bilagrid is None:
            raise RuntimeError(
                "use_bilateral_grid=True but bilagrid is not initialized."
            )
        image_ids = prepared_batch.image_ids
        if image_ids is None:
            raise KeyError("Bilateral grid loss requires `image_id` in the batch.")
        bilagrid_tv = bilagrid.tv_loss(image_ids=image_ids)
        total = total + float(postprocess_cfg.bilateral_grid_tv_weight) * bilagrid_tv
    items["bilagrid_tv"] = bilagrid_tv.detach()

    ppisp_reg = torch.tensor(0.0, device=device)
    if postprocess_cfg.use_ppisp:
        if ppisp is None:
            raise RuntimeError("use_ppisp=True but ppisp is not initialized.")
        ppisp_reg = ppisp.reg_loss()
        total = total + float(postprocess_cfg.ppisp_reg_weight) * ppisp_reg
    items["ppisp_reg"] = ppisp_reg.detach()

    # Optional GNS regularizer.
    # Active only during the configured GNS pruning window.
    # It pushes opacities down over time so low-contribution Gaussians can be pruned.
    gns_reg = torch.tensor(0.0, device=device)
    if gns is not None:
        reg = gns.compute_regularizer(step=step, params=splats)
        if reg is not None:
            gns_reg = reg
            total = total + gns_reg
    items["gns"] = gns_reg.detach()

    items["total"] = total.detach()
    return LossOutput(total=total, items=items)


def maybe_run_evaluation_for_step(
    *,
    step: int,
    train_cfg: TrainConfig,
    eval_loader: Optional[DataLoader],
    splats: torch.nn.ParameterDict,
    bilagrid: Optional[BilateralGridPostProcessor] = None,
    ppisp: Optional[PPISPPostProcessor] = None,
    on_eval_complete: Optional[Callable[[int, Dict[str, float | int]], None]] = None,
) -> Optional[str]:
    """Run evaluation for the current step when configured and due.

    Returns a short human-readable summary string when evaluation runs, else None.
    """
    if not should_run_evaluation(eval_cfg=train_cfg.eval, step=int(step)):
        return None
    if eval_loader is None:
        raise RuntimeError("Evaluation is enabled but eval_loader is not initialized.")

    eval_output = run_evaluation(
        cfg=train_cfg,
        step=int(step),
        eval_loader=eval_loader,
        splats=splats,
        bilagrid=bilagrid,
        ppisp=ppisp,
    )
    save_eval_stats(
        io_cfg=train_cfg.io,
        eval_cfg=train_cfg.eval,
        step=int(step),
        stats=eval_output.stats,
    )
    if on_eval_complete is not None:
        on_eval_complete(int(step), eval_output.stats)
    lpips_suffix = ""
    lpips_value = eval_output.stats.get("lpips")
    if lpips_value is not None:
        lpips_suffix = f" lpips={float(lpips_value):.4f}"
    cc_suffix = ""
    cc_psnr = eval_output.stats.get("cc_psnr")
    cc_ssim = eval_output.stats.get("cc_ssim")
    cc_lpips = eval_output.stats.get("cc_lpips")
    if cc_psnr is not None and cc_ssim is not None and cc_lpips is not None:
        cc_suffix = (
            f" cc_psnr={float(cc_psnr):.3f}"
            f" cc_ssim={float(cc_ssim):.4f}"
            f" cc_lpips={float(cc_lpips):.4f}"
        )
    return (
        "eval "
        f"step={int(step) + 1} "
        f"psnr={eval_output.stats['psnr']:.3f} "
        f"ssim={eval_output.stats['ssim']:.4f}"
        f"{lpips_suffix} "
        f"{cc_suffix} "
        f"sec/img={eval_output.stats['seconds_per_image']:.4f}"
    )


def filter_train_loss_items_for_logging(
    *,
    train_cfg: TrainConfig,
    loss_items: Mapping[str, object],
) -> dict[str, object]:
    """Filter training loss items for TB/viewer logging.

    Rules:
    - Always drop: rgb_l1, rgb_ssim, gns.
    - Keep core items: total, rgb.
    - Keep optional items only when corresponding feature is enabled.
    """
    keep_keys: set[str] = {"total", "rgb"}
    reg_cfg = train_cfg.reg
    post_cfg = train_cfg.postprocess

    if float(reg_cfg.sky_loss_weight) > 0.0:
        keep_keys.add("sky")
    if float(reg_cfg.depth_loss_weight) > 0.0:
        keep_keys.add("depth")
    if float(reg_cfg.normal_loss_weight) > 0.0:
        keep_keys.add("render_normal")
    if float(reg_cfg.surf_normal_loss_weight) > 0.0:
        keep_keys.add("surf_normal")
    if float(reg_cfg.consistency_normal_loss_weight) > 0.0:
        keep_keys.add("consistency_normal")
    if float(reg_cfg.flat_reg_weight) > 0.0:
        keep_keys.add("flat_reg")
    if float(reg_cfg.scale_reg_weight) > 0.0:
        keep_keys.add("scale_reg")

    if bool(post_cfg.use_bilateral_grid):
        keep_keys.add("bilagrid_tv")
    if bool(post_cfg.use_ppisp):
        keep_keys.add("ppisp_reg")

    drop_keys = {"rgb_l1", "rgb_ssim", "gns"}

    filtered: dict[str, object] = {}
    for key, value in loss_items.items():
        if key in drop_keys:
            continue
        if key in keep_keys:
            filtered[key] = value
    return filtered


def extract_prefixed_scalar_dict(
    *,
    values: Mapping[str, object],
    prefix: str,
    exclude_keys: Collection[str] = (),
) -> dict[str, float]:
    """Extract numeric scalars and prefix names for viewer logging."""
    excluded = set(str(x) for x in exclude_keys)
    out: dict[str, float] = {}
    for key, value in values.items():
        key_s = str(key)
        if key_s in excluded:
            continue
        if isinstance(value, torch.Tensor):
            if int(value.numel()) != 1:
                continue
            out[f"{prefix}{key_s}"] = float(value.detach().item())
        elif isinstance(value, (int, float)):
            out[f"{prefix}{key_s}"] = float(value)
    return out


def build_viewer_train_scalars(
    *,
    device: torch.device,
    splats: torch.nn.ParameterDict,
    filtered_loss_items: Mapping[str, object],
    optimizer_coordinator: Any,
) -> dict[str, float]:
    """Build per-step scalar payload for viewer universal plot."""
    train_scalars = extract_prefixed_scalar_dict(
        values=filtered_loss_items,
        prefix="train/",
    )
    means_opt = optimizer_coordinator.splat_optimizers.get("means")
    if means_opt is not None and len(means_opt.param_groups) > 0:
        lr_value = means_opt.param_groups[0].get("lr")
        if isinstance(lr_value, (int, float)):
            train_scalars["train/lr_means"] = float(lr_value)
    train_scalars["train/num_gs"] = float(splats["means"].shape[0])
    if device.type == "cuda":
        train_scalars["train/mem_gb"] = float(
            torch.cuda.max_memory_allocated(device=device) / (1024**3)
        )
    return train_scalars


def handle_eval_complete_logging(
    *,
    eval_step: int,
    stats: Mapping[str, object],
    tb_runtime: Any,
    viewer_runtime: Any,
) -> None:
    """Forward eval completion outputs to TensorBoard + viewer."""
    stats_dict = dict(stats)
    tb_runtime.log_eval(
        step=int(eval_step),
        stats=stats_dict,
        stage="eval",
    )
    viewer_runtime.push_eval_metrics(
        step=int(eval_step),
        stats=stats_dict,
    )
    eval_scalars = extract_prefixed_scalar_dict(
        values=stats_dict,
        prefix="eval/",
        exclude_keys=("step", "train_step"),
    )
    eval_train_step = int(stats_dict.get("train_step", int(eval_step) + 1))
    viewer_runtime.log_scalars(
        step=int(eval_train_step),
        scalars=eval_scalars,
    )


def maybe_log_training_scalars_for_step(
    *,
    step: int,
    device: torch.device,
    splats: torch.nn.ParameterDict,
    loss_output: LossOutput,
    optimizer_coordinator: Any,
    tb_runtime: Any,
    loss_items_override: Optional[Mapping[str, object]] = None,
) -> None:
    """Collect and write training scalars to TensorBoard (if enabled)."""
    means_opt = optimizer_coordinator.splat_optimizers.get("means")
    lr_means = None
    if means_opt is not None and len(means_opt.param_groups) > 0:
        lr_value = means_opt.param_groups[0].get("lr")
        if isinstance(lr_value, (int, float)):
            lr_means = float(lr_value)

    mem_gb = None
    if bool(tb_runtime.enabled) and bool(tb_runtime.log_memory) and device.type == "cuda":
        mem_gb = float(torch.cuda.max_memory_allocated(device=device) / (1024**3))

    tb_runtime.log_train(
        step=int(step),
        loss_items=loss_items_override
        if loss_items_override is not None
        else loss_output.items,
        num_gs=int(splats["means"].shape[0]),
        lr_means=lr_means,
        mem_gb=mem_gb,
    )
