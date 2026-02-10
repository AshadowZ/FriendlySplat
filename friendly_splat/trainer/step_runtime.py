from __future__ import annotations

from typing import Optional

import torch

from friendly_splat.data.dataloader import DataLoader, PreparedBatch
from friendly_splat.models.camera_opt import CameraOptModule, apply_pose_adjust
from friendly_splat.models.gaussian import GaussianModel
from friendly_splat.models.postprocess import (
    PostProcessor,
    apply_postprocess,
    get_postprocess_regularizer,
)
from friendly_splat.renderer.renderer import RenderOutput, render_splats
from friendly_splat.trainer.configs import (
    OptimConfig,
    RegConfig,
    TrainConfig,
)
from friendly_splat.trainer.eval_runtime import (
    EvalOutput,
    build_eval_summary,
    run_evaluation,
    should_run_evaluation,
)
from friendly_splat.trainer.io_utils import save_eval_stats
from friendly_splat.trainer.losses import LossOutput, compute_losses
from friendly_splat.trainer.step_schedule import StepSchedule, compute_step_schedule
from gsplat.strategy.natural_selection import NaturalSelectionPolicy


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
    gaussian_model: GaussianModel,
    optim_cfg: OptimConfig,
    schedule: StepSchedule,
    absgrad: bool = False,
    postprocessor: Optional[PostProcessor] = None,
) -> RenderOutput:
    out = render_splats(
        splats=gaussian_model.splats,
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

    pred_rgb = apply_postprocess(
        pred_rgb=pred_rgb,
        image_ids=image_ids,
        postprocessor=postprocessor,
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
    schedule: StepSchedule,
    step: int,
    prepared_batch: PreparedBatch,
    render_out: RenderOutput,
    gaussian_model: GaussianModel,
    postprocessor: Optional[PostProcessor] = None,
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
        gaussian_model=gaussian_model,
        Ks=prepared_batch.Ks,
    )

    total = base.total
    items = dict(base.items)

    # Postprocess-specific regularizers.
    regularizer = get_postprocess_regularizer(
        postprocessor=postprocessor,
        image_ids=prepared_batch.image_ids,
    )
    if regularizer is not None:
        total = total + regularizer.value
        items[regularizer.name] = regularizer.value.detach()

    # Optional GNS regularizer.
    # Active only during the configured GNS pruning window.
    # It pushes opacities down over time so low-contribution Gaussians can be pruned.
    if gns is not None:
        reg = gns.compute_regularizer(step=step, params=gaussian_model.splats)
        if reg is not None:
            total = total + reg
            items["gns"] = reg.detach()

    items["total"] = total.detach()
    return LossOutput(total=total, items=items)


def maybe_run_evaluation_for_step(
    *,
    step: int,
    train_cfg: TrainConfig,
    eval_loader: Optional[DataLoader],
    gaussian_model: GaussianModel,
    postprocessor: Optional[PostProcessor] = None,
) -> Optional[EvalOutput]:
    """Run evaluation for the current step when configured and due.

    Returns eval output when evaluation runs, else None.
    """
    if not should_run_evaluation(eval_cfg=train_cfg.eval, step=int(step)):
        return None
    if eval_loader is None:
        raise RuntimeError("Evaluation is enabled but eval_loader is not initialized.")

    eval_output = run_evaluation(
        cfg=train_cfg,
        step=int(step),
        eval_loader=eval_loader,
        gaussian_model=gaussian_model,
        postprocessor=postprocessor,
    )
    save_eval_stats(
        io_cfg=train_cfg.io,
        eval_cfg=train_cfg.eval,
        step=int(step),
        stats=eval_output.stats,
    )
    eval_step = int(eval_output.stats.get("step", int(step)))
    print(
        build_eval_summary(
            eval_step=eval_step,
            stats=eval_output.stats,
        ),
        flush=True,
    )
    return eval_output
