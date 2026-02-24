from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from friendly_splat.trainer.configs import OptimConfig, RegConfig


@dataclass(frozen=True)
class StepSchedule:
    """Per-step schedule for regularizations and required renderer outputs."""

    do_sky_loss: bool
    do_depth_reg: bool
    do_render_normal_reg: bool
    do_surf_normal_reg: bool
    do_consistency_normal_reg: bool
    do_flat_reg: bool
    do_scale_ratio_reg: bool
    active_sh_degree: int
    render_mode: Literal["RGB", "RGB+ED", "RGB+N+ED"]


def compute_step_schedule(
    *,
    step: int,
    optim_cfg: OptimConfig,
    reg_cfg: RegConfig,
    has_depth_prior: bool,
    has_normal_prior: bool,
    has_sky_mask: bool,
) -> StepSchedule:
    """Decide which regularizations to apply this step and which renderer outputs are required."""
    max_sh_degree = int(optim_cfg.sh_degree)
    if max_sh_degree < 0:
        raise ValueError(f"sh_degree must be >= 0, got {max_sh_degree}")

    if int(optim_cfg.sh_degree_interval) > 0:
        active_sh_degree = min(max_sh_degree, step // int(optim_cfg.sh_degree_interval))
    else:
        active_sh_degree = max_sh_degree

    depth_due = reg_cfg.depth_reg_every_n > 0 and (
        step % reg_cfg.depth_reg_every_n == 0
    )
    sky_due = reg_cfg.sky_loss_every_n > 0 and (step % reg_cfg.sky_loss_every_n == 0)

    prior_every_n = int(reg_cfg.prior_normal_reg_every_n)
    consistency_every_n = int(reg_cfg.consistency_normal_reg_every_n)

    prior_normal_due = prior_every_n > 0 and (step % prior_every_n == 0)
    consistency_normal_due = consistency_every_n > 0 and (step % consistency_every_n == 0)
    scale_ratio_due = reg_cfg.scale_ratio_reg_every_n > 0 and (
        step % reg_cfg.scale_ratio_reg_every_n == 0
    )

    do_sky_loss = has_sky_mask and reg_cfg.sky_loss_weight > 0.0 and sky_due
    do_depth_reg = (
        has_depth_prior
        and reg_cfg.depth_loss_weight > 0.0
        and step >= reg_cfg.depth_loss_activation_step
        and depth_due
    )
    do_render_normal_reg = (
        has_normal_prior
        and reg_cfg.normal_loss_weight > 0.0
        and step >= reg_cfg.normal_loss_activation_step
        and prior_normal_due
    )
    do_surf_normal_reg = (
        has_normal_prior
        and reg_cfg.surf_normal_loss_weight > 0.0
        and step >= reg_cfg.surf_normal_loss_activation_step
        and prior_normal_due
    )
    do_consistency_normal_reg = (
        reg_cfg.consistency_normal_loss_weight > 0.0
        and step >= reg_cfg.consistency_normal_loss_activation_step
        and consistency_normal_due
    )

    do_flat_reg = reg_cfg.flat_reg_weight > 0.0
    do_scale_ratio_reg = reg_cfg.scale_ratio_reg_weight > 0.0 and scale_ratio_due

    need_expected_depth = (
        do_depth_reg or do_surf_normal_reg or do_consistency_normal_reg
    )
    need_render_normals = do_render_normal_reg or do_consistency_normal_reg

    if need_render_normals:
        render_mode: Literal["RGB", "RGB+ED", "RGB+N+ED"] = "RGB+N+ED"
    elif need_expected_depth:
        render_mode = "RGB+ED"
    else:
        render_mode = "RGB"

    return StepSchedule(
        do_sky_loss=do_sky_loss,
        do_depth_reg=do_depth_reg,
        do_render_normal_reg=do_render_normal_reg,
        do_surf_normal_reg=do_surf_normal_reg,
        do_consistency_normal_reg=do_consistency_normal_reg,
        do_flat_reg=do_flat_reg,
        do_scale_ratio_reg=do_scale_ratio_reg,
        active_sh_degree=int(active_sh_degree),
        render_mode=render_mode,
    )
