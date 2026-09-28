from __future__ import annotations

"""Validation for the `--fast` code path.

Kept in the FastGS folder so `trainer/configs.py` only needs a single call.
Like `presets.py`, this imports nothing from `trainer.configs`.
"""

from typing import Any


def validate_fastgs_config(cfg: Any) -> None:
    """Validate FastGS settings. No-op unless `cfg.fast` is set."""
    if not bool(getattr(cfg, "fast", False)):
        return

    fast = cfg.fastgs

    # --- incompatible switches -------------------------------------------
    if str(cfg.strategy.impl).strip().lower() == "mcmc":
        raise ValueError(
            "fast=True replaces the densification strategy and is incompatible "
            "with strategy.impl='mcmc'. Drop --fast or pick a densification "
            "strategy ('improved'/'default')."
        )
    if bool(cfg.optim.mu_enable):
        raise ValueError(
            "fast=True brings its own optimizer step schedule and is incompatible "
            "with optim.mu_enable=True. Disable one of them "
            "(--fastgs.no-optim-gate-enable keeps MU)."
        )
    if bool(cfg.optim.sparse_grad):
        raise ValueError(
            "fast=True is incompatible with optim.sparse_grad=True: the FastGS "
            "optimizer schedule accumulates dense gradients across steps."
        )
    if not bool(cfg.strategy.absgrad):
        raise ValueError(
            "fast=True splits Gaussians on absolute gradients and requires "
            "strategy.absgrad=True (enabled automatically unless "
            "--fastgs.no-apply-presets is passed)."
        )

    # --- scoring ----------------------------------------------------------
    if int(fast.score_num_views) <= 0:
        raise ValueError(
            f"fastgs.score_num_views must be > 0, got {fast.score_num_views}"
        )
    if fast.view_cache_size is not None and int(fast.view_cache_size) <= 0:
        raise ValueError(
            f"fastgs.view_cache_size must be > 0 or None, got {fast.view_cache_size}"
        )
    if not (0.0 < float(fast.loss_thresh) < 1.0):
        raise ValueError(
            "fastgs.loss_thresh must be in (0, 1) (it thresholds a min-max "
            f"normalized error map), got {fast.loss_thresh}"
        )
    if str(fast.score_backend) not in ("weighted", "tracked"):
        raise ValueError(
            "fastgs.score_backend must be 'weighted' or 'tracked', got "
            f"{fast.score_backend!r}"
        )
    if not (0.0 < float(fast.tracked_vis_thresh) < 1.0):
        raise ValueError(
            f"fastgs.tracked_vis_thresh must be in (0, 1), got {fast.tracked_vis_thresh}"
        )
    if str(fast.score_backend) == "tracked" and bool(cfg.optim.packed):
        raise ValueError(
            "fastgs.score_backend='tracked' relies on gsplat's pixel-Gaussian "
            "tracker, which requires optim.packed=False. Use "
            "score_backend='weighted' with packed rasterization."
        )

    # --- densification ----------------------------------------------------
    if int(fast.densify_every) <= 0:
        raise ValueError(f"fastgs.densify_every must be > 0, got {fast.densify_every}")
    if float(fast.grad_abs_thresh) <= 0.0:
        raise ValueError(
            f"fastgs.grad_abs_thresh must be > 0, got {fast.grad_abs_thresh}"
        )
    if float(fast.percent_dense) <= 0.0:
        raise ValueError(f"fastgs.percent_dense must be > 0, got {fast.percent_dense}")
    if fast.importance_thresh is not None and float(fast.importance_thresh) < 0.0:
        raise ValueError(
            f"fastgs.importance_thresh must be >= 0 or None, got {fast.importance_thresh}"
        )
    if not (0.0 < float(fast.prune_budget_ratio) <= 1.0):
        raise ValueError(
            f"fastgs.prune_budget_ratio must be in (0, 1], got {fast.prune_budget_ratio}"
        )
    if not (0.0 < float(fast.opacity_clamp) <= 1.0):
        raise ValueError(
            f"fastgs.opacity_clamp must be in (0, 1], got {fast.opacity_clamp}"
        )
    if float(fast.opacity_reset_multiplier) <= 1.0:
        raise ValueError(
            "fastgs.opacity_reset_multiplier must be > 1 (the reset target has to "
            f"sit above strategy.prune_opa), got {fast.opacity_reset_multiplier}"
        )
    if float(fast.opacity_reset_multiplier) * float(cfg.strategy.prune_opa) >= 1.0:
        raise ValueError(
            "fastgs.opacity_reset_multiplier * strategy.prune_opa must be < 1 "
            "(it is a post-sigmoid opacity), got "
            f"{float(fast.opacity_reset_multiplier) * float(cfg.strategy.prune_opa)}"
        )

    # --- post-densification pruning ---------------------------------------
    if bool(fast.final_prune_enable):
        # `strategy.refine_stop_iter` is 0-based; the prune window is 1-based.
        if int(fast.final_prune_start_step) <= int(cfg.strategy.refine_stop_iter):
            raise ValueError(
                "fastgs.final_prune_start_step must be strictly after "
                "densification. Got "
                f"final_prune_start_step={int(fast.final_prune_start_step)} (1-based) "
                f"and strategy.refine_stop_iter={int(cfg.strategy.refine_stop_iter)} "
                "(0-based)."
            )
        if int(fast.final_prune_every) <= 0:
            raise ValueError(
                f"fastgs.final_prune_every must be > 0, got {fast.final_prune_every}"
            )
        if int(fast.final_prune_stop_step) < int(fast.final_prune_start_step):
            raise ValueError(
                "fastgs.final_prune_stop_step must be >= "
                f"fastgs.final_prune_start_step, got {fast.final_prune_stop_step} < "
                f"{fast.final_prune_start_step}"
            )
        if not (0.0 < float(fast.final_prune_min_opacity) < 1.0):
            raise ValueError(
                "fastgs.final_prune_min_opacity must be in (0, 1), got "
                f"{fast.final_prune_min_opacity}"
            )
        if not (0.0 < float(fast.final_prune_score_thresh) <= 1.0):
            raise ValueError(
                "fastgs.final_prune_score_thresh must be in (0, 1], got "
                f"{fast.final_prune_score_thresh}"
            )

    # --- optimizer schedule ------------------------------------------------
    if bool(fast.optim_gate_enable):
        valid_groups = {"means", "scales", "quats", "opacities", "sh0", "shN"}
        unknown = sorted(set(fast.phase1_gated_groups) - valid_groups)
        if len(unknown) > 0:
            raise ValueError(
                f"fastgs.phase1_gated_groups contains unknown groups {unknown}; "
                f"valid names are {sorted(valid_groups)}."
            )
        if int(fast.shN_every) <= 0:
            raise ValueError(f"fastgs.shN_every must be > 0, got {fast.shN_every}")
        if int(fast.phase2_every) <= 0:
            raise ValueError(
                f"fastgs.phase2_every must be > 0, got {fast.phase2_every}"
            )
        if int(fast.phase3_every) <= 0:
            raise ValueError(
                f"fastgs.phase3_every must be > 0, got {fast.phase3_every}"
            )
        if int(fast.optim_phase2_end_step) < int(fast.optim_phase1_end_step):
            raise ValueError(
                "fastgs.optim_phase2_end_step must be >= "
                f"fastgs.optim_phase1_end_step, got {fast.optim_phase2_end_step} < "
                f"{fast.optim_phase1_end_step}"
            )
        # Phase 1 must cover densification: throttling position/scale updates
        # while Gaussians are still being created breaks the growth criteria.
        if int(fast.optim_phase1_end_step) < int(cfg.strategy.refine_stop_iter):
            raise ValueError(
                "fastgs.optim_phase1_end_step must be >= "
                "strategy.refine_stop_iter so per-step updates cover the whole "
                f"densification window, got {fast.optim_phase1_end_step} < "
                f"{int(cfg.strategy.refine_stop_iter)}."
            )

    # --- learning rates ----------------------------------------------------
    if bool(fast.apply_presets):
        for name in ("opacity_lr", "sh0_lr", "shN_lr"):
            value = float(getattr(fast, name))
            if value <= 0.0:
                raise ValueError(f"fastgs.{name} must be > 0, got {value}")
