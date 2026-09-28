from __future__ import annotations

"""Configuration for the FastGS training recipe.

FastGS (`Training 3D Gaussian Splatting in 100 Seconds
<https://arxiv.org/abs/2511.04283>`_, CVPR 2026) accelerates 3DGS training with
three ingredients:

1. A *multi-view consistency* score that tells densification/pruning which
   Gaussians actually sit on badly reconstructed pixels.
2. A densification rule that combines that score with AbsGS-style gradients
   (clone by mean gradient, split by absolute gradient).
3. A sparse-in-time optimizer schedule that stops updating every parameter on
   every iteration once the geometry has settled.

Everything is off unless `--fast` is passed. All knobs below live under
`--fastgs.*` and are only read when `--fast` is enabled.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple


@dataclass(frozen=True)
class FastGSConfig:
    """FastGS knobs (only used when `TrainConfig.fast=True`)."""

    # ---------------------------------------------------------------
    # Multi-view consistency scoring
    # ---------------------------------------------------------------

    # Number of views used per scoring event (FastGS samples 10 cameras).
    score_num_views: int = 10
    # Size of the recently-seen-view cache scored views are drawn from.
    # None means `2 * score_num_views`. Views are cached as uint8 on the training
    # device, so scoring never touches the disk (FastGS relies on all images
    # being resident in GPU memory; FriendlySplat streams them instead).
    view_cache_size: Optional[int] = None
    # Threshold on the min-max normalized per-pixel L1 error. Pixels above it are
    # considered "badly reconstructed" and form the metric map. Lower keeps more
    # Gaussians (FastGS `--loss_thresh`).
    loss_thresh: float = 0.1
    # How per-Gaussian responsibility for those pixels is measured:
    # - "weighted": autograd on a 1-channel feature render, giving each Gaussian
    #   `sum_{p in metric map} alpha_p * T_p` (its expected number of bad pixels).
    #   Complete, bounded memory, no thresholds. Two renders per view.
    # - "tracked": hard pixel counts via gsplat's `track_pixel_gaussians`, i.e.
    #   the number of bad pixels where the Gaussian contributes more than
    #   `tracked_vis_thresh`. Closest to the upstream CUDA counter and needs only
    #   one render per view, but allocates
    #   `H * W * ceil(1 / tracked_vis_thresh)` index pairs per view.
    # The two backends live on different scales; see `importance_thresh`.
    score_backend: Literal["weighted", "tracked"] = "weighted"
    # Minimum contribution (alpha * T) for a Gaussian to be counted at a pixel.
    # Only used by `score_backend="tracked"`.
    tracked_vis_thresh: float = 0.05

    # ---------------------------------------------------------------
    # Densification (FastGS multi-view consistent densification)
    # ---------------------------------------------------------------

    # Run densification every N steps. FastGS uses 500 ("base", fastest) or
    # 100 ("big", higher quality). This overrides `strategy.refine_every`.
    # Default 100: it matches FriendlySplat's own `strategy.refine_every`, and on
    # garden under the MipNeRF360 protocol it is what beat the `improved`
    # baseline (see README). Use 500 for FastGS's "base" speed/quality point.
    densify_every: int = 100
    # Absolute-gradient threshold used for *splitting* (FastGS `--grad_abs_thresh`,
    # AbsGS convention). Cloning uses `strategy.grow_grad2d` (FastGS `--grad_thresh`).
    grad_abs_thresh: float = 0.0012
    # Scale threshold (fraction of scene extent) separating clone from split
    # candidates (FastGS `--dense`, i.e. `percent_dense`).
    percent_dense: float = 0.001
    # Minimum per-view importance for a Gaussian to be allowed to densify.
    # None resolves to a backend-specific default (see `resolve_importance_thresh`):
    # 0.15 for "weighted", 0.6 for "tracked". FastGS uses 5 hard pixels/view with
    # its own CUDA counter, so this value always needs a bit of per-scene tuning;
    # `--fastgs.verbose` prints the score distribution to make that easy.
    # 0.15 is the measured value on garden (MipNeRF360 protocol); the earlier
    # default of 0.5 throttled growth hard enough that the model shrank. The
    # "tracked" default is 0.15 scaled by the measured mean ratio between the two
    # backends (~3.7x) and has not been validated on its own.
    importance_thresh: Optional[float] = None
    # Fraction of the opacity/size prune candidates actually removed per event,
    # highest pruning-score first (FastGS's "remove budget"; 1.0 prunes all).
    prune_budget_ratio: float = 0.5
    # NOTE: the size-based prune uses FriendlySplat's `strategy.prune_scale3d`
    # and `strategy.prune_scale2d` (the latter gated by
    # `strategy.refine_scale2d_stop_iter`). FastGS's absolute 20-pixel screen
    # radius is deliberately not used: it is calibrated for ~1.6K-wide images and
    # deletes the coarse structure at lower resolutions (measured on bonsai at
    # 1/4 resolution: 18.1 dB with it, 28.5 dB without).
    # Clamp opacities to this value after every densification event (FastGS
    # applies `min(opacity, 0.8)` at the end of `densify_and_prune`).
    # 1.0 disables the clamp.
    opacity_clamp: float = 0.8
    # Periodic opacity reset value, as a multiple of `strategy.prune_opa`
    # (reset happens every `strategy.reset_every` steps). FastGS and vanilla 3DGS
    # reset to 0.01 with a 0.005 prune threshold, i.e. 2x. gsplat's own
    # strategies use 2x (`default`) or 10x (`improved`); a larger multiple leaves
    # more headroom above the prune threshold and so prunes less aggressively
    # right after a reset.
    opacity_reset_multiplier: float = 2.0

    # ---------------------------------------------------------------
    # Post-densification multi-view consistent pruning
    # ---------------------------------------------------------------

    # Enable the aggressive late pruning phase.
    final_prune_enable: bool = True
    # First (1-based) step of the phase; must be after densification stops.
    final_prune_start_step: int = 18_000
    # Cadence (1-based steps, anchored at `final_prune_start_step`).
    final_prune_every: int = 3000
    # Last (1-based) step of the phase (inclusive).
    final_prune_stop_step: int = 27_000
    # Opacity below this is pruned during the phase.
    final_prune_min_opacity: float = 0.1
    # Normalized (0..1) pruning score above which a Gaussian is pruned.
    final_prune_score_thresh: float = 0.9

    # ---------------------------------------------------------------
    # Sparse-in-time optimizer schedule
    # ---------------------------------------------------------------

    # Enable FastGS's optimizer cadence (gradients accumulate between steps).
    optim_gate_enable: bool = True
    # Phase 1 (1-based, inclusive): all groups step every iteration, except the
    # groups in `phase1_gated_groups`, which step every `shN_every` iterations.
    optim_phase1_end_step: int = 15_000
    shN_every: int = 16
    # Which splat parameter groups are throttled during phase 1. FastGS throttles
    # only the high-order SH coefficients, which is why it also raises their
    # learning rate. Valid names: means, scales, quats, opacities, sh0, shN.
    phase1_gated_groups: Tuple[str, ...] = ("shN",)
    # Phase 2 (1-based, inclusive): every group steps every `phase2_every` iters.
    optim_phase2_end_step: int = 20_000
    phase2_every: int = 32
    # Phase 3 (rest of training): every group steps every `phase3_every` iters.
    phase3_every: int = 64

    # ---------------------------------------------------------------
    # Learning-rate / strategy presets
    # ---------------------------------------------------------------

    # Apply FastGS's learning rates and densification cadence on top of the
    # FriendlySplat defaults (printed on startup). Disable to keep FriendlySplat
    # defaults and only get FastGS's densification/pruning/optimizer schedule.
    apply_presets: bool = True
    # FastGS `--opacity_lr`.
    opacity_lr: float = 0.025
    # FastGS `--lowfeature_lr` (SH DC coefficients).
    sh0_lr: float = 0.0025
    # FastGS `--highfeature_lr` (SH rest); the optimizer LR is this divided by 20.
    shN_lr: float = 0.005

    # Print densification / pruning / scoring statistics.
    verbose: bool = True


def resolve_importance_thresh(fast_cfg: FastGSConfig) -> float:
    """Return the densification importance threshold for the active backend."""
    if fast_cfg.importance_thresh is not None:
        return float(fast_cfg.importance_thresh)
    return 0.15 if str(fast_cfg.score_backend) == "weighted" else 0.6


def resolve_view_cache_size(fast_cfg: FastGSConfig) -> int:
    """Return the number of views kept for scoring."""
    if fast_cfg.view_cache_size is not None:
        return max(1, int(fast_cfg.view_cache_size))
    return max(1, 2 * int(fast_cfg.score_num_views))
