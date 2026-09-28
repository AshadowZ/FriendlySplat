from __future__ import annotations

"""FastGS densification strategy (gsplat `Strategy` implementation).

Port of FastGS's `densify_and_prune_fastgs` onto gsplat's parameter/optimizer
conventions:

- clone Gaussians whose *mean* 2D gradient is high and whose scale is small,
- split Gaussians whose *absolute* 2D gradient (AbsGS) is high and whose scale is
  large,
- in both cases only if the Gaussian is responsible for enough badly
  reconstructed pixels (the multi-view consistency filter, see `scoring.py`),
- prune by opacity / size, keeping only a fraction of the candidates per event,
- clamp opacities after each event, and reset them periodically.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union

import torch
from typing_extensions import Literal

from friendly_splat.fastgs.scoring import MultiViewScores
from gsplat.strategy import Strategy
from gsplat.strategy.ops import duplicate, remove, reset_opa, split


@dataclass
class FastGSStrategy(Strategy):
    """Multi-view consistent densification (FastGS, CVPR 2026).

    Args:
        prune_opa: Prune Gaussians with opacity below this value.
        grow_grad2d: Mean 2D-gradient threshold for cloning (FastGS `grad_thresh`).
        grad_abs_thresh: Absolute 2D-gradient threshold for splitting
            (FastGS `grad_abs_thresh`); requires `rasterization(absgrad=True)`.
        percent_dense: Scale threshold (as a fraction of the scene extent) that
            separates clone candidates from split candidates (FastGS `dense`).
        importance_thresh: Minimum per-view multi-view importance required to
            densify a Gaussian.
        prune_budget_ratio: Fraction of the prune candidates removed per event,
            highest pruning score first (FastGS's "remove budget").
        prune_scale3d: 3D scale (relative to scene scale) above which Gaussians
            are pruned once opacities have been reset at least once.
        prune_scale2d: Projected size (as a fraction of the larger image side)
            above which Gaussians are pruned. FastGS/vanilla 3DGS use an absolute
            20-pixel radius, which is resolution-dependent and prunes the coarse
            structure away at reduced resolutions; this follows gsplat's
            normalized convention instead.
        refine_scale2d_stop_iter: Stop the projected-size prune after this step
            (gsplat convention; 0 disables the projected-size prune entirely).
        opacity_clamp: Opacities are clamped to this value after each event.
        refine_start_iter: Start densifying after this step.
        refine_stop_iter: Stop densifying at this step.
        refine_every: Densify every this many steps.
        reset_every: Reset opacities every this many steps.
        opacity_reset_multiplier: Periodic reset target, as a multiple of
            `prune_opa`.
        budget: Hard cap on the number of Gaussians (0 disables the cap). FastGS
            has no budget; this is a safety valve inherited from FriendlySplat's
            `strategy.densification_budget`.
        absgrad: Whether the renderer accumulates absolute gradients.
        verbose: Print per-event statistics.
        key_for_gradient: Which `info` entry carries the 2D gradient.
        score_fn: Callable `(step, need_importance) -> Optional[MultiViewScores]`
            supplying the multi-view scores. When it returns None the strategy
            degrades gracefully to plain gradient-based densification.
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    grad_abs_thresh: float = 0.0012
    percent_dense: float = 0.001
    importance_thresh: float = 0.5
    prune_budget_ratio: float = 0.5
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 4000
    opacity_clamp: float = 0.8
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    refine_every: int = 500
    reset_every: int = 3000
    opacity_reset_multiplier: float = 2.0
    budget: int = 0
    absgrad: bool = True
    verbose: bool = True
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    score_fn: Optional[Callable[..., Optional[MultiViewScores]]] = None

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Running state; tensors are allocated lazily on the first update."""
        return {
            # Accumulated mean 2D gradient norm (clone criterion).
            "grad2d": None,
            # Accumulated absolute 2D gradient norm (split criterion).
            "grad2d_abs": None,
            # Number of views each Gaussian was visible in.
            "count": None,
            # Max projected size, normalized by the larger image side.
            "radii_norm": None,
            "scene_scale": scene_scale,
        }

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ) -> None:
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_pre_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
    ) -> None:
        del params, optimizers, state, step
        assert (
            self.key_for_gradient in info
        ), "The 2D means of the Gaussians is required but missing."
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
        lr: float | None = None,
    ) -> None:
        del lr
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and state.get("grad2d") is not None
        ):
            scores = (
                self.score_fn(step=int(step), need_importance=True)
                if self.score_fn is not None
                else None
            )
            # Pruning runs first so the multi-view scores stay index-aligned with
            # the parameters: gsplat's clone/split ops append and reorder
            # Gaussians, which would invalidate the score vector. `remove`
            # reindexes the running state, and we reindex the scores the same way
            # before growing. (Upstream FastGS grows first and prunes after; a
            # Gaussian that would be pruned is not worth cloning anyway.)
            n_prune, keep = self._prune_gs(params, optimizers, state, step, scores)
            if keep is not None:
                scores = _reindex_scores(scores=scores, keep=keep)
            n_dupli, n_split = self._grow_gs(params, optimizers, state, scores)
            # FastGS caps opacities after every densification event.
            if float(self.opacity_clamp) < 1.0:
                reset_opa(
                    params=params,
                    optimizers=optimizers,
                    state=state,
                    value=float(self.opacity_clamp),
                )
            if self.verbose:
                print(
                    f"[FastGS] step {step}: -{n_prune} pruned, +{n_dupli} cloned, "
                    f"+{n_split} split. Now having {len(params['means'])} GSs.",
                    flush=True,
                )
            state["grad2d"].zero_()
            state["grad2d_abs"].zero_()
            state["count"].zero_()
            state["radii_norm"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=float(self.prune_opa) * float(self.opacity_reset_multiplier),
            )
            if self.verbose:
                print(
                    f"[FastGS] step {step}: reset opacities to "
                    f"{float(self.prune_opa) * float(self.opacity_reset_multiplier)}.",
                    flush=True,
                )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ) -> None:
        """Accumulate both the mean and the absolute 2D gradients."""
        for key in ["width", "height", "n_cameras", "radii", self.key_for_gradient]:
            assert key in info, f"{key} is required but missing."

        means2d = info[self.key_for_gradient]
        if means2d.grad is None:
            return
        grads = means2d.grad.clone()
        grads_abs = grads
        if self.absgrad:
            absgrad = getattr(means2d, "absgrad", None)
            if absgrad is None:
                raise RuntimeError(
                    "FastGS needs absolute gradients for splitting; call "
                    "rasterization(..., absgrad=True) (strategy.absgrad=True)."
                )
            grads_abs = absgrad.clone()

        # Normalize gradients to [-1, 1] screen space (gsplat convention).
        scale_x = info["width"] / 2.0 * info["n_cameras"]
        scale_y = info["height"] / 2.0 * info["n_cameras"]
        grads[..., 0] *= scale_x
        grads[..., 1] *= scale_y
        if grads_abs is not grads:
            grads_abs[..., 0] *= scale_x
            grads_abs[..., 1] *= scale_y

        n_gaussian = len(list(params.values())[0])
        device = grads.device
        for key in ("grad2d", "grad2d_abs", "count", "radii_norm"):
            if state.get(key) is None:
                state[key] = torch.zeros(n_gaussian, device=device)

        if packed:
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"].max(dim=-1).values  # [nnz]
        else:
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            grads_abs = grads_abs[sel]  # [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["grad2d_abs"].index_add_(0, gs_ids, grads_abs.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        # Normalize by the larger image side so the threshold is resolution
        # independent (gsplat convention).
        state["radii_norm"][gs_ids] = torch.maximum(
            state["radii_norm"][gs_ids],
            radii.float() / float(max(int(info["width"]), int(info["height"]))),
        )

    def _metric_mask(
        self,
        *,
        scores: Optional[MultiViewScores],
        num_gaussians: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Multi-view consistency filter for densification candidates."""
        if scores is None or scores.importance is None:
            # No scores available (e.g. not enough cached views yet): fall back to
            # pure gradient-based densification instead of blocking growth.
            return torch.ones((num_gaussians,), dtype=torch.bool, device=device)
        importance = scores.importance
        if int(importance.numel()) != int(num_gaussians):
            raise RuntimeError(
                "FastGS score size mismatch: got "
                f"{int(importance.numel())} scores for {int(num_gaussians)} Gaussians."
            )
        mask = importance > float(self.importance_thresh)
        if self.verbose:
            print(
                f"[FastGS] scores over {scores.num_views} views: "
                f"importance mean={float(importance.mean()):.4f} "
                f"max={float(importance.max()):.4f} "
                f"selected={int(mask.sum())}/{int(num_gaussians)} "
                f"(thresh={float(self.importance_thresh)}).",
                flush=True,
            )
        return mask

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        scores: Optional[MultiViewScores],
    ) -> tuple[int, int]:
        count = state["count"].clamp_min(1.0)
        grads = state["grad2d"] / count
        grads_abs = state["grad2d_abs"] / count
        num_gaussians = int(params["means"].shape[0])
        device = grads.device

        metric_mask = self._metric_mask(
            scores=scores, num_gaussians=num_gaussians, device=device
        )

        scales_max = torch.exp(params["scales"]).max(dim=-1).values
        is_small = scales_max <= float(self.percent_dense) * state["scene_scale"]

        is_dupli = (grads > float(self.grow_grad2d)) & is_small & metric_mask
        is_split = (grads_abs > float(self.grad_abs_thresh)) & (~is_small) & metric_mask

        # Optional hard cap on the total number of Gaussians. Cloning adds one
        # Gaussian per selected splat, splitting adds one (2 replace 1).
        if int(self.budget) > 0:
            headroom = int(self.budget) - num_gaussians
            is_dupli, is_split = _cap_growth(
                is_dupli=is_dupli,
                is_split=is_split,
                dupli_scores=grads,
                split_scores=grads_abs,
                headroom=headroom,
            )

        n_dupli = int(is_dupli.sum())
        if n_dupli > 0:
            duplicate(params=params, optimizers=optimizers, state=state, mask=is_dupli)
        # Gaussians appended by cloning must not be split in the same event.
        if n_dupli > 0:
            is_split = torch.cat(
                [
                    is_split,
                    torch.zeros(n_dupli, dtype=torch.bool, device=device),
                ]
            )
        n_split = int(is_split.sum())
        if n_split > 0:
            split(params=params, optimizers=optimizers, state=state, mask=is_split)
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        scores: Optional[MultiViewScores],
    ) -> tuple[int, Optional[torch.Tensor]]:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < float(self.prune_opa)
        if step > self.reset_every:
            scales_max = torch.exp(params["scales"]).max(dim=-1).values
            is_too_big = scales_max > float(self.prune_scale3d) * state["scene_scale"]
            # Projected-size prune. FastGS inherits vanilla 3DGS's absolute
            # `max_radii2D > 20 px`, which is calibrated for ~1.6K-wide images and
            # deletes the coarse structure at lower resolutions. gsplat's
            # resolution-normalized threshold is used instead, on the same window
            # as FriendlySplat's other strategies.
            if (
                int(self.refine_scale2d_stop_iter) > 0
                and step < int(self.refine_scale2d_stop_iter)
                and float(self.prune_scale2d) > 0.0
            ):
                radii_norm = state.get("radii_norm")
                if isinstance(radii_norm, torch.Tensor) and int(
                    radii_norm.numel()
                ) == int(is_prune.numel()):
                    is_too_big = is_too_big | (radii_norm > float(self.prune_scale2d))
            is_prune = is_prune | is_too_big

        n_candidates = int(is_prune.sum())
        if n_candidates <= 0:
            return 0, None

        ratio = float(self.prune_budget_ratio)
        if ratio < 1.0 and scores is not None:
            # FastGS only removes a fraction of the candidates per event. We keep
            # the ones the multi-view score blames the most (upstream samples them
            # stochastically with weights 1 / (1 - pruning_score)).
            budget = int(ratio * n_candidates)
            if budget <= 0:
                return 0, None
            pruning = scores.pruning
            if int(pruning.numel()) != int(is_prune.numel()):
                raise RuntimeError(
                    "FastGS pruning-score size mismatch: got "
                    f"{int(pruning.numel())} scores for {int(is_prune.numel())} Gaussians."
                )
            candidate_scores = torch.where(
                is_prune, pruning, torch.full_like(pruning, -1.0)
            )
            worst_idx = torch.topk(candidate_scores, k=budget, largest=True).indices
            selected = torch.zeros_like(is_prune)
            selected[worst_idx] = True
            is_prune = is_prune & selected

        n_prune = int(is_prune.sum())
        if n_prune <= 0:
            return 0, None
        if n_prune >= int(is_prune.numel()):
            # Never remove every Gaussian.
            return 0, None
        keep = ~is_prune
        remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
        return n_prune, keep


@torch.no_grad()
def _reindex_scores(
    *, scores: Optional[MultiViewScores], keep: torch.Tensor
) -> Optional[MultiViewScores]:
    """Restrict scores to the Gaussians that survived pruning."""
    if scores is None:
        return None
    return MultiViewScores(
        importance=(scores.importance[keep] if scores.importance is not None else None),
        pruning=scores.pruning[keep],
        num_views=scores.num_views,
    )


@torch.no_grad()
def _cap_growth(
    *,
    is_dupli: torch.Tensor,
    is_split: torch.Tensor,
    dupli_scores: torch.Tensor,
    split_scores: torch.Tensor,
    headroom: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Trim the growth masks so at most `headroom` Gaussians are added."""
    headroom = int(headroom)
    if headroom <= 0:
        empty = torch.zeros_like(is_dupli)
        return empty, empty.clone()
    n_dupli = int(is_dupli.sum())
    n_split = int(is_split.sum())
    if n_dupli + n_split <= headroom:
        return is_dupli, is_split

    # Prefer splitting (it refines existing detail) then cloning, keeping the
    # highest-gradient candidates in each group.
    keep_split = min(n_split, headroom)
    keep_dupli = max(0, headroom - keep_split)
    return (
        _keep_topk(mask=is_dupli, scores=dupli_scores, keep=keep_dupli),
        _keep_topk(mask=is_split, scores=split_scores, keep=keep_split),
    )


@torch.no_grad()
def _keep_topk(*, mask: torch.Tensor, scores: torch.Tensor, keep: int) -> torch.Tensor:
    keep = int(keep)
    if keep <= 0:
        return torch.zeros_like(mask)
    if keep >= int(mask.sum()):
        return mask
    masked_scores = torch.where(mask, scores, torch.full_like(scores, -1.0))
    idx = torch.topk(masked_scores, k=keep, largest=True).indices
    out = torch.zeros_like(mask)
    out[idx] = True
    return out & mask
