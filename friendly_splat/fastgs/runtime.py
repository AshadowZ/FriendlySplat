from __future__ import annotations

"""Glue between the FriendlySplat training loop and the FastGS pieces.

`FastGSRuntime` owns everything `--fast` adds:

- the cache of recently trained views used for scoring,
- the `FastGSStrategy` used in place of the configured densification strategy,
- the post-densification multi-view consistent pruning phase,
- the sparse-in-time optimizer gate.

The training loop only needs three calls (`observe_step`, `maybe_final_prune`)
plus handing `step_gate` to the optimizer coordinator, so `--fast` stays a
strictly additive code path.
"""

from typing import Any, Dict, Optional

import torch

from friendly_splat.fastgs.config import (
    FastGSConfig,
    resolve_importance_thresh,
    resolve_view_cache_size,
)
from friendly_splat.fastgs.optim_schedule import FastGSStepGate
from friendly_splat.fastgs.scoring import (
    MultiViewScores,
    ViewCache,
    compute_multiview_scores,
)
from friendly_splat.fastgs.strategy import FastGSStrategy
from friendly_splat.modules.gaussian import GaussianModel
from gsplat.strategy.ops import remove


class FastGSRuntime:
    """Owns FastGS state for one training run."""

    def __init__(
        self,
        *,
        fast_cfg: FastGSConfig,
        strategy_cfg: Any,
        optim_cfg: Any,
        reg_cfg: Any,
        scene_scale: float,
        seed: int,
    ) -> None:
        self.cfg = fast_cfg
        self.scene_scale = float(scene_scale)
        self.ssim_lambda = float(reg_cfg.ssim_lambda)
        self.rasterize_mode = (
            "antialiased" if bool(optim_cfg.antialiased) else "classic"
        )
        self.max_steps = int(optim_cfg.max_steps)
        self.verbose = bool(fast_cfg.verbose)

        self.view_cache = ViewCache(capacity=resolve_view_cache_size(fast_cfg))
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(int(seed) + 0x5A5A)

        self.strategy = FastGSStrategy(
            prune_opa=float(strategy_cfg.prune_opa),
            grow_grad2d=float(strategy_cfg.grow_grad2d),
            grad_abs_thresh=float(fast_cfg.grad_abs_thresh),
            percent_dense=float(fast_cfg.percent_dense),
            importance_thresh=resolve_importance_thresh(fast_cfg),
            prune_budget_ratio=float(fast_cfg.prune_budget_ratio),
            prune_scale3d=float(strategy_cfg.prune_scale3d),
            prune_scale2d=float(strategy_cfg.prune_scale2d),
            refine_scale2d_stop_iter=int(strategy_cfg.refine_scale2d_stop_iter),
            opacity_clamp=float(fast_cfg.opacity_clamp),
            refine_start_iter=int(strategy_cfg.refine_start_iter),
            refine_stop_iter=int(strategy_cfg.refine_stop_iter),
            refine_every=int(strategy_cfg.refine_every),
            reset_every=int(strategy_cfg.reset_every),
            opacity_reset_multiplier=float(fast_cfg.opacity_reset_multiplier),
            budget=int(strategy_cfg.densification_budget),
            absgrad=bool(strategy_cfg.absgrad),
            verbose=bool(fast_cfg.verbose),
            key_for_gradient=str(strategy_cfg.key_for_gradient),
            score_fn=self.compute_scores,
        )

        self.step_gate: Optional[FastGSStepGate] = None
        if bool(fast_cfg.optim_gate_enable):
            self.step_gate = FastGSStepGate(
                phase1_end_step=int(fast_cfg.optim_phase1_end_step),
                phase2_end_step=int(fast_cfg.optim_phase2_end_step),
                shN_every=int(fast_cfg.shN_every),
                phase2_every=int(fast_cfg.phase2_every),
                phase3_every=int(fast_cfg.phase3_every),
                max_steps=int(optim_cfg.max_steps),
                phase1_gated_groups=tuple(str(g) for g in fast_cfg.phase1_gated_groups),
            )

        self._gaussian_model: Optional[GaussianModel] = None
        self._bilateral_grid: Any = None
        self._active_sh_degree: int = 0

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def bind(
        self,
        *,
        gaussian_model: GaussianModel,
        bilateral_grid: Any = None,
    ) -> None:
        """Attach what scoring needs to render (called once at build time)."""
        self._gaussian_model = gaussian_model
        self._bilateral_grid = bilateral_grid

    def observe_step(
        self,
        *,
        pixels: torch.Tensor,
        camtoworlds: torch.Tensor,
        Ks: torch.Tensor,
        active_sh_degree: int,
        image_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Record the current batch so later scoring events can reuse it."""
        self._active_sh_degree = int(active_sh_degree)
        self.view_cache.observe(
            pixels=pixels, camtoworlds=camtoworlds, Ks=Ks, image_ids=image_ids
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_scores(
        self, *, step: int, need_importance: bool
    ) -> Optional[MultiViewScores]:
        """Score the current model against a sample of recent views."""
        del step
        if self._gaussian_model is None:
            raise RuntimeError(
                "FastGSRuntime.bind(gaussian_model=...) must be called before scoring."
            )
        views = self.view_cache.sample(
            num_views=int(self.cfg.score_num_views), generator=self._generator
        )
        if len(views) == 0:
            return None
        return compute_multiview_scores(
            gaussian_model=self._gaussian_model,
            views=views,
            fast_cfg=self.cfg,
            sh_degree=int(self._active_sh_degree),
            ssim_lambda=float(self.ssim_lambda),
            rasterize_mode=str(self.rasterize_mode),
            need_importance=bool(need_importance),
            photometric_adapter=self._bilateral_grid,
        )

    # ------------------------------------------------------------------
    # Post-densification pruning
    # ------------------------------------------------------------------

    def _final_prune_due(self, *, step: int) -> bool:
        cfg = self.cfg
        if not bool(cfg.final_prune_enable):
            return False
        train_step = int(step) + 1  # 1-based, like the config
        start = int(cfg.final_prune_start_step)
        every = max(1, int(cfg.final_prune_every))
        if train_step < start or train_step > int(cfg.final_prune_stop_step):
            return False
        return ((train_step - start) % every) == 0

    def maybe_final_prune(
        self,
        *,
        step: int,
        gaussian_model: GaussianModel,
        splat_optimizers: Dict[str, torch.optim.Optimizer],
        strategy_state: Dict[str, Any],
    ) -> None:
        """FastGS's late multi-view consistent pruning.

        Once densification is over the model has essentially converged, so
        Gaussians that the multi-view score still blames for the remaining error
        can be removed outright (upstream: `final_prune_fastgs`).
        """
        if not self._final_prune_due(step=int(step)):
            return
        n_before = int(gaussian_model.num_gaussians)
        if n_before <= 1:
            return

        scores = self.compute_scores(step=int(step), need_importance=False)
        if scores is None:
            return

        cfg = self.cfg
        with torch.no_grad():
            opacities = torch.sigmoid(gaussian_model.opacity_logits.flatten())
            is_prune = opacities < float(cfg.final_prune_min_opacity)
            pruning = scores.pruning
            if int(pruning.numel()) != int(is_prune.numel()):
                raise RuntimeError(
                    "FastGS pruning-score size mismatch: got "
                    f"{int(pruning.numel())} scores for {int(is_prune.numel())} Gaussians."
                )
            is_prune = is_prune | (pruning > float(cfg.final_prune_score_thresh))
            n_prune = int(is_prune.sum())
            if n_prune <= 0:
                return
            if n_prune >= n_before:
                # Never remove every Gaussian.
                return
            remove(
                params=gaussian_model.splats,
                optimizers=splat_optimizers,
                state=strategy_state,
                mask=is_prune,
            )
        torch.cuda.empty_cache()
        if self.verbose:
            print(
                f"[FastGS] step {int(step)}: final prune removed {n_prune} GSs "
                f"(opacity < {float(cfg.final_prune_min_opacity)} or score > "
                f"{float(cfg.final_prune_score_thresh)}). "
                f"Now having {int(gaussian_model.num_gaussians)} GSs.",
                flush=True,
            )


def build_fastgs_runtime(
    *,
    fast_cfg: FastGSConfig,
    strategy_cfg: Any,
    optim_cfg: Any,
    reg_cfg: Any,
    scene_scale: float,
    seed: int,
    gaussian_model: GaussianModel,
    bilateral_grid: Any = None,
) -> FastGSRuntime:
    """Create and bind a `FastGSRuntime`."""
    runtime = FastGSRuntime(
        fast_cfg=fast_cfg,
        strategy_cfg=strategy_cfg,
        optim_cfg=optim_cfg,
        reg_cfg=reg_cfg,
        scene_scale=float(scene_scale),
        seed=int(seed),
    )
    runtime.bind(gaussian_model=gaussian_model, bilateral_grid=bilateral_grid)
    return runtime
