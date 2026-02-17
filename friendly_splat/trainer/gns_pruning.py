from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from friendly_splat.modules.gaussian import GaussianModel
from friendly_splat.trainer.configs import GNSConfig

from gsplat.strategy.ops import _multinomial_sample, remove


def auto_gns_reg_interval(num_train_images: int) -> int:
    """Heuristic for choosing GNS reg_interval based on training set size.

    Picks the closest multiple of 50 to (num_train_images / 10), with a minimum
    of 50. This matches the "images/10 then snap to {50,100,150,200,...}" rule.
    """
    if num_train_images <= 0:
        raise ValueError(f"num_train_images must be > 0, got {num_train_images}")
    target = float(num_train_images) / 10.0
    k = int(math.floor((target + 25.0) / 50.0))
    k = max(1, k)
    return int(k * 50)


@dataclass
class NaturalSelectionState:
    finished: bool = False
    start_count: Optional[int] = None
    opacity_reg_weight: float = 0.0
    stop_step: Optional[int] = None
    opacity_lr_scaled: bool = False


@dataclass
class NaturalSelectionPolicy:
    """Natural Selection pruning + opacity regularization policy (FriendlySplat).

    This policy is intentionally trainer-scoped:
      - It is configured via `GNSConfig`.
      - It operates on `GaussianModel` and FriendlySplat's optimizer dicts.
    """

    cfg: GNSConfig
    densify_stop_step: int
    reg_interval: int
    min_opacity: float = 0.001
    opacity_lr_scale: float = 4.0
    verbose: bool = True
    state: NaturalSelectionState = field(init=False)

    # Mirror legacy attribute names used by trainer code.
    enable: bool = field(init=False)
    reg_start: int = field(init=False)  # 1-based train step (inclusive)
    reg_end: int = field(init=False)  # 1-based train step (inclusive)
    final_budget: int = field(init=False)
    opacity_reg_weight: float = field(init=False)

    def __post_init__(self) -> None:
        self.enable = bool(self.cfg.gns_enable)
        self.reg_start = int(self.cfg.reg_start)
        self.reg_end = int(self.cfg.reg_end)
        self.final_budget = int(self.cfg.final_budget)
        self.opacity_reg_weight = float(self.cfg.opacity_reg_weight)

        self.state = NaturalSelectionState(opacity_reg_weight=float(self.opacity_reg_weight))
        if self.enable and self.verbose:
            print(
                f"[GNS] Enabled: densify_stop_step={int(self.densify_stop_step)}, "
                f"reg_start={int(self.reg_start)} (1-based), reg_end={int(self.reg_end)} (1-based), reg_interval={int(self.reg_interval)}, "
                f"final_budget={int(self.final_budget)}, opacity_reg_weight={float(self.opacity_reg_weight)}",
                flush=True,
            )

    def _train_step(self, *, step: int) -> int:
        return int(step) + 1

    def window_active(self, *, step: int) -> bool:
        if not self.enable or self.state.finished:
            return False
        train_step = int(self._train_step(step=int(step)))
        return int(self.reg_start) <= int(train_step) <= int(self.reg_end)

    @staticmethod
    def _num_gaussians(gaussian_model: GaussianModel) -> int:
        return int(gaussian_model.num_gaussians)

    def maybe_scale_opacity_lr(
        self, *, step: int, optimizers: Dict[str, torch.optim.Optimizer]
    ) -> None:
        st = self.state
        if not self.enable or st.finished:
            return
        train_step = int(self._train_step(step=int(step)))
        if int(train_step) != int(self.reg_start) or bool(st.opacity_lr_scaled):
            return
        if "opacities" not in optimizers:
            raise KeyError("GNS requires optimizers['opacities'] to scale opacity LR.")
        if self.verbose:
            print(
                f"[GNS] Starting Natural Selection: Scaling Opacity LR by {float(self.opacity_lr_scale)}x at step {int(train_step)}",
                flush=True,
            )
        for param_group in optimizers["opacities"].param_groups:
            param_group["lr"] *= float(self.opacity_lr_scale)
        st.opacity_lr_scaled = True

    def maybe_restore_opacity_lr(
        self, *, step: int, optimizers: Dict[str, torch.optim.Optimizer]
    ) -> None:
        st = self.state
        if not self.enable:
            return
        if st.stop_step is None:
            return
        if int(step) != int(st.stop_step) + 1000:
            return
        if "opacities" not in optimizers:
            raise KeyError(
                "GNS requires optimizers['opacities'] to restore opacity LR."
            )
        if st.opacity_lr_scaled:
            if self.verbose:
                print(
                    f"[GNS] Restoring Opacity LR (1000 steps after stop) at step {int(step)}",
                    flush=True,
                )
            for param_group in optimizers["opacities"].param_groups:
                param_group["lr"] /= float(self.opacity_lr_scale)
            st.opacity_lr_scaled = False
        st.stop_step = None

    def compute_regularizer(
        self,
        *,
        step: int,
        gaussian_model: GaussianModel,
    ) -> Optional[torch.Tensor]:
        """Compute the GNS opacity regularizer term for the current step."""
        st = self.state
        if not self.enable or st.finished:
            return None
        train_step = int(self._train_step(step=int(step)))
        if not (int(self.reg_start) <= int(train_step) <= int(self.reg_end)):
            return None
        if (int(train_step) - 1) % int(self.reg_interval) != 0:
            return None

        params = gaussian_model.splats
        if "opacities" not in params:
            raise KeyError("GNS requires params['opacities'] (opacity logits).")
        opacities_logits = params["opacities"].flatten()

        # Dynamic opacity_reg_weight adjustment on reg_interval cadence.
        if (int(train_step) - 1) % int(self.reg_interval) == 0:
            current_count = int(opacities_logits.shape[0])
            if st.start_count is None:
                st.start_count = current_count
                if int(st.start_count) < int(self.final_budget):
                    st.start_count = int(self.final_budget) + 1000

            den = float(int(self.reg_end) - int(self.reg_start)) or 1.0
            progress = float(int(train_step) - int(self.reg_start)) / den
            progress = max(0.0, min(1.0, progress))
            expected_count = float(st.start_count) - (
                float(int(st.start_count) - int(self.final_budget)) * progress
            )

            if float(current_count) > expected_count * 1.05:
                st.opacity_reg_weight *= 1.2
            elif float(current_count) < expected_count * 0.95:
                st.opacity_reg_weight *= 0.8
            st.opacity_reg_weight = float(max(1e-7, min(st.opacity_reg_weight, 1e-2)))

        w = float(st.opacity_reg_weight)
        if int(train_step) < int(self.reg_start) + 1000:
            current_opacities = torch.sigmoid(opacities_logits)
            rate_l = torch.maximum(
                torch.ones_like(current_opacities) * 0.05,
                1.0 - current_opacities,
            )
            term = (opacities_logits + 20.0) / rate_l
            return (w * (torch.mean(term) ** 2)).to(opacities_logits.dtype)

        mean_val = torch.mean(opacities_logits)
        return (3.0 * w * ((mean_val + 20.0) ** 2)).to(opacities_logits.dtype)

    @torch.no_grad()
    def maybe_prune_after_update(
        self,
        *,
        step: int,
        gaussian_model: GaussianModel,
        optimizers: Dict[str, torch.optim.Optimizer],
        strategy_state: Dict[str, Any],
    ) -> None:
        """Run pruning actions after optimizer/strategy updates for this step."""
        st = self.state
        if not self.enable or st.finished:
            return
        train_step = int(self._train_step(step=int(step)))
        if int(train_step) < int(self.reg_start) or int(train_step) > int(self.reg_end):
            return

        params = gaussian_model.splats
        if "opacities" not in params:
            raise KeyError("GNS requires params['opacities'] (opacity logits).")

        current_gs_count = int(params["opacities"].numel())

        # Early stop: if we're already close to the budget, force a final prune and finish.
        if int(train_step) > int(self.reg_start) and current_gs_count < float(self.final_budget) * 1.05:
            if self.verbose:
                print(
                    f"[GNS] Count {current_gs_count} < 1.05 * Budget. "
                    f"Stopping Natural Selection early at step {int(train_step)}.",
                    flush=True,
                )
                print(
                    f"[GNS] Step {int(train_step)}: Running Final Budget Prune to {int(self.final_budget)}...",
                    flush=True,
                )
            n_pruned = self._final_prune(
                gaussian_model=gaussian_model,
                optimizers=optimizers,
                strategy_state=strategy_state,
            )
            if self.verbose:
                print(
                    f"[GNS] Final Prune removed {n_pruned} gaussians. "
                    f"Now having {self._num_gaussians(gaussian_model)} GSs.",
                    flush=True,
                )
            st.finished = True
            st.stop_step = int(step)
            return

        # Window pruning: periodically remove very transparent Gaussians.
        if (
            int(train_step) < int(self.reg_end)
            and int(train_step) % int(self.reg_interval) == 0
            and int(train_step) >= int(self.reg_start) + 1000
        ):
            n_pruned = 0
            if current_gs_count > int(self.final_budget):
                n_pruned = self._opacity_prune(
                    gaussian_model=gaussian_model,
                    optimizers=optimizers,
                    strategy_state=strategy_state,
                    min_opacity=float(self.min_opacity),
                )
            if self.verbose:
                # Always print on the pruning cadence, even when no Gaussians were pruned,
                # to make it easy to verify that the policy is active.
                print(
                    f"[GNS] Step {int(train_step)}: Removed {n_pruned} GSs "
                    f"below opacity threshold. Now having {self._num_gaussians(gaussian_model)} GSs.",
                    flush=True,
                )

        # Final prune at reg_end: enforce the budget via probabilistic survival.
        if int(train_step) == int(self.reg_end):
            if self.verbose:
                print(
                    f"[GNS] Step {int(train_step)}: Running Final Budget Prune to {int(self.final_budget)}...",
                    flush=True,
                )
            n_pruned = self._final_prune(
                gaussian_model=gaussian_model,
                optimizers=optimizers,
                strategy_state=strategy_state,
            )
            if self.verbose:
                print(
                    f"[GNS] Final Prune removed {n_pruned} gaussians. "
                    f"Now having {self._num_gaussians(gaussian_model)} GSs.",
                    flush=True,
                )
            st.finished = True
            if st.stop_step is None:
                st.stop_step = int(step)

    def _opacity_prune(
        self,
        *,
        gaussian_model: GaussianModel,
        optimizers: Dict[str, torch.optim.Optimizer],
        strategy_state: Dict[str, Any],
        min_opacity: float,
    ) -> int:
        params = gaussian_model.splats
        opacities = torch.sigmoid(params["opacities"].flatten())
        is_prune = opacities < float(min_opacity)
        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            remove(
                params=params,
                optimizers=optimizers,
                state=strategy_state,
                mask=is_prune,
            )
        return n_prune

    def _final_prune(
        self,
        *,
        gaussian_model: GaussianModel,
        optimizers: Dict[str, torch.optim.Optimizer],
        strategy_state: Dict[str, Any],
    ) -> int:
        target_budget = int(self.final_budget)
        params = gaussian_model.splats
        opacities = torch.sigmoid(params["opacities"].flatten())
        n_curr = int(opacities.shape[0])
        if n_curr <= target_budget:
            return 0

        keep_indices = _multinomial_sample(opacities, target_budget, replacement=False)
        is_prune = torch.ones(n_curr, dtype=torch.bool, device=opacities.device)
        is_prune[keep_indices] = False

        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            remove(
                params=params,
                optimizers=optimizers,
                state=strategy_state,
                mask=is_prune,
            )
        return n_prune
