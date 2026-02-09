from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Union

import torch

from .ops import _multinomial_sample, remove


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
    """Natural Selection pruning + opacity regularization policy.

    Responsibilities:
      - Opacity LR scaling at reg_start and restoration after stop+1000 steps.
      - Opacity regularizer (GNS loss) computed on a fixed cadence.
      - Pruning inside the window, and final probabilistic pruning to a budget.
    """

    enable: bool
    densify_stop_step: int
    reg_start: int
    reg_end: int
    reg_interval: int
    final_budget: int
    opacity_reg_weight: float
    min_opacity: float = 0.001
    opacity_lr_scale: float = 4.0
    verbose: bool = True
    state: NaturalSelectionState = field(init=False)

    def __post_init__(self) -> None:
        self.state = NaturalSelectionState(
            opacity_reg_weight=float(self.opacity_reg_weight)
        )
        self._validate()
        if self.enable and self.verbose:
            print(
                f"[GNS] Enabled: densify_stop_step={self.densify_stop_step}, "
                f"reg_start={self.reg_start}, reg_end={self.reg_end}, reg_interval={self.reg_interval}, "
                f"final_budget={self.final_budget}, opacity_reg_weight={self.opacity_reg_weight}",
                flush=True,
            )

    @staticmethod
    def _num_gaussians(
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]
    ) -> int:
        if "means" in params:
            return int(len(params["means"]))
        return int(params["opacities"].numel())

    def _validate(self) -> None:
        cfg = self
        if not cfg.enable:
            return
        if cfg.densify_stop_step < 0:
            raise ValueError(
                f"densify_stop_step must be >= 0, got {cfg.densify_stop_step}"
            )
        if cfg.reg_interval <= 0:
            raise ValueError(f"reg_interval must be > 0, got {cfg.reg_interval}")
        if cfg.final_budget <= 0:
            raise ValueError(f"final_budget must be > 0, got {cfg.final_budget}")
        if cfg.reg_end < cfg.reg_start:
            raise ValueError(
                f"reg_end must be >= reg_start, got {cfg.reg_end} < {cfg.reg_start}"
            )
        if cfg.reg_start < cfg.densify_stop_step:
            raise ValueError(
                "Natural selection must start after densification finishes, got "
                f"reg_start={cfg.reg_start} < densify_stop_step={cfg.densify_stop_step}"
            )
        if cfg.opacity_reg_weight <= 0.0:
            raise ValueError(
                f"opacity_reg_weight must be > 0, got {cfg.opacity_reg_weight}"
            )
        if cfg.opacity_lr_scale <= 0.0:
            raise ValueError(
                f"opacity_lr_scale must be > 0, got {cfg.opacity_lr_scale}"
            )
        if cfg.min_opacity < 0.0:
            raise ValueError(f"min_opacity must be >= 0, got {cfg.min_opacity}")

    def maybe_scale_opacity_lr(
        self, *, step: int, optimizers: Dict[str, torch.optim.Optimizer]
    ) -> None:
        cfg = self
        st = self.state
        if not cfg.enable or st.finished:
            return
        if step != cfg.reg_start or st.opacity_lr_scaled:
            return
        if "opacities" not in optimizers:
            raise KeyError("GNS requires optimizers['opacities'] to scale opacity LR.")
        if cfg.verbose:
            print(
                f"[GNS] Starting Natural Selection: Scaling Opacity LR by {cfg.opacity_lr_scale}x at step {step}",
                flush=True,
            )
        for param_group in optimizers["opacities"].param_groups:
            param_group["lr"] *= float(cfg.opacity_lr_scale)
        st.opacity_lr_scaled = True

    def maybe_restore_opacity_lr(
        self, *, step: int, optimizers: Dict[str, torch.optim.Optimizer]
    ) -> None:
        cfg = self
        st = self.state
        if not cfg.enable:
            return
        if st.stop_step is None:
            return
        if step != int(st.stop_step) + 1000:
            return
        if "opacities" not in optimizers:
            raise KeyError(
                "GNS requires optimizers['opacities'] to restore opacity LR."
            )
        if st.opacity_lr_scaled:
            if cfg.verbose:
                print(
                    f"[GNS] Restoring Opacity LR (1000 steps after stop) at step {step}",
                    flush=True,
                )
            for param_group in optimizers["opacities"].param_groups:
                param_group["lr"] /= float(cfg.opacity_lr_scale)
            st.opacity_lr_scaled = False
        st.stop_step = None

    def compute_regularizer(
        self,
        *,
        step: int,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    ) -> Optional[torch.Tensor]:
        """Compute the GNS opacity regularizer term for the current step."""
        cfg = self
        st = self.state
        if not cfg.enable or st.finished:
            return None
        if not (cfg.reg_start <= step <= cfg.reg_end):
            return None
        if (step - 1) % int(cfg.reg_interval) != 0:
            return None

        if "opacities" not in params:
            raise KeyError("GNS requires params['opacities'] (opacity logits).")
        opacities_logits = params["opacities"].flatten()

        # Dynamic opacity_reg_weight adjustment on reg_interval cadence.
        if (step - 1) % int(cfg.reg_interval) == 0:
            current_count = (
                int(params["means"].shape[0])
                if "means" in params
                else int(opacities_logits.shape[0])
            )
            if st.start_count is None:
                st.start_count = current_count
                if st.start_count < int(cfg.final_budget):
                    st.start_count = int(cfg.final_budget) + 1000

            den = (
                float(cfg.reg_end - cfg.reg_start)
                if cfg.reg_end != cfg.reg_start
                else 1.0
            )
            progress = float(step - cfg.reg_start) / den
            progress = max(0.0, min(1.0, progress))
            expected_count = float(st.start_count) - (
                float(st.start_count - cfg.final_budget) * progress
            )

            if float(current_count) > expected_count * 1.05:
                st.opacity_reg_weight *= 1.2
            elif float(current_count) < expected_count * 0.95:
                st.opacity_reg_weight *= 0.8
            st.opacity_reg_weight = float(max(1e-7, min(st.opacity_reg_weight, 1e-2)))

        w = float(st.opacity_reg_weight)
        if step < cfg.reg_start + 1000:
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
    def step_post_update(
        self,
        *,
        step: int,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        strategy_state: Dict[str, Any],
    ) -> None:
        """Run pruning actions after optimizer/strategy updates for this step."""
        cfg = self
        st = self.state
        if not cfg.enable or st.finished:
            return
        if step < cfg.reg_start or step > cfg.reg_end:
            return

        if "opacities" not in params:
            raise KeyError("GNS requires params['opacities'] (opacity logits).")

        current_gs_count = int(params["opacities"].numel())

        # Early stop: if we're already close to the budget, force a final prune and finish.
        if step > cfg.reg_start and current_gs_count < float(cfg.final_budget) * 1.05:
            if cfg.verbose:
                print(
                    f"[GNS] Count {current_gs_count} < 1.05 * Budget. "
                    f"Stopping Natural Selection early at step {step}.",
                    flush=True,
                )
                print(
                    f"[GNS] Step {step}: Running Final Budget Prune to {cfg.final_budget}...",
                    flush=True,
                )
            n_pruned = self._final_prune(
                params=params, optimizers=optimizers, strategy_state=strategy_state
            )
            if cfg.verbose:
                print(
                    f"[GNS] Final Prune removed {n_pruned} gaussians. "
                    f"Now having {self._num_gaussians(params)} GSs.",
                    flush=True,
                )
            st.finished = True
            st.stop_step = int(step)
            return

        # Window pruning: periodically remove very transparent Gaussians.
        if step < cfg.reg_end and step % int(cfg.reg_interval) == 0:
            n_pruned = 0
            if current_gs_count > int(cfg.final_budget):
                n_pruned = self._opacity_prune(
                    params=params,
                    optimizers=optimizers,
                    strategy_state=strategy_state,
                    min_opacity=float(cfg.min_opacity),
                )
            if cfg.verbose:
                # Always print on the pruning cadence, even when no Gaussians were pruned,
                # to make it easy to verify that the policy is active.
                print(
                    f"[GNS] Step {step}: Removed {n_pruned} GSs "
                    f"below opacity threshold. Now having {self._num_gaussians(params)} GSs.",
                    flush=True,
                )

        # Final prune at reg_end: enforce the budget via probabilistic survival.
        if step == cfg.reg_end:
            if cfg.verbose:
                print(
                    f"[GNS] Step {step}: Running Final Budget Prune to {cfg.final_budget}...",
                    flush=True,
                )
            n_pruned = self._final_prune(
                params=params, optimizers=optimizers, strategy_state=strategy_state
            )
            if cfg.verbose:
                print(
                    f"[GNS] Final Prune removed {n_pruned} gaussians. "
                    f"Now having {self._num_gaussians(params)} GSs.",
                    flush=True,
                )
            st.finished = True
            if st.stop_step is None:
                st.stop_step = int(step)

    def _opacity_prune(
        self,
        *,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        strategy_state: Dict[str, Any],
        min_opacity: float,
    ) -> int:
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
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        strategy_state: Dict[str, Any],
    ) -> int:
        target_budget = int(self.final_budget)
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
