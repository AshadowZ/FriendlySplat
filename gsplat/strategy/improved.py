from dataclasses import dataclass
from typing import Any, Dict, Union
import math

import torch
from typing_extensions import Literal

from .base import Strategy
from .ops import long_axis_split, remove, reset_opa


@dataclass
class ImprovedStrategy(Strategy):
    """An improved strategy with budget-based Gaussian splitting.

    This strategy is based on the papers:
    `Improving Densification in 3D Gaussian Splatting for High-Fidelity Rendering
    <https://arxiv.org/abs/2508.12313>`

    The strategy will:

    - Periodically split Gaussians along their long axis based on importance sampling
      from high image-plane gradients (subject to a time-varying budget).
    - Periodically prune Gaussians with low opacity.
    - Periodically reset Gaussians to a lower opacity.

    If `absgrad=True`, it uses absolute gradients instead of average gradients for
    splitting, following the AbsGS paper:

    `AbsGS: Recovering Fine Details for 3D Gaussian Splatting <https://arxiv.org/abs/2404.10484>`

    This typically leads to better results but requires setting `grow_grad2d` to a
    higher value (e.g. 0.0008). Also, the `rasterization` function should be
    called with `absgrad=True` so that absolute gradients are computed.

    Args:
        prune_opa: Gaussians with opacity below this value will be pruned.
        grow_grad2d: Gaussians with image-plane gradient above this value will be
            candidates for splitting.
        prune_scale3d: Gaussians with 3D scale (normalized by `scene_scale`) above this
            value will be pruned.
        prune_scale2d: Gaussians with 2D scale (normalized by image resolution) above
            this value will be pruned.
        refine_scale2d_stop_iter: Stop refining Gaussians based on 2D scale after this
            iteration. Set to a positive value to enable this feature.
        refine_start_iter: Start refining Gaussians after this iteration.
        refine_stop_iter: Stop refining Gaussians after this iteration.
        reset_every: Reset opacities every this steps.
        refine_every: Refine Gaussians every this steps.
        max_steps: Total number of training steps (used to trigger an extra prune at
            the final step). Set to 0 if unknown / not used.
        absgrad: Use absolute gradients for splitting.
        verbose: Whether to print verbose information.
        key_for_gradient: Which variable is used for densification.
            3DGS uses "means2d"; 2DGS uses "gradient_2dgs".
        budget: Maximum number of Gaussians allowed (upper bound for the time-varying budget).

    Examples:

        >>> from gsplat import ImprovedStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = ImprovedStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     strategy.step_pre_backward(params, optimizers, strategy_state, step, info)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info)
    """

    prune_opa: float = 0.005
    grow_grad2d: float = 0.0002
    prune_scale3d: float = 0.08
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 4000
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    max_steps: int = 30000
    absgrad: bool = True
    verbose: bool = True
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    budget: int = 1_000_000

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image-plane gradients for each Gaussian.
        # - count: running accum of how many times each Gaussian is visible.
        state: Dict[str, Any] = {"grad2d": None, "count": None, "scene_scale": scene_scale}
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None
        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        .. note::
            It is not required but highly recommended to call this function after
            initializing the strategy to ensure the convention of parameters and
            optimizers is as expected.
        """
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
    ):
        """Callback executed before `loss.backward()`."""
        del params, optimizers, state, step
        assert self.key_for_gradient in info, (
            "The 2D means (or equivalent) of the Gaussians is required but missing."
        )
        info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Callback executed after `loss.backward()`."""
        if step >= self.refine_stop_iter:
            return

        # Update running statistics needed for densification decisions.
        self._update_state(params, state, info, packed=packed)

        if step > self.refine_start_iter and step % self.refine_every == 0:
            # Grow Gaussians.
            n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            if step < self.refine_stop_iter - self.refine_every:
                # Prune Gaussians (skip pruning in the last refinement iteration).
                n_prune = self._prune_gs(params, optimizers, state, step)
                if self.verbose:
                    print(
                        f"Step {step}: {n_prune} GSs pruned. "
                        f"Now having {len(params['means'])} GSs."
                    )
            else:
                if self.verbose:
                    print(f"Step {step}: Skipping pruning in the last refinement iteration.")

            # Reset running stats.
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 and step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 10.0,
            )
            if self.verbose:
                print(
                    f"Step {step}: reset opacities to {self.prune_opa * 10.0}. "
                    f"Now having {len(params['means'])} GSs."
                )

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        """Update running statistics used by the strategy."""
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # Normalize gradients to [-1, 1] screen space.
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # Initialize state on the first run.
        n_gaussian = len(list(params.values())[0])
        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)
        if self.refine_scale2d_stop_iter > 0 and state.get("radii") is None:
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # Update the running state.
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"].max(dim=-1).values  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]

        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32))

        if self.refine_scale2d_stop_iter > 0:
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        """Split Gaussians using a time-varying budget and importance sampling."""
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        is_grad_high = grads > self.grow_grad2d

        # Time-varying budget: grows from 0 to `budget` over the refinement window.
        start_i = self.refine_start_iter
        end_i = self.refine_stop_iter - 500
        den = end_i - start_i
        # Compute rate while avoiding division-by-zero and keeping float precision.
        if den == 0:
            rate = 1.0
        else:
            rate = float((step - start_i) / den)
        # Clamp to [0, 1] to avoid negative or >1 edge cases.
        rate = max(0.0, min(1.0, rate))

        if rate >= 1.0:
            budget = int(self.budget)
        else:
            # Use math.sqrt on the float before scaling with the budget.
            budget = int(math.sqrt(rate) * float(self.budget))

        total_qualified = int(torch.sum(is_grad_high).item())
        curr_points = int(params["means"].shape[0])
        theoretical_max = total_qualified + curr_points
        final_budget = min(budget, theoretical_max)
        new_points_needed = final_budget - curr_points

        # Initialize split mask with False.
        is_split = torch.zeros_like(is_grad_high, dtype=torch.bool, device=device)
        # Create importance scores restricted to high-gradient candidates.
        importance_scores = grads.clone()
        importance_scores[~is_grad_high] = 0.0
        # Ensure non-negative scores and that at least one candidate exists.
        if torch.any(importance_scores > 0):
            num_available = int((importance_scores > 0).sum().item())
            actual_split_count = min(max(new_points_needed, 0), num_available)
            if actual_split_count > 0:
                selected_indices = torch.multinomial(
                    importance_scores, actual_split_count, replacement=False
                )
                is_split[selected_indices] = True

        n_split = int(is_split.sum().item())
        if n_split > 0:
            long_axis_split(params=params, optimizers=optimizers, state=state, mask=is_split)
        return n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        """Prune Gaussians by opacity and (optionally) size."""
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )
            # The official code also implements screen-size pruning but it is often disabled
            # in practice. This implementation keeps it optional via `refine_scale2d_stop_iter`.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d
            is_prune = is_prune | is_too_big

        n_prune = int(is_prune.sum().item())
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)
        return n_prune
