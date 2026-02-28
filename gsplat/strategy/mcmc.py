import math
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import inject_noise_to_position, relocate, sample_add


@dataclass
class MCMCStrategy(Strategy):
    """Strategy based on "3D Gaussian Splatting as Markov Chain Monte Carlo".

    This implementation matches the public gsplat reference design:
    - Periodically relocate low-opacity ("dead") Gaussians to high-opacity regions.
    - Periodically sample-add new Gaussians from the opacity distribution.
    - Inject position noise each step (optionally stopped by `noise_injection_stop_iter`).
    """

    cap_max: int = 1_000_000
    noise_lr: float = 5e5
    refine_start_iter: int = 500
    refine_stop_iter: int = 25_000
    # Stop injecting noise after this iteration. -1 disables the stop (never stop).
    noise_injection_stop_iter: int = -1
    refine_every: int = 100
    min_opacity: float = 0.005
    verbose: bool = False

    def initialize_state(self, scene_scale: float = 1.0) -> Dict[str, Any]:
        del scene_scale
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        return {"binoms": binoms}

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        super().check_sanity(params, optimizers)
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
        lr: float | None = None,
    ):
        del info, packed
        if lr is None:
            raise ValueError("MCMCStrategy.step_post_backward requires lr!=None.")
        state["binoms"] = state["binoms"].to(params["means"].device)
        binoms: Tensor = state["binoms"]
        # Periodic relocate + sample-add within refine window.
        if (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            n_relocated = self._relocate_gs(params, optimizers, binoms)
            if self.verbose:
                print(f"Step {step}: relocated {n_relocated} GSs.")

            n_new = self._add_new_gs(params, optimizers, binoms)
            if self.verbose:
                print(
                    f"Step {step}: added {n_new} GSs. Now having {len(params['means'])} GSs."
                )

            torch.cuda.empty_cache()

        # Noise injection can continue past refine_stop_iter (gsplat default).
        noise_stop = (
            int(self.noise_injection_stop_iter)
            if int(self.noise_injection_stop_iter) >= 0
            else float("inf")
        )
        if step < noise_stop:
            inject_noise_to_position(
                params=params,
                optimizers=optimizers,
                state={},
                scaler=float(lr) * float(self.noise_lr),
            )

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        opacities = torch.sigmoid(params["opacities"].flatten())
        dead_mask = opacities <= float(self.min_opacity)
        n_dead = int(dead_mask.sum().item())
        if n_dead <= 0:
            return 0
        relocate(
            params=params,
            optimizers=optimizers,
            state={},
            mask=dead_mask,
            binoms=binoms,
            min_opacity=float(self.min_opacity),
        )
        return n_dead

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        current_n = int(len(params["means"]))
        n_target = min(int(self.cap_max), int(1.05 * float(current_n)))
        n_new = max(0, int(n_target - current_n))
        if n_new <= 0:
            return 0
        sample_add(
            params=params,
            optimizers=optimizers,
            state={},
            n=int(n_new),
            binoms=binoms,
            min_opacity=float(self.min_opacity),
        )
        return int(n_new)
