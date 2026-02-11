from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from friendly_splat.models.gaussian import GaussianModel
from friendly_splat.trainer.configs import OptimConfig

from gsplat.strategy.natural_selection import NaturalSelectionPolicy


@dataclass
class OptimizerBundle:
    splat_optimizers: Dict[str, torch.optim.Optimizer]
    pose_optimizers: list[torch.optim.Optimizer]
    postprocess_optimizers: list[torch.optim.Optimizer]
    schedulers: list[object]


class OptimizerCoordinator:
    """Own and coordinate optimizer lifecycle for one training step."""

    def __init__(
        self,
        *,
        optim_cfg: OptimConfig,
        device: torch.device,
        gaussian_model: GaussianModel,
        optimizers: OptimizerBundle,
        gns: Optional[NaturalSelectionPolicy],
    ) -> None:
        self.optim_cfg = optim_cfg
        self.device = device
        self.gaussian_model = gaussian_model
        self.optimizers = optimizers
        self.gns = gns
        self._gns_opacity_visibility: Optional[torch.Tensor] = None

    @property
    def splat_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        return self.optimizers.splat_optimizers

    def prepare_step(self, *, step: int) -> None:
        splat_optimizers = self.optimizers.splat_optimizers
        gns = self.gns

        # Apply GNS LR policy on opacity optimizer (scale at start, restore after stop+1000).
        if gns is not None:
            gns.maybe_restore_opacity_lr(step=step, optimizers=splat_optimizers)
            gns.maybe_scale_opacity_lr(step=step, optimizers=splat_optimizers)

    def zero_grad(self) -> None:
        """Clear gradients on all managed optimizers before backprop.

        Uses ``set_to_none=True`` to reduce memory traffic and let PyTorch
        allocate grad tensors lazily on the next backward pass.
        """
        for opt in self.optimizers.splat_optimizers.values():
            opt.zero_grad(set_to_none=True)
        for opt in self.optimizers.pose_optimizers:
            opt.zero_grad(set_to_none=True)
        for opt in self.optimizers.postprocess_optimizers:
            opt.zero_grad(set_to_none=True)

    def step_all(
        self,
        *,
        step: int,
        meta: Dict[str, torch.Tensor],
        batch_size: int,
    ) -> None:
        """Apply one optimizer step across all managed modules.

        This includes:
        - optional dense->sparse grad conversion for packed sparse training;
        - optional visibility-gated updates for SelectiveAdam;
        - GNS-specific opacity handling in the pruning window;
        - stepping all auxiliary optimizers and schedulers.
        """
        optim_cfg = self.optim_cfg
        gaussian_model = self.gaussian_model
        splat_params = gaussian_model.splat_parameters()
        splat_optimizers = self.optimizers.splat_optimizers
        gns = self.gns

        # In packed+sparse mode, convert dense grads to sparse COO so SparseAdam
        # updates only active Gaussian entries.
        if optim_cfg.sparse_grad:
            gaussian_ids = meta.get("gaussian_ids")
            if not isinstance(gaussian_ids, torch.Tensor):
                raise KeyError(
                    "meta['gaussian_ids'] missing; required for sparse_grad in packed mode."
                )
            ids = gaussian_ids.to(dtype=torch.int64)
            is_coalesced = int(batch_size) == 1
            for param in splat_params.values():
                grad = param.grad
                if grad is None or grad.is_sparse:
                    continue
                param.grad = torch.sparse_coo_tensor(
                    indices=ids[None],  # [1, nnz]
                    values=grad[ids],  # [nnz, ...]
                    size=param.size(),
                    is_coalesced=is_coalesced,
                )

        # Build visibility mask for SelectiveAdam (visible-only updates).
        visibility = None
        if optim_cfg.visible_adam:
            opacity_logits = gaussian_model.opacity_logits
            if optim_cfg.packed:
                gaussian_ids = meta.get("gaussian_ids")
                if not isinstance(gaussian_ids, torch.Tensor):
                    raise KeyError(
                        "meta['gaussian_ids'] missing; required for visible_adam in packed mode."
                    )
                visibility = torch.zeros_like(opacity_logits, dtype=torch.bool)
                visibility.scatter_(0, gaussian_ids.to(dtype=torch.int64), True)
            else:
                radii = meta.get("radii")
                if not isinstance(radii, torch.Tensor):
                    raise KeyError(
                        "meta['radii'] missing; required for visible_adam in unpacked mode."
                    )
                vis = radii > 0
                # Handle common shapes: [..., C, N, 2] -> [..., C, N]
                if vis.dim() >= 2 and int(vis.shape[-1]) != int(
                    opacity_logits.shape[0]
                ):
                    vis = vis.all(dim=-1)
                reduce_dims = tuple(range(vis.dim() - 1))
                visibility = vis.any(dim=reduce_dims)

        gns_window_active = (
            gns is not None
            and gns.enable
            and (not gns.state.finished)
            and int(gns.reg_start) <= int(step) <= int(gns.reg_end)
        )

        # Step splat optimizers.
        # During the GNS window, opacities are forced to "fully visible" so the
        # global opacity regularizer can affect every Gaussian.
        for name, opt in splat_optimizers.items():
            if optim_cfg.visible_adam:
                assert visibility is not None
                # During GNS, update *all* opacities regardless of visibility so the global
                # opacity regularizer can actually push Gaussians below the pruning threshold.
                if gns_window_active and name == "opacities":
                    opacity_logits = gaussian_model.opacity_logits
                    if (
                        self._gns_opacity_visibility is None
                        or int(self._gns_opacity_visibility.numel())
                        != int(opacity_logits.numel())
                        or self._gns_opacity_visibility.device
                        != opacity_logits.device
                    ):
                        self._gns_opacity_visibility = torch.ones_like(
                            opacity_logits, dtype=torch.bool
                        )
                    opt.step(self._gns_opacity_visibility)
                else:
                    opt.step(visibility)
            else:
                opt.step()

        # Step non-splat optimizers and then schedulers.
        for opt in self.optimizers.pose_optimizers:
            opt.step()
        for opt in self.optimizers.postprocess_optimizers:
            opt.step()
        for sch in self.optimizers.schedulers:
            sch.step()
