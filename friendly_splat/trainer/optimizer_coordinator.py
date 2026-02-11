from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from friendly_splat.modules.gaussian import GaussianModel
from friendly_splat.trainer.configs import OptimConfig

from gsplat.strategy.natural_selection import NaturalSelectionPolicy


@dataclass
class OptimizerBundle:
    splat_optimizers: Dict[str, torch.optim.Optimizer]
    extra_optimizers: Dict[str, torch.optim.Optimizer]
    lr_schedulers: Dict[str, torch.optim.lr_scheduler.LRScheduler]

    @classmethod
    def build_from_param_groups(
        cls,
        *,
        optim_cfg: OptimConfig,
        batch_size: int,
        world_size: int,
        device: torch.device,
        scene_scale: float,
        param_groups: Dict[str, list[torch.nn.Parameter]],
        splat_group_names: set[str],
    ) -> "OptimizerBundle":
        """Build optimizers and schedulers from named parameter groups.

        Policy inputs come from:
        - global optimizer switches: `optim_cfg.sparse_grad` / `optim_cfg.visible_adam`;
        - per-group config: `optim_cfg.optimizers` (nerfstudio-style).
        """
        # Scale learning rate based on batch size, reference:
        # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
        BS = int(batch_size) * int(world_size)
        if BS <= 0:
            raise ValueError(f"batch_size*world_size must be > 0, got {BS}")
        lr_scale = float(BS) ** 0.5

        beta1 = max(0.0, 1.0 - float(BS) * (1.0 - 0.9))
        beta2 = max(0.0, 1.0 - float(BS) * (1.0 - 0.999))
        betas = (beta1, beta2)

        def _make_splat_optimizer(
            *,
            params: list[torch.nn.Parameter],
            lr: float,
            eps: float,
        ) -> torch.optim.Optimizer:
            lr_scaled = float(lr) * lr_scale
            if optim_cfg.sparse_grad:
                return torch.optim.SparseAdam(
                    [{"params": params, "lr": lr_scaled}], betas=betas, eps=float(eps)
                )
            if optim_cfg.visible_adam:
                from gsplat.optimizers import SelectiveAdam  # noqa: WPS433

                return SelectiveAdam(
                    [{"params": params, "lr": lr_scaled}],
                    eps=float(eps),
                    betas=betas,
                )

            if device.type == "cuda":
                try:
                    return torch.optim.Adam(
                        [{"params": params, "lr": lr_scaled}],
                        eps=float(eps),
                        betas=betas,
                        fused=True,
                    )
                except TypeError:
                    pass
            return torch.optim.Adam(
                [{"params": params, "lr": lr_scaled}],
                eps=float(eps),
                betas=betas,
            )

        group_cfgs = optim_cfg.optimizers.as_dict()
        missing = sorted(set(param_groups.keys()) - set(group_cfgs.keys()))
        if len(missing) > 0:
            raise RuntimeError(
                "Optimizer config missing for parameter groups: "
                f"{missing}. Available optimizer group configs: {sorted(group_cfgs.keys())}"
            )

        scene_scale = float(scene_scale)
        splat_optimizers: Dict[str, torch.optim.Optimizer] = {}
        extra_optimizers: Dict[str, torch.optim.Optimizer] = {}
        lr_schedulers: Dict[str, torch.optim.lr_scheduler.LRScheduler] = {}

        for name, params in param_groups.items():
            entry = group_cfgs[name]
            opt_cfg = entry.optimizer
            base_lr = float(opt_cfg.lr)
            lr = base_lr * (scene_scale if bool(entry.lr_mult_scene_scale) else 1.0)

            if name in splat_group_names:
                if len(params) != 1:
                    raise RuntimeError(
                        f"Gaussian splat param group {name!r} must have exactly 1 parameter, got {len(params)}."
                    )
                # Apply BS scaling rule for eps on splat optimizers (legacy behavior).
                eps = float(opt_cfg.eps) / float(lr_scale)
                opt = _make_splat_optimizer(params=params, lr=lr, eps=eps)
                splat_optimizers[name] = opt
            else:
                opt = torch.optim.Adam(
                    params,
                    lr=float(lr) * lr_scale,
                    eps=float(opt_cfg.eps),
                    weight_decay=float(opt_cfg.weight_decay),
                )
                extra_optimizers[name] = opt

            sch_cfg = entry.scheduler
            if sch_cfg is not None:
                max_steps = (
                    int(sch_cfg.max_steps)
                    if sch_cfg.max_steps is not None
                    else int(optim_cfg.max_steps)
                )
                if max_steps <= 0:
                    raise ValueError(f"scheduler.max_steps must be > 0, got {max_steps}")
                ratio = float(sch_cfg.lr_final) / float(base_lr)
                gamma = float(ratio) ** (1.0 / float(max_steps))
                if int(sch_cfg.warmup_steps) > 0:
                    # Use a tiny-but-nonzero warmup start factor to satisfy torch scheduler
                    # constraints.
                    start_factor = 1e-8
                    lr_schedulers[name] = torch.optim.lr_scheduler.ChainedScheduler(
                        [
                            torch.optim.lr_scheduler.LinearLR(
                                opt,
                                start_factor=float(start_factor),
                                total_iters=int(sch_cfg.warmup_steps),
                            ),
                            torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma),
                        ]
                    )
                else:
                    lr_schedulers[name] = torch.optim.lr_scheduler.ExponentialLR(
                        opt, gamma=gamma
                    )

        return cls(
            splat_optimizers=splat_optimizers,
            extra_optimizers=extra_optimizers,
            lr_schedulers=lr_schedulers,
        )


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
        for opt in self.optimizers.extra_optimizers.values():
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
        - stepping all auxiliary optimizers and LR schedulers.
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

        # Step non-splat optimizers and then LR schedulers.
        for opt in self.optimizers.extra_optimizers.values():
            opt.step()
        for sch in self.optimizers.lr_schedulers.values():
            sch.step()
