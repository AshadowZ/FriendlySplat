from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import torch


class PostProcessor(torch.nn.Module, ABC):
    """Unified interface for optional train-time RGB postprocessing modules.

    These modules mainly absorb cross-image photometric inconsistencies
    (exposure/lighting/ISP differences), so core splat parameters can focus
    on geometry and view-dependent appearance instead of per-image tone shifts.
    """

    regularization_name: Optional[str] = None

    def __init__(self, *, regularization_weight: float = 1.0) -> None:
        super().__init__()
        self.regularization_weight = float(regularization_weight)

    @abstractmethod
    def apply(self, *, rgb: torch.Tensor, image_ids: torch.Tensor) -> torch.Tensor:
        """Apply postprocess transform to RGB predictions."""
        raise NotImplementedError

    def regularization_loss(
        self, *, image_ids: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Return optional postprocess regularization term (unweighted)."""
        del image_ids
        return None

    def get_regularizer(
        self, *, image_ids: Optional[torch.Tensor] = None
    ) -> Optional["PostprocessRegularizer"]:
        value = self.regularization_loss(image_ids=image_ids)
        if value is None:
            return None
        if self.regularization_name is None:
            raise RuntimeError(
                f"{self.__class__.__name__} returned a regularization loss but has no regularization_name."
            )
        return PostprocessRegularizer(
            name=self.regularization_name,
            value=float(self.regularization_weight) * value,
        )

    def build_optimizers(
        self,
        *,
        batch_size: int,
    ) -> list[torch.optim.Optimizer]:
        # Implementations may or may not use batch_size for LR scaling.
        del batch_size
        return []

    def build_schedulers(
        self,
        *,
        optimizers: list[torch.optim.Optimizer],
        max_steps: int,
    ) -> list[torch.optim.lr_scheduler.LRScheduler]:
        # Implementations may provide custom scheduler stacks.
        del optimizers, max_steps
        return []

    def checkpoint_payload(self) -> Optional[tuple[str, Any]]:
        return None


@dataclass(frozen=True)
class PostprocessRegularizer:
    """Named postprocess regularizer term returned to trainer runtime."""

    name: str
    value: torch.Tensor


def apply_postprocess(
    *,
    pred_rgb: torch.Tensor,
    image_ids: Optional[torch.Tensor],
    postprocessor: Optional[PostProcessor],
) -> torch.Tensor:
    """Apply enabled postprocessor to handle per-image photometric mismatch."""
    if postprocessor is None:
        return pred_rgb
    if image_ids is None:
        raise KeyError("Postprocess requires `image_id` in the batch.")
    return postprocessor.apply(rgb=pred_rgb, image_ids=image_ids)


def get_postprocess_regularizer(
    *,
    postprocessor: Optional[PostProcessor],
    image_ids: Optional[torch.Tensor],
) -> Optional[PostprocessRegularizer]:
    """Fetch weighted postprocess regularizer when available."""
    if postprocessor is None:
        return None
    return postprocessor.get_regularizer(image_ids=image_ids)


def create_postprocessor(
    *,
    use_bilateral_grid: bool,
    use_ppisp: bool,
    bilateral_grid_shape: tuple[int, int, int],
    bilateral_grid_lr: float,
    bilateral_grid_tv_weight: float,
    ppisp_reg_weight: float,
    num_frames: int,
    device: torch.device,
) -> Optional[PostProcessor]:
    """Create one mutually-exclusive photometric adapter (if enabled)."""
    if bool(use_bilateral_grid) and bool(use_ppisp):
        raise ValueError(
            "use_bilateral_grid and use_ppisp are mutually exclusive."
        )
    if bool(use_bilateral_grid):
        from friendly_splat.models.bilateral_grid import BilateralGridPostProcessor

        return BilateralGridPostProcessor.create(
            num_frames=int(num_frames),
            grid_shape=(
                int(bilateral_grid_shape[0]),
                int(bilateral_grid_shape[1]),
                int(bilateral_grid_shape[2]),
            ),
            optimizer_lr=float(bilateral_grid_lr),
            regularization_weight=float(bilateral_grid_tv_weight),
            device=device,
        )
    if bool(use_ppisp):
        from friendly_splat.models.ppisp import PPISPPostProcessor

        return PPISPPostProcessor.create(
            num_frames=int(num_frames),
            regularization_weight=float(ppisp_reg_weight),
            device=device,
        )
    return None
