from __future__ import annotations

from typing import Any

import torch

from friendly_splat.models.postprocess import PostProcessor


class BilateralGridPostProcessor(PostProcessor):
    """Fused bilateral-grid postprocessor with TV regularization.

    Used as a lightweight per-image color/lighting adapter to absorb
    cross-frame ISP and exposure inconsistencies during training.

    Reference:
    Bilateral Guided Radiance Field Processing (arXiv:2406.00448)
    https://arxiv.org/abs/2406.00448
    """

    regularization_name = "bilagrid_tv"

    def __init__(
        self,
        *,
        bil_grids: torch.nn.Module,
        slice_fn: Any,
        total_variation_loss_fn: Any,
        optimizer_lr: float,
        regularization_weight: float,
    ) -> None:
        super().__init__(regularization_weight=regularization_weight)
        self.bil_grids = bil_grids
        self._slice = slice_fn
        self._total_variation_loss = total_variation_loss_fn
        self.optimizer_lr = float(optimizer_lr)
        # [1,H,W,2] in [0,1]
        self.register_buffer("cached_grid_xy", None, persistent=False)
        self.cached_h: int = -1
        self.cached_w: int = -1

    @classmethod
    def create(
        cls,
        *,
        num_frames: int,
        grid_shape: tuple[int, int, int],
        optimizer_lr: float,
        regularization_weight: float,
        device: torch.device,
    ):
        """Create fused bilateral-grid module and wrap it into postprocessor runtime."""
        try:
            from fused_bilagrid import BilateralGrid, slice, total_variation_loss  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "To use bilateral grid, install `fused_bilagrid` and enable --postprocess.use_bilateral_grid."
            ) from e

        grid_x, grid_y, grid_w = (
            int(grid_shape[0]),
            int(grid_shape[1]),
            int(grid_shape[2]),
        )
        bil_grids = BilateralGrid(
            int(num_frames), grid_X=grid_x, grid_Y=grid_y, grid_W=grid_w
        ).to(device)
        return cls(
            bil_grids=bil_grids,
            slice_fn=slice,
            total_variation_loss_fn=total_variation_loss,
            optimizer_lr=optimizer_lr,
            regularization_weight=regularization_weight,
        ).to(device)

    @property
    def device(self) -> torch.device:
        # BilateralGrid usually has trainable tensors; use parameter device as source of truth.
        try:
            return next(self.bil_grids.parameters()).device
        except StopIteration:
            if isinstance(self.cached_grid_xy, torch.Tensor):
                return self.cached_grid_xy.device
            return torch.device("cpu")

    def _get_normalized_grid_xy(self, *, height: int, width: int) -> torch.Tensor:
        """Return cached normalized XY grid in shape [1, H, W, 2]."""
        if (
            self.cached_grid_xy is None
            or self.cached_h != int(height)
            or self.cached_w != int(width)
        ):
            self.cached_h, self.cached_w = int(height), int(width)
            grid_y, grid_x = torch.meshgrid(
                (torch.arange(self.cached_h, device=self.device) + 0.5)
                / float(self.cached_h),
                (torch.arange(self.cached_w, device=self.device) + 0.5)
                / float(self.cached_w),
                indexing="ij",
            )
            self.cached_grid_xy = (
                torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).detach()
            )
        assert isinstance(self.cached_grid_xy, torch.Tensor)
        return self.cached_grid_xy

    def apply(self, *, rgb: torch.Tensor, image_ids: torch.Tensor) -> torch.Tensor:
        # rgb: [B,H,W,3], image_ids: [B]
        # Applies image-conditioned bilateral correction to reduce
        # frame-to-frame photometric mismatch.
        if image_ids.dim() == 0:
            image_ids = image_ids.unsqueeze(0)
        B, H, W = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])
        grid_xy = self._get_normalized_grid_xy(height=H, width=W).expand(B, -1, -1, -1)
        out = self._slice(self.bil_grids, grid_xy, rgb, image_ids.unsqueeze(-1))  # type: ignore[misc]
        return out["rgb"]

    def tv_loss(self, *, image_ids: torch.Tensor) -> torch.Tensor:
        # Only regularize active grids for this batch.
        if image_ids.dim() == 0:
            image_ids = image_ids.unsqueeze(0)
        unique_ids = torch.unique(image_ids)
        active_grids = self.bil_grids.grids[unique_ids]  # type: ignore[attr-defined]
        return self._total_variation_loss(active_grids)  # type: ignore[misc]

    def regularization_loss(
        self, *, image_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        if image_ids is None:
            raise KeyError("Bilateral grid loss requires `image_id` in the batch.")
        return self.tv_loss(image_ids=image_ids)

    def build_optimizers(
        self,
        *,
        batch_size: int,
    ) -> list[torch.optim.Optimizer]:
        # Bilateral grid applies trainer-side LR scaling by sqrt(batch_size).
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=float(self.optimizer_lr) * float(batch_size) ** 0.5,
            eps=1e-15,
        )
        return [optimizer]

    def build_schedulers(
        self,
        *,
        optimizers: list[torch.optim.Optimizer],
        max_steps: int,
    ) -> list[torch.optim.lr_scheduler.LRScheduler]:
        if len(optimizers) == 0:
            raise RuntimeError(
                "Bilateral grid scheduler requires a postprocess optimizer."
            )
        gamma = 0.01 ** (1.0 / float(max_steps))
        return [
            torch.optim.lr_scheduler.ChainedScheduler(
                [
                    torch.optim.lr_scheduler.LinearLR(
                        optimizers[0],
                        start_factor=0.01,
                        total_iters=1000,
                    ),
                    torch.optim.lr_scheduler.ExponentialLR(
                        optimizers[0],
                        gamma=gamma,
                    ),
                ]
            )
        ]

    def checkpoint_payload(self) -> tuple[str, Any]:
        return "bilagrid", self.bil_grids.state_dict()
