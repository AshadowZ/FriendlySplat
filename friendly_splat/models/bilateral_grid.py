from __future__ import annotations

from typing import Any

import torch


class BilateralGridPostProcessor(torch.nn.Module):
    def __init__(
        self,
        *,
        bil_grids: torch.nn.Module,
        slice_fn: Any,
        total_variation_loss_fn: Any,
    ) -> None:
        super().__init__()
        self.bil_grids = bil_grids
        self._slice = slice_fn
        self._total_variation_loss = total_variation_loss_fn
        # [1,H,W,2] in [0,1]
        self.register_buffer("cached_grid_xy", None, persistent=False)
        self.cached_h: int = -1
        self.cached_w: int = -1

    @classmethod
    def create(
        cls, *, num_frames: int, grid_shape: tuple[int, int, int], device: torch.device
    ):
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

    def _grid_xy(self, *, height: int, width: int) -> torch.Tensor:
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
        if image_ids.dim() == 0:
            image_ids = image_ids.unsqueeze(0)
        B, H, W = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])
        grid_xy = self._grid_xy(height=H, width=W).expand(B, -1, -1, -1)
        out = self._slice(self.bil_grids, grid_xy, rgb, image_ids.unsqueeze(-1))  # type: ignore[misc]
        return out["rgb"]

    def tv_loss(self, *, image_ids: torch.Tensor) -> torch.Tensor:
        # Only regularize active grids for this batch.
        if image_ids.dim() == 0:
            image_ids = image_ids.unsqueeze(0)
        unique_ids = torch.unique(image_ids)
        active_grids = self.bil_grids.grids[unique_ids]  # type: ignore[attr-defined]
        return self._total_variation_loss(active_grids)  # type: ignore[misc]
