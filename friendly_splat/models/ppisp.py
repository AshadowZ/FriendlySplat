from __future__ import annotations

from typing import Any

import torch


class PPISPPostProcessor(torch.nn.Module):
    def __init__(
        self,
        *,
        module: torch.nn.Module,
        optimizers: list[torch.optim.Optimizer],
    ) -> None:
        super().__init__()
        self.module = module
        self.optimizers = optimizers
        # [H,W,2] in pixel space (+0.5)
        self.register_buffer("cached_pixel_coords", None, persistent=False)
        self.cached_h: int = -1
        self.cached_w: int = -1

    @classmethod
    def create(cls, *, num_frames: int, device: torch.device):
        try:
            from ppisp import PPISP, PPISPConfig  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "To use PPISP, install the `ppisp` package and enable --postprocess.use_ppisp."
            ) from e

        ppisp_config = PPISPConfig(
            use_controller=False,
            controller_distillation=False,
            controller_activation_ratio=0.0,
        )

        module = PPISP(
            num_cameras=1, num_frames=int(num_frames), config=ppisp_config
        ).to(device)
        optimizers = list(module.create_optimizers())
        return cls(module=module, optimizers=optimizers).to(device)

    @property
    def device(self) -> torch.device:
        try:
            return next(self.module.parameters()).device
        except StopIteration:
            if isinstance(self.cached_pixel_coords, torch.Tensor):
                return self.cached_pixel_coords.device
            return torch.device("cpu")

    def _pixel_coords(self, *, height: int, width: int) -> torch.Tensor:
        if (
            self.cached_pixel_coords is None
            or self.cached_h != int(height)
            or self.cached_w != int(width)
        ):
            self.cached_h, self.cached_w = int(height), int(width)
            pixel_y, pixel_x = torch.meshgrid(
                torch.arange(self.cached_h, device=self.device) + 0.5,
                torch.arange(self.cached_w, device=self.device) + 0.5,
                indexing="ij",
            )
            self.cached_pixel_coords = torch.stack([pixel_x, pixel_y], dim=-1).detach()
        assert isinstance(self.cached_pixel_coords, torch.Tensor)
        return self.cached_pixel_coords

    def apply(self, *, rgb: torch.Tensor, image_ids: torch.Tensor) -> torch.Tensor:
        # "Cheap" batch support: loop over B and call PPISP with B=1 each time.
        if image_ids.dim() == 0:
            image_ids = image_ids.unsqueeze(0)
        B, H, W = int(rgb.shape[0]), int(rgb.shape[1]), int(rgb.shape[2])
        pixel_coords = self._pixel_coords(height=H, width=W)
        outs = []
        for b in range(B):
            frame_idx = int(image_ids[b].item())
            out = self.module(
                rgb=rgb[b : b + 1],
                pixel_coords=pixel_coords,
                resolution=(W, H),
                camera_idx=0,
                frame_idx=frame_idx,
                exposure_prior=None,
            )
            outs.append(out)
        return torch.cat(outs, dim=0)

    def reg_loss(self) -> torch.Tensor:
        if not hasattr(self.module, "get_regularization_loss"):
            return torch.tensor(0.0, device=self.device)
        return self.module.get_regularization_loss()  # type: ignore[no-any-return]
