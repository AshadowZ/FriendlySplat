from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to a 3x3 rotation matrix.

    Implementation follows Zhou et al. "On the Continuity of Rotation Representations
    in Neural Networks" (CVPR 2019), via Gram-Schmidt orthogonalization.
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def apply_pose_adjust(
    *,
    camtoworlds: torch.Tensor,
    image_ids: Optional[torch.Tensor],
    pose_opt: bool,
    pose_adjust: Optional["CameraOptModule"],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply pose optimization transforms."""
    if pose_opt and image_ids is None:
        raise RuntimeError("pose_opt requires image_ids in the dataloader batch.")

    camtoworlds_gt = camtoworlds
    ids = image_ids

    if pose_opt:
        if pose_adjust is None:
            raise RuntimeError("pose_opt=True but pose_adjust is not initialized.")
        if ids is None:
            raise RuntimeError("pose_opt requires image_ids in the dataloader batch.")
        camtoworlds = pose_adjust(camtoworlds, ids)

    return camtoworlds, camtoworlds_gt


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module (per-image SE(3) deltas)."""

    def __init__(self, n: int):
        super().__init__()
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        # Delta position (3D) + Delta rotation (6D).
        self.embeds = torch.nn.Embedding(int(n), 9)
        # Identity rotation in 6D representation.
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32))

    def zero_init(self) -> None:
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float) -> None:
        torch.nn.init.normal_(self.embeds.weight, std=float(std))

    def forward(self, camtoworlds: torch.Tensor, embed_ids: torch.Tensor) -> torch.Tensor:
        """Apply pose deltas to camtoworld matrices.

        Args:
            camtoworlds: (..., 4, 4) camera-to-world matrices.
            embed_ids: (...,) indices into the embedding table.

        Returns:
            Updated camtoworlds: (..., 4, 4)
        """

        if camtoworlds.shape[-2:] != (4, 4):
            raise ValueError(f"camtoworlds must have shape (...,4,4), got {tuple(camtoworlds.shape)}")
        if camtoworlds.shape[:-2] != embed_ids.shape:
            raise ValueError(
                "camtoworlds batch dims must match embed_ids shape, "
                f"got camtoworlds={tuple(camtoworlds.shape)} vs embed_ids={tuple(embed_ids.shape)}"
            )

        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]

        rot = rotation_6d_to_matrix(drot + self.identity.expand(*batch_dims, -1))  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device, dtype=pose_deltas.dtype).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)
