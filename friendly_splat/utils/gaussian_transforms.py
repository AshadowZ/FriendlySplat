from __future__ import annotations

from typing import MutableMapping

import torch

from friendly_splat.modules.gaussian import GaussianModel


def rotmat_to_quat_wxyz(rot: torch.Tensor) -> torch.Tensor:
    """Convert a 3x3 rotation matrix to a unit quaternion in wxyz order."""
    if tuple(rot.shape) != (3, 3):
        raise ValueError(f"Expected rot shape (3, 3), got {tuple(rot.shape)}")

    r = rot.detach().to(device=rot.device, dtype=torch.float64)
    trace = float(r[0, 0] + r[1, 1] + r[2, 2])
    if trace > 0.0:
        s = float(torch.sqrt(torch.tensor(trace + 1.0, dtype=r.dtype))) * 2.0
        w = 0.25 * s
        x = float(r[2, 1] - r[1, 2]) / s
        y = float(r[0, 2] - r[2, 0]) / s
        z = float(r[1, 0] - r[0, 1]) / s
    elif float(r[0, 0]) > float(r[1, 1]) and float(r[0, 0]) > float(r[2, 2]):
        s = float(
            torch.sqrt(
                torch.tensor(
                    1.0 + float(r[0, 0]) - float(r[1, 1]) - float(r[2, 2]),
                    dtype=r.dtype,
                )
            )
        ) * 2.0
        w = float(r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = float(r[0, 1] + r[1, 0]) / s
        z = float(r[0, 2] + r[2, 0]) / s
    elif float(r[1, 1]) > float(r[2, 2]):
        s = float(
            torch.sqrt(
                torch.tensor(
                    1.0 + float(r[1, 1]) - float(r[0, 0]) - float(r[2, 2]),
                    dtype=r.dtype,
                )
            )
        ) * 2.0
        w = float(r[0, 2] - r[2, 0]) / s
        x = float(r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = float(r[1, 2] + r[2, 1]) / s
    else:
        s = float(
            torch.sqrt(
                torch.tensor(
                    1.0 + float(r[2, 2]) - float(r[0, 0]) - float(r[1, 1]),
                    dtype=r.dtype,
                )
            )
        ) * 2.0
        w = float(r[1, 0] - r[0, 1]) / s
        x = float(r[0, 2] + r[2, 0]) / s
        y = float(r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    q = torch.tensor([w, x, y, z], dtype=rot.dtype, device=rot.device)
    return q / torch.linalg.norm(q).clamp(min=1e-12)


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication in wxyz convention: a ⊗ b."""
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    w = aw * bw - ax * bx - ay * by - az * bz
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack([w, x, y, z], dim=-1)


def transform_gaussian_tensors(
    *,
    means: torch.Tensor,
    log_scales: torch.Tensor,
    quats: torch.Tensor,
    transform_src_to_dst: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a similarity transform to Gaussian means/log-scales/quaternions."""
    T = transform_src_to_dst.to(device=means.device, dtype=means.dtype)
    A = T[:3, :3]
    t = T[:3, 3]

    means_out = means @ A.T + t

    col_norms = torch.linalg.norm(A, dim=0)
    length_scale = float(col_norms.mean().item())
    log_length_scale = float(
        torch.log(torch.tensor(max(length_scale, 1e-12), device=A.device)).item()
    )
    log_scales_out = log_scales + log_length_scale

    rot = A / float(max(length_scale, 1e-12))
    q_rot = rotmat_to_quat_wxyz(rot)
    quats_out = quat_mul_wxyz(q_rot.unsqueeze(0), quats)
    quats_out = quats_out / torch.linalg.norm(
        quats_out, dim=-1, keepdim=True
    ).clamp(min=1e-12)

    return means_out, log_scales_out, quats_out


def apply_similarity_transform_to_splats_inplace(
    *,
    splats: MutableMapping[str, torch.Tensor],
    transform_src_to_dst: torch.Tensor,
) -> None:
    """Apply a similarity transform to a splat dict in place."""
    means_out, log_scales_out, quats_out = transform_gaussian_tensors(
        means=splats["means"],
        log_scales=splats["scales"],
        quats=splats["quats"],
        transform_src_to_dst=transform_src_to_dst,
    )
    splats["means"] = means_out
    splats["scales"] = log_scales_out
    splats["quats"] = quats_out


@torch.no_grad()
def apply_similarity_transform_to_model_inplace(
    *,
    gaussian_model: GaussianModel,
    transform_src_to_dst: torch.Tensor,
) -> None:
    """Apply a similarity transform to a GaussianModel in place."""
    means_out, log_scales_out, quats_out = transform_gaussian_tensors(
        means=gaussian_model.means,
        log_scales=gaussian_model.log_scales,
        quats=gaussian_model.quats,
        transform_src_to_dst=transform_src_to_dst,
    )
    gaussian_model.means.copy_(means_out)
    gaussian_model.log_scales.copy_(log_scales_out)
    gaussian_model.quats.copy_(quats_out)
