from __future__ import annotations

import os
import math
from dataclasses import asdict
from typing import Dict, Optional, Set

import torch
import yaml

from friendly_splat.modules.gaussian import GaussianModel
from friendly_splat.modules.bilateral_grid import BilateralGridPostProcessor
from friendly_splat.trainer.configs import (
    IOConfig,
    PoseConfig,
    TrainConfig,
)


def init_output_paths(*, io_cfg: IOConfig) -> None:
    os.makedirs(io_cfg.result_dir, exist_ok=True)

    if io_cfg.save_ckpt:
        os.makedirs(os.path.join(io_cfg.result_dir, "ckpts"), exist_ok=True)

    if io_cfg.export_ply:
        os.makedirs(os.path.join(io_cfg.result_dir, "ply"), exist_ok=True)


def save_train_config_snapshot(
    *,
    io_cfg: IOConfig,
    train_cfg: TrainConfig,
) -> str:
    out_path = os.path.join(io_cfg.result_dir, "cfg.yml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            asdict(train_cfg),
            f,
            sort_keys=False,
            allow_unicode=True,
        )
    print(f"Saved config snapshot: {out_path}", flush=True)
    return out_path


def should_save_checkpoint(
    *,
    io_cfg: IOConfig,
    step: int,
    max_steps: int,
    save_steps: Set[int],
) -> bool:
    if not io_cfg.save_ckpt:
        return False
    train_step = int(step) + 1
    return int(step) == int(max_steps) - 1 or train_step in save_steps


def save_checkpoint(
    *,
    train_cfg: TrainConfig,
    pose_cfg: PoseConfig,
    step: int,
    gaussian_model: GaussianModel,
    ckpt_dir: str,
    pose_adjust: Optional[torch.nn.Module] = None,
    bilateral_grid: Optional[BilateralGridPostProcessor] = None,
) -> str:
    train_step = int(step) + 1  # 1-based step number for user-facing I/O.
    ckpt_path = os.path.join(str(ckpt_dir), f"ckpt_step{train_step:06d}.pt")
    data: Dict[str, object] = {
        "step": int(step),
        "train_step": int(train_step),
        "cfg": asdict(train_cfg),
        # Store splat tensors under canonical keys (means/scales/quats/opacities/sh0/shN).
        # Use a plain dict to keep `viewer.py` checkpoint loading strict and predictable.
        "splats": dict(gaussian_model.splats.state_dict().items()),
    }
    if pose_cfg.pose_opt and pose_adjust is not None:
        data["pose_adjust"] = pose_adjust.state_dict()
    if bilateral_grid is not None:
        data["bilagrid"] = bilateral_grid.bil_grids.state_dict()

    torch.save(data, ckpt_path)
    print(f"Saved checkpoint: {ckpt_path}", flush=True)
    return ckpt_path


def should_export_ply(
    *,
    io_cfg: IOConfig,
    step: int,
    ply_steps: Set[int],
) -> bool:
    if not io_cfg.export_ply:
        return False
    train_step = int(step) + 1
    return int(train_step) in ply_steps


def export_ply(
    *,
    step: int,
    ply_dir: str,
    ply_format: str,
    gaussian_model: GaussianModel,
    active_sh_degree: int,
    scene_transform: Optional[torch.Tensor] = None,
) -> str:
    train_step = int(step) + 1

    from gsplat import export_splats  # noqa: WPS433

    def _rotmat_to_quat_wxyz(rot: torch.Tensor) -> torch.Tensor:
        """Convert a single 3x3 rotation matrix to a wxyz quaternion."""
        r = rot.detach().cpu().double()
        t = float(r[0, 0] + r[1, 1] + r[2, 2])
        if t > 0.0:
            s = math.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = float(r[2, 1] - r[1, 2]) / s
            y = float(r[0, 2] - r[2, 0]) / s
            z = float(r[1, 0] - r[0, 1]) / s
        elif float(r[0, 0]) > float(r[1, 1]) and float(r[0, 0]) > float(r[2, 2]):
            s = math.sqrt(1.0 + float(r[0, 0]) - float(r[1, 1]) - float(r[2, 2])) * 2.0
            w = float(r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = float(r[0, 1] + r[1, 0]) / s
            z = float(r[0, 2] + r[2, 0]) / s
        elif float(r[1, 1]) > float(r[2, 2]):
            s = math.sqrt(1.0 + float(r[1, 1]) - float(r[0, 0]) - float(r[2, 2])) * 2.0
            w = float(r[0, 2] - r[2, 0]) / s
            x = float(r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = float(r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + float(r[2, 2]) - float(r[0, 0]) - float(r[1, 1])) * 2.0
            w = float(r[1, 0] - r[0, 1]) / s
            x = float(r[0, 2] + r[2, 0]) / s
            y = float(r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
        q = torch.tensor([w, x, y, z], dtype=rot.dtype, device=rot.device)
        return q / torch.linalg.norm(q).clamp(min=1e-12)

    def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Quaternion multiplication in wxyz convention: a ⊗ b."""
        aw, ax, ay, az = a.unbind(dim=-1)
        bw, bx, by, bz = b.unbind(dim=-1)
        w = aw * bw - ax * bx - ay * by - az * bz
        x = aw * bx + ax * bw + ay * bz - az * by
        y = aw * by - ax * bz + ay * bw + az * bx
        z = aw * bz + ax * by - ay * bx + az * bw
        return torch.stack([w, x, y, z], dim=-1)

    out_path = os.path.join(str(ply_dir), f"splats_step{int(train_step):06d}.ply")
    with torch.no_grad():
        sh0 = gaussian_model.sh0.detach()
        shN = gaussian_model.shN.detach()

        means = gaussian_model.means.detach()
        log_scales = gaussian_model.log_scales.detach()
        quats = gaussian_model.quats.detach()

        # Default behavior: export PLY in the original (COLMAP) coordinate system.
        # Training may use `normalize_world_space=True`, which applies a similarity
        # transform to cameras/points. Checkpoints stay in that normalized space,
        # but PLY is usually consumed by external tools that expect COLMAP coords.
        if scene_transform is not None:
            T = scene_transform.detach().cpu().to(dtype=torch.float64)
            inv_T = torch.linalg.inv(T).to(device=means.device, dtype=means.dtype)
            inv_A = inv_T[:3, :3]
            inv_t = inv_T[:3, 3]

            means = means @ inv_A.T + inv_t

            col_norms = torch.linalg.norm(inv_A, dim=0)
            length_scale = float(col_norms.mean().item())
            log_scales = log_scales + float(math.log(max(length_scale, 1e-12)))

            rot = inv_A / float(max(length_scale, 1e-12))
            q_rot = _rotmat_to_quat_wxyz(rot)
            quats = _quat_mul_wxyz(q_rot, quats)

        export_splats(
            means=means,
            scales=log_scales,  # log-scales (3DGS convention)
            quats=quats,
            opacities=gaussian_model.opacity_logits.detach(),  # logits (3DGS convention)
            sh0=sh0,
            shN=shN,
            format=str(ply_format),
            save_to=out_path,
        )
    print(f"Saved PLY: {out_path}", flush=True)
    return out_path


def maybe_save_outputs(
    *,
    io_cfg: IOConfig,
    pose_cfg: PoseConfig,
    train_cfg: TrainConfig,
    step: int,
    max_steps: int,
    gaussian_model: GaussianModel,
    active_sh_degree: int,
    pose_adjust: Optional[torch.nn.Module] = None,
    bilateral_grid: Optional[BilateralGridPostProcessor] = None,
    scene_transform: Optional[torch.Tensor] = None,
) -> None:
    ckpt_dir = os.path.join(io_cfg.result_dir, "ckpts")
    ply_dir = os.path.join(io_cfg.result_dir, "ply")
    save_steps = (
        set(int(step_id) for step_id in io_cfg.save_steps)
        if io_cfg.save_ckpt
        else set()
    )
    ply_steps = (
        set(int(step_id) for step_id in io_cfg.ply_steps)
        if io_cfg.export_ply
        else set()
    )
    ply_format = str(io_cfg.ply_format) if io_cfg.export_ply else "ply"

    if should_save_checkpoint(
        io_cfg=io_cfg,
        step=int(step),
        max_steps=int(max_steps),
        save_steps=save_steps,
    ):
        save_checkpoint(
            train_cfg=train_cfg,
            pose_cfg=pose_cfg,
            step=int(step),
            gaussian_model=gaussian_model,
            ckpt_dir=ckpt_dir,
            pose_adjust=pose_adjust,
            bilateral_grid=bilateral_grid,
        )

    if should_export_ply(
        io_cfg=io_cfg,
        step=int(step),
        ply_steps=ply_steps,
    ):
        export_ply(
            step=int(step),
            ply_dir=ply_dir,
            ply_format=ply_format,
            gaussian_model=gaussian_model,
            active_sh_degree=int(active_sh_degree),
            scene_transform=scene_transform,
        )
