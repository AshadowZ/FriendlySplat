from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Dict, Optional, Set

import torch
import yaml

from friendly_splat.models.gaussian import GaussianModel
from friendly_splat.models.postprocess import PostProcessor
from friendly_splat.trainer.configs import (
    EvalConfig,
    IOConfig,
    PoseConfig,
    TrainConfig,
)


def init_output_paths(*, io_cfg: IOConfig, eval_cfg: Optional[EvalConfig] = None) -> None:
    os.makedirs(io_cfg.result_dir, exist_ok=True)

    if io_cfg.save_ckpt:
        os.makedirs(os.path.join(io_cfg.result_dir, "ckpts"), exist_ok=True)

    if io_cfg.export_ply:
        os.makedirs(os.path.join(io_cfg.result_dir, "ply"), exist_ok=True)

    if eval_cfg is not None and bool(eval_cfg.enable):
        os.makedirs(os.path.join(io_cfg.result_dir, "stats"), exist_ok=True)


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


def save_eval_stats(
    *,
    io_cfg: IOConfig,
    eval_cfg: EvalConfig,
    step: int,
    stats: Dict[str, float | int],
) -> str:
    split = str(eval_cfg.split)
    train_step = int(step) + 1
    stats_dir = os.path.join(io_cfg.result_dir, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    out_path = os.path.join(stats_dir, f"{split}_step{train_step:06d}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
    history_path = os.path.join(stats_dir, f"{split}_history.jsonl")
    with open(history_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(stats, sort_keys=True) + "\n")
    print(f"Saved eval stats: {out_path}", flush=True)
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
    postprocessor: Optional[PostProcessor] = None,
) -> str:
    train_step = int(step) + 1  # 1-based step number for user-facing I/O.
    ckpt_path = os.path.join(str(ckpt_dir), f"ckpt_step{train_step:06d}.pt")
    data: Dict[str, object] = {
        "step": int(step),
        "train_step": int(train_step),
        "cfg": asdict(train_cfg),
        "splats": gaussian_model.splats_state_dict(),
    }
    if pose_cfg.pose_opt and pose_adjust is not None:
        data["pose_adjust"] = pose_adjust.state_dict()
    if postprocessor is not None:
        payload = postprocessor.checkpoint_payload()
        if payload is not None:
            key, value = payload
            data[str(key)] = value

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
) -> str:
    train_step = int(step) + 1

    from gsplat import export_splats  # noqa: WPS433

    out_path = os.path.join(str(ply_dir), f"splats_step{int(train_step):06d}.ply")
    with torch.no_grad():
        sh0 = gaussian_model.sh0.detach()
        shN = gaussian_model.shN.detach()
        export_splats(
            means=gaussian_model.means.detach(),
            scales=gaussian_model.log_scales.detach(),  # log-scales (3DGS convention)
            quats=gaussian_model.quats.detach(),
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
    postprocessor: Optional[PostProcessor] = None,
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
            postprocessor=postprocessor,
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
        )
