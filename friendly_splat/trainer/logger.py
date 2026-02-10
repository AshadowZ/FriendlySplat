from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping, Optional

import torch

from friendly_splat.trainer.configs import TrainConfig


@dataclass(frozen=True)
class LogPayload:
    train_scalars: dict[str, float]
    eval_metrics: Optional[dict[str, float]]
    step: int


def filter_train_loss_items_for_logging(
    *,
    train_cfg: TrainConfig,
    loss_items: Mapping[str, object],
) -> dict[str, object]:
    """Filter training loss items for TB/viewer logging.

    Rules:
    - Always drop: rgb_l1, rgb_ssim, gns.
    - Keep core items: total, rgb.
    - Keep optional items only when corresponding feature is enabled.
    """
    keep_keys: set[str] = {"total", "rgb"}
    reg_cfg = train_cfg.reg
    post_cfg = train_cfg.postprocess

    if float(reg_cfg.sky_loss_weight) > 0.0:
        keep_keys.add("sky")
    if float(reg_cfg.depth_loss_weight) > 0.0:
        keep_keys.add("depth")
    if float(reg_cfg.normal_loss_weight) > 0.0:
        keep_keys.add("render_normal")
    if float(reg_cfg.surf_normal_loss_weight) > 0.0:
        keep_keys.add("surf_normal")
    if float(reg_cfg.consistency_normal_loss_weight) > 0.0:
        keep_keys.add("consistency_normal")
    if float(reg_cfg.flat_reg_weight) > 0.0:
        keep_keys.add("flat_reg")
    if float(reg_cfg.scale_reg_weight) > 0.0:
        keep_keys.add("scale_reg")

    if bool(post_cfg.use_bilateral_grid):
        keep_keys.add("bilagrid_tv")
    if bool(post_cfg.use_ppisp):
        keep_keys.add("ppisp_reg")

    drop_keys = {"rgb_l1", "rgb_ssim", "gns"}

    filtered: dict[str, object] = {}
    for key, value in loss_items.items():
        if key in drop_keys:
            continue
        if key in keep_keys:
            filtered[key] = value
    return filtered


def extract_prefixed_scalar_dict(
    *,
    values: Mapping[str, object],
    prefix: str,
    exclude_keys: Collection[str] = (),
) -> dict[str, float]:
    """Extract numeric scalars and prefix names for viewer logging."""
    excluded = set(str(x) for x in exclude_keys)
    out: dict[str, float] = {}
    for key, value in values.items():
        key_s = str(key)
        if key_s in excluded:
            continue
        if isinstance(value, torch.Tensor):
            if int(value.numel()) != 1:
                continue
            out[f"{prefix}{key_s}"] = float(value.detach().item())
        elif isinstance(value, (int, float)):
            out[f"{prefix}{key_s}"] = float(value)
    return out


def build_viewer_train_scalars(
    *,
    device: torch.device,
    filtered_loss_items: Mapping[str, object],
    num_gs: int,
) -> dict[str, float]:
    """Build per-step scalar payload for viewer universal plot."""
    train_scalars = extract_prefixed_scalar_dict(
        values=filtered_loss_items,
        prefix="train/",
    )
    train_scalars["train/num_gs"] = float(int(num_gs))
    if device.type == "cuda":
        train_scalars["train/mem_gb"] = float(
            torch.cuda.max_memory_allocated(device=device) / (1024**3)
        )
    return train_scalars


def log_eval_for_step(
    *,
    eval_step: int,
    stats: Mapping[str, object],
    tb_runtime: Any,
) -> None:
    """Write eval stats to TensorBoard."""
    stats_dict = dict(stats)
    tb_runtime.log_eval(
        step=int(eval_step),
        stats=stats_dict,
        stage="eval",
    )


def build_eval_metrics(
    *,
    eval_step: int,
    stats: Mapping[str, object],
) -> tuple[int, dict[str, float]]:
    """Build eval scalar payload and aligned train step."""
    stats_dict = dict(stats)
    eval_metrics = extract_prefixed_scalar_dict(
        values=stats_dict,
        prefix="",
        exclude_keys=("step", "train_step"),
    )
    eval_train_step = int(stats_dict.get("train_step", int(eval_step) + 1))
    return eval_train_step, eval_metrics


def maybe_log_training_scalars_for_step(
    *,
    step: int,
    device: torch.device,
    num_gs: int,
    filtered_loss_items: Mapping[str, object],
    tb_runtime: Any,
) -> None:
    """Collect and write training scalars to TensorBoard (if enabled)."""
    if not bool(getattr(tb_runtime, "enabled", False)):
        return

    every_n = max(1, int(getattr(tb_runtime, "every_n", 1)))
    train_step = int(step) + 1
    if int(train_step) % int(every_n) != 0:
        return

    mem_gb = None
    if device.type == "cuda":
        mem_gb = float(torch.cuda.max_memory_allocated(device=device) / (1024**3))

    tb_runtime.log_train(
        step=int(step),
        loss_items=filtered_loss_items,
        num_gs=int(num_gs),
        mem_gb=mem_gb,
    )


def handle_step_logging(
    *,
    step: int,
    train_cfg: TrainConfig,
    device: torch.device,
    num_gs: int,
    train_loss_items: Mapping[str, object],
    eval_stats: Optional[Mapping[str, object]],
    tb_runtime: Any,
) -> LogPayload:
    """Prepare step log payload and write TensorBoard metrics."""
    train_step = int(step) + 1
    filtered_train_loss_items = filter_train_loss_items_for_logging(
        train_cfg=train_cfg,
        loss_items=train_loss_items,
    )

    maybe_log_training_scalars_for_step(
        step=int(step),
        device=device,
        num_gs=int(num_gs),
        filtered_loss_items=filtered_train_loss_items,
        tb_runtime=tb_runtime,
    )

    train_scalars = build_viewer_train_scalars(
        device=device,
        filtered_loss_items=filtered_train_loss_items,
        num_gs=int(num_gs),
    )

    if eval_stats is None:
        return LogPayload(
            train_scalars=train_scalars,
            eval_metrics=None,
            step=int(train_step),
        )

    eval_stats_dict = dict(eval_stats)
    eval_step = int(eval_stats_dict.get("step", int(step)))
    log_eval_for_step(
        eval_step=eval_step,
        stats=eval_stats_dict,
        tb_runtime=tb_runtime,
    )
    eval_train_step, eval_metrics = build_eval_metrics(
        eval_step=eval_step,
        stats=eval_stats_dict,
    )
    return LogPayload(
        train_scalars=train_scalars,
        eval_metrics=eval_metrics,
        step=int(eval_train_step),
    )
