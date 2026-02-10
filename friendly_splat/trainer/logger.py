from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Optional

import torch

from friendly_splat.trainer.configs import IOConfig, TensorBoardConfig

TRAIN_LOG_DROP_KEYS = frozenset({"rgb_l1", "rgb_ssim", "gns"})
EVAL_META_KEYS = frozenset({"step", "train_step"})


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, torch.Tensor):
        if int(value.numel()) != 1:
            return None
        return float(value.detach().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _collect_numeric_scalars(
    *,
    values: Mapping[str, object],
    drop_keys: Collection[str] = (),
    exclude_keys: Collection[str] = (),
) -> dict[str, float]:
    dropped = set(str(key) for key in drop_keys)
    excluded = set(str(key) for key in exclude_keys)
    numeric_scalars: dict[str, float] = {}
    for key, value in values.items():
        key_name = str(key)
        if key_name in dropped or key_name in excluded:
            continue
        scalar = _as_float(value)
        if scalar is None:
            continue
        numeric_scalars[key_name] = float(scalar)
    return numeric_scalars


def _prefix_scalar_names(
    *,
    scalars: Mapping[str, float],
    prefix: str,
) -> dict[str, float]:
    return {f"{prefix}{str(key)}": float(value) for key, value in scalars.items()}


class TensorBoardWriter:
    """Small TensorBoard helper for trainer/eval scalar logging."""

    def __init__(self, *, io_cfg: IOConfig, tb_cfg: TensorBoardConfig) -> None:
        self.enabled = bool(tb_cfg.enable)
        self.every_n = int(tb_cfg.every_n)
        self.flush_every_n = int(tb_cfg.flush_every_n)
        self._writer: Optional[Any] = None

        if not self.enabled:
            return

        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise ImportError(
                "TensorBoard logging requested (tb.enable=True) but dependency is missing. "
                "Install `tensorboard` (or run with --tb.enable False)."
            ) from e

        log_dir = os.path.join(io_cfg.result_dir, "tb")
        self._writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard enabled: {log_dir}", flush=True)

    def log_scalars(
        self,
        *,
        step: int,
        scalars: Mapping[str, object],
        prefix: str = "",
        respect_every_n: bool = False,
        flush: bool = False,
    ) -> bool:
        if not self.enabled or self._writer is None:
            return False
        train_step = int(step) + 1
        if respect_every_n and train_step % int(self.every_n) != 0:
            return False

        wrote_any = False
        for key, value in sorted(scalars.items()):
            scalar = _as_float(value)
            if scalar is None:
                continue
            tag = f"{prefix}{key}" if len(prefix) > 0 else str(key)
            self._writer.add_scalar(tag, float(scalar), train_step)
            wrote_any = True

        if not wrote_any:
            return False
        if flush or (respect_every_n and train_step % int(self.flush_every_n) == 0):
            self._writer.flush()
        return True

    def log_train(
        self,
        *,
        step: int,
        loss_items: Mapping[str, object],
        num_gs: int,
        mem_gb: Optional[float] = None,
    ) -> None:
        train_scalars = _collect_numeric_scalars(values=loss_items)
        train_scalars["num_gs"] = float(int(num_gs))
        if mem_gb is not None:
            train_scalars["mem_gb"] = float(mem_gb)
        self.log_scalars(
            step=int(step),
            scalars=train_scalars,
            prefix="train/",
            respect_every_n=True,
        )

    def log_eval(
        self,
        *,
        step: int,
        stats: Mapping[str, object],
        stage: str = "eval",
    ) -> None:
        self.log_scalars(
            step=int(step),
            scalars=stats,
            prefix=f"{str(stage)}/",
            flush=True,
        )

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.flush()
        self._writer.close()


@dataclass(frozen=True)
class LogPayload:
    train_scalars: dict[str, float]
    eval_metrics: Optional[dict[str, float]]
    step: int


def filter_train_loss_items_for_logging(
    *,
    loss_items: Mapping[str, object],
) -> dict[str, float]:
    """Filter training loss items for TB/viewer logging.

    Rules:
    - Always drop: rgb_l1, rgb_ssim, gns.
    - Keep scalar-like values only (Python numeric or scalar tensor).
    """
    return _collect_numeric_scalars(
        values=loss_items,
        drop_keys=TRAIN_LOG_DROP_KEYS,
    )


def handle_step_logging(
    *,
    step: int,
    device: torch.device,
    num_gs: int,
    train_loss_items: Mapping[str, object],
    eval_stats: Optional[Mapping[str, object]],
    tb_writer: TensorBoardWriter,
) -> LogPayload:
    """Prepare step log payload and write TensorBoard metrics."""
    train_step_index = int(step) + 1
    mem_gb = None
    if device.type == "cuda":
        mem_gb = float(torch.cuda.max_memory_allocated(device=device) / (1024**3))

    train_loss_scalars = filter_train_loss_items_for_logging(
        loss_items=train_loss_items,
    )
    all_train_metrics = dict(train_loss_scalars)
    all_train_metrics["num_gs"] = float(int(num_gs))
    if mem_gb is not None:
        all_train_metrics["mem_gb"] = float(mem_gb)

    tb_writer.log_scalars(
        step=int(step),
        scalars=all_train_metrics,
        prefix="train/",
        respect_every_n=True,
    )
    train_scalars = _prefix_scalar_names(
        scalars=all_train_metrics,
        prefix="train/",
    )

    if eval_stats is None:
        return LogPayload(
            train_scalars=train_scalars,
            eval_metrics=None,
            step=int(train_step_index),
        )

    eval_stats_dict = dict(eval_stats)
    eval_step_index = int(eval_stats_dict.get("step", int(step)))
    tb_writer.log_scalars(
        step=int(eval_step_index),
        scalars=eval_stats_dict,
        prefix="eval/",
        flush=True,
    )
    eval_train_step_index = int(eval_stats_dict.get("train_step", int(eval_step_index) + 1))
    eval_metrics = _collect_numeric_scalars(
        values=eval_stats_dict,
        exclude_keys=EVAL_META_KEYS,
    )
    return LogPayload(
        train_scalars=train_scalars,
        eval_metrics=eval_metrics,
        step=int(eval_train_step_index),
    )
