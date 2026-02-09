from __future__ import annotations

import os
from typing import Any, Mapping, Optional

import torch

from friendly_splat.trainer.configs import IOConfig, TensorBoardConfig


def _as_float(value: object) -> Optional[float]:
    if isinstance(value, torch.Tensor):
        if int(value.numel()) != 1:
            return None
        return float(value.detach().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


class TensorBoardRuntime:
    """Small TensorBoard helper for trainer/eval scalar logging."""

    def __init__(self, *, io_cfg: IOConfig, tb_cfg: TensorBoardConfig) -> None:
        self.enabled = bool(tb_cfg.enable)
        self.every_n = int(tb_cfg.every_n)
        self.flush_every_n = int(tb_cfg.flush_every_n)
        self.log_memory = bool(tb_cfg.log_memory)
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

    def log_train(
        self,
        *,
        step: int,
        loss_items: Mapping[str, object],
        num_gs: int,
        lr_means: Optional[float] = None,
        mem_gb: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return
        train_step = int(step) + 1
        if train_step % int(self.every_n) != 0:
            return
        if self._writer is None:
            return

        for key, value in sorted(loss_items.items()):
            scalar = _as_float(value)
            if scalar is None:
                continue
            self._writer.add_scalar(f"train/{key}", scalar, train_step)
        self._writer.add_scalar("train/num_gs", float(num_gs), train_step)
        if lr_means is not None:
            self._writer.add_scalar("train/lr_means", float(lr_means), train_step)
        if mem_gb is not None:
            self._writer.add_scalar("train/mem_gb", float(mem_gb), train_step)

        if train_step % int(self.flush_every_n) == 0:
            self._writer.flush()

    def log_eval(
        self,
        *,
        step: int,
        stats: Mapping[str, object],
        stage: str = "eval",
    ) -> None:
        if not self.enabled or self._writer is None:
            return
        train_step = int(step) + 1
        for key, value in sorted(stats.items()):
            scalar = _as_float(value)
            if scalar is None:
                continue
            self._writer.add_scalar(f"{stage}/{key}", scalar, train_step)
        self._writer.flush()

    def close(self) -> None:
        if self._writer is None:
            return
        self._writer.flush()
        self._writer.close()

