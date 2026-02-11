from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Collection, Mapping, Optional

import torch

from friendly_splat.trainer.configs import IOConfig, TensorBoardConfig

TRAIN_LOG_DROP_KEYS = frozenset({"rgb_l1", "rgb_ssim", "gns"})
EVAL_META_KEYS = frozenset({"step", "train_step"})


def _as_float(value: object) -> Optional[float]:
    # Accept Python numerics or scalar tensors only; ignore non-scalars silently.
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
    # Filter a heterogeneous dict (tensors, strings, etc.) down to float scalars.
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
    # Viewer expects already-prefixed names (e.g. "train/loss"), unlike TensorBoard tags.
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

    def _require_enabled(self) -> Any:
        if not self.enabled or self._writer is None:
            raise RuntimeError("TensorBoardWriter is disabled (tb.enable=False).")
        return self._writer

    def is_train_step_due(self, *, step: int, every_n: Optional[int] = None) -> bool:
        """Return True when (step+1) hits the configured cadence.

        This is used as the single source of truth for step-based cadence decisions
        (TensorBoard writes, viewer payload emission, flushing, etc.).
        """
        n = int(self.every_n if every_n is None else every_n)
        if n <= 0:
            return False
        # Cadences are configured in user-facing (1-based) train steps.
        train_step = int(step) + 1
        return (train_step % n) == 0

    def should_log(self, *, step: int, respect_every_n: bool = False) -> bool:
        """Return True if a call to `log_scalars` would write scalars."""
        if not self.enabled or self._writer is None:
            return False
        if not respect_every_n:
            return True
        return self.is_train_step_due(step=int(step))

    def should_flush(self, *, step: int) -> bool:
        if not self.enabled or self._writer is None:
            return False
        return self.is_train_step_due(step=int(step), every_n=int(self.flush_every_n))

    def log_scalars(
        self,
        *,
        step: int,
        scalars: Mapping[str, object],
        prefix: str = "",
        flush: bool = False,
    ) -> bool:
        # Callers must gate with `should_log` first; this method focuses on I/O only.
        writer = self._require_enabled()
        train_step = int(step) + 1

        wrote_any = False
        for key, value in sorted(scalars.items()):
            scalar = _as_float(value)
            if scalar is None:
                continue
            tag = f"{prefix}{key}" if len(prefix) > 0 else str(key)
            writer.add_scalar(tag, float(scalar), train_step)
            wrote_any = True

        if not wrote_any:
            return False
        if flush:
            writer.flush()
        return True

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


def maybe_handle_step_logging(
    *,
    step: int,
    device: torch.device,
    num_gs: int,
    train_loss_items: Mapping[str, object],
    eval_stats: Optional[Mapping[str, object]],
    tb_writer: TensorBoardWriter,
    emit_payload: bool = True,
) -> Optional[LogPayload]:
    """Optionally write TensorBoard metrics and return a viewer log payload.

    When `emit_payload=False`, only TensorBoard side-effects are performed and this
    function returns None.
    """
    train_step_index = int(step) + 1
    train_due = tb_writer.is_train_step_due(step=int(step))
    eval_due = eval_stats is not None

    want_payload = bool(emit_payload) and (bool(train_due) or bool(eval_due))
    want_tb_train = bool(tb_writer.enabled) and bool(train_due)
    want_tb_eval = bool(tb_writer.enabled) and bool(eval_due)

    # Fast-path: nothing to write (TB) and nothing to emit (viewer).
    if not want_payload and not want_tb_train and not want_tb_eval:
        return None

    train_scalars: dict[str, float] = {}
    if train_due and (want_payload or want_tb_train):
        mem_gb = None
        if device.type == "cuda":
            # Useful for profiling; only computed when we actually log/emit payload.
            mem_gb = float(torch.cuda.max_memory_allocated(device=device) / (1024**3))

        train_loss_scalars = _collect_numeric_scalars(
            values=train_loss_items,
            drop_keys=TRAIN_LOG_DROP_KEYS,
        )
        all_train_metrics = dict(train_loss_scalars)
        all_train_metrics["num_gs"] = float(int(num_gs))
        if mem_gb is not None:
            all_train_metrics["mem_gb"] = float(mem_gb)

        if want_tb_train:
            tb_writer.log_scalars(
                step=int(step),
                scalars=all_train_metrics,
                prefix="train/",
                flush=tb_writer.should_flush(step=int(step)),
            )
        if want_payload:
            train_scalars = _prefix_scalar_names(
                scalars=all_train_metrics,
                prefix="train/",
            )

    if eval_stats is None:
        if not want_payload:
            return None
        return LogPayload(
            train_scalars=train_scalars,
            eval_metrics=None,
            step=int(train_step_index),
        )

    eval_stats_dict = dict(eval_stats)
    eval_step_index = int(eval_stats_dict.get("step", int(step)))
    if want_tb_eval:
        tb_writer.log_scalars(
            step=int(eval_step_index),
            scalars=eval_stats_dict,
            prefix="eval/",
            flush=True,
        )
    eval_metrics = (
        _collect_numeric_scalars(
            values=eval_stats_dict,
            exclude_keys=EVAL_META_KEYS,
        )
        if want_payload
        else None
    )
    if not want_payload:
        return None
    return LogPayload(
        train_scalars=train_scalars,
        eval_metrics=eval_metrics,
        # Viewer uses a single monotonically increasing index; use train step.
        step=int(train_step_index),
    )


# Backward-compatible alias (prefer `maybe_handle_step_logging`).
handle_step_logging = maybe_handle_step_logging
