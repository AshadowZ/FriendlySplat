from __future__ import annotations

"""FastGS's sparse-in-time optimizer schedule.

FastGS stops updating every parameter on every iteration once the scene has
converged (`GaussianModel.optimizer_step` upstream):

- steps 1..`phase1_end`: every group steps every iteration, except the high-order
  SH coefficients, which step every `shN_every` iterations;
- `phase1_end`+1..`phase2_end`: every group steps every `phase2_every` iterations;
- afterwards: every group steps every `phase3_every` iterations.

Gradients are *not* cleared on the iterations a group does not step, so skipped
iterations still contribute (gradient accumulation). This is what makes the tail
of training nearly free, and it is the reason FastGS raises the SH learning rate.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class FastGSStepGate:
    """Decides which splat parameter groups step on a given (0-based) step."""

    phase1_end_step: int = 15_000
    phase2_end_step: int = 20_000
    shN_every: int = 16
    phase2_every: int = 32
    phase3_every: int = 64
    max_steps: int = 30_000
    # Groups throttled during phase 1 (high-order SH by default).
    phase1_gated_groups: Tuple[str, ...] = field(default=("shN",))

    def update_every(self, *, group: str, step: int) -> int:
        """Return the update period (in steps) for `group` at `step`."""
        train_step = int(step) + 1  # 1-based, like upstream FastGS
        if train_step <= int(self.phase1_end_step):
            if str(group) in self.phase1_gated_groups:
                return max(1, int(self.shN_every))
            return 1
        if train_step <= int(self.phase2_end_step):
            return max(1, int(self.phase2_every))
        return max(1, int(self.phase3_every))

    def should_step(self, *, group: str, step: int) -> bool:
        """Whether `group` applies an optimizer step at `step`."""
        every = int(self.update_every(group=str(group), step=int(step)))
        if every <= 1:
            return True
        train_step = int(step) + 1
        if train_step % every == 0:
            return True
        # Always flush the accumulated gradients on the very last step.
        return int(step) == int(self.max_steps) - 1
