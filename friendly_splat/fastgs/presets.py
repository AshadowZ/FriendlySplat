from __future__ import annotations

"""FastGS training presets.

`--fast` is a *recipe*, not only a densification strategy: FastGS also retunes a
few learning rates and the densification cadence. Those overrides are applied
here (and printed), on top of whatever the user passed on the command line, so
that `--fast` alone lands on the upstream recipe. Pass
`--fastgs.no-apply-presets` to keep FriendlySplat's own values and take only
FastGS's densification / pruning / optimizer schedule.

This module deliberately imports nothing from `friendly_splat.trainer.configs`
(it only uses `dataclasses.replace`), so the config module can depend on the
FastGS package without an import cycle.
"""

from dataclasses import replace
from typing import Any, List, Tuple


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def apply_fastgs_presets(cfg: Any) -> Any:
    """Return a copy of `cfg` with FastGS's presets applied.

    No-op unless `cfg.fast` and `cfg.fastgs.apply_presets` are both set.
    """
    if not bool(getattr(cfg, "fast", False)):
        return cfg
    fast_cfg = cfg.fastgs
    if not bool(fast_cfg.apply_presets):
        return cfg

    changes: List[Tuple[str, Any, Any]] = []

    # --- densification cadence / gradients -------------------------------
    strategy = cfg.strategy
    if int(strategy.refine_every) != int(fast_cfg.densify_every):
        changes.append(
            ("strategy.refine_every", strategy.refine_every, fast_cfg.densify_every)
        )
        strategy = replace(strategy, refine_every=int(fast_cfg.densify_every))
    if not bool(strategy.absgrad):
        # Splitting uses AbsGS gradients, which the renderer must accumulate.
        changes.append(("strategy.absgrad", strategy.absgrad, True))
        strategy = replace(strategy, absgrad=True)

    # --- learning rates --------------------------------------------------
    optimizers = cfg.optim.optimizers
    preset_lrs = {
        "opacities": float(fast_cfg.opacity_lr),
        "sh0": float(fast_cfg.sh0_lr),
        # FastGS exposes `highfeature_lr` and divides it by 20 for the optimizer.
        "shN": float(fast_cfg.shN_lr) / 20.0,
    }
    updates = {}
    for name, lr in preset_lrs.items():
        entry = getattr(optimizers, name)
        if abs(float(entry.optimizer.lr) - float(lr)) > 1e-12:
            changes.append(
                (f"optim.optimizers.{name}.optimizer.lr", entry.optimizer.lr, lr)
            )
            updates[name] = replace(
                entry, optimizer=replace(entry.optimizer, lr=float(lr))
            )
    if updates:
        optimizers = replace(optimizers, **updates)

    optim = cfg.optim
    if optimizers is not cfg.optim.optimizers:
        optim = replace(optim, optimizers=optimizers)

    if not changes:
        return cfg

    if bool(fast_cfg.verbose):
        print(
            "[FastGS] applying presets (disable with --fastgs.no-apply-presets):",
            flush=True,
        )
        for key, old, new in changes:
            print(f"[FastGS]   {key}: {_fmt(old)} -> {_fmt(new)}", flush=True)

    return replace(cfg, strategy=strategy, optim=optim)
