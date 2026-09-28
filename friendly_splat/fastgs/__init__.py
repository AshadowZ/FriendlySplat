"""FastGS: multi-view consistent densification for FriendlySplat (`--fast`).

Integration of `FastGS: Training 3D Gaussian Splatting in 100 Seconds
<https://github.com/fastgs/FastGS>`_ (CVPR 2026). Everything FastGS adds lives in
this folder and is inert unless `--fast` is passed; see `README.md`.
"""

from friendly_splat.fastgs.config import (
    FastGSConfig,
    resolve_importance_thresh,
    resolve_view_cache_size,
)
from friendly_splat.fastgs.optim_schedule import FastGSStepGate
from friendly_splat.fastgs.presets import apply_fastgs_presets
from friendly_splat.fastgs.runtime import FastGSRuntime, build_fastgs_runtime
from friendly_splat.fastgs.scoring import (
    CachedView,
    MultiViewScores,
    ViewCache,
    compute_multiview_scores,
)
from friendly_splat.fastgs.strategy import FastGSStrategy
from friendly_splat.fastgs.validate import validate_fastgs_config

__all__ = [
    "CachedView",
    "FastGSConfig",
    "FastGSRuntime",
    "FastGSStepGate",
    "FastGSStrategy",
    "MultiViewScores",
    "ViewCache",
    "apply_fastgs_presets",
    "build_fastgs_runtime",
    "compute_multiview_scores",
    "resolve_importance_thresh",
    "resolve_view_cache_size",
    "validate_fastgs_config",
]
