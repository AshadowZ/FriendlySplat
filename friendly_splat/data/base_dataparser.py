from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class DataparserOutputs:
    """Standardized outputs produced by a dataparser.

    This is the "contract" between dataset-specific parsing logic and the training
    code. Datasets read samples on-demand from these outputs.
    """

    # Global (full) image set.
    image_names: List[str]
    image_paths: List[str]
    camtoworlds: np.ndarray  # [N, 4, 4] float32
    Ks: np.ndarray  # [N, 3, 3] float32 (per-image intrinsics)

    # Split selection (indices into the global image set).
    split: str
    indices: np.ndarray  # [M] int64 (global indices)

    # Scene/normalization info.
    scene_scale: float
    transform: np.ndarray  # [4, 4] float32
    scale: float

    # Optional SfM point cloud for initialization.
    points: np.ndarray  # [P, 3] float32 (may be empty)
    points_rgb: np.ndarray  # [P, 3] uint8 (may be empty)

    # Optional per-image prior paths (length N when present).
    depth_paths: Optional[List[str]] = None
    normal_paths: Optional[List[str]] = None
    dynamic_mask_paths: Optional[List[str]] = None
    sky_mask_paths: Optional[List[str]] = None

    # Free-form extra metadata for future dataset types.
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataParser:
    """Base dataparser interface."""

    def get_dataparser_outputs(self, *, split: str) -> DataparserOutputs:  # pragma: no cover
        raise NotImplementedError
