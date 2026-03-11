"""Shared utilities for FriendlySplat.

This package should remain small and focused; prefer placing code in the most
specific subpackage (trainer/data/viewer) unless it is genuinely shared.
"""

from .gaussian_transforms import (
    apply_similarity_transform_to_model_inplace,
    apply_similarity_transform_to_splats_inplace,
    quat_mul_wxyz,
    rotmat_to_quat_wxyz,
    transform_gaussian_tensors,
)

__all__ = [
    "apply_similarity_transform_to_model_inplace",
    "apply_similarity_transform_to_splats_inplace",
    "quat_mul_wxyz",
    "rotmat_to_quat_wxyz",
    "transform_gaussian_tensors",
]
