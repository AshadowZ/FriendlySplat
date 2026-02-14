from __future__ import annotations

import torch

from .modules.lpips import LPIPS


def lpips(
    x: torch.Tensor,
    y: torch.Tensor,
    net_type: str = "alex",
    version: str = "0.1",
) -> torch.Tensor:
    r"""Learned Perceptual Image Patch Similarity (LPIPS).

    This is a vendored reference implementation from the original 3DGS codebase
    (graphdeco-inria/gaussian-splatting), which in turn follows the PerceptualSimilarity
    (richzhang/PerceptualSimilarity) weights and conventions.
    """
    device = x.device
    criterion = LPIPS(net_type, version).to(device)
    return criterion(x, y)


__all__ = [
    "LPIPS",
    "lpips",
]

