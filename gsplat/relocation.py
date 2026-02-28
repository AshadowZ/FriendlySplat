from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

from .cuda._wrapper import _make_lazy_cuda_func


def compute_relocation(
    opacities: Tensor,  # [N]
    scales: Tensor,  # [N, 3]
    ratios: Tensor,  # [N]
    binoms: Tensor,  # [n_max, n_max]
) -> Tuple[Tensor, Tensor]:
    """Compute new Gaussians from a set of old Gaussians (MCMC relocation).

    This is an implementation of "3D Gaussian Splatting as Markov Chain Monte Carlo"
    (https://arxiv.org/abs/2404.09591), and mirrors the upstream gsplat API.
    """
    N = opacities.shape[0]
    n_max, _ = binoms.shape
    assert scales.shape == (N, 3), scales.shape
    assert ratios.shape == (N,), ratios.shape

    opacities = opacities.contiguous()
    scales = scales.contiguous()
    ratios = ratios.clamp(min=1, max=int(n_max)).int().contiguous()
    binoms = binoms.contiguous()

    new_opacities, new_scales = _make_lazy_cuda_func("relocation")(
        opacities, scales, ratios, binoms, int(n_max)
    )
    return new_opacities, new_scales
