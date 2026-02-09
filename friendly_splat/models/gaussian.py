from __future__ import annotations

from typing import Dict

import torch

from friendly_splat.utils.common_utils import knn_distances, logit, rgb_to_sh


class GaussianModel(torch.nn.Module):
    """Trainable Gaussian parameters used by 3DGS training."""

    def __init__(self, params: Dict[str, torch.nn.Parameter]) -> None:
        super().__init__()
        self.params = torch.nn.ParameterDict(params)

    @property
    def splats(self) -> torch.nn.ParameterDict:
        return self.params

    def as_parameter_dict(self) -> torch.nn.ParameterDict:
        return self.params

    @classmethod
    def from_sfm(
        cls,
        *,
        points: torch.Tensor,
        points_rgb: torch.Tensor,
        sh_degree: int,
        init_scale: float,
        init_opacity: float,
        device: torch.device,
    ) -> "GaussianModel":
        if points.dim() != 2 or points.shape[-1] != 3:
            raise ValueError(f"points must have shape [N,3], got {tuple(points.shape)}")
        if points_rgb.shape != points.shape:
            raise ValueError(
                f"points_rgb must match points shape [N,3], got {tuple(points_rgb.shape)} vs {tuple(points.shape)}"
            )
        return cls._build_from_points(
            points=points.float(),
            rgbs=(points_rgb.float() / 255.0).clamp(0.0, 1.0),
            sh_degree=int(sh_degree),
            init_scale=float(init_scale),
            init_opacity=float(init_opacity),
            device=device,
        )

    @classmethod
    def from_random(
        cls,
        *,
        num_points: int,
        scene_scale: float,
        init_extent: float,
        sh_degree: int,
        init_scale: float,
        init_opacity: float,
        device: torch.device,
    ) -> "GaussianModel":
        n = int(num_points)
        if n <= 0:
            raise ValueError(f"num_points must be > 0, got {n}")
        extent = float(init_extent) * float(scene_scale)
        points = extent * (torch.rand((n, 3)) * 2.0 - 1.0)
        rgbs = torch.rand((n, 3))
        return cls._build_from_points(
            points=points,
            rgbs=rgbs,
            sh_degree=int(sh_degree),
            init_scale=float(init_scale),
            init_opacity=float(init_opacity),
            device=device,
        )

    @classmethod
    def _build_from_points(
        cls,
        *,
        points: torch.Tensor,
        rgbs: torch.Tensor,
        sh_degree: int,
        init_scale: float,
        init_opacity: float,
        device: torch.device,
    ) -> "GaussianModel":
        n = int(points.shape[0])
        if n <= 0:
            raise ValueError("Gaussian initialization requires at least one point.")

        if sh_degree < 0:
            raise ValueError(f"sh_degree must be >= 0, got {sh_degree}")

        # KNN includes self as nearest neighbor. Use K=4 then drop [:, 0].
        dists = knn_distances(points, k=4)
        dist2_avg = (dists[:, 1:] ** 2).mean(dim=-1)
        dist_avg = torch.sqrt(dist2_avg).clamp(min=1e-8)
        scales = torch.log(dist_avg * float(init_scale)).unsqueeze(-1).repeat(1, 3)

        quats = torch.rand((n, 4))
        opacities = logit(torch.full((n,), float(init_opacity), dtype=torch.float32))

        n_coeff = (sh_degree + 1) ** 2
        sh = torch.zeros((n, n_coeff, 3), dtype=torch.float32)
        sh[:, 0, :] = rgb_to_sh(rgbs)

        params: Dict[str, torch.nn.Parameter] = {
            "means": torch.nn.Parameter(points.to(device)),
            # Convention: store log-scales in `scales`.
            "scales": torch.nn.Parameter(scales.to(device)),
            "quats": torch.nn.Parameter(quats.to(device)),
            "opacities": torch.nn.Parameter(opacities.to(device)),
            "sh0": torch.nn.Parameter(sh[:, :1, :].to(device)),
            "shN": torch.nn.Parameter(sh[:, 1:, :].to(device)),
        }
        return cls(params=params)
