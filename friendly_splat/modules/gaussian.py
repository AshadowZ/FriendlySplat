from __future__ import annotations

import math
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

    @property
    def means(self) -> torch.nn.Parameter:
        return self.params["means"]

    @property
    def log_scales(self) -> torch.nn.Parameter:
        # Convention: store log-scales in `scales`.
        return self.params["scales"]

    @property
    def quats(self) -> torch.nn.Parameter:
        return self.params["quats"]

    @property
    def opacity_logits(self) -> torch.nn.Parameter:
        # Convention: store opacity logits in `opacities`.
        return self.params["opacities"]

    @property
    def sh0(self) -> torch.nn.Parameter:
        return self.params["sh0"]

    @property
    def shN(self) -> torch.nn.Parameter:
        return self.params["shN"]

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[0])

    @property
    def scales(self) -> torch.Tensor:
        """3D Gaussian scales in linear space (exp of log-scales)."""
        return torch.exp(self.log_scales)

    @property
    def opacities(self) -> torch.Tensor:
        """3D Gaussian opacities in [0,1] (sigmoid of logits)."""
        return torch.sigmoid(self.opacity_logits)

    @property
    def num_sh_coeffs(self) -> int:
        """Total number of SH coefficients per Gaussian."""
        return 1 + int(self.shN.shape[1])

    @property
    def max_sh_degree(self) -> int:
        """Maximum SH degree supported by the stored SH coefficients."""
        total = int(self.num_sh_coeffs)
        root = math.isqrt(total)
        if root * root != total:
            raise ValueError(
                f"Invalid SH coefficient count: 1+shN={total} is not a perfect square."
            )
        max_degree = int(root) - 1
        if max_degree < 0:
            raise ValueError(
                f"Invalid SH coefficient count: 1+shN={total} yields max_degree={max_degree}."
            )
        return max_degree

    def sh_coeffs(self, *, sh_degree: int) -> torch.Tensor:
        """Return SH coefficients sliced to the active SH degree.

        Args:
            sh_degree: Active SH degree (0 uses only DC term).
        """
        sh_degree = int(sh_degree)
        if sh_degree < 0:
            raise ValueError(f"sh_degree must be >= 0, got {sh_degree}")
        if sh_degree > int(self.max_sh_degree):
            raise ValueError(
                f"sh_degree={sh_degree} exceeds max_sh_degree={self.max_sh_degree} "
                f"(num_sh_coeffs={self.num_sh_coeffs})."
            )
        if sh_degree == 0:
            return self.sh0
        sh_coeffs_full = torch.cat([self.sh0, self.shN], dim=1)
        active_k = (sh_degree + 1) ** 2
        return sh_coeffs_full[:, :active_k, :]

    def to_render_tensors(self, *, sh_degree: int) -> dict[str, torch.Tensor]:
        """Return tensors in the form expected by gsplat rasterization."""
        return {
            "means": self.means,
            "quats": self.quats,
            "scales": self.scales,
            "opacities": self.opacities,
            "colors": self.sh_coeffs(sh_degree=int(sh_degree)),
        }

    def get_param_groups(self) -> dict[str, list[torch.nn.Parameter]]:
        """Return named parameter groups for optimizer construction.

        This method intentionally returns only structural information (group name
        to parameters). Optimizer/scheduler policies live in trainer code.
        """
        return {name: [param] for name, param in self.splat_parameters().items()}

    def splat_parameters(self) -> dict[str, torch.nn.Parameter]:
        """Return the canonical set of trainable splat parameters."""
        return {
            "means": self.means,
            "scales": self.log_scales,
            "quats": self.quats,
            "opacities": self.opacity_logits,
            "sh0": self.sh0,
            "shN": self.shN,
        }

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

        # KNN includes self as nearest neighbor. Use K<=4 then drop [:, 0].
        if n < 2:
            dist_avg = torch.ones((n,), device=points.device, dtype=points.dtype)
        else:
            k = min(4, n)
            dists = knn_distances(points, k=k)
            neighbor_dists = dists[:, 1:] if k > 1 else dists
            if int(neighbor_dists.numel()) == 0:
                dist_avg = torch.ones((n,), device=points.device, dtype=points.dtype)
            else:
                dist2_avg = (neighbor_dists**2).mean(dim=-1)
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
