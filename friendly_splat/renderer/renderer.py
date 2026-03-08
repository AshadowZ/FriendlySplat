from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch

from friendly_splat.modules.gaussian import GaussianModel


@dataclass(frozen=True)
class RenderOutput:
    pred_rgb: torch.Tensor
    alphas: torch.Tensor
    meta: Dict
    expected_depth: Optional[torch.Tensor]
    render_normals: Optional[torch.Tensor]
    active_sh_degree: int


def render_splats(
    *,
    gaussian_model: GaussianModel,
    camtoworlds: torch.Tensor,
    Ks: torch.Tensor,
    width: int,
    height: int,
    sh_degree: int,
    render_mode: Literal["RGB", "RGB+ED", "RGB+N+ED"],
    absgrad: bool = False,
    packed: bool = False,
    sparse_grad: bool = False,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    backgrounds: Optional[torch.Tensor] = None,
    rasterize_mode: Literal["classic", "antialiased"] = "classic",
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
    colors_override: Optional[torch.Tensor] = None,
    sh_degree_override: Optional[int] = None,
) -> RenderOutput:
    from gsplat.rendering import rasterization

    sh_degree = int(
        int(sh_degree_override) if sh_degree_override is not None else int(sh_degree)
    )

    render_tensors = gaussian_model.to_render_tensors(sh_degree=sh_degree)
    means = render_tensors["means"]
    quats = render_tensors["quats"]
    scales = render_tensors["scales"]
    opacities = render_tensors["opacities"]
    sh_coeffs = (
        colors_override.contiguous()
        if isinstance(colors_override, torch.Tensor)
        else render_tensors["colors"]
    )

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=sh_coeffs,
        viewmats=torch.linalg.inv(camtoworlds),  # world-to-camera
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=int(sh_degree),
        packed=bool(packed),
        rasterize_mode=str(rasterize_mode),
        backgrounds=backgrounds,
        render_mode=render_mode,
        sparse_grad=bool(sparse_grad),
        absgrad=absgrad,
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        radius_clip=float(radius_clip),
        eps2d=float(eps2d),
        camera_model=str(camera_model),
    )

    pred_rgb = renders[..., 0:3].clamp(0.0, 1.0)
    need_expected_depth = render_mode in ("RGB+ED", "RGB+N+ED")
    need_render_normals = render_mode == "RGB+N+ED"
    expected_depth = renders[..., -1:].contiguous() if need_expected_depth else None
    render_normals = renders[..., 3:6].contiguous() if need_render_normals else None

    return RenderOutput(
        pred_rgb=pred_rgb,
        alphas=alphas,
        meta=meta,
        expected_depth=expected_depth,
        render_normals=render_normals,
        active_sh_degree=int(sh_degree),
    )
