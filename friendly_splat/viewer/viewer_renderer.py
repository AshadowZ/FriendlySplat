from __future__ import annotations

from typing import Any, Callable, Optional

import torch

from friendly_splat.modules.gaussian import GaussianModel

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from nerfview import apply_float_colormap  # type: ignore
except ImportError:  # pragma: no cover
    apply_float_colormap = None  # type: ignore[assignment]

try:
    from friendly_splat.viewer.gsplat_viewer import GsplatRenderTabState  # type: ignore
except ImportError:  # pragma: no cover
    GsplatRenderTabState = None  # type: ignore[assignment]

try:
    from gsplat.utils import get_implied_normal_from_depth
except ImportError:  # pragma: no cover
    get_implied_normal_from_depth = None  # type: ignore[assignment]

try:
    from friendly_splat.renderer.renderer import render_splats
except ImportError:  # pragma: no cover
    render_splats = None  # type: ignore[assignment]


class ViewerRenderer:
    """Render backend used by ViewerRuntime."""

    def __init__(
        self,
        *,
        device: torch.device,
        gaussian_model: GaussianModel,
        packed: bool = False,
        sparse_grad: bool = False,
        absgrad: bool = False,
        update_counts_fn: Optional[Callable[[Optional[dict[str, torch.Tensor]]], None]] = None,
    ) -> None:
        self.device = device
        self.gaussian_model = gaussian_model
        self.packed = bool(packed)
        self.sparse_grad = bool(sparse_grad)
        self.absgrad = bool(absgrad)
        self._update_counts_fn = update_counts_fn

        self._validate_dependencies()

    @staticmethod
    def _validate_dependencies() -> None:
        missing: list[str] = []
        if np is None:
            missing.append("numpy")
        if apply_float_colormap is None:
            missing.append("nerfview")
        if GsplatRenderTabState is None:
            missing.append("friendly_splat.viewer.gsplat_viewer")
        if get_implied_normal_from_depth is None:
            missing.append("gsplat.utils")
        if render_splats is None:
            missing.append("friendly_splat.renderer.renderer")
        if len(missing) > 0:
            raise ImportError(
                "Viewer renderer dependencies are missing: "
                + ", ".join(missing)
            )

    def _max_sh_degree_supported(self) -> int:
        sh0 = self.gaussian_model.sh0
        shN = self.gaussian_model.shN
        if not isinstance(sh0, torch.Tensor) or not isinstance(shN, torch.Tensor):
            return 0
        total_k = int(sh0.shape[1]) + int(shN.shape[1])
        if total_k <= 0:
            return 0
        degree = int((total_k**0.5) - 1)
        return max(0, degree)

    def _handle_mode_depth(
        self,
        *,
        mode: str,
        out: Any,
        render_tab_state: Any,
        apply_float_colormap: Any,
        height: int,
        width: int,
        np_module: Any,
    ) -> Any:
        if mode == "expected_depth":
            depth = out.expected_depth[0, ..., 0:1] if out.expected_depth is not None else None
        else:
            median = out.meta.get("render_median")
            depth = median[0] if isinstance(median, torch.Tensor) else None
            if depth is None and out.expected_depth is not None:
                depth = out.expected_depth[0, ..., 0:1]

        if depth is None:
            return np_module.zeros((height, width, 3), dtype=np_module.uint8)

        if render_tab_state.normalize_nearfar:
            near_plane = float(render_tab_state.near_plane)
            far_plane = float(render_tab_state.far_plane)
        else:
            near_plane = float(depth.min().item())
            far_plane = float(depth.max().item())

        depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
        depth_norm = depth_norm.clamp(0.0, 1.0)
        if bool(render_tab_state.inverse):
            depth_norm = 1.0 - depth_norm
        return apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()

    def _handle_mode_render_normal(
        self,
        *,
        out: Any,
        height: int,
        width: int,
        np_module: Any,
    ) -> Any:
        normals = out.render_normals[0] if out.render_normals is not None else None
        if normals is None:
            return np_module.zeros((height, width, 3), dtype=np_module.uint8)
        return self._process_normal_map(
            normals=normals,
            np_module=np_module,
        )

    def _handle_mode_surf_normal(
        self,
        *,
        out: Any,
        K: torch.Tensor,
        get_implied_normal_from_depth_fn: Any,
        height: int,
        width: int,
        np_module: Any,
    ) -> Any:
        depth = out.expected_depth
        if depth is None:
            return np_module.zeros((height, width, 3), dtype=np_module.uint8)
        normals = get_implied_normal_from_depth_fn(depth, K).squeeze(0)
        return self._process_normal_map(
            normals=normals,
            np_module=np_module,
        )

    @staticmethod
    def _process_normal_map(*, normals: torch.Tensor, np_module: Any) -> Any:
        mapped = (normals + 1.0) * 0.5
        mapped = 1.0 - mapped
        return (mapped.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(
            np_module.uint8
        )

    @torch.no_grad()
    def render(self, camera_state: Any, render_tab_state: Any):
        assert GsplatRenderTabState is not None
        assert apply_float_colormap is not None
        assert get_implied_normal_from_depth is not None
        assert render_splats is not None
        assert np is not None

        assert isinstance(render_tab_state, GsplatRenderTabState)

        if render_tab_state.preview_render:
            width = int(render_tab_state.render_width)
            height = int(render_tab_state.render_height)
        else:
            width = int(render_tab_state.viewer_width)
            height = int(render_tab_state.viewer_height)

        c2w = torch.from_numpy(np.asarray(camera_state.c2w)).float().to(self.device)
        K = (
            torch.from_numpy(np.asarray(camera_state.get_K((width, height))))
            .float()
            .to(self.device)
        )

        max_degree = self._max_sh_degree_supported()
        active_sh_degree = min(int(render_tab_state.sh_degree), int(max_degree))
        backgrounds = (
            torch.tensor(
                [render_tab_state.backgrounds], device=self.device, dtype=torch.float32
            )
            / 255.0
        )
        render_kwargs = dict(
            gaussian_model=self.gaussian_model,
            camtoworlds=c2w[None],
            Ks=K[None],
            width=width,
            height=height,
            sh_degree=active_sh_degree,
            absgrad=self.absgrad,
            packed=self.packed,
            sparse_grad=self.sparse_grad,
            near_plane=float(render_tab_state.near_plane),
            far_plane=float(render_tab_state.far_plane),
            radius_clip=float(render_tab_state.radius_clip),
            eps2d=float(render_tab_state.eps2d),
            backgrounds=backgrounds,
            rasterize_mode=str(render_tab_state.rasterize_mode),
            camera_model=str(render_tab_state.camera_model),
        )

        mode = str(render_tab_state.render_mode)

        if mode == "rgb":
            out = render_splats(render_mode="RGB", **render_kwargs)
            if self._update_counts_fn is not None:
                self._update_counts_fn(out.meta)
            return out.pred_rgb[0].clamp(0.0, 1.0).detach().cpu().numpy()
        if mode in ("expected_depth", "median_depth"):
            out = render_splats(render_mode="RGB+ED", **render_kwargs)
            if self._update_counts_fn is not None:
                self._update_counts_fn(out.meta)
            return self._handle_mode_depth(
                mode=mode,
                out=out,
                render_tab_state=render_tab_state,
                apply_float_colormap=apply_float_colormap,
                height=height,
                width=width,
                np_module=np,
            )
        if mode == "alpha":
            out = render_splats(render_mode="RGB", **render_kwargs)
            if self._update_counts_fn is not None:
                self._update_counts_fn(out.meta)
            alpha = out.alphas[0, ..., 0:1]
            if bool(render_tab_state.inverse):
                alpha = 1.0 - alpha
            return apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
        if mode == "render_normal":
            out = render_splats(render_mode="RGB+N+ED", **render_kwargs)
            if self._update_counts_fn is not None:
                self._update_counts_fn(out.meta)
            return self._handle_mode_render_normal(
                out=out,
                height=height,
                width=width,
                np_module=np,
            )
        if mode == "surf_normal":
            out = render_splats(render_mode="RGB+ED", **render_kwargs)
            if self._update_counts_fn is not None:
                self._update_counts_fn(out.meta)
            return self._handle_mode_surf_normal(
                out=out,
                K=K,
                get_implied_normal_from_depth_fn=get_implied_normal_from_depth,
                height=height,
                width=width,
                np_module=np,
            )
        raise ValueError(f"Unsupported render mode: {mode}")
