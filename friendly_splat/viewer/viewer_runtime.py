from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import torch


@dataclass
class ViewerRuntime:
    """Small wrapper to keep trainer.py slim.

    This object:
      - optionally starts the viser/nerfview server,
      - provides per-step hooks (pause/lock/update),
      - owns the viewer render callback.
    """

    disable_viewer: bool
    port: int
    device: torch.device
    splats: torch.nn.ParameterDict
    output_dir: Path
    packed: bool = False
    sparse_grad: bool = False
    absgrad: bool = False

    server: Any = None
    viewer: Any = None

    def __post_init__(self) -> None:
        if self.disable_viewer:
            return

        try:
            import viser  # type: ignore

            from friendly_splat.viewer.gsplat_viewer import GsplatViewer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Online viewer requested but dependencies are missing. "
                "Install `viser` and `nerfview` (see friendly_splat/requirements.txt) or run with disable_viewer=True."
            ) from e

        self.server = viser.ViserServer(port=int(self.port), verbose=False)
        self.viewer = GsplatViewer(
            self.server,
            self.render,  # callback
            output_dir=Path(self.output_dir),
            mode="training",
        )

    @property
    def enabled(self) -> bool:
        return self.viewer is not None

    def before_step(self) -> Optional[float]:
        if self.viewer is None:
            return None
        while self.viewer.state == "paused":
            time.sleep(0.01)
        self.viewer.lock.acquire()
        return time.time()

    def after_step(
        self,
        *,
        step: int,
        tic: Optional[float],
        batch_size: int,
        height: int,
        width: int,
        meta: Optional[dict[str, torch.Tensor]] = None,
    ) -> None:
        if self.viewer is None:
            return

        self._update_counts(meta)

        if tic is not None:
            num_train_steps_per_sec = 1.0 / max(time.time() - tic, 1e-10)
            num_train_rays_per_step = int(batch_size) * int(height) * int(width)
            self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            self.viewer.update(int(step), int(num_train_rays_per_step))

        self.viewer.lock.release()

    def complete(self) -> None:
        if self.viewer is None:
            return
        self.viewer.complete()

    def keep_alive(self) -> None:
        """Block the main thread so the viewer server keeps running (Ctrl+C to exit)."""
        if self.viewer is None:
            return
        print("Viewer running... Ctrl+C to exit.", flush=True)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return

    def _update_counts(self, meta: Optional[dict[str, torch.Tensor]]) -> None:
        if self.viewer is None:
            return
        self.viewer.render_tab_state.total_gs_count = int(self.splats["means"].shape[0])
        if meta is None:
            return
        radii = meta.get("radii")
        if not isinstance(radii, torch.Tensor):
            return
        self.viewer.render_tab_state.rendered_gs_count = int(self._count_rendered_gaussians(radii))

    @staticmethod
    def _count_rendered_gaussians(radii: torch.Tensor) -> int:
        # radii shapes differ by packed/unpacked modes:
        # - packed: [nnz, 2]
        # - unpacked: [..., C, N, 2]
        if radii.numel() == 0:
            return 0
        if radii.dim() == 2:
            return int((radii > 0).all(dim=-1).sum().item())
        # Collapse all leading dims except the last (2) and count per-gaussian entries.
        flat = radii.reshape(-1, int(radii.shape[-1]))
        return int((flat > 0).all(dim=-1).sum().item())

    def _max_sh_degree_supported(self) -> int:
        # Infer maximum SH degree from SH coefficient tensor shapes.
        sh0 = self.splats.get("sh0")
        shN = self.splats.get("shN")
        if not isinstance(sh0, torch.Tensor) or not isinstance(shN, torch.Tensor):
            return 0
        total_k = int(sh0.shape[1]) + int(shN.shape[1])
        if total_k <= 0:
            return 0
        # total_k == (degree+1)^2 for SH representation.
        degree = int((total_k ** 0.5) - 1)
        return max(0, degree)

    @torch.no_grad()
    def render(self, camera_state: Any, render_tab_state: Any):
        # Local imports so training without viewer dependencies keeps working.
        import numpy as np  # type: ignore

        from nerfview import apply_float_colormap  # type: ignore

        from friendly_splat.viewer.gsplat_viewer import GsplatRenderTabState  # type: ignore
        from friendly_splat.utils.common_utils import get_implied_normal_from_depth  # noqa: WPS433
        from friendly_splat.renderer.renderer import render_splats  # noqa: WPS433

        assert isinstance(render_tab_state, GsplatRenderTabState)

        if render_tab_state.preview_render:
            width = int(render_tab_state.render_width)
            height = int(render_tab_state.render_height)
        else:
            width = int(render_tab_state.viewer_width)
            height = int(render_tab_state.viewer_height)

        c2w = torch.from_numpy(np.asarray(camera_state.c2w)).float().to(self.device)
        K = torch.from_numpy(np.asarray(camera_state.get_K((width, height)))).float().to(self.device)

        max_degree = self._max_sh_degree_supported()
        active_sh_degree = min(int(render_tab_state.sh_degree), int(max_degree))
        backgrounds = torch.tensor([render_tab_state.backgrounds], device=self.device, dtype=torch.float32) / 255.0

        mode = str(render_tab_state.render_mode)

        def _render(
            *,
            render_mode: Literal["RGB", "RGB+ED", "RGB+N+ED"],
        ):
            return render_splats(
                splats=self.splats,
                camtoworlds=c2w[None],
                Ks=K[None],
                width=width,
                height=height,
                sh_degree=active_sh_degree,
                render_mode=render_mode,
                absgrad=bool(self.absgrad),
                packed=bool(self.packed),
                sparse_grad=bool(self.sparse_grad),
                near_plane=float(render_tab_state.near_plane),
                far_plane=float(render_tab_state.far_plane),
                radius_clip=float(render_tab_state.radius_clip),
                eps2d=float(render_tab_state.eps2d),
                backgrounds=backgrounds,
                rasterize_mode=str(render_tab_state.rasterize_mode),
                camera_model=str(render_tab_state.camera_model),
            )

        if mode == "rgb":
            out = _render(render_mode="RGB")
            self._update_counts(out.meta)
            return out.pred_rgb[0].clamp(0.0, 1.0).detach().cpu().numpy()

        if mode in ("expected_depth", "median_depth"):
            out = _render(render_mode="RGB+ED")
            self._update_counts(out.meta)

            if mode == "expected_depth":
                depth = out.expected_depth[0, ..., 0:1] if out.expected_depth is not None else None
            else:
                median = out.meta.get("render_median")
                depth = median[0] if isinstance(median, torch.Tensor) else None
                if depth is None and out.expected_depth is not None:
                    depth = out.expected_depth[0, ..., 0:1]

            if depth is None:
                return np.zeros((height, width, 3), dtype=np.uint8)

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

        if mode == "alpha":
            out = _render(render_mode="RGB")
            self._update_counts(out.meta)
            alpha = out.alphas[0, ..., 0:1]
            if bool(render_tab_state.inverse):
                alpha = 1.0 - alpha
            return apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()

        if mode == "render_normal":
            out = _render(render_mode="RGB+N+ED")
            self._update_counts(out.meta)
            normals = out.render_normals[0] if out.render_normals is not None else None
            if normals is None:
                return np.zeros((height, width, 3), dtype=np.uint8)
            normals = (normals + 1.0) * 0.5
            normals = 1.0 - normals
            return (normals.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)

        if mode == "surf_normal":
            out = _render(render_mode="RGB+ED")
            self._update_counts(out.meta)
            depth = out.expected_depth
            if depth is None:
                return np.zeros((height, width, 3), dtype=np.uint8)
            normals = get_implied_normal_from_depth(depth, K).squeeze(0)
            normals = (normals + 1.0) * 0.5
            normals = 1.0 - normals
            return (normals.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(np.uint8)

        return np.zeros((height, width, 3), dtype=np.uint8)
