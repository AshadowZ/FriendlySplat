from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple

import viser
from nerfview import RenderTabState, Viewer
from nerfview.render_panel import populate_general_render_tab


class GsplatRenderTabState(RenderTabState):
    # Non-controllable parameters (updated by trainer).
    total_gs_count: int = 0
    rendered_gs_count: int = 0

    # Controllable parameters (GUI).
    sh_degree: int = 3
    near_plane: float = 1e-2
    far_plane: float = 1e2
    radius_clip: float = 0.0
    eps2d: float = 0.3
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal[
        "rgb",
        "expected_depth",
        "median_depth",
        "alpha",
        "render_normal",
        "surf_normal",
        "instance",
    ] = "rgb"
    normalize_nearfar: bool = False
    inverse: bool = False
    colormap: Literal[
        "turbo", "viridis", "magma", "inferno", "cividis", "gray"
    ] = "turbo"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"


class GsplatViewer(Viewer):
    """Viewer for gsplat-style trainers (viser GUI + nerfview tabs)."""

    HANDLE_TOTAL_GS_COUNT = "total_gs_count_number"
    HANDLE_RENDERED_GS_COUNT = "rendered_gs_count_number"
    HANDLE_VIEWER_RES_SLIDER = "viewer_res_slider"

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        enable_instance_mode: bool = False,
        after_render_hook: Optional[Callable[[], None]] = None,
        after_render_tab_populated_hook: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._enable_instance_mode = enable_instance_mode
        self._after_render_hook = after_render_hook
        self._after_render_tab_populated_hook = after_render_tab_populated_hook
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

    def _init_rendering_tab(self) -> None:
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._render_tabs = self.server.gui.add_tab_group()

        settings_icon = getattr(viser.Icon, "SETTINGS", None)
        chart_icon = getattr(viser.Icon, "CHART_DOTS", None)

        controls_kwargs = {"icon": settings_icon} if settings_icon is not None else {}
        metrics_kwargs = {"icon": chart_icon} if chart_icon is not None else {}

        self.controls_tab = self._render_tabs.add_tab("Controls", **controls_kwargs)
        self.metrics_tab = self._render_tabs.add_tab("Metrics", **metrics_kwargs)

        with self.controls_tab:
            self._rendering_folder = self.server.gui.add_folder("Rendering")

    def _bind_render_state_attr(
        self,
        handle: Any,
        *,
        attr_name: str,
        cast: Optional[Callable[[Any], Any]] = None,
        rerender: bool = True,
    ) -> None:
        @handle.on_update
        def _(_event: Any) -> None:
            value = handle.value
            if cast is not None:
                value = cast(value)
            setattr(self.render_tab_state, attr_name, value)
            if rerender:
                self.rerender(_event)

    def _bind_callback(self, handle: Any, callback: Callable[[], None]) -> None:
        @handle.on_update
        def _(_event: Any) -> None:
            callback()
            self.rerender(_event)

    def _populate_rendering_tab(self) -> None:
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder("Gsplat"):
                total_gs_count_number = server.gui.add_number(
                    "Total",
                    initial_value=self.render_tab_state.total_gs_count,
                    disabled=True,
                    hint="Total number of splats in the scene.",
                )
                rendered_gs_count_number = server.gui.add_number(
                    "Rendered",
                    initial_value=self.render_tab_state.rendered_gs_count,
                    disabled=True,
                    hint="Number of splats rendered.",
                )

                sh_degree_number = server.gui.add_number(
                    "SH Degree",
                    initial_value=self.render_tab_state.sh_degree,
                    min=0,
                    max=5,
                    step=1,
                    hint="Spherical harmonics degree used for shading.",
                )
                self._bind_render_state_attr(
                    sh_degree_number,
                    attr_name="sh_degree",
                    cast=int,
                )

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(
                        self.render_tab_state.near_plane,
                        self.render_tab_state.far_plane,
                    ),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )
                self._bind_callback(
                    near_far_plane_vec2,
                    callback=lambda: (
                        setattr(
                            self.render_tab_state,
                            "near_plane",
                            near_far_plane_vec2.value[0],
                        ),
                        setattr(
                            self.render_tab_state,
                            "far_plane",
                            near_far_plane_vec2.value[1],
                        ),
                    ),
                )

                radius_clip_slider = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip (pixels). Gaussians <= clip are skipped.",
                )
                self._bind_render_state_attr(
                    radius_clip_slider,
                    attr_name="radius_clip",
                    cast=float,
                )

                eps2d_slider = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to projected 2D covariances (anti-aliasing).",
                )
                self._bind_render_state_attr(
                    eps2d_slider,
                    attr_name="eps2d",
                    cast=float,
                )

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )
                self._bind_render_state_attr(
                    backgrounds_slider,
                    attr_name="backgrounds",
                )

                render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    tuple(
                        [
                            "rgb",
                            "expected_depth",
                            "median_depth",
                            "alpha",
                            "render_normal",
                            "surf_normal",
                        ]
                        + (["instance"] if self._enable_instance_mode else [])
                    ),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode.",
                )

                normalize_nearfar_checkbox = server.gui.add_checkbox(
                    "Normalize Near/Far",
                    initial_value=self.render_tab_state.normalize_nearfar,
                    disabled=True,
                    hint="Normalize depth with near/far plane.",
                )
                inverse_checkbox = server.gui.add_checkbox(
                    "Inverse",
                    initial_value=self.render_tab_state.inverse,
                    disabled=True,
                    hint="Invert depth/alpha colormap.",
                )

                @render_mode_dropdown.on_update
                def _(_event: Any) -> None:
                    if render_mode_dropdown.value in ("expected_depth", "median_depth"):
                        normalize_nearfar_checkbox.disabled = False
                        inverse_checkbox.disabled = False
                    else:
                        normalize_nearfar_checkbox.disabled = True
                        inverse_checkbox.disabled = True
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_event)

                self._bind_render_state_attr(
                    normalize_nearfar_checkbox,
                    attr_name="normalize_nearfar",
                    cast=bool,
                )
                self._bind_render_state_attr(
                    inverse_checkbox,
                    attr_name="inverse",
                    cast=bool,
                )

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for depth/alpha.",
                )
                self._bind_render_state_attr(
                    colormap_dropdown,
                    attr_name="colormap",
                )

                rasterize_mode_dropdown = server.gui.add_dropdown(
                    "Anti-Aliasing",
                    ("classic", "antialiased"),
                    initial_value=self.render_tab_state.rasterize_mode,
                    hint="Rasterization mode.",
                )
                self._bind_render_state_attr(
                    rasterize_mode_dropdown,
                    attr_name="rasterize_mode",
                )

                camera_model_dropdown = server.gui.add_dropdown(
                    "Camera",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for rendering.",
                )
                self._bind_render_state_attr(
                    camera_model_dropdown,
                    attr_name="camera_model",
                )

        self._rendering_tab_handles.update(
            {
                self.HANDLE_TOTAL_GS_COUNT: total_gs_count_number,
                self.HANDLE_RENDERED_GS_COUNT: rendered_gs_count_number,
                "near_far_plane_vec2": near_far_plane_vec2,
                "radius_clip_slider": radius_clip_slider,
                "eps2d_slider": eps2d_slider,
                "backgrounds_slider": backgrounds_slider,
                "render_mode_dropdown": render_mode_dropdown,
                "normalize_nearfar_checkbox": normalize_nearfar_checkbox,
                "inverse_checkbox": inverse_checkbox,
                "colormap_dropdown": colormap_dropdown,
                "rasterize_mode_dropdown": rasterize_mode_dropdown,
                "camera_model_dropdown": camera_model_dropdown,
            }
        )

        # Populate the common rendering controls (Render Res / video export / path tools).
        with self._rendering_folder:
            viewer_res_slider = self.server.gui.add_slider(
                "Viewer Res",
                min=64,
                max=2048,
                step=1,
                initial_value=2048,
                hint="Maximum resolution of the viewer rendered image.",
            )
            self._bind_render_state_attr(
                viewer_res_slider,
                attr_name="viewer_res",
                cast=int,
            )

            self._rendering_tab_handles[
                self.HANDLE_VIEWER_RES_SLIDER
            ] = viewer_res_slider

        extra_handles = self._rendering_tab_handles.copy()
        if self.mode == "training":
            extra_handles.update(self._training_tab_handles)
        handles = populate_general_render_tab(
            self.server,
            output_dir=self.output_dir,
            folder=self._rendering_folder,
            render_tab_state=self.render_tab_state,
            extra_handles=extra_handles,
        )
        self._rendering_tab_handles.update(handles)
        if self._after_render_tab_populated_hook is not None:
            self._after_render_tab_populated_hook(self)

    def _after_render(self) -> None:
        # Sync GUI read-only values from state.
        self._rendering_tab_handles[
            self.HANDLE_TOTAL_GS_COUNT
        ].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles[
            self.HANDLE_RENDERED_GS_COUNT
        ].value = self.render_tab_state.rendered_gs_count
        if self._after_render_hook is not None:
            self._after_render_hook()
