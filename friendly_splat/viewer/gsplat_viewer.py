from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal, Tuple

import viser
from nerfview import RenderTabState, Viewer


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
    colormap: Literal["turbo", "viridis", "magma", "inferno", "cividis", "gray"] = "turbo"
    rasterize_mode: Literal["classic", "antialiased"] = "classic"
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"


class GsplatViewer(Viewer):
    """Viewer for gsplat-style trainers (viser GUI + nerfview tabs)."""

    def __init__(
        self,
        server: viser.ViserServer,
        render_fn: Callable,
        output_dir: Path,
        mode: Literal["rendering", "training"] = "rendering",
        enable_instance_mode: bool = False,
    ) -> None:
        self._enable_instance_mode = enable_instance_mode
        super().__init__(server, render_fn, output_dir, mode)
        server.gui.set_panel_label("gsplat viewer")

    def _init_rendering_tab(self) -> None:
        self.render_tab_state = GsplatRenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")

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

                @sh_degree_number.on_update
                def _(_) -> None:
                    self.render_tab_state.sh_degree = int(sh_degree_number.value)
                    self.rerender(_)

                near_far_plane_vec2 = server.gui.add_vector2(
                    "Near/Far",
                    initial_value=(self.render_tab_state.near_plane, self.render_tab_state.far_plane),
                    min=(1e-3, 1e1),
                    max=(1e1, 1e3),
                    step=1e-3,
                    hint="Near and far plane for rendering.",
                )

                @near_far_plane_vec2.on_update
                def _(_) -> None:
                    self.render_tab_state.near_plane = near_far_plane_vec2.value[0]
                    self.render_tab_state.far_plane = near_far_plane_vec2.value[1]
                    self.rerender(_)

                radius_clip_slider = server.gui.add_number(
                    "Radius Clip",
                    initial_value=self.render_tab_state.radius_clip,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    hint="2D radius clip (pixels). Gaussians <= clip are skipped.",
                )

                @radius_clip_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.radius_clip = radius_clip_slider.value
                    self.rerender(_)

                eps2d_slider = server.gui.add_number(
                    "2D Epsilon",
                    initial_value=self.render_tab_state.eps2d,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Epsilon added to projected 2D covariances (anti-aliasing).",
                )

                @eps2d_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.eps2d = eps2d_slider.value
                    self.rerender(_)

                backgrounds_slider = server.gui.add_rgb(
                    "Background",
                    initial_value=self.render_tab_state.backgrounds,
                    hint="Background color for rendering.",
                )

                @backgrounds_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.backgrounds = backgrounds_slider.value
                    self.rerender(_)

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
                def _(_) -> None:
                    if render_mode_dropdown.value in ("expected_depth", "median_depth"):
                        normalize_nearfar_checkbox.disabled = False
                        inverse_checkbox.disabled = False
                    else:
                        normalize_nearfar_checkbox.disabled = True
                        inverse_checkbox.disabled = True
                    self.render_tab_state.render_mode = render_mode_dropdown.value
                    self.rerender(_)

                @normalize_nearfar_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.normalize_nearfar = normalize_nearfar_checkbox.value
                    self.rerender(_)

                @inverse_checkbox.on_update
                def _(_) -> None:
                    self.render_tab_state.inverse = inverse_checkbox.value
                    self.rerender(_)

                colormap_dropdown = server.gui.add_dropdown(
                    "Colormap",
                    ("turbo", "viridis", "magma", "inferno", "cividis", "gray"),
                    initial_value=self.render_tab_state.colormap,
                    hint="Colormap used for depth/alpha.",
                )

                @colormap_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.colormap = colormap_dropdown.value
                    self.rerender(_)

                rasterize_mode_dropdown = server.gui.add_dropdown(
                    "Anti-Aliasing",
                    ("classic", "antialiased"),
                    initial_value=self.render_tab_state.rasterize_mode,
                    hint="Rasterization mode.",
                )

                @rasterize_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.rasterize_mode = rasterize_mode_dropdown.value
                    self.rerender(_)

                camera_model_dropdown = server.gui.add_dropdown(
                    "Camera",
                    ("pinhole", "ortho", "fisheye"),
                    initial_value=self.render_tab_state.camera_model,
                    hint="Camera model used for rendering.",
                )

                @camera_model_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.camera_model = camera_model_dropdown.value
                    self.rerender(_)

        self._rendering_tab_handles.update(
            {
                "total_gs_count_number": total_gs_count_number,
                "rendered_gs_count_number": rendered_gs_count_number,
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
        super()._populate_rendering_tab()

    def _after_render(self) -> None:
        # Sync GUI read-only values from state.
        self._rendering_tab_handles["total_gs_count_number"].value = self.render_tab_state.total_gs_count
        self._rendering_tab_handles["rendered_gs_count_number"].value = self.render_tab_state.rendered_gs_count
