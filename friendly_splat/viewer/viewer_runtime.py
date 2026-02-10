from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import torch

if TYPE_CHECKING:
    from friendly_splat.data.dataset import InputDataset


class ViewerRuntime:
    """Runtime controller for viewer lifecycle and camera-frustum interactions."""
    _UNIVERSAL_EMPTY_OPTION = "(no data yet)"

    def __init__(
        self,
        *,
        disable_viewer: bool,
        port: int,
        device: torch.device,
        splats: torch.nn.ParameterDict,
        output_dir: Path | str,
        packed: bool = False,
        sparse_grad: bool = False,
        absgrad: bool = False,
        train_dataset: Optional["InputDataset"] = None,
        max_display_cameras: int = 128,
        camera_frustum_scale: float = 0.025,
        show_camera_frustums: bool = True,
        sync_frustums_to_render: bool = True,
        frustum_show_after_static_sec: float = 0.12,
        focus_frustum_on_click: bool = True,
        metrics_max_points: int = 2000,
        scalar_max_points: int = 5000,
    ) -> None:
        self.disable_viewer = bool(disable_viewer)
        self.port = int(port)
        self.device = device
        self.splats = splats
        self.output_dir = Path(output_dir)
        self.packed = bool(packed)
        self.sparse_grad = bool(sparse_grad)
        self.absgrad = bool(absgrad)
        self.train_dataset = train_dataset
        self.max_display_cameras = int(max_display_cameras)
        self.camera_frustum_scale = float(camera_frustum_scale)
        self.show_camera_frustums = bool(show_camera_frustums)
        self.sync_frustums_to_render = bool(sync_frustums_to_render)
        self.frustum_show_after_static_sec = float(frustum_show_after_static_sec)
        self.focus_frustum_on_click = bool(focus_frustum_on_click)
        self.metrics_max_points = max(64, int(metrics_max_points))
        self.scalar_max_points = max(256, int(scalar_max_points))

        self.server: Any = None
        self.viewer: Any = None
        self.camera_handles: dict[int, Any] = {}
        self._frustums_hidden_for_sync = False
        self._last_camera_move_time = 0.0
        self._focused_camera_idx: Optional[int] = None
        self._programmatic_camera_update_until = 0.0
        self._deps_ready = False
        self._np: Any = None
        self._apply_float_colormap: Any = None
        self._render_splats_fn: Any = None
        self._get_implied_normal_from_depth_fn: Any = None
        self._gsplat_render_tab_state_cls: Any = None
        self._metrics_history: dict[str, list[tuple[int, float]]] = {
            "psnr": [],
            "ssim": [],
            "lpips": [],
        }
        self._metrics_plot_handles: dict[str, Any] = {}
        self._metrics_show_checkbox: Any = None
        self._metrics_window_slider: Any = None
        self._metrics_last_value_handles: dict[str, Any] = {}
        self._metrics_latest_train_step: int = 0
        self._scalar_history: dict[str, list[tuple[int, float]]] = {}
        self._scalar_latest_train_step: int = 0
        self._universal_metric_dropdown: Any = None
        self._universal_metric_plot: Any = None
        self._universal_metric_window_slider: Any = None
        self._universal_metric_latest_number: Any = None

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

        self.server = viser.ViserServer(port=self.port, verbose=False)
        self.viewer = GsplatViewer(
            self.server,
            self.render,  # callback
            output_dir=self.output_dir,
            mode="training",
            after_render_hook=self._on_after_render,
            after_render_tab_populated_hook=self._install_render_tab_extras,
        )
        self._init_train_camera_frustums()
        self._install_frustum_sync_callbacks()

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
            self.viewer.render_tab_state.num_train_rays_per_sec = (
                num_train_rays_per_step * num_train_steps_per_sec
            )
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

    def push_eval_metrics(
        self, *, step: int, stats: dict[str, float | int]
    ) -> None:
        """Record one eval point and refresh viewer plots."""
        if self.viewer is None or self.server is None:
            return

        train_step_raw = stats.get("train_step", int(step) + 1)
        train_step = int(train_step_raw)
        self._metrics_latest_train_step = max(
            int(self._metrics_latest_train_step),
            int(train_step),
        )
        updated = False
        for metric_name in ("psnr", "ssim", "lpips"):
            value = self._to_float_scalar(stats.get(metric_name))
            if value is None:
                continue
            series = self._metrics_history[metric_name]
            series.append((train_step, float(value)))
            max_points = int(self.metrics_max_points)
            if len(series) > max_points:
                del series[: len(series) - max_points]
            updated = True

        if not updated:
            return

        self._refresh_metric_plots()

    def log_scalars(self, *, step: int, scalars: dict[str, object]) -> None:
        """Record arbitrary scalar streams for universal metric plotting."""
        if self.viewer is None or self.server is None:
            return

        train_step = max(0, int(step))
        if train_step > 0:
            self._scalar_latest_train_step = max(
                int(self._scalar_latest_train_step),
                train_step,
            )

        any_updated = False
        for key, raw_value in scalars.items():
            if not isinstance(key, str) or len(key.strip()) == 0:
                continue
            scalar = self._to_float_scalar(raw_value)
            if scalar is None:
                continue

            tag = key.strip()
            series = self._scalar_history.setdefault(tag, [])
            series.append((train_step, float(scalar)))
            max_points = int(self.scalar_max_points)
            if len(series) > max_points:
                del series[: len(series) - max_points]
            any_updated = True

        if not any_updated:
            return

        self._update_universal_metric_dropdown_options()
        self._refresh_universal_metric_plot()

    @staticmethod
    def _to_float_scalar(value: object) -> Optional[float]:
        if isinstance(value, torch.Tensor):
            if int(value.numel()) != 1:
                return None
            return float(value.detach().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _pick_drawn_camera_indices(self, total_num: int) -> list[int]:
        if total_num <= 0:
            return []
        max_display = int(self.max_display_cameras)
        if max_display <= 0 or max_display >= total_num:
            return list(range(total_num))

        import numpy as np  # type: ignore

        return np.linspace(
            0, total_num - 1, max_display, dtype=np.int32
        ).tolist()

    def _set_camera_frustums_visible(self, visible: bool) -> None:
        if self.server is None:
            return
        with self.server.atomic():
            for handle in self.camera_handles.values():
                handle.visible = bool(visible)

    def _apply_frustum_visibility_state(self) -> None:
        if len(self.camera_handles) == 0:
            return
        if not self.show_camera_frustums:
            self._set_camera_frustums_visible(False)
            return
        if self._frustums_hidden_for_sync:
            self._set_camera_frustums_visible(False)
            return
        if self._focused_camera_idx is not None:
            focused_idx = int(self._focused_camera_idx)
            if self.server is None:
                return
            with self.server.atomic():
                for cam_idx, handle in self.camera_handles.items():
                    handle.visible = int(cam_idx) == focused_idx
            return
        self._set_camera_frustums_visible(True)

    def _install_render_tab_extras(self, viewer: Any) -> None:
        self._install_frustum_visibility_toggle(viewer)
        self._install_metrics_panel(viewer)

    def _install_frustum_visibility_toggle(self, viewer: Any) -> None:
        if self.server is None:
            return
        toggle_order = None
        viewer_res_key = getattr(
            viewer,
            "HANDLE_VIEWER_RES_SLIDER",
            "viewer_res_slider",
        )
        viewer_res_handle = viewer._rendering_tab_handles.get(viewer_res_key)
        if viewer_res_handle is not None:
            try:
                toggle_order = float(viewer_res_handle.order) - 0.01
            except Exception:
                toggle_order = None
        with viewer._rendering_folder:
            toggle = self.server.gui.add_checkbox(
                "Show Frustums",
                initial_value=bool(self.show_camera_frustums),
                hint="Show or hide all camera frustums.",
                order=toggle_order,
            )

        @toggle.on_update
        def _(_: Any) -> None:
            self.show_camera_frustums = bool(toggle.value)
            if self.show_camera_frustums:
                self._frustums_hidden_for_sync = False
            self._apply_frustum_visibility_state()

    def _install_metrics_panel(self, viewer: Any) -> None:
        if self.server is None:
            return
        try:
            import numpy as np  # type: ignore
            from viser import uplot  # type: ignore
        except Exception:
            return

        default_x = np.asarray([0.0, 1.0], dtype=np.float32)
        default_y = np.asarray([0.0, 0.0], dtype=np.float32)
        metric_specs = (
            ("psnr", "PSNR", "#00c389"),
            ("ssim", "SSIM", "#ff8c42"),
            ("lpips", "LPIPS", "#5b8ff9"),
        )

        def _populate_metrics_controls() -> None:
            self._metrics_show_checkbox = self.server.gui.add_checkbox(
                "Show Metric Plots",
                initial_value=True,
                hint="Show or hide PSNR/SSIM/LPIPS curves (eval updates only).",
            )
            self._metrics_window_slider = self.server.gui.add_slider(
                "Window (points)",
                min=0,
                max=int(self.metrics_max_points),
                step=10,
                initial_value=0,
                hint="0 means full history; otherwise show only the latest N points.",
            )

            for metric_name, metric_title, metric_color in metric_specs:
                self._metrics_last_value_handles[metric_name] = self.server.gui.add_number(
                    f"{metric_title} (latest)",
                    initial_value=0.0,
                    disabled=True,
                )
                self._metrics_plot_handles[metric_name] = self.server.gui.add_uplot(
                    data=(default_x, default_y),
                    series=(
                        uplot.Series(label="step", show=False),
                        uplot.Series(
                            label=metric_title,
                            stroke=metric_color,
                            width=2.0,
                        ),
                    ),
                    title=f"{metric_title} vs Train Step",
                    scales={
                        "x": uplot.Scale(
                            time=False,
                            auto=False,
                            range=(0.0, 1.0),
                        ),
                    },
                    aspect=2.0,
                )

        metrics_tab = getattr(viewer, "metrics_tab", None)
        if metrics_tab is not None:
            with metrics_tab:
                self._install_universal_plot_panel(np_module=np, uplot_module=uplot)
                _populate_metrics_controls()
        else:
            with viewer._rendering_folder:
                with self.server.gui.add_folder("Universal Plot"):
                    self._install_universal_plot_panel(np_module=np, uplot_module=uplot)
                with self.server.gui.add_folder("Eval Metrics"):
                    _populate_metrics_controls()

        @self._metrics_show_checkbox.on_update
        def _(_: Any) -> None:
            self._apply_metrics_visibility()

        @self._metrics_window_slider.on_update
        def _(_: Any) -> None:
            self._refresh_metric_plots()

        self._apply_metrics_visibility()
        self._refresh_metric_plots()
        self._update_universal_metric_dropdown_options()
        self._refresh_universal_metric_plot()

    def _install_universal_plot_panel(self, *, np_module: Any, uplot_module: Any) -> None:
        self._universal_metric_dropdown = self.server.gui.add_dropdown(
            "Select Metric",
            (self._UNIVERSAL_EMPTY_OPTION,),
            initial_value=self._UNIVERSAL_EMPTY_OPTION,
            hint="Select any runtime scalar stream to visualize.",
        )
        self._universal_metric_window_slider = self.server.gui.add_slider(
            "Universal Window (points)",
            min=0,
            max=int(self.scalar_max_points),
            step=10,
            initial_value=0,
            hint="0 means full history; otherwise show only the latest N points.",
        )
        self._universal_metric_latest_number = self.server.gui.add_number(
            "Selected Metric (latest)",
            initial_value=0.0,
            disabled=True,
        )
        self._universal_metric_plot = self.server.gui.add_uplot(
            data=(
                np_module.asarray([0.0, 1.0], dtype=np_module.float32),
                np_module.asarray([0.0, 0.0], dtype=np_module.float32),
            ),
            series=(
                uplot_module.Series(label="step", show=False),
                uplot_module.Series(
                    label="value",
                    stroke="#7aa2ff",
                    width=2.0,
                ),
            ),
            title="Universal Metric Plot",
            scales={
                "x": uplot_module.Scale(
                    time=False,
                    auto=False,
                    range=(0.0, 1.0),
                ),
            },
            aspect=2.0,
        )

        @self._universal_metric_dropdown.on_update
        def _(_: Any) -> None:
            self._refresh_universal_metric_plot()

        @self._universal_metric_window_slider.on_update
        def _(_: Any) -> None:
            self._refresh_universal_metric_plot()

    def _update_universal_metric_dropdown_options(self) -> None:
        dropdown = self._universal_metric_dropdown
        if dropdown is None:
            return

        keys = sorted(self._scalar_history.keys())
        if len(keys) == 0:
            options = (self._UNIVERSAL_EMPTY_OPTION,)
            dropdown.options = options
            dropdown.value = self._UNIVERSAL_EMPTY_OPTION
            return

        options = tuple(keys)
        if tuple(dropdown.options) != options:
            dropdown.options = options
        if dropdown.value not in self._scalar_history:
            dropdown.value = options[0]

    def _get_universal_metric_view_data(
        self, metric_name: str
    ) -> tuple["np.ndarray", "np.ndarray"]:
        import numpy as np  # type: ignore

        history = self._scalar_history.get(metric_name, [])
        if len(history) == 0:
            return (
                np.asarray([0.0, 1.0], dtype=np.float32),
                np.asarray([0.0, 0.0], dtype=np.float32),
            )

        window = 0
        if self._universal_metric_window_slider is not None:
            window = int(self._universal_metric_window_slider.value)
        if window > 0 and len(history) > window:
            history = history[-window:]

        x = np.asarray([float(step) for step, _ in history], dtype=np.float32)
        y = np.asarray([float(value) for _, value in history], dtype=np.float32)

        # Keep UI responsive when a metric accumulates many points.
        max_plot_points = 2000
        if int(x.shape[0]) > max_plot_points:
            indices = np.linspace(0, int(x.shape[0]) - 1, max_plot_points, dtype=np.int64)
            x = x[indices]
            y = y[indices]
        return x, y

    def _refresh_universal_metric_plot(self) -> None:
        if self.server is None or self._universal_metric_plot is None:
            return
        dropdown = self._universal_metric_dropdown
        if dropdown is None:
            return
        selected = str(dropdown.value)
        current_step = max(1, int(self._scalar_latest_train_step))

        with self.server.atomic():
            if selected not in self._scalar_history:
                import numpy as np  # type: ignore

                self._universal_metric_plot.data = (
                    np.asarray([0.0, 1.0], dtype=np.float32),
                    np.asarray([0.0, 0.0], dtype=np.float32),
                )
                self._universal_metric_plot.title = "Universal Metric Plot"
                self._universal_metric_plot.scales = {
                    "x": {
                        "time": False,
                        "auto": False,
                        "range": (0.0, float(current_step)),
                    }
                }
                if self._universal_metric_latest_number is not None:
                    self._universal_metric_latest_number.value = 0.0
                return

            x, y = self._get_universal_metric_view_data(selected)
            self._universal_metric_plot.data = (x, y)
            self._universal_metric_plot.title = f"{selected} vs Train Step"
            self._universal_metric_plot.scales = {
                "x": {
                    "time": False,
                    "auto": False,
                    "range": (0.0, float(current_step)),
                }
            }
            if self._universal_metric_latest_number is not None:
                self._universal_metric_latest_number.value = (
                    float(y[-1]) if int(y.shape[0]) > 0 else 0.0
                )

    def _apply_metrics_visibility(self) -> None:
        visible = True
        if self._metrics_show_checkbox is not None:
            visible = bool(self._metrics_show_checkbox.value)
        for handle in self._metrics_plot_handles.values():
            handle.visible = visible
        for handle in self._metrics_last_value_handles.values():
            handle.visible = visible
        if self._metrics_window_slider is not None:
            self._metrics_window_slider.visible = visible

    def _get_metric_view_data(
        self, metric_name: str
    ) -> tuple["np.ndarray", "np.ndarray"]:
        import numpy as np  # type: ignore

        history = self._metrics_history.get(metric_name, [])
        if len(history) == 0:
            return (
                np.asarray([0.0, 1.0], dtype=np.float32),
                np.asarray([0.0, 0.0], dtype=np.float32),
            )

        window = 0
        if self._metrics_window_slider is not None:
            window = int(self._metrics_window_slider.value)
        if window > 0 and len(history) > window:
            history = history[-window:]

        x = np.asarray([float(step) for step, _ in history], dtype=np.float32)
        y = np.asarray([float(value) for _, value in history], dtype=np.float32)
        return x, y

    def _refresh_metric_plots(self) -> None:
        if self.server is None or len(self._metrics_plot_handles) == 0:
            return
        current_step = max(1, int(self._metrics_latest_train_step))
        with self.server.atomic():
            for metric_name, plot_handle in self._metrics_plot_handles.items():
                x, y = self._get_metric_view_data(metric_name)
                plot_handle.data = (x, y)
                plot_handle.scales = {
                    "x": {
                        "time": False,
                        "auto": False,
                        "range": (0.0, float(current_step)),
                    }
                }
                latest = float(y[-1]) if int(y.shape[0]) > 0 else 0.0
                latest_handle = self._metrics_last_value_handles.get(metric_name)
                if latest_handle is not None:
                    latest_handle.value = latest

    def _on_after_render(self) -> None:
        if not self.sync_frustums_to_render:
            return
        if not self._frustums_hidden_for_sync:
            return
        static_delay = max(float(self.frustum_show_after_static_sec), 0.0)
        if time.time() - float(self._last_camera_move_time) < static_delay:
            return
        self._frustums_hidden_for_sync = False
        self._apply_frustum_visibility_state()

    def _install_frustum_sync_callbacks(self) -> None:
        if not self.sync_frustums_to_render:
            return
        if self.server is None:
            return

        def _on_connect(client: Any) -> None:
            @client.camera.on_update
            def _(_: Any) -> None:
                # If user starts dragging after a click-focus, automatically restore all.
                is_programmatic = (
                    time.time() <= float(self._programmatic_camera_update_until)
                )
                self._last_camera_move_time = time.time()
                if (
                    self._focused_camera_idx is not None
                    and not is_programmatic
                    and len(self.camera_handles) > 0
                ):
                    self._focused_camera_idx = None
                    self._apply_frustum_visibility_state()
                # Skip hide/show sync for programmatic camera jumps triggered by frustum click.
                if is_programmatic:
                    return
                if not self.show_camera_frustums:
                    return
                if len(self.camera_handles) == 0:
                    return
                if self._frustums_hidden_for_sync:
                    return
                self._frustums_hidden_for_sync = True
                self._apply_frustum_visibility_state()

        self.server.on_client_connect(_on_connect)

    def _create_camera_on_click_callback(self, capture_idx: int):
        def _on_click(event: Any) -> None:
            self._programmatic_camera_update_until = time.time() + 0.30
            self._frustums_hidden_for_sync = False
            if self.focus_frustum_on_click:
                self._focused_camera_idx = int(capture_idx)
            else:
                self._focused_camera_idx = None
            self._apply_frustum_visibility_state()
            with event.client.atomic():
                event.client.camera.position = event.target.position
                event.client.camera.wxyz = event.target.wxyz

        return _on_click

    def _init_train_camera_frustums(self) -> None:
        if self.viewer is None or self.server is None or self.train_dataset is None:
            return

        try:
            import numpy as np  # type: ignore
            import viser.transforms as vtf  # type: ignore
        except ImportError:
            return

        dataset = self.train_dataset
        parsed_scene = dataset.parsed_scene
        total_num = int(len(dataset))
        if total_num <= 0:
            return

        drawn_indices = self._pick_drawn_camera_indices(total_num)
        if len(drawn_indices) == 0:
            return

        # Keep frustum size stable across scenes; avoid scaling with scene extent.
        frustum_scale = max(float(self.camera_frustum_scale), 1e-4)
        self.camera_handles.clear()
        for idx in drawn_indices:
            dataset_index = int(idx)
            image_index = int(parsed_scene.indices[dataset_index])
            K_np = parsed_scene.Ks[image_index]
            c2w_np = parsed_scene.camtoworlds[image_index]
            if K_np.shape != (3, 3) or c2w_np.shape != (4, 4):
                continue

            # Infer image size from intrinsics: width≈2*cx, height≈2*cy.
            original_w = max(1.0, float(K_np[0, 2]) * 2.0)
            original_h = max(1.0, float(K_np[1, 2]) * 2.0)
            fx = max(float(K_np[0, 0]), 1e-8)
            fov_x = float(2.0 * np.arctan((0.5 * original_w) / fx))
            aspect = float(original_w / original_h)

            R = vtf.SO3.from_matrix(c2w_np[:3, :3])
            handle = self.server.scene.add_camera_frustum(
                name=f"/cameras/camera_{dataset_index:05d}",
                fov=fov_x,
                scale=frustum_scale,
                line_width=2.0,
                aspect=aspect,
                wxyz=R.wxyz,
                position=c2w_np[:3, 3],
            )
            handle.on_click(self._create_camera_on_click_callback(dataset_index))
            self.camera_handles[dataset_index] = handle
        self._apply_frustum_visibility_state()

    def _update_counts(self, meta: Optional[dict[str, torch.Tensor]]) -> None:
        if self.viewer is None:
            return
        self.viewer.render_tab_state.total_gs_count = int(self.splats["means"].shape[0])
        if meta is None:
            return
        radii = meta.get("radii")
        if not isinstance(radii, torch.Tensor):
            return
        self.viewer.render_tab_state.rendered_gs_count = int(
            self._count_rendered_gaussians(radii)
        )

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
        degree = int((total_k**0.5) - 1)
        return max(0, degree)

    def _ensure_render_dependencies(self) -> None:
        if self._deps_ready:
            return
        import numpy as np  # type: ignore
        from nerfview import apply_float_colormap  # type: ignore

        from friendly_splat.viewer.gsplat_viewer import GsplatRenderTabState  # type: ignore
        from friendly_splat.utils.common_utils import (
            get_implied_normal_from_depth,
        )
        from friendly_splat.renderer.renderer import render_splats

        self._np = np
        self._apply_float_colormap = apply_float_colormap
        self._gsplat_render_tab_state_cls = GsplatRenderTabState
        self._get_implied_normal_from_depth_fn = get_implied_normal_from_depth
        self._render_splats_fn = render_splats
        self._deps_ready = True

    def _handle_mode_rgb(self, *, render_once: Any) -> Any:
        out = render_once(render_mode="RGB")
        self._update_counts(out.meta)
        return out.pred_rgb[0].clamp(0.0, 1.0).detach().cpu().numpy()

    def _handle_mode_depth(
        self,
        *,
        mode: str,
        render_once: Any,
        render_tab_state: Any,
        apply_float_colormap: Any,
        height: int,
        width: int,
        np_module: Any,
    ) -> Any:
        out = render_once(render_mode="RGB+ED")
        self._update_counts(out.meta)

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

    def _handle_mode_alpha(
        self,
        *,
        render_once: Any,
        render_tab_state: Any,
        apply_float_colormap: Any,
    ) -> Any:
        out = render_once(render_mode="RGB")
        self._update_counts(out.meta)
        alpha = out.alphas[0, ..., 0:1]
        if bool(render_tab_state.inverse):
            alpha = 1.0 - alpha
        return apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()

    def _handle_mode_render_normal(
        self,
        *,
        render_once: Any,
        height: int,
        width: int,
        np_module: Any,
    ) -> Any:
        out = render_once(render_mode="RGB+N+ED")
        self._update_counts(out.meta)
        normals = out.render_normals[0] if out.render_normals is not None else None
        if normals is None:
            return np_module.zeros((height, width, 3), dtype=np_module.uint8)
        normals = (normals + 1.0) * 0.5
        normals = 1.0 - normals
        return (normals.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(
            np_module.uint8
        )

    def _handle_mode_surf_normal(
        self,
        *,
        render_once: Any,
        K: torch.Tensor,
        get_implied_normal_from_depth_fn: Any,
        height: int,
        width: int,
        np_module: Any,
    ) -> Any:
        out = render_once(render_mode="RGB+ED")
        self._update_counts(out.meta)
        depth = out.expected_depth
        if depth is None:
            return np_module.zeros((height, width, 3), dtype=np_module.uint8)
        normals = get_implied_normal_from_depth_fn(depth, K).squeeze(0)
        normals = (normals + 1.0) * 0.5
        normals = 1.0 - normals
        return (normals.clamp(0.0, 1.0).detach().cpu().numpy() * 255.0).astype(
            np_module.uint8
        )

    @torch.no_grad()
    def render(self, camera_state: Any, render_tab_state: Any):
        self._ensure_render_dependencies()
        np = self._np
        apply_float_colormap = self._apply_float_colormap
        get_implied_normal_from_depth = self._get_implied_normal_from_depth_fn
        render_splats = self._render_splats_fn

        assert isinstance(render_tab_state, self._gsplat_render_tab_state_cls)

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

        handlers = {
            "rgb": lambda: self._handle_mode_rgb(render_once=_render),
            "expected_depth": lambda: self._handle_mode_depth(
                mode=mode,
                render_once=_render,
                render_tab_state=render_tab_state,
                apply_float_colormap=apply_float_colormap,
                height=height,
                width=width,
                np_module=np,
            ),
            "median_depth": lambda: self._handle_mode_depth(
                mode=mode,
                render_once=_render,
                render_tab_state=render_tab_state,
                apply_float_colormap=apply_float_colormap,
                height=height,
                width=width,
                np_module=np,
            ),
            "alpha": lambda: self._handle_mode_alpha(
                render_once=_render,
                render_tab_state=render_tab_state,
                apply_float_colormap=apply_float_colormap,
            ),
            "render_normal": lambda: self._handle_mode_render_normal(
                render_once=_render,
                height=height,
                width=width,
                np_module=np,
            ),
            "surf_normal": lambda: self._handle_mode_surf_normal(
                render_once=_render,
                K=K,
                get_implied_normal_from_depth_fn=get_implied_normal_from_depth,
                height=height,
                width=width,
                np_module=np,
            ),
        }
        handler = handlers.get(mode)
        if handler is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        return handler()
