from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from friendly_splat.modules.gaussian import GaussianModel
from friendly_splat.viewer import viewer_panels
from friendly_splat.viewer.viewer_renderer import ViewerRenderer

try:
    import imageio.v3 as iio  # type: ignore
except ImportError:  # pragma: no cover
    iio = None  # type: ignore[assignment]

try:
    import numpy as np  # type: ignore
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import viser  # type: ignore
except ImportError:  # pragma: no cover
    viser = None  # type: ignore[assignment]

try:
    import viser.transforms as vtf  # type: ignore
except ImportError:  # pragma: no cover
    vtf = None  # type: ignore[assignment]

try:
    from viser import uplot  # type: ignore
except ImportError:  # pragma: no cover
    uplot = None  # type: ignore[assignment]

try:
    from friendly_splat.viewer.gsplat_viewer import GsplatViewer  # type: ignore
except ImportError as e:  # pragma: no cover
    _GSPLAT_VIEWER_IMPORT_ERROR: ImportError | None = e
    GsplatViewer = None  # type: ignore[assignment]
else:
    _GSPLAT_VIEWER_IMPORT_ERROR = None

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
        gaussian_model: GaussianModel,
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
        self.disable_viewer = disable_viewer
        self.port = port
        self.device = device
        self.gaussian_model = gaussian_model
        self.output_dir = Path(output_dir)
        self.packed = packed
        self.sparse_grad = sparse_grad
        self.absgrad = absgrad
        self.train_dataset = train_dataset
        self.max_display_cameras = max_display_cameras
        self.camera_frustum_scale = camera_frustum_scale
        self.show_camera_frustums = show_camera_frustums
        self.sync_frustums_to_render = sync_frustums_to_render
        self.frustum_show_after_static_sec = frustum_show_after_static_sec
        self.focus_frustum_on_click = focus_frustum_on_click
        self.metrics_max_points = max(64, metrics_max_points)
        self.scalar_max_points = max(256, scalar_max_points)

        self.server: Any = None
        self.viewer: Any = None
        self.renderer: Optional[ViewerRenderer] = None
        self.camera_handles: dict[int, Any] = {}
        self._frustums_hidden_for_sync = False
        self._last_camera_move_time = 0.0
        self._focused_camera_idx: Optional[int] = None
        self._programmatic_camera_update_until = 0.0
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

        if (
            viser is None
            or vtf is None
            or uplot is None
            or GsplatViewer is None
            or np is None
        ):
            detail = None
            if _GSPLAT_VIEWER_IMPORT_ERROR is not None:
                detail = f"{type(_GSPLAT_VIEWER_IMPORT_ERROR).__name__}: {_GSPLAT_VIEWER_IMPORT_ERROR}"
            raise ImportError(
                "Online viewer requested but dependencies are missing. "
                "Install `viser` and `nerfview` (see friendly_splat/requirements.txt) or run with disable_viewer=True."
                + (f" (import error: {detail})" if detail else "")
            ) from _GSPLAT_VIEWER_IMPORT_ERROR

        self.server = viser.ViserServer(port=self.port, verbose=False)
        self.renderer = ViewerRenderer(
            device=self.device,
            gaussian_model=self.gaussian_model,
            packed=self.packed,
            sparse_grad=self.sparse_grad,
            absgrad=self.absgrad,
            update_counts_fn=self._update_counts,
        )

        def _on_render_tab_populated(viewer: Any) -> None:
            self._install_frustum_visibility_toggle(viewer)
            self._install_snapshot_panel(viewer)
            self._install_metrics_panel(viewer)

        self.viewer = GsplatViewer(
            self.server,
            self.render,  # callback
            output_dir=self.output_dir,
            mode="training",
            after_render_hook=self._on_after_render,
            after_render_tab_populated_hook=_on_render_tab_populated,
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
            num_train_rays_per_step = batch_size * height * width
            self.viewer.render_tab_state.num_train_rays_per_sec = (
                num_train_rays_per_step * num_train_steps_per_sec
            )
            self.viewer.update(step, num_train_rays_per_step)

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

    def push_eval_metrics(self, *, step: int, stats: dict[str, float | int]) -> None:
        """Record one eval point and refresh viewer plots."""
        if self.viewer is None or self.server is None:
            return

        train_step_raw = stats.get("train_step", int(step) + 1)
        train_step = int(train_step_raw)
        self._metrics_latest_train_step = max(
            self._metrics_latest_train_step,
            train_step,
        )
        updated = False
        for metric_name in ("psnr", "ssim", "lpips"):
            updated = (
                self._log_value_to_history(
                    container=self._metrics_history,
                    key=metric_name,
                    value=stats.get(metric_name),
                    step=train_step,
                    max_len=self.metrics_max_points,
                )
                or updated
            )

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
                self._scalar_latest_train_step,
                train_step,
            )

        any_updated = False
        for key, raw_value in scalars.items():
            any_updated = (
                self._log_value_to_history(
                    container=self._scalar_history,
                    key=key,
                    value=raw_value,
                    step=train_step,
                    max_len=self.scalar_max_points,
                )
                or any_updated
            )

        if not any_updated:
            return

        self._update_universal_metric_dropdown_options()
        self._refresh_universal_metric_plot()

    def log_payload(self, *, payload: object | None) -> None:
        """Consume trainer log payload and update viewer plots."""
        if payload is None:
            return
        if self.viewer is None or self.server is None:
            return

        step_raw = getattr(payload, "step", None)
        if step_raw is None:
            return
        step = max(0, int(step_raw))

        train_scalars = getattr(payload, "train_scalars", None)
        if isinstance(train_scalars, Mapping):
            self.log_scalars(step=step, scalars=dict(train_scalars))

        eval_metrics = getattr(payload, "eval_metrics", None)
        if not isinstance(eval_metrics, Mapping) or len(eval_metrics) == 0:
            return

        eval_stats: dict[str, object] = {"train_step": int(step)}
        eval_scalars: dict[str, object] = {}
        for key, value in eval_metrics.items():
            if not isinstance(key, str):
                continue
            metric_name = key.strip()
            if len(metric_name) == 0:
                continue
            eval_stats[metric_name] = value
            eval_scalars[f"eval/{metric_name}"] = value

        self.push_eval_metrics(step=step, stats=eval_stats)
        self.log_scalars(step=step, scalars=eval_scalars)

    @staticmethod
    def _to_float_scalar(value: object) -> Optional[float]:
        if isinstance(value, torch.Tensor):
            if int(value.numel()) != 1:
                return None
            return float(value.detach().item())
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def _append_history(
        self,
        *,
        container: dict[str, list[tuple[int, float]]],
        key: str,
        step: int,
        value: float,
        max_len: int,
    ) -> bool:
        series = container.setdefault(str(key), [])
        series.append((int(step), float(value)))
        limit = max(1, int(max_len))
        if len(series) > limit:
            del series[: len(series) - limit]
        return True

    def _log_value_to_history(
        self,
        *,
        container: dict[str, list[tuple[int, float]]],
        key: object,
        value: object,
        step: int,
        max_len: int,
    ) -> bool:
        if not isinstance(key, str):
            return False
        key_name = key.strip()
        if len(key_name) == 0:
            return False
        scalar = self._to_float_scalar(value)
        if scalar is None:
            return False
        return self._append_history(
            container=container,
            key=key_name,
            step=step,
            value=scalar,
            max_len=max_len,
        )

    def _apply_frustum_visibility_state(self) -> None:
        if len(self.camera_handles) == 0 or self.server is None:
            return
        if not self.show_camera_frustums:
            with self.server.atomic():
                for handle in self.camera_handles.values():
                    handle.visible = False
            return
        if self._frustums_hidden_for_sync:
            with self.server.atomic():
                for handle in self.camera_handles.values():
                    handle.visible = False
            return
        if self._focused_camera_idx is not None:
            focused_idx = self._focused_camera_idx
            with self.server.atomic():
                for cam_idx, handle in self.camera_handles.items():
                    handle.visible = int(cam_idx) == focused_idx
            return
        with self.server.atomic():
            for handle in self.camera_handles.values():
                handle.visible = True

    def _install_frustum_visibility_toggle(self, viewer: Any) -> None:
        if self.server is None:
            return
        toggle = viewer_panels.add_frustum_visibility_toggle(
            server=self.server,
            viewer=viewer,
            initial_value=self.show_camera_frustums,
        )

        @toggle.on_update
        def _(_: Any) -> None:
            self.show_camera_frustums = toggle.value
            if self.show_camera_frustums:
                self._frustums_hidden_for_sync = False
            self._apply_frustum_visibility_state()

    def _install_snapshot_panel(self, viewer: Any) -> None:
        if self.server is None:
            return
        handles = viewer_panels.add_snapshot_panel(
            server=self.server,
            viewer=viewer,
        )

        @handles.save_button.on_click
        def _(event: Any) -> None:
            self._save_current_render_png(event=event)

    def _save_current_render_png(self, *, event: Any) -> None:
        if (
            self.viewer is None
            or self.renderer is None
            or np is None
            or iio is None
        ):
            return

        client = getattr(event, "client", None)
        if client is None:
            return

        notification = None
        if hasattr(client, "add_notification"):
            notification = client.add_notification(
                "Save PNG",
                "Rendering current view...",
                loading=True,
                with_close_button=True,
            )

        try:
            camera_state = self.viewer.get_camera_state(client)
            with self.viewer.lock:
                image = self.renderer.render(camera_state, self.viewer.render_tab_state)

            if not isinstance(image, np.ndarray):
                raise TypeError(
                    f"Expected renderer to return a numpy array, got {type(image)!r}."
                )

            if image.dtype != np.uint8:
                image = (np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

            screenshots_dir = self.output_dir / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            render_mode = str(
                getattr(self.viewer.render_tab_state, "render_mode", "render")
            ).strip() or "render"
            filename = f"{timestamp}_{render_mode}.png"
            out_path = screenshots_dir / filename
            iio.imwrite(out_path, image)
            print(f"Saved viewer snapshot: {out_path}", flush=True)

            if notification is not None:
                notification.loading = False
                notification.color = "green"
                notification.body = f"Saved to {out_path.name}"
                notification.auto_close_seconds = 3.0
        except Exception as exc:
            print(f"Failed to save viewer snapshot: {exc}", flush=True)
            if notification is not None:
                notification.loading = False
                notification.color = "red"
                notification.body = str(exc)
        finally:
            if notification is not None and notification.loading:
                notification.loading = False

    def _install_metrics_panel(self, viewer: Any) -> None:
        if self.server is None or np is None or uplot is None:
            return

        default_x = np.asarray([0.0, 1.0], dtype=np.float32)
        default_y = np.asarray([0.0, 0.0], dtype=np.float32)
        metric_specs = (
            ("psnr", "PSNR", "#00c389"),
            ("ssim", "SSIM", "#ff8c42"),
            ("lpips", "LPIPS", "#5b8ff9"),
        )

        def _populate_metrics_controls() -> None:
            handles = viewer_panels.add_eval_metrics_panel(
                server=self.server,
                uplot_module=uplot,
                metric_specs=metric_specs,
                metrics_max_points=self.metrics_max_points,
                default_x=default_x,
                default_y=default_y,
            )
            self._metrics_show_checkbox = handles.show_checkbox
            self._metrics_window_slider = handles.window_slider
            self._metrics_last_value_handles = handles.last_value_handles
            self._metrics_plot_handles = handles.plot_handles

        def _populate_universal_controls() -> None:
            handles = viewer_panels.add_universal_plot_panel(
                server=self.server,
                np_module=np,
                uplot_module=uplot,
                empty_option=self._UNIVERSAL_EMPTY_OPTION,
                scalar_max_points=self.scalar_max_points,
            )
            self._universal_metric_dropdown = handles.dropdown
            self._universal_metric_window_slider = handles.window_slider
            self._universal_metric_latest_number = handles.latest_number
            self._universal_metric_plot = handles.plot

            @self._universal_metric_dropdown.on_update
            def _(_: Any) -> None:
                self._refresh_universal_metric_plot()

            @self._universal_metric_window_slider.on_update
            def _(_: Any) -> None:
                self._refresh_universal_metric_plot()

        metrics_tab = getattr(viewer, "metrics_tab", None)
        if metrics_tab is not None:
            with metrics_tab:
                _populate_universal_controls()
                _populate_metrics_controls()
        else:
            with viewer._rendering_folder:
                with self.server.gui.add_folder("Universal Plot"):
                    _populate_universal_controls()
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

    def _process_plot_data(
        self,
        *,
        history: list[tuple[int, float]],
        window_slider: Optional[Any],
        max_plot_points: int = 2000,
    ) -> tuple["np.ndarray", "np.ndarray"]:
        if len(history) == 0:
            return (
                np.asarray([0.0, 1.0], dtype=np.float32),
                np.asarray([0.0, 0.0], dtype=np.float32),
            )

        window = 0
        if window_slider is not None:
            window = int(window_slider.value)
        if window > 0 and len(history) > window:
            history = history[-window:]

        x = np.asarray([float(step) for step, _ in history], dtype=np.float32)
        y = np.asarray([float(value) for _, value in history], dtype=np.float32)

        # Keep UI responsive when a metric accumulates many points.
        if int(x.shape[0]) > max_plot_points:
            indices = np.linspace(
                0,
                int(x.shape[0]) - 1,
                max_plot_points,
                dtype=np.int64,
            )
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
        current_step = max(1, self._scalar_latest_train_step)

        with self.server.atomic():
            if selected not in self._scalar_history:
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

            x, y = self._process_plot_data(
                history=self._scalar_history.get(selected, []),
                window_slider=self._universal_metric_window_slider,
            )
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
            visible = self._metrics_show_checkbox.value
        for handle in self._metrics_plot_handles.values():
            handle.visible = visible
        for handle in self._metrics_last_value_handles.values():
            handle.visible = visible
        if self._metrics_window_slider is not None:
            self._metrics_window_slider.visible = visible

    def _refresh_metric_plots(self) -> None:
        if self.server is None or len(self._metrics_plot_handles) == 0:
            return
        current_step = max(1, self._metrics_latest_train_step)
        with self.server.atomic():
            for metric_name, plot_handle in self._metrics_plot_handles.items():
                x, y = self._process_plot_data(
                    history=self._metrics_history.get(metric_name, []),
                    window_slider=self._metrics_window_slider,
                )
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
        static_delay = max(self.frustum_show_after_static_sec, 0.0)
        if time.time() - self._last_camera_move_time < static_delay:
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
                is_programmatic = time.time() <= self._programmatic_camera_update_until
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
        if np is None or vtf is None:
            return

        dataset = self.train_dataset
        parsed_scene = dataset.parsed_scene
        total_num = int(len(dataset))
        if total_num <= 0:
            return

        max_display = self.max_display_cameras
        if max_display <= 0 or max_display >= total_num:
            drawn_indices = list(range(total_num))
        else:
            drawn_indices = np.linspace(
                0, total_num - 1, max_display, dtype=np.int32
            ).tolist()
        if len(drawn_indices) == 0:
            return

        # Keep frustum size stable across scenes; avoid scaling with scene extent.
        frustum_scale = max(self.camera_frustum_scale, 1e-4)
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
        self.viewer.render_tab_state.total_gs_count = int(
            self.gaussian_model.num_gaussians
        )
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

    def render(self, camera_state: Any, render_tab_state: Any):
        if self.renderer is None:
            raise RuntimeError("Viewer renderer is not initialized.")
        return self.renderer.render(camera_state, render_tab_state)
