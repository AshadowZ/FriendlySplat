from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, Tuple


@dataclass(frozen=True)
class SnapshotPanelHandles:
    save_button: Any


@dataclass(frozen=True)
class UniversalPlotHandles:
    dropdown: Any
    window_slider: Any
    latest_number: Any
    plot: Any


@dataclass(frozen=True)
class EvalMetricPanelHandles:
    show_checkbox: Any
    window_slider: Any
    last_value_handles: dict[str, Any]
    plot_handles: dict[str, Any]


def add_frustum_visibility_toggle(
    *,
    server: Any,
    viewer: Any,
    initial_value: bool,
) -> Any:
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
        return server.gui.add_checkbox(
            "Show Frustums",
            initial_value=bool(initial_value),
            hint="Show or hide all camera frustums.",
            order=toggle_order,
        )


def add_snapshot_panel(
    *,
    server: Any,
    viewer: Any,
) -> SnapshotPanelHandles:
    save_order = None
    viewer_res_key = getattr(
        viewer,
        "HANDLE_VIEWER_RES_SLIDER",
        "viewer_res_slider",
    )
    viewer_res_handle = viewer._rendering_tab_handles.get(viewer_res_key)
    if viewer_res_handle is not None:
        try:
            save_order = float(viewer_res_handle.order) + 0.01
        except Exception:
            save_order = None
    with viewer._rendering_folder:
        save_button = server.gui.add_button(
            "Save PNG",
            hint="Render the current view once and save it as a PNG under screenshots/.",
            order=save_order,
        )
    return SnapshotPanelHandles(save_button=save_button)


def add_universal_plot_panel(
    *,
    server: Any,
    np_module: Any,
    uplot_module: Any,
    empty_option: str,
    scalar_max_points: int,
) -> UniversalPlotHandles:
    dropdown = server.gui.add_dropdown(
        "Select Metric",
        (str(empty_option),),
        initial_value=str(empty_option),
        hint="Select any runtime scalar stream to visualize.",
    )
    window_slider = server.gui.add_slider(
        "Universal Window (points)",
        min=0,
        max=int(scalar_max_points),
        step=10,
        initial_value=0,
        hint="0 means full history; otherwise show only the latest N points.",
    )
    latest_number = server.gui.add_number(
        "Selected Metric (latest)",
        initial_value=0.0,
        disabled=True,
    )
    plot = server.gui.add_uplot(
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
    return UniversalPlotHandles(
        dropdown=dropdown,
        window_slider=window_slider,
        latest_number=latest_number,
        plot=plot,
    )


def add_eval_metrics_panel(
    *,
    server: Any,
    uplot_module: Any,
    metric_specs: Sequence[Tuple[str, str, str]],
    metrics_max_points: int,
    default_x: Any,
    default_y: Any,
) -> EvalMetricPanelHandles:
    show_checkbox = server.gui.add_checkbox(
        "Show Metric Plots",
        initial_value=True,
        hint="Show or hide PSNR/SSIM/LPIPS curves (eval updates only).",
    )
    window_slider = server.gui.add_slider(
        "Window (points)",
        min=0,
        max=int(metrics_max_points),
        step=10,
        initial_value=0,
        hint="0 means full history; otherwise show only the latest N points.",
    )

    last_value_handles: dict[str, Any] = {}
    plot_handles: dict[str, Any] = {}
    for metric_name, metric_title, metric_color in metric_specs:
        last_value_handles[metric_name] = server.gui.add_number(
            f"{metric_title} (latest)",
            initial_value=0.0,
            disabled=True,
        )
        plot_handles[metric_name] = server.gui.add_uplot(
            data=(default_x, default_y),
            series=(
                uplot_module.Series(label="step", show=False),
                uplot_module.Series(
                    label=metric_title,
                    stroke=metric_color,
                    width=2.0,
                ),
            ),
            title=f"{metric_title} vs Train Step",
            scales={
                "x": uplot_module.Scale(
                    time=False,
                    auto=False,
                    range=(0.0, 1.0),
                ),
            },
            aspect=2.0,
        )

    return EvalMetricPanelHandles(
        show_checkbox=show_checkbox,
        window_slider=window_slider,
        last_value_handles=last_value_handles,
        plot_handles=plot_handles,
    )
