# Copyright 2022 the Regents of the University of California, Nerfstudio Team
# and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

# This module is a local fork of nerfview's render panel. We pulled it into the
# repo specifically to replace the upstream Dump Video behavior, because the
# original path could miss the trajectory start on the first frame and depended
# on viewport/client render state instead of exporting directly from the
# trajectory camera states.

import json
import os
import threading
import time
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import viser
import viser.transforms as tf
from rich.console import Console

from nerfview import CameraState
from nerfview.render_panel import CameraPath, Keyframe, RenderTabState


def populate_general_render_tab(
    server: viser.ViserServer,
    output_dir: Path,
    folder: viser.GuiFolderHandle,
    render_tab_state: RenderTabState,
    extra_handles: Optional[Dict[str, viser.GuiInputHandle]] = None,
    scale_ratio: float = 10.0,
    time_enabled: bool = False,
    render_fn: Optional[object] = None,
    render_lock: Optional[Lock] = None,
) -> Dict[str, viser.GuiInputHandle]:
    """FriendlySplat-local copy of nerfview's render panel with Dump Video fixes.

    This exists so FriendlySplat can override the upstream video export path
    without patching nerfview inside site-packages. In particular, we keep a
    local copy here to fix two issues: the first dumped frame not matching the
    trajectory start, and video export depending on viewport/client render state
    instead of rendering directly from trajectory camera states.
    """
    if extra_handles is None:
        extra_handles = {}

    with folder:
        fov_degrees_slider = server.gui.add_slider(
            "FOV",
            initial_value=50.0,
            min=0.1,
            max=175.0,
            step=0.01,
            hint="Field-of-view for rendering, which can also be overridden on a per-keyframe basis.",
        )

        render_time = None
        if time_enabled:
            render_time = server.gui.add_slider(
                "Default Time",
                initial_value=0.0,
                min=0.0,
                max=1.0,
                step=0.01,
                hint="Rendering time step, which can also be overridden on a per-keyframe basis.",
            )

            @render_time.on_update
            def _(_) -> None:
                camera_path.default_render_time = render_time.value

        @fov_degrees_slider.on_update
        def _(_) -> None:
            fov_radians = fov_degrees_slider.value / 180.0 * np.pi
            for client in server.get_clients().values():
                client.camera.fov = fov_radians
            camera_path.default_fov = fov_radians

            camera_path.update_aspect(
                render_res_vec2.value[0] / render_res_vec2.value[1]
            )
            compute_and_update_preview_camera_state()

        render_res_vec2 = server.gui.add_vector2(
            "Render Res",
            initial_value=(1280, 960),
            min=(50, 50),
            max=(10_000, 10_000),
            step=1,
            hint="Rendering resolution.",
        )

        @render_res_vec2.on_update
        def _(_) -> None:
            camera_path.update_aspect(
                render_res_vec2.value[0] / render_res_vec2.value[1]
            )
            compute_and_update_preview_camera_state()
            render_tab_state.render_width = int(render_res_vec2.value[0])
            render_tab_state.render_height = int(render_res_vec2.value[1])

        add_keyframe_button = server.gui.add_button(
            "Add Keyframe",
            icon=viser.Icon.PLUS,
            hint="Add a new keyframe at the current pose.",
        )

        @add_keyframe_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client_id is not None
            camera = server.get_clients()[event.client_id].camera

            camera_path.add_camera(
                Keyframe.from_camera(
                    camera,
                    aspect=render_res_vec2.value[0] / render_res_vec2.value[1],
                ),
            )
            duration_number.value = camera_path.compute_duration()
            camera_path.update_spline()

        clear_keyframes_button = server.gui.add_button(
            "Clear Keyframes",
            icon=viser.Icon.TRASH,
            hint="Remove all keyframes from the render path.",
        )

        @clear_keyframes_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client_id is not None
            client = server.get_clients()[event.client_id]
            with client.atomic(), client.gui.add_modal("Confirm") as modal:
                client.gui.add_markdown("Clear all keyframes?")
                confirm_button = client.gui.add_button(
                    "Yes", color="red", icon=viser.Icon.TRASH
                )
                exit_button = client.gui.add_button("Cancel")

                @confirm_button.on_click
                def _(_) -> None:
                    camera_path.reset()
                    modal.close()

                    duration_number.value = camera_path.compute_duration()

                    if len(transform_controls) > 0:
                        for t in transform_controls:
                            t.remove()
                        transform_controls.clear()
                        return

                @exit_button.on_click
                def _(_) -> None:
                    modal.close()

        reset_up_button = server.gui.add_button(
            "Reset Up Direction",
            icon=viser.Icon.ARROW_BIG_UP_LINES,
            color="gray",
            hint="Set the up direction of the camera orbit controls to the camera's current up direction.",
        )

        @reset_up_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            event.client.camera.up_direction = tf.SO3(
                event.client.camera.wxyz
            ) @ np.array([0.0, -1.0, 0.0])

        loop_checkbox = server.gui.add_checkbox(
            "Loop",
            False,
            hint="Add a segment between the first and last keyframes.",
        )

        @loop_checkbox.on_update
        def _(_) -> None:
            camera_path.loop = loop_checkbox.value
            duration_number.value = camera_path.compute_duration()

        tension_slider = server.gui.add_slider(
            "Spline tension",
            min=0.0,
            max=1.0,
            initial_value=0.0,
            step=0.01,
            hint="Tension parameter for adjusting smoothness of spline interpolation.",
        )

        @tension_slider.on_update
        def _(_) -> None:
            camera_path.tension = tension_slider.value
            camera_path.update_spline()

        move_checkbox = server.gui.add_checkbox(
            "Move keyframes",
            initial_value=False,
            hint="Toggle move handles for keyframes in the scene.",
        )

        transform_controls: List[viser.SceneNodeHandle] = []

        @move_checkbox.on_update
        def _(event: viser.GuiEvent) -> None:
            if move_checkbox.value is False:
                for t in transform_controls:
                    t.remove()
                transform_controls.clear()
                return

            def _make_transform_controls_callback(
                keyframe: Tuple[Keyframe, viser.SceneNodeHandle],
                controls: viser.TransformControlsHandle,
            ) -> None:
                @controls.on_update
                def _(_) -> None:
                    keyframe[0].wxyz = controls.wxyz
                    keyframe[0].position = controls.position
                    keyframe[1].wxyz = controls.wxyz
                    keyframe[1].position = controls.position
                    camera_path.update_spline()

            assert event.client is not None
            for keyframe_index, keyframe in camera_path._keyframes.items():
                controls = event.client.scene.add_transform_controls(
                    f"/keyframe_move/{keyframe_index}",
                    scale=0.4,
                    wxyz=keyframe[0].wxyz,
                    position=keyframe[0].position,
                )
                transform_controls.append(controls)
                _make_transform_controls_callback(keyframe, controls)

        show_keyframe_checkbox = server.gui.add_checkbox(
            "Show keyframes",
            initial_value=True,
            hint="Show keyframes in the scene.",
        )

        @show_keyframe_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            camera_path.set_keyframes_visible(show_keyframe_checkbox.value)

        show_spline_checkbox = server.gui.add_checkbox(
            "Show spline",
            initial_value=True,
            hint="Show camera path spline in the scene.",
        )

        @show_spline_checkbox.on_update
        def _(_) -> None:
            camera_path.show_spline = show_spline_checkbox.value
            camera_path.update_spline()

        transition_sec_number = server.gui.add_number(
            "Transition (sec)",
            min=0.001,
            max=30.0,
            step=0.001,
            initial_value=2.0,
            hint="Time in seconds between each keyframe, which can also be overridden on a per-transition basis.",
        )
        framerate_number = server.gui.add_number(
            "FPS", min=0.1, max=240.0, step=1e-2, initial_value=30.0
        )
        duration_number = server.gui.add_number(
            "Duration (sec)",
            min=0.0,
            max=1e8,
            step=0.001,
            initial_value=0.0,
            disabled=True,
        )

        @transition_sec_number.on_update
        def _(_) -> None:
            camera_path.default_transition_sec = transition_sec_number.value
            duration_number.value = camera_path.compute_duration()

        trajectory_name_text = server.gui.add_text(
            "Name",
            initial_value="default",
            hint="Name of the trajectory",
        )

        load_camera_path_button = server.gui.add_button(
            "Load Trajectory",
            icon=viser.Icon.FOLDER_OPEN,
            hint="Load an existing camera path.",
        )

        save_camera_path_button = server.gui.add_button(
            "Save Trajectory",
            icon=viser.Icon.FILE_EXPORT,
            hint="Save the current trajectory to a json file.",
        )

        play_button = server.gui.add_button("Play", icon=viser.Icon.PLAYER_PLAY)
        pause_button = server.gui.add_button(
            "Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False
        )
        preview_save_camera_path_button = server.gui.add_button(
            "Preview Render",
            icon=viser.Icon.EYE,
            hint="Show a preview of the render in the viewport.",
        )
        preview_render_stop_button = server.gui.add_button(
            "Exit Render Preview", color="red", visible=False
        )
        dump_video_button = server.gui.add_button(
            "Dump Video",
            color="green",
            icon=viser.Icon.PLAYER_PLAY,
            hint="Dump the current trajectory as a video.",
        )

    def get_max_frame_index() -> int:
        return max(1, int(framerate_number.value * duration_number.value) - 1)

    preview_camera_handle: Optional[viser.SceneNodeHandle] = None

    def remove_preview_camera() -> None:
        nonlocal preview_camera_handle
        if preview_camera_handle is not None:
            preview_camera_handle.remove()
            preview_camera_handle = None

    def compute_and_update_preview_camera_state(
        frame_index: Optional[int] = None,
    ) -> Optional[Union[Tuple[tf.SE3, float], Tuple[tf.SE3, float, float]]]:
        """Update preview render state and return the interpolated camera state."""
        if preview_frame_slider is None:
            return None
        if frame_index is None:
            frame_index = int(preview_frame_slider.value)
        maybe_pose_and_fov_rad = camera_path.interpolate_pose_and_fov_rad(
            float(frame_index) / float(get_max_frame_index())
        )
        if maybe_pose_and_fov_rad is None:
            remove_preview_camera()
            return None
        time_value = None
        if len(maybe_pose_and_fov_rad) == 3:
            pose, fov_rad, time_value = maybe_pose_and_fov_rad
            render_tab_state.preview_time = time_value
        else:
            pose, fov_rad = maybe_pose_and_fov_rad
        render_tab_state.preview_fov = fov_rad
        render_tab_state.preview_aspect = camera_path.get_aspect()

        if time_value is not None:
            return pose, fov_rad, time_value
        return pose, fov_rad

    def sync_clients_to_preview_frame(frame_index: int) -> bool:
        assert preview_frame_slider is not None
        preview_frame_slider.value = int(frame_index)
        maybe_pose_and_fov_rad = compute_and_update_preview_camera_state(
            frame_index=int(frame_index)
        )
        if maybe_pose_and_fov_rad is None:
            remove_preview_camera()
            return False
        if len(maybe_pose_and_fov_rad) == 3:
            pose, fov_rad, _time_value = maybe_pose_and_fov_rad
        else:
            pose, fov_rad = maybe_pose_and_fov_rad
        for current_client in server.get_clients().values():
            current_client.camera.wxyz = pose.rotation().wxyz
            current_client.camera.position = pose.translation()
            current_client.camera.fov = fov_rad
        return True

    def camera_state_from_preview_frame(frame_index: int) -> Optional[CameraState]:
        maybe_pose_and_fov_rad = compute_and_update_preview_camera_state(
            frame_index=int(frame_index)
        )
        if maybe_pose_and_fov_rad is None:
            return None
        if len(maybe_pose_and_fov_rad) == 3:
            pose, fov_rad, _time_value = maybe_pose_and_fov_rad
        else:
            pose, fov_rad = maybe_pose_and_fov_rad
        c2w = pose.as_matrix().astype(np.float32)
        aspect = float(render_res_vec2.value[0]) / float(render_res_vec2.value[1])
        return CameraState(
            fov=float(fov_rad),
            aspect=aspect,
            c2w=c2w,
        )

    def render_video_frame(frame_index: int) -> Optional[np.ndarray]:
        camera_state = camera_state_from_preview_frame(frame_index)
        if camera_state is None:
            return None
        if render_fn is None:
            return None
        if render_lock is None:
            rendered = render_fn(camera_state, render_tab_state)
        else:
            with render_lock:
                rendered = render_fn(camera_state, render_tab_state)
        if isinstance(rendered, tuple):
            image = rendered[0]
        else:
            image = rendered
        image_np = np.asarray(image)
        if np.issubdtype(image_np.dtype, np.floating):
            image_np = np.clip(image_np, 0.0, 1.0)
            image_np = (image_np * 255.0).astype(np.uint8)
        elif image_np.dtype != np.uint8:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        return image_np

    def add_preview_frame_slider() -> Optional[viser.GuiInputHandle[int]]:
        with folder:
            preview_frame_slider = server.gui.add_slider(
                "Preview frame",
                min=0,
                max=get_max_frame_index(),
                step=1,
                initial_value=0,
                order=trajectory_name_text.order + 0.01,
                disabled=get_max_frame_index() == 1,
            )
            play_button.disabled = preview_frame_slider.disabled
            preview_save_camera_path_button.disabled = preview_frame_slider.disabled
            save_camera_path_button.disabled = preview_frame_slider.disabled
            dump_video_button.disabled = preview_frame_slider.disabled

        @preview_frame_slider.on_update
        def _(_) -> None:
            nonlocal preview_camera_handle
            maybe_pose_and_fov_rad = compute_and_update_preview_camera_state()
            if maybe_pose_and_fov_rad is None:
                return
            if len(maybe_pose_and_fov_rad) == 3:
                pose, fov_rad, _time_value = maybe_pose_and_fov_rad
            else:
                pose, fov_rad = maybe_pose_and_fov_rad

            preview_camera_handle = server.scene.add_camera_frustum(
                "/preview_camera",
                fov=fov_rad,
                aspect=render_res_vec2.value[0] / render_res_vec2.value[1],
                scale=0.35,
                wxyz=pose.rotation().wxyz,
                position=pose.translation(),
                color=(10, 200, 30),
            )
            if render_tab_state.preview_render:
                for client in server.get_clients().values():
                    client.camera.wxyz = pose.rotation().wxyz
                    client.camera.position = pose.translation()
                    client.camera.fov = fov_rad

        return preview_frame_slider

    camera_pose_backup_from_id: Dict[int, tuple] = {}

    @preview_save_camera_path_button.on_click
    def _(_) -> None:
        render_tab_state.preview_render = True
        preview_save_camera_path_button.visible = False
        preview_render_stop_button.visible = True
        dump_video_button.disabled = True

        maybe_pose_and_fov_rad = compute_and_update_preview_camera_state()
        if maybe_pose_and_fov_rad is None:
            remove_preview_camera()
            return
        if len(maybe_pose_and_fov_rad) == 3:
            pose, _fov, _time_value = maybe_pose_and_fov_rad
        else:
            pose, _fov = maybe_pose_and_fov_rad

        server.scene.set_global_visibility(False)

        for client in server.get_clients().values():
            camera_pose_backup_from_id[client.client_id] = (
                client.camera.position,
                client.camera.look_at,
                client.camera.up_direction,
            )
            client.camera.wxyz = pose.rotation().wxyz
            client.camera.position = pose.translation()

    @preview_render_stop_button.on_click
    def _(_) -> None:
        render_tab_state.preview_render = False
        preview_save_camera_path_button.visible = True
        preview_render_stop_button.visible = False
        dump_video_button.disabled = False

        for client in server.get_clients().values():
            if client.client_id not in camera_pose_backup_from_id:
                continue
            cam_position, cam_look_at, cam_up = camera_pose_backup_from_id.pop(
                client.client_id
            )
            client.camera.position = cam_position
            client.camera.look_at = cam_look_at
            client.camera.up_direction = cam_up
            client.flush()

        server.scene.set_global_visibility(True)

    preview_frame_slider = add_preview_frame_slider()
    handles = {
        "fov_degrees_slider": fov_degrees_slider,
        "render_res_vec2": render_res_vec2,
        "add_keyframe_button": add_keyframe_button,
        "clear_keyframes_button": clear_keyframes_button,
        "reset_up_button": reset_up_button,
        "loop_checkbox": loop_checkbox,
        "tension_slider": tension_slider,
        "move_checkbox": move_checkbox,
        "show_keyframe_checkbox": show_keyframe_checkbox,
        "show_spline_checkbox": show_spline_checkbox,
        "transition_sec_number": transition_sec_number,
        "framerate_number": framerate_number,
        "duration_number": duration_number,
        "trajectory_name_text": trajectory_name_text,
        "preview_frame_slider": preview_frame_slider,
        "load_camera_path_button": load_camera_path_button,
        "save_camera_path_button": save_camera_path_button,
        "play_button": play_button,
        "pause_button": pause_button,
        "preview_save_camera_path_button": preview_save_camera_path_button,
        "preview_render_stop_button": preview_render_stop_button,
        "dump_video_button": dump_video_button,
    }
    if time_enabled:
        handles["render_time"] = render_time

    @duration_number.on_update
    @framerate_number.on_update
    def _(_) -> None:
        remove_preview_camera()

        nonlocal preview_frame_slider
        old = preview_frame_slider
        assert old is not None

        preview_frame_slider = add_preview_frame_slider()
        if preview_frame_slider is not None:
            old.remove()
        else:
            preview_frame_slider = old

        handles["preview_frame_slider"] = preview_frame_slider
        camera_path.framerate = framerate_number.value
        camera_path.update_spline()

    @play_button.on_click
    def _(_) -> None:
        play_button.visible = False
        pause_button.visible = True
        dump_video_button.disabled = True

        def play() -> None:
            while not play_button.visible:
                max_frame = int(framerate_number.value * duration_number.value)
                if max_frame > 0:
                    assert preview_frame_slider is not None
                    preview_frame_slider.value = (
                        preview_frame_slider.value + 1
                    ) % max_frame
                time.sleep(1.0 / framerate_number.value)

        play_thread = threading.Thread(target=play)
        play_thread.start()
        play_thread.join()
        dump_video_button.disabled = False

    @pause_button.on_click
    def _(_) -> None:
        play_button.visible = True
        pause_button.visible = False

    @load_camera_path_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        camera_path_dir = output_dir / "camera_paths"
        camera_path_dir.mkdir(parents=True, exist_ok=True)
        preexisting_camera_paths = list(camera_path_dir.glob("*.json"))
        preexisting_camera_filenames = [p.name for p in preexisting_camera_paths]

        with event.client.gui.add_modal("Load Path") as modal:
            if len(preexisting_camera_filenames) == 0:
                event.client.gui.add_markdown("No existing paths found")
            else:
                event.client.gui.add_markdown("Select existing camera path:")
                camera_path_dropdown = event.client.gui.add_dropdown(
                    label="Camera Path",
                    options=[str(p) for p in preexisting_camera_filenames],
                    initial_value=str(preexisting_camera_filenames[0]),
                )
                load_button = event.client.gui.add_button("Load")

                @load_button.on_click
                def _(_) -> None:
                    json_path = output_dir / "camera_paths" / camera_path_dropdown.value
                    with open(json_path, "r") as f:
                        json_data = json.load(f)

                    keyframes = json_data["keyframes"]
                    camera_path.reset()
                    for i in range(len(keyframes)):
                        frame = keyframes[i]
                        pose = tf.SE3.from_matrix(
                            np.array(frame["matrix"]).reshape(4, 4)
                        )
                        pose = tf.SE3.from_rotation_and_translation(
                            pose.rotation() @ tf.SO3.from_x_radians(np.pi),
                            pose.translation(),
                        )
                        camera_path.add_camera(
                            Keyframe(
                                position=pose.translation() * scale_ratio,
                                wxyz=pose.rotation().wxyz,
                                override_fov_enabled=abs(
                                    frame["fov"] - json_data.get("default_fov", 0.0)
                                )
                                > 1e-3,
                                override_fov_rad=frame["fov"] / 180.0 * np.pi,
                                override_time_enabled=frame.get(
                                    "override_time_enabled", False
                                ),
                                override_time_val=frame.get("render_time", None),
                                aspect=frame["aspect"],
                                override_transition_enabled=frame.get(
                                    "override_transition_enabled", None
                                ),
                                override_transition_sec=frame.get(
                                    "override_transition_sec", None
                                ),
                            ),
                        )

                    transition_sec_number.value = json_data.get(
                        "default_transition_sec", 0.5
                    )
                    trajectory_name_text.value = json_path.stem
                    camera_path.update_spline()
                    modal.close()
                    server.scene.set_global_visibility(True)

            cancel_button = event.client.gui.add_button("Cancel")

            @cancel_button.on_click
            def _(_) -> None:
                modal.close()

    @save_camera_path_button.on_click
    def _(event: viser.GuiEvent) -> None:
        assert event.client is not None
        num_frames = int(framerate_number.value * duration_number.value)
        json_data = {}
        keyframes = []
        for keyframe, _dummy in camera_path._keyframes.values():
            pose = tf.SE3.from_rotation_and_translation(
                tf.SO3(keyframe.wxyz) @ tf.SO3.from_x_radians(np.pi),
                keyframe.position / scale_ratio,
            )
            keyframe_dict = {
                "matrix": pose.as_matrix().flatten().tolist(),
                "fov": (
                    np.rad2deg(keyframe.override_fov_rad)
                    if keyframe.override_fov_enabled
                    else fov_degrees_slider.value
                ),
                "aspect": keyframe.aspect,
                "override_transition_enabled": keyframe.override_transition_enabled,
                "override_transition_sec": keyframe.override_transition_sec,
            }
            keyframes.append(keyframe_dict)
        json_data["default_fov"] = fov_degrees_slider.value
        json_data["default_transition_sec"] = transition_sec_number.value
        json_data["keyframes"] = keyframes
        json_data["render_height"] = render_res_vec2.value[1]
        json_data["render_width"] = render_res_vec2.value[0]
        json_data["fps"] = framerate_number.value
        json_data["seconds"] = duration_number.value
        json_data["is_cycle"] = loop_checkbox.value
        json_data["smoothness_value"] = tension_slider.value
        camera_path_list = []
        for i in range(num_frames):
            maybe_pose_and_fov = camera_path.interpolate_pose_and_fov_rad(
                i / num_frames
            )
            if maybe_pose_and_fov is None:
                return
            time_value = None
            if len(maybe_pose_and_fov) == 3:
                pose, fov, time_value = maybe_pose_and_fov
            else:
                pose, fov = maybe_pose_and_fov
            pose = tf.SE3.from_rotation_and_translation(
                pose.rotation() @ tf.SO3.from_x_radians(np.pi),
                pose.translation() / scale_ratio,
            )
            camera_path_list_dict = {
                "camera_to_world": pose.as_matrix().flatten().tolist(),
                "fov": np.rad2deg(fov),
                "aspect": render_res_vec2.value[0] / render_res_vec2.value[1],
            }
            if time_value is not None:
                camera_path_list_dict["render_time"] = time_value
            camera_path_list.append(camera_path_list_dict)
        json_data["camera_path"] = camera_path_list

        try:
            json_outfile = (
                output_dir / "camera_paths" / f"{trajectory_name_text.value}.json"
            )
            json_outfile.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            Console(width=120).print(
                "[bold yellow]Warning: Failed to write the camera path to the data directory. Saving to the output directory instead."
            )
            json_outfile = (
                output_dir / "camera_paths" / f"{trajectory_name_text.value}.json"
            )
            json_outfile.parent.mkdir(parents=True, exist_ok=True)
        with open(json_outfile.absolute(), "w") as outfile:
            json.dump(json_data, outfile)
            print(f"Camera path saved to {json_outfile.absolute()}")

    @dump_video_button.on_click
    def _(event: viser.GuiEvent) -> None:
        _client = event.client
        assert _client is not None

        render_tab_state.preview_render = True
        maybe_pose_and_fov_rad = compute_and_update_preview_camera_state()
        if maybe_pose_and_fov_rad is None:
            remove_preview_camera()
            return
        if len(maybe_pose_and_fov_rad) == 3:
            pose, _fov, _time_value = maybe_pose_and_fov_rad
        else:
            pose, _fov = maybe_pose_and_fov_rad

        server.scene.set_global_visibility(False)

        for current_client in server.get_clients().values():
            camera_pose_backup_from_id[current_client.client_id] = (
                current_client.camera.position,
                current_client.camera.look_at,
                current_client.camera.up_direction,
            )
            current_client.camera.wxyz = pose.rotation().wxyz
            current_client.camera.position = pose.translation()

        handles_to_disable = list(handles.values()) + list(extra_handles.values())
        original_disabled = [handle.disabled for handle in handles_to_disable]
        for handle in handles_to_disable:
            handle.disabled = True

        def dump() -> None:
            os.makedirs(output_dir / "videos", exist_ok=True)
            writer = imageio.get_writer(
                f"{output_dir}/videos/traj_{trajectory_name_text.value}.mp4",
                fps=framerate_number.value,
                macro_block_size=1,
            )
            max_frame = int(framerate_number.value * duration_number.value)
            assert max_frame > 0 and preview_frame_slider is not None
            for frame_idx in range(max_frame):
                synced = sync_clients_to_preview_frame(frame_idx)
                if not synced:
                    break
                image = render_video_frame(frame_idx)
                if image is None:
                    break
                writer.append_data(image)
            writer.close()
            print(f"Video saved to videos/traj_{trajectory_name_text.value}.mp4")

        dump_thread = threading.Thread(target=dump)
        dump_thread.start()
        dump_thread.join()

        for handle, was_disabled in zip(handles_to_disable, original_disabled):
            handle.disabled = was_disabled

        render_tab_state.preview_render = False

        for current_client in server.get_clients().values():
            if current_client.client_id not in camera_pose_backup_from_id:
                continue
            cam_position, cam_look_at, cam_up = camera_pose_backup_from_id.pop(
                current_client.client_id
            )
            current_client.camera.position = cam_position
            current_client.camera.look_at = cam_look_at
            current_client.camera.up_direction = cam_up
            current_client.flush()

        server.scene.set_global_visibility(True)

    camera_path = CameraPath(server, duration_number)
    camera_path.tension = tension_slider.value
    camera_path.default_fov = fov_degrees_slider.value / 180.0 * np.pi
    camera_path.default_transition_sec = transition_sec_number.value

    return handles
