from __future__ import annotations

import numpy as np

from ..rendering import render_s5_pybullet_demo_video, render_s5_pybullet_episode, render_sphere_episode


class S5RenderingMixin:
    def render_episode(self, scene, trajectory, output_path, **kwargs):
        geometry = dict((scene or {}).get("geometry", {}))
        camera_name = str(kwargs.get("camera", "default_3d"))
        presets = self.get_render_camera_presets()
        preset = dict(presets.get(camera_name, presets["default_3d"]))
        backend = str(kwargs.get("backend", preset.get("backend", "matplotlib"))).lower()
        traj = np.asarray(trajectory, dtype=float)[:, :3]
        sphere_center = geometry.get("sphere_center", self.sphere_center.tolist())
        sphere_radius = float(geometry.get("sphere_radius", self.sphere_radius))
        if backend in {"matplotlib", "mpl"}:
            return render_sphere_episode(
                trajectory=traj,
                output_path=output_path,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                cutpoints=kwargs.get("cutpoints"),
                title=kwargs.get("title", str(self.eval_tag)),
                elev=float(kwargs.get("elev", preset.get("elev", 24.0))),
                azim=float(kwargs.get("azim", preset.get("azim", 38.0))),
            )
        if backend == "pybullet":
            tool_axis = kwargs.get("tool_axis")
            if tool_axis is None:
                tool_axis = self._lookup_cached_tool_axis_trace(traj)
            if tool_axis is None:
                tool_axis = self._estimate_tool_axis_from_geometry(traj)
            return render_s5_pybullet_episode(
                trajectory=traj,
                output_path=output_path,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                cutpoints=kwargs.get("cutpoints"),
                overlay_cutpoints=kwargs.get("overlay_cutpoints"),
                tool_axis=np.asarray(tool_axis, dtype=float),
                title=kwargs.get("title", str(self.eval_tag)),
                center_world=kwargs.get("center_world", self.pybullet_world_center),
                world_scale=float(kwargs.get("world_scale", self.pybullet_world_scale)),
                main_yaw=float(kwargs.get("main_yaw", preset.get("main_yaw", 42.0))),
                inset_yaw=float(kwargs.get("inset_yaw", preset.get("inset_yaw", 205.0))),
                main_pitch=float(kwargs.get("main_pitch", -18.0)),
                inset_pitch=float(kwargs.get("inset_pitch", -16.0)),
                main_distance=float(kwargs.get("main_distance", 1.32)),
                inset_distance=float(kwargs.get("inset_distance", 1.36)),
                tube_radius=float(kwargs.get("tube_radius", 0.0065)),
            )
        if backend in {"pybullet_video", "video"}:
            tool_axis = kwargs.get("tool_axis")
            if tool_axis is None:
                tool_axis = self._lookup_cached_tool_axis_trace(traj)
            if tool_axis is None:
                tool_axis = self._estimate_tool_axis_from_geometry(traj)
            return render_s5_pybullet_demo_video(
                trajectory=traj,
                output_path=output_path,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                cutpoints=kwargs.get("cutpoints"),
                tool_axis=np.asarray(tool_axis, dtype=float),
                joint_positions=kwargs.get("joint_positions"),
                title=kwargs.get("title", str(self.eval_tag)),
                center_world=kwargs.get("center_world", self.pybullet_world_center),
                world_scale=float(kwargs.get("world_scale", self.pybullet_world_scale)),
                urdf_path=kwargs.get("urdf_path", self.pybullet_ur5_urdf_path),
                ur5_base_xyz=kwargs.get("ur5_base_xyz", self.pybullet_ur5_base_xyz),
                ur5_base_rpy=kwargs.get("ur5_base_rpy", self.pybullet_ur5_base_rpy),
                gui=int(kwargs.get("gui", 1)),
                fps=float(kwargs.get("fps", 30.0)),
                width=int(kwargs.get("width", 1024)),
                height=int(kwargs.get("height", 768)),
                render_frame_stride=int(kwargs.get("render_frame_stride", 1)),
                realtime=bool(kwargs.get("realtime", False)),
                gui_hold_seconds=float(kwargs.get("gui_hold_seconds", 0.0)),
                camera_yaw=float(kwargs.get("camera_yaw", preset.get("main_yaw", 90.0))),
                camera_pitch=float(kwargs.get("camera_pitch", -34.0)),
                camera_distance=float(kwargs.get("camera_distance", 1.45)),
                camera_target=kwargs.get("camera_target"),
                camera_fov=float(kwargs.get("camera_fov", 38.0)),
                tube_radius=float(kwargs.get("tube_radius", 0.0055)),
                stage4_shell_offset=float(
                    kwargs.get("stage4_shell_offset", self.get_true_constraints().get("surface_near_target", 0.0))
                ),
                sphere_texture_name=str(kwargs.get("sphere_texture_name", "")),
                trace_stride=int(kwargs.get("trace_stride", 1)),
                draw_stage_trace=bool(kwargs.get("draw_stage_trace", True)),
                draw_executed_trace=bool(kwargs.get("draw_executed_trace", True)),
                trace_width=float(kwargs.get("trace_width", 3.0)),
                draw_current_marker=bool(kwargs.get("draw_current_marker", False)),
                hide_gripper=bool(kwargs.get("hide_gripper", True)),
                draw_tool_bar=bool(kwargs.get("draw_tool_bar", False)),
                tool_bar_length=float(kwargs.get("tool_bar_length", 0.205)),
                tool_bar_radius=float(kwargs.get("tool_bar_radius", 0.005)),
                suppress_urdf_warnings=bool(
                    kwargs.get("suppress_urdf_warnings", self.pybullet_suppress_urdf_warnings)
                ),
                connect_client=bool(kwargs.get("connect_client", True)),
                feature_overlay=bool(kwargs.get("feature_overlay", False)),
                feature_overlay_features=kwargs.get("feature_overlay_features"),
                feature_overlay_names=kwargs.get("feature_overlay_names"),
                feature_overlay_units=kwargs.get("feature_overlay_units"),
                feature_overlay_specs=kwargs.get("feature_overlay_specs"),
                feature_overlay_true_constraints=kwargs.get("feature_overlay_true_constraints"),
                feature_overlay_title=kwargs.get("feature_overlay_title"),
                playback_speed=float(kwargs.get("playback_speed", 1.0)),
                playback_label=kwargs.get("playback_label"),
                save_frame_indices=kwargs.get("save_frame_indices"),
                save_frame_dir=kwargs.get("save_frame_dir"),
                save_frame_prefix=str(kwargs.get("save_frame_prefix", "s5_frame")),
            )
        raise ValueError(f"Unsupported S5 render backend '{backend}'.")

        return F[:, 0], F[:, 2]


