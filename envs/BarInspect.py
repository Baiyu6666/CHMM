from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .base import TaskBundle


@dataclass(frozen=True)
class BarInspectScene:
    bar_pose_optitrack: np.ndarray | None = None
    obstacle_pose_optitrack: np.ndarray | None = None
    obstacle_poses_optitrack: np.ndarray | None = None
    bar_lateral_centerline: dict | None = None

    def __post_init__(self):
        if self.bar_pose_optitrack is not None:
            bar_pose = np.asarray(self.bar_pose_optitrack, dtype=float)
            if bar_pose.shape not in {(7,)} and not (
                bar_pose.ndim == 2 and bar_pose.shape[1] == 7
            ):
                raise ValueError("bar_pose_optitrack must have shape (7,) or (samples, 7).")
            if not np.all(np.isfinite(bar_pose)):
                raise ValueError("bar_pose_optitrack must be finite.")
            object.__setattr__(self, "bar_pose_optitrack", bar_pose.copy())

        if self.obstacle_pose_optitrack is not None:
            obstacle_pose = np.asarray(self.obstacle_pose_optitrack, dtype=float)
            if obstacle_pose.shape not in {(3,), (7,)} and not (
                obstacle_pose.ndim == 2 and obstacle_pose.shape[1] in (3, 7)
            ):
                raise ValueError(
                    "obstacle_pose_optitrack must have shape (3,), (7,), "
                    "(samples, 3), or (samples, 7)."
                )
            if not np.all(np.isfinite(obstacle_pose)):
                raise ValueError("obstacle_pose_optitrack must be finite.")
            object.__setattr__(self, "obstacle_pose_optitrack", obstacle_pose.copy())

        if self.obstacle_poses_optitrack is not None:
            obstacle_poses = np.asarray(self.obstacle_poses_optitrack, dtype=float)
            if obstacle_poses.ndim != 2 or obstacle_poses.shape[1] not in (3, 7):
                raise ValueError(
                    "obstacle_poses_optitrack must have shape (obstacles, 3 or 7)."
                )
            if len(obstacle_poses) < 1 or not np.all(np.isfinite(obstacle_poses)):
                raise ValueError("obstacle_poses_optitrack must be nonempty and finite.")
            object.__setattr__(
                self, "obstacle_poses_optitrack", obstacle_poses.copy()
            )
        if self.bar_lateral_centerline is not None:
            object.__setattr__(
                self, "bar_lateral_centerline", dict(self.bar_lateral_centerline)
            )

    def to_dict(self):
        return {
            "bar_pose_optitrack": (
                None
                if self.bar_pose_optitrack is None
                else np.asarray(self.bar_pose_optitrack, dtype=float).tolist()
            ),
            "obstacle_pose_optitrack": (
                None
                if self.obstacle_pose_optitrack is None
                else np.asarray(self.obstacle_pose_optitrack, dtype=float).tolist()
            ),
            "obstacle_poses_optitrack": (
                None
                if self.obstacle_poses_optitrack is None
                else np.asarray(self.obstacle_poses_optitrack, dtype=float).tolist()
            ),
            "bar_lateral_centerline": (
                None
                if self.bar_lateral_centerline is None
                else dict(self.bar_lateral_centerline)
            ),
        }


class BarInspectEnv:
    """Four-stage bar-inspection task expressed in the robot-base frame.

    A trajectory sample is ``[x, y, z, qx, qy, qz, qw]`` for the IIWA
    end-effector link. Position-dependent features are evaluated at the flange
    origin, assuming the inspection camera is mounted directly on the flange.

    Stages (zero-based indices in ``constraint_specs``):

    0. approach the bar while keeping obstacle clearance;
    1. scan 15 cm along the bar at fixed table standoff and 90 degree tool pitch;
    2. scan another 15 cm at the same table standoff and 30 degree tool pitch;
    3. leave the bar without a task constraint.

    Only the configured tool axis is used for orientation features.  Rotation
    about that axis is deliberately absent from the feature set and is free.
    The steel-bar pose supplies direction only for learned task features;
    ``table_dist`` is measured from a separately calibrated fixed table plane.
    """

    task_name = "BarInspect"
    default_learning_features = (
        "obstacle_clearance",
        "table_dist",
        "bar_lateral_offset",
        "tool_pitch",
        "tool_roll",
    )

    def __init__(
        self,
        bar_surface_point=(0.0, 0.0, 0.0),
        bar_axis=(1.0, 0.0, 0.0),
        bar_axis_local=(1.0, 0.0, 0.0),
        bar_surface_offset_local=(0.0, 0.0, 0.0),
        optitrack_to_robot_rotation=((0.0, 0.0, 1.0),
                                     (1.0, 0.0, 0.0),
                                     (0.0, 1.0, 0.0)),
        optitrack_to_robot_translation=(0.0, 0.0, 0.0),
        table_surface_point=(0.0, 0.0, 0.14584),
        table_normal=(0.0, 0.0, 1.0),
        bar_outline_u=(-0.11177, 0.18629),
        bar_outline_v=(-0.01787, 0.04452),
        bar_height=0.02780,
        scan_start_progress=-0.18,
        stage2_scan_distance=0.15,
        stage3_scan_distance=0.15,
        scan_lateral_offset=0.0,
        scan_standoff=0.080,
        stage2_pitch_deg=90.0,
        stage3_pitch_deg=70.0,
        obstacle_center=(-0.285, -0.090, 0.14584),
        obstacle_radius=0.025,
        obstacle_min_clearance=0.040,
        nominal_start_tcp=(-0.40, -0.27, 0.27),
        tcp_offset_local=(0.0, 0.0, 0.0),
        tool_axis_local=(0.0, 0.0, 1.0),
        tool_lateral_axis_local=(0.0, 1.0, 0.0),
        task_frame_snapshot_policy="frozen_per_task",
        seg_lengths=(38, 32, 32, 28),
        seg_length_jitter=(5, 4, 4, 5),
        dt=0.10,
        position_noise_std=0.0012,
        pitch_noise_deg=1.0,
        plane_noise_deg=0.8,
        roll_variation_deg=35.0,
        scene=None,
    ):
        self.bar_surface_point = np.asarray(bar_surface_point, dtype=float).reshape(3)
        self.table_surface_point = np.asarray(table_surface_point, dtype=float).reshape(3)
        self.table_normal = self._unit(table_normal, "table_normal")
        raw_bar_axis = np.asarray(bar_axis, dtype=float).reshape(3)
        raw_bar_axis = raw_bar_axis - self.table_normal * float(np.dot(raw_bar_axis, self.table_normal))
        self.bar_axis = self._unit(raw_bar_axis, "bar_axis projected into the table plane")
        self.bar_axis_local = self._unit(bar_axis_local, "bar_axis_local")
        self.bar_surface_offset_local = np.asarray(
            bar_surface_offset_local, dtype=float
        ).reshape(3)
        self.optitrack_to_robot_rotation = np.asarray(
            optitrack_to_robot_rotation, dtype=float
        ).reshape(3, 3)
        self.optitrack_to_robot_translation = np.asarray(
            optitrack_to_robot_translation, dtype=float
        ).reshape(3)
        if not np.allclose(
            self.optitrack_to_robot_rotation.T @ self.optitrack_to_robot_rotation,
            np.eye(3),
            atol=1e-6,
        ) or not np.isclose(np.linalg.det(self.optitrack_to_robot_rotation), 1.0, atol=1e-6):
            raise ValueError("optitrack_to_robot_rotation must be a proper rotation matrix.")
        self.bar_lateral = self._unit(
            np.cross(self.table_normal, self.bar_axis),
            "table_normal x bar_axis",
        )

        self.bar_outline_u = np.asarray(bar_outline_u, dtype=float).reshape(2)
        self.bar_outline_v = np.asarray(bar_outline_v, dtype=float).reshape(2)
        if self.bar_outline_u[1] <= self.bar_outline_u[0]:
            raise ValueError("bar_outline_u must contain increasing pivot-relative bounds.")
        if self.bar_outline_v[1] <= self.bar_outline_v[0]:
            raise ValueError("bar_outline_v must contain increasing pivot-relative bounds.")
        self.bar_length = float(self.bar_outline_u[1] - self.bar_outline_u[0])
        self.bar_width = float(self.bar_outline_v[1] - self.bar_outline_v[0])
        self.bar_height = float(bar_height)
        self.scan_start_progress = float(scan_start_progress)
        self.stage2_scan_distance = float(stage2_scan_distance)
        self.stage3_scan_distance = float(stage3_scan_distance)
        self.scan_lateral_offset = float(scan_lateral_offset)
        self.scan_standoff = float(scan_standoff)
        self.stage2_pitch = float(np.deg2rad(stage2_pitch_deg))
        self.stage3_pitch = float(np.deg2rad(stage3_pitch_deg))

        self.obstacle_center = np.asarray(obstacle_center, dtype=float).reshape(3)
        self.obstacle_radius = float(obstacle_radius)
        self.obstacle_min_clearance = float(obstacle_min_clearance)
        self.nominal_start_tcp = np.asarray(nominal_start_tcp, dtype=float).reshape(3)
        self.task_frame_snapshot_policy = str(task_frame_snapshot_policy)
        if self.task_frame_snapshot_policy != "frozen_per_task":
            raise ValueError(
                "task_frame_snapshot_policy must be 'frozen_per_task'."
            )

        self.tcp_offset_local = np.asarray(tcp_offset_local, dtype=float).reshape(3)
        self.tool_axis_local = self._unit(tool_axis_local, "tool_axis_local")
        raw_lateral = np.asarray(tool_lateral_axis_local, dtype=float).reshape(3)
        raw_lateral = raw_lateral - self.tool_axis_local * float(
            np.dot(raw_lateral, self.tool_axis_local)
        )
        self.tool_lateral_axis_local = self._unit(raw_lateral, "tool_lateral_axis_local")
        self.tool_third_axis_local = self._unit(
            np.cross(self.tool_axis_local, self.tool_lateral_axis_local),
            "tool local third axis",
        )
        self.local_tool_frame = np.column_stack(
            [self.tool_axis_local, self.tool_lateral_axis_local, self.tool_third_axis_local]
        )

        self.seg_lengths = tuple(int(value) for value in seg_lengths)
        self.seg_length_jitter = tuple(int(value) for value in seg_length_jitter)
        self.dt = float(dt)
        self.position_noise_std = float(position_noise_std)
        self.pitch_noise = float(np.deg2rad(pitch_noise_deg))
        self.plane_noise = float(np.deg2rad(plane_noise_deg))
        self.roll_variation = float(np.deg2rad(roll_variation_deg))

        if len(self.seg_lengths) != 4 or len(self.seg_length_jitter) != 4:
            raise ValueError("BarInspect requires four segment lengths and four jitters.")
        if min(self.seg_lengths) < 4 or min(self.seg_length_jitter) < 0:
            raise ValueError("BarInspect segment lengths must be >= 4 and jitters non-negative.")
        if self.dt <= 0.0 or self.stage2_scan_distance <= 0.0 or self.stage3_scan_distance <= 0.0:
            raise ValueError("dt and both scan distances must be positive.")
        if self.scan_standoff <= 0.0 or self.obstacle_radius <= 0.0:
            raise ValueError("scan_standoff and obstacle_radius must be positive.")

        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self.stage_specs = self.get_stage_specs()
        self.hide_true_stage_end_markers = True
        self.scene = self._coerce_scene(scene)
        self.demo_scenes = []

    @staticmethod
    def _unit(value, name="vector"):
        vector = np.asarray(value, dtype=float).reshape(3)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError(f"{name} must be finite and non-zero.")
        return vector / norm

    @staticmethod
    def _smoothstep(value):
        u = np.asarray(value, dtype=float)
        return u * u * (3.0 - 2.0 * u)

    @staticmethod
    def _coerce_scene(scene):
        if scene is None or isinstance(scene, BarInspectScene):
            return scene
        if not isinstance(scene, dict):
            raise TypeError("scene must be a BarInspectScene, dict, or None.")
        return BarInspectScene(
            bar_pose_optitrack=scene.get("bar_pose_optitrack"),
            obstacle_pose_optitrack=scene.get("obstacle_pose_optitrack"),
            obstacle_poses_optitrack=scene.get("obstacle_poses_optitrack"),
            bar_lateral_centerline=scene.get("bar_lateral_centerline"),
        )

    def set_scene(self, scene):
        self.scene = self._coerce_scene(scene)

    def set_demo_scenes(self, scenes):
        self.demo_scenes = [self._coerce_scene(scene) for scene in scenes]

    def get_demo_scene(self, demo_index):
        index = int(demo_index)
        if not 0 <= index < len(self.demo_scenes):
            raise IndexError(f"No BarInspect scene is available for demo {index}.")
        return self.demo_scenes[index]

    def get_top_view_scene_geometry(self, demo_index=None):
        active_scene = self.scene
        if demo_index is not None and self.demo_scenes:
            active_scene = self.get_demo_scene(demo_index)
        elif self.demo_scenes:
            active_scene = self.demo_scenes[0]
        probe = np.zeros((1, 7), dtype=float)
        bar_reference, bar_axis, bar_lateral = self._bar_geometry_trace(
            probe,
            scene=active_scene,
        )
        obstacle_center = self._obstacle_center_trace(probe, scene=active_scene)[0]
        reference_xy = bar_reference[0, :2]
        axis_xy = bar_axis[0, :2]
        lateral_xy = bar_lateral[0, :2]
        u_min, u_max = (float(value) for value in self.bar_outline_u)
        v_min, v_max = (float(value) for value in self.bar_outline_v)
        bar_corners = np.asarray(
            [
                reference_xy + u_min * axis_xy + v_min * lateral_xy,
                reference_xy + u_max * axis_xy + v_min * lateral_xy,
                reference_xy + u_max * axis_xy + v_max * lateral_xy,
                reference_xy + u_min * axis_xy + v_max * lateral_xy,
            ],
            dtype=float,
        )
        return {
            "bar_corners_xy": bar_corners,
            "bar_reference_xy": reference_xy.copy(),
            "bar_axis_xy": axis_xy.copy(),
            "obstacle_center_xy": obstacle_center[:2].copy(),
            "obstacle_radius": float(self.obstacle_radius),
        }

    @staticmethod
    def _pose_trace(pose, length, name):
        values = np.asarray(pose, dtype=float)
        if values.ndim == 1:
            return np.repeat(values[None, :], int(length), axis=0)
        if len(values) != int(length):
            raise ValueError(
                f"{name} contains {len(values)} samples for a {length}-sample trajectory."
            )
        return values

    def _obstacle_center_trace(self, trajectory, scene=None):
        active_scene = self.scene if scene is None else self._coerce_scene(scene)
        if active_scene is not None and active_scene.obstacle_pose_optitrack is not None:
            poses = self._pose_trace(
                active_scene.obstacle_pose_optitrack,
                len(trajectory),
                "obstacle_pose_optitrack",
            )
            poses = np.repeat(poses[0:1], len(poses), axis=0)
            return (
                poses[:, :3] @ self.optitrack_to_robot_rotation.T
                + self.optitrack_to_robot_translation[None, :]
            )
        return np.repeat(self.obstacle_center[None, :], len(trajectory), axis=0)

    def _bar_geometry_trace(self, trajectory, scene=None):
        demo = np.asarray(trajectory, dtype=float)
        active_scene = self.scene if scene is None else self._coerce_scene(scene)
        if active_scene is None or active_scene.bar_pose_optitrack is None:
            surface_point = np.repeat(self.bar_surface_point[None, :], len(demo), axis=0)
            bar_axis = np.repeat(self.bar_axis[None, :], len(demo), axis=0)
            bar_lateral = np.repeat(self.bar_lateral[None, :], len(demo), axis=0)
            return surface_point, bar_axis, bar_lateral

        bar_pose = self._pose_trace(
            active_scene.bar_pose_optitrack,
            len(demo),
            "bar_pose_optitrack",
        )
        # The bar/table task frame is a per-execution scene snapshot.  Never
        # let live marker jitter or a later bar movement deform one trajectory.
        bar_pose = np.repeat(bar_pose[0:1], len(bar_pose), axis=0)
        self._quat_normalize(bar_pose[:, 3:7])

        tracker_rotation = self._quat_to_matrix(bar_pose[:, 3:7])
        bar_rotation = np.einsum(
            "ij,tjk->tik", self.optitrack_to_robot_rotation, tracker_rotation
        )
        bar_position = (
            bar_pose[:, :3] @ self.optitrack_to_robot_rotation.T
            + self.optitrack_to_robot_translation[None, :]
        )
        surface_point = bar_position + np.einsum(
            "tij,j->ti", bar_rotation, self.bar_surface_offset_local
        )
        bar_axis = np.einsum("tij,j->ti", bar_rotation, self.bar_axis_local)
        bar_axis -= np.outer(bar_axis @ self.table_normal, self.table_normal)
        axis_norm = np.linalg.norm(bar_axis, axis=1, keepdims=True)
        if np.any(axis_norm <= 1e-9):
            raise ValueError("An OptiTrack bar +X axis is parallel to the configured table normal.")
        bar_axis /= axis_norm
        bar_lateral = np.cross(self.table_normal[None, :], bar_axis)
        bar_lateral /= np.maximum(np.linalg.norm(bar_lateral, axis=1, keepdims=True), 1e-12)
        return surface_point, bar_axis, bar_lateral

    @staticmethod
    def _quat_normalize(quaternion):
        quat = np.asarray(quaternion, dtype=float)
        norms = np.linalg.norm(quat, axis=-1, keepdims=True)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
            raise ValueError("Trajectory contains an invalid quaternion.")
        return quat / norms

    @classmethod
    def _quat_to_matrix(cls, quaternion):
        q = cls._quat_normalize(quaternion)
        x, y, z, w = np.moveaxis(q, -1, 0)
        matrix = np.empty(q.shape[:-1] + (3, 3), dtype=float)
        matrix[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
        matrix[..., 0, 1] = 2.0 * (x * y - z * w)
        matrix[..., 0, 2] = 2.0 * (x * z + y * w)
        matrix[..., 1, 0] = 2.0 * (x * y + z * w)
        matrix[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
        matrix[..., 1, 2] = 2.0 * (y * z - x * w)
        matrix[..., 2, 0] = 2.0 * (x * z - y * w)
        matrix[..., 2, 1] = 2.0 * (y * z + x * w)
        matrix[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
        return matrix

    @staticmethod
    def _matrix_to_quat(matrix):
        rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = 2.0 * np.sqrt(max(trace + 1.0, 0.0))
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=float,
            )
        else:
            index = int(np.argmax(np.diag(rotation)))
            if index == 0:
                scale = 2.0 * np.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 0.0))
                quaternion = np.array(
                    [0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale,
                     (rotation[0, 2] + rotation[2, 0]) / scale,
                     (rotation[2, 1] - rotation[1, 2]) / scale],
                    dtype=float,
                )
            elif index == 1:
                scale = 2.0 * np.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 0.0))
                quaternion = np.array(
                    [(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                     (rotation[1, 2] + rotation[2, 1]) / scale,
                     (rotation[0, 2] - rotation[2, 0]) / scale],
                    dtype=float,
                )
            else:
                scale = 2.0 * np.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 0.0))
                quaternion = np.array(
                    [(rotation[0, 2] + rotation[2, 0]) / scale,
                     (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale,
                     (rotation[1, 0] - rotation[0, 1]) / scale],
                    dtype=float,
                )
        quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1e-12)
        return quaternion if quaternion[3] >= 0.0 else -quaternion

    @classmethod
    def _quat_slerp(cls, first, second, values):
        q0 = cls._quat_normalize(np.asarray(first, dtype=float).reshape(4))
        q1 = cls._quat_normalize(np.asarray(second, dtype=float).reshape(4))
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        u = np.asarray(values, dtype=float).reshape(-1)
        if dot > 0.9995:
            result = (1.0 - u[:, None]) * q0[None, :] + u[:, None] * q1[None, :]
            return cls._quat_normalize(result)
        theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
        denominator = np.sin(theta)
        result = (
            np.sin((1.0 - u) * theta)[:, None] * q0[None, :]
            + np.sin(u * theta)[:, None] * q1[None, :]
        ) / denominator
        return cls._quat_normalize(result)

    def _tool_rotation(self, pitch, plane_error=0.0, roll=0.0):
        in_plane_axis = (
            np.cos(float(pitch)) * self.bar_axis
            - np.sin(float(pitch)) * self.table_normal
        )
        tool_axis = (
            np.cos(float(plane_error)) * in_plane_axis
            + np.sin(float(plane_error)) * self.bar_lateral
        )
        tool_axis = self._unit(tool_axis, "generated tool axis")
        base_lateral = self.bar_lateral - tool_axis * float(np.dot(self.bar_lateral, tool_axis))
        base_lateral = self._unit(base_lateral, "generated tool lateral axis")
        base_third = self._unit(np.cross(tool_axis, base_lateral), "generated tool third axis")
        world_lateral = np.cos(float(roll)) * base_lateral + np.sin(float(roll)) * base_third
        world_third = self._unit(np.cross(tool_axis, world_lateral), "rolled tool third axis")
        world_frame = np.column_stack([tool_axis, world_lateral, world_third])
        return world_frame @ self.local_tool_frame.T

    def _scan_tcp(self, progress, lateral=None, standoff=None):
        lateral_value = self.scan_lateral_offset if lateral is None else np.asarray(lateral, dtype=float)
        standoff_value = self.scan_standoff if standoff is None else np.asarray(standoff, dtype=float)
        progress_value, lateral_value, standoff_value = np.broadcast_arrays(
            np.asarray(progress, dtype=float), lateral_value, standoff_value
        )
        scan_origin = self.bar_surface_point + self.table_normal * float(
            np.dot(self.table_surface_point - self.bar_surface_point, self.table_normal)
        )
        return (
            scan_origin[None, :]
            + progress_value.reshape(-1, 1) * self.bar_axis[None, :]
            + lateral_value.reshape(-1, 1) * self.bar_lateral[None, :]
            + standoff_value.reshape(-1, 1) * self.table_normal[None, :]
        )

    @staticmethod
    def _resample_polyline(points, count):
        vertices = np.asarray(points, dtype=float).reshape(-1, 3)
        segment_lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        if cumulative[-1] <= 1e-12:
            return np.repeat(vertices[:1], int(count), axis=0)
        targets = np.linspace(0.0, cumulative[-1], int(count))
        output = np.empty((int(count), 3), dtype=float)
        for dimension in range(3):
            output[:, dimension] = np.interp(targets, cumulative, vertices[:, dimension])
        return output

    @staticmethod
    def _smooth_noise(rng, length, dimensions, scale, knots=7):
        if float(scale) <= 0.0:
            return np.zeros((int(length), int(dimensions)), dtype=float)
        knot_count = max(3, min(int(knots), int(length)))
        knot_values = rng.normal(scale=float(scale), size=(knot_count, int(dimensions)))
        knot_values[0] = 0.0
        knot_values[-1] = 0.0
        knot_t = np.linspace(0.0, 1.0, knot_count)
        sample_t = np.linspace(0.0, 1.0, int(length))
        output = np.empty((int(length), int(dimensions)), dtype=float)
        for dimension in range(int(dimensions)):
            output[:, dimension] = np.interp(sample_t, knot_t, knot_values[:, dimension])
        return output

    def _project_outside_obstacle(self, points, margin=0.002):
        output = np.asarray(points, dtype=float).copy()
        safe_radius = self.obstacle_radius + self.obstacle_min_clearance + float(margin)
        relative = output - self.obstacle_center[None, :]
        radial = relative - np.outer(relative @ self.table_normal, self.table_normal)
        distance = np.linalg.norm(radial, axis=1)
        mask = distance < safe_radius
        if np.any(mask):
            direction = radial[mask] / np.maximum(distance[mask, None], 1e-12)
            zero = distance[mask] <= 1e-12
            if np.any(zero):
                direction[zero] = self.bar_lateral
            output[mask] += (safe_radius - distance[mask])[:, None] * direction
        return output

    def _pose_from_tcp_and_quaternion(self, tcp_position, quaternion):
        tcp = np.asarray(tcp_position, dtype=float).reshape(-1, 3)
        quat = self._quat_normalize(np.asarray(quaternion, dtype=float).reshape(-1, 4))
        rotations = self._quat_to_matrix(quat)
        flange = tcp - np.einsum("tij,j->ti", rotations, self.tcp_offset_local)
        return np.column_stack([flange, quat])

    def get_feature_schema(self):
        return [
            {"id": 0, "column_idx": 0, "name": "obstacle_clearance", "unit": "m",
             "frame": "frozen scene snapshot",
             "description": "EE radial clearance from the infinite vertical obstacle cylinder; positive outside"},
            {"id": 1, "column_idx": 1, "name": "table_dist", "unit": "m",
             "frame": "bar_table_task.z / calibrated table",
             "description": "Signed EE distance to the fixed calibrated table plane along its normal"},
            {"id": 2, "column_idx": 2, "name": "bar_lateral_offset", "unit": "m",
             "frame": "bar_table_task.y",
             "description": "Signed in-table offset from the bar centerline"},
            {"id": 3, "column_idx": 3, "name": "tool_pitch", "unit": "rad",
             "frame": "bar_table_task orientation",
             "description": "Signed tool-axis pitch above the table plane; downward vertical is pi/2"},
            {"id": 4, "column_idx": 4, "name": "tool_roll", "unit": "rad",
             "frame": "bar_table_task orientation",
             "description": "Signed tool-axis deviation from the plane spanned by bar axis and table normal"},
            {"id": 5, "column_idx": 5, "name": "motion_axis_err", "unit": "rad",
             "description": "Signed bar-relative motion-direction error; diagnostic and excluded from the default learner"},
            {"id": 6, "column_idx": 6, "name": "speed", "unit": "m/s",
             "description": "Camera/flange-point translational speed, diagnostic and excluded from the default learner"},
            {"id": 7, "column_idx": 7, "name": "angular_speed", "unit": "rad/s",
             "description": "End-effector angular speed, diagnostic and excluded from the default learner"},
        ]

    def get_true_constraints(self):
        return {
            "obstacle_min_clearance": float(self.obstacle_min_clearance),
            "table_distance_target": float(self.scan_standoff),
            "bar_lateral_target": float(self.scan_lateral_offset),
            "stage2_pitch_target": float(self.stage2_pitch),
            "stage3_pitch_target": float(self.stage3_pitch),
            "tool_roll_target": 0.0,
        }

    def get_constraint_specs(self):
        stage_scan_common = [
            {"feature_name": "table_dist", "semantics": "target_value",
             "oracle_key": "table_distance_target"},
            {"feature_name": "bar_lateral_offset", "semantics": "target_value",
             "oracle_key": "bar_lateral_target"},
            {"feature_name": "tool_roll", "semantics": "target_value",
             "oracle_key": "tool_roll_target"},
        ]
        specs = [
            {"feature_name": "obstacle_clearance", "stage": 0, "semantics": "lower_bound",
             "oracle_key": "obstacle_min_clearance"}
        ]
        for stage, pitch_key in ((1, "stage2_pitch_target"), (2, "stage3_pitch_target")):
            specs.extend([{**spec, "stage": stage} for spec in stage_scan_common])
            specs.append(
                {"feature_name": "tool_pitch", "stage": stage, "semantics": "target_value",
                 "oracle_key": pitch_key}
            )
        return specs

    def get_stage_specs(self):
        return [
            {"stage": 0, "name": "approach", "description": "Approach the bar and avoid the obstacle"},
            {"stage": 1, "name": "vertical_scan", "distance_m": float(self.stage2_scan_distance),
             "pitch_deg": float(np.rad2deg(self.stage2_pitch))},
            {"stage": 2, "name": "oblique_scan", "distance_m": float(self.stage3_scan_distance),
             "pitch_deg": float(np.rad2deg(self.stage3_pitch))},
            {"stage": 3, "name": "free_depart", "description": "Leave with no task constraint"},
        ]

    def get_observation_spec(self):
        return {
            "trajectory_columns": ["ee_x", "ee_y", "ee_z", "qx", "qy", "qz", "qw"],
            "quaternion_order": "xyzw",
            "position_frame": "iiwa base/world frame used by the demo recorder",
            "position_evaluation_point": "camera/flange point = end-effector position + R * tcp_offset_local",
            "tcp_offset_local": self.tcp_offset_local.tolist(),
            "table_plane": {
                "surface_point_base": self.table_surface_point.tolist(),
                "normal_base": self.table_normal.tolist(),
                "table_dist_definition": "dot(ee_position - surface_point_base, normal_base)",
                "calibration_source": "demo_surface stable contact segment, 2026-08-19",
            },
            "optitrack_obstacle_pose": {
                "topic": "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14",
                "columns": ["obstacle_x", "obstacle_y", "obstacle_z", "qx", "qy", "qz", "qw"],
                "scene_field": "obstacle_pose_optitrack",
                "geometry": "infinite cylinder parallel to table_normal",
                "radius_m": float(self.obstacle_radius),
                "orientation_usage": "ignored",
                "fallback": "static obstacle_center from the environment config",
            },
            "optitrack_bar_pose": {
                "topic": "/vrpn_client_node/baiyu_bar/pose_from_iiwa14",
                "columns": ["bar_x", "bar_y", "bar_z", "qx", "qy", "qz", "qw"],
                "bar_axis_local": self.bar_axis_local.tolist(),
                "bar_reference_offset_local": self.bar_surface_offset_local.tolist(),
                "tracker_to_robot_rotation": self.optitrack_to_robot_rotation.tolist(),
                "tracker_to_robot_translation": self.optitrack_to_robot_translation.tolist(),
                "scene_field": "bar_pose_optitrack",
                "usage": "bar direction and diagnostic bar-relative coordinates only; not table_dist",
                "fallback": "static bar reference point and bar axis from the environment config",
            },
            "orientation_definition": {
                "tool_axis_local": self.tool_axis_local.tolist(),
                "pitch_zero": "parallel to the table and pointing along positive bar axis",
                "pitch_positive": "toward the bar surface",
                "free_rotation": "rotation about tool_axis_local is intentionally not featured",
            },
            "task_frame": {
                "frame_id": "bar_table_task",
                "origin": "tracked bar center projected onto the calibrated table plane",
                "x_axis": "tracked bar local +X projected into the table plane",
                "z_axis": "calibrated table normal",
                "y_axis": "z_axis cross x_axis",
                "motive_orientation_usage": "bar local +X only",
                "snapshot_policy": self.task_frame_snapshot_policy,
            },
            "default_learning_features": list(self.default_learning_features),
            "feature_schema": self.get_feature_schema(),
        }

    def get_render_camera_presets(self):
        return {"default_3d": {"elev": 26.0, "azim": -55.0, "equal_aspect": True}}

    def get_asset_handles(self):
        return {
            "bar": {"type": "box", "length": self.bar_length, "width": self.bar_width,
                    "height": self.bar_height, "outline_u": self.bar_outline_u.tolist(),
                    "outline_v": self.bar_outline_v.tolist()},
            "obstacle": {
                "type": "cylinder",
                "radius": self.obstacle_radius,
                "axis": "table_normal",
                "infinite_for_features": True,
            },
            "tool": {"type": "pose_axis", "axis": "local_+z"},
        }

    def sample_scene(self, seed=None, rng=None):
        return {
            "task_name": self.task_name,
            "geometry": {
                "bar_surface_point": self.bar_surface_point.tolist(),
                "bar_axis": self.bar_axis.tolist(),
                "bar_lateral": self.bar_lateral.tolist(),
                "table_surface_point": self.table_surface_point.tolist(),
                "table_normal": self.table_normal.tolist(),
                "bar_length": float(self.bar_length),
                "bar_width": float(self.bar_width),
                "bar_height": float(self.bar_height),
                "bar_outline_u": self.bar_outline_u.tolist(),
                "bar_outline_v": self.bar_outline_v.tolist(),
                "obstacle_center": self.obstacle_center.tolist(),
                "obstacle_radius": float(self.obstacle_radius),
            },
            "task": {
                "stage_specs": self.get_stage_specs(),
                "scan_standoff": float(self.scan_standoff),
                "scan_lateral_offset": float(self.scan_lateral_offset),
                "dt": float(self.dt),
            },
        }

    def generate_demo(self, seed=0, rng=None):
        local_rng = np.random.RandomState(int(seed)) if rng is None else rng
        lengths = [
            max(4, base + local_rng.randint(-jitter, jitter + 1))
            for base, jitter in zip(self.seg_lengths, self.seg_length_jitter)
        ]
        n1, n2, n3, n4 = lengths

        scan_entry = self._scan_tcp([self.scan_start_progress])[0]
        start = self.nominal_start_tcp + local_rng.normal(
            scale=np.array([0.025, 0.035, 0.025], dtype=float), size=3
        )
        obstacle_to_line = 0.5 * (start + scan_entry) - self.obstacle_center
        lateral_sign = np.sign(float(np.dot(obstacle_to_line, self.bar_lateral)))
        detour_direction = (-1.0 if lateral_sign == 0.0 else lateral_sign) * self.bar_lateral
        detour = self.obstacle_center + (
            self.obstacle_radius + self.obstacle_min_clearance + 0.045
        ) * detour_direction
        detour += self.table_normal * float(
            np.dot(0.5 * (start + scan_entry) - detour, self.table_normal)
        )
        detour += local_rng.normal(scale=0.012, size=3)
        approach_tcp = self._resample_polyline([start, detour, scan_entry], n1)
        approach_tcp += self._smooth_noise(
            local_rng, n1, 3, self.position_noise_std * 2.0, knots=7
        )
        approach_tcp = self._project_outside_obstacle(approach_tcp)
        approach_tcp[0] = start
        approach_tcp[-1] = scan_entry

        stage2_u = np.arange(1, n2 + 1, dtype=float) / float(n2)
        stage3_u = np.arange(1, n3 + 1, dtype=float) / float(n3)
        stage2_lateral = self.scan_lateral_offset + self._smooth_noise(
            local_rng, n2, 1, self.position_noise_std, knots=6
        )[:, 0]
        stage2_height = self.scan_standoff + self._smooth_noise(
            local_rng, n2, 1, self.position_noise_std * 0.65, knots=6
        )[:, 0]
        stage3_lateral = self.scan_lateral_offset + self._smooth_noise(
            local_rng, n3, 1, self.position_noise_std, knots=6
        )[:, 0]
        stage3_height = self.scan_standoff + self._smooth_noise(
            local_rng, n3, 1, self.position_noise_std * 0.65, knots=6
        )[:, 0]
        stage2_tcp = self._scan_tcp(
            self.scan_start_progress + self.stage2_scan_distance * stage2_u,
            stage2_lateral,
            stage2_height,
        )
        stage3_tcp = self._scan_tcp(
            self.scan_start_progress
            + self.stage2_scan_distance
            + self.stage3_scan_distance * stage3_u,
            stage3_lateral,
            stage3_height,
        )

        depart_direction = self._unit(
            local_rng.uniform(-0.45, 0.45) * self.bar_axis
            + local_rng.choice([-1.0, 1.0]) * local_rng.uniform(0.55, 1.0) * self.bar_lateral
            + local_rng.uniform(0.45, 0.95) * self.table_normal,
            "departure direction",
        )
        depart_distance = local_rng.uniform(0.18, 0.28)
        depart_u = np.arange(1, n4 + 1, dtype=float) / float(n4)
        depart_curve = self._smoothstep(depart_u)
        depart_tcp = stage3_tcp[-1][None, :] + (
            depart_distance * depart_curve[:, None] * depart_direction[None, :]
        )
        depart_tcp += (
            0.025
            * np.sin(np.pi * depart_u)[:, None]
            * local_rng.uniform(-1.0, 1.0)
            * self.bar_axis[None, :]
        )

        stage2_pitch = self.stage2_pitch + self._smooth_noise(
            local_rng, n2, 1, self.pitch_noise, knots=6
        )[:, 0]
        stage3_pitch = self.stage3_pitch + self._smooth_noise(
            local_rng, n3, 1, self.pitch_noise, knots=6
        )[:, 0]
        stage2_plane = self._smooth_noise(local_rng, n2, 1, self.plane_noise, knots=6)[:, 0]
        stage3_plane = self._smooth_noise(local_rng, n3, 1, self.plane_noise, knots=6)[:, 0]
        roll_phase = local_rng.uniform(-np.pi, np.pi)
        stage2_roll = local_rng.uniform(-np.pi, np.pi) + self.roll_variation * np.sin(
            2.0 * np.pi * stage2_u + roll_phase
        )
        stage3_roll = local_rng.uniform(-np.pi, np.pi) + self.roll_variation * np.sin(
            2.0 * np.pi * stage3_u + roll_phase
        )
        stage2_quat = np.asarray(
            [
                self._matrix_to_quat(self._tool_rotation(pitch, plane, roll))
                for pitch, plane, roll in zip(stage2_pitch, stage2_plane, stage2_roll)
            ],
            dtype=float,
        )
        stage3_quat = np.asarray(
            [
                self._matrix_to_quat(self._tool_rotation(pitch, plane, roll))
                for pitch, plane, roll in zip(stage3_pitch, stage3_plane, stage3_roll)
            ],
            dtype=float,
        )

        start_quat = self._quat_normalize(local_rng.normal(size=4))
        approach_quat = self._quat_slerp(
            start_quat,
            stage2_quat[0],
            self._smoothstep(np.linspace(0.0, 1.0, n1)),
        )
        end_quat = self._quat_normalize(local_rng.normal(size=4))
        depart_quat = self._quat_slerp(
            stage3_quat[-1],
            end_quat,
            self._smoothstep(depart_u),
        )

        trajectory = np.vstack(
            [
                self._pose_from_tcp_and_quaternion(approach_tcp, approach_quat),
                self._pose_from_tcp_and_quaternion(stage2_tcp, stage2_quat),
                self._pose_from_tcp_and_quaternion(stage3_tcp, stage3_quat),
                self._pose_from_tcp_and_quaternion(depart_tcp, depart_quat),
            ]
        )
        labels = np.concatenate(
            [np.full(length, stage, dtype=int) for stage, length in enumerate(lengths)]
        )
        cutpoints = np.where(np.diff(labels) != 0)[0].astype(int)
        return trajectory, labels, cutpoints

    def rollout_demo(self, scene, seed=None, rng=None, **kwargs):
        trajectory, labels, cutpoints = self.generate_demo(
            seed=0 if seed is None else int(seed), rng=rng
        )
        return {
            "trajectory": trajectory,
            "timestamps": np.arange(len(trajectory), dtype=float) * self.dt,
            "true_labels": labels,
            "true_cutpoints": cutpoints,
        }

    def compute_observation(self, latent_rollout, scene):
        trajectory = np.asarray(latent_rollout["trajectory"], dtype=float)
        bar_pose = latent_rollout.get("bar_pose")
        obstacle_pose = latent_rollout.get("obstacle_pose")
        observation_scene = self._coerce_scene(scene)
        if bar_pose is not None or obstacle_pose is not None:
            observation_scene = BarInspectScene(
                bar_pose_optitrack=bar_pose,
                obstacle_pose_optitrack=obstacle_pose,
            )
        return {
            "trajectory": trajectory,
            "timestamps": np.asarray(
                latent_rollout.get("timestamps", np.arange(len(trajectory), dtype=float) * self.dt),
                dtype=float,
            ),
            "features": self.compute_all_features_matrix(
                trajectory,
                scene=observation_scene,
            ),
            "true_labels": np.asarray(latent_rollout.get("true_labels", []), dtype=int),
            "true_cutpoints": np.asarray(latent_rollout.get("true_cutpoints", []), dtype=int),
            "feature_schema": self.get_feature_schema(),
            "observation_spec": self.get_observation_spec(),
            "scene": dict(scene or {}),
            "bar_pose": None if bar_pose is None else np.asarray(bar_pose, dtype=float),
            "obstacle_pose": (
                None if obstacle_pose is None else np.asarray(obstacle_pose, dtype=float)
            ),
        }

    def compute_all_features_matrix(self, traj, feat_ids=None, scene=None):
        trajectory = np.asarray(traj, dtype=float)
        if trajectory.ndim != 2 or trajectory.shape[1] < 7:
            raise ValueError(
                "BarInspect trajectories must contain [x, y, z, qx, qy, qz, qw]. "
                "Orientation features cannot be inferred from XYZ alone."
            )
        quaternion = self._quat_normalize(trajectory[:, 3:7])
        rotations = self._quat_to_matrix(quaternion)
        tcp = trajectory[:, :3] + np.einsum("tij,j->ti", rotations, self.tcp_offset_local)
        tool_axis = np.einsum("tij,j->ti", rotations, self.tool_axis_local)
        bar_reference_point, bar_axis, bar_lateral = self._bar_geometry_trace(
            trajectory,
            scene=scene,
        )
        obstacle_center = self._obstacle_center_trace(trajectory, scene=scene)

        relative_obstacle = tcp - obstacle_center
        obstacle_radial = relative_obstacle - np.outer(
            relative_obstacle @ self.table_normal, self.table_normal
        )
        obstacle_clearance = np.linalg.norm(obstacle_radial, axis=1) - self.obstacle_radius
        relative_bar = tcp - bar_reference_point
        relative_table = tcp - self.table_surface_point[None, :]
        table_dist = relative_table @ self.table_normal
        bar_progress = np.sum(relative_bar * bar_axis, axis=1)
        bar_lateral_offset = np.sum(relative_bar * bar_lateral, axis=1)
        down_component = -(tool_axis @ self.table_normal)
        forward_component = np.sum(tool_axis * bar_axis, axis=1)
        tool_pitch = np.arctan2(down_component, forward_component)
        tool_roll = np.arcsin(
            np.clip(np.sum(tool_axis * bar_lateral, axis=1), -1.0, 1.0)
        )

        velocity = np.zeros_like(tcp)
        if len(tcp) > 1:
            velocity[1:] = np.diff(tcp, axis=0) / self.dt
            velocity[0] = velocity[1]
        along_velocity = np.zeros(len(tcp), dtype=float)
        lateral_velocity = np.zeros(len(tcp), dtype=float)
        if len(tcp) > 1:
            along_velocity[1:] = np.diff(bar_progress) / self.dt
            lateral_velocity[1:] = np.diff(bar_lateral_offset) / self.dt
            along_velocity[0] = along_velocity[1]
            lateral_velocity[0] = lateral_velocity[1]
        motion_axis_err = np.arctan2(lateral_velocity, along_velocity)
        planar_speed = np.hypot(along_velocity, lateral_velocity)
        stationary = planar_speed <= 1e-8
        if np.any(stationary):
            motion_axis_err[stationary] = 0.0
        speed = np.linalg.norm(velocity, axis=1)

        angular_speed = np.zeros(len(trajectory), dtype=float)
        if len(trajectory) > 1:
            quaternion_dot = np.sum(quaternion[1:] * quaternion[:-1], axis=1)
            angle = 2.0 * np.arccos(np.clip(np.abs(quaternion_dot), -1.0, 1.0))
            angular_speed[1:] = angle / self.dt
            angular_speed[0] = angular_speed[1]

        features = np.column_stack(
            [
                obstacle_clearance,
                table_dist,
                bar_lateral_offset,
                tool_pitch,
                tool_roll,
                motion_axis_err,
                speed,
                angular_speed,
            ]
        )
        if not np.all(np.isfinite(features)):
            raise ValueError("BarInspect feature extraction produced non-finite values.")
        if feat_ids is None:
            return features
        if len(feat_ids) > 0 and isinstance(feat_ids[0], str):
            name_to_column = {spec["name"]: spec["column_idx"] for spec in self.feature_schema}
            feat_ids = [name_to_column[name] for name in feat_ids]
        return features[:, feat_ids]

    def compute_features_all(self, traj):
        features = self.compute_all_features_matrix(traj)
        return tuple(features[:, index] for index in range(features.shape[1]))


def load_BarInspect(
    n_demos=10,
    seed=2026,
    env_kwargs=None,
    demo_kwargs=None,
    processed_demo_path=None,
    **extra_env_kwargs,
):
    env_config = dict(env_kwargs or {})
    env_config.update(extra_env_kwargs)
    run_config = dict(demo_kwargs or {})
    env = BarInspectEnv(**env_config)

    if processed_demo_path is not None:
        data_path = Path(processed_demo_path).expanduser()
        if not data_path.is_absolute():
            data_path = Path(__file__).resolve().parents[1] / data_path
        with np.load(data_path, allow_pickle=False) as archive:
            required = {
                "trajectory",
                "features",
                "feature_names",
                "timestamps",
                "coarse_bounds_indices",
                "locked_bar_pose",
                "demo_obstacle_poses",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(
                    f"Processed BarInspect data {data_path} is missing keys: {missing}"
                )
            all_trajectories = np.asarray(archive["trajectory"], dtype=float)
            all_features = np.asarray(archive["features"], dtype=float)
            all_timestamps = np.asarray(archive["timestamps"], dtype=float)
            bounds = np.asarray(archive["coarse_bounds_indices"], dtype=int)
            feature_names = [str(name) for name in archive["feature_names"].tolist()]
            locked_bar_pose = np.asarray(archive["locked_bar_pose"], dtype=float)
            demo_obstacle_poses = np.asarray(
                archive["demo_obstacle_poses"],
                dtype=float,
            )
            cutpoint_annotation_kind = str(
                np.asarray(
                    archive["cutpoint_annotation_kind"]
                    if "cutpoint_annotation_kind" in archive.files
                    else "heuristic_coarse_stage_boundaries"
                ).item()
            )
            cutpoint_evaluation_role = str(
                np.asarray(
                    archive["cutpoint_evaluation_role"]
                    if "cutpoint_evaluation_role" in archive.files
                    else "heuristic_reference"
                ).item()
            )

        dataset_scene = BarInspectScene(
            bar_pose_optitrack=locked_bar_pose,
        )
        demo_scenes = [
            BarInspectScene(
                bar_pose_optitrack=locked_bar_pose,
                obstacle_pose_optitrack=obstacle_pose,
            )
            for obstacle_pose in demo_obstacle_poses
        ]
        env.set_scene(dataset_scene)
        env.set_demo_scenes(demo_scenes)

        expected_names = [spec["name"] for spec in env.get_feature_schema()]
        if feature_names != expected_names:
            raise ValueError(
                "Processed BarInspect feature order does not match the environment schema: "
                f"got {feature_names}, expected {expected_names}."
            )
        if bounds.ndim != 2 or bounds.shape[1] != 5:
            raise ValueError(
                "coarse_bounds_indices must have shape (num_demos, 5) for four stages."
            )
        if int(n_demos) > len(bounds):
            raise ValueError(
                f"Requested {n_demos} demos, but {data_path} contains only {len(bounds)}."
            )
        if demo_obstacle_poses.shape != (len(bounds), 7):
            raise ValueError(
                "demo_obstacle_poses must have shape (num_demos, 7), got "
                f"{demo_obstacle_poses.shape}."
            )

        demos = []
        features = []
        true_labels = []
        true_cutpoints = []
        timestamps = []
        scene_specs = []
        for demo_index, row in enumerate(bounds[: int(n_demos)]):
            begin, s2, s3, s4, end = (int(value) for value in row)
            if not (0 <= begin < s2 < s3 < s4 < end <= len(all_trajectories)):
                raise ValueError(
                    f"Invalid four-stage bounds for demo {demo_index}: {row.tolist()}"
                )
            demos.append(all_trajectories[begin:end].copy())
            features.append(all_features[begin:end].copy())
            stage_lengths = np.diff(row)
            true_labels.append(
                np.repeat(np.arange(4, dtype=int), stage_lengths.astype(int))
            )
            true_cutpoints.append(
                np.asarray([s2 - begin, s3 - begin, s4 - begin], dtype=int)
            )
            demo_time = all_timestamps[begin:end].copy()
            timestamps.append(demo_time - demo_time[0])
            scene_specs.append(
                {
                    "demo_index": int(demo_index),
                    "source": "processed_real_demo",
                    "processed_demo_path": str(data_path),
                    "recording_bounds": row.tolist(),
                    **demo_scenes[demo_index].to_dict(),
                }
            )

        return TaskBundle(
            name=env.task_name,
            demos=demos,
            features=features,
            env=env,
            true_taus=None,
            true_cutpoints=true_cutpoints,
            true_labels=true_labels,
            feature_schema=env.get_feature_schema(),
            true_constraints=env.get_true_constraints(),
            constraint_specs=env.get_constraint_specs(),
            meta={
                "seed": int(seed),
                "task_name": env.task_name,
                "data_source": "processed_real_demo",
                "processed_demo_path": str(data_path),
                "scene": dataset_scene.to_dict(),
                "cutpoint_annotation_kind": cutpoint_annotation_kind,
                "cutpoint_evaluation_role": cutpoint_evaluation_role,
                "cutpoint_annotations": {
                    "kind": cutpoint_annotation_kind,
                    "is_ground_truth": False,
                    "usage": "task-informed heuristic cutpoints used as stage supervision and evaluation references",
                },
                "stage_specs": env.get_stage_specs(),
                "scene_specs": scene_specs,
                "timestamps": timestamps,
                "observation_specs": env.get_observation_spec(),
                "render_camera_presets": env.get_render_camera_presets(),
                "asset_handles": env.get_asset_handles(),
                "default_learning_features": list(env.default_learning_features),
            },
        )

    demos = []
    features = []
    true_labels = []
    true_cutpoints = []
    timestamps = []
    scene_specs = []
    for demo_index in range(int(n_demos)):
        scene = env.sample_scene(seed=int(seed) + demo_index)
        scene["demo_index"] = int(demo_index)
        scene["rollout_seed"] = int(seed) + demo_index
        latent = env.rollout_demo(
            scene,
            seed=int(seed) + demo_index,
            **run_config,
        )
        observation = env.compute_observation(latent, scene)
        demos.append(np.asarray(observation["trajectory"], dtype=float))
        features.append(np.asarray(observation["features"], dtype=float))
        true_labels.append(np.asarray(observation["true_labels"], dtype=int))
        true_cutpoints.append(np.asarray(observation["true_cutpoints"], dtype=int))
        timestamps.append(np.asarray(observation["timestamps"], dtype=float))
        scene_specs.append(scene)

    return TaskBundle(
        name=env.task_name,
        demos=demos,
        features=features,
        env=env,
        true_taus=None,
        true_cutpoints=true_cutpoints,
        true_labels=true_labels,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(),
        constraint_specs=env.get_constraint_specs(),
        meta={
            "seed": int(seed),
            "task_name": env.task_name,
            "stage_specs": env.get_stage_specs(),
            "scene_specs": scene_specs,
            "timestamps": timestamps,
            "observation_specs": env.get_observation_spec(),
            "render_camera_presets": env.get_render_camera_presets(),
            "asset_handles": env.get_asset_handles(),
            "default_learning_features": list(env.default_learning_features),
        },
    )


__all__ = ["BarInspectEnv", "BarInspectScene", "load_BarInspect"]
