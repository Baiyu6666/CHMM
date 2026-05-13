from __future__ import annotations

import numpy as np

from .S4SlideInsert import S4SlideInsertEnv
from .base import TaskBundle
from .rendering import render_planar_episode
from .s4_pybullet_backend import simulate_s4_demo_from_reference


class S4SlideInsertRealisticEnv(S4SlideInsertEnv):
    """Robot-friendly S4 copy with a lateral clearance dimension.

    State is [x, y, z, theta] in the robot-friendly units used by this task.
    """

    def __init__(
        self,
        slot_half_width: float = 0.032,
        clearance_target: float = 0.0,
        clearance_align_max: float = 0.006,
        clearance_insert_max: float = 0.0035,
        normal_load_min: float = 0.40,
        rollout_backend: str = "analytic",
        observation_backend: str | None = None,
        pybullet_world_center=(0.55, 0.0, 0.52),
        pybullet_sim_dt: float = 1.0 / 120.0,
        pybullet_steps_per_sample=None,
        pybullet_gravity_z: float = -9.81,
        pybullet_solver_iterations: int = 80,
        pybullet_ur5_home_q=(0.0, -1.35, 1.85, -2.05, -1.57, 0.0),
        pybullet_ur5_tool_axis: str = "-x",
        pybullet_ur5_tip_offset: float = 0.0,
        pybullet_ur5_ee_link_index: int = -1,
        pybullet_ur5_urdf_path=None,
        pybullet_ur5_base_xyz=(0.0, 0.0, 0.0),
        pybullet_ur5_base_rpy=(0.0, 0.0, 0.0),
        pybullet_ur5_ik_iterations: int = 120,
        pybullet_ur5_ik_damping: float = 0.02,
        pybullet_ur5_position_gain: float = 0.08,
        pybullet_ur5_velocity_gain: float = 1.0,
        pybullet_ur5_max_force: float = 500.0,
        pybullet_s4_track_orientation: bool = True,
        pybullet_visualize_normal_load: bool = False,
        pybullet_normal_load_arrow_scale: float = 0.055,
        pybullet_suppress_urdf_warnings: bool = True,
        pybullet_marker_radius: float = 0.002,
        pybullet_table_half_extents=(0.45, 0.20, 0.015),
        slider_half_extents=(0.080, 0.026, 0.018),
        pybullet_grasp_height: float = 0.070,
        slot_wall_length: float = 0.19,
        slot_wall_forward_extension: float = 0.065,
        slot_wall_thickness: float = 0.010,
        slot_wall_height: float = 0.030,
        rail_shape: str = "straight",
        rail_polyline=None,
        rail_bend_amp: float = 0.012,
        surface_tilt_x: float = 0.0,
        surface_tilt_y: float = 0.0,
        surface_z0: float = 0.0,
        pybullet_render_width: int = 1280,
        pybullet_render_height: int = 900,
        pybullet_camera_target=(0.55, 0.08, 0.54),
        pybullet_camera_distance: float = 0.84,
        pybullet_camera_yaw: float = 128.0,
        pybullet_camera_pitch: float = -29.0,
        pybullet_camera_fov: float = 42.0,
        **kwargs,
    ):
        realistic_defaults = {
            "dt": 0.40,
            "seg_lengths": (35, 14, 67, 21),
            "seg_length_jitter": (9, 5, 11, 7),
            "start": (-0.2240, 0.0560),
            "start_jitter": (0.0160, 0.0080),
            "stage1_end": (-0.0400, 0.00224),
            "stage2_end": (-0.0320, 0.0),
            "stage3_end": (0.1120, 0.0),
            "stage4_end": (0.1600, 0.0),
            "stage_end_jitter": (
                (0.00512, 0.00192),
                (0.00384, 0.00096),
                (0.00608, 0.00096),
                (0.00224, 0.00048),
            ),
            "stage2_end_x_range": (-0.1120, 0.0),
            "stage2_end_z_range": (-0.00160, 0.00160),
            "slot_x": 0.1600,
            "noise_pos": 0.00048,
            "v1_target": 0.00960,
            "v2_target": 0.00112,
            "v3_target": 0.00720,
            "v4_target": 0.00288,
        }
        base_kwargs = dict(realistic_defaults)
        base_kwargs.update(kwargs)
        self.slot_half_width = float(slot_half_width)
        self.clearance_target = float(clearance_target)
        self.clearance_align_max = float(clearance_align_max)
        self.clearance_insert_max = float(clearance_insert_max)
        self.normal_load_min = float(normal_load_min)
        self.v_align_max = float(base_kwargs["v2_target"])
        self.v_insert_max = float(base_kwargs["v3_target"])
        self.v_seat_max = float(base_kwargs["v4_target"])
        super().__init__(**base_kwargs)
        self.v_align_max = float(self.v2_target)
        self.v_insert_max = float(self.v3_target)
        self.v_seat_max = float(self.v4_target)
        self.slot_half_width = float(slot_half_width)
        self.clearance_target = float(clearance_target)
        self.clearance_align_max = float(clearance_align_max)
        self.clearance_insert_max = float(clearance_insert_max)
        self.normal_load_min = float(normal_load_min)
        self.rollout_backend = str(rollout_backend).lower()
        self.observation_backend = str(observation_backend or self.rollout_backend).lower()
        self.pybullet_world_center = tuple(float(x) for x in np.asarray(pybullet_world_center, dtype=float).reshape(3))
        self.pybullet_sim_dt = float(pybullet_sim_dt)
        self.pybullet_steps_per_sample = None if pybullet_steps_per_sample is None else int(pybullet_steps_per_sample)
        self.pybullet_gravity_z = float(pybullet_gravity_z)
        self.pybullet_solver_iterations = int(pybullet_solver_iterations)
        self.pybullet_ur5_home_q = tuple(float(x) for x in pybullet_ur5_home_q)
        self.pybullet_ur5_tool_axis = str(pybullet_ur5_tool_axis)
        self.pybullet_ur5_tip_offset = float(pybullet_ur5_tip_offset)
        self.pybullet_ur5_ee_link_index = int(pybullet_ur5_ee_link_index)
        self.pybullet_ur5_urdf_path = pybullet_ur5_urdf_path
        self.pybullet_ur5_base_xyz = tuple(float(x) for x in pybullet_ur5_base_xyz)
        self.pybullet_ur5_base_rpy = tuple(float(x) for x in pybullet_ur5_base_rpy)
        self.pybullet_ur5_ik_iterations = int(pybullet_ur5_ik_iterations)
        self.pybullet_ur5_ik_damping = float(pybullet_ur5_ik_damping)
        self.pybullet_ur5_position_gain = float(pybullet_ur5_position_gain)
        self.pybullet_ur5_velocity_gain = float(pybullet_ur5_velocity_gain)
        self.pybullet_ur5_max_force = float(pybullet_ur5_max_force)
        self.pybullet_s4_track_orientation = bool(pybullet_s4_track_orientation)
        self.pybullet_visualize_normal_load = bool(pybullet_visualize_normal_load)
        self.pybullet_normal_load_arrow_scale = float(pybullet_normal_load_arrow_scale)
        self.pybullet_suppress_urdf_warnings = bool(pybullet_suppress_urdf_warnings)
        self.pybullet_marker_radius = float(pybullet_marker_radius)
        self.pybullet_table_half_extents = tuple(float(x) for x in pybullet_table_half_extents)
        self.slider_half_extents = tuple(float(x) for x in slider_half_extents)
        self.pybullet_grasp_height = float(pybullet_grasp_height)
        self.slot_wall_length = float(slot_wall_length)
        self.slot_wall_forward_extension = float(slot_wall_forward_extension)
        self.slot_wall_thickness = float(slot_wall_thickness)
        self.slot_wall_height = float(slot_wall_height)
        self.rail_shape = str(rail_shape).strip().lower()
        self.rail_polyline = self._coerce_rail_polyline(rail_polyline)
        self.rail_bend_amp = float(rail_bend_amp)
        self.surface_tilt_x = float(surface_tilt_x)
        self.surface_tilt_y = float(surface_tilt_y)
        self.surface_z0 = float(surface_z0)
        self.pybullet_render_width = int(pybullet_render_width)
        self.pybullet_render_height = int(pybullet_render_height)
        self.pybullet_camera_target = tuple(float(x) for x in pybullet_camera_target)
        self.pybullet_camera_distance = float(pybullet_camera_distance)
        self.pybullet_camera_yaw = float(pybullet_camera_yaw)
        self.pybullet_camera_pitch = float(pybullet_camera_pitch)
        self.pybullet_camera_fov = float(pybullet_camera_fov)
        self.eval_tag = "S4SlideInsertRealistic"
        self.subgoal = np.asarray([self.stage2_end[0], self.clearance_target, self.stage2_end[1], self.theta_stage2_end], dtype=float)
        self.goal = np.asarray([self.stage4_end[0], self.clearance_target, self.stage4_end[1], self.theta_stage4_end], dtype=float)
        self._cached_normal_load_traces = {}
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self.feature_schema = self.get_feature_schema()

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "surf_dist", "description": "Distance to the table/guide surface"},
            {"id": 1, "name": "centerline_dist", "description": "Absolute lateral distance from the slot centerline"},
            {"id": 2, "name": "orient_err", "description": "Absolute angle error relative to the slot"},
            {"id": 3, "name": "speed", "description": "3D translational speed"},
            {"id": 4, "name": "angular_speed", "description": "Absolute angular speed"},
            {"id": 5, "name": "normal_load", "description": "Normal preload applied against the guide"},
            {"id": 6, "name": "noise", "description": "Auxiliary irrelevant feature"},
            {"id": 7, "name": "start_dist", "description": "Distance to the demo start pose"},
            {"id": 8, "name": "insertion_err", "description": "Distance remaining to the slot target along x"},
        ]

    def get_overlay_feature_names(self):
        return [
            "surf_dist",
            "centerline_dist",
            "orient_err",
            "speed",
            "normal_load",
            "start_dist",
            "insertion_err",
        ]

    def get_true_constraints(self):
        return {
            "surface_target": 0.0,
            "clearance_target": float(self.clearance_target),
            "clearance_align_max": float(self.clearance_align_max),
            "clearance_insert_max": float(self.clearance_insert_max),
            "normal_load_stage2_min": float(self.f_contact_min),
            "normal_load_stage3_min": float(self.f_slide_min),
            "normal_load_stage4_min": float(self.f_insert_min),
            "v_align_max": float(self.v_align_max),
            "v_insert_max": float(self.v_insert_max),
            "v_seat_max": float(self.v_seat_max),
            "orient_aligned_max": float(self.orient_err_max_stage3),
        }

    def get_constraint_specs(self):
        return [
            {"feature_name": "surf_dist", "stage": 1, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "centerline_dist", "stage": 1, "semantics": "target_value", "oracle_key": "clearance_target"},
            {"feature_name": "normal_load", "stage": 1, "semantics": "lower_bound", "oracle_key": "normal_load_stage2_min"},
            {"feature_name": "surf_dist", "stage": 2, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "centerline_dist", "stage": 2, "semantics": "target_value", "oracle_key": "clearance_target"},
            {"feature_name": "orient_err", "stage": 2, "semantics": "upper_bound", "oracle_key": "orient_aligned_max"},
            {"feature_name": "normal_load", "stage": 2, "semantics": "lower_bound", "oracle_key": "normal_load_stage3_min"},
            {"feature_name": "speed", "stage": 2, "semantics": "upper_bound", "oracle_key": "v_insert_max"},
            {"feature_name": "surf_dist", "stage": 3, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "centerline_dist", "stage": 3, "semantics": "target_value", "oracle_key": "clearance_target"},
            {"feature_name": "orient_err", "stage": 3, "semantics": "upper_bound", "oracle_key": "orient_aligned_max"},
            {"feature_name": "normal_load", "stage": 3, "semantics": "lower_bound", "oracle_key": "normal_load_stage4_min"},
            {"feature_name": "speed", "stage": 3, "semantics": "upper_bound", "oracle_key": "v_seat_max"},
        ]

    def get_observation_spec(self):
        return {
            "feature_schema": self.get_feature_schema(),
            "default_rollout_backend": self.rollout_backend,
            "default_observation_backend": self.observation_backend,
            "state_schema": ["x", "y", "z", "theta"],
            "normal_load_semantics": "contact preload; controller/debug signal, not insertion push effort",
        }

    def sample_scene(self, seed=None, rng=None):
        scene = super().sample_scene(seed=seed, rng=rng)
        scene["task_name"] = "S4SlideInsertRealistic"
        scene["geometry"].update(
            {
                "start": self.start.tolist(),
                "stage1_end": self.stage1_end.tolist(),
                "stage2_end": self.stage2_end.tolist(),
                "stage3_end": self.stage3_end.tolist(),
                "stage4_end": self.stage4_end.tolist(),
                "slot_x": float(self.slot_x),
                "slot_center_y": float(self.clearance_target),
                "slot_half_width": float(self.slot_half_width),
                "rail_shape": str(self.rail_shape),
                "rail_polyline": self.get_rail_polyline(num=64).tolist(),
                "surface_z": float(self.surface_z0),
                "surface_tilt_x": float(self.surface_tilt_x),
                "surface_tilt_y": float(self.surface_tilt_y),
            }
        )
        return scene

    def surface_height(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=float)
        flat = pts.reshape(-1, 2)
        z = (
            float(self.surface_z0)
            + float(self.surface_tilt_x) * (flat[:, 0] - float(self.slot_x))
            + float(self.surface_tilt_y) * (flat[:, 1] - float(self.clearance_target))
        )
        return z.reshape(pts.shape[:-1])

    @staticmethod
    def _coerce_rail_polyline(polyline):
        if polyline is None:
            return None
        if isinstance(polyline, str):
            text = polyline.strip()
            if not text:
                return None
            pts = []
            for item in text.split(";"):
                xy = [float(v.strip()) for v in item.split(",") if v.strip()]
                if len(xy) != 2:
                    raise ValueError(f"Invalid rail polyline point {item!r}; expected 'x,y'.")
                pts.append(xy)
            arr = np.asarray(pts, dtype=float)
        else:
            arr = np.asarray(polyline, dtype=float)
        arr = np.asarray(arr, dtype=float).reshape(-1, 2)
        if arr.shape[0] < 2:
            raise ValueError("rail_polyline must contain at least two 2D points.")
        return arr

    def get_rail_polyline(self, num: int = 96) -> np.ndarray:
        if self.rail_polyline is not None:
            return np.asarray(self.rail_polyline, dtype=float).reshape(-1, 2).copy()
        x0 = float(self.slot_x) - float(self.slot_wall_length)
        x1 = float(self.slot_x)
        y0 = float(self.clearance_target)
        shape = str(self.rail_shape or "straight").strip().lower()
        if shape == "polyline":
            xm = 0.5 * (x0 + x1)
            return np.asarray([[x0, y0], [xm, y0 + float(self.rail_bend_amp)], [x1, y0]], dtype=float)
        n = max(int(num), 2)
        u = np.linspace(0.0, 1.0, n)
        x = x0 + (x1 - x0) * u
        if shape in {"sine", "curve", "curved"}:
            y = y0 + float(self.rail_bend_amp) * np.sin(np.pi * u)
        else:
            y = np.full_like(x, y0)
        return np.c_[x, y]

    def _rail_segments(self):
        pts = self.get_rail_polyline(num=128)
        seg = pts[1:] - pts[:-1]
        lengths = np.linalg.norm(seg, axis=1)
        keep = lengths > 1e-10
        if not np.any(keep):
            pts = np.asarray([[float(self.start[0]), float(self.clearance_target)], [float(self.slot_x), float(self.clearance_target)]], dtype=float)
            seg = pts[1:] - pts[:-1]
            lengths = np.linalg.norm(seg, axis=1)
            keep = lengths > 1e-10
        seg = seg[keep]
        starts = pts[:-1][keep]
        lengths = lengths[keep]
        tangents = seg / lengths[:, None]
        normals = np.c_[-tangents[:, 1], tangents[:, 0]]
        cum = np.r_[0.0, np.cumsum(lengths)]
        return starts, lengths, tangents, normals, cum

    def rail_total_length(self) -> float:
        *_, cum = self._rail_segments()
        return float(cum[-1])

    def rail_pose_at_s(self, s):
        starts, lengths, tangents, normals, cum = self._rail_segments()
        s_arr = np.asarray(s, dtype=float)
        flat = np.clip(s_arr.reshape(-1), 0.0, float(cum[-1]))
        idx = np.searchsorted(cum[1:], flat, side="right")
        idx = np.clip(idx, 0, len(lengths) - 1)
        local = (flat - cum[idx]) / np.maximum(lengths[idx], 1e-12)
        points = starts[idx] + tangents[idx] * (local[:, None] * lengths[idx, None])
        angles = np.arctan2(tangents[idx, 1], tangents[idx, 0])
        shape = s_arr.shape
        return (
            points.reshape(shape + (2,)),
            tangents[idx].reshape(shape + (2,)),
            normals[idx].reshape(shape + (2,)),
            angles.reshape(shape),
        )

    def project_to_rail(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        pts = np.asarray(xy, dtype=float).reshape(-1, 2)
        starts, lengths, tangents, normals, cum = self._rail_segments()
        best_d2 = np.full(pts.shape[0], np.inf, dtype=float)
        best_s = np.zeros(pts.shape[0], dtype=float)
        best_signed = np.zeros(pts.shape[0], dtype=float)
        best_angle = np.zeros(pts.shape[0], dtype=float)
        for i in range(len(lengths)):
            rel = pts - starts[i]
            t = np.clip(rel @ tangents[i] / max(lengths[i], 1e-12), 0.0, lengths[i])
            proj = starts[i] + t[:, None] * tangents[i]
            delta = pts - proj
            d2 = np.sum(delta * delta, axis=1)
            take = d2 < best_d2
            if np.any(take):
                best_d2[take] = d2[take]
                best_s[take] = cum[i] + t[take]
                best_signed[take] = delta[take] @ normals[i]
                best_angle[take] = np.arctan2(tangents[i, 1], tangents[i, 0])
        total = float(cum[-1])
        return {
            "s": best_s,
            "signed_dist": best_signed,
            "dist": np.abs(best_signed),
            "angle": best_angle,
            "remaining": total - best_s,
        }

    @staticmethod
    def _traj_cache_key(traj: np.ndarray):
        arr = np.ascontiguousarray(np.asarray(traj, dtype=np.float64))
        return arr.shape, arr.tobytes()

    def register_normal_load_trace(self, traj: np.ndarray, load: np.ndarray):
        self._cached_normal_load_traces[self._traj_cache_key(traj)] = np.asarray(load, dtype=float).copy()

    def _lookup_cached_normal_load_trace(self, traj: np.ndarray):
        load = self._cached_normal_load_traces.get(self._traj_cache_key(traj))
        return None if load is None else np.asarray(load, dtype=float)

    def _normal_load_profile(self, labels: np.ndarray, rng: np.random.RandomState | None = None):
        labels = np.asarray(labels, dtype=int)
        load = np.zeros(len(labels), dtype=float)
        for stage_idx, base in [(1, self.normal_load_min), (2, 1.08 * self.normal_load_min), (3, 1.05 * self.normal_load_min)]:
            mask = labels == stage_idx
            if not np.any(mask):
                continue
            n = int(mask.sum())
            u = np.linspace(0.0, 1.0, n)
            profile = base * (1.0 + 0.08 * np.sin(2.0 * np.pi * u - 0.5 * np.pi) * np.sin(np.pi * u))
            if rng is not None:
                profile += self._smooth_noise(rng, n, 0.025 * base, kernel_size=5)
            load[mask] = np.maximum(0.92 * base, profile)
        return load

    def apply_execution_normal_load_noise(self, normal_load: np.ndarray, *, noise_std: float = 0.0, noise_smooth: float = 0.85, seed=None):
        load = np.asarray(normal_load, dtype=float).reshape(-1)
        executed = load.copy()
        std = float(noise_std)
        if std <= 0.0 or executed.size == 0:
            return executed, np.zeros_like(executed)
        rng = np.random.RandomState(0 if seed is None else int(seed) + 9173)
        raw = rng.normal(0.0, std, size=executed.shape)
        smooth = float(np.clip(float(noise_smooth), 0.0, 0.999))
        noise = np.zeros_like(raw, dtype=float)
        for i in range(1, raw.size):
            noise[i] = smooth * noise[i - 1] + (1.0 - smooth) * raw[i]
        active = load > 1e-9
        executed[active] = np.maximum(0.0, load[active] + noise[active])
        return executed, executed - load

    def _lift_planar_demo_to_4d(self, pos_planar: np.ndarray, theta: np.ndarray, labels: np.ndarray, rng: np.random.RandomState):
        pos_planar = np.asarray(pos_planar, dtype=float)
        labels = np.asarray(labels, dtype=int)
        x = pos_planar[:, 0]
        z = pos_planar[:, 1]
        y = np.zeros(len(x), dtype=float)
        stage0 = labels == 0
        if np.any(stage0):
            n0 = int(stage0.sum())
            y0_start = float(rng.uniform(-0.055, 0.055))
            y[stage0] = np.linspace(y0_start, 0.0, n0, endpoint=True)
        for stage_idx, amp in [(1, 0.00018), (2, 0.00100), (3, 0.00055)]:
            mask = labels == stage_idx
            if not np.any(mask):
                continue
            n = int(mask.sum())
            u = np.linspace(0.0, 1.0, n)
            y_wave = amp * np.sin(2.0 * np.pi * u + float(rng.uniform(-np.pi, np.pi))) * np.sin(np.pi * u)
            y_noise = self._smooth_noise(rng, n, 0.22 * amp, kernel_size=5) * np.sin(np.pi * u)
            y[mask] = np.clip(y_wave + y_noise, -1.25 * amp, 1.25 * amp)
        z[labels >= 1] = np.clip(z[labels >= 1], -0.002, 0.002)
        return np.c_[x, y, z, np.asarray(theta, dtype=float)]

    @staticmethod
    def _sample_polyline_by_edge_weights(path: np.ndarray, num_points: int, edge_weights: np.ndarray) -> np.ndarray:
        pts = np.asarray(path, dtype=float)
        n = int(num_points)
        if len(pts) <= 1 or n <= 1:
            return np.repeat(pts[:1], max(n, 1), axis=0)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= 1e-12:
            out = np.repeat(pts[:1], n, axis=0)
            out[0] = pts[0]
            out[-1] = pts[-1]
            return out
        weights = np.asarray(edge_weights, dtype=float).reshape(-1)
        expected = max(n - 1, 1)
        if weights.size != expected:
            raise ValueError(f"edge_weights produced {weights.size} values, expected {expected}.")
        weights = np.clip(weights, 1e-6, None)
        targets = np.concatenate([[0.0], np.cumsum(weights)])
        targets = targets / float(targets[-1]) * total
        out = np.empty((n, pts.shape[1]), dtype=float)
        for d in range(pts.shape[1]):
            out[:, d] = np.interp(targets, s, pts[:, d])
        out[0] = pts[0]
        out[-1] = pts[-1]
        return out

    def _speed_profile_weights(self, stage_idx: int, num_edges: int, rng: np.random.RandomState, phase: float) -> np.ndarray:
        n = int(num_edges)
        if n <= 0:
            return np.zeros(0, dtype=float)
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        if int(stage_idx) == 0:
            weights = 1.38 - 0.60 * u + 0.05 * np.sin(2.0 * np.pi * u + phase) * np.sin(np.pi * u)
        elif int(stage_idx) == 1:
            valley = np.exp(-0.5 * ((u - 0.58) / 0.12) ** 2)
            weights = 1.0 - 0.16 * valley + 0.035 * np.sin(2.5 * np.pi * u + phase) * np.sin(np.pi * u)
        elif int(stage_idx) == 2:
            valleys = (
                0.10 * np.exp(-0.5 * ((u - 0.32) / 0.040) ** 2)
                + 0.07 * np.exp(-0.5 * ((u - 0.61) / 0.050) ** 2)
                + 0.06 * np.exp(-0.5 * ((u - 0.80) / 0.035) ** 2)
            )
            micro_slowdown = 0.010 * np.abs(self._smooth_noise(rng, n, 1.0, kernel_size=5)) * np.sin(np.pi * u)
            weights = 1.0 - valleys - micro_slowdown
        else:
            valleys = (
                0.075 * np.exp(-0.5 * ((u - 0.45) / 0.055) ** 2)
                + 0.050 * np.exp(-0.5 * ((u - 0.72) / 0.040) ** 2)
            )
            micro_slowdown = 0.006 * np.abs(self._smooth_noise(rng, n, 1.0, kernel_size=3)) * np.sin(np.pi * u)
            weights = 1.0 - valleys - micro_slowdown
        if int(stage_idx) == 2:
            weights += self._smooth_noise(rng, n, 0.004, kernel_size=5)
            weights = np.minimum(weights, 1.0)
        elif int(stage_idx) == 3:
            weights += self._smooth_noise(rng, n, 0.003, kernel_size=3)
            weights = np.minimum(weights, 1.0)
        else:
            weights += self._smooth_noise(rng, n, 0.012, kernel_size=5)
        return np.clip(weights, 0.45, None)

    @staticmethod
    def _planar_curve(
        start: np.ndarray,
        end: np.ndarray,
        rng: np.random.RandomState,
        *,
        amp: float,
        cycles: float,
        z_bias: float = 0.0,
        n: int = 512,
    ) -> np.ndarray:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        u = np.linspace(0.0, 1.0, int(n), endpoint=True)
        base = start[None, :] + u[:, None] * (end - start)[None, :]
        direction = end - start
        dist = float(np.linalg.norm(direction))
        if dist <= 1e-12:
            normal = np.array([0.0, 1.0], dtype=float)
        else:
            tangent = direction / dist
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
        phase = float(rng.uniform(-0.5 * np.pi, 0.5 * np.pi))
        sign = -1.0 if rng.rand() < 0.5 else 1.0
        envelope = np.sin(np.pi * u)
        curve = base + sign * float(amp) * envelope[:, None] * np.sin(float(cycles) * np.pi * u + phase)[:, None] * normal[None, :]
        if z_bias != 0.0:
            curve[:, 1] += float(z_bias) * envelope**2
        curve[0] = start
        curve[-1] = end
        return curve

    def _resample_planar_segment(
        self,
        path: np.ndarray,
        num_points: int,
        stage_idx: int,
        rng: np.random.RandomState,
        phase: float,
    ) -> np.ndarray:
        n = int(num_points)
        weights = self._speed_profile_weights(stage_idx, max(n - 1, 1), rng, phase)
        return self._sample_polyline_by_edge_weights(path, n, weights)

    def generate_demo(self, seed: int):
        rng = np.random.RandomState(seed)
        l1, l2, l3, l4 = self._sample_segment_lengths(rng)
        latents = self._sample_demo_latents(rng)
        latents["surface_wobble"] = float(rng.uniform(0.00024, 0.00080))
        phase = float(latents["phase"])

        v1_demo = self.v1_target * rng.uniform(0.94, 1.06)
        v2_demo = self.v2_target * rng.uniform(0.82, 0.90)
        v3_demo = self.v3_target * rng.uniform(0.95, 0.98)
        v4_demo = self.v4_target * rng.uniform(0.92, 0.97)

        start_local = self.start + rng.randn(2) * self.start_jitter
        start_local[0] = float(np.clip(start_local[0], -0.2640, -0.1792))
        start_local[1] = float(np.clip(start_local[1], 0.0384, 0.0736))

        stage4_end_local = self.stage4_end + rng.randn(2) * self.stage_end_jitter[3]
        stage4_end_local[0] = float(np.clip(stage4_end_local[0], 0.1536, 0.1648))
        stage4_end_local[1] = float(np.clip(stage4_end_local[1], -0.00128, 0.00128))

        seat_len = v4_demo * self.dt * max(l4 - 1, 1) * rng.uniform(0.94, 1.04)
        stage3_end_local = self.stage3_end + rng.randn(2) * self.stage_end_jitter[2]
        stage3_end_local[0] = float(np.clip(stage4_end_local[0] - seat_len, 0.1248, 0.1488))
        stage3_end_local[1] = float(np.clip(stage3_end_local[1], -0.00160, 0.00160))

        insert_len = v3_demo * self.dt * max(l3, 1) * rng.uniform(0.96, 1.03)
        stage2_end_local = np.array(
            [
                stage3_end_local[0] - insert_len,
                rng.uniform(*self.stage2_end_z_range),
            ],
            dtype=float,
        )
        stage2_end_local[0] = float(np.clip(stage2_end_local[0], -0.0960, 0.0060))

        align_len = v2_demo * self.dt * max(l2, 1) * rng.uniform(0.92, 1.08)
        stage1_end_local = self.stage1_end + rng.randn(2) * self.stage_end_jitter[0]
        stage1_end_local[0] = float(np.clip(stage2_end_local[0] - align_len, -0.1120, stage2_end_local[0] - 0.00045))
        stage1_end_local[1] = float(np.clip(rng.uniform(0.00064, 0.00224), 0.00048, 0.00320))

        seg1_path = self._planar_curve(start_local, stage1_end_local, rng, amp=0.0060, cycles=1.0, z_bias=0.0060)
        seg2_path = self._planar_curve(stage1_end_local, stage2_end_local, rng, amp=0.00025, cycles=1.1, z_bias=-0.00012)
        seg3_path = self._planar_curve(stage2_end_local, stage3_end_local, rng, amp=0.00120, cycles=2.3, z_bias=0.0000)
        seg4_path = self._planar_curve(stage3_end_local, stage4_end_local, rng, amp=0.00055, cycles=1.2, z_bias=0.0)

        seg1 = self._resample_planar_segment(seg1_path, l1, 0, rng, phase)
        seg2 = self._resample_planar_segment(seg2_path, l2 + 1, 1, rng, phase + 0.4)[1:]
        seg3 = self._resample_planar_segment(seg3_path, l3 + 1, 2, rng, phase + 0.8)[1:]
        seg4 = self._resample_planar_segment(seg4_path, l4 + 1, 3, rng, phase + 1.2)[1:]

        pos = np.vstack([seg1, seg2, seg3, seg4])
        labels = np.repeat(np.arange(4), [l1, l2, l3, l4])

        theta_start_local = self.theta_start + self.theta_start_jitter * rng.randn()
        theta_stage1_end = self.theta_stage1_end + self.theta_end_jitter[0] * rng.randn()
        theta_stage2_end = float(rng.uniform(*self.stage2_theta_end_range))
        theta_stage3_end = self.theta_stage3_end + self.theta_end_jitter[2] * rng.randn()
        theta_stage4_end = self.theta_stage4_end + self.theta_end_jitter[3] * rng.randn()

        theta1 = np.linspace(theta_start_local, theta_stage1_end, l1, endpoint=False)
        theta2 = np.linspace(theta_stage1_end, theta_stage2_end, l2, endpoint=False)
        theta3 = np.zeros(l3, dtype=float)
        theta4 = np.zeros(l4, dtype=float)
        sign3 = -1.0 if float(theta_stage2_end) < 0.0 else 1.0
        sign4 = sign3 if abs(float(theta_stage4_end)) < 1e-6 else (1.0 if float(theta_stage4_end) >= 0.0 else -1.0)
        if l3 > 0:
            u3_theta = np.linspace(0.0, 1.0, l3, endpoint=False)
            half_wave3 = np.maximum(np.sin(2.35 * np.pi * u3_theta - 0.5 * np.pi + 0.20 * latents["phase"]), 0.0)
            margin3 = 0.62 * self.orient_err_max_stage3 * half_wave3 - 0.18 * self.orient_err_max_stage3
            abs_theta3 = np.clip(self.orient_err_max_stage3 - margin3, 0.0, 0.96 * self.orient_err_max_stage3)
            theta3 = sign3 * self._smooth_trace(abs_theta3, kernel_size=3)
        if l4 > 0:
            u4_theta = np.linspace(0.0, 1.0, l4, endpoint=True)
            half_wave4 = np.maximum(np.sin(1.95 * np.pi * u4_theta - 0.5 * np.pi + 0.16 * latents["phase"]), 0.0)
            margin4 = 0.58 * self.orient_err_max_stage4 * half_wave4 - 0.16 * self.orient_err_max_stage4
            abs_theta4 = np.clip(self.orient_err_max_stage4 - margin4, 0.0, 0.96 * self.orient_err_max_stage4)
            theta4 = sign4 * self._smooth_trace(abs_theta4, kernel_size=3)
        theta = np.concatenate([theta1, theta2, theta3, theta4])
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 - 1, half_window=self.transition_half_window).ravel()
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 + l2 - 1, half_window=self.transition_half_window).ravel()
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 + l2 + l3 - 1, half_window=self.transition_half_window).ravel()

        theta_noise_scale = np.take(np.array([1.0, 0.30, 0.08, 0.05], dtype=float), labels)
        theta += self._smooth_noise(rng, len(theta), 0.28 * self.noise_misc, kernel_size=11) * theta_noise_scale
        theta += latents["theta_wobble"] * np.sin(np.linspace(0.0, 4.5 * np.pi, len(theta)) + latents["phase"]) * np.r_[
            np.linspace(0.3, 1.0, l1 + l2),
            np.linspace(0.18, 0.08, l3 + l4),
        ] * theta_noise_scale
        theta[l1 + l2:l1 + l2 + l3] = np.clip(
            theta[l1 + l2:l1 + l2 + l3],
            -0.95 * self.orient_err_max_stage3,
            0.95 * self.orient_err_max_stage3,
        )
        theta[l1 + l2 + l3:] = np.clip(
            theta[l1 + l2 + l3:],
            -0.95 * self.orient_err_max_stage4,
            0.95 * self.orient_err_max_stage4,
        )
        for boundary in (l1 - 1, l1 + l2 - 1, l1 + l2 + l3 - 1):
            theta = self._blend_segment_boundary(theta[:, None], boundary=boundary, half_window=self.transition_half_window).ravel()
        theta[l1 + l2:l1 + l2 + l3] = np.clip(
            theta[l1 + l2:l1 + l2 + l3],
            -0.98 * self.orient_err_max_stage3,
            0.98 * self.orient_err_max_stage3,
        )
        theta[l1 + l2 + l3:] = np.clip(
            theta[l1 + l2 + l3:],
            -0.98 * self.orient_err_max_stage4,
            0.98 * self.orient_err_max_stage4,
        )

        normal_load = self._compute_force_signal(pos, theta, stage3_end_local[0], labels, rng, latents)
        rng = np.random.RandomState(int(seed) + 100003)
        traj4 = self._lift_planar_demo_to_4d(pos, theta, labels, rng)
        return traj4, np.asarray(labels, dtype=int), normal_load

    def rollout_demo(self, scene, seed=None, rng=None, backend=None, **kwargs):
        local_seed = int(seed) if seed is not None else int((scene or {}).get("rollout_seed", 0))
        traj4, labels, normal_load = self.generate_demo(seed=local_seed)
        cutpoints = np.where(np.diff(labels) != 0)[0].astype(int)
        active_backend = str(backend or self.rollout_backend).lower()
        if active_backend == "pybullet":
            execution_normal_load, load_noise = self.apply_execution_normal_load_noise(
                normal_load,
                noise_std=float(kwargs.get("execution_normal_load_noise_std", 0.0)),
                noise_smooth=float(kwargs.get("execution_normal_load_noise_smooth", 0.85)),
                seed=kwargs.get("execution_normal_load_noise_seed", kwargs.get("execution_noise_seed", None)),
            )
            sim = simulate_s4_demo_from_reference(
                self,
                scene=scene,
                reference_traj=traj4,
                true_cutpoints=cutpoints,
                gui=int(kwargs.get("gui", 0)),
                video_path=kwargs.get("video_path"),
                fps=float(kwargs.get("fps", 15.0)),
                width=kwargs.get("width"),
                height=kwargs.get("height"),
                render_frame_stride=int(kwargs.get("render_frame_stride", 1)),
                video_end_hold_seconds=float(kwargs.get("video_end_hold_seconds", 2.0)),
                realtime=bool(kwargs.get("realtime", False)),
                gui_hold_seconds=float(kwargs.get("gui_hold_seconds", 0.0)),
                normal_load_trace=execution_normal_load,
                visualize_normal_load=bool(kwargs.get("visualize_normal_load", self.pybullet_visualize_normal_load)),
                feature_overlay=bool(kwargs.get("feature_overlay", True)),
                feature_overlay_title=kwargs.get("feature_overlay_title", None),
            )
            sim["planned_trajectory"] = traj4
            sim["normal_load_trace"] = execution_normal_load
            sim["planned_normal_load_trace"] = normal_load
            sim["execution_normal_load_noise"] = load_noise
            sim["true_labels"] = labels
            return sim
        return {
            "trajectory": traj4,
            "planned_trajectory": traj4,
            "true_cutpoints": cutpoints,
            "true_labels": labels,
            "normal_load_trace": normal_load,
        }

    def plan_episode_from_constraints(self, scene, constraint_values: dict, *, seed: int = 0, stage_lengths=None, speed_safety: float = 1.0):
        lengths = [int(x) for x in self.seg_lengths]
        for key, value in dict(stage_lengths or {}).items():
            text = str(key).strip().lower()
            if text.startswith("stage"):
                idx = int(text.replace("stage", "")) - 1
            elif text.startswith("s"):
                idx = int(text.replace("s", "")) - 1
            else:
                idx = int(text)
            if 0 <= idx < len(lengths):
                lengths[idx] = max(int(value), 3)
        l1, l2, l3, l4 = lengths

        def cv(key: str, default: float) -> float:
            value = dict(constraint_values or {}).get(key)
            if value is None or not np.isfinite(float(value)):
                return float(default)
            return float(value)

        surf2 = cv("s2:surf_dist", float(self.true_constraints.get("surface_target", 0.0)))
        surf3 = cv("s3:surf_dist", surf2)
        surf4 = cv("s4:surf_dist", surf3)
        center2 = cv("s2:centerline_dist", float(self.clearance_target))
        center3 = cv("s3:centerline_dist", center2)
        center4 = cv("s4:centerline_dist", center3)
        theta2 = cv("s2:orient_err", float(self.theta_stage2_end))
        theta3 = cv("s3:orient_err", theta2)
        theta4 = cv("s4:orient_err", theta3)
        v2 = max(cv("s2:speed", float(self.v_align_max)) * float(speed_safety), 1e-5)
        v3 = max(cv("s3:speed", float(self.v_insert_max)) * float(speed_safety), 1e-5)
        v4 = max(cv("s4:speed", float(self.v_seat_max)) * float(speed_safety), 1e-5)

        rail_total = float(self.rail_total_length())
        s4 = rail_total
        s3 = max(0.0, s4 - v4 * float(self.dt) * max(l4 - 1, 1))
        s2 = max(0.0, s3 - v3 * float(self.dt) * max(l3, 1))
        s1 = max(0.0, s2 - v2 * float(self.dt) * max(l2, 1))

        start_y = 0.75 * float(self.slot_half_width)
        w0 = np.asarray([float(self.start[0]), start_y, float(self.start[1]), float(self.theta_start)], dtype=float)

        def line(a, b, n, *, endpoint=False):
            return np.linspace(np.asarray(a, dtype=float), np.asarray(b, dtype=float), int(n), endpoint=bool(endpoint))

        def rail_segment(sa, sb, n, center, surf, theta_err, *, endpoint=False):
            ss = np.linspace(float(sa), float(sb), int(n), endpoint=bool(endpoint))
            points, _tangents, normals, angles = self.rail_pose_at_s(ss)
            traj = np.zeros((int(n), 4), dtype=float)
            traj[:, :2] = points + normals * float(center)
            traj[:, 2] = self.surface_height(traj[:, :2]) + float(surf)
            traj[:, 3] = angles + float(theta_err)
            return traj

        w1 = rail_segment(s1, s1, 1, center2, surf2, float(self.theta_stage1_end), endpoint=True)[0]
        seg1 = line(w0, w1, l1, endpoint=False)
        seg2 = rail_segment(s1, s2, l2, center2, surf2, theta2, endpoint=False)
        seg3 = rail_segment(s2, s3, l3, center3, surf3, theta3, endpoint=False)
        seg4 = rail_segment(s3, s4, l4, center4, surf4, theta4, endpoint=True)
        traj = np.vstack([seg1, seg2, seg3, seg4])
        labels = np.repeat(np.arange(4), [l1, l2, l3, l4])
        cutpoints = np.where(np.diff(labels) != 0)[0].astype(int)

        normal_load = np.zeros(len(traj), dtype=float)
        for stage_idx, key in [(1, "s2:normal_load"), (2, "s3:normal_load"), (3, "s4:normal_load")]:
            normal_load[labels == int(stage_idx)] = max(cv(key, float(self.normal_load_min)), 0.0)

        return {
            "trajectory": traj,
            "planned_trajectory": traj,
            "true_cutpoints": cutpoints,
            "true_labels": labels,
            "normal_load_trace": normal_load,
            "constraint_values": dict(constraint_values or {}),
            "stage_lengths": {"stage1": int(l1), "stage2": int(l2), "stage3": int(l3), "stage4": int(l4)},
            "planner": "s4_clean_geometric_insert_planner",
            "scene": dict(scene or {}),
            "seed": int(seed),
        }

    def execute_plan_pybullet(self, scene, planned_episode, **kwargs):
        traj = np.asarray(planned_episode["trajectory"], dtype=float)
        cutpoints = np.asarray(planned_episode.get("true_cutpoints", []), dtype=int)
        normal_load = np.asarray(planned_episode.get("normal_load_trace", np.zeros(len(traj))), dtype=float)
        execution_normal_load, load_noise = self.apply_execution_normal_load_noise(
            normal_load,
            noise_std=float(kwargs.get("execution_normal_load_noise_std", 0.0)),
            noise_smooth=float(kwargs.get("execution_normal_load_noise_smooth", 0.85)),
            seed=kwargs.get("execution_normal_load_noise_seed", kwargs.get("execution_noise_seed", None)),
        )
        sim = simulate_s4_demo_from_reference(
            self,
            scene=scene,
            reference_traj=traj,
            true_cutpoints=cutpoints,
            gui=int(kwargs.get("gui", 0)),
            video_path=kwargs.get("video_path"),
            fps=float(kwargs.get("fps", 15.0)),
            width=kwargs.get("width"),
            height=kwargs.get("height"),
            render_frame_stride=int(kwargs.get("render_frame_stride", 1)),
            video_end_hold_seconds=float(kwargs.get("video_end_hold_seconds", 2.0)),
            realtime=bool(kwargs.get("realtime", False)),
            gui_hold_seconds=float(kwargs.get("gui_hold_seconds", 0.0)),
            normal_load_trace=execution_normal_load,
            visualize_normal_load=bool(kwargs.get("visualize_normal_load", self.pybullet_visualize_normal_load)),
            feature_overlay=bool(kwargs.get("feature_overlay", True)),
            feature_overlay_title=kwargs.get("feature_overlay_title", None),
            execution_joint_noise_std=float(kwargs.get("execution_joint_noise_std", 0.0)),
            execution_joint_noise_smooth=float(kwargs.get("execution_joint_noise_smooth", 0.90)),
            execution_noise_seed=kwargs.get("execution_noise_seed", None),
        )
        sim["planned_trajectory"] = traj
        sim["normal_load_trace"] = execution_normal_load
        sim["planned_normal_load_trace"] = normal_load
        sim["execution_normal_load_noise"] = load_noise
        sim["true_labels"] = np.asarray(planned_episode.get("true_labels", []), dtype=int)
        sim["planner"] = str(planned_episode.get("planner", "s4_clean_geometric_insert_planner"))
        sim["planned_constraint_values"] = dict(planned_episode.get("constraint_values", {}))
        return sim

    def compute_observation(self, latent_rollout, scene, backend=None):
        traj = np.asarray(latent_rollout["trajectory"], dtype=float)
        load = latent_rollout.get("normal_load_trace")
        if load is not None:
            self.register_normal_load_trace(traj, np.asarray(load, dtype=float))
        features = np.asarray(self.compute_all_features_matrix(traj), dtype=float)
        out = {
            "trajectory": traj,
            "features": features,
            "true_cutpoints": np.asarray(latent_rollout.get("true_cutpoints", []), dtype=int),
            "true_labels": np.asarray(latent_rollout.get("true_labels", []), dtype=int),
            "feature_schema": self.get_feature_schema(),
            "observation_spec": self.get_observation_spec(),
            "scene": dict(scene or {}),
        }
        for key in (
            "planned_trajectory",
            "reference_trajectory",
            "normal_load_trace",
            "joint_positions",
            "joint_velocities",
            "joint_position_commands",
            "joint_position_commands_nominal",
            "execution_joint_noise",
            "planned_normal_load_trace",
            "execution_normal_load_noise",
            "reference_trajectory_world",
            "realized_ee_trajectory_world",
            "realized_ee_quaternions",
            "target_quaternions",
            "ik_position_error_world",
            "robot_backend",
            "sim_dt",
            "steps_per_sample",
        ):
            if key in latent_rollout:
                out[key] = latent_rollout[key]
        return out

    def compute_all_features_matrix(self, traj: np.ndarray, feat_ids=None) -> np.ndarray:
        traj = np.asarray(traj, dtype=float)
        T = traj.shape[0]
        xyz = traj[:, :3]
        theta = traj[:, 3]
        vel = np.zeros_like(xyz)
        omega = np.zeros(T, dtype=float)
        if T > 1:
            vel[:-1] = np.diff(xyz, axis=0) / max(self.dt, 1e-12)
            vel[-1] = vel[-2]
            dtheta = self._wrap_to_pi(np.diff(theta)) / max(self.dt, 1e-12)
            omega[:-1] = dtheta
            omega[-1] = omega[-2]
        speed = np.linalg.norm(vel, axis=1)
        angular_speed = np.abs(omega)
        surf_dist = np.abs(xyz[:, 2] - self.surface_height(xyz[:, :2]))
        rail_proj = self.project_to_rail(xyz[:, :2])
        centerline_dist = np.asarray(rail_proj["dist"], dtype=float)
        orient_err = np.abs(self._wrap_to_pi(theta - np.asarray(rail_proj["angle"], dtype=float)))
        normal_load = self._lookup_cached_normal_load_trace(traj)
        if normal_load is None:
            normal_load = np.where(surf_dist < 0.004, self.normal_load_min, 0.0)
        start_dist = np.linalg.norm(xyz - xyz[0][None, :], axis=1)
        insertion_err = np.asarray(rail_proj["remaining"], dtype=float)
        noise = 0.35 * np.sin(0.19 * np.arange(T)) + 0.15 * np.cos(0.07 * np.arange(T))
        F = np.c_[
            surf_dist,
            centerline_dist,
            orient_err,
            speed,
            angular_speed,
            normal_load,
            noise,
            start_dist,
            insertion_err,
        ]
        return F if feat_ids is None else F[:, feat_ids]

    def render_episode(self, scene, trajectory, output_path, **kwargs):
        cutpoints = kwargs.get("cutpoints")
        return render_planar_episode(
            trajectory=np.asarray(trajectory, dtype=float)[:, :2],
            output_path=output_path,
            cutpoints=cutpoints,
            title=kwargs.get("title", "S4SlideInsertRealistic top view"),
            obstacles=None,
            reference_lines=[{"point": [0.0, self.clearance_target], "direction": [1.0, 0.0], "color": "#64748B"}],
            markers=[{"point": [self.slot_x, self.clearance_target], "color": "#16A34A", "marker": "s", "size": 34}],
            xlabel="x",
            ylabel="y / clearance",
            equal_aspect=True,
        )


def load_S4SlideInsertRealistic(n_demos: int = 10, seed: int = 123, env_kwargs=None, demo_kwargs=None, **extra_env_kwargs):
    env_cfg = dict(env_kwargs or {})
    env_cfg.update(extra_env_kwargs)
    env = S4SlideInsertRealisticEnv(**env_cfg)
    run_kwargs = dict(demo_kwargs or {})
    demos = []
    labels = []
    cutpoints = []
    scene_specs = []
    for i in range(int(n_demos)):
        scene = env.sample_scene()
        scene["demo_index"] = int(i)
        latent = env.rollout_demo(scene, seed=int(seed) + int(i), **run_kwargs)
        observation = env.compute_observation(latent, scene)
        demo = np.asarray(observation["trajectory"], dtype=float)
        demos.append(demo)
        labels.append(np.asarray(observation["true_labels"], dtype=int))
        cutpoints.append(np.asarray(observation["true_cutpoints"], dtype=int))
        scene_specs.append(dict(scene))
    env.demo_subgoals = [np.asarray(x[int(c[1]), :4], dtype=float).copy() for x, c in zip(demos, cutpoints)]
    env.demo_goals = [np.asarray(x[-1, :4], dtype=float).copy() for x in demos]
    env.demo_stage_lengths = [np.bincount(np.asarray(z, dtype=int), minlength=env.n_segments).astype(int) for z in labels]
    env.subgoal = np.mean(np.stack(env.demo_subgoals, axis=0), axis=0)
    env.goal = np.mean(np.stack(env.demo_goals, axis=0), axis=0)
    return TaskBundle(
        name="S4SlideInsertRealistic",
        demos=demos,
        env=env,
        true_taus=None,
        true_cutpoints=[np.asarray(c, dtype=int) for c in cutpoints],
        true_labels=labels,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(),
        constraint_specs=env.get_constraint_specs(),
        meta={
            "seed": int(seed),
            "cutpoints": [c.tolist() for c in cutpoints],
            "task_name": "S4SlideInsertRealistic",
            "scene_specs": scene_specs,
            "observation_specs": env.get_observation_spec(),
            "render_camera_presets": env.get_render_camera_presets(),
            "asset_handles": env.get_asset_handles(),
        },
    )
