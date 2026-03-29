from __future__ import annotations

import numpy as np

from .base import TaskBundle
from planner import optimize_trajectory, resample_polyline


class SphereInspectEnv3D:
    """
    3D spherical surface inspection task.
    """

    def __init__(
        self,
        sphere_center=(0.0, 0.0, 0.0),
        sphere_radius=1.0,
        shell_thickness=0.14,
        seg_lengths=(18, 34, 26, 18),
        seg_length_jitter=(3, 5, 4, 3),
        approach_offset=0.42,
        depart_offset=0.50,
        stage1_speed_max=0.12,
        stage2_speed_max=0.05,
        stage3_speed_max=0.05,
        stage4_speed_max=0.09,
        stage1_accel_max=0.08,
        stage2_accel_max=0.03,
        stage3_accel_max=0.07,
        stage4_accel_max=0.06,
        tool_align_max_stage2=0.12,
        angular_speed_max_stage2=0.22,
        angular_speed_max_stage3=0.55,
        dt=0.8,
        noise_std=0.004,
        surface_near_target_ratio=0.5,
        split_stage3_transition=False,
        transition_stage_fraction=1.0 / 3.0,
        stage2_trace_angle_range=(0.55, 1.05),
        stage2_surface_detour_angle=0.0,
        stage2_length_scale_range=(1.0, 1.0),
        stage4_length_scale_range=(1.0, 1.0),
        feature_boundary_ramp_half_windows=None,
        eval_tag="3DSphereInspect",
    ):
        self.sphere_center = np.asarray(sphere_center, dtype=float)
        self.sphere_radius = float(sphere_radius)
        self.shell_thickness = float(shell_thickness)
        self.seg_lengths = tuple(int(x) for x in seg_lengths)
        self.seg_length_jitter = tuple(int(x) for x in seg_length_jitter)
        self.approach_offset = float(approach_offset)
        self.depart_offset = float(depart_offset)
        self.stage1_speed_max = float(stage1_speed_max)
        self.stage2_speed_max = float(stage2_speed_max)
        self.stage3_speed_max = float(stage3_speed_max)
        self.stage4_speed_max = float(stage4_speed_max)
        self.stage1_accel_max = float(stage1_accel_max)
        self.stage2_accel_max = float(stage2_accel_max)
        self.stage3_accel_max = float(stage3_accel_max)
        self.stage4_accel_max = float(stage4_accel_max)
        self.tool_align_max_stage2 = float(tool_align_max_stage2)
        self.angular_speed_max_stage2 = float(angular_speed_max_stage2)
        self.angular_speed_max_stage3 = float(angular_speed_max_stage3)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.surface_near_target_ratio = float(surface_near_target_ratio)
        self.split_stage3_transition = bool(split_stage3_transition)
        self.transition_stage_fraction = float(transition_stage_fraction)
        angle_lo, angle_hi = stage2_trace_angle_range
        self.stage2_trace_angle_range = (float(angle_lo), float(angle_hi))
        self.stage2_surface_detour_angle = float(stage2_surface_detour_angle)
        stage2_scale_lo, stage2_scale_hi = stage2_length_scale_range
        stage4_scale_lo, stage4_scale_hi = stage4_length_scale_range
        self.stage2_length_scale_range = (float(stage2_scale_lo), float(stage2_scale_hi))
        self.stage4_length_scale_range = (float(stage4_scale_lo), float(stage4_scale_hi))
        self.feature_boundary_ramp_half_windows = feature_boundary_ramp_half_windows
        self.eval_tag = str(eval_tag)

        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self._cached_tool_axis_traces = {}
        self._cached_feature_traces = {}

        nominal_contact = self.sphere_center + np.array([0.0, self.sphere_radius, 0.0], dtype=float)
        nominal_shell = self.sphere_center + np.array(
            [0.0, self.sphere_radius + self.surface_near_target_ratio * self.shell_thickness, 0.0],
            dtype=float,
        )
        self.subgoal = nominal_contact.copy()
        self.goal = nominal_shell.copy()

    @staticmethod
    def _unit(vec):
        arr = np.asarray(vec, dtype=float)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return arr / norm

    @staticmethod
    def _smoothstep(u):
        u = np.asarray(u, dtype=float)
        return u * u * (3.0 - 2.0 * u)

    @staticmethod
    def _traj_cache_key(traj: np.ndarray):
        arr = np.ascontiguousarray(np.asarray(traj, dtype=np.float64))
        return arr.shape, arr.tobytes()

    @staticmethod
    def _smooth_trace(values: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        k = max(1, int(kernel_size))
        if k <= 1 or len(vals) == 0:
            return vals
        kernel = np.ones(k, dtype=float) / float(k)
        pad_left = k // 2
        pad_right = k - 1 - pad_left
        padded = np.pad(vals, (pad_left, pad_right), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _blend_segment_boundary(values: np.ndarray, boundary: int, half_window: int = 2) -> np.ndarray:
        out = np.asarray(values, dtype=float).copy()
        squeeze = False
        if out.ndim == 1:
            out = out[:, None]
            squeeze = True
        left = max(0, int(boundary) - int(half_window))
        right = min(len(out) - 1, int(boundary) + int(half_window) + 1)
        if left < 1 or right >= len(out) - 1 or right - left < 2:
            return out[:, 0] if squeeze else out

        p0 = out[left].copy()
        p1 = out[right].copy()
        span = float(right - left)
        m0 = (out[left] - out[left - 1]) * span
        m1 = (out[right + 1] - out[right]) * span

        u = np.linspace(0.0, 1.0, right - left + 1)
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        out[left : right + 1] = (
            h00[:, None] * p0
            + h10[:, None] * m0
            + h01[:, None] * p1
            + h11[:, None] * m1
        )
        return out[:, 0] if squeeze else out

    def _smooth_noise(self, rng, length: int, scale: float, kernel_size: int = 7) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.zeros(0, dtype=float)
        noise = rng.randn(n) * float(scale)
        return self._smooth_trace(noise, kernel_size=kernel_size)

    @staticmethod
    def _half_sine_wave(length: int, cycles: float, phase: float = 0.0) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.zeros(0, dtype=float)
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        base_wave = np.sin(2.0 * np.pi * float(cycles) * u - 0.5 * np.pi + float(phase))
        return np.maximum(base_wave, 0.0)

    def _make_stage_margin_profile(
        self,
        length: int,
        *,
        offset: float,
        amplitude: float,
        cycles: float,
        phase: float = 0.0,
        noise_scale: float = 0.0,
        rng=None,
        kernel_size: int = 5,
    ) -> np.ndarray:
        trace = float(amplitude) * self._half_sine_wave(length, cycles=cycles, phase=phase) - float(offset)
        if rng is not None and float(noise_scale) > 0.0:
            trace = trace + self._smooth_noise(rng, length, scale=float(noise_scale), kernel_size=kernel_size)
        trace = self._smooth_trace(trace, kernel_size=kernel_size)
        return np.asarray(trace, dtype=float)

    def _make_target_stage_trace(
        self,
        length: int,
        *,
        target: float,
        amplitude: float,
        cycles: float,
        phase: float = 0.0,
        noise_scale: float = 0.0,
        rng=None,
        kernel_size: int = 5,
        lower: float | None = 0.0,
        upper: float | None = None,
    ) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.zeros(0, dtype=float)
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        envelope = np.sin(np.pi * u)
        trace = float(target) + float(amplitude) * envelope * np.sin(2.0 * np.pi * float(cycles) * u + float(phase))
        if rng is not None and float(noise_scale) > 0.0:
            trace = trace + self._smooth_noise(rng, n, scale=float(noise_scale), kernel_size=kernel_size)
        trace = self._smooth_trace(trace, kernel_size=kernel_size)
        if lower is not None or upper is not None:
            lo = -np.inf if lower is None else float(lower)
            hi = np.inf if upper is None else float(upper)
            trace = np.clip(trace, lo, hi)
        return np.asarray(trace, dtype=float)

    def _make_irregular_positive_stage_trace(
        self,
        length: int,
        *,
        base: float,
        amplitude: float,
        phase: float = 0.0,
        noise_scale: float = 0.0,
        rng=None,
        kernel_size: int = 5,
        lower: float = 0.0,
        upper: float | None = None,
    ) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.zeros(0, dtype=float)
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        env1 = np.sin(np.pi * u)
        env2 = np.sin(np.pi * np.clip(1.15 * u, 0.0, 1.0)) ** 1.15
        wave = (
            0.55 * env1 * np.sin(4.5 * np.pi * u + float(phase))
            + 0.35 * env2 * np.sin(7.8 * np.pi * u - 0.55 * float(phase))
            + 0.20 * np.sin(11.6 * np.pi * u + 0.7 * float(phase))
        )
        trace = float(base) + float(amplitude) * wave
        if rng is not None and float(noise_scale) > 0.0:
            trace = trace + self._smooth_noise(rng, n, scale=float(noise_scale), kernel_size=kernel_size)
        trace = self._smooth_trace(trace, kernel_size=kernel_size)
        hi = np.inf if upper is None else float(upper)
        return np.clip(trace, float(lower), hi)

    def register_tool_axis_trace(self, traj: np.ndarray, tool_axis: np.ndarray):
        self._cached_tool_axis_traces[self._traj_cache_key(traj)] = np.asarray(tool_axis, dtype=float).copy()

    def _lookup_cached_tool_axis_trace(self, traj: np.ndarray):
        axis = self._cached_tool_axis_traces.get(self._traj_cache_key(traj))
        if axis is None:
            return None
        return np.asarray(axis, dtype=float)

    def register_feature_trace(self, traj: np.ndarray, features: np.ndarray):
        self._cached_feature_traces[self._traj_cache_key(traj)] = np.asarray(features, dtype=float).copy()

    def _lookup_cached_feature_trace(self, traj: np.ndarray):
        features = self._cached_feature_traces.get(self._traj_cache_key(traj))
        if features is None:
            return None
        return np.asarray(features, dtype=float)

    def _resolve_feature_boundary_ramp_half_windows(self, num_boundaries: int) -> np.ndarray:
        num_boundaries = max(int(num_boundaries), 0)
        feature_names = [
            "surface_distance",
            "tool_normal_alignment_error",
            "speed",
            "angular_speed",
        ]
        defaults = {
            "surface_distance": [2] * num_boundaries,
            "tool_normal_alignment_error": [0] * num_boundaries,
            "speed": ([1] + [2] * max(num_boundaries - 1, 0))[:num_boundaries],
            "angular_speed": [0] * num_boundaries,
        }
        cfg = self.feature_boundary_ramp_half_windows
        resolved = np.asarray(
            [np.asarray(defaults[name], dtype=int) for name in feature_names],
            dtype=int,
        )
        if cfg is None:
            return resolved
        if not isinstance(cfg, dict):
            raise ValueError("feature_boundary_ramp_half_windows must be a dict keyed by feature name.")
        for feat_idx, feature_name in enumerate(feature_names):
            value = cfg.get(feature_name, defaults[feature_name])
            if np.isscalar(value):
                resolved[feat_idx, :] = int(max(int(value), 0))
                continue
            arr = np.asarray(value, dtype=int).reshape(-1)
            if arr.size == 0:
                resolved[feat_idx, :] = 0
                continue
            if arr.size == 1:
                resolved[feat_idx, :] = int(max(int(arr[0]), 0))
                continue
            if arr.size != num_boundaries:
                raise ValueError(
                    f"feature_boundary_ramp_half_windows['{feature_name}'] must have length 1 or {num_boundaries}, got {arr.size}."
                )
            resolved[feat_idx, :] = np.maximum(arr.astype(int), 0)
        return resolved

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "surface_distance", "description": "Absolute radial distance to the sphere surface"},
            {"id": 1, "name": "tool_normal_alignment_error", "description": "Angle between tool axis and sphere normal"},
            {"id": 2, "name": "speed", "description": "3D speed magnitude"},
            {"id": 3, "name": "angular_speed", "description": "Tool-axis angular speed magnitude"},
            {"id": 4, "name": "noise_aux", "description": "Deterministic auxiliary irrelevant feature"},
        ]

    def get_true_constraints(self):
        base = {
            "surface_trace_target": 0.0,
            "surface_near_target": float(self.surface_near_target_ratio * self.shell_thickness),
            "surface_trace_max": float(0.018 * self.sphere_radius),
            "surface_near_max": float(self.shell_thickness),
            "tool_align_max_stage2": float(self.tool_align_max_stage2),
            "v23_max": float(self.stage2_speed_max),
        }
        return base

    def get_constraint_specs(self):
        if self.split_stage3_transition:
            return [
                {"feature_name": "surface_distance", "stage": 1, "semantics": "target_value", "oracle_key": "surface_trace_target"},
                {"feature_name": "tool_normal_alignment_error", "stage": 1, "semantics": "upper_bound", "oracle_key": "tool_align_max_stage2"},
                {"feature_name": "speed", "stage": 1, "semantics": "upper_bound", "oracle_key": "v23_max"},
                {"feature_name": "surface_distance", "stage": 3, "semantics": "target_value", "oracle_key": "surface_near_target"},
                {"feature_name": "speed", "stage": 3, "semantics": "upper_bound", "oracle_key": "v23_max"},
            ]
        return [
            {"feature_name": "surface_distance", "stage": 1, "semantics": "target_value", "oracle_key": "surface_trace_target"},
            {"feature_name": "tool_normal_alignment_error", "stage": 1, "semantics": "upper_bound", "oracle_key": "tool_align_max_stage2"},
            {"feature_name": "speed", "stage": 1, "semantics": "upper_bound", "oracle_key": "v23_max"},
            {"feature_name": "surface_distance", "stage": 2, "semantics": "target_value", "oracle_key": "surface_near_target"},
            {"feature_name": "speed", "stage": 2, "semantics": "upper_bound", "oracle_key": "v23_max"},
        ]

    def _sample_segment_lengths(self, rng):
        out = []
        for base, jitter in zip(self.seg_lengths, self.seg_length_jitter):
            delta = 0 if int(jitter) <= 0 else int(rng.randint(-int(jitter), int(jitter) + 1))
            out.append(max(int(base) + delta, 8))
        return tuple(out)

    def _orthonormal_frame(self, normal, rng):
        normal = self._unit(normal)
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(normal, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        t1 = np.cross(normal, ref)
        t1 = self._unit(t1)
        if float(np.linalg.norm(t1)) <= 1e-12:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(normal, ref))
        phase = float(rng.uniform(-np.pi, np.pi))
        t1 = self._unit(np.cos(phase) * t1 + np.sin(phase) * np.cross(normal, t1))
        t2 = self._unit(np.cross(normal, t1))
        return normal, t1, t2

    def _slerp_unit(self, u0, u1, num_points, endpoint=True):
        u0 = self._unit(u0)
        u1 = self._unit(u1)
        dots = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
        if dots > 0.9995:
            t = np.linspace(0.0, 1.0, int(num_points), endpoint=endpoint)
            out = (1.0 - t)[:, None] * u0[None, :] + t[:, None] * u1[None, :]
            return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        omega = float(np.arccos(dots))
        sin_omega = float(np.sin(omega))
        t = np.linspace(0.0, 1.0, int(num_points), endpoint=endpoint)
        out = (
            np.sin((1.0 - t) * omega)[:, None] * u0[None, :]
            + np.sin(t * omega)[:, None] * u1[None, :]
        ) / max(sin_omega, 1e-12)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    def _make_surface_path(self, n_start, n_end, num_points):
        normals = self._slerp_unit(n_start, n_end, num_points, endpoint=True)
        detour_angle = float(max(self.stage2_surface_detour_angle, 0.0))
        if detour_angle > 1e-8 and len(normals) > 2:
            axis = np.cross(self._unit(n_start), self._unit(n_end))
            if float(np.linalg.norm(axis)) <= 1e-8:
                _, _, detour_dir = self._orthonormal_frame(n_start, np.random.RandomState(0))
            else:
                axis = self._unit(axis)
                detour_dir = np.cross(axis, normals)
                detour_dir = detour_dir / np.maximum(np.linalg.norm(detour_dir, axis=1, keepdims=True), 1e-12)
            u = np.linspace(0.0, 1.0, len(normals), endpoint=True)
            bend = detour_angle * np.sin(np.pi * u)
            normals = (
                np.cos(bend)[:, None] * normals
                + np.sin(bend)[:, None] * detour_dir
            )
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        return self.sphere_center[None, :] + self.sphere_radius * normals

    def _resample_with_speed(self, path, v_max, a_max):
        ref = resample_polyline(np.asarray(path, dtype=float), max(float(v_max) * self.dt, 1e-3))
        return optimize_trajectory(
            ref,
            dt=self.dt,
            v_max=float(v_max),
            a_max=float(a_max),
            projector=None,
        )

    @staticmethod
    def _sample_range_value(rng, value_range):
        arr = np.asarray(value_range, dtype=float).reshape(-1)
        if arr.size == 0:
            return 1.0
        if arr.size == 1:
            return float(arr[0])
        lo = float(np.min(arr[:2]))
        hi = float(np.max(arr[:2]))
        return float(rng.uniform(lo, hi))

    @staticmethod
    def _split_polyline_by_fraction(path, fraction: float):
        pts = np.asarray(path, dtype=float)
        if len(pts) < 3:
            return pts.copy(), pts.copy()
        frac = float(np.clip(fraction, 0.1, 0.9))
        edges = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(edges)])
        total = float(cum[-1])
        if total <= 1e-9:
            split_idx = max(1, min(len(pts) - 2, int(round(frac * (len(pts) - 1)))))
            return pts[: split_idx + 1].copy(), pts[split_idx:].copy()
        target = frac * total
        split_idx = int(np.searchsorted(cum, target, side="right") - 1)
        split_idx = max(0, min(split_idx, len(pts) - 2))
        edge_len = float(edges[split_idx])
        if edge_len <= 1e-9:
            alpha = 0.0
        else:
            alpha = float(np.clip((target - cum[split_idx]) / edge_len, 0.0, 1.0))
        split_pt = (1.0 - alpha) * pts[split_idx] + alpha * pts[split_idx + 1]
        first = np.vstack([pts[: split_idx + 1], split_pt[None, :]])
        second = np.vstack([split_pt[None, :], pts[split_idx + 1 :]])
        return np.asarray(first, dtype=float), np.asarray(second, dtype=float)

    def _build_stage1(self, p_start, p_contact, n_points, v_max, a_max):
        mid = 0.35 * np.asarray(p_start, dtype=float) + 0.65 * np.asarray(p_contact, dtype=float)
        ctrl = np.vstack([p_start, mid, p_contact])
        return self._resample_with_speed(ctrl, v_max=v_max, a_max=a_max)

    def _build_stage3(self, n_start, n_end, n_points, rng):
        n_points = int(max(n_points, 8))
        n_start = self._unit(n_start)
        n_end = self._unit(n_end)
        _, t1, t2 = self._orthonormal_frame(n_start, rng)
        bend_sign = -1.0 if float(np.dot(t2, n_end)) < 0.0 else 1.0
        detour_dir = self._unit(0.45 * t1 + bend_sign * 0.55 * t2)
        if self.split_stage3_transition:
            detour_angle = float(rng.uniform(0.78, 1.00))
        else:
            detour_angle = float(rng.uniform(0.68, 0.92))
        n_mid = self._unit(np.cos(detour_angle) * n_start + np.sin(detour_angle) * detour_dir)

        split = max(3, n_points // 2)
        normals_a = self._slerp_unit(n_start, n_mid, split, endpoint=False)
        normals_b = self._slerp_unit(n_mid, n_end, n_points - split + 1, endpoint=True)
        normals = np.vstack([normals_a, normals_b[1:]])
        if len(normals) != n_points:
            normals = self._slerp_unit(n_start, n_end, n_points, endpoint=True)

        u = np.linspace(0.0, 1.0, len(normals))
        envelope = np.sin(np.pi * u) ** 1.15
        phase = float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi))
        if self.split_stage3_transition:
            base = 0.27 + 0.16 * np.sin(1.8 * np.pi * u + phase)
            ripple = 0.06 * np.sin(4.0 * np.pi * u - 0.4 * phase)
            radial_frac = np.clip((base + ripple) * envelope + 0.145, 0.08, 0.76)
        else:
            base = 0.24 + 0.15 * np.sin(1.8 * np.pi * u + phase)
            ripple = 0.05 * np.sin(4.0 * np.pi * u - 0.4 * phase)
            radial_frac = np.clip((base + ripple) * envelope + 0.13, 0.07, 0.72)
        radius = self.sphere_radius + self.shell_thickness * radial_frac
        return self.sphere_center[None, :] + radius[:, None] * normals

    def _build_stage4(self, p_start, n_start, rng):
        normal, t1, t2 = self._orthonormal_frame(n_start, rng)
        depart_radius = self.sphere_radius + self.depart_offset * rng.uniform(0.85, 1.15)
        lateral = 0.10 * self.sphere_radius * rng.uniform(-1.0, 1.0)
        vertical = 0.08 * self.sphere_radius * rng.uniform(-1.0, 1.0)
        p_end = (
            self.sphere_center
            + depart_radius * normal
            + lateral * t1
            + vertical * t2
        )
        ctrl = np.vstack([p_start, 0.55 * p_start + 0.45 * p_end, p_end])
        return ctrl

    def _interpolate_unit_axes(self, axis_start, axis_end, num_points):
        return self._slerp_unit(axis_start, axis_end, num_points, endpoint=True)

    def _make_irregular_axis_transition(self, axis_start, axis_end, num_points, rng, max_tilt):
        base = self._interpolate_unit_axes(axis_start, axis_end, num_points)
        n = len(base)
        if n == 0:
            return base
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        env = np.sin(np.pi * u) ** 0.78
        tilt = (
            0.30
            + 0.46 * np.sin(3.4 * np.pi * u - 0.15 * np.pi)
            + 0.24 * np.sin(7.2 * np.pi * u + 0.33 * np.pi)
            + 0.16 * np.sin(11.4 * np.pi * u - 0.52 * np.pi)
        )
        tilt = env * np.abs(tilt)
        tilt = self._smooth_trace(tilt, kernel_size=3)
        tilt = np.clip(tilt * float(max_tilt), 0.0, float(max_tilt))

        tangent_phase = (
            float(rng.uniform(-np.pi, np.pi))
            + 1.10 * np.sin(4.2 * np.pi * u)
            + 0.75 * np.sin(8.6 * np.pi * u - 0.40)
            + 0.35 * np.sin(14.0 * np.pi * u + 0.25)
        )
        tangent_phase = tangent_phase + self._smooth_noise(rng, n, scale=0.22, kernel_size=3)

        out = np.empty_like(base)
        for i, axis in enumerate(base):
            axis = self._unit(axis)
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(axis, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(axis, ref))
            t2 = self._unit(np.cross(axis, t1))
            tangent = np.cos(tangent_phase[i]) * t1 + np.sin(tangent_phase[i]) * t2
            out[i] = self._unit(np.cos(tilt[i]) * axis + np.sin(tilt[i]) * tangent)
        out[0] = self._unit(axis_start)
        out[-1] = self._unit(axis_end)
        return out

    def _make_aligned_axis_trace(self, normals, rng, max_error):
        normals = np.asarray(normals, dtype=float)
        n = len(normals)
        out = np.empty_like(normals)
        if n == 0:
            return out
        max_error = float(max(max_error, 1e-4))
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        phase0 = float(rng.uniform(-np.pi, np.pi))
        angle_margin = self._make_stage_margin_profile(
            n,
            offset=0.08 * max_error,
            amplitude=1.02 * max_error,
            cycles=4.6,
            phase=0.0,
            noise_scale=0.0,
            rng=None,
            kernel_size=1,
        )
        angle = np.minimum(max_error, max_error - angle_margin)
        angle = np.clip(angle, 0.24 * max_error, 0.95 * max_error)

        tangent_phase = (
            phase0
            + 1.15 * np.sin(2.9 * np.pi * u + 0.18 * phase0)
            + 0.55 * np.sin(5.8 * np.pi * u - 0.33 * phase0)
        )
        tangent_phase = self._smooth_trace(tangent_phase, kernel_size=7)

        for i, normal in enumerate(normals):
            normal = self._unit(normal)
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(normal, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(normal, ref))
            t2 = self._unit(np.cross(normal, t1))
            tangent = np.cos(tangent_phase[i]) * t1 + np.sin(tangent_phase[i]) * t2
            out[i] = self._unit(np.cos(angle[i]) * normal + np.sin(angle[i]) * tangent)
        return out

    def _generate_tool_axis_trace(self, traj, stage_lengths, normals_stage2, normals_stage3, n_contact, rng):
        lengths = [int(x) for x in stage_lengths]
        total = int(sum(stage_lengths))
        axis_start = self._unit(0.76 * self._unit(n_contact) + 0.24 * self._unit(rng.randn(3)))
        l1 = lengths[0]
        stage1 = self._make_irregular_axis_transition(
            axis_start,
            n_contact,
            l1,
            rng=rng,
            max_tilt=0.38 * float(self.tool_align_max_stage2),
        )
        stage2 = self._make_aligned_axis_trace(normals_stage2, rng, max_error=self.tool_align_max_stage2)

        if len(lengths) == 5:
            _, _, l3, l4, l5 = lengths
            mid_anchor = self._unit(0.72 * self._unit(normals_stage3[0]) + 0.28 * self._unit(rng.randn(3)))
            stage3 = self._make_irregular_axis_transition(
                stage2[-1],
                mid_anchor,
                l3,
                rng=rng,
                max_tilt=0.46 * float(self.tool_align_max_stage2),
            )
            free_axis = self._unit(0.6 * normals_stage3[-1] + 0.4 * self._unit(rng.randn(3)))
            stage4 = self._make_irregular_axis_transition(
                stage3[-1],
                free_axis,
                l4,
                rng=rng,
                max_tilt=0.92 * float(self.tool_align_max_stage2),
            )
            stage5_end = self._unit(0.78 * self._unit(stage4[-1]) + 0.22 * self._unit(rng.randn(3)))
            stage5 = self._make_irregular_axis_transition(
                stage4[-1],
                stage5_end,
                l5,
                rng=rng,
                max_tilt=0.36 * float(self.tool_align_max_stage2),
            )
            axis = np.vstack([stage1, stage2, stage3, stage4, stage5])
        else:
            _, _, l3, l4 = lengths
            free_axis = self._unit(0.6 * normals_stage3[-1] + 0.4 * self._unit(rng.randn(3)))
            stage3 = self._make_irregular_axis_transition(
                stage2[-1],
                free_axis,
                l3,
                rng=rng,
                max_tilt=0.92 * float(self.tool_align_max_stage2),
            )
            stage4_end = self._unit(0.78 * self._unit(stage3[-1]) + 0.22 * self._unit(rng.randn(3)))
            stage4 = self._make_irregular_axis_transition(
                stage3[-1],
                stage4_end,
                l4,
                rng=rng,
                max_tilt=0.36 * float(self.tool_align_max_stage2),
            )
            axis = np.vstack([stage1, stage2, stage3, stage4])
        if axis.shape[0] != total:
            axis = np.asarray(axis[:total], dtype=float)
        return axis

    def generate_demo(self, rng=None, **kwargs):
        rng = np.random if rng is None else rng
        l1, l2, l3, l4 = self._sample_segment_lengths(rng)

        theta0 = float(rng.uniform(-0.55 * np.pi, 0.15 * np.pi))
        phi0 = float(rng.uniform(0.22 * np.pi, 0.42 * np.pi))
        n_contact = self._unit(
            [
                np.cos(theta0) * np.sin(phi0),
                np.sin(theta0) * np.sin(phi0),
                np.cos(phi0),
            ]
        )
        normal0, t1, t2 = self._orthonormal_frame(n_contact, rng)

        trace_angle = float(rng.uniform(self.stage2_trace_angle_range[0], self.stage2_trace_angle_range[1]))
        n_trace_end = self._unit(np.cos(trace_angle) * normal0 + np.sin(trace_angle) * t1)

        repos_angle = float(rng.uniform(0.65, 1.10))
        blend_dir = self._unit(0.55 * t2 + 0.45 * n_trace_end)
        n_repos_end = self._unit(np.cos(repos_angle) * n_trace_end + np.sin(repos_angle) * blend_dir)

        p_contact = self.sphere_center + self.sphere_radius * n_contact
        p_precontact = self.sphere_center + (self.sphere_radius + 0.18) * n_contact
        p_start = (
            self.sphere_center
            + (self.sphere_radius + self.approach_offset * rng.uniform(0.85, 1.15)) * n_contact
            + 0.18 * self.sphere_radius * rng.uniform(-1.0, 1.0) * t1
            + 0.12 * self.sphere_radius * rng.uniform(-1.0, 1.0) * t2
        )

        stage1 = self._build_stage1(p_start, p_contact, l1, v_max=self.stage1_speed_max, a_max=self.stage1_accel_max)
        stage2_length_scale = max(self._sample_range_value(rng, self.stage2_length_scale_range), 1e-3)
        stage4_length_scale = max(self._sample_range_value(rng, self.stage4_length_scale_range), 1e-3)
        stage2_raw = self._make_surface_path(n_contact, n_trace_end, l2)
        stage2 = self._resample_with_speed(
            stage2_raw,
            v_max=self.stage2_speed_max / stage2_length_scale,
            a_max=self.stage2_accel_max / stage2_length_scale,
        )
        stage3_raw_full = self._build_stage3(n_trace_end, n_repos_end, l3, rng=rng)
        if self.split_stage3_transition:
            stage3_raw, stage4_raw = self._split_polyline_by_fraction(
                stage3_raw_full,
                fraction=self.transition_stage_fraction,
            )
            stage3 = self._resample_with_speed(stage3_raw, v_max=self.stage2_speed_max, a_max=self.stage3_accel_max)
            stage4 = self._resample_with_speed(
                stage4_raw,
                v_max=self.stage2_speed_max / stage4_length_scale,
                a_max=self.stage3_accel_max / stage4_length_scale,
            )
            stage5_ctrl = self._build_stage4(stage4[-1], n_repos_end, rng=rng)
            stage5 = self._resample_with_speed(stage5_ctrl, v_max=self.stage4_speed_max, a_max=self.stage4_accel_max)
            traj = np.vstack([stage1, stage2[1:], stage3[1:], stage4[1:], stage5[1:]])
        else:
            stage3 = self._resample_with_speed(stage3_raw_full, v_max=self.stage2_speed_max, a_max=self.stage3_accel_max)
            stage4_ctrl = self._build_stage4(stage3[-1], n_repos_end, rng=rng)
            stage4 = self._resample_with_speed(stage4_ctrl, v_max=self.stage4_speed_max, a_max=self.stage4_accel_max)
            traj = np.vstack([stage1, stage2[1:], stage3[1:], stage4[1:]])
        if self.noise_std > 0.0:
            traj = traj + rng.randn(*traj.shape) * self.noise_std
            radii = np.linalg.norm(traj - self.sphere_center[None, :], axis=1)
            contact_mask = slice(len(stage1), len(stage1) + len(stage2) - 1)
            safe = np.maximum(radii[contact_mask], 1e-12)
            traj[contact_mask] = self.sphere_center[None, :] + (
                self.sphere_radius * (traj[contact_mask] - self.sphere_center[None, :]) / safe[:, None]
            )

        if self.split_stage3_transition:
            true_cutpoints = np.asarray(
                [
                    int(len(stage1) - 1),
                    int(len(stage1) + len(stage2) - 2),
                    int(len(stage1) + len(stage2) + len(stage3) - 3),
                    int(len(stage1) + len(stage2) + len(stage3) + len(stage4) - 4),
                ],
                dtype=int,
            )
            stage_lengths = (len(stage1), len(stage2) - 1, len(stage3) - 1, len(stage4) - 1, len(stage5) - 1)
            stage2_slice = slice(len(stage1), len(stage1) + len(stage2) - 1)
            stage4_slice = slice(
                len(stage1) + len(stage2) + len(stage3) - 2,
                len(stage1) + len(stage2) + len(stage3) + len(stage4) - 3,
            )
            normals_stage2 = traj[stage2_slice] - self.sphere_center[None, :]
            normals_stage2 = normals_stage2 / np.maximum(np.linalg.norm(normals_stage2, axis=1, keepdims=True), 1e-12)
            normals_stage4 = traj[stage4_slice] - self.sphere_center[None, :]
            normals_stage4 = normals_stage4 / np.maximum(np.linalg.norm(normals_stage4, axis=1, keepdims=True), 1e-12)
            tool_axis = self._generate_tool_axis_trace(
                traj=traj,
                stage_lengths=stage_lengths,
                normals_stage2=normals_stage2,
                normals_stage3=normals_stage4,
                n_contact=n_contact,
                rng=rng,
            )
        else:
            true_cutpoints = np.asarray(
                [
                    int(len(stage1) - 1),
                    int(len(stage1) + len(stage2) - 2),
                    int(len(stage1) + len(stage2) + len(stage3) - 3),
                ],
                dtype=int,
            )
            stage_lengths = (len(stage1), len(stage2) - 1, len(stage3) - 1, len(stage4) - 1)
            stage2_slice = slice(len(stage1), len(stage1) + len(stage2) - 1)
            stage3_slice = slice(len(stage1) + len(stage2) - 1, len(stage1) + len(stage2) + len(stage3) - 2)
            normals_stage2 = traj[stage2_slice] - self.sphere_center[None, :]
            normals_stage2 = normals_stage2 / np.maximum(np.linalg.norm(normals_stage2, axis=1, keepdims=True), 1e-12)
            normals_stage3 = traj[stage3_slice] - self.sphere_center[None, :]
            normals_stage3 = normals_stage3 / np.maximum(np.linalg.norm(normals_stage3, axis=1, keepdims=True), 1e-12)
            tool_axis = self._generate_tool_axis_trace(
                traj=traj,
                stage_lengths=stage_lengths,
                normals_stage2=normals_stage2,
                normals_stage3=normals_stage3,
                n_contact=n_contact,
                rng=rng,
            )
        if tool_axis.shape[0] != len(traj):
            axis_fixed = np.empty((len(traj), 3), dtype=float)
            axis_fixed[: min(len(tool_axis), len(traj))] = tool_axis[: min(len(tool_axis), len(traj))]
            if len(tool_axis) < len(traj):
                axis_fixed[len(tool_axis) :] = tool_axis[-1]
            tool_axis = axis_fixed
        self.register_tool_axis_trace(traj, tool_axis)
        feature_trace = self._synthesize_feature_trace(
            traj,
            tool_axis=tool_axis,
            stage_lengths=stage_lengths,
            rng=rng,
        )
        self.register_feature_trace(traj, feature_trace)
        return traj, true_cutpoints

    def generate_demos(self, n_demos=10, rng=None, **kwargs):
        rng = np.random if rng is None else rng
        demos = []
        true_cutpoints = []
        for _ in range(int(n_demos)):
            traj, cutpoints = self.generate_demo(rng=rng, **kwargs)
            demos.append(np.asarray(traj, dtype=float))
            true_cutpoints.append(np.asarray(cutpoints, dtype=int))
        return demos, true_cutpoints

    def _estimate_tool_axis_from_geometry(self, traj):
        pts = np.asarray(traj, dtype=float)
        rel = pts - self.sphere_center[None, :]
        normals = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
        return normals

    def _compute_geometry_feature_traces(self, traj, tool_axis=None):
        traj = np.asarray(traj, dtype=float)
        T = len(traj)
        rel = traj - self.sphere_center[None, :]
        radial_dist = np.linalg.norm(rel, axis=1)
        surface_distance = np.abs(radial_dist - self.sphere_radius)

        if tool_axis is None:
            tool_axis = self._lookup_cached_tool_axis_trace(traj)
        if tool_axis is None:
            tool_axis = self._estimate_tool_axis_from_geometry(traj)
        tool_axis = np.asarray(tool_axis, dtype=float)
        tool_axis = tool_axis / np.maximum(np.linalg.norm(tool_axis, axis=1, keepdims=True), 1e-12)
        normals = rel / np.maximum(radial_dist[:, None], 1e-12)
        cos_align = np.sum(tool_axis * normals, axis=1)
        cos_align = np.clip(cos_align, -1.0, 1.0)
        tool_normal_alignment_error = np.arccos(cos_align)

        speed = np.zeros(T, dtype=float)
        if T > 1:
            speed_edge = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.dt
            speed[0] = speed_edge[0]
            speed[1:] = speed_edge

        angular_speed = np.zeros(T, dtype=float)
        if T > 1:
            dots = np.sum(tool_axis[1:] * tool_axis[:-1], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            ang = np.arccos(dots) / self.dt
            angular_speed[0] = ang[0]
            angular_speed[1:] = ang

        return surface_distance, tool_normal_alignment_error, speed, angular_speed

    def _synthesize_feature_trace(self, traj, tool_axis, stage_lengths, rng):
        surface_distance, tool_alignment_error, speed, angular_speed = self._compute_geometry_feature_traces(
            traj,
            tool_axis=tool_axis,
        )

        lengths = [int(x) for x in stage_lengths]
        starts = np.cumsum([0] + lengths[:-1]).tolist()
        ends = [s + l for s, l in zip(starts, lengths)]
        phase = float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi))

        if len(lengths) == 5:
            speed_maxima = [
                float(self.stage1_speed_max),
                float(self.stage2_speed_max),
                float(self.stage2_speed_max),
                float(self.stage2_speed_max),
                float(self.stage4_speed_max),
            ]
            speed_offsets = [0.26, 0.20, 0.24, 0.20, 0.24]
            speed_amps = [0.52, 0.60, 0.38, 0.42, 0.50]
            speed_cycles = [1.55, 2.15, 1.75, 1.90, 1.65]
        else:
            speed_maxima = [
                float(self.stage1_speed_max),
                float(self.stage2_speed_max),
                float(self.stage2_speed_max),
                float(self.stage4_speed_max),
            ]
            speed_offsets = [0.26, 0.20, 0.28, 0.24]
            speed_amps = [0.52, 0.60, 0.42, 0.50]
            speed_cycles = [1.55, 2.15, 1.90, 1.65]
        for stage_idx, (s, e, vmax) in enumerate(zip(starts, ends, speed_maxima)):
            if stage_idx == 0:
                profile = self._make_irregular_positive_stage_trace(
                    e - s,
                    base=0.52 * vmax,
                    amplitude=0.34 * vmax,
                    phase=phase,
                    noise_scale=0.08 * vmax,
                    rng=rng,
                    kernel_size=3,
                    lower=0.18 * vmax,
                    upper=0.995 * vmax,
                )
                speed[s:e] = 0.28 * np.asarray(speed[s:e], dtype=float) + 0.72 * profile
                continue
            if (len(lengths) == 5 and stage_idx in {2, 4}) or (len(lengths) != 5 and stage_idx == 3):
                base = 0.60 * vmax if stage_idx == 2 else 0.56 * vmax
                amplitude = 0.34 * vmax if stage_idx == 2 else 0.40 * vmax
                upper = 1.08 * vmax if stage_idx == 2 else 1.16 * vmax
                profile = self._make_irregular_positive_stage_trace(
                    e - s,
                    base=base,
                    amplitude=amplitude,
                    phase=phase + 0.50 + 0.20 * stage_idx,
                    noise_scale=0.04 * vmax if stage_idx == 2 else 0.05 * vmax,
                    rng=rng,
                    kernel_size=3,
                    lower=0.16 * vmax,
                    upper=upper,
                )
                mix = 0.78 if stage_idx == 2 else 0.76
                speed[s:e] = (1.0 - mix) * np.asarray(speed[s:e], dtype=float) + mix * profile
                continue
            margin = self._make_stage_margin_profile(
                e - s,
                offset=speed_offsets[stage_idx] * vmax,
                amplitude=speed_amps[stage_idx] * vmax,
                cycles=speed_cycles[stage_idx],
                phase=phase + 0.18 * stage_idx,
                noise_scale=(0.028 if stage_idx == 1 else 0.022) * vmax,
                rng=rng,
                kernel_size=5,
            )
            profile = np.minimum(vmax, vmax - margin)
            profile = np.clip(profile, 0.0, None)
            speed[s:e] = 0.12 * np.asarray(speed[s:e], dtype=float) + 0.88 * profile

        s2, e2 = starts[1], ends[1]
        surface_distance[s2:e2] = self._make_target_stage_trace(
            e2 - s2,
            target=float(self.true_constraints["surface_trace_target"]),
            amplitude=0.22 * self.true_constraints["surface_trace_max"],
            cycles=1.35,
            phase=phase,
            noise_scale=max(0.18 * self.true_constraints["surface_trace_max"], 1e-4),
            rng=rng,
            kernel_size=5,
            lower=0.0,
            upper=float(self.true_constraints["surface_trace_max"]),
        )
        near_stage_idx = 3 if len(lengths) == 5 else 2
        s3, e3 = starts[near_stage_idx], ends[near_stage_idx]
        if len(lengths) == 5:
            near_amplitude = 0.12 * self.true_constraints["surface_near_target"]
            near_cycles = 0.85
            near_noise_scale = max(0.05 * self.true_constraints["surface_near_target"], 1e-4)
            near_kernel = 7
        else:
            near_amplitude = 0.22 * self.true_constraints["surface_near_target"]
            near_cycles = 1.2
            near_noise_scale = max(0.10 * self.true_constraints["surface_near_target"], 1e-4)
            near_kernel = 5
        surface_distance[s3:e3] = self._make_target_stage_trace(
            e3 - s3,
            target=float(self.true_constraints["surface_near_target"]),
            amplitude=near_amplitude,
            cycles=near_cycles,
            phase=phase - 0.25,
            noise_scale=near_noise_scale,
            rng=rng,
            kernel_size=near_kernel,
            lower=0.0,
            upper=float(self.true_constraints["surface_near_max"]),
        )

        boundaries = np.cumsum(lengths[:-1]) - 1
        feature_matrix_raw = np.stack(
            [
                surface_distance,
                tool_alignment_error,
                speed,
                angular_speed,
            ],
            axis=1,
        )
        feature_matrix = np.asarray(feature_matrix_raw, dtype=float).copy()
        ramp_windows = self._resolve_feature_boundary_ramp_half_windows(len(boundaries))
        for feat_idx in range(feature_matrix.shape[1]):
            trace = np.asarray(feature_matrix_raw[:, feat_idx], dtype=float).copy()
            for boundary_idx, boundary in enumerate(boundaries.tolist()):
                half_window = int(ramp_windows[feat_idx, boundary_idx])
                if half_window <= 0:
                    continue
                trace = self._blend_segment_boundary(
                    trace,
                    boundary=int(boundary),
                    half_window=half_window,
                )
            feature_matrix[:, feat_idx] = trace

        for stage_idx, (s, e, vmax) in enumerate(zip(starts, ends, speed_maxima)):
            if len(lengths) == 5:
                unconstrained_speed_stage = {2, 4}
            else:
                unconstrained_speed_stage = {3}
            upper = None if stage_idx in unconstrained_speed_stage else float(vmax)
            feature_matrix[s:e, 2] = np.clip(feature_matrix[s:e, 2], 0.0, np.inf if upper is None else upper)
        feature_matrix[s2:e2, 0] = np.clip(feature_matrix[s2:e2, 0], 0.0, float(self.true_constraints["surface_trace_max"]))
        feature_matrix[s3:e3, 0] = np.clip(feature_matrix[s3:e3, 0], 0.0, float(self.true_constraints["surface_near_max"]))
        return np.asarray(feature_matrix, dtype=float)

    def compute_all_features_matrix(self, traj, feat_ids=None):
        traj = np.asarray(traj, dtype=float)
        T = len(traj)
        cached_features = self._lookup_cached_feature_trace(traj)
        if cached_features is not None and cached_features.shape[0] == T:
            surface_distance = np.asarray(cached_features[:, 0], dtype=float)
            tool_normal_alignment_error = np.asarray(cached_features[:, 1], dtype=float)
            speed = np.asarray(cached_features[:, 2], dtype=float)
            angular_speed = np.asarray(cached_features[:, 3], dtype=float)
        else:
            surface_distance, tool_normal_alignment_error, speed, angular_speed = self._compute_geometry_feature_traces(traj)

        t = np.linspace(0.0, 2.0 * np.pi, T)
        phase = float(0.31 * np.mean(traj[:, 0]) - 0.27 * np.mean(traj[:, 1]) + 0.43 * np.mean(traj[:, 2]))
        noise_aux = 0.15 * np.sin(4.3 * t + phase) + 0.08 * np.cos(1.7 * t - 0.5 * phase)

        F = np.stack(
            [
                surface_distance,
                tool_normal_alignment_error,
                speed,
                angular_speed,
                noise_aux,
            ],
            axis=1,
        )
        return F if feat_ids is None else F[:, feat_ids]

    def compute_features_all(self, traj):
        F = self.compute_all_features_matrix(traj)
        return F[:, 0], F[:, 2]


def _build_sphere_inspect_bundle(
    *,
    task_name: str,
    n_demos: int,
    seed: int,
    env_kwargs=None,
    demo_kwargs=None,
    **extra_env_kwargs,
) -> TaskBundle:
    env_cfg = dict(env_kwargs or {})
    env_cfg.update(extra_env_kwargs)
    run_kwargs = dict(demo_kwargs or {})
    rng = np.random.RandomState(seed)
    env = SphereInspectEnv3D(**env_cfg)
    demos, true_cutpoints = env.generate_demos(n_demos=n_demos, rng=rng, **run_kwargs)
    true_taus = [None for _ in demos]
    return TaskBundle(
        name=task_name,
        demos=demos,
        env=env,
        true_taus=true_taus,
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=dict(env.true_constraints),
        constraint_specs=env.get_constraint_specs(),
        meta={"seed": seed, "task_name": task_name},
    )


def load_3d_sphere_inspect(
    n_demos: int = 10,
    seed: int = 0,
    env_kwargs=None,
    demo_kwargs=None,
    **extra_env_kwargs,
) -> TaskBundle:
    env_cfg = dict(env_kwargs or {})
    env_cfg.update(extra_env_kwargs)
    env_cfg.setdefault("seg_lengths", (18, 34, 33, 18))
    env_cfg.setdefault("seg_length_jitter", (3, 5, 5, 3))
    env_cfg.setdefault("surface_near_target_ratio", 0.62)
    env_cfg.setdefault("split_stage3_transition", True)
    env_cfg.setdefault("transition_stage_fraction", 0.40)
    env_cfg.setdefault("stage2_speed_max", 0.047)
    env_cfg.setdefault("stage3_speed_max", 0.047)
    env_cfg.setdefault("stage2_trace_angle_range", (2.85, 3.12))
    env_cfg.setdefault("stage2_surface_detour_angle", 0.42)
    env_cfg.setdefault("stage2_length_scale_range", (0.4, 0.8))
    env_cfg.setdefault("stage4_length_scale_range", (0.5, 1.0))
    env_cfg.setdefault(
        "feature_boundary_ramp_half_windows",
        {
            "surface_distance": [3, 2, 1, 5],
            "tool_normal_alignment_error": [1, 3, 2, 1],
            "speed": [1, 2, 2, 1],
            "angular_speed": [1, 2, 2, 1],
        },
    )
    env_cfg.setdefault("eval_tag", "3DSphereInspect")
    return _build_sphere_inspect_bundle(
        task_name="3DSphereInspect",
        n_demos=n_demos,
        seed=seed,
        env_kwargs=env_cfg,
        demo_kwargs=demo_kwargs,
    )
