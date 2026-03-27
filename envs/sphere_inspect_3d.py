from __future__ import annotations

import numpy as np

from .base import TaskBundle
from planner import optimize_trajectory, resample_polyline


class SphereInspectEnv3D:
    """
    Four-stage 3D spherical surface inspection task.

    Stage 1: approach the sphere with bounded speed.
    Stage 2: trace on the sphere surface with aligned tool normal.
    Stage 3: reposition near the sphere surface within a shell.
    Stage 4: depart from the sphere with bounded speed.
    """

    def __init__(
        self,
        sphere_center=(0.0, 0.0, 0.0),
        sphere_radius=1.0,
        shell_thickness=0.14,
        seg_lengths=(18, 34, 22, 18),
        seg_length_jitter=(3, 5, 4, 3),
        approach_offset=0.42,
        depart_offset=0.50,
        stage1_speed_max=0.12,
        stage2_speed_max=0.05,
        stage3_speed_max=0.11,
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
        self.eval_tag = "3DSphereInspect4"

        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self._cached_tool_axis_traces = {}

        nominal_contact = self.sphere_center + np.array([0.0, self.sphere_radius, 0.0], dtype=float)
        nominal_shell = self.sphere_center + np.array([0.0, self.sphere_radius + 0.5 * self.shell_thickness, 0.0], dtype=float)
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

    def register_tool_axis_trace(self, traj: np.ndarray, tool_axis: np.ndarray):
        self._cached_tool_axis_traces[self._traj_cache_key(traj)] = np.asarray(tool_axis, dtype=float).copy()

    def _lookup_cached_tool_axis_trace(self, traj: np.ndarray):
        axis = self._cached_tool_axis_traces.get(self._traj_cache_key(traj))
        if axis is None:
            return None
        return np.asarray(axis, dtype=float)

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "surface_distance", "description": "Absolute radial distance to the sphere surface"},
            {"id": 1, "name": "tool_normal_alignment_error", "description": "Angle between tool axis and sphere normal"},
            {"id": 2, "name": "speed", "description": "3D speed magnitude"},
            {"id": 3, "name": "angular_speed", "description": "Tool-axis angular speed magnitude"},
            {"id": 4, "name": "noise_aux", "description": "Deterministic auxiliary irrelevant feature"},
        ]

    def get_true_constraints(self):
        return {
            "surface_trace_max": float(0.018 * self.sphere_radius),
            "surface_near_max": float(self.shell_thickness),
            "tool_align_max_stage2": float(self.tool_align_max_stage2),
            "v1_max": float(self.stage1_speed_max),
            "v2_max": float(self.stage2_speed_max),
            "v3_max": float(self.stage3_speed_max),
            "v4_max": float(self.stage4_speed_max),
            "w2_max": float(self.angular_speed_max_stage2),
            "w3_max": float(self.angular_speed_max_stage3),
        }

    def get_constraint_specs(self):
        return [
            {"feature_name": "speed", "stage": 0, "semantics": "upper_bound", "oracle_key": "v1_max"},
            {"feature_name": "surface_distance", "stage": 1, "semantics": "upper_bound", "oracle_key": "surface_trace_max"},
            {"feature_name": "tool_normal_alignment_error", "stage": 1, "semantics": "upper_bound", "oracle_key": "tool_align_max_stage2"},
            {"feature_name": "speed", "stage": 1, "semantics": "upper_bound", "oracle_key": "v2_max"},
            {"feature_name": "angular_speed", "stage": 1, "semantics": "upper_bound", "oracle_key": "w2_max"},
            {"feature_name": "surface_distance", "stage": 2, "semantics": "upper_bound", "oracle_key": "surface_near_max"},
            {"feature_name": "speed", "stage": 2, "semantics": "upper_bound", "oracle_key": "v3_max"},
            {"feature_name": "angular_speed", "stage": 2, "semantics": "upper_bound", "oracle_key": "w3_max"},
            {"feature_name": "speed", "stage": 3, "semantics": "upper_bound", "oracle_key": "v4_max"},
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

    def _build_stage1(self, p_start, p_contact, n_points, v_max, a_max):
        mid = 0.35 * np.asarray(p_start, dtype=float) + 0.65 * np.asarray(p_contact, dtype=float)
        ctrl = np.vstack([p_start, mid, p_contact])
        return self._resample_with_speed(ctrl, v_max=v_max, a_max=a_max)

    def _build_stage3(self, n_start, n_end, n_points, rng):
        normals = self._slerp_unit(n_start, n_end, n_points, endpoint=True)
        u = np.linspace(0.0, 1.0, int(n_points))
        envelope = np.sin(np.pi * u) ** 1.2
        phase = float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi))
        base = 0.18 + 0.20 * np.sin(1.6 * np.pi * u + phase)
        radial_frac = np.clip(base * envelope + 0.12, 0.05, 0.78)
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

    def _jitter_axis_about_normal(self, normals, rng, max_error):
        normals = np.asarray(normals, dtype=float)
        out = np.empty_like(normals)
        max_error = float(max(max_error, 1e-4))
        for i, normal in enumerate(normals):
            normal = self._unit(normal)
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(normal, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(normal, ref))
            t2 = self._unit(np.cross(normal, t1))
            angle = float(rng.uniform(0.0, max_error))
            phase = float(rng.uniform(-np.pi, np.pi))
            tangent = np.cos(phase) * t1 + np.sin(phase) * t2
            out[i] = self._unit(np.cos(angle) * normal + np.sin(angle) * tangent)
        return out

    def _generate_tool_axis_trace(self, traj, stage_lengths, normals_stage2, normals_stage3, n_contact, rng):
        l1, l2, l3, l4 = [int(x) for x in stage_lengths]
        total = int(sum(stage_lengths))
        axis_start = self._unit(rng.randn(3))
        stage1 = self._interpolate_unit_axes(axis_start, n_contact, l1)
        stage2 = self._jitter_axis_about_normal(normals_stage2, rng, max_error=0.65 * self.tool_align_max_stage2)

        free_axis = self._unit(0.6 * normals_stage3[-1] + 0.4 * self._unit(rng.randn(3)))
        stage3_base = self._interpolate_unit_axes(stage2[-1], free_axis, l3)
        stage3 = np.asarray(stage3_base, dtype=float)

        stage4_end = self._unit(rng.randn(3))
        stage4 = self._interpolate_unit_axes(stage3[-1], stage4_end, l4)

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

        trace_angle = float(rng.uniform(0.55, 1.05))
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
        stage2_raw = self._make_surface_path(n_contact, n_trace_end, l2)
        stage2 = self._resample_with_speed(stage2_raw, v_max=self.stage2_speed_max, a_max=self.stage2_accel_max)
        stage3_raw = self._build_stage3(n_trace_end, n_repos_end, l3, rng=rng)
        stage3 = self._resample_with_speed(stage3_raw, v_max=self.stage3_speed_max, a_max=self.stage3_accel_max)
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

        true_cutpoints = np.asarray(
            [
                int(len(stage1) - 1),
                int(len(stage1) + len(stage2) - 2),
                int(len(stage1) + len(stage2) + len(stage3) - 3),
            ],
            dtype=int,
        )

        normals_stage2 = self._slerp_unit(n_contact, n_trace_end, len(stage2), endpoint=True)
        normals_stage3 = self._slerp_unit(n_trace_end, n_repos_end, len(stage3), endpoint=True)
        tool_axis = self._generate_tool_axis_trace(
            traj=traj,
            stage_lengths=(len(stage1), len(stage2) - 1, len(stage3) - 1, len(stage4) - 1),
            normals_stage2=normals_stage2[1:],
            normals_stage3=normals_stage3[1:],
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

    def compute_all_features_matrix(self, traj, feat_ids=None):
        traj = np.asarray(traj, dtype=float)
        T = len(traj)
        rel = traj - self.sphere_center[None, :]
        radial_dist = np.linalg.norm(rel, axis=1)
        surface_distance = np.abs(radial_dist - self.sphere_radius)

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


def load_3d_sphere_inspect_4(
    n_demos: int = 10,
    seed: int = 0,
    env_kwargs=None,
    demo_kwargs=None,
) -> TaskBundle:
    env_cfg = dict(env_kwargs or {})
    run_kwargs = dict(demo_kwargs or {})
    rng = np.random.RandomState(seed)
    env = SphereInspectEnv3D(**env_cfg)
    demos, true_cutpoints = env.generate_demos(n_demos=n_demos, rng=rng, **run_kwargs)
    true_taus = [None for _ in demos]
    return TaskBundle(
        name="3DSphereInspect4",
        demos=demos,
        env=env,
        true_taus=true_taus,
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=dict(env.true_constraints),
        constraint_specs=env.get_constraint_specs(),
        meta={"seed": seed, "task_name": "3DSphereInspect4"},
    )
