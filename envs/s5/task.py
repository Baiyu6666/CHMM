from __future__ import annotations

import numpy as np

from .config import S5_METRIC_SCALE, S5SyntheticPreset, S5_SYNTHETIC_V23
from .execution import S5ExecutionMixin
from .features import S5FeatureMixin
from .generator import S5DemoGeneratorMixin
from .planner import S5ConstraintPlannerMixin
from .rendering import S5RenderingMixin

_S5_METRIC_SCALE = S5_METRIC_SCALE


class S5SphereInspectEnv(
    S5FeatureMixin,
    S5DemoGeneratorMixin,
    S5ExecutionMixin,
    S5ConstraintPlannerMixin,
    S5RenderingMixin,
):
    """
    3D spherical surface inspection task.
    """

    preset_name = S5_SYNTHETIC_V23.name
    preset_version = S5_SYNTHETIC_V23.version

    @classmethod
    def from_preset(
        cls,
        preset: S5SyntheticPreset = S5_SYNTHETIC_V23,
        **overrides,
    ):
        env_kwargs = preset.apply(overrides)
        env_kwargs.pop("cache_demos", None)
        instance = cls(**env_kwargs)
        instance.preset_name = str(preset.name)
        instance.preset_version = int(preset.version)
        return instance

    def __init__(
        self,
        sphere_center=(0.0, 0.0, 0.0),
        sphere_radius=_S5_METRIC_SCALE,
        shell_thickness=0.24 * _S5_METRIC_SCALE,
        seg_lengths=(18, 34, 24, 18),
        seg_length_jitter=(3, 5, 5, 3),
        approach_offset=0.42 * _S5_METRIC_SCALE,
        depart_offset=0.50 * _S5_METRIC_SCALE,
        stage1_speed_max=0.12 * _S5_METRIC_SCALE,
        stage2_speed_max=0.05 * _S5_METRIC_SCALE,
        stage3_speed_max=0.06 * _S5_METRIC_SCALE,
        stage4_speed_max=0.09 * _S5_METRIC_SCALE,
        stage1_accel_max=0.08 * _S5_METRIC_SCALE,
        stage2_accel_max=0.03 * _S5_METRIC_SCALE,
        stage3_accel_max=0.07 * _S5_METRIC_SCALE,
        stage4_accel_max=0.06 * _S5_METRIC_SCALE,
        tool_align_max_stage2=0.04,
        dt=0.8,
        noise_std=0.004 * _S5_METRIC_SCALE,
        surface_near_target_ratio=0.75,
        contact_phi_range=(0.20 * np.pi, 0.34 * np.pi),
        stage2_trace_angle_range=(1.184, 1.376),
        stage2_lateral_center_theta=0.0,
        stage2_lateral_phi_bump_range=(-0.035 * np.pi, 0.035 * np.pi),
        stage4_trace_angle_range=(0.52, 0.66),
        stage1_target_speed_ratio=0.68,
        stage1_speed_taper_fraction=1.0,
        stage1_speed_taper_end_ratio=0.78,
        stage2_target_speed_ratio=0.99,
        stage3_target_speed_ratio=0.75,
        stage4_target_speed_ratio=0.99,
        stage5_target_speed_ratio=0.62,
        stage2_speed_valley_depths=(0.07, 0.18, 0.07),
        stage2_speed_valley_centers=(0.30, 0.58, 0.80),
        stage2_speed_valley_widths=(0.018, 0.025, 0.018),
        stage3_speed_jitter_std=0.04,
        stage3_speed_jitter_clip=0.09,
        stage3_speed_jitter_kernel=5,
        stage4_speed_valley_depths=(0.20, 0.10),
        stage4_speed_valley_centers=(0.35, 0.72),
        stage4_speed_valley_widths=(0.055, 0.045),
        stage2_normal_error_policy="random_control_points_quantile_matched",
        stage2_normal_error_control_point_count_range=(6, 10),
        stage2_normal_error_depth_scale_std=0.05,
        stage2_normal_error_bias_std=0.004,
        stage2_noise_scale=0.28,
        stage4_noise_scale=0.24,
        stage4_tool_normal_max_error=0.30,
        stage5_tool_normal_max_error=0.18,
        trajectory_noise_kernel=9,
        segment_count_slack=0.35,
        stage3_shell_blend_range=(0.44, 0.58),
        stage345_top_phi_range=(0.10 * np.pi, 0.18 * np.pi),
        stage345_top_theta_pull=0.45,
        stage345_top_theta_jitter=0.10 * np.pi,
        rollout_backend="analytic",
        observation_backend=None,
        pybullet_sim_dt=1.0 / 120.0,
        pybullet_steps_per_sample=None,
        pybullet_gravity_z=0.0,
        pybullet_solver_iterations=80,
        pybullet_world_scale=1.0,
        pybullet_world_center=(0.55, 0.0, 0.52),
        pybullet_ur5_urdf_path=None,
        pybullet_ur5_base_xyz=(0.0, 0.0, 0.0),
        pybullet_ur5_base_rpy=(0.0, 0.0, 0.0),
        pybullet_ur5_ee_link_index=-1,
        pybullet_ur5_tool_axis="-x",
        pybullet_ur5_tip_offset=0.0,
        pybullet_ur5_home_q=(0.0, -1.25, 1.85, -2.10, -1.57, 0.0),
        pybullet_ur5_ik_iterations=120,
        pybullet_ur5_ik_damping=0.02,
        pybullet_ur5_rest_home_blend=0.03,
        pybullet_ur5_axis_error_weight=0.02,
        pybullet_ur5_stage1_axis_error_weight=None,
        pybullet_ur5_stage1_axis_weight_ramp_points=5,
        pybullet_ur5_ik_position_error_fallback_threshold=0.0005,
        pybullet_ur5_ik_fallback_axis_error_weight=0.005,
        pybullet_filter_ik_valid=True,
        pybullet_filter_max_attempts=80,
        pybullet_filter_max_position_error=0.012 * _S5_METRIC_SCALE,
        pybullet_filter_max_axis_error=0.30,
        pybullet_filter_global_axis_error=False,
        pybullet_filter_constrained_max_axis_error=0.45,
        pybullet_filter_max_speed_ratio=1.25,
        pybullet_precheck_ik_waypoints=True,
        pybullet_precheck_points_per_stage=3,
        pybullet_suppress_urdf_warnings=True,
        pybullet_ur5_position_gain=0.08,
        pybullet_ur5_velocity_gain=1.0,
        pybullet_ur5_max_force=500.0,
        pybullet_ur5_settle_steps=None,
        pybullet_contact_surface_tol=None,
        pybullet_sphere_collision=False,
        goal_dist_mode="demo_goal",
        eval_tag="S5SphereInspect",
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
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.surface_near_target_ratio = float(surface_near_target_ratio)
        phi_lo, phi_hi = contact_phi_range
        self.contact_phi_range = (float(phi_lo), float(phi_hi))
        angle_lo, angle_hi = stage2_trace_angle_range
        self.stage2_trace_angle_range = (float(angle_lo), float(angle_hi))
        self.stage2_lateral_center_theta = float(stage2_lateral_center_theta)
        bump_lo, bump_hi = stage2_lateral_phi_bump_range
        self.stage2_lateral_phi_bump_range = (float(bump_lo), float(bump_hi))
        self.stage4_trace_angle_range = tuple(float(x) for x in stage4_trace_angle_range)
        self.stage1_target_speed_ratio = float(stage1_target_speed_ratio)
        self.stage1_speed_taper_fraction = float(stage1_speed_taper_fraction)
        self.stage1_speed_taper_end_ratio = (
            None if stage1_speed_taper_end_ratio is None else float(stage1_speed_taper_end_ratio)
        )
        self.stage2_target_speed_ratio = float(stage2_target_speed_ratio)
        self.stage3_target_speed_ratio = float(stage3_target_speed_ratio)
        self.stage4_target_speed_ratio = float(stage4_target_speed_ratio)
        self.stage5_target_speed_ratio = float(stage5_target_speed_ratio)
        self.stage2_speed_valley_depths = tuple(float(x) for x in np.asarray(stage2_speed_valley_depths, dtype=float).reshape(-1))
        self.stage2_speed_valley_centers = tuple(float(x) for x in np.asarray(stage2_speed_valley_centers, dtype=float).reshape(-1))
        self.stage2_speed_valley_widths = tuple(float(x) for x in np.asarray(stage2_speed_valley_widths, dtype=float).reshape(-1))
        self.stage3_speed_jitter_std = float(stage3_speed_jitter_std)
        self.stage3_speed_jitter_clip = float(stage3_speed_jitter_clip)
        self.stage3_speed_jitter_kernel = int(max(int(stage3_speed_jitter_kernel), 1))
        self.stage4_speed_valley_depths = tuple(float(x) for x in np.asarray(stage4_speed_valley_depths, dtype=float).reshape(-1))
        self.stage4_speed_valley_centers = tuple(float(x) for x in np.asarray(stage4_speed_valley_centers, dtype=float).reshape(-1))
        self.stage4_speed_valley_widths = tuple(float(x) for x in np.asarray(stage4_speed_valley_widths, dtype=float).reshape(-1))
        self.stage2_normal_error_policy = str(stage2_normal_error_policy).strip().lower()
        if self.stage2_normal_error_policy not in {
            "fixed_periodic_v20",
            "periodic_quantile_matched_v21",
            "random_control_points_quantile_matched",
        }:
            raise ValueError(
                "Unsupported stage2_normal_error_policy "
                f"'{stage2_normal_error_policy}'."
            )
        control_count_lo, control_count_hi = stage2_normal_error_control_point_count_range
        self.stage2_normal_error_control_point_count_range = (
            max(int(control_count_lo), 3),
            max(int(control_count_hi), max(int(control_count_lo), 3)),
        )
        self.stage2_normal_error_depth_scale_std = float(max(stage2_normal_error_depth_scale_std, 0.0))
        self.stage2_normal_error_bias_std = float(max(stage2_normal_error_bias_std, 0.0))
        self.stage2_noise_scale = float(stage2_noise_scale)
        self.stage4_noise_scale = float(stage4_noise_scale)
        self.stage4_tool_normal_max_error = float(stage4_tool_normal_max_error)
        self.stage5_tool_normal_max_error = float(stage5_tool_normal_max_error)
        self.trajectory_noise_kernel = int(max(int(trajectory_noise_kernel), 1))
        self.segment_count_slack = float(segment_count_slack)
        shell_blend_lo, shell_blend_hi = stage3_shell_blend_range
        self.stage3_shell_blend_range = (float(shell_blend_lo), float(shell_blend_hi))
        top_phi_lo, top_phi_hi = stage345_top_phi_range
        self.stage345_top_phi_range = (float(top_phi_lo), float(top_phi_hi))
        self.stage345_top_theta_pull = float(stage345_top_theta_pull)
        self.stage345_top_theta_jitter = float(stage345_top_theta_jitter)
        self.rollout_backend = str(rollout_backend).lower()
        requested_observation_backend = self.rollout_backend if observation_backend is None else observation_backend
        self.observation_backend = self._normalize_observation_backend(requested_observation_backend)
        if self.rollout_backend not in {"analytic", "pybullet"}:
            raise ValueError(f"Unsupported S5 rollout_backend '{self.rollout_backend}'.")
        if self.observation_backend not in {"analytic_raw", "pybullet"}:
            raise ValueError(f"Unsupported S5 observation_backend '{self.observation_backend}'.")
        self.pybullet_sim_dt = float(pybullet_sim_dt)
        self.pybullet_steps_per_sample = None if pybullet_steps_per_sample is None else int(pybullet_steps_per_sample)
        self.pybullet_gravity_z = float(pybullet_gravity_z)
        self.pybullet_solver_iterations = int(pybullet_solver_iterations)
        self.pybullet_world_scale = float(pybullet_world_scale)
        self.pybullet_world_center = tuple(float(x) for x in np.asarray(pybullet_world_center, dtype=float).reshape(3))
        self.pybullet_ur5_urdf_path = pybullet_ur5_urdf_path
        self.pybullet_ur5_base_xyz = tuple(float(x) for x in np.asarray(pybullet_ur5_base_xyz, dtype=float).reshape(3))
        self.pybullet_ur5_base_rpy = tuple(float(x) for x in np.asarray(pybullet_ur5_base_rpy, dtype=float).reshape(3))
        self.pybullet_ur5_ee_link_index = int(pybullet_ur5_ee_link_index)
        self.pybullet_ur5_tool_axis = str(pybullet_ur5_tool_axis)
        self.pybullet_ur5_tip_offset = float(pybullet_ur5_tip_offset)
        self.pybullet_ur5_home_q = tuple(float(x) for x in np.asarray(pybullet_ur5_home_q, dtype=float).reshape(6))
        self.pybullet_ur5_ik_iterations = int(pybullet_ur5_ik_iterations)
        self.pybullet_ur5_ik_damping = float(pybullet_ur5_ik_damping)
        self.pybullet_ur5_rest_home_blend = float(pybullet_ur5_rest_home_blend)
        self.pybullet_ur5_axis_error_weight = float(pybullet_ur5_axis_error_weight)
        self.pybullet_ur5_stage1_axis_error_weight = (
            None if pybullet_ur5_stage1_axis_error_weight is None else float(pybullet_ur5_stage1_axis_error_weight)
        )
        self.pybullet_ur5_stage1_axis_weight_ramp_points = int(max(int(pybullet_ur5_stage1_axis_weight_ramp_points), 0))
        self.pybullet_ur5_ik_position_error_fallback_threshold = float(pybullet_ur5_ik_position_error_fallback_threshold)
        self.pybullet_ur5_ik_fallback_axis_error_weight = float(pybullet_ur5_ik_fallback_axis_error_weight)
        self.pybullet_filter_ik_valid = bool(pybullet_filter_ik_valid)
        self.pybullet_filter_max_attempts = int(max(int(pybullet_filter_max_attempts), 1))
        self.pybullet_filter_max_position_error = float(pybullet_filter_max_position_error)
        self.pybullet_filter_max_axis_error = float(pybullet_filter_max_axis_error)
        self.pybullet_filter_global_axis_error = bool(pybullet_filter_global_axis_error)
        self.pybullet_filter_constrained_max_axis_error = float(pybullet_filter_constrained_max_axis_error)
        self.pybullet_filter_max_speed_ratio = float(pybullet_filter_max_speed_ratio)
        self.pybullet_precheck_ik_waypoints = bool(pybullet_precheck_ik_waypoints)
        self.pybullet_precheck_points_per_stage = int(max(int(pybullet_precheck_points_per_stage), 2))
        self.pybullet_suppress_urdf_warnings = bool(pybullet_suppress_urdf_warnings)
        self.pybullet_ur5_position_gain = float(pybullet_ur5_position_gain)
        self.pybullet_ur5_velocity_gain = float(pybullet_ur5_velocity_gain)
        self.pybullet_ur5_max_force = float(pybullet_ur5_max_force)
        self.pybullet_ur5_settle_steps = None if pybullet_ur5_settle_steps is None else int(pybullet_ur5_settle_steps)
        default_contact_tol = 0.025 * self.sphere_radius
        self.pybullet_contact_surface_tol = (
            float(default_contact_tol) if pybullet_contact_surface_tol is None else float(pybullet_contact_surface_tol)
        )
        self.pybullet_sphere_collision = bool(pybullet_sphere_collision)
        self.goal_dist_mode = self._normalize_goal_dist_mode(goal_dist_mode)
        self.eval_tag = str(eval_tag)

        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self._cached_tool_axis_traces = {}
        self._cached_timestamp_traces = {}
        self._cached_goal_positions = {}
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

    def sample_scene(self, seed=None, rng=None):
        return {
            "task_name": str(self.eval_tag),
            "geometry": {
                "sphere_center": self.sphere_center.tolist(),
                "sphere_radius": float(self.sphere_radius),
                "shell_thickness": float(self.shell_thickness),
                "surface_near_target_ratio": float(self.surface_near_target_ratio),
            },
            "task": {
                "generator_mode": "five_stage_lateral_inspection",
                "contact_phi_range": list(self.contact_phi_range),
                "stage2_trace_angle_range": list(self.stage2_trace_angle_range),
                "stage2_lateral_center_theta": float(self.stage2_lateral_center_theta),
                "stage2_lateral_phi_bump_range": list(self.stage2_lateral_phi_bump_range),
                "stage3_shell_blend_range": list(self.stage3_shell_blend_range),
                "stage345_top_phi_range": list(self.stage345_top_phi_range),
                "stage345_top_theta_pull": float(self.stage345_top_theta_pull),
                "stage345_top_theta_jitter": float(self.stage345_top_theta_jitter),
                "stage4_trace_angle_range": list(self.stage4_trace_angle_range),
                "stage2_deliberate_slowdowns": {
                    "kind": "gaussian_speed_intent_events",
                    "depths": list(self.stage2_speed_valley_depths),
                    "centers": list(self.stage2_speed_valley_centers),
                    "widths": list(self.stage2_speed_valley_widths),
                },
                "stage3_speed_jitter": {
                    "std": float(self.stage3_speed_jitter_std),
                    "clip": float(self.stage3_speed_jitter_clip),
                    "kernel": int(self.stage3_speed_jitter_kernel),
                },
                "stage4_deliberate_slowdowns": {
                    "kind": "gaussian_speed_intent_events",
                    "depths": list(self.stage4_speed_valley_depths),
                    "centers": list(self.stage4_speed_valley_centers),
                    "widths": list(self.stage4_speed_valley_widths),
                },
                "tool_axis": {
                    "stage2_normal_error_policy": str(self.stage2_normal_error_policy),
                    "stage2_control_point_count_range": list(
                        self.stage2_normal_error_control_point_count_range
                    ),
                    "stage2_depth_scale_std": float(self.stage2_normal_error_depth_scale_std),
                    "stage2_bias_std": float(self.stage2_normal_error_bias_std),
                    "stage4_normal_max_error": float(self.stage4_tool_normal_max_error),
                    "stage5_normal_max_error": float(self.stage5_tool_normal_max_error),
                },
                "trajectory_noise": {
                    "noise_std": float(self.noise_std),
                    "kernel": int(self.trajectory_noise_kernel),
                    "stage2_scale": float(self.stage2_noise_scale),
                    "stage4_scale": float(self.stage4_noise_scale),
                },
            },
        }

    def _rollout_demo_analytic(self, scene, seed=None, rng=None, **kwargs):
        if scene is not None and "demo_index" in scene and "demo_index" not in kwargs:
            kwargs["demo_index"] = int(scene["demo_index"])
        if rng is not None:
            traj, cutpoints, generation_metadata = self.generate_demo(
                rng=rng,
                return_metadata=True,
                **kwargs,
            )
            generation_metadata["rollout_seed"] = None
        else:
            local_seed = int(seed) if seed is not None else int((scene or {}).get("rollout_seed", 0))
            local_rng = np.random.RandomState(local_seed)
            traj, cutpoints, generation_metadata = self.generate_demo(
                rng=local_rng,
                return_metadata=True,
                **kwargs,
            )
            generation_metadata["rollout_seed"] = int(local_seed)
        tool_axis = self._lookup_cached_tool_axis_trace(traj)
        timestamps = self._lookup_cached_timestamp_trace(traj)
        if timestamps is None:
            timestamps = np.arange(len(traj), dtype=float) * float(self.dt)
        return {
            "trajectory": np.asarray(traj, dtype=float),
            "timestamps": np.asarray(timestamps, dtype=float),
            "true_cutpoints": np.asarray(cutpoints, dtype=int),
            "tool_axis": None if tool_axis is None else np.asarray(tool_axis, dtype=float),
            "generation_metadata": generation_metadata,
            "rollout_backend": "analytic",
            "observation_backend": str(self.observation_backend),
        }
