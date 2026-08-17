from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np


S5_METRIC_SCALE = 0.18
_LOADER_DEFAULT_FIELDS = (
    "seg_lengths",
    "seg_length_jitter",
    "sphere_radius",
    "shell_thickness",
    "approach_offset",
    "depart_offset",
    "surface_near_target_ratio",
    "split_stage3_transition",
    "contact_theta_range",
    "contact_phi_range",
    "stage1_speed_max",
    "stage2_speed_max",
    "stage3_speed_max",
    "stage4_speed_max",
    "stage1_accel_max",
    "stage2_accel_max",
    "stage3_accel_max",
    "stage4_accel_max",
    "stage2_trace_angle_range",
    "stage2_robot_lateral_trace",
    "stage2_lateral_center_theta",
    "stage2_lateral_phi_bump_range",
    "stage3_shell_blend_range",
    "stage345_top_phi_range",
    "stage345_top_theta_pull",
    "stage345_top_theta_jitter",
    "stage2_surface_detour_angle",
    "stage4_shell_detour_angle",
    "stage2_length_scale_range",
    "stage4_length_scale_range",
    "stage1_speed_taper_fraction",
    "stage1_speed_taper_end_ratio",
    "stage2_target_speed_ratio",
    "stage3_target_speed_ratio",
    "stage4_target_speed_ratio",
    "stage2_speed_valley_depths",
    "stage2_speed_valley_centers",
    "stage2_speed_valley_widths",
    "stage3_speed_jitter_std",
    "stage3_speed_jitter_clip",
    "stage3_speed_jitter_kernel",
    "stage4_speed_valley_depth",
    "stage4_speed_valley_center",
    "stage4_speed_valley_width",
    "noise_std",
    "stage2_noise_scale",
    "stage4_noise_scale",
    "trajectory_noise_kernel",
    "pybullet_world_scale",
    "pybullet_filter_max_position_error",
)
_V20_COMPAT_DEFAULTS = {
    "split_stage3_transition": True,
    "stage2_robot_lateral_trace": True,
    "stage2_surface_detour_angle": 0.0,
    "stage2_length_scale_range": (1.0, 1.0),
    "stage4_length_scale_range": (1.0, 1.0),
    "contact_theta_range": (-0.12 * np.pi, 0.16 * np.pi),
}


@dataclass(frozen=True)
class S5GeometryConfig:
    sphere_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    seg_lengths: tuple[int, int, int, int] = (18, 34, 24, 18)
    seg_length_jitter: tuple[int, int, int, int] = (3, 5, 5, 3)
    sphere_radius: float = 1.0 * S5_METRIC_SCALE
    shell_thickness: float = 0.24 * S5_METRIC_SCALE
    approach_offset: float = 0.42 * S5_METRIC_SCALE
    depart_offset: float = 0.50 * S5_METRIC_SCALE
    surface_near_target_ratio: float = 0.75
    contact_phi_range: tuple[float, float] = (0.20 * np.pi, 0.34 * np.pi)


@dataclass(frozen=True)
class S5PathConfig:
    stage2_trace_angle_range: tuple[float, float] = (1.184, 1.376)
    stage2_lateral_center_theta: float = 0.0
    stage2_lateral_phi_bump_range: tuple[float, float] = (-0.035 * np.pi, 0.035 * np.pi)
    stage3_shell_blend_range: tuple[float, float] = (0.44, 0.58)
    stage345_top_phi_range: tuple[float, float] = (0.10 * np.pi, 0.18 * np.pi)
    stage345_top_theta_pull: float = 0.45
    stage345_top_theta_jitter: float = 0.10 * np.pi
    stage4_shell_detour_angle: float = 0.10


@dataclass(frozen=True)
class S5TimingConfig:
    dt: float = 0.8
    stage1_speed_max: float = 0.12 * S5_METRIC_SCALE
    stage2_speed_max: float = 0.047 * S5_METRIC_SCALE
    stage3_speed_max: float = 0.060 * S5_METRIC_SCALE
    stage4_speed_max: float = 0.09 * S5_METRIC_SCALE
    stage1_accel_max: float = 0.08 * S5_METRIC_SCALE
    stage2_accel_max: float = 0.03 * S5_METRIC_SCALE
    stage3_accel_max: float = 0.07 * S5_METRIC_SCALE
    stage4_accel_max: float = 0.06 * S5_METRIC_SCALE
    stage1_target_speed_ratio: float = 0.68
    stage1_speed_taper_fraction: float = 1.0
    stage1_speed_taper_end_ratio: float = 0.78
    stage2_target_speed_ratio: float = 0.99
    stage3_target_speed_ratio: float = 0.75
    stage4_target_speed_ratio: float = 0.99
    stage5_target_speed_ratio: float = 0.62
    stage2_speed_valley_depths: tuple[float, float, float] = (0.07, 0.18, 0.07)
    stage2_speed_valley_centers: tuple[float, float, float] = (0.30, 0.58, 0.80)
    stage2_speed_valley_widths: tuple[float, float, float] = (0.018, 0.025, 0.018)
    stage3_speed_jitter_std: float = 0.04
    stage3_speed_jitter_clip: float = 0.09
    stage3_speed_jitter_kernel: int = 5
    stage4_speed_valley_depth: float = 0.08
    stage4_speed_valley_center: float = 0.54
    stage4_speed_valley_width: float = 0.025
    segment_count_slack: float = 0.35


@dataclass(frozen=True)
class S5OrientationConfig:
    tool_align_max_stage2: float = 0.04
    stage2_normal_error_policy: str = "random_control_points_quantile_matched"
    stage2_normal_error_control_point_count_range: tuple[int, int] = (6, 10)
    stage2_normal_error_depth_scale_std: float = 0.05
    stage2_normal_error_bias_std: float = 0.004
    stage4_tool_normal_max_error: float = 0.30
    stage5_tool_normal_max_error: float = 0.18


@dataclass(frozen=True)
class S5NoiseConfig:
    noise_std: float = 0.004 * S5_METRIC_SCALE
    stage2_noise_scale: float = 0.28
    stage4_noise_scale: float = 0.24
    trajectory_noise_kernel: int = 9


@dataclass(frozen=True)
class S5ExecutionConfig:
    rollout_backend: str = "pybullet"
    observation_backend: str = "pybullet"
    pybullet_sim_dt: float = 1.0 / 120.0
    pybullet_steps_per_sample: Any = None
    pybullet_gravity_z: float = 0.0
    pybullet_solver_iterations: int = 80
    pybullet_world_scale: float = 1.0
    pybullet_world_center: tuple[float, float, float] = (0.55, 0.0, 0.52)
    pybullet_ur5_urdf_path: Any = None
    pybullet_ur5_base_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pybullet_ur5_base_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pybullet_ur5_ee_link_index: int = -1
    pybullet_ur5_tool_axis: str = "-x"
    pybullet_ur5_tip_offset: float = 0.0
    pybullet_ur5_home_q: tuple[float, float, float, float, float, float] = (0.0, -1.25, 1.85, -2.10, -1.57, 0.0)
    pybullet_ur5_ik_iterations: int = 120
    pybullet_ur5_ik_damping: float = 0.02
    pybullet_ur5_rest_home_blend: float = 0.03
    pybullet_ur5_axis_error_weight: float = 0.02
    pybullet_ur5_stage1_axis_error_weight: Any = None
    pybullet_ur5_stage1_axis_weight_ramp_points: int = 5
    pybullet_ur5_ik_position_error_fallback_threshold: float = 0.0005
    pybullet_ur5_ik_fallback_axis_error_weight: float = 0.005
    pybullet_filter_ik_valid: bool = True
    pybullet_filter_max_attempts: int = 80
    pybullet_filter_max_position_error: float = 0.012 * S5_METRIC_SCALE
    pybullet_filter_max_axis_error: float = 0.30
    pybullet_filter_global_axis_error: bool = False
    pybullet_filter_constrained_max_axis_error: float = 0.45
    pybullet_filter_max_speed_ratio: float = 1.25
    pybullet_precheck_ik_waypoints: bool = True
    pybullet_precheck_points_per_stage: int = 3
    pybullet_suppress_urdf_warnings: bool = True
    pybullet_ur5_position_gain: float = 0.08
    pybullet_ur5_velocity_gain: float = 1.0
    pybullet_ur5_max_force: float = 500.0
    pybullet_ur5_settle_steps: Any = None
    pybullet_contact_surface_tol: Any = None
    pybullet_sphere_collision: bool = False


@dataclass(frozen=True)
class S5DatasetConfig:
    cache_demos: bool = True
    goal_dist_mode: str = "demo_goal"
    eval_tag: str = "S5SphereInspect"


@dataclass(frozen=True)
class S5SyntheticPreset:
    name: str
    version: int
    geometry: S5GeometryConfig = field(default_factory=S5GeometryConfig)
    path: S5PathConfig = field(default_factory=S5PathConfig)
    timing: S5TimingConfig = field(default_factory=S5TimingConfig)
    orientation: S5OrientationConfig = field(default_factory=S5OrientationConfig)
    noise: S5NoiseConfig = field(default_factory=S5NoiseConfig)
    execution: S5ExecutionConfig = field(default_factory=S5ExecutionConfig)
    dataset: S5DatasetConfig = field(default_factory=S5DatasetConfig)

    def to_env_kwargs(self) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for group in (
            self.geometry,
            self.path,
            self.timing,
            self.orientation,
            self.noise,
            self.execution,
            self.dataset,
        ):
            defaults.update(asdict(group))
        return defaults

    def apply(self, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        resolved = self.to_env_kwargs()
        resolved.update(dict(overrides or {}))
        return resolved

    def historical_loader_defaults(self) -> dict[str, Any]:
        resolved = self.to_env_kwargs()
        resolved.update(_V20_COMPAT_DEFAULTS)
        return {key: resolved[key] for key in _LOADER_DEFAULT_FIELDS}


S5_SYNTHETIC_V20 = S5SyntheticPreset(
    name="s5_synthetic_v20",
    version=20,
    orientation=replace(
        S5OrientationConfig(),
        stage2_normal_error_policy="fixed_periodic_v20",
    ),
)
S5_SYNTHETIC_V21 = S5SyntheticPreset(
    name="s5_synthetic_v21",
    version=21,
    orientation=replace(
        S5OrientationConfig(),
        stage2_normal_error_policy="periodic_quantile_matched_v21",
    ),
)
S5_SYNTHETIC_V22 = S5SyntheticPreset(name="s5_synthetic_v22", version=22)
S5_SYNTHETIC_V23 = S5SyntheticPreset(name="s5_synthetic_v23", version=23)


def cache_compatible_s5_loader_config(env_cfg: dict[str, Any] | None) -> dict[str, Any]:
    resolved = dict(env_cfg or {})
    for key, value in S5_SYNTHETIC_V20.historical_loader_defaults().items():
        resolved.setdefault(key, value)
    return resolved


def active_s5_env_kwargs(env_cfg: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(env_cfg)
    for key, expected in _V20_COMPAT_DEFAULTS.items():
        value = resolved.pop(key, expected)
        if value != expected:
            raise ValueError(
                f"S5 v20 formal generator no longer supports '{key}={value}'. "
                f"The only supported compatibility value is {expected}."
            )
    return resolved


def apply_default_s5_loader_config(env_cfg: dict[str, Any] | None) -> dict[str, Any]:
    return active_s5_env_kwargs(cache_compatible_s5_loader_config(env_cfg))
