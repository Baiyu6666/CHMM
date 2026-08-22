from __future__ import annotations

import numpy as np

from .BarInspect import BarInspectEnv, BarInspectScene
from .base import TaskBundle


class BarCleanEnv(BarInspectEnv):
    """Five-stage bar-cleaning task with a transverse debris-discharge stroke.

    Stages:

    0. approach the bar while keeping obstacle clearance;
    1. clean along the bar using the same constrained motion as BarInspect stage 2;
    2. move freely from the cleaning endpoint to the discharge start pose;
    3. discharge debris along the bar-lateral direction while holding axial position;
    4. leave the bar without a task constraint.

    The bar axis is treated as task-relative x and the bar-lateral axis as
    task-relative y.  Therefore stage 4 constrains ``bar_axial_offset`` while
    allowing the trajectory to progress along y.
    """

    task_name = "BarClean"
    default_learning_features = (
        "obstacle_clearance",
        "surface_dist",
        "bar_lateral_offset",
        "tool_pitch",
        "tool_plane_err",
        "bar_axial_offset",
    )

    def __init__(
        self,
        *args,
        scan_start_progress=-0.18,
        stage2_scan_distance=0.15,
        transition_axial_advance=0.0,
        discharge_start_lateral=0.07,
        discharge_distance=0.26,
        discharge_axial_noise_std=0.0008,
        discharge_height_variation=0.025,
        seg_lengths=(38, 32, 24, 28, 28),
        seg_length_jitter=(5, 4, 4, 4, 5),
        **kwargs,
    ):
        requested_lengths = tuple(int(value) for value in seg_lengths)
        requested_jitters = tuple(int(value) for value in seg_length_jitter)
        if len(requested_lengths) != 5 or len(requested_jitters) != 5:
            raise ValueError("BarClean requires five segment lengths and five jitters.")
        if min(requested_lengths) < 4 or min(requested_jitters) < 0:
            raise ValueError("BarClean segment lengths must be >= 4 and jitters non-negative.")

        self.transition_axial_advance = float(transition_axial_advance)
        self.discharge_start_lateral = float(discharge_start_lateral)
        self.discharge_distance = float(discharge_distance)
        self.discharge_axial_noise_std = float(discharge_axial_noise_std)
        self.discharge_height_variation = float(discharge_height_variation)
        self.discharge_axial_progress = (
            float(scan_start_progress)
            + float(stage2_scan_distance)
            + self.transition_axial_advance
        )
        if self.discharge_distance <= 0.0:
            raise ValueError("discharge_distance must be positive.")
        if self.discharge_axial_noise_std < 0.0 or self.discharge_height_variation < 0.0:
            raise ValueError(
                "discharge_axial_noise_std and discharge_height_variation must be non-negative."
            )

        super().__init__(
            *args,
            scan_start_progress=scan_start_progress,
            stage2_scan_distance=stage2_scan_distance,
            seg_lengths=(
                requested_lengths[0],
                requested_lengths[1],
                requested_lengths[2],
                requested_lengths[4],
            ),
            seg_length_jitter=(
                requested_jitters[0],
                requested_jitters[1],
                requested_jitters[2],
                requested_jitters[4],
            ),
            **kwargs,
        )
        self.seg_lengths = requested_lengths
        self.seg_length_jitter = requested_jitters
        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self.stage_specs = self.get_stage_specs()

    def get_feature_schema(self):
        schema = [dict(spec) for spec in super().get_feature_schema()]
        schema.append(
            {
                "id": 8,
                "column_idx": 8,
                "name": "bar_axial_offset",
                "unit": "m",
                "description": (
                    "Signed bar-axis displacement from the transverse-discharge line; "
                    "task-relative x error"
                ),
            }
        )
        return schema

    def get_true_constraints(self):
        constraints = dict(super().get_true_constraints())
        constraints.pop("stage3_pitch_target", None)
        constraints["discharge_axial_target"] = 0.0
        constraints["discharge_axial_progress"] = float(self.discharge_axial_progress)
        return constraints

    def get_constraint_specs(self):
        return [
            {
                "feature_name": "obstacle_clearance",
                "stage": 0,
                "semantics": "lower_bound",
                "oracle_key": "obstacle_min_clearance",
            },
            {
                "feature_name": "surface_dist",
                "stage": 1,
                "semantics": "target_value",
                "oracle_key": "surface_distance_target",
            },
            {
                "feature_name": "bar_lateral_offset",
                "stage": 1,
                "semantics": "target_value",
                "oracle_key": "bar_lateral_target",
            },
            {
                "feature_name": "tool_pitch",
                "stage": 1,
                "semantics": "target_value",
                "oracle_key": "stage2_pitch_target",
            },
            {
                "feature_name": "tool_plane_err",
                "stage": 1,
                "semantics": "target_value",
                "oracle_key": "tool_plane_target",
            },
            {
                "feature_name": "bar_axial_offset",
                "stage": 3,
                "semantics": "target_value",
                "oracle_key": "discharge_axial_target",
            },
        ]

    def get_stage_specs(self):
        return [
            {
                "stage": 0,
                "name": "approach",
                "description": "Approach the bar and avoid the obstacle",
            },
            {
                "stage": 1,
                "name": "longitudinal_clean",
                "distance_m": float(self.stage2_scan_distance),
                "pitch_deg": float(np.rad2deg(self.stage2_pitch)),
            },
            {
                "stage": 2,
                "name": "free_reposition",
                "description": "Move freely to the transverse-discharge start pose",
            },
            {
                "stage": 3,
                "name": "transverse_discharge",
                "distance_m": float(self.discharge_distance),
                "fixed_axial_progress_m": float(self.discharge_axial_progress),
            },
            {
                "stage": 4,
                "name": "free_depart",
                "description": "Leave with no task constraint",
            },
        ]

    def sample_scene(self, seed=None, rng=None):
        scene = super().sample_scene(seed=seed, rng=rng)
        scene["task"] = {
            "stage_specs": self.get_stage_specs(),
            "clean_standoff": float(self.scan_standoff),
            "clean_lateral_offset": float(self.scan_lateral_offset),
            "discharge_axial_progress": float(self.discharge_axial_progress),
            "discharge_start_lateral": float(self.discharge_start_lateral),
            "discharge_distance": float(self.discharge_distance),
            "dt": float(self.dt),
        }
        return scene

    def compute_all_features_matrix(self, traj, feat_ids=None, scene=None):
        trajectory = np.asarray(traj, dtype=float)
        base_features = super().compute_all_features_matrix(
            trajectory,
            feat_ids=None,
            scene=scene,
        )
        quaternion = self._quat_normalize(trajectory[:, 3:7])
        rotations = self._quat_to_matrix(quaternion)
        tcp = trajectory[:, :3] + np.einsum(
            "tij,j->ti", rotations, self.tcp_offset_local
        )
        bar_reference, bar_axis, _ = self._bar_geometry_trace(
            trajectory,
            scene=scene,
        )
        bar_progress = np.sum((tcp - bar_reference) * bar_axis, axis=1)
        bar_axial_offset = bar_progress - self.discharge_axial_progress
        features = np.column_stack([base_features, bar_axial_offset])
        if not np.all(np.isfinite(features)):
            raise ValueError("BarClean feature extraction produced non-finite values.")
        if feat_ids is None:
            return features
        if len(feat_ids) > 0 and isinstance(feat_ids[0], str):
            name_to_column = {
                spec["name"]: spec["column_idx"] for spec in self.feature_schema
            }
            feat_ids = [name_to_column[name] for name in feat_ids]
        return features[:, feat_ids]

    def generate_demo(self, seed=0, rng=None):
        local_rng = np.random.RandomState(int(seed)) if rng is None else rng
        lengths = [
            max(4, base + local_rng.randint(-jitter, jitter + 1))
            for base, jitter in zip(self.seg_lengths, self.seg_length_jitter)
        ]
        n1, n2, n3, n4, n5 = lengths

        clean_entry = self._scan_tcp([self.scan_start_progress])[0]
        start = self.nominal_start_tcp + local_rng.normal(
            scale=np.array([0.025, 0.035, 0.025], dtype=float), size=3
        )
        obstacle_to_line = 0.5 * (start + clean_entry) - self.obstacle_center
        lateral_sign = np.sign(float(np.dot(obstacle_to_line, self.bar_lateral)))
        detour_direction = (
            -1.0 if lateral_sign == 0.0 else lateral_sign
        ) * self.bar_lateral
        detour = self.obstacle_center + (
            self.obstacle_radius + self.obstacle_min_clearance + 0.045
        ) * detour_direction
        detour += self.table_normal * float(
            np.dot(0.5 * (start + clean_entry) - detour, self.table_normal)
        )
        detour += local_rng.normal(scale=0.012, size=3)
        approach_tcp = self._resample_polyline([start, detour, clean_entry], n1)
        approach_tcp += self._smooth_noise(
            local_rng, n1, 3, self.position_noise_std * 2.0, knots=7
        )
        approach_tcp = self._project_outside_obstacle(approach_tcp)
        approach_tcp[0] = start
        approach_tcp[-1] = clean_entry

        clean_u = np.arange(1, n2 + 1, dtype=float) / float(n2)
        clean_lateral = self.scan_lateral_offset + self._smooth_noise(
            local_rng, n2, 1, self.position_noise_std, knots=6
        )[:, 0]
        clean_height = self.scan_standoff + self._smooth_noise(
            local_rng, n2, 1, self.position_noise_std * 0.65, knots=6
        )[:, 0]
        clean_tcp = self._scan_tcp(
            self.scan_start_progress + self.stage2_scan_distance * clean_u,
            clean_lateral,
            clean_height,
        )

        clean_pitch = self.stage2_pitch + self._smooth_noise(
            local_rng, n2, 1, self.pitch_noise, knots=6
        )[:, 0]
        clean_plane = self._smooth_noise(
            local_rng, n2, 1, self.plane_noise, knots=6
        )[:, 0]
        roll_phase = local_rng.uniform(-np.pi, np.pi)
        clean_roll = local_rng.uniform(-np.pi, np.pi) + self.roll_variation * np.sin(
            2.0 * np.pi * clean_u + roll_phase
        )
        clean_quat = np.asarray(
            [
                self._matrix_to_quat(self._tool_rotation(pitch, plane, roll))
                for pitch, plane, roll in zip(clean_pitch, clean_plane, clean_roll)
            ],
            dtype=float,
        )
        start_quat = self._quat_normalize(local_rng.normal(size=4))
        approach_quat = self._quat_slerp(
            start_quat,
            clean_quat[0],
            self._smoothstep(np.linspace(0.0, 1.0, n1)),
        )

        discharge_start_height = self.scan_standoff + local_rng.uniform(
            -self.discharge_height_variation,
            self.discharge_height_variation,
        )
        discharge_end_height = self.scan_standoff + local_rng.uniform(
            -self.discharge_height_variation,
            self.discharge_height_variation,
        )
        discharge_start = self._scan_tcp(
            [self.discharge_axial_progress],
            lateral=[self.discharge_start_lateral],
            standoff=[discharge_start_height],
        )[0]
        transition_u = np.arange(1, n3 + 1, dtype=float) / float(n3)
        transition_curve = self._smoothstep(transition_u)
        transition_tcp = (
            (1.0 - transition_curve[:, None]) * clean_tcp[-1][None, :]
            + transition_curve[:, None] * discharge_start[None, :]
        )
        transition_tcp += self._smooth_noise(
            local_rng, n3, 3, self.position_noise_std * 2.0, knots=5
        )
        transition_tcp[-1] = discharge_start

        discharge_start_quat = self._quat_normalize(local_rng.normal(size=4))
        transition_quat = self._quat_slerp(
            clean_quat[-1],
            discharge_start_quat,
            transition_curve,
        )

        discharge_u = np.arange(1, n4 + 1, dtype=float) / float(n4)
        discharge_progress = self.discharge_axial_progress + self._smooth_noise(
            local_rng,
            n4,
            1,
            self.discharge_axial_noise_std,
            knots=6,
        )[:, 0]
        discharge_lateral = (
            self.discharge_start_lateral - self.discharge_distance * discharge_u
        )
        discharge_lateral += self._smooth_noise(
            local_rng, n4, 1, self.position_noise_std, knots=6
        )[:, 0]
        discharge_height = (
            (1.0 - discharge_u) * discharge_start_height
            + discharge_u * discharge_end_height
        )
        discharge_height += self._smooth_noise(
            local_rng, n4, 1, self.position_noise_std * 1.5, knots=6
        )[:, 0]
        discharge_tcp = self._scan_tcp(
            discharge_progress,
            lateral=discharge_lateral,
            standoff=discharge_height,
        )
        discharge_end_quat = self._quat_normalize(local_rng.normal(size=4))
        discharge_quat = self._quat_slerp(
            discharge_start_quat,
            discharge_end_quat,
            self._smoothstep(discharge_u),
        )

        depart_direction = self._unit(
            local_rng.uniform(-0.45, 0.45) * self.bar_axis
            + local_rng.choice([-1.0, 1.0])
            * local_rng.uniform(0.55, 1.0)
            * self.bar_lateral
            + local_rng.uniform(0.45, 0.95) * self.table_normal,
            "departure direction",
        )
        depart_distance = local_rng.uniform(0.18, 0.28)
        depart_u = np.arange(1, n5 + 1, dtype=float) / float(n5)
        depart_curve = self._smoothstep(depart_u)
        depart_tcp = discharge_tcp[-1][None, :] + (
            depart_distance * depart_curve[:, None] * depart_direction[None, :]
        )
        depart_tcp += (
            0.025
            * np.sin(np.pi * depart_u)[:, None]
            * local_rng.uniform(-1.0, 1.0)
            * self.bar_axis[None, :]
        )
        end_quat = self._quat_normalize(local_rng.normal(size=4))
        depart_quat = self._quat_slerp(
            discharge_quat[-1],
            end_quat,
            depart_curve,
        )

        trajectory = np.vstack(
            [
                self._pose_from_tcp_and_quaternion(approach_tcp, approach_quat),
                self._pose_from_tcp_and_quaternion(clean_tcp, clean_quat),
                self._pose_from_tcp_and_quaternion(transition_tcp, transition_quat),
                self._pose_from_tcp_and_quaternion(discharge_tcp, discharge_quat),
                self._pose_from_tcp_and_quaternion(depart_tcp, depart_quat),
            ]
        )
        labels = np.concatenate(
            [np.full(length, stage, dtype=int) for stage, length in enumerate(lengths)]
        )
        cutpoints = np.where(np.diff(labels) != 0)[0].astype(int)
        return trajectory, labels, cutpoints


def load_BarClean(
    n_demos=10,
    seed=2026,
    env_kwargs=None,
    demo_kwargs=None,
    **extra_env_kwargs,
):
    env_config = dict(env_kwargs or {})
    env_config.update(extra_env_kwargs)
    run_config = dict(demo_kwargs or {})
    env = BarCleanEnv(**env_config)

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


__all__ = ["BarCleanEnv", "BarInspectScene", "load_BarClean"]
