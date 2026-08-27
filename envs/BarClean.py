from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .BarInspect import BarInspectEnv, BarInspectScene
from .base import TaskBundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASK_DEFINITION = (
    PROJECT_ROOT
    / "robot"
    / "stage_cons_iiwa14"
    / "ros_ws"
    / "src"
    / "stage_constraint_planner"
    / "config"
    / "bar_clean_true.json"
)
PLANNER_PYTHON_ROOT = DEFAULT_TASK_DEFINITION.parent.parent / "src"
if str(PLANNER_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANNER_PYTHON_ROOT))

from stage_constraint_planner.optimizer import (  # noqa: E402
    BarFeatureEvaluator,
    tool_yaw_from_quaternion,
)


def _load_task_definition(path):
    resolved = Path(path or DEFAULT_TASK_DEFINITION).expanduser().resolve()
    definition = json.loads(resolved.read_text(encoding="utf-8"))
    if definition.get("task_id") != "BarClean":
        raise ValueError(f"{resolved} is not a BarClean task definition.")
    return resolved, definition


def _required_term(definition, stage, feature_name):
    matches = [
        dict(term)
        for term in definition["constraint_terms"]
        if int(term["stage"]) == int(stage)
        and str(term["feature_name"]) == str(feature_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"BarClean requires exactly one stage {stage} {feature_name} constraint."
        )
    return matches[0]


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
        "table_dist",
        "bar_lateral_offset",
        "tool_pitch",
        "tool_roll",
        "tool_yaw",
        "bar_axial_offset",
    )

    def __init__(
        self,
        *args,
        task_definition_path=None,
        discharge_axial_noise_std=0.0008,
        discharge_height_variation=0.003,
        obstacle_center=(-0.285, -0.090, 0.13437),
        seg_lengths=(38, 32, 24, 28, 28),
        seg_length_jitter=(5, 4, 4, 4, 5),
        **kwargs,
    ):
        self.task_definition_path, self.task_definition = _load_task_definition(
            task_definition_path
        )
        definition = self.task_definition
        endpoints = np.asarray(definition["stage_endpoint_positions_bar"], dtype=float)
        if endpoints.shape != (4, 3):
            raise ValueError("BarClean requires four task-frame endpoint positions.")
        self.stage_endpoint_positions_bar = endpoints.copy()
        self.feature_definition = dict(definition["feature_definition"])

        stage0_clearance = _required_term(definition, 0, "obstacle_clearance")
        clean_surface = _required_term(definition, 1, "table_dist")
        clean_lateral = _required_term(definition, 1, "bar_lateral_offset")
        clean_pitch = _required_term(definition, 1, "tool_pitch")
        clean_yaw = _required_term(definition, 1, "tool_yaw")
        discharge_axial = _required_term(definition, 3, "bar_axial_offset")
        discharge_surface = _required_term(definition, 3, "table_dist")
        discharge_pitch = _required_term(definition, 3, "tool_pitch")
        discharge_yaw = _required_term(definition, 3, "tool_yaw")
        for term in (
            clean_surface,
            clean_lateral,
            clean_pitch,
            clean_yaw,
            discharge_axial,
            discharge_surface,
            discharge_pitch,
            discharge_yaw,
        ):
            if str(term["semantics"]) != "target_value":
                raise ValueError(
                    "Synthetic BarClean demonstrations require equality targets for "
                    f"{term['feature_name']} at stage {term['stage']}."
                )

        requested_lengths = tuple(int(value) for value in seg_lengths)
        requested_jitters = tuple(int(value) for value in seg_length_jitter)
        if len(requested_lengths) != 5 or len(requested_jitters) != 5:
            raise ValueError("BarClean requires five segment lengths and five jitters.")
        if min(requested_lengths) < 4 or min(requested_jitters) < 0:
            raise ValueError("BarClean segment lengths must be >= 4 and jitters non-negative.")

        discharge_axial_reference = float(definition["bar_axial_offset_reference"])
        discharge_standoff = float(discharge_surface["value"])
        self.clean_yaw = float(clean_yaw["value"])
        self.discharge_yaw = float(discharge_yaw["value"])
        self.discharge_axial_noise_std = float(discharge_axial_noise_std)
        self.discharge_height_variation = float(discharge_height_variation)
        if discharge_standoff <= 0.0:
            raise ValueError("The stage-four surface target must be positive.")
        # Endpoints are approximate task goals, not constraint definitions.  Real
        # demonstrations need not hit them exactly, and planning profiles may tune
        # endpoint positions independently from learned/true feature targets.
        if self.discharge_axial_noise_std < 0.0 or self.discharge_height_variation < 0.0:
            raise ValueError(
                "discharge_axial_noise_std and discharge_height_variation must be non-negative."
            )

        geometry = dict(definition["scene_geometry"])
        feature_definition = dict(definition["feature_definition"])
        super().__init__(
            *args,
            bar_axis_local=definition["bar_axis_local"],
            table_surface_point=definition["table_surface_point"],
            table_normal=definition["table_normal"],
            bar_outline_u=geometry["bar_outline_u"],
            bar_outline_v=geometry["bar_outline_v"],
            bar_height=geometry["bar_height"],
            scan_start_progress=float(endpoints[0, 0]),
            stage2_scan_distance=float(endpoints[1, 0] - endpoints[0, 0]),
            stage3_scan_distance=float(
                np.linalg.norm(endpoints[2] - endpoints[1])
            ),
            scan_standoff=float(clean_surface["value"]),
            scan_lateral_offset=float(clean_lateral["value"]),
            stage2_pitch_deg=float(np.rad2deg(clean_pitch["value"])),
            stage3_pitch_deg=float(np.rad2deg(discharge_pitch["value"])),
            obstacle_center=obstacle_center,
            obstacle_radius=float(definition["obstacle_radius"]),
            obstacle_min_clearance=float(stage0_clearance["value"]),
            tcp_offset_local=feature_definition["tcp_offset_local"],
            tool_axis_local=feature_definition["tool_axis_local"],
            task_frame_snapshot_policy=definition["task_frame"]["snapshot_policy"],
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
        self._planner_feature_evaluator = BarFeatureEvaluator(definition)

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
                "frame": "bar_table_task.x",
            }
        )
        schema.append(
            {
                "id": 9,
                "column_idx": 9,
                "name": "tool_yaw",
                "unit": "rad",
                "description": "Tool-X heading relative to the frozen bar axis",
                "frame": "bar_table_task.xy",
            }
        )
        return schema

    def get_true_constraints(self):
        return {
            self._oracle_key(term): float(term["value"])
            for term in self.task_definition["constraint_terms"]
        }

    @staticmethod
    def _oracle_key(term):
        return "stage_{}_{}".format(int(term["stage"]), str(term["feature_name"]))

    def get_constraint_specs(self):
        return [
            {
                "feature_name": str(term["feature_name"]),
                "stage": int(term["stage"]),
                "semantics": str(term["semantics"]),
                "oracle_key": self._oracle_key(term),
            }
            for term in self.task_definition["constraint_terms"]
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
                "endpoint_bar_table_task": self.stage_endpoint_positions_bar[1].tolist(),
            },
            {
                "stage": 2,
                "name": "free_reposition",
                "description": "Move freely to the transverse-discharge start pose",
            },
            {
                "stage": 3,
                "name": "transverse_discharge",
                "endpoint_bar_table_task": self.stage_endpoint_positions_bar[3].tolist(),
            },
            {
                "stage": 4,
                "name": "free_depart",
                "description": "Leave with no task constraint",
            },
        ]

    def get_planning_profile(self):
        keys = self.task_definition["planning_profile_fields"]
        return json.loads(
            json.dumps({key: self.task_definition[key] for key in keys})
        )

    def get_observation_spec(self):
        spec = dict(super().get_observation_spec())
        spec["task_frame"] = dict(self.task_definition["task_frame"])
        spec["feature_definition"] = dict(self.feature_definition)
        return spec

    def sample_scene(self, seed=None, rng=None):
        scene = super().sample_scene(seed=seed, rng=rng)
        scene["task"] = {
            "stage_specs": self.get_stage_specs(),
            "endpoints_bar_table_task": self.stage_endpoint_positions_bar.tolist(),
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
        bar_reference, bar_axis, bar_lateral = self._bar_geometry_trace(
            trajectory,
            scene=scene,
        )
        obstacle_center = self._obstacle_center_trace(trajectory, scene=scene)
        task_origins = bar_reference - np.outer(
            (bar_reference - self.table_surface_point[None, :]) @ self.table_normal,
            self.table_normal,
        )
        if not (
            np.allclose(task_origins, task_origins[:1])
            and np.allclose(bar_axis, bar_axis[:1])
            and np.allclose(bar_lateral, bar_lateral[:1])
            and np.allclose(obstacle_center, obstacle_center[:1])
        ):
            raise ValueError("BarClean feature evaluation requires one frozen task snapshot.")
        task_frame = {
            "origin": task_origins[0],
            "axial": bar_axis[0],
            "lateral": bar_lateral[0],
            "normal": self.table_normal,
        }
        tool_yaw = np.asarray(
            [tool_yaw_from_quaternion(value, task_frame) for value in quaternion],
            dtype=float,
        )
        obstacle_pose = np.concatenate(
            [obstacle_center[0], np.asarray([0.0, 0.0, 0.0, 1.0])]
        )
        planner_features = self._planner_feature_evaluator.evaluate(
            tcp,
            np.einsum("tij,j->ti", rotations, self.tool_axis_local),
            task_frame,
            obstacle_pose,
            tool_yaws=tool_yaw,
        )
        for column, name in enumerate(
            (
                "obstacle_clearance",
                "table_dist",
                "bar_lateral_offset",
                "tool_pitch",
                "tool_roll",
            )
        ):
            base_features[:, column] = planner_features[name]
        features = np.column_stack(
            [
                base_features,
                planner_features["bar_axial_offset"],
                planner_features["tool_yaw"],
            ]
        )
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

    def _tool_rotation_with_yaw(self, pitch, plane_error, yaw):
        in_plane_axis = (
            np.cos(float(pitch)) * self.bar_axis
            - np.sin(float(pitch)) * self.table_normal
        )
        tool_axis = self._unit(
            np.cos(float(plane_error)) * in_plane_axis
            + np.sin(float(plane_error)) * self.bar_lateral,
            "generated tool axis",
        )
        heading = (
            np.cos(float(yaw)) * self.bar_axis
            + np.sin(float(yaw)) * self.bar_lateral
        )
        normal_component = float(self.table_normal @ tool_axis)
        if abs(normal_component) > 1e-6:
            tool_x = heading - self.table_normal * float(heading @ tool_axis) / normal_component
        else:
            tool_x = heading - tool_axis * float(heading @ tool_axis)
        tool_x = self._unit(tool_x, "generated Tool-X axis")
        tool_y = self._unit(np.cross(tool_axis, tool_x), "generated Tool-Y axis")
        return np.column_stack([tool_x, tool_y, tool_axis])

    def generate_demo(self, seed=0, rng=None):
        local_rng = np.random.RandomState(int(seed)) if rng is None else rng
        lengths = [
            max(4, base + local_rng.randint(-jitter, jitter + 1))
            for base, jitter in zip(self.seg_lengths, self.seg_length_jitter)
        ]
        n1, n2, n3, n4, n5 = lengths

        endpoint_tcp = self._scan_tcp(
            self.stage_endpoint_positions_bar[:, 0],
            lateral=self.stage_endpoint_positions_bar[:, 1],
            standoff=self.stage_endpoint_positions_bar[:, 2],
        )
        clean_entry = endpoint_tcp[0]
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
        clean_coordinates = (
            (1.0 - clean_u[:, None]) * self.stage_endpoint_positions_bar[0]
            + clean_u[:, None] * self.stage_endpoint_positions_bar[1]
        )
        clean_coordinates[:, 0] += self._smooth_noise(
            local_rng, n2, 1, self.position_noise_std, knots=6
        )[:, 0]
        clean_coordinates[:, 1] += self._smooth_noise(
            local_rng, n2, 1, self.position_noise_std, knots=6
        )[:, 0]
        clean_coordinates[:, 2] += self._smooth_noise(
            local_rng, n2, 1, self.position_noise_std * 0.65, knots=6
        )[:, 0]
        clean_tcp = self._scan_tcp(
            clean_coordinates[:, 0],
            clean_coordinates[:, 1],
            clean_coordinates[:, 2],
        )

        clean_pitch = self.stage2_pitch + self._smooth_noise(
            local_rng, n2, 1, self.pitch_noise, knots=6
        )[:, 0]
        clean_plane = self._smooth_noise(
            local_rng, n2, 1, self.plane_noise, knots=6
        )[:, 0]
        clean_yaw = self.clean_yaw + self._smooth_noise(
            local_rng, n2, 1, self.plane_noise, knots=6
        )[:, 0]
        clean_quat = np.asarray(
            [
                self._matrix_to_quat(self._tool_rotation_with_yaw(pitch, plane, yaw))
                for pitch, plane, yaw in zip(clean_pitch, clean_plane, clean_yaw)
            ],
            dtype=float,
        )
        start_quat = self._quat_normalize(local_rng.normal(size=4))
        approach_quat = self._quat_slerp(
            start_quat,
            clean_quat[0],
            self._smoothstep(np.linspace(0.0, 1.0, n1)),
        )

        discharge_start_height = self.stage_endpoint_positions_bar[2, 2] + local_rng.uniform(
            -self.discharge_height_variation,
            self.discharge_height_variation,
        )
        discharge_end_height = self.stage_endpoint_positions_bar[3, 2] + local_rng.uniform(
            -self.discharge_height_variation,
            self.discharge_height_variation,
        )
        discharge_start = self._scan_tcp(
            [self.stage_endpoint_positions_bar[2, 0]],
            lateral=[self.stage_endpoint_positions_bar[2, 1]],
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

        discharge_start_quat = self._matrix_to_quat(
            self._tool_rotation_with_yaw(
                self.stage2_pitch,
                0.0,
                self.discharge_yaw,
            )
        )
        transition_quat = self._quat_slerp(
            clean_quat[-1],
            discharge_start_quat,
            transition_curve,
        )

        discharge_u = np.arange(1, n4 + 1, dtype=float) / float(n4)
        discharge_coordinates = (
            (1.0 - discharge_u[:, None]) * self.stage_endpoint_positions_bar[2]
            + discharge_u[:, None] * self.stage_endpoint_positions_bar[3]
        )
        discharge_coordinates[:, 0] += self._smooth_noise(
            local_rng,
            n4,
            1,
            self.discharge_axial_noise_std,
            knots=6,
        )[:, 0]
        discharge_coordinates[:, 1] += self._smooth_noise(
            local_rng, n4, 1, self.position_noise_std, knots=6
        )[:, 0]
        discharge_coordinates[:, 2] = (
            (1.0 - discharge_u) * discharge_start_height
            + discharge_u * discharge_end_height
        )
        discharge_coordinates[:, 2] += self._smooth_noise(
            local_rng, n4, 1, self.position_noise_std * 1.5, knots=6
        )[:, 0]
        discharge_tcp = self._scan_tcp(
            discharge_coordinates[:, 0],
            lateral=discharge_coordinates[:, 1],
            standoff=discharge_coordinates[:, 2],
        )
        discharge_pitch = self.stage2_pitch + self._smooth_noise(
            local_rng, n4, 1, self.pitch_noise, knots=6
        )[:, 0]
        discharge_plane = self._smooth_noise(
            local_rng, n4, 1, self.plane_noise, knots=6
        )[:, 0]
        discharge_yaw = self.discharge_yaw + self._smooth_noise(
            local_rng, n4, 1, self.plane_noise, knots=6
        )[:, 0]
        discharge_quat = np.asarray(
            [
                self._matrix_to_quat(self._tool_rotation_with_yaw(pitch, plane, yaw))
                for pitch, plane, yaw in zip(
                    discharge_pitch,
                    discharge_plane,
                    discharge_yaw,
                )
            ],
            dtype=float,
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
    processed_demo_path=None,
    source_demo_ids=None,
    **extra_env_kwargs,
):
    if processed_demo_path is None:
        raise ValueError(
            "BarClean requires processed_demo_path; synthetic dataset generation is not supported"
        )

    env_config = dict(env_kwargs or {})
    env_config.update(extra_env_kwargs)

    if processed_demo_path is not None:
        data_path = Path(processed_demo_path).expanduser()
        if not data_path.is_absolute():
            data_path = PROJECT_ROOT / data_path
        with np.load(data_path, allow_pickle=False) as archive:
            required = {
                "trajectory",
                "features",
                "feature_names",
                "timestamps",
                "coarse_bounds_indices",
                "demo_bar_poses",
                "demo_obstacle_poses",
                "downsample_hz",
                "optitrack_to_robot_rotation",
                "optitrack_to_robot_translation",
            }
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(
                    f"Processed BarClean data {data_path} is missing keys: {missing}"
                )
            all_trajectories = np.asarray(archive["trajectory"], dtype=float)
            all_features = np.asarray(archive["features"], dtype=float)
            all_timestamps = np.asarray(archive["timestamps"], dtype=float)
            bounds = np.asarray(archive["coarse_bounds_indices"], dtype=int)
            feature_names = [str(name) for name in archive["feature_names"].tolist()]
            demo_bar_poses = np.asarray(archive["demo_bar_poses"], dtype=float)
            demo_obstacle_poses = np.asarray(
                archive["demo_obstacle_poses"], dtype=float
            )
            downsample_hz = float(np.asarray(archive["downsample_hz"]).item())
            tracker_rotation = np.asarray(
                archive["optitrack_to_robot_rotation"], dtype=float
            )
            tracker_translation = np.asarray(
                archive["optitrack_to_robot_translation"], dtype=float
            )
            archive_source_demo_ids = np.asarray(
                archive["source_demo_ids"]
                if "source_demo_ids" in archive.files
                else np.arange(len(bounds)),
                dtype=int,
            )
            cutpoint_annotation_kind = str(
                np.asarray(
                    archive["cutpoint_annotation_kind"]
                    if "cutpoint_annotation_kind" in archive.files
                    else "motion_phase_stage_boundaries"
                ).item()
            )
            cutpoint_evaluation_role = str(
                np.asarray(
                    archive["cutpoint_evaluation_role"]
                    if "cutpoint_evaluation_role" in archive.files
                    else "task_informed_reference"
                ).item()
            )

        if not np.isfinite(downsample_hz) or downsample_hz <= 0.0:
            raise ValueError(
                f"Processed BarClean downsample_hz must be positive, got {downsample_hz}."
            )
        env_config.update(
            {
                "dt": 1.0 / downsample_hz,
                "optitrack_to_robot_rotation": tracker_rotation,
                "optitrack_to_robot_translation": tracker_translation,
            }
        )
        env = BarCleanEnv(**env_config)

        expected_names = [spec["name"] for spec in env.get_feature_schema()]
        if feature_names != expected_names:
            raise ValueError(
                "Processed BarClean feature order does not match the environment schema: "
                f"got {feature_names}, expected {expected_names}."
            )
        if bounds.ndim != 2 or bounds.shape[1] != 6:
            raise ValueError(
                "coarse_bounds_indices must have shape (num_demos, 6) for five stages."
            )
        if demo_bar_poses.shape != (len(bounds), 7):
            raise ValueError(
                "demo_bar_poses must have shape (num_demos, 7), got "
                f"{demo_bar_poses.shape}."
            )
        if demo_obstacle_poses.shape != (len(bounds), 7):
            raise ValueError(
                "demo_obstacle_poses must have shape (num_demos, 7), got "
                f"{demo_obstacle_poses.shape}."
            )
        if archive_source_demo_ids.shape != (len(bounds),):
            raise ValueError(
                "source_demo_ids must have shape (num_demos,), got "
                f"{archive_source_demo_ids.shape}."
            )

        if source_demo_ids is None:
            requested_count = int(n_demos)
            if requested_count <= 0:
                raise ValueError("n_demos must be positive")
            if requested_count > len(bounds):
                raise ValueError(
                    f"Requested {n_demos} demos, but {data_path} contains only {len(bounds)}."
                )
            selected_archive_indices = list(range(requested_count))
        else:
            requested_source_ids = [int(value) for value in source_demo_ids]
            if not requested_source_ids:
                raise ValueError("source_demo_ids must contain at least one demo ID")
            if len(set(requested_source_ids)) != len(requested_source_ids):
                raise ValueError(
                    f"source_demo_ids contains duplicates: {requested_source_ids}"
                )
            archive_id_to_index = {}
            duplicate_archive_ids = set()
            for archive_index, source_id in enumerate(archive_source_demo_ids.tolist()):
                source_id = int(source_id)
                if source_id in archive_id_to_index:
                    duplicate_archive_ids.add(source_id)
                archive_id_to_index[source_id] = int(archive_index)
            if duplicate_archive_ids:
                raise ValueError(
                    "Processed BarClean data contains duplicate source_demo_ids: "
                    f"{sorted(duplicate_archive_ids)}"
                )
            missing_source_ids = [
                source_id
                for source_id in requested_source_ids
                if source_id not in archive_id_to_index
            ]
            if missing_source_ids:
                raise ValueError(
                    f"Requested source_demo_ids {missing_source_ids} are not present in {data_path}; "
                    f"available IDs are {archive_source_demo_ids.tolist()}."
                )
            selected_archive_indices = [
                archive_id_to_index[source_id] for source_id in requested_source_ids
            ]

        demo_scenes = [
            BarInspectScene(
                bar_pose_optitrack=bar_pose,
                obstacle_pose_optitrack=obstacle_pose,
            )
            for bar_pose, obstacle_pose in zip(demo_bar_poses, demo_obstacle_poses)
        ]
        selected_demo_scenes = [
            demo_scenes[archive_index]
            for archive_index in selected_archive_indices
        ]
        env.set_scene(selected_demo_scenes[0])
        env.set_demo_scenes(selected_demo_scenes)

        demos = []
        features = []
        true_labels = []
        true_cutpoints = []
        timestamps = []
        scene_specs = []
        for demo_index, archive_index in enumerate(selected_archive_indices):
            row = bounds[archive_index]
            row = np.asarray(row, dtype=int)
            if not (
                row[0] >= 0
                and row[-1] <= len(all_trajectories)
                and np.all(np.diff(row) > 0)
            ):
                raise ValueError(
                    f"Invalid five-stage bounds for demo {demo_index}: {row.tolist()}"
                )
            begin, end = int(row[0]), int(row[-1])
            demos.append(all_trajectories[begin:end].copy())
            features.append(all_features[begin:end].copy())
            stage_lengths = np.diff(row)
            true_labels.append(
                np.repeat(np.arange(5, dtype=int), stage_lengths.astype(int))
            )
            true_cutpoints.append((row[1:-1] - begin).astype(int))
            demo_time = all_timestamps[begin:end].copy()
            timestamps.append(demo_time - demo_time[0])
            scene_specs.append(
                {
                    "demo_index": int(demo_index),
                    "archive_demo_index": int(archive_index),
                    "source_demo_id": int(archive_source_demo_ids[archive_index]),
                    "source": "processed_real_demo",
                    "processed_demo_path": str(data_path),
                    "recording_bounds": row.tolist(),
                    **demo_scenes[archive_index].to_dict(),
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
                "source_demo_ids": [
                    int(archive_source_demo_ids[index])
                    for index in selected_archive_indices
                ],
                "cutpoint_annotation_kind": cutpoint_annotation_kind,
                "cutpoint_evaluation_role": cutpoint_evaluation_role,
                "cutpoint_annotations": {
                    "kind": cutpoint_annotation_kind,
                    "is_ground_truth": False,
                    "usage": "5 Hz motion-phase stage references",
                },
                "stage_specs": env.get_stage_specs(),
                "scene_specs": scene_specs,
                "timestamps": timestamps,
                "observation_specs": env.get_observation_spec(),
                "planning_profile": env.get_planning_profile(),
                "render_camera_presets": env.get_render_camera_presets(),
                "asset_handles": env.get_asset_handles(),
                "default_learning_features": list(env.default_learning_features),
            },
        )

__all__ = ["BarCleanEnv", "BarInspectScene", "load_BarClean"]
