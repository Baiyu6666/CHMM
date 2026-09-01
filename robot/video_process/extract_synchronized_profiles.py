#!/usr/bin/env python3
"""Extract video-synchronized executed and controller-planned feature profiles."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pybullet as bullet
from rosbags.highlevel import AnyReader

from render_experiment_profiles import probe_video


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROS_SOURCE_ROOT = PROJECT_ROOT / "robot" / "stage_cons_iiwa14" / "ros_ws" / "src"
IIWA_XACRO = ROS_SOURCE_ROOT / "iiwa_description" / "urdf" / "iiwa14.urdf.xacro"
BARCLEAN_CONFIG = (
    ROS_SOURCE_ROOT
    / "stage_constraint_planner"
    / "config"
    / "bar_clean_true.json"
)
CONTROLLER_TOPIC = "/iiwa14/PositionTrajectoryController/state"
PLAN_TOPIC = "/stage_cons/plan"
ORIENTATION_TOPIC = "/stage_cons/plan_orientation_constraints"
CACHE_FILENAME = "synchronized_profiles.json"


@dataclass(frozen=True)
class ControllerSeries:
    joint_names: tuple[str, ...]
    absolute_times_s: np.ndarray
    actual_positions: np.ndarray
    desired_positions: np.ndarray
    plan_positions: np.ndarray
    plan_quaternions: np.ndarray
    stage_boundaries: np.ndarray
    transition_windows: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract cached profiles aligned to video content time"
    )
    parser.add_argument("--run-directories", nargs="+", type=Path, required=True)
    parser.add_argument("--camera-calibration", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return payload


def _stamp_seconds(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def _read_controller_series(path: Path) -> ControllerSeries:
    joint_names = None
    times = []
    actual = []
    desired = []
    plan_positions = None
    plan_quaternions = None
    stage_boundaries = None
    transition_windows = None
    with AnyReader([path]) as reader:
        connections = [
            connection
            for connection in reader.connections
            if connection.topic in {CONTROLLER_TOPIC, PLAN_TOPIC, ORIENTATION_TOPIC}
        ]
        for connection, _, raw in reader.messages(connections=connections):
            message = reader.deserialize(raw, connection.msgtype)
            if connection.topic == CONTROLLER_TOPIC:
                names = tuple(str(value) for value in message.joint_names)
                actual_values = np.asarray(message.actual.positions, dtype=float)
                desired_values = np.asarray(message.desired.positions, dtype=float)
                if joint_names is None:
                    joint_names = names
                if names != joint_names:
                    raise RuntimeError("controller joint order changed inside the bag")
                if actual_values.shape != desired_values.shape or actual_values.shape != (
                    len(names),
                ):
                    continue
                times.append(_stamp_seconds(message.header.stamp))
                actual.append(actual_values)
                desired.append(desired_values)
            elif connection.topic == PLAN_TOPIC:
                plan_positions = np.asarray(
                    [
                        [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                        for pose in message.poses
                    ],
                    dtype=float,
                )
                plan_quaternions = np.asarray(
                    [
                        [
                            pose.pose.orientation.x,
                            pose.pose.orientation.y,
                            pose.pose.orientation.z,
                            pose.pose.orientation.w,
                        ]
                        for pose in message.poses
                    ],
                    dtype=float,
                )
            elif connection.topic == ORIENTATION_TOPIC:
                payload = json.loads(message.data)
                timing = payload["stage_timing"]
                stage_boundaries = np.asarray(timing["boundaries"], dtype=int)
                transition_windows = np.asarray(
                    timing["transition_windows"], dtype=int
                ).reshape(-1, 2)
    if joint_names is None or not times:
        raise RuntimeError(f"{path} contains no controller state samples")
    if plan_positions is None or plan_quaternions is None:
        raise RuntimeError(f"{path} contains no Cartesian plan")
    if stage_boundaries is None or transition_windows is None:
        raise RuntimeError(f"{path} contains no stage-timing metadata")
    order = np.argsort(np.asarray(times, dtype=float), kind="stable")
    sorted_times = np.asarray(times, dtype=float)[order]
    unique = np.concatenate(([True], np.diff(sorted_times) > 1e-9))
    return ControllerSeries(
        joint_names=joint_names,
        absolute_times_s=sorted_times[unique],
        actual_positions=np.asarray(actual, dtype=float)[order][unique],
        desired_positions=np.asarray(desired, dtype=float)[order][unique],
        plan_positions=plan_positions,
        plan_quaternions=plan_quaternions,
        stage_boundaries=stage_boundaries,
        transition_windows=transition_windows,
    )


class IiwaKinematics:
    def __init__(self) -> None:
        environment = dict(os.environ)
        existing = environment.get("ROS_PACKAGE_PATH", "")
        environment["ROS_PACKAGE_PATH"] = str(ROS_SOURCE_ROOT) + (
            os.pathsep + existing if existing else ""
        )
        result = subprocess.run(
            [
                "/opt/ros/noetic/bin/xacro",
                str(IIWA_XACRO),
                "robot_name:=iiwa14",
                "simple_collision:=true",
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        )
        expanded = result.stdout.replace(
            "package://", str(ROS_SOURCE_ROOT.resolve()) + "/"
        )
        self._temporary = tempfile.TemporaryDirectory(prefix="iiwa14_video_sync_")
        urdf_path = Path(self._temporary.name) / "iiwa14.urdf"
        urdf_path.write_text(expanded, encoding="utf-8")
        self._client = bullet.connect(bullet.DIRECT)
        self._robot = bullet.loadURDF(str(urdf_path), useFixedBase=True)
        self._joint_indices = {}
        self._link_indices = {}
        for index in range(bullet.getNumJoints(self._robot)):
            info = bullet.getJointInfo(self._robot, index)
            self._joint_indices[info[1].decode("utf-8")] = index
            self._link_indices[info[12].decode("utf-8")] = index
        self._tip_index = self._link_indices["iiwa14_link_7"]

    def close(self) -> None:
        bullet.disconnect(self._client)
        self._temporary.cleanup()

    def poses(
        self, joint_names: Sequence[str], joint_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        indices = [self._joint_indices[name] for name in joint_names]
        positions = np.empty((len(joint_positions), 3), dtype=float)
        quaternions = np.empty((len(joint_positions), 4), dtype=float)
        for row, values in enumerate(np.asarray(joint_positions, dtype=float)):
            for index, value in zip(indices, values):
                bullet.resetJointState(self._robot, index, float(value))
            state = bullet.getLinkState(
                self._robot, self._tip_index, computeForwardKinematics=True
            )
            positions[row] = state[4]
            quaternions[row] = state[5]
        return positions, quaternions


def _interpolate_joints(
    source_times: np.ndarray,
    positions: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    return np.column_stack(
        [
            np.interp(target_times, source_times, positions[:, column])
            for column in range(positions.shape[1])
        ]
    )


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(first * second, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _bar_centerline_lateral_offset(
    axial_coordinates: np.ndarray, specification: object
) -> np.ndarray:
    axial = np.asarray(axial_coordinates, dtype=float)
    spec = dict(specification) if isinstance(specification, dict) else {"type": "straight"}
    centerline_type = str(spec.get("type", "straight"))
    if centerline_type == "straight":
        return np.zeros_like(axial)
    if centerline_type != "circular_arc_chord":
        raise ValueError(f"Unsupported bar lateral centerline type {centerline_type}")
    radius = float(spec["radius_m"])
    lower, upper = (float(value) for value in spec["axial_bounds_m"])
    bulge_sign = float(spec["bulge_sign"])
    if (
        not math.isfinite(radius)
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or not math.isfinite(bulge_sign)
        or radius <= 0.5 * (upper - lower)
        or upper <= lower
        or abs(bulge_sign) != 1.0
    ):
        raise ValueError("Invalid circular-arc bar centerline geometry")
    midpoint = 0.5 * (lower + upper)
    half_chord = 0.5 * (upper - lower)
    chord_height = math.sqrt(radius * radius - half_chord * half_chord)
    offset = np.zeros_like(axial)
    active = (axial >= lower) & (axial <= upper)
    offset[active] = bulge_sign * (
        np.sqrt(
            np.maximum(radius * radius - np.square(axial[active] - midpoint), 0.0)
        )
        - chord_height
    )
    return offset


def _monotonic_plan_mapping(
    plan_positions: np.ndarray,
    plan_quaternions: np.ndarray,
    desired_positions: np.ndarray,
    desired_quaternions: np.ndarray,
) -> tuple[np.ndarray, dict]:
    position_cost = np.linalg.norm(
        desired_positions[:, None, :] - plan_positions[None, :, :], axis=2
    ) / 0.005
    orientation_cost = _quaternion_angle(
        desired_quaternions[:, None, :], plan_quaternions[None, :, :]
    ) / math.radians(2.0)
    costs = position_cost + orientation_cost
    rows, columns = costs.shape
    accumulated = np.full((rows + 1, columns + 1), np.inf, dtype=float)
    accumulated[0, 0] = 0.0
    parent = np.zeros((rows, columns), dtype=np.uint8)
    for row in range(rows):
        for column in range(columns):
            choices = (
                accumulated[row, column + 1],
                accumulated[row + 1, column],
                accumulated[row, column],
            )
            choice = int(np.argmin(choices))
            accumulated[row + 1, column + 1] = costs[row, column] + choices[choice]
            parent[row, column] = choice
    row = rows - 1
    column = columns - 1
    pairs = []
    while row >= 0 and column >= 0:
        pairs.append((row, column))
        choice = int(parent[row, column])
        if choice == 0:
            row -= 1
        elif choice == 1:
            column -= 1
        else:
            row -= 1
            column -= 1
    pairs.reverse()
    grouped = [[] for _ in range(columns)]
    for row, column in pairs:
        grouped[column].append(row)
    mapping = np.asarray(
        [int(round(float(np.median(values)))) for values in grouped], dtype=int
    )
    matched_position_error = np.linalg.norm(
        desired_positions[mapping] - plan_positions, axis=1
    )
    matched_orientation_error = _quaternion_angle(
        desired_quaternions[mapping], plan_quaternions
    )
    return mapping, {
        "median_position_error_m": float(np.median(matched_position_error)),
        "maximum_position_error_m": float(np.max(matched_position_error)),
        "median_orientation_error_deg": float(
            np.degrees(np.median(matched_orientation_error))
        ),
        "maximum_orientation_error_deg": float(
            np.degrees(np.max(matched_orientation_error))
        ),
    }


def _feature_values(
    positions: np.ndarray,
    quaternions: np.ndarray,
    visualization: dict,
    definition: dict,
) -> np.ndarray:
    scene = visualization["scene_geometry"]
    bar = scene["bar"]
    obstacle = scene["obstacle"]
    axis_xy = np.asarray(bar["axis"], dtype=float)
    axis_xy /= np.linalg.norm(axis_xy)
    axis = np.asarray([axis_xy[0], axis_xy[1], 0.0], dtype=float)
    lateral = np.asarray([-axis_xy[1], axis_xy[0], 0.0], dtype=float)
    normal = np.asarray(definition["table_normal"], dtype=float)
    normal /= np.linalg.norm(normal)
    table_point = np.asarray(definition["table_surface_point"], dtype=float)
    pivot = np.asarray([*bar["pivot"], 0.0], dtype=float)
    obstacle_radius = float(obstacle["radius"])
    if "center" in obstacle:
        center = np.asarray(obstacle["center"], dtype=float)
        obstacle_distance = np.linalg.norm(positions[:, :2] - center, axis=1)
    else:
        endpoints = np.asarray(obstacle["endpoints"], dtype=float)
        segment = endpoints[1] - endpoints[0]
        phase = np.clip(
            ((positions[:, :2] - endpoints[0]) @ segment) / (segment @ segment),
            0.0,
            1.0,
        )
        closest = endpoints[0] + phase[:, None] * segment
        obstacle_distance = np.linalg.norm(positions[:, :2] - closest, axis=1)
    x, y, z, w = quaternions.T
    tool_axis = np.column_stack(
        (
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        )
    )
    tool_x = np.column_stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        )
    )
    tool_x_horizontal = tool_x - (tool_x @ normal)[:, None] * normal[None, :]
    tool_x_horizontal /= np.linalg.norm(tool_x_horizontal, axis=1)[:, None]
    relative_bar = positions - pivot
    raw_axial = relative_bar @ axis
    raw_lateral = relative_bar @ lateral
    axial_reference = float(
        visualization["feature_series"]["true_constraints"].get(
            "bar_axial_offset_reference",
            definition.get("bar_axial_offset_reference", 0.0),
        )
    )
    values = {
        "obs_dist": obstacle_distance - obstacle_radius,
        "table_dist": (positions - table_point) @ normal,
        "lateral_offset": raw_lateral
        - _bar_centerline_lateral_offset(
            raw_axial, bar.get("lateral_centerline")
        ),
        "tool_pitch": np.arctan2(-(tool_axis @ normal), tool_axis @ axis),
        "tool_roll": np.arcsin(np.clip(tool_axis @ lateral, -1.0, 1.0)),
        "tool_yaw": np.arctan2(
            tool_x_horizontal @ lateral, tool_x_horizontal @ axis
        ),
        "axial_offset": raw_axial - axial_reference,
    }
    names = [str(item["name"]) for item in visualization["feature_series"]["schema"]]
    return np.column_stack([values[name] for name in names])


def extract_run(
    run_directory: Path,
    kinematics: IiwaKinematics,
    camera_delay_s: float,
    *,
    overwrite: bool,
) -> dict:
    run_directory = run_directory.resolve()
    output_path = run_directory / CACHE_FILENAME
    if output_path.exists() and not overwrite:
        raise RuntimeError(f"refusing to overwrite {output_path}")
    visualization = _load_json(run_directory / "visualization.json")
    video_metadata = _load_json(run_directory / "execution_video_metadata.json")
    definition = _load_json(BARCLEAN_CONFIG)
    controller = _read_controller_series(run_directory / "real_task.bag")
    video_info = probe_video(run_directory / "execution.mp4")
    motion_start_s = float(video_metadata["motion_start_unix_ns"]) / 1e9
    motion_end_s = float(video_metadata["motion_end_unix_ns"]) / 1e9
    motion_duration_s = motion_end_s - motion_start_s
    if abs(float(video_info["duration"]) - motion_duration_s) > 0.08:
        raise RuntimeError("video duration does not match its recorded motion window")
    video_times = np.asarray(
        [
            0.0,
            *(
                controller.absolute_times_s
                - motion_start_s
                + float(camera_delay_s)
            ).tolist(),
            float(video_info["duration"]),
        ],
        dtype=float,
    )
    video_times = np.unique(
        np.clip(
            video_times[
                (video_times >= 0.0)
                & (video_times <= float(video_info["duration"]))
            ],
            0.0,
            float(video_info["duration"]),
        )
    )
    robot_absolute_times = motion_start_s + video_times - float(camera_delay_s)
    leading_hold_s = max(
        0.0, float(controller.absolute_times_s[0] - robot_absolute_times[0])
    )
    trailing_hold_s = max(
        0.0, float(robot_absolute_times[-1] - controller.absolute_times_s[-1])
    )
    if leading_hold_s > float(camera_delay_s) + 0.1 or trailing_hold_s > 0.1:
        raise RuntimeError("controller-state coverage gap is too large to hold safely")
    actual_joints = _interpolate_joints(
        controller.absolute_times_s,
        controller.actual_positions,
        robot_absolute_times,
    )
    desired_joints = _interpolate_joints(
        controller.absolute_times_s,
        controller.desired_positions,
        robot_absolute_times,
    )
    actual_positions, actual_quaternions = kinematics.poses(
        controller.joint_names, actual_joints
    )
    desired_positions, desired_quaternions = kinematics.poses(
        controller.joint_names, desired_joints
    )
    actual_features = _feature_values(
        actual_positions, actual_quaternions, visualization, definition
    )
    desired_features = _feature_values(
        desired_positions, desired_quaternions, visualization, definition
    )
    motion_mask = (
        controller.absolute_times_s >= motion_start_s
    ) & (controller.absolute_times_s <= motion_end_s)
    motion_times = controller.absolute_times_s[motion_mask] - motion_start_s
    motion_desired_positions, motion_desired_quaternions = kinematics.poses(
        controller.joint_names, controller.desired_positions[motion_mask]
    )
    mapping, mapping_validation = _monotonic_plan_mapping(
        controller.plan_positions,
        controller.plan_quaternions,
        motion_desired_positions,
        motion_desired_quaternions,
    )
    boundary_times = (
        motion_times[mapping[controller.stage_boundaries[:-1]]]
        + float(camera_delay_s)
    )
    transition_times = (
        motion_times[mapping[controller.transition_windows[:, 1]]]
        + float(camera_delay_s)
    )
    if np.any(np.diff(boundary_times) <= 0.0) or np.any(
        transition_times < boundary_times
    ):
        raise RuntimeError("mapped stage timing is not monotonic")
    schema = json.loads(json.dumps(visualization["feature_series"]["schema"]))
    synchronized = {
        "schema_version": 1,
        "task_id": str(visualization.get("task_id", "")),
        "run_id": run_directory.name,
        "timing": {
            "basis": "controller_state_header_stamp_aligned_to_video_content",
            "motion_start_unix_ns": int(video_metadata["motion_start_unix_ns"]),
            "motion_end_unix_ns": int(video_metadata["motion_end_unix_ns"]),
            "camera_pipeline_delay_s": float(camera_delay_s),
            "phone_wall_clock_used": False,
            "video_timestamp_clock": "workstation CLOCK_MONOTONIC via V4L2",
            "video_duration_s": float(video_info["duration"]),
            "motion_window_duration_s": motion_duration_s,
            "duration_error_s": float(video_info["duration"]) - motion_duration_s,
        },
        "feature_series": {
            "source": "controller_actual_joint_state_fk",
            "schema": schema,
            "samples": np.column_stack((video_times, actual_features)).tolist(),
        },
        "planned_feature_series": {
            "source": "controller_desired_joint_state_fk",
            "schema": schema,
            "constraint_specs": json.loads(
                json.dumps(
                    visualization["planned_feature_series"].get(
                        "constraint_specs", []
                    )
                )
            ),
            "planning_constraint_specs": json.loads(
                json.dumps(
                    visualization["planned_feature_series"].get(
                        "planning_constraint_specs", []
                    )
                )
            ),
            "planning_constraint_source": str(
                visualization["planned_feature_series"].get(
                    "planning_constraint_source", ""
                )
            ),
            "samples": np.column_stack((video_times, desired_features)).tolist(),
        },
        "stage_boundary_indices": controller.stage_boundaries.tolist(),
        "stage_boundary_times": boundary_times.tolist(),
        "stage_transition_end_times": transition_times.tolist(),
        "validation": {
            "controller_sample_count": int(len(controller.absolute_times_s)),
            "synchronized_sample_count": int(len(video_times)),
            "plan_point_count": int(len(controller.plan_positions)),
            "leading_stationary_hold_s": leading_hold_s,
            "trailing_stationary_hold_s": trailing_hold_s,
            **mapping_validation,
        },
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(synchronized, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return {
        "run": run_directory.name,
        "output": str(output_path),
        "samples": len(video_times),
        "camera_pipeline_delay_s": float(camera_delay_s),
        "duration_error_s": synchronized["timing"]["duration_error_s"],
        **mapping_validation,
    }


def main() -> None:
    options = build_parser().parse_args()
    calibration = _load_json(options.camera_calibration)
    camera_delay_s = float(calibration["selected_camera_pipeline_delay_s"])
    kinematics = IiwaKinematics()
    try:
        results = [
            extract_run(
                directory,
                kinematics,
                camera_delay_s,
                overwrite=bool(options.overwrite),
            )
            for directory in options.run_directories
        ]
    finally:
        kinematics.close()
    print(json.dumps({"ok": True, "results": results}, indent=2))


if __name__ == "__main__":
    main()
