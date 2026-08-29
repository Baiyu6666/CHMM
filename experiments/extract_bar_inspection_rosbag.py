#!/usr/bin/env python3
"""Build a BarInspect training dataset from one ROS1 recording."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.BarInspect import BarInspectEnv, BarInspectScene  # noqa: E402
from experiments.bar_inspect_processing import (  # noqa: E402
    annotation_arrays,
    dominant_static_pose_cluster,
    downsample_processed_arrays,
    load_cutpoint_annotations,
    nearest_indices,
    robust_static_pose,
)


JOINT_TOPIC = "/iiwa14/joint_states"
BAR_TOPIC = "/vrpn_client_node/baiyu_bar/pose_from_iiwa14"
OBSTACLE_TOPIC = "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14"
OBSTACLE_B_TOPIC = "/vrpn_client_node/baiyu_obs_bar_b/pose_from_iiwa14"
JOINT_ORDER = tuple(f"iiwa14_joint_{index}" for index in range(1, 8))
JOINT_ORIGINS_XYZ = np.asarray(
    [
        [0.0, 0.0, 0.1575],
        [0.0, 0.0, 0.2025],
        [0.0, 0.2045, 0.0],
        [0.0, 0.0, 0.2155],
        [0.0, 0.1845, 0.0],
        [0.0, 0.0, 0.2155],
        [0.0, 0.0810, 0.0],
    ],
    dtype=float,
)
JOINT_ORIGINS_RPY = np.asarray(
    [
        [0.0, 0.0, 0.0],
        [np.pi / 2.0, 0.0, np.pi],
        [np.pi / 2.0, 0.0, np.pi],
        [np.pi / 2.0, 0.0, 0.0],
        [-np.pi / 2.0, np.pi, 0.0],
        [np.pi / 2.0, 0.0, 0.0],
        [-np.pi / 2.0, np.pi, 0.0],
    ],
    dtype=float,
)


def pose_row(message):
    position = message.pose.position
    orientation = message.pose.orientation
    return [
        position.x,
        position.y,
        position.z,
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    ]


def homogeneous(rotation, translation):
    output = np.eye(4)
    output[:3, :3] = np.asarray(rotation, dtype=float)
    output[:3, 3] = np.asarray(translation, dtype=float)
    return output


def iiwa14_fk(joint_positions):
    origin_transforms = [
        homogeneous(Rotation.from_euler("xyz", rpy).as_matrix(), xyz)
        for xyz, rpy in zip(JOINT_ORIGINS_XYZ, JOINT_ORIGINS_RPY)
    ]
    joint_trace = np.asarray(joint_positions, dtype=float)
    poses = np.empty((len(joint_trace), 7), dtype=float)
    for row_index, joint_row in enumerate(joint_trace):
        transform = np.eye(4)
        for origin_transform, angle in zip(origin_transforms, joint_row):
            transform = transform @ origin_transform @ homogeneous(
                Rotation.from_rotvec([0.0, 0.0, float(angle)]).as_matrix(),
                [0.0, 0.0, 0.0],
            )
        poses[row_index, :3] = transform[:3, 3]
        poses[row_index, 3:] = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return poses


def read_recording(bag_path, joint_topic, bar_topic, obstacle_topic):
    try:
        from rosbags.highlevel import AnyReader
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Reading ROS1 bags requires the 'rosbags' Python package."
        ) from error

    obstacle_topics = (
        (obstacle_topic,)
        if isinstance(obstacle_topic, str)
        else tuple(obstacle_topic)
    )
    topics = (joint_topic, bar_topic, *obstacle_topics)
    raw_times = {topic: [] for topic in topics}
    raw_values = {topic: [] for topic in topics}
    with AnyReader([Path(bag_path)]) as reader:
        selected = [connection for connection in reader.connections if connection.topic in topics]
        present = {connection.topic for connection in selected}
        missing = sorted(set(topics).difference(present))
        if missing:
            raise RuntimeError(f"ROS bag is missing required topics: {missing}")
        for connection, timestamp, raw_data in reader.messages(connections=selected):
            message = reader.deserialize(raw_data, connection.msgtype)
            topic = connection.topic
            if topic == joint_topic:
                name_to_index = {name: index for index, name in enumerate(message.name)}
                if not all(name in name_to_index for name in JOINT_ORDER):
                    continue
                value = [message.position[name_to_index[name]] for name in JOINT_ORDER]
            else:
                value = pose_row(message)
            raw_times[topic].append(float(timestamp) / 1e9)
            raw_values[topic].append(value)
    for topic in topics:
        raw_times[topic] = np.asarray(raw_times[topic], dtype=float)
        raw_values[topic] = np.asarray(raw_values[topic], dtype=float)
        if len(raw_times[topic]) == 0:
            raise RuntimeError(f"No usable samples found on {topic}.")
    return raw_times, raw_values


def load_environment_config(path, feature_dt):
    with Path(path).open(encoding="utf-8") as stream:
        config = dict(json.load(stream))
    for key in ("name", "n_demos", "seed", "processed_demo_path", "method_overrides"):
        config.pop(key, None)
    config["dt"] = float(feature_dt)
    return config


def build_feature_grid(raw_times, raw_values, joint_topic, feature_hz):
    dt = 1.0 / float(feature_hz)
    common_start = max(values[0] for values in raw_times.values())
    common_end = min(values[-1] for values in raw_times.values())
    grid_absolute = np.arange(np.ceil(common_start / dt) * dt, common_end, dt)
    joint_indices, joint_gaps = nearest_indices(raw_times[joint_topic], grid_absolute)
    return (
        grid_absolute - grid_absolute[0],
        raw_values[joint_topic][joint_indices],
        joint_gaps,
        float(grid_absolute[0]),
    )


def scene_window_mask(raw_times, common_grid_start, start_s, end_s):
    relative_times = np.asarray(raw_times, dtype=float) - float(common_grid_start)
    mask = relative_times >= float(start_s)
    if end_s is not None:
        mask &= relative_times <= float(end_s)
    if not np.any(mask):
        raise ValueError("The requested scene-lock window contains no samples.")
    return mask


def lock_demo_obstacle_poses(
    obstacle_times,
    obstacle_poses,
    common_grid_start,
    bounds_times,
    position_radius_m,
):
    relative_times = np.asarray(obstacle_times, dtype=float) - float(common_grid_start)
    locks = []
    for demo_index, row in enumerate(np.asarray(bounds_times, dtype=float)):
        mask = (relative_times >= float(row[0])) & (relative_times <= float(row[-1]))
        if not np.any(mask):
            raise ValueError(f"Demo {demo_index} contains no obstacle tracker samples.")
        lock = dominant_static_pose_cluster(
            np.asarray(obstacle_poses, dtype=float)[mask],
            position_radius_m=position_radius_m,
        )
        lock["sample_count"] = int(np.count_nonzero(mask))
        locks.append(lock)
    return locks


def concatenate_annotated_demos(
    timestamps,
    joint_positions,
    joint_gaps,
    trajectory,
    bounds_times,
    environment,
    demo_scenes,
    max_joint_gap,
):
    source_bounds, _, _ = annotation_arrays(timestamps, bounds_times)
    output = {
        "timestamps": [],
        "joint_positions": [],
        "trajectory": [],
        "features": [],
        "demo_id": [],
        "coarse_stage_labels": [],
    }
    output_bounds = []
    selected_gaps = []
    offset = 0
    for demo_index, row in enumerate(source_bounds):
        begin, stage2, stage3, stage4, end = (int(value) for value in row)
        demo_slice = slice(begin, end)
        demo_gaps = joint_gaps[demo_slice]
        if float(np.max(demo_gaps)) > float(max_joint_gap):
            raise RuntimeError(
                f"Joint-state gap too large inside demo {demo_index}: "
                f"{np.max(demo_gaps):.6f}s"
            )
        demo_trajectory = trajectory[demo_slice]
        local_bounds = np.asarray(
            [0, stage2 - begin, stage3 - begin, stage4 - begin, end - begin],
            dtype=np.int64,
        )
        labels = np.concatenate(
            [
                np.full(length, stage, dtype=np.int64)
                for stage, length in enumerate(np.diff(local_bounds))
            ]
        )
        output["timestamps"].append(timestamps[demo_slice])
        output["joint_positions"].append(joint_positions[demo_slice])
        output["trajectory"].append(demo_trajectory)
        output["features"].append(
            environment.compute_all_features_matrix(
                demo_trajectory,
                scene=demo_scenes[demo_index],
            )
        )
        output["demo_id"].append(
            np.full(end - begin, demo_index, dtype=np.int64)
        )
        output["coarse_stage_labels"].append(labels)
        output_bounds.append(local_bounds + offset)
        selected_gaps.append(demo_gaps)
        offset += end - begin
    arrays = {
        key: np.concatenate(value, axis=0)
        for key, value in output.items()
    }
    return arrays, np.asarray(output_bounds, dtype=np.int64), np.concatenate(selected_gaps)


def validate_demo_start_positions(
    timestamps,
    trajectory,
    bounds_times,
    environment,
    demo_scenes,
    minimum_north_offset_m,
):
    source_bounds, _, _ = annotation_arrays(timestamps, bounds_times)
    start_indices = source_bounds[:, 0]
    trajectory = np.asarray(trajectory, dtype=float)
    north_offsets = np.asarray(
        [
            trajectory[start_index, 1]
            - environment._obstacle_center_trace(
                trajectory[start_index : start_index + 1],
                scene=demo_scenes[demo_index],
            )[0, 1]
            for demo_index, start_index in enumerate(start_indices)
        ],
        dtype=float,
    )
    invalid = np.flatnonzero(north_offsets < float(minimum_north_offset_m))
    if len(invalid):
        details = ", ".join(
            f"demo {int(index)}: {north_offsets[index]:.3f}m"
            for index in invalid
        )
        raise RuntimeError(
            "Annotated demo starts must be north of the obstacle by at least "
            f"{float(minimum_north_offset_m):.3f}m; {details}. "
            "This usually means the reset from the previous attempt was included."
        )
    return north_offsets


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate one robust static BarInspect scene, compute features at 10 Hz, "
            "then optionally subsample complete rows for training."
        )
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument(
        "--env-config",
        type=Path,
        default=PROJECT_ROOT / "configs/envs/BarInspect.json",
    )
    parser.add_argument("--joint-topic", default=JOINT_TOPIC)
    parser.add_argument("--bar-topic", default=BAR_TOPIC)
    parser.add_argument("--obstacle-topic", default=OBSTACLE_TOPIC)
    parser.add_argument("--feature-hz", type=float, default=10.0)
    parser.add_argument("--output-hz", type=float, default=5.0)
    parser.add_argument("--max-joint-gap", type=float, default=0.05)
    parser.add_argument("--scene-position-outlier-m", type=float, default=0.020)
    parser.add_argument(
        "--demo-obstacle-cluster-radius-m",
        type=float,
        default=0.005,
        help=(
            "Fixed radius used to select the dominant static obstacle-pose cluster "
            "inside each annotated demo."
        ),
    )
    parser.add_argument("--bar-axis-outlier-deg", type=float, default=5.0)
    parser.add_argument(
        "--min-start-north-offset-m",
        type=float,
        default=0.03,
        help=(
            "Reject annotations whose first TCP sample is not this far north "
            "(+Y in the robot base) of the tracked obstacle center."
        ),
    )
    parser.add_argument(
        "--scene-lock-start-s",
        type=float,
        default=0.0,
        help="Scene-lock window start relative to the common processing grid.",
    )
    parser.add_argument(
        "--scene-lock-end-s",
        type=float,
        help="Optional scene-lock window end relative to the common processing grid.",
    )
    parser.add_argument(
        "--only-annotated",
        action="store_true",
        help="Export only annotated demo intervals, excluding resets and failed attempts.",
    )
    parser.add_argument(
        "--feature-output",
        type=Path,
        help="Optionally save the annotated feature-rate arrays before downsampling.",
    )
    args = parser.parse_args()

    if args.feature_hz <= 0.0 or args.output_hz <= 0.0:
        parser.error("--feature-hz and --output-hz must be positive")
    ratio = float(args.feature_hz) / float(args.output_hz)
    factor = int(round(ratio))
    if factor < 1 or not np.isclose(ratio, factor, atol=1e-9):
        parser.error("--feature-hz must be an integer multiple of --output-hz")
    if args.scene_position_outlier_m <= 0.0:
        parser.error("--scene-position-outlier-m must be positive")
    if args.demo_obstacle_cluster_radius_m <= 0.0:
        parser.error("--demo-obstacle-cluster-radius-m must be positive")
    if not 0.0 < args.bar_axis_outlier_deg < 180.0:
        parser.error("--bar-axis-outlier-deg must be between 0 and 180 degrees")
    if args.min_start_north_offset_m < 0.0:
        parser.error("--min-start-north-offset-m must be non-negative")
    if args.scene_lock_start_s < 0.0:
        parser.error("--scene-lock-start-s must be non-negative")
    if (
        args.scene_lock_end_s is not None
        and args.scene_lock_end_s <= args.scene_lock_start_s
    ):
        parser.error("--scene-lock-end-s must be greater than --scene-lock-start-s")

    annotation_path, annotation_payload, bounds_times, confidence = load_cutpoint_annotations(
        args.annotations
    )
    raw_times, raw_values = read_recording(
        args.bag,
        args.joint_topic,
        args.bar_topic,
        args.obstacle_topic,
    )
    timestamps, joint_positions, joint_gaps, common_grid_start = build_feature_grid(
        raw_times,
        raw_values,
        args.joint_topic,
        args.feature_hz,
    )
    trajectory = iiwa14_fk(joint_positions)
    environment = BarInspectEnv(
        **load_environment_config(args.env_config, 1.0 / args.feature_hz)
    )

    bar_scene_mask = scene_window_mask(
        raw_times[args.bar_topic],
        common_grid_start,
        args.scene_lock_start_s,
        args.scene_lock_end_s,
    )
    obstacle_scene_mask = scene_window_mask(
        raw_times[args.obstacle_topic],
        common_grid_start,
        args.scene_lock_start_s,
        args.scene_lock_end_s,
    )
    bar_lock = robust_static_pose(
        raw_values[args.bar_topic][bar_scene_mask],
        position_floor_m=args.scene_position_outlier_m,
        local_axis=environment.bar_axis_local,
        axis_floor_rad=np.deg2rad(args.bar_axis_outlier_deg),
    )
    obstacle_lock = robust_static_pose(
        raw_values[args.obstacle_topic][obstacle_scene_mask],
        position_floor_m=args.scene_position_outlier_m,
    )
    demo_obstacle_locks = lock_demo_obstacle_poses(
        raw_times[args.obstacle_topic],
        raw_values[args.obstacle_topic],
        common_grid_start,
        bounds_times,
        args.demo_obstacle_cluster_radius_m,
    )
    demo_scenes = [
        BarInspectScene(
            bar_pose_optitrack=bar_lock["pose"],
            obstacle_pose_optitrack=lock["pose"],
        )
        for lock in demo_obstacle_locks
    ]
    demo_start_north_offsets = validate_demo_start_positions(
        timestamps,
        trajectory,
        bounds_times,
        environment,
        demo_scenes,
        args.min_start_north_offset_m,
    )
    feature_names = np.asarray(
        [spec["name"] for spec in environment.get_feature_schema()]
    )
    if args.only_annotated:
        annotated, bounds, selected_joint_gaps = concatenate_annotated_demos(
            timestamps,
            joint_positions,
            joint_gaps,
            trajectory,
            bounds_times,
            environment,
            demo_scenes,
            args.max_joint_gap,
        )
        output_timestamps = annotated["timestamps"]
        output_joint_positions = annotated["joint_positions"]
        output_trajectory = annotated["trajectory"]
        features = annotated["features"]
        demo_id = annotated["demo_id"]
        stage_labels = annotated["coarse_stage_labels"]
    else:
        if float(np.max(joint_gaps)) > float(args.max_joint_gap):
            raise RuntimeError(
                f"Joint-state gap too large for {args.feature_hz:g} Hz processing: "
                f"{np.max(joint_gaps):.6f}s"
            )
        output_timestamps = timestamps
        output_joint_positions = joint_positions
        output_trajectory = trajectory
        bounds, demo_id, stage_labels = annotation_arrays(timestamps, bounds_times)
        obstacle_pose_trace = np.repeat(
            obstacle_lock["pose"][None, :],
            len(trajectory),
            axis=0,
        )
        for row, lock in zip(bounds, demo_obstacle_locks):
            obstacle_pose_trace[int(row[0]) : int(row[-1])] = lock["pose"]
        feature_scene = BarInspectScene(
            bar_pose_optitrack=bar_lock["pose"],
            obstacle_pose_optitrack=obstacle_pose_trace,
        )
        features = environment.compute_all_features_matrix(
            trajectory,
            scene=feature_scene,
        )
        selected_joint_gaps = joint_gaps

    feature_rate_arrays = {
        "schema_version": np.asarray(3, dtype=np.int64),
        "timestamps": output_timestamps,
        "joint_positions": output_joint_positions,
        "trajectory": output_trajectory,
        "features": features,
        "feature_names": feature_names,
        "locked_bar_pose": bar_lock["pose"],
        "recording_reference_obstacle_pose": obstacle_lock["pose"],
        "demo_obstacle_poses": np.asarray(
            [lock["pose"] for lock in demo_obstacle_locks],
            dtype=float,
        ),
        "demo_id": demo_id,
        "coarse_stage_labels": stage_labels,
        "coarse_bounds_indices": bounds,
        "coarse_bounds_times_s": bounds_times,
        "demo_start_confidence": np.asarray(confidence),
        "demo_start_segmentation_basis": np.asarray(
            annotation_payload.get("demo_start_segmentation_basis", "")
        ),
        "demo_start_north_offset_m": demo_start_north_offsets,
        "minimum_demo_start_north_offset_m": np.asarray(
            args.min_start_north_offset_m
        ),
        "raw_joint_state_hz": np.asarray(
            1.0 / np.median(np.diff(raw_times[args.joint_topic]))
        ),
        "feature_computation_hz": np.asarray(args.feature_hz),
        "downsample_hz": np.asarray(args.feature_hz),
        "source_bag": np.asarray(args.bag.name),
        "source_annotations": np.asarray(annotation_path.name),
        "cutpoint_annotation_kind": np.asarray(annotation_payload["kind"]),
        "cutpoint_evaluation_role": np.asarray(
            annotation_payload.get("evaluation_role", "heuristic_reference")
        ),
        "cutpoint_confidence": np.asarray(
            annotation_payload.get("heuristic_cutpoint_confidence", [])
        ),
        "cutpoint_segmentation_basis": np.asarray(
            annotation_payload.get("segmentation_basis", [])
        ),
        "bar_scene_policy": np.asarray(
            "windowed_robust_static_lock"
            if args.scene_lock_end_s is not None or args.scene_lock_start_s > 0.0
            else "whole_recording_robust_static_lock"
        ),
        "obstacle_scene_policy": np.asarray(
            "per_demo_dominant_static_position_cluster"
        ),
        "scene_lock_window_s": np.asarray(
            [
                args.scene_lock_start_s,
                np.nan if args.scene_lock_end_s is None else args.scene_lock_end_s,
            ],
            dtype=float,
        ),
        "export_policy": np.asarray(
            "annotated_intervals_only" if args.only_annotated else "whole_recording"
        ),
        "bar_scene_inlier_count": np.asarray(np.count_nonzero(bar_lock["inlier"])),
        "bar_scene_sample_count": np.asarray(len(bar_lock["inlier"])),
        "demo_obstacle_scene_inlier_count": np.asarray(
            [np.count_nonzero(lock["inlier"]) for lock in demo_obstacle_locks],
            dtype=np.int64,
        ),
        "demo_obstacle_scene_sample_count": np.asarray(
            [lock["sample_count"] for lock in demo_obstacle_locks],
            dtype=np.int64,
        ),
        "demo_obstacle_position_p95_m": np.asarray(
            [lock["position_p95_m"] for lock in demo_obstacle_locks],
            dtype=float,
        ),
        "bar_position_gate_m": np.asarray(bar_lock["position_limit_m"]),
        "demo_obstacle_cluster_radius_m": np.asarray(
            args.demo_obstacle_cluster_radius_m
        ),
        "bar_axis_gate_rad": np.asarray(np.deg2rad(args.bar_axis_outlier_deg)),
        "max_joint_resampling_gap_s": np.asarray(np.max(selected_joint_gaps)),
        "recording_max_joint_resampling_gap_s": np.asarray(np.max(joint_gaps)),
    }

    if args.feature_output is not None:
        args.feature_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.feature_output, **feature_rate_arrays)

    output_arrays, selected_indices = downsample_processed_arrays(
        feature_rate_arrays,
        factor,
    )
    output_arrays["downsample_hz"] = np.asarray(args.output_hz)
    output_arrays["downsample_factor"] = np.asarray(factor, dtype=np.int64)
    output_arrays["source_sample_count"] = np.asarray(
        len(output_timestamps),
        dtype=np.int64,
    )
    output_arrays["derivative_feature_policy"] = np.asarray(
        "compute_at_feature_computation_hz_then_subsample_complete_rows"
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_arrays)
    print(
        f"feature_samples={len(output_timestamps)} output_samples={len(selected_indices)} "
        f"feature_hz={args.feature_hz:g} output_hz={args.output_hz:g} "
        f"bar_inliers={np.count_nonzero(bar_lock['inlier'])}/{len(bar_lock['inlier'])} "
        f"demo_obstacle_inliers="
        f"{','.join(str(np.count_nonzero(lock['inlier'])) for lock in demo_obstacle_locks)} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
