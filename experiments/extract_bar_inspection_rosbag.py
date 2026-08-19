#!/usr/bin/env python3
"""Convert one BarInsepect ROS1 bag into synchronized poses and features."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import rosbag


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.BarInsepect import BarInsepectEnv  # noqa: E402


BAR_TOPIC = "/vrpn_client_node/baiyu_bar/pose_from_iiwa14"
OBSTACLE_TOPIC = "/vrpn_client_node/baiyu_obs_ball/pose_from_iiwa14"


def transform_matrix(transform):
    q = transform.transform.rotation
    p = transform.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [p.x, p.y, p.z]
    return matrix


def rotation_quaternion(rotation):
    """Return a numerically stable XYZW quaternion for a rotation matrix."""
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]))
            quaternion = np.array(
                [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            )
        elif index == 1:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]))
            quaternion = np.array(
                [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            )
        else:
            scale = 2.0 * math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]))
            quaternion = np.array(
                [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
            )
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Could not convert an EE rotation to a finite quaternion")
    return quaternion / norm


def message_stamp(message, bag_stamp):
    stamp = message.header.stamp.to_sec()
    return stamp if stamp > 0.0 else bag_stamp.to_sec()


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


def nearest_indices(reference_times, query_times):
    reference = np.asarray(reference_times, dtype=float)
    query = np.asarray(query_times, dtype=float)
    right = np.searchsorted(reference, query, side="left")
    right = np.clip(right, 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    use_left = np.abs(query - reference[left]) <= np.abs(reference[right] - query)
    indices = np.where(use_left, left, right)
    return indices, np.abs(query - reference[indices])


def quaternion_matrices(quaternions):
    """Convert finite XYZW quaternions to rotation matrices."""
    quaternion = np.asarray(quaternions, dtype=float)
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    quaternion = quaternion / np.where(norms > 1e-12, norms, 1.0)
    x, y, z, w = quaternion.T
    matrices = np.empty((len(quaternion), 3, 3), dtype=float)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - z * w)
    matrices[:, 0, 2] = 2.0 * (x * z + y * w)
    matrices[:, 1, 0] = 2.0 * (x * y + z * w)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - x * w)
    matrices[:, 2, 0] = 2.0 * (x * z - y * w)
    matrices[:, 2, 1] = 2.0 * (y * z + x * w)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def average_quaternion(quaternions):
    """Average XYZW quaternions without being affected by q/-q sign flips."""
    quaternion = np.asarray(quaternions, dtype=float)
    quaternion = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    scatter = np.einsum("ti,tj->ij", quaternion, quaternion)
    _, eigenvectors = np.linalg.eigh(scatter)
    result = eigenvectors[:, -1]
    if result[3] < 0.0:
        result = -result
    return result / np.linalg.norm(result)


def robust_static_pose(
    poses,
    position_floor_m,
    local_axis=None,
    axis_floor_rad=None,
):
    """Estimate one fixed pose while rejecting minority OptiTrack jumps.

    Position is estimated by a component-wise median and a median/MAD radial
    gate.  For the bar, a second gate rejects samples whose tracked local +X
    direction disagrees with the robust direction.  The returned pose is then
    repeated for all feature calculations by the caller.
    """
    pose = np.asarray(poses, dtype=float)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError("Static-scene poses must have shape (samples, 7).")

    quaternion_norm = np.linalg.norm(pose[:, 3:7], axis=1)
    valid = (
        np.all(np.isfinite(pose), axis=1)
        & np.isfinite(quaternion_norm)
        & (quaternion_norm > 1e-12)
    )
    if not np.any(valid):
        raise ValueError("No finite OptiTrack poses are available for scene locking.")

    initial_position = np.median(pose[valid, :3], axis=0)
    position_error = np.linalg.norm(pose[:, :3] - initial_position[None, :], axis=1)
    valid_error = position_error[valid]
    median_error = float(np.median(valid_error))
    mad = float(np.median(np.abs(valid_error - median_error)))
    position_limit = max(
        float(position_floor_m),
        median_error + 6.0 * 1.4826 * max(mad, 1e-9),
    )
    position_inlier = valid & (position_error <= position_limit)

    minimum_inliers = max(1, int(math.ceil(0.5 * np.count_nonzero(valid))))
    if np.count_nonzero(position_inlier) < minimum_inliers:
        raise ValueError(
            "OptiTrack scene lock rejected at least half of the finite position samples."
        )

    axis_error = np.full(len(pose), np.nan, dtype=float)
    inlier = position_inlier.copy()
    if local_axis is not None:
        axis = np.asarray(local_axis, dtype=float).reshape(3)
        axis /= np.linalg.norm(axis)
        rotations = quaternion_matrices(pose[:, 3:7])
        tracked_axes = np.einsum("tij,j->ti", rotations, axis)
        robust_axis = np.median(tracked_axes[position_inlier], axis=0)
        robust_axis /= np.linalg.norm(robust_axis)
        axis_error[valid] = np.arccos(
            np.clip(tracked_axes[valid] @ robust_axis, -1.0, 1.0)
        )
        inlier &= axis_error <= float(axis_floor_rad)
        if np.count_nonzero(inlier) < minimum_inliers:
            raise ValueError(
                "OptiTrack scene lock rejected at least half of the finite bar-axis samples."
            )

    locked_pose = np.concatenate(
        [
            np.median(pose[inlier, :3], axis=0),
            average_quaternion(pose[inlier, 3:7]),
        ]
    )
    final_position_error = np.linalg.norm(
        pose[:, :3] - locked_pose[None, :3], axis=1
    )
    return {
        "pose": locked_pose,
        "inlier": inlier,
        "position_error_m": final_position_error,
        "position_limit_m": position_limit,
        "axis_error_rad": axis_error,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("output", help="Output .npz file")
    parser.add_argument("--robot-name", default="iiwa14")
    parser.add_argument("--bar-topic", default=BAR_TOPIC)
    parser.add_argument("--obstacle-topic", default=OBSTACLE_TOPIC)
    parser.add_argument("--max-sync-gap", type=float, default=0.05)
    parser.add_argument(
        "--scene-position-outlier-m",
        type=float,
        default=0.020,
        help="Minimum radial gate used when locking each static OptiTrack object.",
    )
    parser.add_argument(
        "--bar-axis-outlier-deg",
        type=float,
        default=5.0,
        help="Reject bar samples whose local +X direction differs from the robust direction.",
    )
    args = parser.parse_args()

    if args.scene_position_outlier_m <= 0.0:
        parser.error("--scene-position-outlier-m must be positive")
    if not 0.0 < args.bar_axis_outlier_deg < 180.0:
        parser.error("--bar-axis-outlier-deg must be between 0 and 180 degrees")

    ee_times = []
    trajectories = []
    bar_times, bar_poses = [], []
    obstacle_times, obstacle_poses = [], []

    topics = ["/tf", args.bar_topic, args.obstacle_topic]
    with rosbag.Bag(args.bag) as bag:
        for topic, message, bag_stamp in bag.read_messages(topics=topics):
            if topic == args.bar_topic:
                bar_times.append(message_stamp(message, bag_stamp))
                bar_poses.append(pose_row(message))
                continue
            if topic == args.obstacle_topic:
                obstacle_times.append(message_stamp(message, bag_stamp))
                obstacle_poses.append(pose_row(message))
                continue

            links = {item.child_frame_id: item for item in message.transforms}
            if not all(
                f"{args.robot_name}_link_{index}" in links for index in range(1, 8)
            ):
                continue
            matrix = np.eye(4)
            for index in range(1, 8):
                matrix = matrix @ transform_matrix(
                    links[f"{args.robot_name}_link_{index}"]
                )
            stamp = message.transforms[0].header.stamp.to_sec() or bag_stamp.to_sec()
            ee_times.append(stamp)
            trajectories.append(
                [*matrix[:3, 3], *rotation_quaternion(matrix[:3, :3])]
            )

    if not trajectories:
        raise RuntimeError(f"No complete {args.robot_name} link chain found in /tf")
    if not bar_poses:
        raise RuntimeError(f"No bar poses found on {args.bar_topic}")
    if not obstacle_poses:
        raise RuntimeError(f"No obstacle poses found on {args.obstacle_topic}")

    ee_times = np.asarray(ee_times, dtype=float)
    trajectory = np.asarray(trajectories, dtype=float)
    bar_times = np.asarray(bar_times, dtype=float)
    bar_poses = np.asarray(bar_poses, dtype=float)
    obstacle_times = np.asarray(obstacle_times, dtype=float)
    obstacle_poses = np.asarray(obstacle_poses, dtype=float)

    bar_indices, bar_gaps = nearest_indices(bar_times, ee_times)
    obstacle_indices, obstacle_gaps = nearest_indices(obstacle_times, ee_times)
    keep = (bar_gaps <= args.max_sync_gap) & (obstacle_gaps <= args.max_sync_gap)
    if not np.any(keep):
        raise RuntimeError(
            "No EE samples have both bar and obstacle poses within the synchronization limit"
        )

    trajectory = trajectory[keep]
    timestamps = ee_times[keep] - ee_times[keep][0]
    bar_pose = bar_poses[bar_indices[keep]]
    obstacle_pose = obstacle_poses[obstacle_indices[keep]]

    environment = BarInsepectEnv()
    bar_lock = robust_static_pose(
        bar_pose,
        args.scene_position_outlier_m,
        local_axis=environment.bar_axis_local,
        axis_floor_rad=np.deg2rad(args.bar_axis_outlier_deg),
    )
    obstacle_lock = robust_static_pose(
        obstacle_pose,
        args.scene_position_outlier_m,
    )
    locked_bar_pose_trace = np.repeat(
        bar_lock["pose"][None, :], len(trajectory), axis=0
    )
    locked_obstacle_pose_trace = np.repeat(
        obstacle_lock["pose"][None, :], len(trajectory), axis=0
    )
    environment.register_bar_pose_trace(trajectory, locked_bar_pose_trace)
    environment.register_obstacle_pose_trace(trajectory, locked_obstacle_pose_trace)
    features = environment.compute_all_features_matrix(trajectory)
    feature_names = np.asarray(
        [item["name"] for item in environment.get_feature_schema()]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        trajectory=trajectory,
        timestamps=timestamps,
        bar_pose=locked_bar_pose_trace,
        obstacle_pose=locked_obstacle_pose_trace,
        bar_pose_raw=bar_pose,
        obstacle_pose_raw=obstacle_pose,
        locked_bar_pose=bar_lock["pose"],
        locked_obstacle_pose=obstacle_lock["pose"],
        bar_pose_used_for_features=locked_bar_pose_trace,
        obstacle_pose_used_for_features=locked_obstacle_pose_trace,
        bar_scene_inlier=bar_lock["inlier"],
        obstacle_scene_inlier=obstacle_lock["inlier"],
        bar_position_error_m=bar_lock["position_error_m"],
        obstacle_position_error_m=obstacle_lock["position_error_m"],
        bar_axis_error_rad=bar_lock["axis_error_rad"],
        bar_position_gate_m=np.asarray(bar_lock["position_limit_m"]),
        obstacle_position_gate_m=np.asarray(obstacle_lock["position_limit_m"]),
        bar_axis_gate_rad=np.asarray(np.deg2rad(args.bar_axis_outlier_deg)),
        feature_scene_policy=np.asarray("whole_demo_robust_static_lock"),
        features=features,
        feature_names=feature_names,
        bar_sync_gap_s=bar_gaps[keep],
        obstacle_sync_gap_s=obstacle_gaps[keep],
        source_bag=np.asarray(str(Path(args.bag).resolve())),
    )
    print(
        f"samples={len(trajectory)} duration={timestamps[-1]:.3f}s "
        f"max_bar_gap={bar_gaps[keep].max():.6f}s "
        f"max_obstacle_gap={obstacle_gaps[keep].max():.6f}s "
        f"bar_inliers={np.count_nonzero(bar_lock['inlier'])}/{len(trajectory)} "
        f"obstacle_inliers={np.count_nonzero(obstacle_lock['inlier'])}/{len(trajectory)} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
