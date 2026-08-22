from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def nearest_indices(reference_times, query_times):
    reference = np.asarray(reference_times, dtype=float)
    query = np.asarray(query_times, dtype=float)
    if reference.ndim != 1 or len(reference) == 0:
        raise ValueError("reference_times must be a non-empty one-dimensional array.")
    right = np.searchsorted(reference, query, side="left")
    right = np.clip(right, 0, len(reference) - 1)
    left = np.clip(right - 1, 0, len(reference) - 1)
    use_left = np.abs(query - reference[left]) <= np.abs(reference[right] - query)
    indices = np.where(use_left, left, right)
    return indices, np.abs(query - reference[indices])


def quaternion_matrices(quaternions):
    quaternion = np.asarray(quaternions, dtype=float)
    norms = np.linalg.norm(quaternion, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("Quaternions must be finite and non-zero.")
    quaternion = quaternion / norms
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
    position_floor_m=0.020,
    local_axis=None,
    axis_floor_rad=None,
):
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
        if axis_floor_rad is None:
            raise ValueError("axis_floor_rad is required when local_axis is provided.")
        axis = np.asarray(local_axis, dtype=float).reshape(3)
        axis /= np.linalg.norm(axis)
        tracked_axes = np.full((len(pose), 3), np.nan, dtype=float)
        tracked_axes[valid] = np.einsum(
            "tij,j->ti",
            quaternion_matrices(pose[valid, 3:7]),
            axis,
        )
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
    return {
        "pose": locked_pose,
        "inlier": inlier,
        "position_error_m": np.linalg.norm(
            pose[:, :3] - locked_pose[None, :3],
            axis=1,
        ),
        "position_limit_m": position_limit,
        "axis_error_rad": axis_error,
    }


def dominant_static_pose_cluster(
    poses,
    position_radius_m=0.005,
    minimum_inlier_fraction=0.25,
):
    pose = np.asarray(poses, dtype=float)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError("Static-scene poses must have shape (samples, 7).")
    if float(position_radius_m) <= 0.0:
        raise ValueError("position_radius_m must be positive.")
    if not 0.0 < float(minimum_inlier_fraction) <= 1.0:
        raise ValueError("minimum_inlier_fraction must be in (0, 1].")

    quaternion_norm = np.linalg.norm(pose[:, 3:7], axis=1)
    valid = (
        np.all(np.isfinite(pose), axis=1)
        & np.isfinite(quaternion_norm)
        & (quaternion_norm > 1e-12)
    )
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        raise ValueError("No finite OptiTrack poses are available for scene locking.")

    positions = pose[valid_indices, :3]
    tree = cKDTree(positions)
    neighbor_counts = tree.query_ball_point(
        positions,
        float(position_radius_m),
        return_length=True,
    )
    seed_position = positions[int(np.argmax(neighbor_counts))]
    cluster = np.linalg.norm(positions - seed_position[None, :], axis=1) <= float(
        position_radius_m
    )
    locked_position = np.median(positions[cluster], axis=0)
    cluster = np.linalg.norm(positions - locked_position[None, :], axis=1) <= float(
        position_radius_m
    )
    inlier = np.zeros(len(pose), dtype=bool)
    inlier[valid_indices[cluster]] = True
    minimum_inliers = max(
        1,
        int(np.ceil(float(minimum_inlier_fraction) * len(valid_indices))),
    )
    if np.count_nonzero(inlier) < minimum_inliers:
        raise ValueError(
            "Dominant OptiTrack position cluster contains fewer than the required "
            f"{float(minimum_inlier_fraction):.0%} of finite samples."
        )

    locked_pose = np.concatenate(
        [
            np.median(pose[inlier, :3], axis=0),
            average_quaternion(pose[inlier, 3:7]),
        ]
    )
    position_error = np.linalg.norm(
        pose[:, :3] - locked_pose[None, :3],
        axis=1,
    )
    return {
        "pose": locked_pose,
        "inlier": inlier,
        "position_error_m": position_error,
        "position_limit_m": float(position_radius_m),
        "position_p95_m": float(np.quantile(position_error[inlier], 0.95)),
    }


def load_cutpoint_annotations(path):
    annotation_path = Path(path).expanduser().resolve()
    with annotation_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if payload.get("kind") != "heuristic_coarse_stage_boundaries":
        raise ValueError("BarInspect annotations must declare heuristic coarse boundaries.")
    if payload.get("is_ground_truth") is not False:
        raise ValueError("BarInspect heuristic annotations must set is_ground_truth=false.")
    bounds = np.asarray(payload.get("bounds_times_s"), dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 5:
        raise ValueError("bounds_times_s must have shape (num_demos, 5).")
    if np.any(~np.isfinite(bounds)) or np.any(np.diff(bounds, axis=1) <= 0.0):
        raise ValueError("Every annotation row must contain five increasing finite times.")
    confidence = list(payload.get("demo_start_confidence", []))
    if confidence and len(confidence) != len(bounds):
        raise ValueError("demo_start_confidence must align with bounds_times_s.")
    return annotation_path, payload, bounds, confidence


def annotation_arrays(timestamps, bounds_times_s):
    time = np.asarray(timestamps, dtype=float)
    bounds_time = np.asarray(bounds_times_s, dtype=float)
    bounds = np.searchsorted(time, bounds_time, side="left").astype(np.int64)
    if np.any(bounds < 0) or np.any(bounds > len(time)):
        raise ValueError("Annotation times fall outside the processed recording.")
    demo_id = np.full(len(time), -1, dtype=np.int64)
    labels = np.full(len(time), -1, dtype=np.int64)
    for demo_index, row in enumerate(bounds):
        begin, stage2, stage3, stage4, end = (int(value) for value in row)
        if not (0 <= begin < stage2 < stage3 < stage4 < end <= len(time)):
            raise ValueError(
                f"Invalid mapped annotation bounds for demo {demo_index}: {row.tolist()}"
            )
        demo_id[begin:end] = int(demo_index)
        labels[begin:stage2] = 0
        labels[stage2:stage3] = 1
        labels[stage3:stage4] = 2
        labels[stage4:end] = 3
    return bounds, demo_id, labels


def downsample_indices_by_runs(run_ids, factor):
    stride = int(factor)
    if stride < 1:
        raise ValueError("factor must be a positive integer.")
    values = np.asarray(run_ids)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("run_ids must be a non-empty one-dimensional array.")
    if stride == 1:
        return np.arange(len(values), dtype=np.int64)
    run_starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    run_ends = np.r_[run_starts[1:], len(values)]
    selected = []
    for start, end in zip(run_starts, run_ends):
        local = np.arange(int(start), int(end), stride, dtype=np.int64)
        last = int(end) - 1
        if local[-1] != last:
            local = np.r_[local, last]
        selected.append(local)
    return np.unique(np.concatenate(selected))


def downsample_processed_arrays(arrays, factor):
    source = {key: np.asarray(value) for key, value in arrays.items()}
    point_count = len(source["timestamps"])
    selected = downsample_indices_by_runs(source["demo_id"], factor)
    pointwise = {
        key
        for key, value in source.items()
        if value.ndim >= 1 and value.shape[0] == point_count
    }
    output = {
        key: (value[selected] if key in pointwise else value.copy())
        for key, value in source.items()
    }
    output["coarse_bounds_indices"] = np.searchsorted(
        selected,
        source["coarse_bounds_indices"],
        side="left",
    ).astype(np.int64)
    return output, selected


__all__ = [
    "annotation_arrays",
    "dominant_static_pose_cluster",
    "downsample_indices_by_runs",
    "downsample_processed_arrays",
    "load_cutpoint_annotations",
    "nearest_indices",
    "robust_static_pose",
]
