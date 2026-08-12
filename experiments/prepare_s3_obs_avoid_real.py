#!/usr/bin/env python3
"""Prepare the reviewed real S3 demonstrations and estimate their shared geometry."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution


# Manually reviewed setup/cleanup trims for this specific continuous recording.
# Keeping them fixed ensures later stage-boundary refinements do not silently
# change which physical motion belongs to each demonstration.
REVIEWED_DEMO_BOUNDS = (
    (186, 1019),
    (82, 989),
    (83, 1142),
    (393, 1251),
    (158, 1266),
    (60, 1004),
    (213, 921),
)

# Previously reviewed stage-2-to-stage-3 boundaries. These remain fixed while
# refining only the stage-1-to-stage-2 line-entry semantics.
REVIEWED_CP2_INDICES = (814, 700, 866, 1087, 1065, 664, 696)

# Zero-based candidate demo IDs retained after manual quality review.  Demos
# 0, 2, and 3 were rejected due to large systematic tracking errors.
RETAINED_DEMO_INDICES = (1, 4, 5, 6)


def smooth_xy(xy, window=31):
    n = min(int(window), len(xy) - (1 - len(xy) % 2))
    n += 1 - n % 2
    return np.column_stack([savgol_filter(xy[:, axis], n, 3) for axis in range(2)])


def fit_line(points, trim_quantile=0.85, rounds=4):
    kept = np.asarray(points, dtype=float)
    for _ in range(rounds):
        point = kept.mean(axis=0)
        _, _, vectors = np.linalg.svd(kept - point, full_matrices=False)
        direction = vectors[0]
        normal = np.array([-direction[1], direction[0]])
        residual = np.abs((kept - point) @ normal)
        kept = kept[residual <= np.quantile(residual, trim_quantile)]
    point = kept.mean(axis=0)
    _, _, vectors = np.linalg.svd(kept - point, full_matrices=False)
    direction = vectors[0]
    if direction[0] < 0:
        direction = -direction
    normal = np.array([-direction[1], direction[0]])
    return point, direction, float(np.sqrt(np.mean(((kept - point) @ normal) ** 2)))


def fit_circle(points, rounds=6, trim_quantile=0.80):
    kept = np.asarray(points, dtype=float)
    for _ in range(rounds):
        matrix = np.column_stack([2 * kept[:, 0], 2 * kept[:, 1], np.ones(len(kept))])
        rhs = np.sum(kept ** 2, axis=1)
        solution = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        center = solution[:2]
        radius = float(np.sqrt(max(solution[2] + center @ center, 0.0)))
        residual = np.abs(np.linalg.norm(kept - center, axis=1) - radius)
        kept = kept[residual <= np.quantile(residual, trim_quantile)]
    return center, radius, float(np.sqrt(np.mean((np.linalg.norm(kept - center, axis=1) - radius) ** 2)))


def fit_maximum_clearance_obstacle(points):
    """Estimate the largest circular obstacle avoided by every stage-1 trace."""
    samples = np.asarray(points, dtype=float)
    bounds = [(0.60, 0.67), (-0.005, 0.065)]
    result = differential_evolution(
        lambda center: -float(np.min(np.linalg.norm(samples - center, axis=1))),
        bounds=bounds, seed=0, tol=1e-10, polish=True,
    )
    center = np.asarray(result.x, dtype=float)
    observed_clearance = float(np.min(np.linalg.norm(samples - center, axis=1)))
    # One millimetre prevents TF/smoothing noise from turning boundary samples
    # into numerical violations of the estimated physical obstacle.
    radius = max(0.0, observed_clearance - 0.001)
    return center, radius, observed_clearance


def low_speed_trim_start(time, xy, stage1_end):
    speed = np.linalg.norm(np.gradient(xy, time, axis=0), axis=1)
    low = speed < 0.015
    min_samples = max(5, int(round(0.35 / np.median(np.diff(time)))))
    # Search only in the early 65% of stage 1. Pauses near the first line are
    # legitimate stage transitions and must not be mistaken for setup motion.
    latest_allowed = max(1, int(round(0.65 * stage1_end)))
    candidates = []
    start = None
    for index, value in enumerate(low[:latest_allowed]):
        if value and start is None:
            start = index
        if start is not None and (not value or index == latest_allowed - 1):
            end = index if value else index - 1
            if end - start + 1 >= min_samples:
                candidates.append((start, end))
            start = None
    return min((candidates[-1][1] + 1 if candidates else 0), stage1_end - 1)


def resample_stagewise(trajectory, time, cutpoints, target_hz=5.0):
    """Time-resample each stage while retaining both boundary samples exactly."""
    dt = 1.0 / float(target_hz)
    bounds = ((0, int(cutpoints[0])),
              (int(cutpoints[0]), int(cutpoints[1])),
              (int(cutpoints[1]), len(trajectory) - 1))
    output_points = []
    output_times = []
    output_cutpoints = []
    for stage_index, (start, end) in enumerate(bounds):
        source_t = np.asarray(time[start:end + 1], dtype=float)
        source_xy = np.asarray(trajectory[start:end + 1], dtype=float)
        duration = float(source_t[-1] - source_t[0])
        count = max(2, int(round(duration * float(target_hz))) + 1)
        target_t = np.linspace(source_t[0], source_t[-1], count)
        target_xy = np.column_stack([
            np.interp(target_t, source_t, source_xy[:, axis]) for axis in range(2)
        ])
        if stage_index > 0:
            target_t = target_t[1:]
            target_xy = target_xy[1:]
        output_points.append(target_xy)
        output_times.append(target_t)
        if stage_index < 2:
            output_cutpoints.append(sum(len(part) for part in output_points) - 1)
    points = np.vstack(output_points)
    times = np.concatenate(output_times)
    return points, times - times[0], np.asarray(output_cutpoints, dtype=int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--target-hz", type=float, default=5.0)
    args = parser.parse_args()
    files = sorted(args.candidate_dir.glob("demo_*.csv"))
    if len(files) != 7:
        raise ValueError(f"Expected 7 candidate demonstrations, found {len(files)}")

    selected = [
        (original_index, files[original_index], REVIEWED_DEMO_BOUNDS[original_index],
         REVIEWED_CP2_INDICES[original_index])
        for original_index in RETAINED_DEMO_INDICES
    ]
    raw = [np.genfromtxt(path, delimiter=",", skip_header=1) for _, path, _, _ in selected]
    paths = [smooth_xy(item[:, 2:4]) for item in raw]

    line1_seed = np.vstack([path[(path[:, 1] < -0.075) & (path[:, 1] > -0.175) &
                                    (path[:, 0] > 0.59) & (path[:, 0] < 0.67)] for path in paths])
    line2_seed = np.vstack([path[(path[:, 1] < -0.18) & (path[:, 0] > 0.62)] for path in paths])
    line1_point, line1_direction, line1_rms = fit_line(line1_seed)
    line2_point, line2_direction, line2_rms = fit_line(line2_seed)
    normal1 = np.array([-line1_direction[1], line1_direction[0]])
    normal2 = np.array([-line2_direction[1], line2_direction[0]])

    preliminary = []
    for demo_index, (item, path) in enumerate(zip(raw, paths)):
        time = item[:, 0]
        d1 = np.abs((path - line1_point) @ normal1)
        d2 = np.abs((path - line2_point) @ normal2)
        dt = float(np.median(np.diff(time)))
        # A task stage starts only after the end effector has entered and then
        # persistently remains in the corresponding line's narrow tracking band.
        # Use the centre of the persistence window, not its left edge: the latter
        # still contains the lateral approach motion visible before line entry.
        window = max(8, int(round(0.8 / dt)))
        good1 = (d1 < 0.012) & (path[:, 1] < -0.050) & (path[:, 1] > -0.190)
        rate1 = np.convolve(good1.astype(float), np.ones(window) / window, mode="valid")
        candidates1 = np.flatnonzero(rate1 > 0.75)
        cp1 = int(candidates1[0] + window // 2) if len(candidates1) else int(0.35 * len(path))

        _, _, reviewed_bounds, reviewed_cp2 = selected[demo_index]
        cp2 = int(reviewed_cp2)
        start, reviewed_end = reviewed_bounds

        # Remove the final vertical pickup, detected from the rapid terminal z rise.
        z = savgol_filter(item[:, 4], 31, 3)
        dz = np.gradient(z, time)
        pickup = np.flatnonzero((np.arange(len(z)) > cp2) & (dz > 0.06))
        detected_end = max(cp2 + 2, int(pickup[0]) - max(2, int(round(0.15 / dt)))) if len(pickup) else len(path) - 1
        end = int(min(reviewed_end, detected_end))
        preliminary.append((start, end, cp1, cp2))

    circle_points = []
    for path, (start, _, cp1, _) in zip(paths, preliminary):
        margin = max(4, int(0.3 / 0.025))
        circle_points.append(path[start + margin:max(start + margin + 1, cp1 - margin)])
    obstacle_center, obstacle_radius, observed_clearance = fit_maximum_clearance_obstacle(np.vstack(circle_points))

    arrays = {}
    records = []
    for index, (item, path, bounds) in enumerate(zip(raw, paths, preliminary)):
        start, end, cp1, cp2 = bounds
        trajectory = path[start:end + 1]
        times = item[start:end + 1, 0] - item[start, 0]
        cutpoints = np.array([cp1 - start, cp2 - start], dtype=int)
        trajectory, times, cutpoints = resample_stagewise(
            trajectory, times, cutpoints, target_hz=args.target_hz
        )
        labels = np.zeros(len(trajectory), dtype=int)
        labels[cutpoints[0] + 1:cutpoints[1] + 1] = 1
        labels[cutpoints[1] + 1:] = 2
        arrays[f"demo_{index}"] = trajectory
        arrays[f"time_{index}"] = times
        arrays[f"cutpoints_{index}"] = cutpoints
        arrays[f"labels_{index}"] = labels
        original_index, source_file, _, _ = selected[index]
        records.append({
            "demo": index, "original_demo": int(original_index),
            "source_file": source_file.name,
            "candidate_start_index": int(start), "candidate_end_index": int(end),
            "frames": int(len(trajectory)), "duration_s": float(times[-1]),
            "cutpoints": cutpoints.tolist(),
            "cutpoint_times_s": [float(times[value]) for value in cutpoints],
        })

    angle = float(np.degrees(np.arccos(np.clip(abs(line1_direction @ line2_direction), 0.0, 1.0))))
    metadata = {
        "dataset": "S3ObsAvoidReal", "source": "2026-08-12-09-36-21.bag",
        "coordinate_system": "iiwa_link_0_xy_m", "dt_s": 1.0 / float(args.target_hz),
        "sampling_hz": float(args.target_hz),
        "resampling": "stage-wise linear time interpolation after Savitzky-Golay position smoothing",
        "geometry_estimation_status": "estimated_from_demonstrations",
        "obstacle": {"center": obstacle_center.tolist(), "radius_m": obstacle_radius,
                     "minimum_observed_stage1_distance_m": observed_clearance,
                     "estimation": "maximum circle in the common stage-1 trajectory gap with 1 mm margin"},
        "line_1": {"point": line1_point.tolist(), "direction": line1_direction.tolist(), "fit_rms_m": line1_rms},
        "line_2": {"point": line2_point.tolist(), "direction": line2_direction.tolist(), "fit_rms_m": line2_rms},
        "line_angle_degrees": angle, "demonstrations": records,
    }
    arrays["count"] = np.array(len(records), dtype=int)
    arrays["metadata_json"] = np.array(json.dumps(metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
    theta = np.linspace(0, 2 * np.pi, 200)
    colors = ("#D55E00", "#0072B2", "#009E73")
    for index, (axis, path, bounds) in enumerate(zip(axes.flat, paths, preliminary)):
        start, end, cp1, cp2 = bounds
        spans = ((start, cp1), (cp1, cp2), (cp2, end))
        for color, (a, b) in zip(colors, spans):
            axis.plot(path[a:b + 1, 0], path[a:b + 1, 1], color=color, linewidth=1.6)
        axis.plot(obstacle_center[0] + obstacle_radius * np.cos(theta),
                  obstacle_center[1] + obstacle_radius * np.sin(theta), "k--", linewidth=1)
        for point, direction in ((line1_point, line1_direction), (line2_point, line2_direction)):
            extent = np.linspace(-0.25, 0.25, 2)
            line = point + extent[:, None] * direction
            axis.plot(line[:, 0], line[:, 1], color="0.45", linestyle=":", linewidth=1)
        axis.scatter(path[start, 0], path[start, 1], marker="o", color="k", s=12)
        axis.scatter(path[end, 0], path[end, 1], marker="x", color="k", s=18)
        axis.set_title(f"Demo {index} (original {selected[index][0]})")
        axis.set_aspect("equal")
    fig.suptitle(f"S3ObsAvoidReal geometry; line angle={angle:.1f}°")
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
