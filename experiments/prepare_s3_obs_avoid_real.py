#!/usr/bin/env python3
"""Prepare the seven real S3 demonstrations and estimate their shared geometry."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import differential_evolution


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.candidate_dir.glob("demo_*.csv"))
    if len(files) != 7:
        raise ValueError(f"Expected 7 candidate demonstrations, found {len(files)}")

    raw = [np.genfromtxt(path, delimiter=",", skip_header=1) for path in files]
    paths = [smooth_xy(item[:, 2:4]) for item in raw]

    line1_seed = np.vstack([path[(path[:, 1] < -0.075) & (path[:, 1] > -0.175) &
                                    (path[:, 0] > 0.59) & (path[:, 0] < 0.67)] for path in paths])
    line2_seed = np.vstack([path[(path[:, 1] < -0.18) & (path[:, 0] > 0.62)] for path in paths])
    line1_point, line1_direction, line1_rms = fit_line(line1_seed)
    line2_point, line2_direction, line2_rms = fit_line(line2_seed)
    normal1 = np.array([-line1_direction[1], line1_direction[0]])
    normal2 = np.array([-line2_direction[1], line2_direction[0]])

    preliminary = []
    for item, path in zip(raw, paths):
        time = item[:, 0]
        d1 = np.abs((path - line1_point) @ normal1)
        d2 = np.abs((path - line2_point) @ normal2)
        dt = float(np.median(np.diff(time)))
        window = max(8, int(round(1.2 / dt)))
        good1 = (d1 < 0.020) & (path[:, 1] < -0.045)
        rate1 = np.convolve(good1.astype(float), np.ones(window) / window, mode="valid")
        candidates1 = np.flatnonzero(rate1 > 0.75)
        cp1 = int(candidates1[0]) if len(candidates1) else int(0.35 * len(path))

        lo = max(cp1 + window, int(0.45 * len(path)))
        hi = int(0.90 * len(path))
        scores = []
        for cp2 in range(lo, hi):
            scores.append((float(np.mean(d1[cp1:cp2] ** 2) + np.mean(d2[cp2:] ** 2)), cp2))
        cp2 = min(scores)[1]
        start = low_speed_trim_start(time, path, cp1)

        # Remove the final vertical pickup, detected from the rapid terminal z rise.
        z = savgol_filter(item[:, 4], 31, 3)
        dz = np.gradient(z, time)
        pickup = np.flatnonzero((np.arange(len(z)) > cp2) & (dz > 0.06))
        end = max(cp2 + 2, int(pickup[0]) - max(2, int(round(0.15 / dt)))) if len(pickup) else len(path) - 1
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
        labels = np.zeros(len(trajectory), dtype=int)
        labels[cutpoints[0] + 1:cutpoints[1] + 1] = 1
        labels[cutpoints[1] + 1:] = 2
        arrays[f"demo_{index}"] = trajectory
        arrays[f"time_{index}"] = times
        arrays[f"cutpoints_{index}"] = cutpoints
        arrays[f"labels_{index}"] = labels
        records.append({
            "demo": index + 1, "source_file": files[index].name,
            "candidate_start_index": int(start), "candidate_end_index": int(end),
            "frames": int(len(trajectory)), "duration_s": float(times[-1]),
            "cutpoints": cutpoints.tolist(),
            "cutpoint_times_s": [float(times[value]) for value in cutpoints],
        })

    angle = float(np.degrees(np.arccos(np.clip(abs(line1_direction @ line2_direction), 0.0, 1.0))))
    metadata = {
        "dataset": "S3ObsAvoidReal", "source": "2026-08-12-09-36-21.bag",
        "coordinate_system": "iiwa_link_0_xy_m", "dt_s": 0.025,
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

    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
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
        axis.set_title(f"Demo {index + 1}")
        axis.set_aspect("equal")
    axes.flat[-1].axis("off")
    fig.suptitle(f"S3ObsAvoidReal geometry; line angle={angle:.1f}°")
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=180)
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
