#!/usr/bin/env python3
"""Suggest BarInspect demo starts from high-rate ROS bag joint states."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.BarInspect import BarInspectEnv, BarInspectScene  # noqa: E402
from experiments.bar_inspect_processing import (  # noqa: E402
    load_cutpoint_annotations,
    robust_static_pose,
)
from experiments.extract_bar_inspection_rosbag import (  # noqa: E402
    BAR_TOPIC,
    JOINT_TOPIC,
    OBSTACLE_TOPIC,
    iiwa14_fk,
    load_environment_config,
    read_recording,
    scene_window_mask,
)


def odd_window_samples(duration_s, sample_dt, point_count, minimum=5):
    samples = max(int(minimum), int(round(float(duration_s) / float(sample_dt))))
    if samples % 2 == 0:
        samples += 1
    maximum = point_count if point_count % 2 == 1 else point_count - 1
    return min(samples, maximum)


def merge_true_runs(mask, timestamps, maximum_gap_s):
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    split_points = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = [group for group in np.split(indices, split_points) if len(group)]
    merged = [runs[0]]
    for run in runs[1:]:
        gap = float(timestamps[run[0]] - timestamps[merged[-1][-1]])
        if gap <= float(maximum_gap_s):
            merged[-1] = np.concatenate([merged[-1], run])
        else:
            merged.append(run)
    return merged


def stationary_dwell_suggestion(
    timestamps,
    smoothed_positions,
    speed,
    obstacle_y,
    search_start_s,
    stage2_s,
    minimum_stage1_s,
    minimum_north_offset_m,
    stationary_speed_m_s,
    minimum_dwell_s,
    merge_gap_s,
):
    latest_start_s = float(stage2_s) - float(minimum_stage1_s)
    search = (
        (timestamps >= float(search_start_s))
        & (timestamps <= latest_start_s)
        & (smoothed_positions[:, 1] - float(obstacle_y) >= float(minimum_north_offset_m))
    )
    stationary = search & (speed <= float(stationary_speed_m_s))
    qualifying = []
    for run in merge_true_runs(stationary, timestamps, merge_gap_s):
        duration = float(timestamps[run[-1]] - timestamps[run[0]])
        if duration >= float(minimum_dwell_s):
            qualifying.append((run, duration))
    if qualifying:
        run, duration = max(qualifying, key=lambda item: timestamps[item[0][-1]])
        return {
            "suggested_start_s": float(timestamps[run[-1]]),
            "method": "stationary_dwell_end",
            "confidence": "high",
            "dwell_start_s": float(timestamps[run[0]]),
            "dwell_end_s": float(timestamps[run[-1]]),
            "dwell_duration_s": duration,
        }

    fallback = np.flatnonzero(search)
    if len(fallback) == 0:
        return {
            "suggested_start_s": None,
            "method": "no_north_candidate",
            "confidence": "manual_review",
        }
    fallback_index = fallback[int(np.argmax(smoothed_positions[fallback, 1]))]
    return {
        "suggested_start_s": float(timestamps[fallback_index]),
        "method": "north_position_maximum_fallback",
        "confidence": "manual_review",
    }


def rounded(value, digits):
    if value is None:
        return None
    return round(float(value), int(digits))


def build_diagnostics(args):
    print("[1/4] Reading high-rate ROS bag topics...", flush=True)
    annotation_path, _, bounds_times, _ = load_cutpoint_annotations(args.annotations)
    raw_times, raw_values = read_recording(
        args.bag,
        args.joint_topic,
        args.bar_topic,
        args.obstacle_topic,
    )

    common_start = max(values[0] for values in raw_times.values())
    feature_dt = 1.0 / float(args.feature_hz)
    common_grid_start = math.ceil(common_start / feature_dt) * feature_dt
    timestamps = raw_times[args.joint_topic] - common_grid_start
    sample_dt = float(np.median(np.diff(timestamps)))
    raw_hz = 1.0 / sample_dt

    print(
        f"[2/4] Computing FK for {len(timestamps)} joint samples at ~{raw_hz:.1f} Hz...",
        flush=True,
    )
    trajectory = iiwa14_fk(raw_values[args.joint_topic])
    position_window = odd_window_samples(
        args.position_smoothing_s,
        sample_dt,
        len(timestamps),
    )
    smoothed_positions = savgol_filter(
        trajectory[:, :3],
        position_window,
        3,
        axis=0,
    )
    velocity = np.gradient(smoothed_positions, timestamps, axis=0)
    speed_window = odd_window_samples(
        args.speed_smoothing_s,
        sample_dt,
        len(timestamps),
    )
    speed = savgol_filter(np.linalg.norm(velocity, axis=1), speed_window, 2)

    environment = BarInspectEnv(
        **load_environment_config(args.env_config, feature_dt)
    )
    bar_mask = scene_window_mask(
        raw_times[args.bar_topic],
        common_grid_start,
        args.scene_lock_start_s,
        args.scene_lock_end_s,
    )
    obstacle_mask = scene_window_mask(
        raw_times[args.obstacle_topic],
        common_grid_start,
        args.scene_lock_start_s,
        args.scene_lock_end_s,
    )
    bar_lock = robust_static_pose(
        raw_values[args.bar_topic][bar_mask],
        position_floor_m=args.scene_position_outlier_m,
        local_axis=environment.bar_axis_local,
        axis_floor_rad=np.deg2rad(args.bar_axis_outlier_deg),
    )
    obstacle_lock = robust_static_pose(
        raw_values[args.obstacle_topic][obstacle_mask],
        position_floor_m=args.scene_position_outlier_m,
    )
    scene = BarInspectScene(
        bar_pose_optitrack=bar_lock["pose"],
        obstacle_pose_optitrack=obstacle_lock["pose"],
    )
    obstacle_y = float(
        environment._obstacle_center_trace(trajectory[:1], scene=scene)[0, 1]
    )

    print("[3/4] Locating north-side stationary dwells...", flush=True)
    suggestions = []
    previous_end = None
    for demo_index, row in enumerate(bounds_times):
        current_start, stage2 = (float(value) for value in row[:2])
        lookback_start = current_start - float(args.search_lookback_s)
        search_start = max(0.0, lookback_start)
        if previous_end is not None:
            search_start = max(search_start, float(previous_end))
        suggestion = stationary_dwell_suggestion(
            timestamps,
            smoothed_positions,
            speed,
            obstacle_y,
            search_start,
            stage2,
            args.minimum_stage1_s,
            args.minimum_north_offset_m,
            args.stationary_speed_m_s,
            args.minimum_dwell_s,
            args.merge_gap_s,
        )
        suggested_start = suggestion["suggested_start_s"]
        if suggested_start is not None:
            suggested_start = round(suggested_start, args.round_digits)
            sample_index = int(np.argmin(np.abs(timestamps - suggested_start)))
            north_offset = float(smoothed_positions[sample_index, 1] - obstacle_y)
            stage1_duration = stage2 - suggested_start
        else:
            north_offset = None
            stage1_duration = None
        suggestions.append(
            {
                "demo_index": int(demo_index),
                "current_start_s": current_start,
                "stage2_s": stage2,
                "search_start_s": rounded(search_start, args.round_digits + 2),
                **{
                    key: rounded(value, args.round_digits + 2)
                    if key.endswith("_s") and value is not None
                    else value
                    for key, value in suggestion.items()
                },
                "suggested_start_s": suggested_start,
                "suggested_start_north_offset_m": rounded(north_offset, 4),
                "suggested_stage1_duration_s": rounded(stage1_duration, 3),
            }
        )
        previous_end = float(row[-1])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json = args.output_dir / "demo_start_suggestions.json"
    payload = {
        "source_bag": str(Path(args.bag)),
        "source_annotations": str(annotation_path),
        "time_reference": (
            "seconds from the first sample on the common feature-processing grid"
        ),
        "raw_joint_state_hz": raw_hz,
        "obstacle_y_robot_m": obstacle_y,
        "parameters": {
            "feature_hz_for_time_reference": args.feature_hz,
            "position_smoothing_s": args.position_smoothing_s,
            "speed_smoothing_s": args.speed_smoothing_s,
            "stationary_speed_m_s": args.stationary_speed_m_s,
            "minimum_dwell_s": args.minimum_dwell_s,
            "merge_gap_s": args.merge_gap_s,
            "minimum_north_offset_m": args.minimum_north_offset_m,
            "minimum_stage1_s": args.minimum_stage1_s,
        },
        "warning": (
            "Suggestions are diagnostics only. Inspect the plot and edit the annotation "
            "JSON manually; do not apply them blindly."
        ),
        "demos": suggestions,
    }
    output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    rows = int(math.ceil(len(suggestions) / 2))
    figure, axes = plt.subplots(rows, 2, figsize=(16, 3.6 * rows), squeeze=False)
    for demo, ax in zip(suggestions, axes.flat):
        current_start = float(demo["current_start_s"])
        suggested_start = demo["suggested_start_s"]
        stage2 = float(demo["stage2_s"])
        plot_start = min(current_start, suggested_start or current_start) - 2.0
        mask = (timestamps >= plot_start) & (timestamps <= stage2 + 0.25)
        ax.plot(timestamps[mask], smoothed_positions[mask, 1], color="C0", label="TCP y")
        ax.axhline(obstacle_y, color="C0", linestyle=":", alpha=0.55, label="obstacle y")
        ax.axvline(current_start, color="C3", linestyle="--", label="current start")
        if suggested_start is not None:
            ax.axvline(
                suggested_start,
                color="C2",
                linestyle="--",
                label="suggested start",
            )
        ax.axvline(stage2, color="0.2", linestyle="--", label="Stage 2")
        speed_axis = ax.twinx()
        speed_axis.plot(timestamps[mask], speed[mask], color="C1", alpha=0.6, label="speed")
        speed_axis.axhline(args.stationary_speed_m_s, color="C1", linestyle=":", alpha=0.5)
        speed_axis.set_ylim(
            -0.005,
            max(0.22, float(np.percentile(speed[mask], 99)) * 1.05),
        )
        ax.set_title(
            f"Demo {demo['demo_index']}: {demo['method']} ({demo['confidence']})"
        )
        ax.set_xlabel("bag-relative time (s)")
        ax.set_ylabel("robot-base y (m)")
        speed_axis.set_ylabel("TCP speed (m/s)")
        ax.grid(alpha=0.2)
    for ax in axes.flat[len(suggestions):]:
        ax.set_visible(False)
    handles = axes.flat[0].get_lines() + axes.flat[0].twinx().get_lines()
    figure.suptitle("High-rate BarInspect demo-start diagnostics")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    output_plot = args.output_dir / "demo_start_diagnostics.png"
    figure.savefig(output_plot, dpi=160)
    plt.close(figure)

    print(f"[4/4] Wrote {output_json} and {output_plot}", flush=True)
    for demo in suggestions:
        print(
            f"demo={demo['demo_index']:02d} current={demo['current_start_s']:.1f} "
            f"suggested={demo['suggested_start_s']} method={demo['method']} "
            f"confidence={demo['confidence']}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect raw high-rate FK and suggest reset-free BarInspect demo starts. "
            "The script never modifies annotations."
        )
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--env-config",
        type=Path,
        default=PROJECT_ROOT / "configs/envs/BarInspect.json",
    )
    parser.add_argument("--joint-topic", default=JOINT_TOPIC)
    parser.add_argument("--bar-topic", default=BAR_TOPIC)
    parser.add_argument("--obstacle-topic", default=OBSTACLE_TOPIC)
    parser.add_argument("--feature-hz", type=float, default=10.0)
    parser.add_argument("--scene-lock-start-s", type=float, default=0.0)
    parser.add_argument("--scene-lock-end-s", type=float, default=60.0)
    parser.add_argument("--scene-position-outlier-m", type=float, default=0.020)
    parser.add_argument("--bar-axis-outlier-deg", type=float, default=5.0)
    parser.add_argument("--position-smoothing-s", type=float, default=0.40)
    parser.add_argument("--speed-smoothing-s", type=float, default=0.25)
    parser.add_argument("--stationary-speed-m-s", type=float, default=0.030)
    parser.add_argument("--minimum-dwell-s", type=float, default=0.30)
    parser.add_argument("--merge-gap-s", type=float, default=0.15)
    parser.add_argument("--minimum-north-offset-m", type=float, default=0.030)
    parser.add_argument("--minimum-stage1-s", type=float, default=3.0)
    parser.add_argument("--search-lookback-s", type=float, default=30.0)
    parser.add_argument("--round-digits", type=int, default=1)
    args = parser.parse_args()

    positive = {
        "--feature-hz": args.feature_hz,
        "--position-smoothing-s": args.position_smoothing_s,
        "--speed-smoothing-s": args.speed_smoothing_s,
        "--stationary-speed-m-s": args.stationary_speed_m_s,
        "--minimum-dwell-s": args.minimum_dwell_s,
        "--minimum-stage1-s": args.minimum_stage1_s,
        "--search-lookback-s": args.search_lookback_s,
    }
    for name, value in positive.items():
        if value <= 0.0:
            parser.error(f"{name} must be positive")
    if args.merge_gap_s < 0.0 or args.minimum_north_offset_m < 0.0:
        parser.error("--merge-gap-s and --minimum-north-offset-m must be non-negative")
    if args.round_digits < 0:
        parser.error("--round-digits must be non-negative")
    if args.scene_lock_end_s <= args.scene_lock_start_s:
        parser.error("--scene-lock-end-s must be greater than --scene-lock-start-s")
    build_diagnostics(args)


if __name__ == "__main__":
    main()
