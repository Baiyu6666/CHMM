#!/usr/bin/env python3
"""Plot and summarize stage-wise planar speeds for S3ObsAvoidReal."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.registry import load_env


COLORS = ("#D55E00", "#0072B2", "#009E73")


def smooth_speed(trajectory, time, window=31):
    count = min(int(window), len(trajectory) - (1 - len(trajectory) % 2))
    count += 1 - count % 2
    smooth = np.column_stack([
        savgol_filter(trajectory[:, axis], count, 3) for axis in range(2)
    ])
    return np.linalg.norm(np.gradient(smooth, time, axis=0), axis=1)


def stage_slices(cutpoints, length):
    return (
        slice(0, int(cutpoints[0]) + 1),
        slice(int(cutpoints[0]) + 1, int(cutpoints[1]) + 1),
        slice(int(cutpoints[1]) + 1, int(length)),
    )


def interior(values, fraction=0.1):
    margin = int(round(float(fraction) * len(values)))
    return values[margin:-margin] if margin > 0 and len(values) > 2 * margin else values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/S3ObsAvoidReal_stage_speeds.png"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/analysis/S3ObsAvoidReal_stage_speeds.json"))
    args = parser.parse_args()
    bundle = load_env("S3ObsAvoidReal")
    speeds = [smooth_speed(x, t) for x, t in zip(bundle.demos, bundle.meta["times"])]

    fig = plt.figure(figsize=(15, 16))
    grid = fig.add_gridspec(5, 2, height_ratios=(1, 1, 1, 1, 1.15), hspace=0.38, wspace=0.18)
    time_axes = [fig.add_subplot(grid[row, column]) for row in range(4) for column in range(2)]
    time_axes[-1].axis("off")
    box_axis = fig.add_subplot(grid[4, 0])
    overlay_axis = fig.add_subplot(grid[4, 1])

    normalized = [[] for _ in range(3)]
    stage_values = [[] for _ in range(3)]
    per_demo = []
    target_progress = np.linspace(0.0, 1.0, 150)
    for demo_index, (trajectory, time, speed, cutpoints) in enumerate(
        zip(bundle.demos, bundle.meta["times"], speeds, bundle.true_cutpoints)
    ):
        axis = time_axes[demo_index]
        slices = stage_slices(cutpoints, len(trajectory))
        demo_record = {"demo": demo_index + 1, "stages": []}
        for stage_index, segment in enumerate(slices):
            local_time = time[segment]
            local_speed = speed[segment]
            axis.plot(local_time, 1000.0 * local_speed, color=COLORS[stage_index], linewidth=1.0)
            progress = np.linspace(0.0, 1.0, len(local_speed))
            normalized[stage_index].append(np.interp(target_progress, progress, local_speed))
            core = interior(local_speed, 0.1)
            stage_values[stage_index].extend(core.tolist())
            demo_record["stages"].append({
                "stage": stage_index + 1,
                "median_mm_s": float(1000.0 * np.median(core)),
                "mean_mm_s": float(1000.0 * np.mean(core)),
                "std_mm_s": float(1000.0 * np.std(core)),
                "p95_mm_s": float(1000.0 * np.percentile(core, 95)),
                "coefficient_of_variation": float(np.std(core) / max(np.mean(core), 1e-12)),
            })
        for cutpoint in cutpoints:
            axis.axvline(time[int(cutpoint)], color="0.25", linestyle="--", linewidth=0.8)
        axis.set_title(f"Demo {demo_index + 1}")
        axis.set_xlabel("Time from trimmed demo start [s]")
        axis.set_ylabel("Planar speed [mm/s]")
        axis.grid(alpha=0.2)
        per_demo.append(demo_record)

    for stage_index in range(3):
        values = np.asarray(normalized[stage_index]) * 1000.0
        mean = values.mean(axis=0)
        overlay_axis.plot(100.0 * target_progress, mean, color=COLORS[stage_index],
                          linewidth=2.0, label=f"Stage {stage_index + 1}")
        overlay_axis.fill_between(100.0 * target_progress,
                                  np.percentile(values, 25, axis=0),
                                  np.percentile(values, 75, axis=0),
                                  color=COLORS[stage_index], alpha=0.16)
    overlay_axis.set_title("Across-demo stage speed (mean and interquartile band)")
    overlay_axis.set_xlabel("Normalized stage progress [%]")
    overlay_axis.set_ylabel("Planar speed [mm/s]")
    overlay_axis.grid(alpha=0.2)
    overlay_axis.legend()

    positions = np.arange(1, 4)
    box_values = [1000.0 * np.asarray(values) for values in stage_values]
    box = box_axis.boxplot(box_values, positions=positions, widths=0.55, showfliers=False, patch_artist=True)
    for patch, color in zip(box["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    box_axis.set_xticks(positions, ["Stage 1", "Stage 2", "Stage 3"])
    box_axis.set_ylabel("Planar speed [mm/s]")
    box_axis.set_title("Interior 80% speed distributions")
    box_axis.grid(axis="y", alpha=0.2)

    pooled = []
    for stage_index, values in enumerate(stage_values):
        values = np.asarray(values)
        pooled.append({
            "stage": stage_index + 1,
            "median_mm_s": float(1000.0 * np.median(values)),
            "mean_mm_s": float(1000.0 * np.mean(values)),
            "std_mm_s": float(1000.0 * np.std(values)),
            "p05_mm_s": float(1000.0 * np.percentile(values, 5)),
            "p95_mm_s": float(1000.0 * np.percentile(values, 95)),
            "coefficient_of_variation": float(np.std(values) / max(np.mean(values), 1e-12)),
        })
    summary = {"method": "31-frame Savitzky-Golay smoothing followed by time derivative",
               "boundary_trim_fraction": 0.1, "pooled": pooled, "per_demo": per_demo}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("S3ObsAvoidReal planar end-effector speed by stage", fontsize=15)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
