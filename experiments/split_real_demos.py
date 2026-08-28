#!/usr/bin/env python3
"""Split continuously recorded real demonstrations using elevated transfer motions."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


def contiguous_runs(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [[int(start), int(end)] for start, end in zip(starts, ends)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--height-threshold", type=float, default=0.40)
    parser.add_argument("--smooth-window", type=int, default=101)
    parser.add_argument("--min-transfer-duration", type=float, default=0.15)
    parser.add_argument("--merge-gap", type=float, default=0.8)
    args = parser.parse_args()

    with args.csv_path.open(newline="") as stream:
        header = next(csv.reader(stream))
    data = np.genfromtxt(args.csv_path, delimiter=",", skip_header=1)
    time = data[:, 0]
    xyz = data[:, 1:4]
    window = min(args.smooth_window, len(data) - (1 - len(data) % 2))
    window += 1 - window % 2
    smooth_xyz = np.column_stack([savgol_filter(xyz[:, i], window, 3) for i in range(3)])
    velocity = np.gradient(smooth_xyz, time, axis=0)
    speed = np.linalg.norm(velocity, axis=1)

    runs = [run for run in contiguous_runs(smooth_xyz[:, 2] > args.height_threshold)
            if time[run[1]] - time[run[0]] >= args.min_transfer_duration]
    transfers = []
    for start, end in runs:
        if transfers and time[start] - time[transfers[-1][1]] < args.merge_gap:
            transfers[-1][1] = end
        else:
            transfers.append([start, end])

    # The first and last elevated runs are recording setup/cleanup; all elevated
    # runs are excluded and their low-height complements are demonstrations.
    demos = []
    cursor = 0
    for start, end in transfers:
        if start > cursor:
            demos.append([cursor, start - 1])
        cursor = end + 1
    if cursor < len(data):
        demos.append([cursor, len(data) - 1])
    demos = [[start, end] for start, end in demos if time[end] - time[start] >= 5.0]

    args.outdir.mkdir(parents=True, exist_ok=True)
    demo_records = []
    for demo_id, (start, end) in enumerate(demos):
        segment = data[start:end + 1].copy()
        global_time = segment[:, 0].copy()
        segment[:, 0] -= segment[0, 0]
        output = args.outdir / f"demo_{demo_id:02d}.csv"
        output_header = ["time_s", "recording_time_s", *header[1:]]
        np.savetxt(output, np.column_stack([segment[:, 0], global_time, segment[:, 1:]]),
                   delimiter=",", header=",".join(output_header), comments="", fmt="%.10g")
        demo_records.append({
            "demo_id": demo_id,
            "start_time_s": float(time[start]),
            "end_time_s": float(time[end]),
            "duration_s": float(time[end] - time[start]),
            "frames": int(end - start + 1),
            "file": output.name,
        })

    transfer_records = [{
        "start_time_s": float(time[start]),
        "end_time_s": float(time[end]),
        "duration_s": float(time[end] - time[start]),
        "peak_z_m": float(np.max(smooth_xyz[start:end + 1, 2])),
    } for start, end in transfers]
    manifest = {
        "source": str(args.csv_path),
        "status": "automatic_candidate_boundaries_require_visual_review",
        "method": "Savitzky-Golay-smoothed end-effector z above threshold",
        "height_threshold_m": args.height_threshold,
        "sample_count": int(len(data)),
        "recording_duration_s": float(time[-1] - time[0]),
        "transfers": transfer_records,
        "demonstrations": demo_records,
    }
    (args.outdir / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    fig = plt.figure(figsize=(14, 9))
    axes = [fig.add_subplot(3, 1, 1)]
    axes.append(fig.add_subplot(3, 1, 2, sharex=axes[0]))
    axes.append(fig.add_subplot(3, 1, 3))
    axes[0].plot(time, xyz[:, 2], color="0.75", linewidth=0.6, label="raw z")
    axes[0].plot(time, smooth_xyz[:, 2], color="C0", linewidth=1.3, label="smoothed z")
    axes[0].axhline(args.height_threshold, color="C3", linestyle="--", label="transfer threshold")
    axes[0].set_ylabel("EE z [m]")
    axes[0].legend(loc="upper left")
    axes[1].plot(time, speed, color="C1", linewidth=0.8)
    axes[1].set_ylabel("EE speed [m/s]")
    axes[2].plot(smooth_xyz[:, 0], smooth_xyz[:, 1], color="0.7", linewidth=0.7)
    for record, (start, end) in zip(demo_records, demos):
        for axis in axes[:2]:
            axis.axvspan(time[start], time[end], alpha=0.12, label=None)
            axis.text((time[start] + time[end]) / 2, axis.get_ylim()[1], f"D{record['demo_id']}",
                      ha="center", va="top", fontsize=9)
        axes[2].plot(smooth_xyz[start:end + 1, 0], smooth_xyz[start:end + 1, 1],
                     linewidth=1.4, label=f"D{record['demo_id']}")
    axes[2].set_xlabel("EE x [m]")
    axes[2].set_ylabel("EE y [m]")
    axes[2].axis("equal")
    axes[2].legend(ncol=7, fontsize=8)
    axes[1].set_xlabel("Recording time [s]")
    axes[0].tick_params(labelbottom=False)
    fig.tight_layout()
    fig.savefig(args.outdir / "split_diagnostics.png", dpi=180)
    plt.close(fig)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
