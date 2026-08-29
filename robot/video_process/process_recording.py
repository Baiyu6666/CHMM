#!/usr/bin/env python3
"""Discover and render an existing dual-camera iiwa experiment recording."""

import argparse
import csv
import glob
import json
import os
import subprocess
import sys


CAMERA_SUFFIXES = (
    "_cam1.mp4",
    "_cam1_timestamps.csv",
    "_cam2.mp4",
    "_cam2_timestamps.csv",
)


def timestamp_bounds(path):
    first = last = None
    count = 0
    with open(path, newline="") as stream:
        for row in csv.DictReader(stream):
            value = int(row["unix_time_ns"])
            first = value if first is None else first
            last = value
            count += 1
    if first is None or last is None or count < 2:
        raise ValueError("fewer than two timestamp rows: %s" % path)
    return first, last, count


def load_trace_intervals(trace_directory):
    intervals = []
    for path in glob.glob(os.path.join(trace_directory,
                                       "constraint_plan_*.json")):
        if "_sim_" in os.path.basename(path):
            continue
        try:
            with open(path) as stream:
                payload = json.load(stream)
            stamps = [int(sample["unix_time_ns"])
                      for sample in payload.get("samples", [])
                      if sample.get("unix_time_ns") is not None]
            manifold = [sample for sample in payload.get("samples", [])
                        if sample.get("segment") == "following_manifold"]
            if stamps and manifold:
                intervals.append((min(stamps), max(stamps)))
        except (OSError, ValueError, TypeError, KeyError,
                json.JSONDecodeError):
            continue
    return intervals


def trace_overlap_count(intervals, start_ns, end_ns):
    return sum(trace_end >= start_ns and trace_start <= end_ns
               for trace_start, trace_end in intervals)


def discover(recording_directory, trace_directory):
    result = []
    trace_intervals = load_trace_intervals(trace_directory)
    pattern = os.path.join(recording_directory, "*_cam1.mp4")
    for primary_video in sorted(glob.glob(pattern)):
        if primary_video.endswith("_dual_profiles.mp4"):
            continue
        prefix = primary_video[:-len("_cam1.mp4")]
        files = {suffix[1:].replace(".", "_"): prefix + suffix
                 for suffix in CAMERA_SUFFIXES}
        if not all(os.path.isfile(prefix + suffix) and
                   os.path.getsize(prefix + suffix) > 0
                   for suffix in CAMERA_SUFFIXES):
            continue
        try:
            p0, p1, pn = timestamp_bounds(prefix + "_cam1_timestamps.csv")
            a0, a1, an = timestamp_bounds(prefix + "_cam2_timestamps.csv")
            start_ns, end_ns = max(p0, a0), min(p1, a1)
            if end_ns <= start_ns:
                continue
            traces = trace_overlap_count(trace_intervals, start_ns, end_ns)
        except (OSError, ValueError, KeyError):
            continue
        result.append({
            "id": os.path.basename(prefix),
            "prefix": prefix,
            "duration_s": (end_ns - start_ns) / 1e9,
            "primary_frames": pn,
            "auxiliary_frames": an,
            "trace_count": traces,
            "complete": traces > 0,
            "files": files,
        })
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description="Render timestamp-aligned iiwa dual-camera profile video")
    parser.add_argument("--recording-directory", required=True,
                        help="directory containing *_cam1/cam2.mp4 and CSV")
    parser.add_argument("--trace-directory", required=True,
                        help="directory containing constraint_plan_*.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list complete recording groups")
    process = subparsers.add_parser("process", help="render one recording")
    process.add_argument("recording_id",
                         help="recording ID from list, or 'latest'")
    process.add_argument("--output", help="output MP4 path")
    return parser


def main():
    options = build_parser().parse_args()
    recording_directory = os.path.abspath(options.recording_directory)
    trace_directory = os.path.abspath(options.trace_directory)
    recordings = discover(recording_directory, trace_directory)
    if options.command == "list":
        print(json.dumps(recordings, indent=2, ensure_ascii=False))
        return

    complete = [item for item in recordings if item["complete"]]
    if not complete:
        raise SystemExit("no complete dual-camera recording overlaps trace data")
    if options.recording_id == "latest":
        selected = complete[-1]
    else:
        selected = next((item for item in complete
                         if item["id"] == options.recording_id), None)
        if selected is None:
            raise SystemExit("unknown or incomplete recording: %s" %
                             options.recording_id)
    prefix = selected["prefix"]
    output = (os.path.abspath(options.output) if options.output else
              prefix + "_cam1_dual_profiles.mp4")
    renderer = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "render_experiment_profiles.py")
    command = [
        sys.executable, renderer,
        "--video", prefix + "_cam1.mp4",
        "--timestamps", prefix + "_cam1_timestamps.csv",
        "--aux-video", prefix + "_cam2.mp4",
        "--aux-timestamps", prefix + "_cam2_timestamps.csv",
        "--trace-directory", trace_directory,
        "--output", output,
        "--wait-seconds", "0",
    ]
    print("processing %s" % selected["id"], flush=True)
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
