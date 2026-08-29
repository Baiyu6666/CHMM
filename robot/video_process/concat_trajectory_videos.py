#!/usr/bin/env python3
"""Concatenate selected processed trajectory videos with short still pauses."""

import argparse
import json
import subprocess
from pathlib import Path


DEFAULT_DATA_ROOT = Path("kuka_experiment_data/learned_from_human")
DEFAULT_ORDER = [4, 1, 7, 13, 12, 11]
DEFAULT_INPUT_NAME = "manifold_cam1_dual_profiles.mp4"
DEFAULT_OUTPUT = DEFAULT_DATA_ROOT / "selected_trajectories_dual_profiles.mp4"


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="directory containing trajectory_XXX folders")
    parser.add_argument("--order", type=int, nargs="+", default=DEFAULT_ORDER,
                        help="trajectory numbers to concatenate, in order")
    parser.add_argument("--input-name", default=DEFAULT_INPUT_NAME,
                        help="processed video filename inside each trajectory folder")
    parser.add_argument("--pause-seconds", type=float, default=2.0,
                        help="still-frame pause inserted after each non-final clip")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output montage video path")
    parser.add_argument("--crf", type=int, default=18,
                        help="libx264 CRF for the concatenated output")
    return parser.parse_args()


def trajectory_path(data_root, number, input_name):
    return data_root / ("trajectory_%03d" % number) / input_name


def ffprobe(path):
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,duration,nb_frames",
        "-of", "json", str(path)]
    result = subprocess.run(command, check=True, text=True,
                            stdout=subprocess.PIPE)
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError("no video stream found in %s" % path)
    return streams[0]


def build_filter(video_count, width, height, fps, pause_seconds):
    parts = []
    labels = []
    for index in range(video_count):
        label = "v%d" % index
        filters = [
            "fps=%s" % fps,
            "scale=%d:%d:force_original_aspect_ratio=decrease" % (width, height),
            "pad=%d:%d:(ow-iw)/2:(oh-ih)/2" % (width, height),
            "setsar=1",
            "setpts=PTS-STARTPTS",
        ]
        if index < video_count - 1 and pause_seconds > 0:
            filters.append("tpad=stop_mode=clone:stop_duration=%.6f" %
                           pause_seconds)
        parts.append("[%d:v]%s[%s]" % (index, ",".join(filters), label))
        labels.append("[%s]" % label)
    parts.append("".join(labels) + "concat=n=%d:v=1:a=0[outv]" % video_count)
    return ";".join(parts)


def main():
    options = args()
    videos = [trajectory_path(options.data_root, number, options.input_name)
              for number in options.order]
    missing = [path for path in videos if not path.is_file()]
    if missing:
        raise SystemExit("missing input videos:\n%s" %
                         "\n".join(str(path) for path in missing))

    first_stream = ffprobe(videos[0])
    width = int(first_stream["width"])
    height = int(first_stream["height"])
    fps = first_stream.get("avg_frame_rate", "30/1")

    options.output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in videos:
        command.extend(["-i", str(path)])
    command.extend([
        "-filter_complex",
        build_filter(len(videos), width, height, fps, options.pause_seconds),
        "-map", "[outv]",
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", str(options.crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(options.output),
    ])
    subprocess.run(command, check=True)

    output_stream = ffprobe(options.output)
    print(json.dumps({
        "ok": True,
        "output": str(options.output),
        "order": options.order,
        "pause_seconds": options.pause_seconds,
        "width": int(output_stream["width"]),
        "height": int(output_stream["height"]),
        "fps": output_stream.get("avg_frame_rate"),
        "duration": float(output_stream.get("duration", 0.0)),
        "frames": int(output_stream.get("nb_frames", 0)),
    }, ensure_ascii=True))


if __name__ == "__main__":
    main()
