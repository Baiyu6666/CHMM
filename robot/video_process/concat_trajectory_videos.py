#!/usr/bin/env python3
"""Concatenate rendered run clips with a still pause at both ends of each run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-pause-seconds", type=float, default=1.0)
    parser.add_argument("--end-pause-seconds", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=15)
    return parser


def ffprobe(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,nb_frames:format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise RuntimeError(f"no video stream found in {path}")
    stream = streams[0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": str(stream.get("avg_frame_rate", "0/1")),
        "duration": float(payload.get("format", {}).get("duration", 0.0)),
        "frames": int(stream.get("nb_frames") or 0),
    }


def build_filter(
    video_count: int,
    width: int,
    height: int,
    fps: float,
    start_pause_seconds: float,
    end_pause_seconds: float,
) -> str:
    if video_count <= 0:
        raise ValueError("video_count must be positive")
    parts = []
    labels = []
    for index in range(video_count):
        label = f"v{index}"
        filters = [
            f"fps={float(fps):.6f}",
            f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=decrease",
            f"pad={int(width)}:{int(height)}:(ow-iw)/2:(oh-ih)/2",
            "setsar=1",
            "setpts=PTS-STARTPTS",
        ]
        tpad = ["tpad=start_mode=clone", "stop_mode=clone"]
        tpad.append(f"start_duration={max(0.0, float(start_pause_seconds)):.6f}")
        tpad.append(f"stop_duration={max(0.0, float(end_pause_seconds)):.6f}")
        filters.append(":".join(tpad))
        parts.append(f"[{index}:v]{','.join(filters)}[{label}]")
        labels.append(f"[{label}]")
    parts.append("".join(labels) + f"concat=n={video_count}:v=1:a=0[outv]")
    return ";".join(parts)


def concatenate(
    videos: Sequence[Path],
    output: Path,
    *,
    start_pause_seconds: float,
    end_pause_seconds: float,
    fps: float,
    crf: int,
) -> dict:
    resolved = [path.expanduser().resolve() for path in videos]
    missing = [path for path in resolved if not path.is_file()]
    if missing:
        raise RuntimeError(
            "missing rendered input videos:\n" + "\n".join(str(path) for path in missing)
        )
    first = ffprobe(resolved[0])
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]
    for path in resolved:
        command.extend(["-i", str(path)])
    command.extend(
        [
            "-filter_complex",
            build_filter(
                len(resolved),
                first["width"],
                first["height"],
                fps,
                start_pause_seconds,
                end_pause_seconds,
            ),
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    subprocess.run(command, check=True)
    result = ffprobe(output)
    return {
        "ok": True,
        "output": str(output),
        "videos": [str(path) for path in resolved],
        "start_pause_seconds": float(start_pause_seconds),
        "end_pause_seconds": float(end_pause_seconds),
        **result,
    }


def main() -> None:
    options = build_parser().parse_args()
    result = concatenate(
        options.videos,
        options.output,
        start_pause_seconds=options.start_pause_seconds,
        end_pause_seconds=options.end_pause_seconds,
        fps=options.fps,
        crf=options.crf,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
