#!/usr/bin/env python3
"""Overlay human demonstration profiles and an auxiliary camera on video."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from render_experiment_profiles import (
    DEFAULT_AUX_HORIZONTAL_CROP_FRACTION,
    DEFAULT_AUX_MAX_HEIGHT_RATIO,
    DEFAULT_AUX_SCALE,
    DEFAULT_PROFILE_PANEL_SCALE,
    DEFAULT_SIDEBAR_WIDTH_RATIO,
    DEFAULT_VALUE_UPDATE_SECONDS,
    draw_profile_line,
    format_playback_label,
    load_frame_times,
    numeric_visible_count,
    open_writer,
    panel_rectangle,
    place_auxiliary,
    profile_fonts,
)


DEFAULT_DEMO_ROOT = Path("kuka_experiment_data/human demonstration")
DEFAULT_VIDEO = DEFAULT_DEMO_ROOT / "experiment_20260721_152914_689_cam1.mp4"
DEFAULT_TIMESTAMPS = (
    DEFAULT_DEMO_ROOT / "experiment_20260721_152914_689_cam1_timestamps.csv")
DEFAULT_AUX_VIDEO = (
    DEFAULT_DEMO_ROOT / "experiment_20260721_152914_689_cam2.mp4")
DEFAULT_AUX_TIMESTAMPS = (
    DEFAULT_DEMO_ROOT / "experiment_20260721_152914_689_cam2_timestamps.csv")
DEFAULT_DEMO_DATA = Path(
    "datasets/real_iiwa/tasks/RealKuka/processed/train_sets/"
    "human_demo_paper_toolz_arc3mm_cropped_v001.npz")
DEFAULT_OUTPUT = DEFAULT_DEMO_ROOT / "human_demo_dual_profiles.mp4"

PROFILE_GROUPS = [
    ("Position error to true constraint", "mm", 1000.0, 5.0,
     ("position_error_m",), "Human", (90, 96, 108, 255), "solid"),
    ("Orientation error to true constraint", "deg", 1.0, 13.0,
     ("orientation_error_deg",), "Human", (90, 96, 108, 255), "solid"),
    ("End-effector speed", "mm/s", 1000.0, 180.0,
     ("display_speed_mps",), "Speed", (99, 102, 241, 255), "solid"),
]


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--timestamps", type=Path, default=DEFAULT_TIMESTAMPS)
    parser.add_argument("--aux-video", type=Path, default=DEFAULT_AUX_VIDEO)
    parser.add_argument("--aux-timestamps", type=Path,
                        default=DEFAULT_AUX_TIMESTAMPS)
    parser.add_argument("--demo-data", type=Path, default=DEFAULT_DEMO_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--value-update-seconds", type=float,
                        default=DEFAULT_VALUE_UPDATE_SECONDS)
    parser.add_argument("--sidebar-width-ratio", type=float,
                        default=DEFAULT_SIDEBAR_WIDTH_RATIO)
    parser.add_argument("--aux-horizontal-crop-fraction", type=float,
                        default=DEFAULT_AUX_HORIZONTAL_CROP_FRACTION)
    parser.add_argument("--aux-scale", type=float, default=DEFAULT_AUX_SCALE)
    parser.add_argument("--profile-panel-scale", type=float,
                        default=DEFAULT_PROFILE_PANEL_SCALE)
    parser.add_argument("--max-frame-fraction", type=float, default=None,
                        help="render only this fraction of main-camera frames")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="render at most this many main-camera frames")
    parser.add_argument("--start-elapsed-seconds", type=float, default=None,
                        help="first source-video elapsed second to include")
    parser.add_argument("--end-elapsed-seconds", type=float, default=None,
                        help="source-video elapsed second where rendering stops")
    parser.add_argument("--speedup-start-elapsed-seconds", type=float,
                        default=None,
                        help="source-video elapsed second where speedup begins")
    parser.add_argument("--speedup-factor", type=float, default=5.0,
                        help="play source after speedup start this many times faster")
    return parser.parse_args()


def load_demo_samples(path):
    with np.load(path, allow_pickle=False) as data:
        times = np.asarray(data["unix_time_ns"], dtype=np.int64)
        position = np.asarray(data["position_world"], dtype=np.float64)
        samples = {
            "unix_time_ns": times,
            "position_error_m": np.asarray(data["position_error_m"],
                                           dtype=np.float64),
            "orientation_error_deg": np.asarray(data["orientation_error_deg"],
                                                dtype=np.float64),
        }
    if len(times) >= 2:
        seconds = times.astype(np.float64) * 1e-9
        speed = np.zeros(len(times), dtype=np.float64)
        delta_t = np.maximum(np.diff(seconds), 1e-9)
        segment_speed = np.linalg.norm(np.diff(position, axis=0),
                                       axis=1) / delta_t
        speed[1:] = segment_speed
        speed[0] = segment_speed[0]
        samples["display_speed_mps"] = speed
    else:
        samples["display_speed_mps"] = np.zeros(len(times), dtype=np.float64)
    return samples


def clip_demo_samples(samples, start_ns, end_ns):
    times = samples["unix_time_ns"]
    mask = (times >= int(start_ns)) & (times <= int(end_ns))
    if not np.any(mask):
        raise SystemExit("no demonstration samples overlap rendered video range")
    return {key: np.asarray(value)[mask] for key, value in samples.items()}


def sample_value(samples, key, count):
    if count <= 0:
        return np.nan
    values = samples.get(key)
    if values is None or count > len(values):
        return np.nan
    return float(values[count - 1])


def build_frame_schedule(frame_times, fps, options):
    if not frame_times:
        return []
    if (options.start_elapsed_seconds is None and
            options.end_elapsed_seconds is None and
            options.speedup_start_elapsed_seconds is None):
        schedule = list(range(len(frame_times)))
    else:
        elapsed = ((np.asarray(frame_times, dtype=np.float64) -
                    float(frame_times[0])) * 1e-9)
        start = (float(options.start_elapsed_seconds)
                 if options.start_elapsed_seconds is not None else 0.0)
        end = (float(options.end_elapsed_seconds)
               if options.end_elapsed_seconds is not None else float(elapsed[-1]))
        start = max(0.0, start)
        end = min(float(elapsed[-1]), max(start, end))
        speedup_start = (float(options.speedup_start_elapsed_seconds)
                         if options.speedup_start_elapsed_seconds is not None
                         else end)
        speedup_start = min(max(start, speedup_start), end)
        speedup_factor = max(1e-6, float(options.speedup_factor))
        step = 1.0 / max(float(fps), 1e-6)

        source_elapsed = []
        if speedup_start > start:
            source_elapsed.extend(np.arange(start, speedup_start, step))
        if end > speedup_start:
            fast_output_duration = (end - speedup_start) / speedup_factor
            output_elapsed = np.arange(0.0, fast_output_duration, step)
            source_elapsed.extend(speedup_start + output_elapsed * speedup_factor)
        if not source_elapsed:
            source_elapsed = [start]

        schedule = []
        for source_time in source_elapsed:
            index = int(np.searchsorted(elapsed, source_time, side="left"))
            if index >= len(elapsed):
                index = len(elapsed) - 1
            schedule.append(index)

    if options.max_frame_fraction is not None:
        fraction = float(np.clip(options.max_frame_fraction, 0.0, 1.0))
        if fraction > 0:
            limit = max(1, int(round(len(schedule) * fraction)))
            schedule = schedule[:limit]
    if options.max_frames is not None and options.max_frames > 0:
        schedule = schedule[:int(options.max_frames)]
    return schedule


def playback_label_for_timestamp(frame_times, timestamp_ns, options):
    elapsed = (float(timestamp_ns) - float(frame_times[0])) * 1e-9
    if (options.speedup_start_elapsed_seconds is not None and
            elapsed >= float(options.speedup_start_elapsed_seconds)):
        return format_playback_label(options.speedup_factor)
    return "1 X"


def draw_profiles(frame, auxiliary_frame, samples, timestamp_ns,
                  value_update_seconds=DEFAULT_VALUE_UPDATE_SECONDS,
                  sidebar_width_ratio=DEFAULT_SIDEBAR_WIDTH_RATIO,
                  aux_horizontal_crop_fraction=DEFAULT_AUX_HORIZONTAL_CROP_FRACTION,
                  aux_scale=DEFAULT_AUX_SCALE,
                  profile_panel_scale=DEFAULT_PROFILE_PANEL_SCALE,
                  playback_label=None):
    height, width = frame.shape[:2]
    canvas = frame.copy()
    sidebar_ratio = float(np.clip(sidebar_width_ratio, 0.25, 0.6))
    panel_width = min(width - 20, max(430, int(round(width * sidebar_ratio))))
    auxiliary_margin = max(8, int(round(height / 90.0)))
    scale = float(np.clip(height / 720.0, .85, 1.45))
    panel_margin = int(8 * scale)
    profile_panel_width = int(round(
        (panel_width - 2 * panel_margin) *
        float(np.clip(profile_panel_scale, 0.65, 1.0))))

    auxiliary_top = auxiliary_margin
    aux_width = int(round(profile_panel_width *
                          float(np.clip(aux_scale, 0.8, 1.15)) /
                          DEFAULT_PROFILE_PANEL_SCALE))
    if auxiliary_frame is not None and auxiliary_frame.size:
        source_h, source_w = auxiliary_frame.shape[:2]
        aux_aspect = max(0.2, source_w / max(source_h, 1))
    else:
        aux_aspect = 4.0 / 3.0
    auxiliary_height = int(round(aux_width / aux_aspect))
    max_aux_height = int(round(height * DEFAULT_AUX_MAX_HEIGHT_RATIO))
    if auxiliary_height > max_aux_height:
        auxiliary_height = max_aux_height
        aux_width = int(round(auxiliary_height * aux_aspect))
    auxiliary_bottom = auxiliary_top + auxiliary_height
    auxiliary_left = width - panel_margin - aux_width
    place_auxiliary(
        canvas, auxiliary_frame, auxiliary_left, auxiliary_top,
        auxiliary_left + aux_width, auxiliary_bottom,
        aux_horizontal_crop_fraction)

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image, "RGBA")
    font_size = max(13, int(round(14 * scale)))
    title_size = max(16, int(round(19 * scale)))
    font, font_bold, font_title = profile_fonts(font_size, title_size)
    if playback_label:
        playback_font_size = max(34, int(round(40 * scale)))
        try:
            playback_font = profile_fonts(playback_font_size,
                                          playback_font_size)[1]
        except (OSError, IOError):
            playback_font = font_bold
        draw.text((int(24 * scale), height - int(62 * scale)),
                  playback_label, fill=(255, 255, 255, 255),
                  font=playback_font)

    panel_right = width - panel_margin
    panel_left = panel_right - profile_panel_width
    panel_top = auxiliary_bottom + panel_margin
    panel_bottom = height - int(10 * scale)
    x0, x1 = panel_left + int(18 * scale), panel_right - int(18 * scale)
    panel_rectangle(draw, (panel_left, panel_top, panel_right, panel_bottom),
                    radius=max(6, int(7 * scale)), fill=(255, 255, 255, 238),
                    outline=(36, 42, 50, 190), width=max(1, int(scale)))

    header_y = panel_top + int(12 * scale)
    draw.text((x0, header_y), "Human demonstration profiles",
              fill=(20, 24, 30, 255), font=font_title)

    sample_times = samples["unix_time_ns"]
    if timestamp_ns < sample_times[0]:
        visible_count = 0
        numeric_count = 0
        state = "Waiting"
        state_color = (105, 110, 120, 255)
    else:
        visible_count = int(np.searchsorted(sample_times, timestamp_ns,
                                            side="right"))
        visible_count = min(max(visible_count, 0), len(sample_times))
        run = {
            "samples": [None] * len(sample_times),
            "manifold_start_ns": int(sample_times[0]),
            "manifold_end_ns": int(sample_times[-1]),
        }
        numeric_count = numeric_visible_count(
            sample_times, run, timestamp_ns, value_update_seconds)
        state = "Recording" if timestamp_ns <= sample_times[-1] else "Completed"
        state_color = ((37, 99, 235, 255) if
                       timestamp_ns <= sample_times[-1] else
                       (80, 85, 95, 255))
    draw.text((x0, header_y + int(36 * scale)), state,
              fill=state_color, font=font_bold)

    top, bottom = header_y + int(62 * scale), height - int(18 * scale)
    gap = int(6 * scale)
    weights = [1.0, 1.0, .68]
    usable_height = bottom - top - gap * (len(PROFILE_GROUPS) - 1)
    row_heights = [int(usable_height * value / sum(weights))
                   for value in weights]
    row_heights[-1] += usable_height - sum(row_heights)
    cursor_indigo = (99, 102, 241, 210)
    ry0 = top
    for row, (title, unit, value_scale, fixed_vmax, keys, label, color,
              style) in enumerate(PROFILE_GROUPS):
        ry1 = ry0 + row_heights[row]
        label_w = int(min(220 * scale, profile_panel_width * .45))
        px0, px1 = x0 + label_w, x1
        py0, py1 = ry0 + int(23 * scale), ry1 - int(7 * scale)
        draw.rectangle((px0, py0, px1, py1), fill=(248, 250, 252, 220),
                       outline=(188, 196, 206, 220), width=max(1, int(scale)))
        draw.text((x0, ry0 + int(2 * scale)), "%s (%s)" % (title, unit),
                  fill=(18, 24, 38, 255), font=font_bold)
        vmin, vmax = 0.0, fixed_vmax
        draw.text((px0 + 3, py0 + 2), "%.1f" % vmax,
                  fill=(105, 110, 120, 255), font=font)

        legend_y = ry0 + int(25 * scale)
        marker_y = legend_y + int(7 * scale)
        draw_profile_line(draw, [(x0, marker_y),
                                 (x0 + int(20 * scale), marker_y)],
                          color, style, max(2, int(2 * scale)))
        current = sample_value(samples, keys[0], numeric_count) * value_scale
        value_text = "-" if not np.isfinite(current) else "%.2f" % current
        draw.text((x0 + int(25 * scale), legend_y),
                  "%s  %s" % (label, value_text), fill=color, font=font)

        values = np.asarray(samples[keys[0]], dtype=np.float64) * value_scale
        if visible_count:
            points = []
            denominator = max(len(values) - 1, 1)
            for index, value in enumerate(values[:visible_count]):
                if not np.isfinite(value):
                    continue
                x = int(round(px0 + index / denominator * (px1 - px0)))
                y = int(round(py1 - (value - vmin) /
                              max(vmax - vmin, 1e-9) * (py1 - py0)))
                points.append((x, max(py0, min(py1, y))))
            if len(points) >= 2:
                draw_profile_line(draw, points, color, style,
                                  max(2, int(2 * scale)))
            elif points:
                radius = max(2, int(3 * scale))
                draw.ellipse((points[0][0] - radius, points[0][1] - radius,
                              points[0][0] + radius, points[0][1] + radius),
                             fill=color)
            cursor_x = int(round(px0 + (visible_count - 1) /
                                 max(len(values) - 1, 1) * (px1 - px0)))
            draw.line((cursor_x, py0, cursor_x, py1), fill=cursor_indigo,
                      width=max(1, int(scale)))
        ry0 = ry1 + gap

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def main():
    options = args()
    frame_times = load_frame_times(options.timestamps)
    auxiliary_times = load_frame_times(options.aux_timestamps)
    if not frame_times:
        raise SystemExit("timestamp CSV contains no frames")
    if not auxiliary_times:
        raise SystemExit("auxiliary timestamp CSV contains no frames")
    samples = load_demo_samples(options.demo_data)
    capture = cv2.VideoCapture(str(options.video))
    auxiliary_capture = cv2.VideoCapture(str(options.aux_video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    schedule = build_frame_schedule(frame_times, fps, options)
    if not schedule:
        raise SystemExit("frame schedule is empty")
    samples = clip_demo_samples(
        samples, frame_times[schedule[0]], frame_times[schedule[-1]])
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    options.output.parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(str(options.output), fps, (width, height))
    if (not capture.isOpened() or not auxiliary_capture.isOpened() or
            not writer.isOpened()):
        raise SystemExit("cannot open input or output video")

    source_index = 0
    schedule_index = 0
    output_frames = 0
    auxiliary_index = -1
    auxiliary_frame = None
    last_source_index = schedule[-1]
    while source_index <= last_source_index:
        ok, frame = capture.read()
        if not ok:
            break
        while (schedule_index < len(schedule) and
               schedule[schedule_index] == source_index):
            timestamp_ns = frame_times[source_index]
            while (auxiliary_index + 1 < len(auxiliary_times) and
                   auxiliary_times[auxiliary_index + 1] <= timestamp_ns):
                aux_ok, candidate = auxiliary_capture.read()
                if not aux_ok:
                    break
                auxiliary_frame = candidate
                auxiliary_index += 1
            writer.write(draw_profiles(
                frame, auxiliary_frame, samples, timestamp_ns,
                options.value_update_seconds,
                options.sidebar_width_ratio,
                options.aux_horizontal_crop_fraction,
                options.aux_scale,
                options.profile_panel_scale,
                playback_label_for_timestamp(frame_times, timestamp_ns,
                                             options)))
            schedule_index += 1
            output_frames += 1
        source_index += 1

    capture.release()
    auxiliary_capture.release()
    writer.release()
    print(json.dumps({
        "ok": True,
        "output": str(options.output),
        "frames": output_frames,
        "source_frames_read": source_index,
        "auxiliary_frames": auxiliary_index + 1,
        "demo_samples": int(len(samples["unix_time_ns"])),
        "speedup_factor": float(options.speedup_factor),
    }))


if __name__ == "__main__":
    main()
