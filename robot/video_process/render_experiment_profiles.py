#!/usr/bin/env python3
"""Overlay timestamp-aligned robot profiles and an auxiliary camera on video."""

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROFILE_GROUPS = [
    ("Position error to true constraint", "mm", 1000.0, 4.5, [
        (("learned_to_true_surface_position_error_m",), "Learned",
         (234, 88, 12, 255), "dashed"),
        (("reference_true_surface_position_error_m", "true_manifold_position_error_m"),
         "Executed", (22, 163, 74, 255), "dashdot"),
    ]),
    ("Orientation error to true constraint", "deg", 1.0, 6.0, [
        (("learned_to_true_surface_orientation_error_deg",), "Learned",
         (234, 88, 12, 255), "dashed"),
        (("reference_true_surface_orientation_error_deg", "true_manifold_orientation_error_deg"),
         "Executed", (22, 163, 74, 255), "dashdot"),
    ]),
    ("End-effector speed", "mm/s", 1000.0, 80.0, [
        (("display_end_effector_speed_mps", "end_effector_speed_mps"),
         "Speed", (99, 102, 241, 255), "solid"),
    ]),
]

DEFAULT_VALUE_UPDATE_SECONDS = 0.2
DEFAULT_PLAYBACK_SPEED = 1.33
DEFAULT_SIDEBAR_WIDTH_RATIO = 0.34
DEFAULT_AUX_HORIZONTAL_CROP_FRACTION = 0.0
DEFAULT_AUX_SCALE = 0.86
DEFAULT_PROFILE_PANEL_SCALE = 0.86
DEFAULT_AUX_MAX_HEIGHT_RATIO = 0.38
DEFAULT_DEMO_ERROR_SUMMARY = (
    "datasets/real_iiwa/tasks/RealKuka/figures/"
    "human_demo_paper_arc3mm_cropped_summary.json")


def add_smoothed_speed(samples, window_s=0.25):
    """Add a display-only speed estimated by local linear position fitting.

    Raw end_effector_speed_mps remains untouched in the saved experiment log.
    A time-based window makes the result insensitive to small variations in the
    trace sampling interval and avoids amplifying repeated/quantized TF poses.
    """
    if len(samples) < 3:
        return
    times = np.asarray([sample.get("unix_time_ns", 0) for sample in samples],
                       dtype=np.float64) * 1e-9
    positions = np.asarray([sample.get("actual", [np.nan] * 3)
                            for sample in samples], dtype=np.float64)
    half_window = max(0.02, float(window_s) * 0.5)
    for index, sample in enumerate(samples):
        left = int(np.searchsorted(times, times[index] - half_window,
                                   side="left"))
        right = int(np.searchsorted(times, times[index] + half_window,
                                    side="right"))
        if right - left < 3 or not np.all(np.isfinite(positions[left:right])):
            continue
        local_t = times[left:right] - np.mean(times[left:right])
        denominator = float(np.dot(local_t, local_t))
        if denominator <= 1e-12:
            continue
        centered = positions[left:right] - np.mean(positions[left:right], axis=0)
        velocity = np.dot(local_t, centered) / denominator
        sample["display_end_effector_speed_mps"] = float(
            np.linalg.norm(velocity))


def add_tracking_decomposition(samples):
    """Backfill normal/tangential tracking fields for legacy trace logs."""
    for sample in samples:
        if "nominal_normal_tracking_error_m" in sample:
            continue
        actual = sample.get("actual")
        nominal = sample.get("nominal_planned")
        normal = sample.get("planned_tool_z")
        if not actual or not nominal or not normal:
            continue
        normal = np.asarray(normal, dtype=float)
        magnitude = float(np.linalg.norm(normal))
        if magnitude <= 1e-12:
            continue
        normal /= magnitude
        difference = np.asarray(actual, dtype=float) - np.asarray(
            nominal, dtype=float)
        signed = float(np.dot(difference, normal))
        tangential = difference - signed * normal
        sample["nominal_normal_tracking_signed_error_m"] = signed
        sample["nominal_normal_tracking_error_m"] = abs(signed)
        sample["nominal_tangential_tracking_error_m"] = float(
            np.linalg.norm(tangential))


def add_learned_to_true_errors(samples):
    for sample in samples:
        learned_signed = sample.get(
            "reference_learned_surface_signed_position_error_m")
        true_signed = sample.get("reference_true_surface_signed_position_error_m")
        if learned_signed is not None and true_signed is not None:
            # Both signed errors are measured from the same executed point, so
            # their difference estimates learned-vs-true surface separation at
            # this current XY location.
            sample["learned_to_true_surface_position_error_m"] = abs(
                float(true_signed) - float(learned_signed))

        learned_tool_z = sample.get("reference_learned_surface_expected_tool_z")
        true_tool_z = sample.get("reference_true_surface_expected_tool_z")
        if learned_tool_z is None or true_tool_z is None:
            continue
        learned_tool_z = np.asarray(learned_tool_z, dtype=float)
        true_tool_z = np.asarray(true_tool_z, dtype=float)
        learned_norm = float(np.linalg.norm(learned_tool_z))
        true_norm = float(np.linalg.norm(true_tool_z))
        if learned_norm <= 1e-12 or true_norm <= 1e-12:
            continue
        cosine = float(np.dot(learned_tool_z, true_tool_z) /
                       (learned_norm * true_norm))
        cosine = max(-1.0, min(1.0, cosine))
        sample["learned_to_true_surface_orientation_error_deg"] = math.degrees(
            math.acos(cosine))


def profile_fonts(font_size, title_size):
    """Use the same font family as the supplied profile-panel reference."""
    try:
        return (ImageFont.truetype("DejaVuSans.ttf", font_size),
                ImageFont.truetype("DejaVuSans-Bold.ttf", font_size),
                ImageFont.truetype("DejaVuSans-Bold.ttf", title_size))
    except (OSError, IOError):
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback


def panel_rectangle(draw, bounds, radius, fill, outline, width):
    """Draw the reference-style panel on both old and new Pillow releases."""
    if hasattr(draw, "rounded_rectangle"):
        draw.rounded_rectangle(bounds, radius=radius, fill=fill,
                               outline=outline, width=width)
    else:
        draw.rectangle(bounds, fill=fill, outline=outline, width=width)


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--timestamps", required=True)
    parser.add_argument("--aux-video", required=True)
    parser.add_argument("--aux-timestamps", required=True)
    parser.add_argument("--trace-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait-seconds", type=float, default=30.0)
    parser.add_argument("--value-update-seconds", type=float,
                        default=DEFAULT_VALUE_UPDATE_SECONDS,
                        help="seconds between numeric profile value updates")
    parser.add_argument("--playback-speed", type=float,
                        default=DEFAULT_PLAYBACK_SPEED,
                        help="render output at this source playback multiplier")
    parser.add_argument("--sidebar-width-ratio", type=float,
                        default=DEFAULT_SIDEBAR_WIDTH_RATIO,
                        help="fraction of the main video width used by overlays")
    parser.add_argument("--aux-horizontal-crop-fraction", type=float,
                        default=DEFAULT_AUX_HORIZONTAL_CROP_FRACTION,
                        help="fraction cropped from each side of aux video")
    parser.add_argument("--aux-scale", type=float, default=DEFAULT_AUX_SCALE,
                        help="fraction of sidebar width used by aux video")
    parser.add_argument("--profile-panel-scale", type=float,
                        default=DEFAULT_PROFILE_PANEL_SCALE,
                        help="fraction of sidebar width used by profile panel")
    parser.add_argument("--max-elapsed-seconds", type=float, default=None,
                        help="only render and plot data up to this many seconds "
                             "after the first main-camera timestamp")
    parser.add_argument("--demo-error-summary", default=DEFAULT_DEMO_ERROR_SUMMARY,
                        help="optional JSON summary with demonstration MAE values")
    return parser.parse_args()


def load_frame_times(path):
    with open(path, newline="") as stream:
        return [int(row["unix_time_ns"]) for row in csv.DictReader(stream)]


def limit_frame_times(frame_times, max_elapsed_seconds):
    if max_elapsed_seconds is None or max_elapsed_seconds <= 0:
        return frame_times
    end_ns = frame_times[0] + int(round(max_elapsed_seconds * 1e9))
    return [timestamp for timestamp in frame_times if timestamp <= end_ns]


def build_playback_schedule(frame_times, fps, playback_speed):
    if not frame_times:
        return []
    speed = max(1e-6, float(playback_speed))
    if abs(speed - 1.0) <= 1e-9:
        return list(range(len(frame_times)))
    elapsed = ((np.asarray(frame_times, dtype=np.float64) -
                float(frame_times[0])) * 1e-9)
    source_duration = max(0.0, float(elapsed[-1]))
    step = 1.0 / max(float(fps), 1e-6)
    output_elapsed = np.arange(0.0, source_duration / speed + step * 0.5,
                               step)
    source_elapsed = np.minimum(output_elapsed * speed, source_duration)
    schedule = []
    for source_time in source_elapsed:
        index = int(np.searchsorted(elapsed, source_time, side="left"))
        if index >= len(elapsed):
            index = len(elapsed) - 1
        schedule.append(index)
    if schedule[-1] != len(frame_times) - 1:
        schedule.append(len(frame_times) - 1)
    return schedule


def format_playback_label(playback_speed):
    speed = float(playback_speed)
    if abs(speed - round(speed)) <= 1e-6:
        text = str(int(round(speed)))
    else:
        text = ("%.2f" % speed).rstrip("0").rstrip(".")
    return "%s X" % text


def load_demo_errors(path):
    if not path:
        return {}
    try:
        payload = json.load(open(path))
    except (OSError, ValueError, TypeError):
        return {}

    position = payload.get("position_abs_error_mm", {})
    orientation = payload.get("orientation_error_deg", {})
    errors = {}
    if isinstance(position, dict) and position.get("mean") is not None:
        errors["Position error to true constraint"] = float(position["mean"])
    if isinstance(orientation, dict) and orientation.get("mean") is not None:
        errors["Orientation error to true constraint"] = float(
            orientation["mean"])
    return errors


def load_runs(directory, start_ns, end_ns):
    runs = []
    for path in glob.glob(os.path.join(directory, "constraint_plan_*.json")):
        if "_sim_" in os.path.basename(path):
            continue
        try:
            payload = json.load(open(path))
            samples = [sample for sample in payload.get("samples", [])
                       if isinstance(sample.get("unix_time_ns"), int)]
        except (OSError, ValueError, TypeError):
            continue
        if not samples or samples[-1]["unix_time_ns"] < start_ns or samples[0]["unix_time_ns"] > end_ns:
            continue
        manifold = [sample for sample in samples
                    if (sample.get("segment") == "following_manifold" and
                        start_ns <= sample["unix_time_ns"] <= end_ns)]
        if not manifold:
            continue
        add_tracking_decomposition(manifold)
        add_learned_to_true_errors(manifold)
        add_smoothed_speed(manifold)
        # Compatibility for runs where the active execution surface was the
        # same learned surface that the profile is meant to evaluate.  These
        # geometric fields are valid; the old learned_manifold_* fields are
        # deliberately not reused because they measured distance to the path.
        if payload.get("surface_id") == "human_demo_recorded_toolz_oncl":
            for sample in manifold:
                if "reference_learned_surface_position_error_m" not in sample:
                    value = sample.get("true_manifold_position_error_m")
                    if value is not None:
                        sample["reference_learned_surface_position_error_m"] = value
                if "reference_learned_surface_orientation_error_deg" not in sample:
                    value = sample.get("true_manifold_orientation_error_deg")
                    if value is not None:
                        sample["reference_learned_surface_orientation_error_deg"] = value
        runs.append({"plan_id": payload.get("plan_id"), "path": path,
                     "start_ns": samples[0]["unix_time_ns"],
                     "manifold_start_ns": manifold[0]["unix_time_ns"],
                     "manifold_end_ns": manifold[-1]["unix_time_ns"],
                     "samples": manifold})
    return sorted(runs, key=lambda run: run["start_ns"])


def wait_for_runs(directory, start_ns, end_ns, timeout):
    deadline = time.monotonic() + timeout
    while True:
        runs = load_runs(directory, start_ns, end_ns)
        if runs or time.monotonic() >= deadline:
            return runs
        time.sleep(1.0)


def active_run(runs, timestamp_ns):
    selected = None
    for run in runs:
        if run["start_ns"] <= timestamp_ns:
            selected = run
        else:
            break
    return selected


def place_auxiliary(canvas, frame, left, top, right, bottom,
                    horizontal_crop_fraction=DEFAULT_AUX_HORIZONTAL_CROP_FRACTION):
    """Fit the full auxiliary view into its overlay without cropping."""
    if frame is None or frame.size == 0:
        canvas[top:bottom, left:right] = (24, 28, 34)
        return
    source_h, source_w = frame.shape[:2]
    crop_fraction = float(np.clip(horizontal_crop_fraction, 0.0, 0.45))
    crop_x0 = int(round(source_w * crop_fraction))
    crop_x1 = source_w - crop_x0
    if crop_x1 - crop_x0 >= 2:
        frame = frame[:, crop_x0:crop_x1]
        source_h, source_w = frame.shape[:2]
    target_w, target_h = right - left, bottom - top
    canvas[top:bottom, left:right] = (24, 28, 34)
    scale = min(target_w / source_w, target_h / source_h)
    width, height = max(1, int(source_w * scale)), max(1, int(source_h * scale))
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    paste_x = left + max(0, (target_w - width) // 2)
    paste_y = top + max(0, (target_h - height) // 2)
    canvas[paste_y:paste_y + height, paste_x:paste_x + width] = resized


def sample_value(sample, keys):
    """Read a metric while retaining compatibility with older trace names."""
    for key in keys:
        value = sample.get(key)
        if value is not None:
            return value
    return np.nan


def numeric_visible_count(sample_times, run, timestamp_ns, update_seconds):
    """Hold legend values between display updates without slowing curves."""
    if update_seconds <= 0:
        display_timestamp_ns = timestamp_ns
    elif timestamp_ns >= run["manifold_end_ns"]:
        return len(run["samples"])
    else:
        interval_ns = max(1, int(round(update_seconds * 1e9)))
        elapsed_ns = max(0, timestamp_ns - run["manifold_start_ns"])
        display_timestamp_ns = (run["manifold_start_ns"] +
                                (elapsed_ns // interval_ns) * interval_ns)
    count = int(np.searchsorted(sample_times, display_timestamp_ns,
                                side="right"))
    return min(max(count, 0), len(run["samples"]))


def draw_profile_line(draw, points, color, style, width):
    """Draw color plus line style so curves survive compression/grayscale."""
    if len(points) < 2:
        return
    if style == "solid":
        draw.line(points, fill=color, width=width)
        return
    pattern = ((8, 5) if style == "dashed" else (8, 3, 2, 3))
    draw_segment = True
    pattern_index = 0
    remaining = pattern[0]
    for start, end in zip(points[:-1], points[1:]):
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = max(1.0, math.hypot(dx, dy))
        consumed = 0.0
        while consumed < length:
            step = min(remaining, length - consumed)
            a = consumed / length
            b = (consumed + step) / length
            if draw_segment:
                draw.line((start[0] + a * dx, start[1] + a * dy,
                           start[0] + b * dx, start[1] + b * dy),
                          fill=color, width=width)
            consumed += step
            remaining -= step
            if remaining <= 1e-9:
                pattern_index = (pattern_index + 1) % len(pattern)
                remaining = pattern[pattern_index]
                draw_segment = not draw_segment


def draw_profiles(frame, auxiliary_frame, run, timestamp_ns,
                  value_update_seconds=DEFAULT_VALUE_UPDATE_SECONDS,
                  sidebar_width_ratio=DEFAULT_SIDEBAR_WIDTH_RATIO,
                  aux_horizontal_crop_fraction=DEFAULT_AUX_HORIZONTAL_CROP_FRACTION,
                  aux_scale=DEFAULT_AUX_SCALE,
                  profile_panel_scale=DEFAULT_PROFILE_PANEL_SCALE,
                  demo_errors=None,
                  playback_label=None):
    height, width = frame.shape[:2]
    # Preserve the primary video's resolution: overlays occupy its right side.
    canvas = frame.copy()
    sidebar_ratio = float(np.clip(sidebar_width_ratio, 0.25, 0.6))
    panel_width = min(width - 20, max(430, int(round(width * sidebar_ratio))))
    sidebar_left = width - panel_width
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
        canvas, auxiliary_frame,
        auxiliary_left, auxiliary_top,
        auxiliary_left + aux_width, auxiliary_bottom,
        aux_horizontal_crop_fraction)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image, "RGBA")
    font_size = max(13, int(round(14 * scale)))
    title_size = max(16, int(round(19 * scale)))
    font, font_bold, font_title = profile_fonts(font_size, title_size)
    if playback_label:
        playback_font_size = max(34, int(round(40 * scale)))
        try:
            playback_font = ImageFont.truetype(
                "DejaVuSans-Bold.ttf", playback_font_size)
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
    draw.text((x0, header_y), "KUKA iiwa 14 profiles",
              fill=(20, 24, 30, 255), font=font_title)
    if run is None:
        draw.text((x0, header_y + int(38 * scale)), "Waiting for plan",
                  fill=(105, 110, 120, 255), font=font)
        return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    if timestamp_ns < run["manifold_start_ns"]:
        state = "Approaching Entry"
        state_color = (105, 110, 120, 255)
        visible_count = 0
        numeric_count = 0
    else:
        sample_times = np.asarray([sample["unix_time_ns"] for sample in run["samples"]], dtype=np.int64)
        visible_count = int(np.searchsorted(sample_times, timestamp_ns, side="right"))
        visible_count = min(max(visible_count, 0), len(run["samples"]))
        numeric_count = numeric_visible_count(
            sample_times, run, timestamp_ns, value_update_seconds)
        state = "Executing" if timestamp_ns <= run["manifold_end_ns"] else "Completed"
        state_color = ((37, 99, 235, 255) if
                       timestamp_ns <= run["manifold_end_ns"] else
                       (80, 85, 95, 255))
    draw.text((x0, header_y + int(36 * scale)), state,
              fill=state_color, font=font_bold)

    top, bottom = header_y + int(62 * scale), height - int(18 * scale)
    gap = int(6 * scale)
    weights = [1.0, 1.0, .68]
    usable_height = bottom - top - gap * (len(PROFILE_GROUPS) - 1)
    row_heights = [int(usable_height * value / sum(weights)) for value in weights]
    row_heights[-1] += usable_height - sum(row_heights)
    cursor_indigo = (99, 102, 241, 210)
    ry0 = top
    for row, (title, unit, value_scale, fixed_vmax, series) in enumerate(PROFILE_GROUPS):
        ry1 = ry0 + row_heights[row]
        label_w = int(min(220 * scale, profile_panel_width * .45))
        px0, px1 = x0 + label_w, x1
        py0, py1 = ry0 + int(23 * scale), ry1 - int(7 * scale)
        draw.rectangle((px0, py0, px1, py1), fill=(248, 250, 252, 220),
                       outline=(188, 196, 206, 220), width=max(1, int(scale)))
        draw.text((x0, ry0 + int(2 * scale)), "%s (%s)" % (title, unit),
                  fill=(18, 24, 38, 255), font=font_bold)
        arrays = [np.asarray([
            sample_value(sample, specification[0]) * value_scale
            for sample in run["samples"]], dtype=float)
                  for specification in series]
        finite_parts = [values[np.isfinite(values)] for values in arrays
                        if np.any(np.isfinite(values))]
        if not finite_parts:
            draw.text((x0, ry0 + int(28 * scale)), "no data",
                      fill=(120, 126, 136, 255), font=font)
            ry0 = ry1 + gap
            continue
        vmin, vmax = 0.0, fixed_vmax
        draw.text((px0 + 3, py0 + 2), "%.1f" % vmax,
                  fill=(105, 110, 120, 255), font=font)
        legend_y = ry0 + int(25 * scale)
        demo_mae = None if demo_errors is None else demo_errors.get(title)
        demo_color = (90, 96, 108, 255)
        if demo_mae is not None and np.isfinite(demo_mae):
            demo_axis_y = int(round(
                py1 - (demo_mae - vmin) / max(vmax - vmin, 1e-9) *
                (py1 - py0)))
            demo_axis_y = max(py0, min(py1, demo_axis_y))
            draw_profile_line(
                draw, [(px0, demo_axis_y), (px1, demo_axis_y)],
                demo_color, "dashed", max(2, int(2 * scale)))
        denominator = max(len(run["samples"]) - 1, 1)
        for series_index, ((keys, label, color, style), full) in enumerate(
                zip(series, arrays)):
            current = full[numeric_count - 1] if numeric_count else np.nan
            marker_y = legend_y + series_index * int(20 * scale) + int(7 * scale)
            draw_profile_line(
                draw,
                [(x0, marker_y), (x0 + int(20 * scale), marker_y)],
                color, style, max(2, int(2 * scale)))
            value_text = "—" if not np.isfinite(current) else "%.2f" % current
            draw.text((x0 + int(25 * scale),
                       legend_y + series_index * int(20 * scale)),
                      "%s  %s" % (label, value_text), fill=color, font=font)
            if not visible_count:
                continue
            values = full[:visible_count]
            points = []
            for index, value in enumerate(values):
                if not np.isfinite(value):
                    continue
                x = int(round(px0 + index / denominator * (px1 - px0)))
                y = int(round(py1 - (value - vmin) / max(vmax - vmin, 1e-9) * (py1 - py0)))
                points.append((x, max(py0, min(py1, y))))
            if len(points) >= 2:
                draw_profile_line(draw, points, color, style,
                                  max(2, int(2 * scale)))
            elif points:
                radius = max(2, int(3 * scale))
                draw.ellipse((points[0][0] - radius, points[0][1] - radius,
                              points[0][0] + radius, points[0][1] + radius),
                             fill=color)
        if demo_mae is not None and np.isfinite(demo_mae):
            demo_y = legend_y + len(series) * int(20 * scale)
            draw_profile_line(
                draw,
                [(x0, demo_y + int(7 * scale)),
                 (x0 + int(20 * scale), demo_y + int(7 * scale))],
                demo_color, "dashed", max(2, int(2 * scale)))
            draw.text((x0 + int(25 * scale), demo_y),
                      "human mean  %.2f" % demo_mae,
                      fill=demo_color, font=font)
        if visible_count:
            cursor_x = int(round(px0 + (visible_count - 1) / denominator * (px1 - px0)))
            draw.line((cursor_x, py0, cursor_x, py1), fill=cursor_indigo,
                      width=max(1, int(scale)))
        ry0 = ry1 + gap
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def open_writer(path, fps, size):
    return FFmpegVideoWriter(path, fps, size, crf=18)


class FFmpegVideoWriter:
    """Stream rendered BGR frames to FFmpeg/libx264 without avc1 fallback."""

    def __init__(self, path, fps, size, crf=18):
        self.size = (int(size[0]), int(size[1]))
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s:v", "%dx%d" % self.size,
            "-r", "%.6f" % float(fps), "-i", "pipe:0", "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        try:
            self.process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        except (OSError, ValueError):
            self.process = None

    def isOpened(self):
        return (self.process is not None and self.process.poll() is None and
                self.process.stdin is not None)

    def write(self, frame):
        if (not self.isOpened() or frame is None or
                (frame.shape[1], frame.shape[0]) != self.size):
            return False
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
            return True
        except (BrokenPipeError, OSError):
            return False

    def release(self):
        if self.process is None:
            return
        if self.process.stdin is not None:
            try:
                self.process.stdin.close()
            except OSError:
                pass
        try:
            self.process.wait(timeout=60.0)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None


def main():
    options = args()
    demo_errors = load_demo_errors(options.demo_error_summary)
    frame_times = load_frame_times(options.timestamps)
    auxiliary_times = load_frame_times(options.aux_timestamps)
    if not frame_times:
        raise SystemExit("timestamp CSV contains no frames")
    if not auxiliary_times:
        raise SystemExit("auxiliary timestamp CSV contains no frames")
    frame_times = limit_frame_times(frame_times, options.max_elapsed_seconds)
    runs = wait_for_runs(options.trace_directory, frame_times[0], frame_times[-1],
                         options.wait_seconds)
    if not runs:
        raise SystemExit("no timestamped real-robot trace overlaps this video")
    capture = cv2.VideoCapture(options.video)
    auxiliary_capture = cv2.VideoCapture(options.aux_video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    schedule = build_playback_schedule(frame_times, fps, options.playback_speed)
    if not schedule:
        raise SystemExit("frame schedule is empty")
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = open_writer(options.output, fps, (width, height))
    if (not capture.isOpened() or not auxiliary_capture.isOpened() or
            not writer.isOpened()):
        raise SystemExit("cannot open input or output video")
    source_index = 0
    schedule_index = 0
    output_frames = 0
    auxiliary_index = -1
    auxiliary_frame = None
    playback_label = format_playback_label(options.playback_speed)
    last_source_index = schedule[-1]
    while source_index <= last_source_index:
        ok, frame = capture.read()
        if not ok:
            break
        while (schedule_index < len(schedule) and
               schedule[schedule_index] == source_index):
            timestamp_ns = frame_times[source_index]
            # Advance the auxiliary stream to the most recent frame at the main
            # camera timestamp.  Independent absolute timestamp CSVs make this
            # deterministic even when the two cameras have different frame rates.
            while (auxiliary_index + 1 < len(auxiliary_times) and
                   auxiliary_times[auxiliary_index + 1] <= timestamp_ns):
                aux_ok, candidate = auxiliary_capture.read()
                if not aux_ok:
                    break
                auxiliary_frame = candidate
                auxiliary_index += 1
            writer.write(draw_profiles(
                frame, auxiliary_frame, active_run(runs, timestamp_ns),
                timestamp_ns, options.value_update_seconds,
                options.sidebar_width_ratio,
                options.aux_horizontal_crop_fraction,
                options.aux_scale,
                options.profile_panel_scale,
                demo_errors,
                playback_label))
            schedule_index += 1
            output_frames += 1
        source_index += 1
    capture.release(); auxiliary_capture.release(); writer.release()
    print(json.dumps({"ok": True, "output": options.output,
                      "frames": output_frames,
                      "source_frames_read": source_index,
                      "auxiliary_frames": auxiliary_index + 1,
                      "playback_speed": float(options.playback_speed),
                      "runs": [run["plan_id"] for run in runs]}))


if __name__ == "__main__":
    main()
