#!/usr/bin/env python3
"""Render one accepted single-camera run with stage-aware feature profiles."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.barclean_display import BARCLEAN_AXIAL_DISPLAY_SHIFT_M


EXECUTED_COLOR = (37, 99, 235, 255)
LEARNED_CONSTRAINT_COLOR = (234, 88, 12, 245)
LEARNED_CONSTRAINT_TEXT_COLOR = (194, 65, 12, 255)
INEQUALITY_FEASIBLE_COLOR = (254, 240, 138, 145)
EQUALITY_BAND_COLOR = (253, 186, 116, 92)
CURSOR_COLOR = (79, 70, 229, 210)
STAGE_COLORS = (
    (59, 130, 246),
    (16, 185, 129),
    (245, 158, 11),
    (168, 85, 247),
    (239, 68, 68),
)
BARCLEAN_SPONGE_YAW_OFFSET_DEG = 38.30219823954972


FEATURE_DISPLAY = {
    "obs_dist": ("obs_dist", "mm", 1000.0),
    "table_dist": ("table_dist", "mm", 1000.0),
    "lateral_offset": ("lateral_offset", "mm", 1000.0),
    "axial_offset": ("axial_offset", "mm", 1000.0),
    "tool_pitch": ("tool_pitch", "deg", 180.0 / math.pi),
    "tool_roll": ("tool_roll", "deg", 180.0 / math.pi),
    "tool_yaw": ("tool_yaw", "deg", 180.0 / math.pi),
}

FEATURE_ORDER_BY_TASK = {
    "barclean": (
        "obs_dist",
        "table_dist",
        "lateral_offset",
        "axial_offset",
        "tool_pitch",
        "tool_roll",
        "tool_yaw",
    ),
}


@dataclass(frozen=True)
class StageWindow:
    index: int
    start_s: float
    end_s: float


@dataclass(frozen=True)
class ConstraintReference:
    stage: int
    value: float
    semantics: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay stage-aware feature profiles on one final-run video"
    )
    parser.add_argument("--run-directory", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument(
        "--preview-image",
        type=Path,
        help="render only the final source frame to a PNG/JPEG preview",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=15)
    parser.add_argument(
        "--encoder",
        choices=("auto", "cpu", "nvenc"),
        default="auto",
        help="video encoder; auto uses NVIDIA NVENC when available",
    )
    parser.add_argument("--panel-width-ratio", type=float, default=0.238)
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=None,
        help="render only the beginning of a run; intended for layout debugging",
    )
    return parser


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"cannot read JSON file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return payload


def _run_inputs(run_directory: Path) -> tuple[Path, Path]:
    directory = run_directory.expanduser().resolve()
    video = directory / "execution.mp4"
    visualization = directory / "visualization.json"
    missing = [path.name for path in (video, visualization) if not path.is_file()]
    if missing:
        raise RuntimeError(
            f"run {directory.name} is missing required files: {', '.join(missing)}"
        )
    return video, visualization


def _load_render_visualization(run_directory: Path, path: Path) -> dict:
    visualization = _load_json(path)
    if str(visualization.get("mode", "")).lower() != "real":
        return visualization
    synchronized_path = run_directory / "synchronized_profiles.json"
    if not synchronized_path.is_file():
        raise RuntimeError(
            f"real run {run_directory.name} has no synchronized_profiles.json; "
            "run extract_synchronized_profiles.py first"
        )
    synchronized = _load_json(synchronized_path)
    if str(synchronized.get("task_id", "")) != str(
        visualization.get("task_id", "")
    ):
        raise RuntimeError("synchronized profile task does not match visualization")
    for key in (
        "feature_series",
        "planned_feature_series",
        "stage_boundary_indices",
        "stage_boundary_times",
        "stage_transition_end_times",
    ):
        visualization[key] = synchronized[key]
    visualization["profile_synchronization"] = synchronized["timing"]
    return visualization


def probe_video(path: Path) -> dict:
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
        "duration": float(payload.get("format", {}).get("duration", 0.0)),
        "avg_frame_rate": str(stream.get("avg_frame_rate", "0/1")),
        "nb_frames": int(stream.get("nb_frames") or 0),
    }


def load_frame_times(path: Path) -> list[float]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    frames = json.loads(result.stdout).get("frames", [])
    values = []
    for frame in frames:
        try:
            value = float(frame["best_effort_timestamp_time"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    if not values:
        raise RuntimeError(f"ffprobe returned no frame timestamps for {path}")
    first = values[0]
    normalized = [max(0.0, value - first) for value in values]
    if any(current < previous for previous, current in zip(normalized, normalized[1:])):
        raise RuntimeError(f"video frame timestamps are not monotonic in {path}")
    return normalized


def build_frame_schedule(
    frame_times: Sequence[float], duration_s: float, fps: float
) -> tuple[list[int], list[float]]:
    if not frame_times:
        return [], []
    fps = float(fps)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive")
    duration_s = max(0.0, float(duration_s))
    output_count = max(1, int(math.ceil(duration_s * fps)))
    output_times = np.arange(output_count, dtype=float) / fps
    source_times = np.asarray(frame_times, dtype=float)
    schedule = np.searchsorted(source_times, output_times, side="right") - 1
    schedule = np.clip(schedule, 0, len(source_times) - 1)
    return schedule.astype(int).tolist(), output_times.tolist()


def _series_array(series: dict) -> tuple[list[str], dict[str, str], np.ndarray]:
    schema = series.get("schema", [])
    samples = series.get("samples", [])
    if not isinstance(schema, list) or not isinstance(samples, list):
        raise RuntimeError("feature series schema and samples must be arrays")
    names = [str(item.get("name", f"feature_{index}")) for index, item in enumerate(schema)]
    units = {
        name: str(schema[index].get("unit", "")) for index, name in enumerate(names)
    }
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] != len(names) + 1:
        raise RuntimeError(
            "feature samples must have shape (N, 1 + number of schema entries)"
        )
    if not np.all(np.isfinite(array[:, 0])):
        raise RuntimeError("feature sample times must be finite")
    return names, units, array


def _ordered_feature_series(
    task_id: str, names: list[str], units: dict[str, str], values: np.ndarray
) -> tuple[list[str], dict[str, str], np.ndarray]:
    preferred_order = FEATURE_ORDER_BY_TASK.get(str(task_id).strip().lower())
    if preferred_order is None:
        return names, units, values
    ordered_names = [name for name in preferred_order if name in names]
    ordered_names.extend(name for name in names if name not in ordered_names)
    if ordered_names == names:
        return names, units, values
    indices = [names.index(name) for name in ordered_names]
    ordered_values = np.column_stack(
        (values[:, 0], values[:, np.asarray(indices, dtype=int) + 1])
    )
    return ordered_names, {name: units[name] for name in ordered_names}, ordered_values


def stage_timeline(
    visualization: dict, video_duration_s: float
) -> tuple[list[StageWindow], list[tuple[float, float]], float]:
    planned = visualization.get("planned_feature_series", {})
    try:
        _, _, planned_samples = _series_array(planned)
        planned_end = float(planned_samples[-1, 0])
    except RuntimeError:
        planned_end = float(video_duration_s)
    if not math.isfinite(planned_end) or planned_end <= 1e-9:
        planned_end = float(video_duration_s)
    boundaries = [
        float(value)
        for value in visualization.get("stage_boundary_times", [])
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    transition_ends = [
        float(value)
        for value in visualization.get("stage_transition_end_times", [])
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    if len(boundaries) != len(transition_ends):
        raise RuntimeError("stage boundary and transition-end counts do not match")
    if any(current <= previous for previous, current in zip(boundaries, boundaries[1:])):
        raise RuntimeError("stage boundary times must be strictly increasing")
    transitions = []
    for start, end in zip(boundaries, transition_ends):
        transitions.append(
            (
                float(np.clip(start, 0.0, video_duration_s)),
                float(np.clip(max(start, end), 0.0, video_duration_s)),
            )
        )
    stage_splits = [0.5 * (start + end) for start, end in transitions]
    windows = []
    stage_count = len(boundaries) + 1
    for index in range(stage_count):
        start = 0.0 if index == 0 else stage_splits[index - 1]
        end = (
            float(video_duration_s)
            if index == stage_count - 1
            else stage_splits[index]
        )
        windows.append(StageWindow(index, min(start, end), max(start, end)))
    return windows, transitions, 1.0


def constraint_references(
    series: dict, feature_name: str, spec_key: str
) -> list[ConstraintReference]:
    constraints = dict(series.get("true_constraints") or {})
    references = []
    for raw in series.get(spec_key, []) or []:
        if not isinstance(raw, dict) or str(raw.get("feature_name", "")) != feature_name:
            continue
        value = constraints.get(str(raw.get("oracle_key", "")), raw.get("value"))
        try:
            value = float(value)
            stage = int(raw["stage"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value) or stage < 0:
            continue
        references.append(
            ConstraintReference(
                stage=stage,
                value=value,
                semantics=str(raw.get("semantics", "target_value")),
            )
        )
    return references


def _semantics_kind(text: str) -> str:
    value = str(text).strip().lower()
    if value in {"target", "target_value", "equality", "eq", "equal"}:
        return "target"
    if value in {"upper", "upper_bound", "max", "maximum", "<=", "leq"}:
        return "upper"
    if value in {"lower", "lower_bound", "min", "minimum", ">=", "geq"}:
        return "lower"
    return value


def _axis_limits(
    values: Sequence[float], shared_span: float | None = None
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("axis values must contain at least one finite value")
    raw_min, raw_max = float(np.min(finite)), float(np.max(finite))
    raw_span = raw_max - raw_min
    if shared_span is not None:
        span = float(shared_span)
        if math.isfinite(span) and span > 0.0:
            center = 0.5 * (raw_min + raw_max)
            return center - 0.5 * span, center + 0.5 * span
    margin = max(
        1e-4,
        (0.15 * abs(raw_max) + 1e-4) if raw_span < 1e-8 else 0.12 * raw_span,
    )
    return raw_min - margin, raw_max + margin


def _feature_display(name: str, source_unit: str) -> tuple[str, str, float]:
    if name in FEATURE_DISPLAY:
        return FEATURE_DISPLAY[name]
    return name.replace("_", " "), source_unit, 1.0


def _display_feature_values(
    task_id: str,
    feature_name: str,
    values: Sequence[float] | np.ndarray,
    value_scale: float,
) -> np.ndarray:
    displayed = np.asarray(values, dtype=float) * float(value_scale)
    if str(task_id).strip().lower() != "barclean":
        return displayed
    if feature_name == "tool_pitch":
        return displayed - 90.0
    if feature_name == "tool_yaw":
        return (
            displayed + BARCLEAN_SPONGE_YAW_OFFSET_DEG + 180.0
        ) % 360.0 - 180.0
    if feature_name == "axial_offset":
        return displayed + BARCLEAN_AXIAL_DISPLAY_SHIFT_M * float(value_scale)
    return displayed


def _fonts(height: int):
    scale = float(np.clip(height / 1080.0, 0.75, 1.4))
    sizes = (
        max(15, round(17 * scale)),
        max(15, round(17 * scale)),
        max(17, round(19 * scale)),
        max(16, round(18 * scale)),
    )
    try:
        return (
            ImageFont.truetype("DejaVuSans.ttf", sizes[0]),
            ImageFont.truetype("DejaVuSans-Bold.ttf", sizes[1]),
            ImageFont.truetype("DejaVuSans-Bold.ttf", sizes[2]),
            ImageFont.truetype("DejaVuSans.ttf", sizes[3]),
        )
    except (OSError, IOError):
        fallback = ImageFont.load_default()
        return fallback, fallback, fallback, fallback


def _draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    xy: Sequence[float],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash: float,
    gap: float,
) -> None:
    x0, y0, x1, y1 = [float(value) for value in xy]
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 1e-9:
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    position = 0.0
    while position < length:
        end = min(position + dash, length)
        draw.line(
            (
                x0 + ux * position,
                y0 + uy * position,
                x0 + ux * end,
                y0 + uy * end,
            ),
            fill=fill,
            width=width,
        )
        position = end + gap


def _alpha_fill_rectangle(
    image: Image.Image,
    xy: Sequence[float],
    fill: tuple[int, int, int, int],
) -> None:
    x0, y0, x1, y1 = [float(value) for value in xy]
    left = max(0, int(math.floor(min(x0, x1))))
    top = max(0, int(math.floor(min(y0, y1))))
    right = min(image.width, int(math.ceil(max(x0, x1))) + 1)
    bottom = min(image.height, int(math.ceil(max(y0, y1))) + 1)
    if right <= left or bottom <= top:
        return
    layer = Image.new("RGBA", (right - left, bottom - top), fill)
    image.alpha_composite(layer, dest=(left, top))


def _draw_styled_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[tuple[float, float]],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dashed: bool,
) -> None:
    if len(points) < 2:
        return
    if not dashed:
        draw.line(points, fill=fill, width=width)
        return
    for first, second in zip(points, points[1:]):
        _draw_dashed_segment(
            draw,
            (*first, *second),
            fill=fill,
            width=width,
            dash=8.0,
            gap=5.0,
        )


def _value_at_or_before(times: np.ndarray, values: np.ndarray, timestamp_s: float) -> float:
    index = int(np.searchsorted(times, timestamp_s, side="right")) - 1
    if index < 0:
        return float("nan")
    return float(values[min(index, len(values) - 1)])


def _format_value(value: float) -> str:
    if not math.isfinite(value):
        return "—"
    absolute = abs(value)
    if absolute >= 100.0:
        return f"{value:.1f}"
    if absolute >= 10.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


class FeaturePanel:
    def __init__(self, visualization: dict, video_duration_s: float, run_id: str):
        self.visualization = visualization
        self.video_duration_s = float(video_duration_s)
        self.run_id = str(run_id)
        self.task_id = str(visualization.get("task_id", ""))
        self.names, self.units, self.executed = _series_array(
            visualization.get("feature_series", {})
        )
        self.names, self.units, self.executed = _ordered_feature_series(
            self.task_id, self.names, self.units, self.executed
        )
        planned = visualization.get("planned_feature_series", {})
        self.windows, self.transitions, self.planned_time_scale = stage_timeline(
            visualization, self.video_duration_s
        )
        self.learned_constraint_series = planned
        self.show_learned_constraints = bool(
            planned.get("planning_constraint_specs")
        )
        self.shared_angle_span = self._yaw_axis_span()
        self._curve_layer = None
        self._curve_cache_key = None
        self._curve_counts = {}
        self._last_timestamp_s = -math.inf
        self._static_layer = None
        self._row_layouts = []

    def _prepare_curve_cache(
        self, size: tuple[int, int], panel_width_ratio: float, timestamp_s: float
    ) -> None:
        cache_key = (size, round(float(panel_width_ratio), 8))
        if cache_key != self._curve_cache_key:
            self._curve_layer = Image.new("RGBA", size, (0, 0, 0, 0))
            self._curve_cache_key = cache_key
            self._curve_counts = {}
            self._static_layer = None
            self._row_layouts = []
        elif timestamp_s < self._last_timestamp_s:
            self._curve_layer = Image.new("RGBA", size, (0, 0, 0, 0))
            self._curve_counts = {}
        self._last_timestamp_s = float(timestamp_s)

    def _extend_curve(
        self,
        key: tuple[str, int, str],
        times: np.ndarray,
        points: list[tuple[float, float]],
        timestamp_s: float,
        *,
        fill: tuple[int, int, int, int],
        width: int,
        dashed: bool,
    ) -> None:
        if self._curve_layer is None or not points:
            return
        visible_count = int(np.searchsorted(times, timestamp_s, side="right"))
        visible_count = min(visible_count, len(points))
        previous_count = int(self._curve_counts.get(key, 0))
        if visible_count <= previous_count:
            return
        start = max(0, previous_count - 1)
        _draw_styled_polyline(
            ImageDraw.Draw(self._curve_layer),
            points[start:visible_count],
            fill=fill,
            width=width,
            dashed=dashed,
        )
        self._curve_counts[key] = visible_count

    def _draw_cached(
        self, image: Image.Image, timestamp_s: float
    ) -> np.ndarray:
        overlay = self._static_layer.copy()
        for layout in self._row_layouts:
            self._extend_curve(
                ("executed", layout["row"], layout["name"]),
                layout["executed_times"],
                layout["executed_points"],
                timestamp_s,
                fill=EXECUTED_COLOR,
                width=layout["line_width"],
                dashed=False,
            )
        if self._curve_layer is not None:
            overlay = Image.alpha_composite(overlay, self._curve_layer)
        draw = ImageDraw.Draw(overlay)
        for layout in self._row_layouts:
            current_value = _value_at_or_before(
                layout["executed_times"], layout["executed_values"], timestamp_s
            )
            display_unit = layout["display_unit"]
            draw.text(
                layout["value_text_xy"],
                (
                    f"{_format_value(current_value)} {display_unit}"
                    if display_unit
                    else _format_value(current_value)
                ),
                fill=EXECUTED_COLOR,
                font=layout["font"],
            )
            cursor_x = layout["plot_x0"] + float(
                np.clip(timestamp_s, 0.0, self.video_duration_s)
            ) / max(self.video_duration_s, 1e-9) * layout["plot_width"]
            draw.line(
                (cursor_x, layout["py0"], cursor_x, layout["py1"]),
                fill=CURSOR_COLOR,
                width=layout["cursor_width"],
            )
            if math.isfinite(current_value):
                current_y = layout["py1"] - (
                    (current_value - layout["vmin"])
                    / max(layout["vmax"] - layout["vmin"], 1e-12)
                    * (layout["py1"] - layout["py0"])
                )
                radius = layout["radius"]
                draw.ellipse(
                    (
                        cursor_x - radius,
                        current_y - radius,
                        cursor_x + radius,
                        current_y + radius,
                    ),
                    fill=EXECUTED_COLOR,
                )
        composited = Image.alpha_composite(image, overlay).convert("RGB")
        return cv2.cvtColor(np.asarray(composited), cv2.COLOR_RGB2BGR)

    def _yaw_axis_span(self) -> float | None:
        name = "tool_yaw"
        try:
            executed_index = self.names.index(name)
        except ValueError:
            return None
        _, _, value_scale = _feature_display(name, self.units.get(name, ""))
        values = [
            _display_feature_values(
                self.task_id,
                name,
                self.executed[:, executed_index + 1],
                value_scale,
            )
        ]
        references = constraint_references(
            self.learned_constraint_series, name, "planning_constraint_specs"
        )
        if references:
            values.append(
                _display_feature_values(
                    self.task_id,
                    name,
                    [reference.value for reference in references],
                    value_scale,
                )
            )
        combined = np.concatenate(
            [part[np.isfinite(part)] for part in values if part.size]
        )
        vmin, vmax = _axis_limits(combined)
        return vmax - vmin

    def draw(
        self, frame_bgr: np.ndarray, timestamp_s: float, panel_width_ratio: float
    ) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).convert("RGBA")
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        width, height = image.size
        self._prepare_curve_cache(image.size, panel_width_ratio, timestamp_s)
        if self._static_layer is not None:
            return self._draw_cached(image, timestamp_s)
        margin = max(10, int(round(height * 0.012)))
        panel_width = int(
            np.clip(
                width * float(panel_width_ratio),
                min(400, width - 2 * margin),
                width - 2 * margin,
            )
        )
        x0, x1 = width - margin - panel_width, width - margin
        y0, y1 = margin, height - margin
        font, font_bold, font_title, font_legend = _fonts(height)
        scale = float(np.clip(height / 1080.0, 0.75, 1.4))
        line_thin = max(1, int(round(scale)))
        line_mid = max(2, int(round(2.3 * scale)))
        pad = max(12, int(round(14 * scale)))
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=max(7, int(round(9 * scale))),
            fill=(255, 255, 255, 232),
            outline=(36, 42, 50, 190),
            width=line_thin,
        )
        draw.text(
            (x0 + pad, y0 + max(8, pad // 2)),
            "Feature Profiles and Learned Constraints",
            fill=(20, 24, 30, 255),
            font=font_title,
        )
        legend_y = y0 + int(round(42 * scale))
        legend_items = [
            ("executed trajectory", EXECUTED_COLOR, "line"),
        ]
        if self.show_learned_constraints:
            legend_items.extend(
                [
                    (
                        "learned equality constraint",
                        LEARNED_CONSTRAINT_COLOR,
                        "equality",
                    ),
                    (
                        "learned inequality constraint\n(feasible side shaded)",
                        LEARNED_CONSTRAINT_COLOR,
                        "inequality",
                    ),
                ]
            )
        legend_x = x0 + pad
        sample_width = int(round(36 * scale))
        legend_line_height = int(round(24 * scale))
        legend_cursor_y = legend_y
        for label, color, style in legend_items:
            lx = legend_x
            ly = legend_cursor_y
            if style == "equality":
                draw.line(
                    (lx, ly + 7 * scale, lx + sample_width, ly + 7 * scale),
                    fill=color,
                    width=max(4, int(round(3.0 * scale))),
                )
            elif style == "inequality":
                _alpha_fill_rectangle(
                    overlay,
                    (lx, ly + 2 * scale, lx + sample_width, ly + 12 * scale),
                    INEQUALITY_FEASIBLE_COLOR,
                )
                _draw_dashed_segment(
                    draw,
                    (lx, ly + 7 * scale, lx + sample_width, ly + 7 * scale),
                    fill=color,
                    width=line_mid,
                    dash=7 * scale,
                    gap=4 * scale,
                )
            else:
                draw.line(
                    (lx, ly + 7 * scale, lx + sample_width, ly + 7 * scale),
                    fill=color,
                    width=line_mid,
                )
            draw.multiline_text(
                (lx + sample_width + 6 * scale, ly),
                label,
                fill=(
                    LEARNED_CONSTRAINT_TEXT_COLOR
                    if style in {"equality", "inequality"}
                    else color
                ),
                font=font_legend,
                spacing=max(0, int(round(scale))),
            )
            legend_cursor_y += (label.count("\n") + 1) * legend_line_height
        stage_top = legend_cursor_y + int(round(5 * scale))
        label_width = int(
            np.clip(panel_width * 0.30, 145 * scale, 185 * scale)
        )
        plot_x0 = x0 + pad + label_width
        plot_x1 = x1 - pad
        plot_width = max(1.0, float(plot_x1 - plot_x0))

        def x_at(value: float) -> float:
            return plot_x0 + float(
                np.clip(value, 0.0, self.video_duration_s)
            ) / max(self.video_duration_s, 1e-9) * plot_width

        stage_height = int(round(21 * scale))
        stage_bottom = stage_top + stage_height
        table_x0 = x0 + pad
        draw.rectangle(
            (table_x0, stage_top, plot_x1, stage_bottom),
            fill=(250, 251, 253, 215),
            outline=(145, 154, 166, 220),
            width=line_thin,
        )
        draw.line(
            (plot_x0, stage_top, plot_x0, stage_bottom),
            fill=(145, 154, 166, 220),
            width=line_thin,
        )
        draw.text(
            (table_x0 + 6 * scale, stage_top + 1 * scale),
            "Stage",
            fill=(18, 24, 38, 245),
            font=font_bold,
        )
        for window in self.windows:
            xa, xb = x_at(window.start_s), x_at(window.end_s)
            color = STAGE_COLORS[window.index % len(STAGE_COLORS)]
            blend = 0.28 if window.start_s <= timestamp_s <= window.end_s else 0.16
            pastel = tuple(
                int(round(blend * channel + (1.0 - blend) * 255.0))
                for channel in color
            )
            draw.rectangle(
                (xa, stage_top, xb, stage_bottom),
                fill=(*pastel, 225),
                outline=(*color, 210),
                width=line_thin,
            )
            draw.text(
                (xa + 3 * scale, stage_top + 1 * scale),
                f"S{window.index + 1}",
                fill=(18, 24, 38, 245),
                font=font_bold,
            )
        rows_top = stage_top + stage_height + int(round(5 * scale))
        row_height = max(62, int((y1 - pad - rows_top) / max(len(self.names), 1)))
        row_layouts = []
        for row, name in enumerate(self.names):
            row_y0 = rows_top + row * row_height
            row_y1 = min(y1 - pad, row_y0 + row_height)
            if row_y1 - row_y0 < 34:
                break
            py0 = row_y0 + int(round(6 * scale))
            py1 = row_y1 - int(round(7 * scale))
            source_unit = self.units.get(name, "")
            label, display_unit, value_scale = _feature_display(name, source_unit)
            executed_times = self.executed[:, 0]
            executed_values = _display_feature_values(
                self.task_id,
                name,
                self.executed[:, row + 1],
                value_scale,
            )
            learned_refs = (
                constraint_references(
                    self.learned_constraint_series,
                    name,
                    "planning_constraint_specs",
                )
                if self.show_learned_constraints
                else []
            )
            all_values = [executed_values[np.isfinite(executed_values)]]
            reference_values = _display_feature_values(
                self.task_id,
                name,
                [reference.value for reference in learned_refs],
                value_scale,
            )
            if reference_values.size:
                all_values.append(reference_values[np.isfinite(reference_values)])
            finite_parts = [values for values in all_values if values.size]
            if not finite_parts:
                continue
            combined = np.concatenate(finite_parts)
            shared_span = (
                self.shared_angle_span
                if name in {"tool_pitch", "tool_roll", "tool_yaw"}
                else None
            )
            vmin, vmax = _axis_limits(combined, shared_span)

            def y_at(value: float) -> float:
                return py1 - (float(value) - vmin) / max(
                    vmax - vmin, 1e-12
                ) * (py1 - py0)

            draw.text(
                (x0 + pad, row_y0 + int(round(7 * scale))),
                label,
                fill=(18, 24, 38, 255),
                font=font_bold,
            )
            draw.rectangle(
                (plot_x0, py0, plot_x1, py1),
                fill=(248, 250, 252, 205),
                outline=(188, 196, 206, 220),
                width=line_thin,
            )
            for window in self.windows[1:]:
                x = x_at(window.start_s)
                draw.line(
                    (x, py0, x, py1),
                    fill=(145, 154, 166, 135),
                    width=line_thin,
                )
            learned_line_width = max(3, int(round(2.0 * scale)))
            learned_heavy_width = max(4, int(round(3.0 * scale)))
            for reference, reference_value in zip(learned_refs, reference_values):
                self._draw_constraint(
                    overlay,
                    draw,
                    reference,
                    self.windows,
                    x_at,
                    y_at,
                    float(reference_value),
                    py0,
                    py1,
                    LEARNED_CONSTRAINT_COLOR,
                    learned_line_width,
                    learned_heavy_width,
                )
            executed_finite = np.isfinite(executed_values)
            executed_curve_times = executed_times[executed_finite]
            executed_points = [
                (x_at(time_value), y_at(value))
                for time_value, value in zip(
                    executed_curve_times,
                    executed_values[executed_finite],
                )
            ]
            row_layouts.append(
                {
                    "row": row,
                    "name": name,
                    "executed_times": executed_curve_times,
                    "executed_points": executed_points,
                    "executed_values": executed_values[executed_finite],
                    "display_unit": display_unit,
                    "value_text_xy": (
                        x0 + pad,
                        row_y0 + int(round(30 * scale)),
                    ),
                    "font": font,
                    "line_width": line_mid,
                    "cursor_width": line_thin,
                    "radius": max(2, int(round(2.5 * scale))),
                    "plot_x0": plot_x0,
                    "plot_width": plot_width,
                    "py0": py0,
                    "py1": py1,
                    "vmin": vmin,
                    "vmax": vmax,
                }
            )
        self._static_layer = overlay.copy()
        self._row_layouts = row_layouts
        return self._draw_cached(image, timestamp_s)

    @staticmethod
    def _draw_constraint(
        overlay: Image.Image,
        draw: ImageDraw.ImageDraw,
        reference: ConstraintReference,
        windows: Sequence[StageWindow],
        x_at,
        y_at,
        display_value: float,
        py0: float,
        py1: float,
        color: tuple[int, int, int, int],
        line_width: int,
        heavy_line_width: int,
    ) -> None:
        if reference.stage >= len(windows):
            return
        window = windows[reference.stage]
        xa, xb = x_at(window.start_s), x_at(window.end_s)
        y = float(np.clip(y_at(display_value), py0, py1))
        kind = _semantics_kind(reference.semantics)
        if kind == "target":
            band = max(1.5, 0.05 * float(py1 - py0))
            _alpha_fill_rectangle(
                overlay,
                (xa, max(py0, y - band), xb, min(py1, y + band)),
                EQUALITY_BAND_COLOR,
            )
            draw.line((xa, y, xb, y), fill=color, width=heavy_line_width)
            return
        if kind == "upper":
            _alpha_fill_rectangle(
                overlay,
                (xa, max(py0, min(py1, y)), xb, py1),
                INEQUALITY_FEASIBLE_COLOR,
            )
        elif kind == "lower":
            _alpha_fill_rectangle(
                overlay,
                (xa, py0, xb, max(py0, min(py1, y))),
                INEQUALITY_FEASIBLE_COLOR,
            )
        _draw_dashed_segment(
            draw,
            (xa, y, xb, y),
            fill=color,
            width=line_width,
            dash=7.0,
            gap=4.0,
        )


def nvenc_available() -> bool:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        "color=size=1920x1080:rate=30",
        "-frames:v",
        "1",
        "-an",
        "-c:v",
        "h264_nvenc",
        "-f",
        "null",
        "-",
    ]
    return subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


class FFmpegVideoWriter:
    def __init__(
        self,
        path: Path,
        fps: float,
        size: tuple[int, int],
        crf: int,
        encoder: str,
    ):
        self.path = path
        self.size = (int(size[0]), int(size[1]))
        self.log_path = path.with_suffix(path.suffix + ".ffmpeg.log")
        self.log_stream = self.log_path.open("wb")
        requested_encoder = str(encoder).strip().lower()
        use_nvenc = requested_encoder == "nvenc" or (
            requested_encoder == "auto" and nvenc_available()
        )
        self.encoder = "h264_nvenc" if use_nvenc else "libx264"
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.size[0]}x{self.size[1]}",
            "-r",
            f"{float(fps):.6f}",
            "-i",
            "pipe:0",
            "-an",
        ]
        if use_nvenc:
            command.extend(
                [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "fast",
                    "-rc:v",
                    "vbr",
                    "-cq:v",
                    str(int(crf)),
                    "-b:v",
                    "0",
                ]
            )
        else:
            command.extend(
                [
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    str(int(crf)),
                ]
            )
        command.extend(
            [
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ]
        )
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self.log_stream,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError(f"FFmpeg stopped while writing {self.path}")
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise RuntimeError(
                f"rendered frame is {frame.shape[1]}x{frame.shape[0]}, "
                f"expected {self.size[0]}x{self.size[1]}"
            )
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError(f"cannot write rendered frame to FFmpeg: {error}") from error

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait(timeout=120.0)
        self.log_stream.close()
        if return_code:
            detail = self.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"FFmpeg failed with code {return_code}: {detail}")


def render_run(
    run_directory: Path,
    output: Path,
    *,
    fps: float,
    crf: int,
    encoder: str,
    panel_width_ratio: float,
    max_seconds: float | None,
) -> dict:
    video_path, visualization_path = _run_inputs(run_directory)
    visualization = _load_render_visualization(run_directory, visualization_path)
    info = probe_video(video_path)
    full_duration = float(info["duration"])
    if full_duration <= 0.0:
        raise RuntimeError(f"input video has invalid duration: {full_duration}")
    render_duration = full_duration
    if max_seconds is not None and max_seconds > 0.0:
        render_duration = min(render_duration, float(max_seconds))
    frame_times = load_frame_times(video_path)
    schedule, output_times = build_frame_schedule(frame_times, render_duration, fps)
    if not schedule:
        raise RuntimeError("render frame schedule is empty")
    panel = FeaturePanel(visualization, full_duration, run_directory.name)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open input video {video_path}")
    writer = FFmpegVideoWriter(
        output,
        float(fps),
        (int(info["width"]), int(info["height"])),
        int(crf),
        encoder,
    )
    source_index = 0
    schedule_index = 0
    frames_written = 0
    try:
        while schedule_index < len(schedule):
            ok, frame = capture.read()
            if not ok:
                break
            while (
                schedule_index < len(schedule)
                and schedule[schedule_index] == source_index
            ):
                rendered = panel.draw(
                    frame, output_times[schedule_index], panel_width_ratio
                )
                writer.write(rendered)
                frames_written += 1
                schedule_index += 1
            source_index += 1
    finally:
        capture.release()
        writer.close()
    if frames_written != len(schedule):
        raise RuntimeError(
            f"decoded video ended early: wrote {frames_written} of {len(schedule)} frames"
        )
    output_info = probe_video(output)
    return {
        "ok": True,
        "run": run_directory.name,
        "output": str(output),
        "width": output_info["width"],
        "height": output_info["height"],
        "fps": float(fps),
        "frames": frames_written,
        "duration": output_info["duration"],
        "features": panel.names,
        "stages": len(panel.windows),
        "planned_time_scale": panel.planned_time_scale,
        "timing_source": visualization.get("profile_synchronization", {}).get(
            "basis", "native"
        ),
        "encoder": writer.encoder,
    }


def render_final_frame_preview(
    run_directory: Path,
    output: Path,
    *,
    panel_width_ratio: float,
) -> dict:
    video_path, visualization_path = _run_inputs(run_directory)
    visualization = _load_render_visualization(run_directory, visualization_path)
    info = probe_video(video_path)
    duration_s = float(info["duration"])
    if duration_s <= 0.0:
        raise RuntimeError(f"input video has invalid duration: {duration_s}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open input video {video_path}")
    frame_count = max(int(info.get("nb_frames") or 0), 1)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ok, frame = capture.read()
    if not ok:
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, duration_s - 0.1) * 1000.0)
        ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"cannot decode the final frame of {video_path}")
    panel = FeaturePanel(visualization, duration_s, run_directory.name)
    timestamp_s = max(0.0, duration_s - 1.0 / 30.0)
    rendered = panel.draw(frame, timestamp_s, panel_width_ratio)
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), rendered):
        raise RuntimeError(f"cannot write preview image {output}")
    return {
        "ok": True,
        "run": run_directory.name,
        "output": str(output),
        "width": int(rendered.shape[1]),
        "height": int(rendered.shape[0]),
        "timestamp": timestamp_s,
        "features": panel.names,
        "stages": len(panel.windows),
    }


def main() -> None:
    options = build_parser().parse_args()
    if options.preview_image is not None:
        result = render_final_frame_preview(
            options.run_directory,
            options.preview_image,
            panel_width_ratio=options.panel_width_ratio,
        )
    else:
        result = render_run(
            options.run_directory,
            options.output,
            fps=options.fps,
            crf=options.crf,
            encoder=options.encoder,
            panel_width_ratio=options.panel_width_ratio,
            max_seconds=options.max_seconds,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
