from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import pybullet as p
    import pybullet_data
except ModuleNotFoundError:
    p = None
    pybullet_data = None


STAGE_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _save_figure(fig, path: str | Path, *, dpi: int = 220) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if plt is not None:
        plt.close(fig)
    return out_path


def _center_crop_to_aspect(arr: np.ndarray, aspect: float | None) -> np.ndarray:
    if aspect is None or float(aspect) <= 0.0:
        return arr
    h, w = int(arr.shape[0]), int(arr.shape[1])
    current = float(w) / max(float(h), 1e-12)
    target = float(aspect)
    if abs(current - target) <= 1e-6:
        return arr
    if current > target:
        new_w = max(1, int(round(float(h) * target)))
        left = max(0, (w - new_w) // 2)
        return arr[:, left:left + new_w]
    new_h = max(1, int(round(float(w) / target)))
    top = max(0, (h - new_h) // 2)
    return arr[top:top + new_h, :]


def _center_crop_scale(
    arr: np.ndarray,
    scale: float | None,
    *,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    if scale is None:
        return arr
    scale = float(scale)
    if scale <= 0.0 or scale >= 1.0:
        return arr
    h, w = int(arr.shape[0]), int(arr.shape[1])
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    max_left = max(0, w - new_w)
    max_top = max(0, h - new_h)
    left = int(round(0.5 * max_left + float(offset_x) * max_left))
    top = int(round(0.5 * max_top + float(offset_y) * max_top))
    left = int(np.clip(left, 0, max_left))
    top = int(np.clip(top, 0, max_top))
    return arr[top:top + new_h, left:left + new_w]


def _crop_bottom(arr: np.ndarray, fraction: float | None) -> np.ndarray:
    if fraction is None:
        return arr
    fraction = float(fraction)
    if fraction <= 0.0 or fraction >= 1.0:
        return arr
    h = int(arr.shape[0])
    new_h = max(1, int(round(float(h) * (1.0 - fraction))))
    return arr[:new_h, :]


def _save_rgb_frame(
    frame: np.ndarray,
    path: str | Path,
    *,
    crop_aspect: float | None = None,
    crop_scale: float | None = None,
    crop_offset_x: float = 0.0,
    crop_offset_y: float = 0.0,
    crop_bottom_fraction: float | None = None,
) -> Path:
    if Image is None:
        raise RuntimeError("Pillow is required to save rendered frame PNGs.")
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] < 3:
        raise ValueError("frame must have shape (H, W, 3+) to save as an RGB image.")
    arr = np.clip(arr[:, :, :3], 0, 255).astype(np.uint8)
    arr = _center_crop_to_aspect(arr, crop_aspect)
    arr = _center_crop_scale(arr, crop_scale, offset_x=float(crop_offset_x), offset_y=float(crop_offset_y))
    arr = _crop_bottom(arr, crop_bottom_fraction)
    Image.fromarray(arr, mode="RGB").save(out_path)
    return out_path


def stage_segments(length: int, cutpoints: Sequence[int] | None = None) -> list[tuple[int, int]]:
    T = max(int(length), 0)
    if T <= 0:
        return []
    if cutpoints is None:
        return [(0, T - 1)]
    cuts = np.asarray(cutpoints, dtype=int).reshape(-1)
    if cuts.size == 0:
        return [(0, T - 1)]
    cuts = np.sort(cuts[(cuts >= 0) & (cuts < T - 1)])
    ends = cuts.tolist() + [T - 1]
    starts = [0] + [int(v) + 1 for v in ends[:-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _constraint_semantics_kind(spec: dict) -> str:
    text = str(spec.get("semantics", "")).strip().lower()
    if text in {"target", "target_value", "equality", "eq", "equal"}:
        return "target"
    if text in {"upper", "upper_bound", "max", "maximum", "<=", "leq"}:
        return "upper"
    if text in {"lower", "lower_bound", "min", "minimum", ">=", "geq"}:
        return "lower"
    return text


def _dash_line(draw, xy, *, fill, width=1, dash=4, gap=3) -> None:
    x0, y0, x1, y1 = [float(v) for v in xy]
    length = float(np.hypot(x1 - x0, y1 - y0))
    if length <= 1e-9:
        draw.line((x0, y0, x1, y1), fill=fill, width=width)
        return
    ux, uy = (x1 - x0) / length, (y1 - y0) / length
    pos = 0.0
    while pos < length:
        end = min(pos + float(dash), length)
        draw.line((x0 + ux * pos, y0 + uy * pos, x0 + ux * end, y0 + uy * end), fill=fill, width=width)
        pos = end + float(gap)


def _overlay_feature_panel(
    frame: np.ndarray,
    *,
    features: np.ndarray,
    feature_names: Sequence[str],
    feature_units: dict[str, str] | None = None,
    current_index: int,
    cutpoints: Sequence[int] | None,
    constraint_specs: Sequence[dict] | None,
    true_constraints: dict | None,
    title: str,
) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return frame
    F_all = np.asarray(features, dtype=float)
    if F_all.ndim != 2 or F_all.shape[0] <= 0:
        return frame
    idx = int(np.clip(int(current_index), 0, F_all.shape[0] - 1))
    F = F_all[: idx + 1]
    names = [str(v) for v in feature_names]
    if not names:
        names = [f"feature_{i}" for i in range(F_all.shape[1])]
    name_to_idx = {name: i for i, name in enumerate(names[: F_all.shape[1]])}
    specs = [dict(s) for s in (constraint_specs or []) if str(s.get("feature_name", "")) in name_to_idx]
    shown = []
    for name in names:
        if name in name_to_idx:
            shown.append(name)
    if not shown:
        return frame
    units = {str(k): str(v) for k, v in dict(feature_units or {}).items() if str(v)}

    def label_for(name: str) -> str:
        unit = units.get(str(name), "")
        return str(name) if not unit else f"{name} [{unit}]"

    spans = stage_segments(F_all.shape[0], cutpoints=cutpoints)
    height, width = int(frame.shape[0]), int(frame.shape[1])
    rows = len(shown)
    y_margin = int(max(10, round(0.018 * height)))
    panel_w = int(min(max(520, 0.36 * width), max(width - 2 * y_margin, 1), 620))
    title_text = str(title or "Executed trajectory feature profile")
    if " (planned with " in title_text:
        first, rest = title_text.split(" (planned with ", 1)
        title_lines = [first, "(planned with " + rest]
    else:
        title_lines = [title_text]
    legend_rows = 3
    target_panel_h = int(max(240, height - 2 * y_margin))
    estimated_header_h = (
        88
        + 19 * max(0, len(title_lines) - 1)
        + 19 * max(0, legend_rows - 2)
        + (28 if len(spans) > 1 else 0)
    )
    row_h = int(np.clip((target_panel_h - estimated_header_h - 14) / max(rows, 1), 56, 112))
    panel_scale = float(np.clip(row_h / 72.0, 1.15, 1.55))
    font_size = int(max(14, round(12 * panel_scale)))
    title_font_size = int(max(15, round(13 * panel_scale)))
    title_line_h = int(round(title_font_size + 5))
    legend_line_h = int(round(font_size + 7))
    pad = int(max(12, round(9 * panel_scale)))
    bottom_pad = int(max(12, round(8 * panel_scale)))
    stage_header_h = int(max(22, round(19 * panel_scale))) if len(spans) > 1 else 0
    header_h = (
        pad
        + len(title_lines) * title_line_h
        + 5
        + legend_rows * legend_line_h
        + (stage_header_h + 8 if stage_header_h > 0 else 4)
    )
    row_h = int(np.clip((target_panel_h - header_h - bottom_pad) / max(rows, 1), 56, 112))
    panel_h = int(min(target_panel_h, header_h + rows * row_h + bottom_pad))
    x0 = max(8, width - panel_w - y_margin)
    y0 = y_margin
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", title_font_size)
    except Exception:
        font = ImageFont.load_default() if ImageFont is not None else None
        font_bold = font
        font_title = font

    line_w_thin = int(max(1, round(1 * panel_scale)))
    line_w_mid = int(max(3, round(2 * panel_scale)))
    line_w_heavy = int(max(4, round(3 * panel_scale)))
    dash_len = int(max(7, round(6 * panel_scale)))
    dash_gap = int(max(4, round(4 * panel_scale)))
    stage_band_h = int(max(16, round(13 * panel_scale)))

    draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=int(round(7 * panel_scale)), fill=(255, 255, 255, 218), outline=(36, 42, 50, 185), width=line_w_thin)
    for line_idx, line in enumerate(title_lines):
        draw.text((x0 + pad, y0 + max(6, pad // 2) + title_line_h * line_idx), line, fill=(20, 24, 30, 255), font=font_title)
    constraint_orange = (234, 88, 12, 245)
    constraint_orange_text = (194, 65, 12, 255)
    feasible_yellow = (254, 240, 138, 145)
    equality_band = (253, 186, 116, 92)
    legend_y = y0 + max(6, pad // 2) + len(title_lines) * title_line_h + 4
    legend_x = x0 + pad
    legend_sample_w = int(round(28 * panel_scale))
    legend_mid_y = legend_y + int(round(0.52 * font_size))
    draw.line((legend_x, legend_mid_y, legend_x + legend_sample_w, legend_mid_y), fill=constraint_orange, width=line_w_heavy)
    draw.text((legend_x + legend_sample_w + int(round(8 * panel_scale)), legend_y), "Ground truth equality constraint target", fill=constraint_orange_text, font=font)
    legend_y2 = legend_y + legend_line_h
    legend_mid_y2 = legend_y2 + int(round(0.52 * font_size))
    draw.rectangle((legend_x, legend_y2 + int(round(2 * panel_scale)), legend_x + legend_sample_w, legend_y2 + int(round(12 * panel_scale))), fill=feasible_yellow)
    _dash_line(draw, (legend_x, legend_mid_y2, legend_x + legend_sample_w, legend_mid_y2), fill=constraint_orange, width=line_w_mid, dash=dash_len, gap=dash_gap)
    legend_text_x = legend_x + legend_sample_w + int(round(8 * panel_scale))
    draw.text((legend_text_x, legend_y2), "Ground truth inequality constraint bound", fill=constraint_orange_text, font=font)
    draw.text((legend_text_x, legend_y2 + legend_line_h), "and feasible region", fill=constraint_orange_text, font=font)

    true_constraints = dict(true_constraints or {})
    try:
        max_label_w = max(draw.textbbox((0, 0), label_for(name), font=font_bold)[2] for name in shown)
    except Exception:
        max_label_w = int(round(108 * panel_scale))
    plot_x0 = x0 + int(min(max(max_label_w + 24, 120 * panel_scale), 0.38 * panel_w))
    plot_x1 = x0 + panel_w - pad
    plot_w = max(1, plot_x1 - plot_x0)
    total_den = max(F_all.shape[0] - 1, 1)
    if stage_header_h > 0:
        stage_y0 = y0 + header_h - stage_header_h + 2
        for stage_idx, (start, end) in enumerate(spans):
            xa = plot_x0 + float(start) / float(total_den) * float(plot_w)
            xb = plot_x0 + float(end) / float(total_den) * float(plot_w)
            rgba = _hex_to_rgba(STAGE_COLORS[stage_idx % len(STAGE_COLORS)], alpha=1.0)
            fill = tuple(int(np.clip(v * 255.0, 0, 255)) for v in (*rgba[:3], 0.18))
            stroke = tuple(int(np.clip(v * 255.0, 0, 255)) for v in (*rgba[:3], 0.90))
            draw.rectangle((xa, stage_y0, xb, stage_y0 + stage_band_h), fill=fill, outline=stroke, width=line_w_thin)
            label = f"s{stage_idx + 1}"
            try:
                bbox = draw.textbbox((0, 0), label, font=font_bold)
                tw = float(bbox[2] - bbox[0])
                th = float(bbox[3] - bbox[1])
            except Exception:
                tw, th = 34.0, 9.0
            tx = min(max(xa + 2.0, 0.5 * (xa + xb) - 0.5 * tw), max(xa + 2.0, xb - tw - 2.0))
            draw.text((tx, stage_y0 + max(0.0, 0.5 * (float(stage_band_h) - th)) - 1.0), label, fill=(18, 24, 38, 245), font=font_bold)
    for row, name in enumerate(shown):
        feat_idx = int(name_to_idx[name])
        ry0 = y0 + header_h + row * row_h
        ry1 = ry0 + row_h
        py0 = ry0 + int(round(6 * panel_scale))
        py1 = ry1 - int(round(8 * panel_scale))
        full = np.asarray(F_all[:, feat_idx], dtype=float)
        trace = np.asarray(F[:, feat_idx], dtype=float)
        finite_vals = full[np.isfinite(full)]
        spec_vals = []
        for spec in specs:
            if str(spec.get("feature_name", "")) != name:
                continue
            key = str(spec.get("oracle_key", ""))
            if key in true_constraints and np.isfinite(float(true_constraints[key])):
                spec_vals.append(float(true_constraints[key]))
        all_vals = np.concatenate([finite_vals, np.asarray(spec_vals, dtype=float)]) if spec_vals else finite_vals
        if all_vals.size == 0:
            continue
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        if abs(vmax - vmin) < 1e-8:
            margin = max(1e-4, abs(vmax) * 0.15 + 1e-4)
        else:
            margin = 0.12 * (vmax - vmin)
        vmin -= margin
        vmax += margin

        def x_at(t: int) -> float:
            return float(plot_x0) + float(np.clip(t, 0, total_den)) / float(total_den) * float(plot_w)

        def y_at(v: float) -> float:
            return float(py1) - (float(v) - vmin) / max(vmax - vmin, 1e-12) * float(py1 - py0)

        label_x = x0 + pad
        label_y = ry0 + int(round(8 * panel_scale))
        draw.text((label_x, label_y), label_for(name), fill=(18, 24, 38, 255), font=font_bold)
        draw.text((label_x, label_y + font_size + int(round(4 * panel_scale))), f"{trace[-1]:.3g}", fill=(37, 99, 235, 255), font=font)
        draw.rectangle((plot_x0, py0, plot_x1, py1), outline=(188, 196, 206, 220), fill=(248, 250, 252, 205), width=line_w_thin)
        for cp in np.asarray(cutpoints if cutpoints is not None else [], dtype=int).reshape(-1):
            x = x_at(int(cp))
            draw.line((x, py0, x, py1), fill=(150, 158, 170, 150), width=line_w_thin)
        for spec in specs:
            if str(spec.get("feature_name", "")) != name:
                continue
            stage_idx = int(spec.get("stage", -1))
            if stage_idx < 0 or stage_idx >= len(spans):
                continue
            key = str(spec.get("oracle_key", ""))
            if key not in true_constraints:
                continue
            value = float(true_constraints[key])
            if not np.isfinite(value):
                continue
            a, b = spans[stage_idx]
            y = y_at(value)
            xa, xb = x_at(a), x_at(b)
            kind = _constraint_semantics_kind(spec)
            if kind == "target":
                band = max(1.5, 0.05 * float(py1 - py0))
                draw.rectangle((xa, max(py0, y - band), xb, min(py1, y + band)), fill=equality_band)
                draw.line((xa, y, xb, y), fill=constraint_orange, width=line_w_heavy)
            elif kind == "upper":
                draw.rectangle((xa, max(py0, min(py1, y)), xb, py1), fill=feasible_yellow)
                _dash_line(draw, (xa, y, xb, y), fill=constraint_orange, width=line_w_mid, dash=dash_len, gap=dash_gap)
            elif kind == "lower":
                draw.rectangle((xa, py0, xb, max(py0, min(py1, y))), fill=feasible_yellow)
                _dash_line(draw, (xa, y, xb, y), fill=constraint_orange, width=line_w_mid, dash=dash_len, gap=dash_gap)
        if trace.size >= 2:
            pts = [(x_at(i), y_at(float(v))) for i, v in enumerate(trace) if np.isfinite(float(v))]
            if len(pts) >= 2:
                draw.line(pts, fill=(37, 99, 235, 255), width=line_w_mid)
        cx = x_at(idx)
        draw.line((cx, py0, cx, py1), fill=(99, 102, 241, 210), width=line_w_thin)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def _overlay_corner_label(frame: np.ndarray, text: str | None) -> np.ndarray:
    label = "" if text is None else str(text).strip()
    if not label or Image is None or ImageDraw is None:
        return frame
    height, width = int(frame.shape[0]), int(frame.shape[1])
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(np.clip(round(height * 0.026), 18, 30)))
    except Exception:
        font = ImageFont.load_default() if ImageFont is not None else None
    margin = int(max(16, round(min(width, height) * 0.022)))
    x0 = margin
    try:
        bbox = draw.textbbox((0, 0), label, font=font)
        text_h = int(bbox[3] - bbox[1])
    except Exception:
        _text_w, text_h = draw.textsize(label, font=font)
    y0 = max(0, height - margin - text_h)
    draw.text((x0, y0), label, fill=(0, 0, 0, 255), font=font)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def _camera_matrices(
    *,
    yaw_deg: float,
    target: np.ndarray,
    distance: float,
    width: int,
    height: int,
    pitch_deg: float,
    fov: float,
) -> tuple[list[float], list[float]]:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=np.asarray(target, dtype=float).tolist(),
        distance=float(distance),
        yaw=float(yaw_deg),
        pitch=float(pitch_deg),
        roll=0.0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=float(fov),
        aspect=float(width) / float(height),
        nearVal=0.05,
        farVal=8.0,
    )
    return view, proj


def _project_world_to_pixel(
    world_pos: np.ndarray,
    *,
    view_matrix: Sequence[float],
    projection_matrix: Sequence[float],
    width: int,
    height: int,
) -> tuple[float, float, bool]:
    pos = np.ones(4, dtype=float)
    pos[:3] = np.asarray(world_pos, dtype=float).reshape(3)
    view = np.asarray(view_matrix, dtype=float).reshape(4, 4, order="F")
    proj = np.asarray(projection_matrix, dtype=float).reshape(4, 4, order="F")
    clip = proj @ (view @ pos)
    if abs(float(clip[3])) < 1e-9:
        return 0.0, 0.0, False
    ndc = clip[:3] / float(clip[3])
    visible = bool(-1.15 <= ndc[0] <= 1.15 and -1.15 <= ndc[1] <= 1.15 and -1.15 <= ndc[2] <= 1.15)
    x = (float(ndc[0]) + 1.0) * 0.5 * float(width)
    y = (1.0 - float(ndc[1])) * 0.5 * float(height)
    return x, y, visible


def _overlay_trajectory_stage_labels(
    frame: np.ndarray,
    *,
    labels: Sequence[dict],
    view_matrix: Sequence[float],
    projection_matrix: Sequence[float],
) -> np.ndarray:
    if Image is None or ImageDraw is None or not labels:
        return frame
    height, width = int(frame.shape[0]), int(frame.shape[1])
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(np.clip(round(height * 0.018), 14, 22)))
    except Exception:
        font = ImageFont.load_default() if ImageFont is not None else None
    used_boxes: list[tuple[float, float, float, float]] = []
    for item in labels:
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        anchor_x, anchor_y, visible = _project_world_to_pixel(
            np.asarray(item.get("pos"), dtype=float),
            view_matrix=view_matrix,
            projection_matrix=projection_matrix,
            width=width,
            height=height,
        )
        if not visible:
            continue
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = float(bbox[2] - bbox[0])
            text_h = float(bbox[3] - bbox[1])
        except Exception:
            text_w, text_h = draw.textsize(text, font=font)
        pad_x = 7.0
        pad_y = 4.0
        box_w = text_w + 2.0 * pad_x
        box_h = text_h + 2.0 * pad_y
        center_pos = item.get("center_pos")
        if center_pos is not None:
            center_x, center_y, center_visible = _project_world_to_pixel(
                np.asarray(center_pos, dtype=float),
                view_matrix=view_matrix,
                projection_matrix=projection_matrix,
                width=width,
                height=height,
            )
        else:
            center_x, center_y, center_visible = 0.5 * float(width), 0.5 * float(height), False
        direction = np.asarray([anchor_x - center_x, anchor_y - center_y], dtype=float)
        norm = float(np.linalg.norm(direction))
        if not center_visible or norm < 1e-6:
            angle = 2.0 * np.pi * float(len(used_boxes)) / max(float(len(labels)), 1.0)
            direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
        else:
            direction = direction / norm
        tangent = np.asarray([-direction[1], direction[0]], dtype=float)
        best_box = None
        best_score = float("inf")

        def _overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b
            pad = 5.0
            ax0 -= pad
            ay0 -= pad
            ax1 += pad
            ay1 += pad
            bx0 -= pad
            by0 -= pad
            bx1 += pad
            by1 += pad
            ow = max(0.0, min(ax1, bx1) - max(ax0, bx0))
            oh = max(0.0, min(ay1, by1) - max(ay0, by0))
            return float(ow * oh)

        for radial_dist in (76.0, 102.0, 130.0, 158.0):
            for tangent_dist in (0.0, -34.0, 34.0, -68.0, 68.0, -102.0, 102.0):
                label_center_x = anchor_x + radial_dist * float(direction[0]) + tangent_dist * float(tangent[0])
                label_center_y = anchor_y + radial_dist * float(direction[1]) + tangent_dist * float(tangent[1])
                x0_c = float(np.clip(label_center_x - 0.5 * box_w, 4.0, max(4.0, width - box_w - 4.0)))
                y0_c = float(np.clip(label_center_y - 0.5 * box_h, 4.0, max(4.0, height - box_h - 4.0)))
                box = (x0_c, y0_c, x0_c + box_w, y0_c + box_h)
                overlap = sum(_overlap_area(box, prev) for prev in used_boxes)
                clipped = abs(x0_c - (label_center_x - 0.5 * box_w)) + abs(y0_c - (label_center_y - 0.5 * box_h))
                score = 10000.0 * overlap + 4.0 * clipped + 0.55 * abs(tangent_dist) + 0.18 * radial_dist
                if score < best_score:
                    best_score = float(score)
                    best_box = box
        if best_box is None:
            best_box = (
                float(np.clip(anchor_x + 12.0, 4.0, max(4.0, width - box_w - 4.0))),
                float(np.clip(anchor_y - 0.5 * box_h, 4.0, max(4.0, height - box_h - 4.0))),
                0.0,
                0.0,
            )
            best_box = (best_box[0], best_box[1], best_box[0] + box_w, best_box[1] + box_h)
        x0, y0, x1, y1 = best_box
        used_boxes.append((x0, y0, x1, y1))
        color = tuple(int(v) for v in item.get("color", (31, 41, 55, 255)))
        outline = color[:3] + (230,)
        fill = (255, 255, 255, 220)
        if x0 > anchor_x:
            line_end = (x0, 0.5 * (y0 + y1))
        else:
            line_end = (x1, 0.5 * (y0 + y1))
        draw.line((anchor_x, anchor_y, line_end[0], line_end[1]), fill=outline, width=2)
        r = 3.0
        draw.ellipse((anchor_x - r, anchor_y - r, anchor_x + r, anchor_y + r), fill=outline)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=5, fill=fill, outline=outline, width=2)
        draw.text((x0 + pad_x, y0 + pad_y - 1.0), text, fill=(15, 23, 42, 255), font=font)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for env.render_episode().")


def _require_pybullet() -> None:
    if p is None or pybullet_data is None:
        raise RuntimeError("pybullet is required for env.render_episode(..., backend='pybullet').")


class _FFmpegVideoWriter:
    def __init__(self, *, out_path: str | Path, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg binary not found in PATH")
        self.out_path = str(Path(out_path).resolve())
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(float(fps), 0.1))
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            self.out_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def append_data(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        arr = np.asarray(frame, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if arr.shape != expected:
            raise ValueError(f"video frame has shape {arr.shape}, expected {expected}")
        self.proc.stdin.write(arr.tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        self.proc.wait()
        if self.proc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg exited with code {self.proc.returncode}")


def render_planar_episode(
    *,
    trajectory: np.ndarray,
    output_path: str | Path,
    cutpoints: Sequence[int] | None = None,
    title: str | None = None,
    obstacles: Iterable[dict] | None = None,
    reference_lines: Iterable[dict] | None = None,
    markers: Iterable[dict] | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    equal_aspect: bool = True,
) -> Path:
    _require_matplotlib()
    pts = np.asarray(trajectory, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("trajectory must have shape (T, 2+) for planar rendering.")

    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=False)

    if obstacles is not None:
        for obs in obstacles:
            center = np.asarray(obs.get("center", [0.0, 0.0]), dtype=float).reshape(2)
            radius = float(obs.get("radius", 0.1))
            facecolor = str(obs.get("facecolor", "#CBD5E1"))
            edgecolor = str(obs.get("edgecolor", "#475569"))
            alpha = float(obs.get("alpha", 0.32))
            circle = plt.Circle(center, radius, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=1.2)
            ax.add_patch(circle)

    if reference_lines is not None:
        for line in reference_lines:
            point = np.asarray(line.get("point", [0.0, 0.0]), dtype=float).reshape(2)
            direction = np.asarray(line.get("direction", [1.0, 0.0]), dtype=float).reshape(2)
            span = float(line.get("span", 4.0))
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            direction = direction / norm
            endpoints = np.vstack([point - span * direction, point + span * direction])
            ax.plot(
                endpoints[:, 0],
                endpoints[:, 1],
                linestyle=str(line.get("linestyle", "--")),
                linewidth=float(line.get("linewidth", 1.0)),
                color=str(line.get("color", "#64748B")),
                alpha=float(line.get("alpha", 0.7)),
            )

    segments = stage_segments(len(pts), cutpoints=cutpoints)
    for stage_idx, (start, end) in enumerate(segments):
        seg = pts[start : end + 1, :2]
        color = STAGE_COLORS[stage_idx % len(STAGE_COLORS)]
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=1.8, alpha=0.96)
        ax.scatter(seg[:, 0], seg[:, 1], color=color, s=10, alpha=0.24)

    ax.scatter(pts[0, 0], pts[0, 1], color="#111827", marker="o", s=24, zorder=6)
    ax.scatter(pts[-1, 0], pts[-1, 1], color="#111827", marker="s", s=24, zorder=6)

    if cutpoints is not None:
        for cp in np.asarray(cutpoints, dtype=int).reshape(-1):
            if 0 <= int(cp) < len(pts):
                ax.scatter(pts[int(cp), 0], pts[int(cp), 1], color="#111827", marker="x", s=36, linewidths=1.4, zorder=7)

    if markers is not None:
        for marker in markers:
            point = np.asarray(marker.get("point", [0.0, 0.0]), dtype=float).reshape(2)
            ax.scatter(
                point[0],
                point[1],
                color=str(marker.get("color", "#1D4ED8")),
                marker=str(marker.get("marker", "o")),
                s=float(marker.get("size", 26.0)),
                alpha=float(marker.get("alpha", 0.95)),
                zorder=8,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(str(title), fontsize=10)
    ax.grid(alpha=0.18)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    mins = np.min(pts[:, :2], axis=0)
    maxs = np.max(pts[:, :2], axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    margin = 0.18 * span
    ax.set_xlim(float(mins[0] - margin[0]), float(maxs[0] + margin[0]))
    ax.set_ylim(float(mins[1] - margin[1]), float(maxs[1] + margin[1]))

    fig.tight_layout(pad=0.25)
    return _save_figure(fig, output_path, dpi=220)


def render_sphere_episode(
    *,
    trajectory: np.ndarray,
    output_path: str | Path,
    sphere_center: Sequence[float],
    sphere_radius: float,
    cutpoints: Sequence[int] | None = None,
    title: str | None = None,
    elev: float = 24.0,
    azim: float = 38.0,
) -> Path:
    _require_matplotlib()
    pts = np.asarray(trajectory, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("trajectory must have shape (T, 3+) for sphere rendering.")

    fig = plt.figure(figsize=(4.6, 3.9), constrained_layout=False)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    center = np.asarray(sphere_center, dtype=float).reshape(3)
    radius = float(sphere_radius)
    th = np.linspace(0.0, 2.0 * np.pi, 28)
    ph = np.linspace(0.0, np.pi, 18)
    th, ph = np.meshgrid(th, ph)
    xx = center[0] + radius * np.cos(th) * np.sin(ph)
    yy = center[1] + radius * np.sin(th) * np.sin(ph)
    zz = center[2] + radius * np.cos(ph)
    ax.plot_wireframe(xx, yy, zz, color="#94A3B8", alpha=0.26, linewidth=0.6, rstride=1, cstride=1)

    segments = stage_segments(len(pts), cutpoints=cutpoints)
    for stage_idx, (start, end) in enumerate(segments):
        seg = pts[start : end + 1, :3]
        color = STAGE_COLORS[stage_idx % len(STAGE_COLORS)]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=1.8, alpha=0.98)
        ax.scatter(seg[:, 0], seg[:, 1], seg[:, 2], color=color, s=8.0, alpha=0.22, depthshade=False)

    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="#111827", marker="o", s=26, depthshade=False)
    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color="#111827", marker="s", s=26, depthshade=False)
    if cutpoints is not None:
        for cp in np.asarray(cutpoints, dtype=int).reshape(-1):
            if 0 <= int(cp) < len(pts):
                ax.scatter(
                    pts[int(cp), 0],
                    pts[int(cp), 1],
                    pts[int(cp), 2],
                    color="#111827",
                    marker="x",
                    s=36,
                    linewidths=1.4,
                    depthshade=False,
                )

    corners = np.array(
        [
            center + np.array([sx, sy, sz], dtype=float) * radius
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )
    all_pts = np.vstack([pts[:, :3], corners])
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    center_box = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.55 * max(span, 1e-3)
    ax.set_xlim(center_box[0] - half, center_box[0] + half)
    ax.set_ylim(center_box[1] - half, center_box[1] + half)
    ax.set_zlim(center_box[2] - half, center_box[2] + half)
    ax.view_init(elev=float(elev), azim=float(azim))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(str(title), fontsize=10)

    fig.tight_layout(pad=0.25)
    return _save_figure(fig, output_path, dpi=220)


def _normalize3(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _hex_to_rgba(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    text = str(color).lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected 6-digit hex color, got '{color}'.")
    rgb = tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))


def _quat_align_z_to_vec(vec: np.ndarray) -> tuple[float, float, float, float]:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    target = _normalize3(vec)
    dot = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))
    if dot >= 1.0 - 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    if dot <= -1.0 + 1e-8:
        return tuple(p.getQuaternionFromEuler((np.pi, 0.0, 0.0)))
    axis = _normalize3(np.cross(z_axis, target))
    angle = float(np.arccos(dot))
    return tuple(p.getQuaternionFromAxisAngle(axis.tolist(), angle))


def _env_to_world(points: np.ndarray, sphere_center: np.ndarray, center_world: np.ndarray, scale: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return center_world[None, :] + float(scale) * (pts - np.asarray(sphere_center, dtype=float)[None, :])


def _spawn_table(
    table_top_z: float,
    *,
    center_xy: Sequence[float] = (0.0, 0.0),
    half_extents_xy: Sequence[float] = (0.54, 0.54),
) -> None:
    center_xy = np.asarray(center_xy, dtype=float).reshape(2)
    half_extents_xy = np.asarray(half_extents_xy, dtype=float).reshape(2)
    half_extents = [float(half_extents_xy[0]), float(half_extents_xy[1]), 0.028]
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.78, 0.79, 0.78, 1.0],
        specularColor=[0.22, 0.22, 0.22],
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[float(center_xy[0]), float(center_xy[1]), table_top_z - half_extents[2]],
    )
    p.changeVisualShape(body_id, -1, rgbaColor=[0.82, 0.82, 0.80, 1.0])
    leg_half = [0.03, 0.03, 0.30]
    leg_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=leg_half)
    leg_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=leg_half,
        rgbaColor=[0.30, 0.27, 0.24, 1.0],
        specularColor=[0.05, 0.05, 0.05],
    )
    leg_margin = 0.10
    for sx in (-half_extents[0] + leg_margin, half_extents[0] - leg_margin):
        for sy in (-half_extents[1] + leg_margin, half_extents[1] - leg_margin):
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=leg_col,
                baseVisualShapeIndex=leg_vis,
                basePosition=[
                    float(center_xy[0] + sx),
                    float(center_xy[1] + sy),
                    table_top_z - 2.0 * half_extents[2] - leg_half[2],
                ],
            )


def _spawn_ground_grid(
    *,
    center_xy: Sequence[float] = (0.0, 0.0),
    half_extents_xy: Sequence[float] = (1.25, 1.00),
    spacing: float = 0.10,
    z: float = 0.0,
    z_lift: float = 0.003,
) -> None:
    floor_id = int(p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, float(z) - 0.02], useFixedBase=True))
    p.changeVisualShape(floor_id, -1, rgbaColor=[0.96, 0.97, 0.99, 1.0])


def _spawn_sphere(
    center_world: np.ndarray,
    radius_world: float,
    *,
    shell_radius_world: float | None = None,
    texture_name: str = "",
    draw_shell_surface: bool = False,
    draw_surface_rings: bool = False,
) -> None:
    data_root = Path(pybullet_data.getDataPath())
    sphere_mesh = data_root / "sphere_smooth.obj"
    col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius_world)
    use_texture = bool(str(texture_name).strip())
    outer_vis = p.createVisualShape(
        p.GEOM_MESH,
        fileName=str(sphere_mesh),
        meshScale=[radius_world, radius_world, radius_world],
        rgbaColor=([0.96, 0.98, 1.00, 1.0] if use_texture else [0.63, 0.75, 0.88, 1.0]),
        specularColor=[0.48, 0.52, 0.58],
    )
    sphere_body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=outer_vis,
        basePosition=center_world.tolist(),
    )
    texture_path = data_root / str(texture_name)
    if use_texture and texture_path.exists():
        try:
            tex_id = p.loadTexture(str(texture_path))
            p.changeVisualShape(sphere_body, -1, textureUniqueId=tex_id, rgbaColor=[1.0, 1.0, 1.0, 1.0])
        except Exception:
            pass

    def _draw_sphere_rings(
        radius: float,
        color: tuple[float, float, float, float],
        ring_scale: float,
        *,
        ring_axes: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> None:
        ring_radius = max(0.0008, float(ring_scale) * float(radius_world))
        axes = ring_axes or (
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
            (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        )
        for basis_a, basis_b in axes:
            theta = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=True)
            ring = center_world[None, :] + float(radius) * (
                np.cos(theta)[:, None] * basis_a[None, :] + np.sin(theta)[:, None] * basis_b[None, :]
            )
            for a, b in zip(ring[:-1], ring[1:]):
                _spawn_capsule_segment(a, b, radius=ring_radius, color=color)

    if bool(draw_surface_rings):
        _draw_sphere_rings(
            1.004 * float(radius_world),
            color=(0.20, 0.30, 0.42, 0.18),
            ring_scale=0.0017,
        )
    if shell_radius_world is not None and float(shell_radius_world) > float(radius_world) + 1e-5:
        if bool(draw_shell_surface):
            shell_vis = p.createVisualShape(
                p.GEOM_MESH,
                fileName=str(sphere_mesh),
                meshScale=[float(shell_radius_world), float(shell_radius_world), float(shell_radius_world)],
                rgbaColor=[0.28, 0.58, 0.95, 0.065],
                specularColor=[0.45, 0.55, 0.70],
            )
            p.createMultiBody(
                baseMass=0.0,
                baseVisualShapeIndex=shell_vis,
                basePosition=center_world.tolist(),
            )
        _draw_sphere_rings(
            float(shell_radius_world),
            color=(0.05, 0.25, 0.62, 0.42),
            ring_scale=0.0022,
            ring_axes=(
                (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
                (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])),
            ),
        )
    stand_radius = 0.42 * float(radius_world)
    stand_height = 0.13 * float(radius_world)
    stand_vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=stand_radius,
        length=stand_height,
        rgbaColor=[0.40, 0.43, 0.45, 1.0],
        specularColor=[0.38, 0.38, 0.38],
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=stand_vis,
        basePosition=[float(center_world[0]), float(center_world[1]), float(center_world[2] - radius_world - 0.5 * stand_height)],
    )


def _spawn_marker(pos_world: np.ndarray, radius: float, color: tuple[float, float, float, float]) -> int:
    vis_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=list(color),
        specularColor=[0.2, 0.2, 0.2],
    )
    return int(p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=vis_id,
        basePosition=np.asarray(pos_world, dtype=float).tolist(),
    ))


def _spawn_oriented_cylinder(
    pos_world: np.ndarray,
    axis_world: np.ndarray,
    length: float,
    radius: float,
    color: tuple[float, float, float, float],
) -> None:
    vis_id = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(radius),
        length=float(length),
        rgbaColor=list(color),
        specularColor=[0.20, 0.20, 0.20],
    )
    orn = _quat_align_z_to_vec(axis_world)
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=vis_id,
        basePosition=np.asarray(pos_world, dtype=float).tolist(),
        baseOrientation=orn,
    )


def _spawn_probe_pose(
    pos_world: np.ndarray,
    axis_world: np.ndarray,
    shaft_len: float = 0.080,
    shaft_radius: float = 0.0045,
    tip_len: float = 0.020,
    tip_radius: float = 0.0075,
) -> None:
    axis = _normalize3(axis_world)
    shaft_center = np.asarray(pos_world, dtype=float) - 0.5 * float(shaft_len) * axis
    _spawn_oriented_cylinder(
        pos_world=shaft_center,
        axis_world=axis,
        length=shaft_len,
        radius=shaft_radius,
        color=(0.18, 0.20, 0.24, 1.0),
    )
    collar_center = np.asarray(pos_world, dtype=float) - 0.12 * float(shaft_len) * axis
    _spawn_oriented_cylinder(
        pos_world=collar_center,
        axis_world=axis,
        length=0.018,
        radius=0.0065,
        color=(0.12, 0.46, 0.84, 1.0),
    )
    tip_center = np.asarray(pos_world, dtype=float) + 0.5 * float(tip_len) * axis
    _spawn_oriented_cylinder(
        pos_world=tip_center,
        axis_world=axis,
        length=tip_len,
        radius=tip_radius,
        color=(0.88, 0.64, 0.18, 1.0),
    )


def _spawn_capsule_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    color: tuple[float, float, float, float],
) -> None:
    vec = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    seg_len = float(np.linalg.norm(vec))
    if seg_len <= 1e-8:
        return
    cyl_len = max(seg_len - 2.0 * float(radius), 1e-4)
    vis_id = p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=float(radius),
        length=cyl_len,
        rgbaColor=list(color),
        specularColor=[0.18, 0.18, 0.18],
    )
    midpoint = 0.5 * (np.asarray(p0, dtype=float) + np.asarray(p1, dtype=float))
    orn = _quat_align_z_to_vec(vec)
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=vis_id,
        basePosition=midpoint.tolist(),
        baseOrientation=orn,
    )


def _render_rgb(
    *,
    yaw_deg: float,
    target: np.ndarray,
    distance: float,
    width: int,
    height: int,
    pitch_deg: float = -23.0,
    fov: float = 37.0,
) -> np.ndarray:
    view, proj = _camera_matrices(
        yaw_deg=float(yaw_deg),
        target=np.asarray(target, dtype=float),
        distance=float(distance),
        width=int(width),
        height=int(height),
        pitch_deg=float(pitch_deg),
        fov=float(fov),
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        lightDirection=[1.8, -1.1, 2.8],
        shadow=1,
    )
    rgba = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3]


def _load_ur5_render_robot(
    *,
    urdf_path: str | None,
    base_xyz: Sequence[float],
    base_rpy: Sequence[float],
    hide_link_geometry_patterns: Sequence[str] | None = None,
    suppress_urdf_warnings: bool = True,
) -> tuple[int, list[int], str | None]:
    from .pybullet_ur5 import _DEFAULT_UR5_URDF, _make_pybullet_friendly_urdf, _suppress_native_output

    path = str(urdf_path or _DEFAULT_UR5_URDF)
    if not os.path.exists(path):
        raise RuntimeError(f"UR5 URDF not found: {path}")
    load_path = path
    patched_path = None
    with open(path, "r", encoding="utf-8") as f:
        if "package://" in f.read():
            patched_path = _make_pybullet_friendly_urdf(path)
            load_path = patched_path
    patterns = [str(v).lower() for v in (hide_link_geometry_patterns or []) if str(v).strip()]
    if patterns:
        stripped_path = _make_urdf_without_link_geometry(load_path, patterns)
        if patched_path:
            try:
                os.remove(patched_path)
            except OSError:
                pass
        patched_path = stripped_path
        load_path = stripped_path
    with _suppress_native_output(bool(suppress_urdf_warnings)):
        robot_id = int(
            p.loadURDF(
                load_path,
                basePosition=np.asarray(base_xyz, dtype=float).reshape(3).tolist(),
                baseOrientation=p.getQuaternionFromEuler(np.asarray(base_rpy, dtype=float).reshape(3).tolist()),
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
            )
        )
    arm_joint_indices: list[int] = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if int(info[2]) == p.JOINT_REVOLUTE:
            arm_joint_indices.append(int(j))
    if len(arm_joint_indices) < 6:
        raise RuntimeError(f"UR5 model has fewer than 6 revolute joints: {len(arm_joint_indices)}")
    return robot_id, arm_joint_indices[:6], patched_path


def _make_urdf_without_link_geometry(urdf_path: str, name_patterns: Sequence[str]) -> str:
    patterns = [str(v).lower() for v in name_patterns if str(v).strip()]
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        name = str(link.attrib.get("name", "")).lower()
        if not any(pattern in name for pattern in patterns):
            continue
        for child in list(link):
            if child.tag in {"visual", "collision"}:
                link.remove(child)
    fd, tmp = tempfile.mkstemp(prefix="s5_ur5_render_hidden_", suffix=".urdf")
    os.close(fd)
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    return tmp


def _hide_robot_links_by_name(robot_id: int, name_patterns: Sequence[str]) -> None:
    patterns = [str(v).lower() for v in name_patterns]
    if not patterns:
        return
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        link_name = info[12].decode("utf-8", errors="ignore").lower()
        if not any(pattern in link_name for pattern in patterns):
            continue
        try:
            p.changeVisualShape(robot_id, j, rgbaColor=[0.0, 0.0, 0.0, 0.0])
        except Exception:
            pass
        try:
            p.setCollisionFilterGroupMask(robot_id, j, collisionFilterGroup=0, collisionFilterMask=0)
        except Exception:
            pass


def _set_robot_q(robot_id: int, joint_indices: Sequence[int], q: np.ndarray) -> None:
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.size < len(joint_indices):
        raise ValueError(f"joint_positions row has {q_arr.size} values, expected at least {len(joint_indices)}")
    for i, joint_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, int(joint_idx), float(q_arr[i]), targetVelocity=0.0)


def _spawn_current_marker(radius: float, color: tuple[float, float, float, float]) -> int:
    vis_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=list(color),
        specularColor=[0.15, 0.15, 0.15],
    )
    return int(p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis_id, basePosition=[0.0, 0.0, 0.0]))


def _spawn_tool_bar(length: float, radius: float, color: tuple[float, float, float, float]) -> int:
    vis_id = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(radius),
        length=float(length),
        rgbaColor=list(color),
        specularColor=[0.18, 0.18, 0.18],
    )
    return int(p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis_id, basePosition=[0.0, 0.0, 0.0]))


def _set_tool_bar_pose(body_id: int, tip_pos_world: np.ndarray, axis_world: np.ndarray, length: float) -> None:
    axis = _normalize3(axis_world)
    # Keep the probe tip fixed and extend the visible shaft toward the wrist.
    center = np.asarray(tip_pos_world, dtype=float).reshape(3) + 0.5 * float(length) * axis
    p.resetBasePositionAndOrientation(
        int(body_id),
        center.tolist(),
        _quat_align_z_to_vec(axis),
    )


def _space_was_triggered() -> bool:
    events = p.getKeyboardEvents()
    state = int(events.get(ord(" "), 0))
    return bool(state & p.KEY_WAS_TRIGGERED)


def render_s5_pybullet_demo_video(
    *,
    trajectory: np.ndarray,
    output_path: str | Path | None,
    sphere_center: Sequence[float],
    sphere_radius: float,
    cutpoints: Sequence[int] | None = None,
    tool_axis: np.ndarray | None = None,
    joint_positions: np.ndarray | None = None,
    title: str | None = None,
    center_world: Sequence[float] = (0.0, 0.0, 0.98),
    world_scale: float = 1.0,
    urdf_path: str | None = None,
    ur5_base_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    ur5_base_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    gui: int = 1,
    fps: float = 30.0,
    width: int = 1024,
    height: int = 768,
    render_frame_stride: int = 1,
    realtime: bool = False,
    gui_hold_seconds: float = 0.0,
    camera_yaw: float = 90.0,
    camera_pitch: float = -34.0,
    camera_distance: float = 1.45,
    camera_target: Sequence[float] | None = None,
    camera_fov: float = 42.0,
    tube_radius: float = 0.0055,
    stage4_shell_offset: float | None = None,
    sphere_texture_name: str = "checker_blue.png",
    trace_stride: int = 1,
    draw_stage_trace: bool = True,
    draw_executed_trace: bool = True,
    trace_width: float = 3.0,
    draw_current_marker: bool = False,
    hide_gripper: bool = True,
    draw_tool_bar: bool = False,
    tool_bar_length: float = 0.205,
    tool_bar_radius: float = 0.005,
    suppress_urdf_warnings: bool = True,
    connect_client: bool = True,
    feature_overlay: bool = False,
    feature_overlay_features: np.ndarray | None = None,
    feature_overlay_names: Sequence[str] | None = None,
    feature_overlay_units: dict[str, str] | None = None,
    feature_overlay_specs: Sequence[dict] | None = None,
    feature_overlay_true_constraints: dict | None = None,
    feature_overlay_title: str | None = None,
    playback_speed: float = 1.0,
    playback_label: str | None = None,
    save_frame_indices: Sequence[int] | None = None,
    save_frame_dir: str | Path | None = None,
    save_frame_prefix: str = "s5_frame",
) -> dict:
    _require_pybullet()
    pts = np.asarray(trajectory, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("trajectory must have shape (T, 3+) for S5 pybullet video rendering.")
    if len(pts) < 2:
        raise ValueError("trajectory must contain at least two points.")
    if tool_axis is None:
        axis = np.zeros((len(pts), 3), dtype=float)
        axis[:, 2] = 1.0
    else:
        axis = np.asarray(tool_axis, dtype=float)
        if axis.shape != pts[:, :3].shape:
            raise ValueError("tool_axis must have the same shape as trajectory[:, :3].")
    axis = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)

    q_path = None if joint_positions is None else np.asarray(joint_positions, dtype=float)
    if q_path is not None and (q_path.ndim != 2 or q_path.shape[0] != len(pts) or q_path.shape[1] < 6):
        raise ValueError("joint_positions must have shape (T, >=6), matching trajectory length.")

    gui_mode = int(gui)
    if gui_mode not in {0, 1, 2}:
        raise ValueError("gui must be one of 0, 1, 2.")
    save_video = gui_mode == 1 and output_path is not None
    use_gui = gui_mode == 2
    if gui_mode == 1 and output_path is None:
        raise ValueError("output_path is required when gui=1.")
    playback_speed = float(max(float(playback_speed), 1e-6))
    output_fps = float(fps) * playback_speed

    center_world = np.asarray(center_world, dtype=float).reshape(3)
    sphere_center = np.asarray(sphere_center, dtype=float).reshape(3)
    radius_world = float(world_scale) * float(sphere_radius)
    table_top_z = float(center_world[2] - 1.03 * radius_world)
    traj_world = _env_to_world(pts[:, :3], sphere_center=sphere_center, center_world=center_world, scale=world_scale)
    shell_radius_world = None
    if stage4_shell_offset is not None:
        shell_radius_world = radius_world + float(world_scale) * max(0.0, float(stage4_shell_offset))
    table_center_xy = (0.5 * float(center_world[0]), 0.5 * float(center_world[1]))
    table_half_extents_xy = (
        max(0.82, 0.5 * abs(float(center_world[0])) + float(radius_world) + 0.46),
        max(0.68, 0.5 * abs(float(center_world[1])) + float(radius_world) + 0.44),
    )
    bounds = stage_segments(len(pts), cutpoints=cutpoints)
    trace_stride = int(max(1, trace_stride))
    trace_width = float(max(0.5, trace_width))
    trace_radius = float(max(0.0009, 0.00055 * trace_width))
    exec_trace_palette = [
        (0.28, 0.14, 0.46, 0.94),
        (0.21, 0.36, 0.55, 0.94),
        (0.13, 0.57, 0.55, 0.94),
        (0.40, 0.76, 0.39, 0.94),
        (0.90, 0.85, 0.22, 0.94),
    ]
    render_frame_stride = int(max(1, render_frame_stride))
    frame_indices_to_save = {
        int(v)
        for v in ([] if save_frame_indices is None else save_frame_indices)
        if 0 <= int(v) < len(pts)
    }
    frame_save_dir = None if save_frame_dir is None else Path(save_frame_dir)
    saved_frame_paths: list[str] = []

    def _stage_index_at(frame_idx: int) -> int:
        for stage_idx, (start, end) in enumerate(bounds):
            if int(start) <= int(frame_idx) <= int(end):
                return int(stage_idx)
        return int(max(0, min(len(bounds) - 1, len(bounds) - 1)))

    client = p.connect(p.GUI if use_gui else p.DIRECT) if bool(connect_client) else None
    writer = None
    patched_urdf = None
    frames_written = 0
    t0 = time.time()
    try:
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        if not use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        _spawn_ground_grid(
            center_xy=(0.48 * float(center_world[0]), 0.40 * float(center_world[1])),
            half_extents_xy=(1.18, 0.92),
            spacing=0.10,
            z=0.0,
        )
        _spawn_table(
            table_top_z=table_top_z,
            center_xy=table_center_xy,
            half_extents_xy=table_half_extents_xy,
        )
        _spawn_sphere(
            center_world=center_world,
            radius_world=radius_world,
            shell_radius_world=shell_radius_world,
            texture_name=str(sphere_texture_name),
            draw_shell_surface=bool(use_gui),
            draw_surface_rings=False,
        )

        if bool(draw_stage_trace):
            for stage_idx, (start, end) in enumerate(bounds):
                idx = np.arange(int(start), int(end) + 1, trace_stride, dtype=int)
                if int(idx[-1]) != int(end):
                    idx = np.concatenate([idx, np.asarray([int(end)], dtype=int)])
                seg = traj_world[idx]
                color = exec_trace_palette[stage_idx % len(exec_trace_palette)]
                for a, b in zip(seg[:-1], seg[1:]):
                    _spawn_capsule_segment(a, b, radius=float(tube_radius), color=color)

        _spawn_marker(traj_world[0], radius=0.0058, color=(0.10, 0.65, 0.25, 1.0))
        _spawn_marker(traj_world[-1], radius=0.0058, color=(0.86, 0.18, 0.18, 1.0))
        pending_cutpoints = [
            int(cp)
            for cp in np.asarray(cutpoints if cutpoints is not None else [], dtype=int).reshape(-1)
            if 0 <= int(cp) < len(traj_world)
        ]
        spawned_cutpoints: set[int] = set()

        robot_id = None
        joint_indices: list[int] = []
        if q_path is not None:
            hidden_link_patterns: list[str] = []
            if bool(hide_gripper):
                hidden_link_patterns.extend(["gripper", "finger", "palm"])
            if bool(draw_tool_bar):
                hidden_link_patterns.extend(["task_tool", "task_tcp"])
            robot_id, joint_indices, patched_urdf = _load_ur5_render_robot(
                urdf_path=urdf_path,
                base_xyz=ur5_base_xyz,
                base_rpy=ur5_base_rpy,
                hide_link_geometry_patterns=hidden_link_patterns,
                suppress_urdf_warnings=bool(suppress_urdf_warnings),
            )
            if bool(hide_gripper):
                _hide_robot_links_by_name(robot_id, ("gripper", "finger", "palm"))
            _set_robot_q(robot_id, joint_indices, q_path[0])

        current_marker_id = None
        if bool(draw_current_marker):
            current_marker_id = _spawn_current_marker(radius=0.0045, color=(0.98, 0.78, 0.16, 1.0))
        tool_bar_id = None
        if bool(draw_tool_bar):
            tool_bar_id = _spawn_tool_bar(
                length=float(tool_bar_length),
                radius=float(tool_bar_radius),
                color=(0.92, 0.50, 0.10, 1.0),
            )
            _set_tool_bar_pose(tool_bar_id, traj_world[0], axis[0], float(tool_bar_length))
        elif q_path is None:
            _spawn_probe_pose(traj_world[0], axis[0], shaft_len=0.080, shaft_radius=0.0045)

        stage_label_items: list[dict] = []
        for stage_idx, (start, end) in enumerate(bounds):
            start_i = int(start)
            end_i = int(end)
            anchor_i = int(round(float(start_i) + 0.25 * float(max(0, end_i - start_i))))
            anchor_i = int(np.clip(anchor_i, 0, len(traj_world) - 1))
            rgba = _hex_to_rgba(STAGE_COLORS[stage_idx % len(STAGE_COLORS)], alpha=1.0)
            stage_label_items.append(
                {
                    "start": start_i,
                    "pos": np.asarray(traj_world[anchor_i], dtype=float),
                    "center_pos": np.asarray(center_world, dtype=float),
                    "stage_index": int(stage_idx),
                    "text": f"Stage {stage_idx + 1}",
                    "color": tuple(int(np.clip(v * 255.0, 0, 255)) for v in rgba),
                }
            )

        target = (
            np.asarray(camera_target, dtype=float).reshape(3)
            if camera_target is not None
            else center_world + np.array([0.0, 0.0, 0.035], dtype=float)
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=float(camera_distance),
            cameraYaw=float(camera_yaw),
            cameraPitch=float(camera_pitch),
            cameraTargetPosition=target.tolist(),
        )
        if save_video:
            writer = _FFmpegVideoWriter(out_path=Path(output_path), width=int(width), height=int(height), fps=float(output_fps))

        pause_text_id = None
        if use_gui:
            pause_text_id = p.addUserDebugText(
                "SPACE: pause/resume. At end, press SPACE for next demo.",
                textPosition=(target + np.array([-0.34, -0.30, 0.34], dtype=float)).tolist(),
                textColorRGB=(0.05, 0.05, 0.05),
                textSize=1.15,
                lifeTime=0.0,
            )

        i = 0
        paused = False
        prev_trace_pos: np.ndarray | None = None
        while i < len(pts):
            if use_gui and _space_was_triggered():
                paused = not paused
                if pause_text_id is not None:
                    p.removeUserDebugItem(pause_text_id)
                pause_text_id = p.addUserDebugText(
                    ("Paused. SPACE: resume." if paused else "SPACE: pause/resume. At end, press SPACE for next demo."),
                    textPosition=(target + np.array([-0.34, -0.30, 0.34], dtype=float)).tolist(),
                    textColorRGB=(0.05, 0.05, 0.05),
                    textSize=1.15,
                    lifeTime=0.0,
                )
            if use_gui and paused:
                time.sleep(0.05)
                continue

            if robot_id is not None and q_path is not None:
                _set_robot_q(robot_id, joint_indices, q_path[i])
            if current_marker_id is not None:
                p.resetBasePositionAndOrientation(current_marker_id, traj_world[i].tolist(), [0.0, 0.0, 0.0, 1.0])
            if tool_bar_id is not None:
                _set_tool_bar_pose(tool_bar_id, traj_world[i], axis[i], float(tool_bar_length))
            for cp in pending_cutpoints:
                if int(i) >= int(cp) and int(cp) not in spawned_cutpoints:
                    _spawn_marker(traj_world[int(cp)], radius=0.0058, color=(0.04, 0.04, 0.04, 0.92))
                    spawned_cutpoints.add(int(cp))
            if bool(draw_executed_trace) and (i % trace_stride == 0 or i == len(pts) - 1):
                radial = _normalize3(traj_world[i] - center_world)
                lift = radial * max(0.006, 2.8 * trace_radius)
                cur_trace_pos = traj_world[i] + lift
                if prev_trace_pos is not None:
                    color = exec_trace_palette[_stage_index_at(i) % len(exec_trace_palette)]
                    _spawn_capsule_segment(prev_trace_pos, cur_trace_pos, radius=trace_radius, color=color)
                prev_trace_pos = cur_trace_pos
            p.stepSimulation()
            write_frame = (i % render_frame_stride == 0) or (i == len(pts) - 1)
            save_still = int(i) in frame_indices_to_save and frame_save_dir is not None
            if (save_video and writer is not None and write_frame) or save_still:
                frame = _render_rgb(
                    yaw_deg=float(camera_yaw),
                    target=target,
                    distance=float(camera_distance),
                    width=int(width),
                    height=int(height),
                    pitch_deg=float(camera_pitch),
                    fov=float(camera_fov),
                )
                view, proj = _camera_matrices(
                    yaw_deg=float(camera_yaw),
                    target=target,
                    distance=float(camera_distance),
                    width=int(width),
                    height=int(height),
                    pitch_deg=float(camera_pitch),
                    fov=float(camera_fov),
                )
                visible_stage_labels = [item for item in stage_label_items if int(i) >= int(item.get("start", 0))]
                if bool(feature_overlay) and feature_overlay_features is not None:
                    frame = _overlay_feature_panel(
                        frame,
                        features=np.asarray(feature_overlay_features, dtype=float),
                        feature_names=list(feature_overlay_names or []),
                        feature_units=dict(feature_overlay_units or {}),
                        current_index=int(i),
                        cutpoints=cutpoints,
                        constraint_specs=list(feature_overlay_specs or []),
                        true_constraints=dict(feature_overlay_true_constraints or {}),
                        title=str(feature_overlay_title or "Executed trajectory feature profile"),
                    )
                frame = _overlay_corner_label(frame, playback_label)
                if save_still:
                    frame_path = _save_rgb_frame(
                        frame,
                        frame_save_dir / f"{str(save_frame_prefix)}_frame_{int(i):06d}.png",
                        crop_aspect=0.9,
                        crop_scale=0.6,
                        crop_offset_x=-0.70,
                        crop_offset_y=0.30,
                        crop_bottom_fraction=0.2,
                    )
                    saved_frame_paths.append(str(frame_path.resolve()))
                if save_video and writer is not None and write_frame:
                    frame = _overlay_trajectory_stage_labels(
                        frame,
                        labels=visible_stage_labels,
                        view_matrix=view,
                        projection_matrix=proj,
                    )
                    writer.append_data(frame)
                    frames_written += 1
            i += 1
            if use_gui and bool(realtime):
                time.sleep(1.0 / max(float(output_fps), 1e-6))
        if save_video and writer is not None and float(gui_hold_seconds) > 0.0:
            hold_frames = int(round(float(gui_hold_seconds) * float(output_fps)))
            if hold_frames > 0:
                frame = _render_rgb(
                    yaw_deg=float(camera_yaw),
                    target=target,
                    distance=float(camera_distance),
                    width=int(width),
                    height=int(height),
                    pitch_deg=float(camera_pitch),
                    fov=float(camera_fov),
                )
                view, proj = _camera_matrices(
                    yaw_deg=float(camera_yaw),
                    target=target,
                    distance=float(camera_distance),
                    width=int(width),
                    height=int(height),
                    pitch_deg=float(camera_pitch),
                    fov=float(camera_fov),
                )
                frame = _overlay_trajectory_stage_labels(
                    frame,
                    labels=stage_label_items,
                    view_matrix=view,
                    projection_matrix=proj,
                )
                if bool(feature_overlay) and feature_overlay_features is not None:
                    frame = _overlay_feature_panel(
                        frame,
                        features=np.asarray(feature_overlay_features, dtype=float),
                        feature_names=list(feature_overlay_names or []),
                        feature_units=dict(feature_overlay_units or {}),
                        current_index=int(len(pts) - 1),
                        cutpoints=cutpoints,
                        constraint_specs=list(feature_overlay_specs or []),
                        true_constraints=dict(feature_overlay_true_constraints or {}),
                        title=str(feature_overlay_title or "Executed trajectory feature profile"),
                    )
                frame = _overlay_corner_label(frame, playback_label)
                for _ in range(hold_frames):
                    writer.append_data(frame)
                frames_written += int(hold_frames)
        if use_gui:
            hold_seconds = float(gui_hold_seconds)
            if hold_seconds < 0.0:
                if pause_text_id is not None:
                    p.removeUserDebugItem(pause_text_id)
                p.addUserDebugText(
                    "Demo finished. Press SPACE for next demo.",
                    textPosition=(target + np.array([-0.30, -0.30, 0.34], dtype=float)).tolist(),
                    textColorRGB=(0.05, 0.05, 0.05),
                    textSize=1.2,
                    lifeTime=0.0,
                )
                try:
                    while True:
                        if _space_was_triggered():
                            break
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
            elif hold_seconds > 0.0:
                time.sleep(hold_seconds)
    finally:
        if writer is not None:
            writer.close()
        if bool(connect_client) and client is not None:
            p.disconnect(client)
        if patched_urdf:
            try:
                os.remove(patched_urdf)
            except OSError:
                pass

    return {
        "video_path": None if not save_video else str(Path(output_path).resolve()),
        "frames_written": int(frames_written),
        "source_frames": int(len(pts)),
        "fps": float(fps),
        "output_fps": float(output_fps),
        "playback_speed": float(playback_speed),
        "playback_label": None if playback_label is None else str(playback_label),
        "gui": int(gui_mode),
        "wall_seconds": float(time.time() - t0),
        "has_robot_joint_playback": bool(q_path is not None),
        "hide_gripper": bool(hide_gripper),
        "draw_tool_bar": bool(draw_tool_bar),
        "draw_stage_trace": bool(draw_stage_trace),
        "draw_executed_trace": bool(draw_executed_trace),
        "trace_stride": int(trace_stride),
        "saved_frames": saved_frame_paths,
        "trace_width": float(trace_width),
        "draw_current_marker": bool(draw_current_marker),
        "stage4_shell_offset": None if stage4_shell_offset is None else float(stage4_shell_offset),
        "sphere_texture_name": str(sphere_texture_name),
        "title": title,
    }


def _compose_paper_view(main_img: np.ndarray, inset_img: np.ndarray, output_path: str | Path, title: str | None) -> Path:
    _require_matplotlib()
    fig = plt.figure(figsize=(5.7, 3.35), dpi=240)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.92])
    ax.imshow(np.asarray(main_img, dtype=np.uint8))
    ax.set_axis_off()
    if title:
        ax.set_title(str(title), fontsize=10, pad=2.0)

    inset_ax = fig.add_axes([0.67, 0.58, 0.28, 0.28])
    inset_ax.imshow(np.asarray(inset_img, dtype=np.uint8))
    inset_ax.set_axis_off()
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_edgecolor((0.15, 0.15, 0.15, 0.95))

    return _save_figure(fig, output_path, dpi=240)


def render_s5_pybullet_episode(
    *,
    trajectory: np.ndarray,
    output_path: str | Path,
    sphere_center: Sequence[float],
    sphere_radius: float,
    cutpoints: Sequence[int] | None = None,
    overlay_cutpoints: Sequence[int] | None = None,
    tool_axis: np.ndarray | None = None,
    title: str | None = None,
    center_world: Sequence[float] = (0.0, 0.0, 0.98),
    world_scale: float = 1.0,
    main_yaw: float = 42.0,
    inset_yaw: float = 205.0,
    main_pitch: float = -18.0,
    inset_pitch: float = -16.0,
    main_distance: float = 1.32,
    inset_distance: float = 1.36,
    tube_radius: float = 0.0065,
) -> Path:
    _require_pybullet()
    pts = np.asarray(trajectory, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("trajectory must have shape (T, 3+) for S5 pybullet rendering.")
    if tool_axis is None:
        raise ValueError("tool_axis is required for S5 pybullet rendering.")

    axis = np.asarray(tool_axis, dtype=float)
    if axis.shape != pts.shape:
        raise ValueError("tool_axis must have the same shape as trajectory.")

    center_world = np.asarray(center_world, dtype=float).reshape(3)
    sphere_center = np.asarray(sphere_center, dtype=float).reshape(3)
    radius_world = float(world_scale) * float(sphere_radius)
    table_top_z = float(center_world[2] - 1.03 * radius_world)
    traj_world = _env_to_world(pts[:, :3], sphere_center=sphere_center, center_world=center_world, scale=world_scale)
    bounds = stage_segments(len(pts), cutpoints=cutpoints)
    overlay_cutpoints = [] if overlay_cutpoints is None else [int(v) for v in np.asarray(overlay_cutpoints, dtype=int).reshape(-1)]

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=35.0,
            cameraPitch=-20.0,
            cameraTargetPosition=[0.0, 0.0, 1.0],
        )

        _spawn_ground_grid(center_xy=(0.0, 0.0), half_extents_xy=(1.12, 0.92), spacing=0.10, z=0.0)
        _spawn_table(table_top_z=table_top_z)
        _spawn_sphere(center_world=center_world, radius_world=radius_world)

        for stage_idx, (start, end) in enumerate(bounds):
            seg = traj_world[start : end + 1]
            color = _hex_to_rgba(STAGE_COLORS[stage_idx % len(STAGE_COLORS)], alpha=1.0)
            for idx in range(len(seg) - 1):
                _spawn_capsule_segment(seg[idx], seg[idx + 1], radius=tube_radius, color=color)

        _spawn_marker(traj_world[0], radius=0.015, color=(0.10, 0.65, 0.25, 1.0))
        _spawn_marker(traj_world[-1], radius=0.014, color=(0.86, 0.18, 0.18, 1.0))
        for cp in overlay_cutpoints:
            if 0 <= int(cp) < len(traj_world):
                _spawn_marker(traj_world[int(cp)], radius=0.011, color=(0.08, 0.08, 0.08, 1.0))

        for start, end in bounds:
            mid = int(round(0.5 * (int(start) + int(end))))
            if 0 <= mid < len(traj_world):
                _spawn_probe_pose(traj_world[mid], axis[mid])

        for _ in range(8):
            p.stepSimulation()

        main_target = center_world + np.array([0.0, 0.0, -0.06], dtype=float)
        inset_target = center_world + np.array([0.0, 0.0, -0.04], dtype=float)
        main_img = _render_rgb(
            yaw_deg=float(main_yaw),
            target=main_target,
            distance=float(main_distance),
            width=1300,
            height=980,
            pitch_deg=float(main_pitch),
        )
        inset_img = _render_rgb(
            yaw_deg=float(inset_yaw),
            target=inset_target,
            distance=float(inset_distance),
            width=720,
            height=720,
            pitch_deg=float(inset_pitch),
        )
    finally:
        p.disconnect(client)

    return _compose_paper_view(main_img=main_img, inset_img=inset_img, output_path=output_path, title=title)
