#!/usr/bin/env python3
"""Build the trajectory-10 four-frame paper figure with per-frame errors."""

from __future__ import annotations

import bisect
import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAJ_DIR = PROJECT_ROOT / "kuka_experiment_data/learned_from_human/trajectory_010"
FRAME_DIR = TRAJ_DIR / "paper_frames"
PLAN_JSON = TRAJ_DIR / "constraint_plan_9_20260721_171531.json"
TIMESTAMPS_CSV = TRAJ_DIR / "manifold_cam1_timestamps.csv"
SECONDS = [2, 4, 6, 9]
FPS = 30.0
OUTPUT_STEM = "traj10_four_frame_original_video_dynamic_narrow_errors"


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _frame_timestamps() -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with TIMESTAMPS_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((int(row["frame_index"]), int(row["unix_time_ns"])))
    return rows


def _nearest_errors() -> dict[int, tuple[float, float]]:
    with PLAN_JSON.open() as handle:
        samples = json.load(handle)["samples"]
    sample_times = [int(sample["unix_time_ns"]) for sample in samples]
    frame_times = _frame_timestamps()
    errors: dict[int, tuple[float, float]] = {}
    for sec in SECONDS:
        target_frame = int(round(sec * FPS))
        frame_index, frame_time = min(
            frame_times,
            key=lambda item: abs(item[0] - target_frame),
        )
        del frame_index
        insert_at = bisect.bisect_left(sample_times, frame_time)
        candidates = [
            index
            for index in (insert_at - 1, insert_at, insert_at + 1)
            if 0 <= index < len(samples)
        ]
        sample = min(candidates, key=lambda index: abs(sample_times[index] - frame_time))
        matched = samples[sample]
        position_mm = 1000.0 * float(matched["reference_true_surface_position_error_m"])
        orientation_deg = float(matched["reference_true_surface_orientation_error_deg"])
        errors[sec] = (position_mm, orientation_deg)
    return errors


def main() -> None:
    errors = _nearest_errors()
    crops = [
        Image.open(FRAME_DIR / f"traj10_dynamic_narrow_crop_{sec:02d}s.png").convert("RGB")
        for sec in SECONDS
    ]
    reference = Image.open(FRAME_DIR / "traj10_four_frame_original_video_dynamic_narrow.png")
    panel_gap = (reference.width - sum(crop.width for crop in crops)) // (len(crops) - 1)
    panel_width = crops[0].width
    panel_height = crops[0].height
    out_width = panel_width * len(crops) + panel_gap * (len(crops) - 1)
    out_height = panel_height
    canvas = Image.new("RGB", (out_width, out_height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(22)

    x = 0
    for sec, crop in zip(SECONDS, crops):
        canvas.paste(crop, (x, 0))
        pos_mm, ori_deg = errors[sec]
        caption = f"pos. {pos_mm:.2f} mm | ori. {ori_deg:.2f} deg"
        bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        pad_x = 10
        pad_y = 5
        box_x0 = int(x + (panel_width - text_width) / 2 - pad_x)
        box_y0 = panel_height - text_height - 23
        box_x1 = int(box_x0 + text_width + 2 * pad_x)
        box_y1 = int(box_y0 + text_height + 2 * pad_y)
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rounded_rectangle(
            (box_x0, box_y0, box_x1, box_y1),
            radius=4,
            fill=(255, 255, 255, 210),
        )
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(canvas)
        draw.text((box_x0 + pad_x, box_y0 + pad_y - 1), caption, font=font, fill=(0, 0, 0))
        x += panel_width + panel_gap

    output_png = FRAME_DIR / f"{OUTPUT_STEM}.png"
    output_pdf = FRAME_DIR / f"{OUTPUT_STEM}.pdf"
    canvas.save(output_png)
    canvas.save(output_pdf, "PDF", resolution=300.0)
    print(output_png)
    print(output_pdf)
    for sec in SECONDS:
        pos_mm, ori_deg = errors[sec]
        print(f"{sec}s: position={pos_mm:.3f} mm, orientation={ori_deg:.3f} deg")


if __name__ == "__main__":
    main()
