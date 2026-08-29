#!/usr/bin/env python3
"""Temporary fixed-view trajectory-10 snapshots for the thesis figure."""

from __future__ import annotations

import bisect
import csv
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAJ_DIR = PROJECT_ROOT / "kuka_experiment_data/learned_from_human/trajectory_010"
FRAME_DIR = TRAJ_DIR / "paper_frames"
PLAN_JSON = TRAJ_DIR / "constraint_plan_9_20260721_171531.json"
TIMESTAMPS_CSV = TRAJ_DIR / "manifold_cam1_timestamps.csv"
THESIS_OUTDIR = Path("/home/baiyu/PycharmProjects/baiyu_thesis/content/oncl/figs/paper_snaps")

SECONDS = [2, 4, 6, 9]
FPS = 30.0

# Fixed crop in raw 1920x1080 camera pixels. It keeps the view wider than the
# older paper crop while removing a little top/bottom clutter.
CROP_BOX = (152, 115, 1368, 1070)
PANEL_HEIGHT = 520
PANEL_GAP = 10
OUTPUT_STEM = "traj10_four_frame_original_video_fixed_wide_temp"


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
        _, frame_time = min(frame_times, key=lambda item: abs(item[0] - target_frame))
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


def _resize_crop(image: Image.Image) -> Image.Image:
    crop = image.crop(CROP_BOX)
    width, height = crop.size
    panel_width = int(round(PANEL_HEIGHT * width / height))
    return crop.resize((panel_width, PANEL_HEIGHT), Image.Resampling.LANCZOS)


def _label_box(
    canvas: Image.Image,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    *,
    fill: tuple[int, int, int, int] = (30, 30, 30, 190),
    text_fill: tuple[int, int, int] = (255, 255, 255),
) -> None:
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad_x = 10
    pad_y = 6
    x, y = xy
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(
        (x, y, x + text_w + 2 * pad_x, y + text_h + 2 * pad_y),
        radius=4,
        fill=fill,
    )
    merged = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
    canvas.paste(merged)
    draw = ImageDraw.Draw(canvas)
    draw.text((x + pad_x, y + pad_y - 1), text, font=font, fill=text_fill)


def _save_image(image: Image.Image, stem: str) -> tuple[Path, Path]:
    THESIS_OUTDIR.mkdir(parents=True, exist_ok=True)
    png_path = THESIS_OUTDIR / f"{stem}.png"
    pdf_path = THESIS_OUTDIR / f"{stem}.pdf"
    image.save(png_path)
    image.save(pdf_path, "PDF", resolution=300.0)
    return png_path, pdf_path


def _copy_setup_images() -> list[Path]:
    THESIS_OUTDIR.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in ("kuka_task.png", "kuka_demo.png"):
        src = TRAJ_DIR / name
        dst = THESIS_OUTDIR / name
        shutil.copy2(src, dst)
        copied.append(dst)
    return copied


def main() -> None:
    errors = _nearest_errors()
    panels = [
        _resize_crop(Image.open(FRAME_DIR / f"traj10_raw_{sec:02d}s.png").convert("RGB"))
        for sec in SECONDS
    ]
    panel_w = panels[0].width
    panel_h = panels[0].height
    out_w = panel_w * len(panels) + PANEL_GAP * (len(panels) - 1)
    montage = Image.new("RGB", (out_w, panel_h), "white")
    time_font = _load_font(34)

    x = 0
    for sec, panel in zip(SECONDS, panels):
        montage.paste(panel, (x, 0))
        _label_box(montage, (x + 14, 14), f"{sec} s", time_font)
        x += panel_w + PANEL_GAP

    png_path, pdf_path = _save_image(montage, OUTPUT_STEM)
    copied = _copy_setup_images()

    print(png_path)
    print(pdf_path)
    for path in copied:
        print(path)
    print(f"crop_box={CROP_BOX}, panel_size={panel_w}x{panel_h}")


if __name__ == "__main__":
    main()
