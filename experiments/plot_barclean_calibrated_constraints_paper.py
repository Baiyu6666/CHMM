from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from visualization.learned_constraints_matrix import (
    plot_learned_constraints_matrix_paper,
    plot_true_vs_learned_constraints_matrix_paper,
)
from visualization.barclean_display import (
    BARCLEAN_AXIAL_DISPLAY_SHIFT_M,
    BARCLEAN_OBSTACLE_NEAR_ENDPOINT_AXIAL_M,
    BARCLEAN_RAW_AXIAL_REFERENCE_M,
)


BARCLEAN_SPONGE_YAW_OFFSET_DEG = 38.30219823954972
DISTANCE_FEATURES = {"obs_dist", "table_dist", "lateral_offset", "axial_offset"}
DISPLAY_FEATURE_NAMES = {
    "obs_dist": "obs_dist [mm]",
    "table_dist": "table_dist [mm]",
    "lateral_offset": "lateral_offset [mm]",
    "axial_offset": "axial_offset [mm]",
    "tool_pitch": "tool_pitch [deg]",
    "tool_roll": "tool_roll [deg]",
    "tool_yaw": "tool_yaw [deg]",
}
MATRIX_KEYS = (
    "ConstraintLearnedValueMatrix",
    "ConstraintLearnedRawValueMatrix",
    "ConstraintTargetMatrix",
)


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def _transform_matrix(
    values: object,
    feature_names: list[str],
    sponge_yaw_offset_deg: float,
) -> list[list[float]]:
    matrix = np.asarray(values, dtype=float).copy()
    if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
        raise ValueError("BarClean constraint matrix must align with ConstraintFeatureNames.")
    for feature_index, feature_name in enumerate(feature_names):
        if feature_name in DISTANCE_FEATURES:
            matrix[:, feature_index] *= 1000.0
            if feature_name == "axial_offset":
                matrix[:, feature_index] += BARCLEAN_AXIAL_DISPLAY_SHIFT_M * 1000.0
        elif feature_name == "tool_pitch":
            matrix[:, feature_index] = np.rad2deg(matrix[:, feature_index]) - 90.0
        elif feature_name == "tool_roll":
            matrix[:, feature_index] = np.rad2deg(matrix[:, feature_index])
        elif feature_name == "tool_yaw":
            matrix[:, feature_index] = _wrap_degrees(
                np.rad2deg(matrix[:, feature_index]) + float(sponge_yaw_offset_deg)
            )
    return matrix.tolist()


def prepare_barclean_paper_payload(
    payload: dict,
    *,
    sponge_yaw_offset_deg: float = BARCLEAN_SPONGE_YAW_OFFSET_DEG,
) -> dict:
    transformed = deepcopy(payload)
    feature_names = [str(value) for value in payload.get("ConstraintFeatureNames", [])]
    missing = sorted(set(DISPLAY_FEATURE_NAMES).difference(feature_names))
    if missing:
        raise ValueError(f"BarClean payload is missing required features: {missing}")
    for key in MATRIX_KEYS:
        if key in payload:
            transformed[key] = _transform_matrix(
                payload[key],
                feature_names,
                sponge_yaw_offset_deg,
            )
    transformed["ConstraintFeatureNames"] = [
        DISPLAY_FEATURE_NAMES.get(name, name) for name in feature_names
    ]
    transformed["BarCleanPaperCalibration"] = {
        "sponge_yaw_offset_deg": float(sponge_yaw_offset_deg),
        "tool_pitch_reference_deg": 90.0,
        "tool_roll_reference_deg": 0.0,
        "axial_offset_source_reference_m": BARCLEAN_RAW_AXIAL_REFERENCE_M,
        "axial_offset_zero_task_x_m": BARCLEAN_OBSTACLE_NEAR_ENDPOINT_AXIAL_M,
        "axial_offset_display_shift_mm": BARCLEAN_AXIAL_DISPLAY_SHIFT_M * 1000.0,
        "distance_unit": "mm",
        "angle_unit": "deg",
    }
    return transformed


def plot_barclean_constraints(
    constraints_json: Path,
    output_path: Path,
    *,
    sponge_yaw_offset_deg: float = BARCLEAN_SPONGE_YAW_OFFSET_DEG,
    compare: bool = False,
) -> Path:
    payload = json.loads(constraints_json.read_text(encoding="utf-8"))
    transformed = prepare_barclean_paper_payload(
        payload,
        sponge_yaw_offset_deg=sponge_yaw_offset_deg,
    )
    plotter = (
        plot_true_vs_learned_constraints_matrix_paper
        if compare
        else plot_learned_constraints_matrix_paper
    )
    out = plotter(
        transformed,
        save_path=output_path,
        dataset_name="display_space",
    )
    if out is None:
        raise RuntimeError("matplotlib is required to plot BarClean constraints.")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BarClean paper constraints in calibrated sponge/error coordinates."
    )
    parser.add_argument("--constraints-json", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--sponge-yaw-offset-deg",
        type=float,
        default=BARCLEAN_SPONGE_YAW_OFFSET_DEG,
    )
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    constraints_json = Path(args.constraints_json)
    if args.output is None:
        filename = (
            "paper_barclean_calibrated_true_vs_learned_constraints.png"
            if args.compare
            else "paper_barclean_calibrated_learned_constraints.png"
        )
        output_path = constraints_json.parent / "paper_figures" / filename
    else:
        output_path = Path(args.output)
    out = plot_barclean_constraints(
        constraints_json,
        output_path,
        sponge_yaw_offset_deg=float(args.sponge_yaw_offset_deg),
        compare=bool(args.compare),
    )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
