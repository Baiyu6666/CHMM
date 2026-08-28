from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.BarClean import BarCleanEnv  # noqa: E402
from envs.BarInspect import BarInspectScene  # noqa: E402


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "robot/stage_cons_iiwa14/data/processed/demo_fixed_scene_loader_export/"
    "demo_fixed_scene_7_demos_20hz_loader_input.npz"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "robot/stage_cons_iiwa14/data/processed/demo_fixed_scene_loader_export/"
    "demo_fixed_scene_7_demos_10hz_training.npz"
)
DEFAULT_ANNOTATION_REFERENCE = (
    PROJECT_ROOT
    / "robot/stage_cons_iiwa14/data/processed/demo_fixed_scene_loader_export/"
    "demo_fixed_scene_7_demos_5hz_training.npz"
)


def _resolve(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    return resolved.resolve()


def _stored_path(path: str | Path) -> str:
    resolved = _resolve(path)
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _required(archive: np.lib.npyio.NpzFile, names: set[str]) -> None:
    missing = sorted(names.difference(archive.files))
    if missing:
        raise ValueError(f"Segmented BarClean archive is missing keys: {missing}")


def _downsample_demo_indices(begin: int, end: int, factor: int) -> np.ndarray:
    selected = np.arange(int(begin), int(end), int(factor), dtype=np.int64)
    if len(selected) == 0:
        raise ValueError(f"Empty BarClean demo interval {begin}:{end}.")
    last = int(end) - 1
    if selected[-1] != last:
        selected = np.r_[selected, last]
    return selected


def _task_coordinates(features: np.ndarray, env: BarCleanEnv) -> np.ndarray:
    columns = {spec["name"]: int(spec["column_idx"]) for spec in env.feature_schema}
    axial = (
        features[:, columns["bar_axial_offset"]]
        + float(env.task_definition["bar_axial_offset_reference"])
    )
    lateral = features[:, columns["bar_lateral_offset"]]
    height = features[:, columns["table_dist"]]
    return np.column_stack([axial, lateral, height])


def _true_runs(mask: np.ndarray) -> np.ndarray:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return np.column_stack((edges[::2], edges[1::2]))


def _merge_nearby_runs(runs: list[tuple[int, int]], maximum_gap: int) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for begin, end in runs:
        if merged and int(begin) - merged[-1][1] <= int(maximum_gap):
            merged[-1] = (merged[-1][0], int(end))
        else:
            merged.append((int(begin), int(end)))
    return merged


def _reference_cutpoint_times(
    reference_path: str | Path,
) -> tuple[dict[int, np.ndarray], float, str]:
    path = _resolve(reference_path)
    with np.load(path, allow_pickle=False) as archive:
        _required(
            archive,
            {
                "timestamps",
                "coarse_bounds_indices",
                "source_demo_ids",
                "downsample_hz",
                "cutpoint_annotation_kind",
            },
        )
        timestamps = np.asarray(archive["timestamps"], dtype=float)
        bounds = np.asarray(archive["coarse_bounds_indices"], dtype=np.int64)
        source_demo_ids = np.asarray(archive["source_demo_ids"], dtype=np.int64)
        reference_hz = float(np.asarray(archive["downsample_hz"]).item())
        annotation_kind = str(np.asarray(archive["cutpoint_annotation_kind"]).item())

    if bounds.shape != (len(source_demo_ids), 6):
        raise ValueError("Reference BarClean bounds must have shape (num_demos, 6).")
    result: dict[int, np.ndarray] = {}
    for source_demo_id, row in zip(source_demo_ids, bounds):
        if np.any(np.diff(row) <= 0):
            raise ValueError(f"Invalid reference bounds: {row.tolist()}")
        result[int(source_demo_id)] = timestamps[row[1:-1]].copy()
    return result, reference_hz, annotation_kind


def _bounds_from_cutpoint_times(
    timestamps: np.ndarray,
    cutpoint_times: np.ndarray,
) -> np.ndarray:
    internal = np.asarray(
        [int(np.argmin(np.abs(timestamps - value))) for value in cutpoint_times],
        dtype=np.int64,
    )
    bounds = np.r_[0, internal, len(timestamps)].astype(np.int64)
    if np.any(np.diff(bounds) <= 0):
        raise ValueError(f"Mapped BarClean bounds are not strictly ordered: {bounds.tolist()}")
    return bounds


def _stage_diagnostics(
    task_xyz: np.ndarray,
    bounds: np.ndarray,
) -> dict[str, object]:
    return {
        "boundary_task_xyz_m": task_xyz[bounds[1:-1]].tolist(),
        "axial_progress_m": float(
            task_xyz[bounds[2] - 1, 0] - task_xyz[bounds[1], 0]
        ),
        "lateral_progress_m": float(
            task_xyz[bounds[3], 1] - task_xyz[bounds[4] - 1, 1]
        ),
    }


def motion_phase_stage_bounds(
    features: np.ndarray,
    env: BarCleanEnv,
    *,
    minimum_stage_samples: int = 5,
    progress_speed_mps: float = 0.025,
    cross_axis_speed_mps: float = 0.045,
    smoothing_sigma_samples: float = 1.0,
    bridge_samples: int = 3,
    merge_axial_gap_samples: int = 8,
) -> tuple[np.ndarray, dict[str, object]]:
    """Split five phases from motion, without using task endpoints.

    Stage 2 is the dominant positive bar-axial run with little lateral/vertical
    motion. Stage 4 is the subsequent dominant negative bar-lateral run with
    little axial/vertical motion. Their starts and ends define the four internal
    boundaries; the free approach/reposition/depart phases occupy the gaps.
    """
    task_xyz = _task_coordinates(features, env)
    point_count = len(task_xyz)
    if point_count < 5 * int(minimum_stage_samples):
        raise ValueError("BarClean demo is too short for five ordered stages.")

    smooth_xyz = gaussian_filter1d(
        task_xyz,
        sigma=float(smoothing_sigma_samples),
        axis=0,
        mode="nearest",
    )
    velocity = np.gradient(smooth_xyz, float(env.dt), axis=0)
    structure = np.ones(max(1, int(bridge_samples)), dtype=bool)
    axial_mask = (
        (velocity[:, 0] > float(progress_speed_mps))
        & (np.abs(velocity[:, 1]) < float(cross_axis_speed_mps))
        & (np.abs(velocity[:, 2]) < float(cross_axis_speed_mps))
    )
    lateral_mask = (
        (velocity[:, 1] < -float(progress_speed_mps))
        & (np.abs(velocity[:, 0]) < float(cross_axis_speed_mps))
        & (np.abs(velocity[:, 2]) < float(cross_axis_speed_mps))
    )
    axial_runs = [
        (int(begin), int(end))
        for begin, end in _true_runs(binary_closing(axial_mask, structure=structure))
        if int(end) - int(begin) >= int(minimum_stage_samples)
        and int(begin) >= int(0.15 * point_count)
    ]
    axial_runs = _merge_nearby_runs(axial_runs, merge_axial_gap_samples)
    if not axial_runs:
        raise ValueError("No sustained axial-cleaning phase was found.")
    axial_begin, axial_end = max(
        axial_runs,
        key=lambda run: float(task_xyz[run[1] - 1, 0] - task_xyz[run[0], 0]),
    )

    lateral_runs = [
        (int(begin), int(end))
        for begin, end in _true_runs(binary_closing(lateral_mask, structure=structure))
        if int(end) - int(begin) >= int(minimum_stage_samples)
        and int(begin) > int(axial_end)
    ]
    if not lateral_runs:
        raise ValueError("No sustained transverse-discharge phase was found.")
    lateral_begin, lateral_end = max(
        lateral_runs,
        key=lambda run: float(task_xyz[run[0], 1] - task_xyz[run[1] - 1, 1]),
    )

    bounds = np.asarray(
        [0, axial_begin, axial_end, lateral_begin, lateral_end, point_count],
        dtype=np.int64,
    )
    if np.any(np.diff(bounds) < int(minimum_stage_samples)):
        raise ValueError(f"Invalid five-stage bounds: {bounds.tolist()}")
    return bounds, _stage_diagnostics(task_xyz, bounds)


def process_bar_clean_archive(
    input_path: str | Path = DEFAULT_INPUT,
    output_path: str | Path = DEFAULT_OUTPUT,
    *,
    output_hz: float = 10.0,
    exclude_demo_ids: tuple[int, ...] = (),
    annotation_reference_path: str | Path | None = DEFAULT_ANNOTATION_REFERENCE,
) -> dict[str, object]:
    source_path = _resolve(input_path)
    destination = _resolve(output_path)
    with np.load(source_path, allow_pickle=False) as archive:
        _required(
            archive,
            {
                "flange_pose",
                "demo_time_s",
                "demo_bounds_indices",
                "demo_bar_poses_optitrack",
                "demo_obstacle_poses_optitrack",
                "sampling_hz",
                "optitrack_to_robot_rotation",
                "optitrack_to_robot_translation",
            },
        )
        trajectory_all = np.asarray(archive["flange_pose"], dtype=float)
        time_all = np.asarray(archive["demo_time_s"], dtype=float)
        source_bounds = np.asarray(archive["demo_bounds_indices"], dtype=np.int64)
        bar_poses = np.asarray(archive["demo_bar_poses_optitrack"], dtype=float)
        obstacle_poses = np.asarray(
            archive["demo_obstacle_poses_optitrack"], dtype=float
        )
        source_hz = float(np.asarray(archive["sampling_hz"]).item())
        tracker_rotation = np.asarray(
            archive["optitrack_to_robot_rotation"], dtype=float
        )
        tracker_translation = np.asarray(
            archive["optitrack_to_robot_translation"], dtype=float
        )

    factor_float = source_hz / float(output_hz)
    factor = int(round(factor_float))
    if factor < 1 or not np.isclose(factor_float, factor, atol=1e-9):
        raise ValueError(
            f"Source rate {source_hz:g} Hz must be an integer multiple of "
            f"output rate {float(output_hz):g} Hz."
        )
    if source_bounds.ndim != 2 or source_bounds.shape[1] != 2:
        raise ValueError("demo_bounds_indices must have shape (num_demos, 2).")
    source_demo_count = len(source_bounds)
    if bar_poses.shape != (source_demo_count, 7) or obstacle_poses.shape != (source_demo_count, 7):
        raise ValueError("Per-demo bar and obstacle poses must have shape (num_demos, 7).")
    excluded = {int(value) for value in exclude_demo_ids}
    source_demo_ids = np.asarray(
        [index for index in range(source_demo_count) if index not in excluded],
        dtype=np.int64,
    )
    if len(source_demo_ids) == 0:
        raise ValueError("exclude_demo_ids removed every BarClean demonstration.")

    reference_cutpoints: dict[int, np.ndarray] | None = None
    reference_hz: float | None = None
    reference_annotation_kind: str | None = None
    if annotation_reference_path is not None:
        reference_cutpoints, reference_hz, reference_annotation_kind = (
            _reference_cutpoint_times(annotation_reference_path)
        )
        missing_reference_ids = sorted(set(source_demo_ids.tolist()) - reference_cutpoints.keys())
        if missing_reference_ids:
            raise ValueError(
                "Reference annotations are missing source demo IDs: "
                f"{missing_reference_ids}"
            )

    env = BarCleanEnv(
        dt=1.0 / float(output_hz),
        optitrack_to_robot_rotation=tracker_rotation,
        optitrack_to_robot_translation=tracker_translation,
    )
    feature_names = np.asarray([spec["name"] for spec in env.feature_schema])
    trajectories: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    features: list[np.ndarray] = []
    demo_ids: list[np.ndarray] = []
    stage_labels: list[np.ndarray] = []
    global_bounds: list[np.ndarray] = []
    phase_diagnostics: list[dict[str, object]] = []
    output_lengths: list[int] = []
    offset = 0

    for demo_id, source_demo_id in enumerate(source_demo_ids):
        begin, end = source_bounds[source_demo_id]
        bar_pose = bar_poses[source_demo_id]
        obstacle_pose = obstacle_poses[source_demo_id]
        selected = _downsample_demo_indices(int(begin), int(end), factor)
        trajectory = trajectory_all[selected].copy()
        demo_time = time_all[selected].copy()
        demo_time -= demo_time[0]
        scene = BarInspectScene(
            bar_pose_optitrack=bar_pose,
            obstacle_pose_optitrack=obstacle_pose,
        )
        demo_features = env.compute_all_features_matrix(trajectory, scene=scene)
        if reference_cutpoints is None:
            local_bounds, diagnostics = motion_phase_stage_bounds(demo_features, env)
        else:
            local_bounds = _bounds_from_cutpoint_times(
                demo_time,
                reference_cutpoints[int(source_demo_id)],
            )
            diagnostics = _stage_diagnostics(
                _task_coordinates(demo_features, env),
                local_bounds,
            )
        labels = np.repeat(np.arange(5, dtype=np.int64), np.diff(local_bounds))
        if len(labels) != len(trajectory):
            raise RuntimeError("BarClean stage labels do not cover the complete demo.")

        trajectories.append(trajectory)
        timestamps.append(demo_time)
        features.append(demo_features)
        demo_ids.append(np.full(len(trajectory), demo_id, dtype=np.int64))
        stage_labels.append(labels)
        global_bounds.append(local_bounds + offset)
        phase_diagnostics.append(diagnostics)
        output_lengths.append(len(trajectory))
        offset += len(trajectory)

    output = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "timestamps": np.concatenate(timestamps),
        "trajectory": np.concatenate(trajectories),
        "features": np.concatenate(features),
        "feature_names": feature_names,
        "demo_id": np.concatenate(demo_ids),
        "coarse_stage_labels": np.concatenate(stage_labels),
        "coarse_bounds_indices": np.asarray(global_bounds, dtype=np.int64),
        "demo_bar_poses": bar_poses[source_demo_ids],
        "demo_obstacle_poses": obstacle_poses[source_demo_ids],
        "source_demo_ids": source_demo_ids,
        "excluded_source_demo_ids": np.asarray(sorted(excluded), dtype=np.int64),
        "boundary_task_xyz_m": np.asarray(
            [item["boundary_task_xyz_m"] for item in phase_diagnostics],
            dtype=float,
        ),
        "axial_progress_m": np.asarray(
            [item["axial_progress_m"] for item in phase_diagnostics], dtype=float
        ),
        "lateral_progress_m": np.asarray(
            [item["lateral_progress_m"] for item in phase_diagnostics], dtype=float
        ),
        "source_demo_bounds_indices": source_bounds,
        "source_sample_count": np.asarray(len(trajectory_all), dtype=np.int64),
        "source_hz": np.asarray(source_hz),
        "downsample_hz": np.asarray(float(output_hz)),
        "downsample_factor": np.asarray(factor, dtype=np.int64),
        "source_archive": np.asarray(_stored_path(source_path)),
        "cutpoint_annotation_kind": np.asarray(
            f"{float(output_hz):g}hz_time_mapped_from_"
            f"{reference_annotation_kind}"
            if reference_annotation_kind is not None
            else f"{float(output_hz):g}hz_task_motion_direction_change_points"
        ),
        "cutpoint_evaluation_role": np.asarray("motion_phase_reference"),
        "scene_pose_policy": np.asarray("per_demo_robust_static_lock"),
        "optitrack_to_robot_rotation": tracker_rotation,
        "optitrack_to_robot_translation": tracker_translation,
    }
    if annotation_reference_path is not None:
        output["cutpoint_annotation_source"] = np.asarray(
            _stored_path(annotation_reference_path)
        )
        output["cutpoint_annotation_source_hz"] = np.asarray(reference_hz)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **output)
    return {
        "output": str(destination),
        "demo_count": len(source_demo_ids),
        "source_demo_ids": source_demo_ids.tolist(),
        "excluded_source_demo_ids": sorted(excluded),
        "source_hz": source_hz,
        "output_hz": float(output_hz),
        "points_per_demo": output_lengths,
        "stage_lengths_per_demo": np.diff(np.asarray(global_bounds), axis=1).tolist(),
        "boundary_task_xyz_m": [
            item["boundary_task_xyz_m"] for item in phase_diagnostics
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert motion-trimmed BarClean demos into a reusable training archive."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-hz", type=float, default=10.0)
    parser.add_argument(
        "--exclude-demo-ids",
        default="",
        help="Comma-separated source demo IDs to omit; default: none.",
    )
    parser.add_argument(
        "--annotation-reference",
        type=Path,
        default=DEFAULT_ANNOTATION_REFERENCE,
        help="Archive whose cutpoint times are mapped onto the new sampling grid.",
    )
    args = parser.parse_args()
    summary = process_bar_clean_archive(
        args.input,
        args.output,
        output_hz=args.output_hz,
        exclude_demo_ids=tuple(
            int(value)
            for value in str(args.exclude_demo_ids).split(",")
            if value.strip()
        ),
        annotation_reference_path=args.annotation_reference,
    )
    print(summary)


if __name__ == "__main__":
    main()
