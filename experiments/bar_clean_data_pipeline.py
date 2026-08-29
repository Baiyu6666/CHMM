#!/usr/bin/env python3
"""BarClean rosbag pipeline with interactive Matplotlib review gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import threading

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_CONFIG = (
    PROJECT_ROOT
    / "robot/stage_cons_iiwa14/ros_ws/src/stage_iiwa_sim/config/demo_scene.json"
)
DEFAULT_ENV_CONFIG = PROJECT_ROOT / "configs/envs/BarClean.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.BarClean import BarCleanEnv, load_BarClean  # noqa: E402
from envs.BarInspect import BarInspectScene  # noqa: E402
from experiments.bar_inspect_processing import nearest_indices, robust_static_pose  # noqa: E402
from experiments.extract_bar_inspection_rosbag import (  # noqa: E402
    BAR_TOPIC,
    JOINT_TOPIC,
    OBSTACLE_TOPIC,
    iiwa14_fk,
    read_recording,
)
from experiments.process_bar_clean_demos import (  # noqa: E402
    _task_coordinates,
    motion_phase_stage_bounds,
    process_bar_clean_archive,
)


REVIEW_SCHEMA_VERSION = 1
ANALYSIS_ARCHIVE = "analysis.npz"
REVIEW_FILE = "review.json"
LOADER_INPUT_ARCHIVE = "loader_input.npz"
ANNOTATION_ARCHIVE = "reviewed_cutpoints.npz"
DEFAULT_OUTPUT_HZ = 5.0
DEFAULT_TRAINING_ARCHIVE = "training_5hz.npz"


def _json_dump(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _stored_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _config_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _true_runs(mask: np.ndarray) -> np.ndarray:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return np.column_stack((edges[::2], edges[1::2])).astype(np.int64)


def detect_stationary_runs(
    speed: np.ndarray,
    sampling_hz: float,
    *,
    threshold_rad_s: float,
    minimum_dwell_s: float,
    bridge_s: float,
) -> np.ndarray:
    """Return half-open stationary runs on a uniform analysis grid."""
    structure = np.ones(max(1, int(round(bridge_s * sampling_hz))), dtype=bool)
    stationary = binary_closing(
        np.asarray(speed, dtype=float) < float(threshold_rad_s),
        structure=structure,
    )
    runs = _true_runs(stationary)
    minimum_samples = max(1, int(round(minimum_dwell_s * sampling_hz)))
    return runs[(runs[:, 1] - runs[:, 0]) >= minimum_samples]


def _endpoint_diagnostics(task_xyz: np.ndarray, endpoints: np.ndarray) -> dict:
    distances = np.linalg.norm(
        np.asarray(task_xyz)[:, None, :] - np.asarray(endpoints)[None, :, :],
        axis=2,
    )
    nearest = np.argmin(distances, axis=0)
    minimum = distances[nearest, np.arange(len(endpoints))]
    ordered = bool(
        nearest[0] < min(nearest[1], nearest[2])
        and max(nearest[1], nearest[2]) < nearest[3]
    )
    return {
        "endpoint_min_distance_m": np.round(minimum, 6).tolist(),
        "endpoint_nearest_indices": nearest.astype(int).tolist(),
        "endpoint_order_valid": ordered,
    }


def propose_demo_intervals(
    dwell_runs: np.ndarray,
    task_xyz: np.ndarray,
    sampling_hz: float,
    endpoints: np.ndarray,
    *,
    minimum_duration_s: float,
    endpoint_tolerance_m: float,
    bar_north_end_axial_m: float,
    north_start_margin_m: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """Classify between-dwell intervals as BarClean demo proposals."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    for interval_id in range(max(0, len(dwell_runs) - 1)):
        begin = int(dwell_runs[interval_id, 1])
        end = int(dwell_runs[interval_id + 1, 0])
        if end <= begin:
            continue
        diagnostics = _endpoint_diagnostics(task_xyz[begin:end], endpoints)
        duration_s = float((end - begin) / sampling_hz)
        close = max(diagnostics["endpoint_min_distance_m"]) <= endpoint_tolerance_m
        long_enough = duration_s >= minimum_duration_s
        start_window_samples = max(1, int(round(0.25 * sampling_hz)))
        start_window_end = min(end, begin + start_window_samples)
        start_task_xyz = np.median(task_xyz[begin:start_window_end], axis=0)
        north_limit = float(bar_north_end_axial_m - north_start_margin_m)
        starts_north_of_bar = bool(start_task_xyz[0] < north_limit)
        row = {
            "source_interval_id": int(interval_id),
            "start_index": begin,
            "end_index": end,
            "duration_s": duration_s,
            "start_task_xyz_m": np.round(start_task_xyz, 6).tolist(),
            "bar_north_end_axial_m": float(bar_north_end_axial_m),
            "north_start_limit_axial_m": north_limit,
            "start_north_of_bar": starts_north_of_bar,
            **diagnostics,
        }
        reasons = []
        if not long_enough:
            reasons.append("duration below minimum")
        if not close:
            reasons.append("not all task endpoints reached")
        if not starts_north_of_bar:
            reasons.append("start is not north of the bar's north end")
        if reasons:
            row["rejection_reasons"] = reasons
            rejected.append(row)
        else:
            accepted.append(row)
    return accepted, rejected


def _fallback_stage_bounds(
    task_xyz: np.ndarray,
    endpoints: np.ndarray,
    minimum_stage_samples: int = 5,
) -> np.ndarray:
    """Use sequential endpoint proximity when the motion detector has no proposal."""
    cursor = minimum_stage_samples
    cutpoints = []
    point_count = len(task_xyz)
    for endpoint_index, endpoint in enumerate(endpoints):
        remaining_endpoints = len(endpoints) - endpoint_index - 1
        stop = point_count - minimum_stage_samples * (remaining_endpoints + 1)
        if stop <= cursor:
            raise ValueError("Demo is too short for fallback endpoint cutpoints.")
        local = int(
            np.argmin(
                np.linalg.norm(task_xyz[cursor:stop] - endpoint[None, :], axis=1)
            )
        )
        cutpoint = cursor + local
        cutpoints.append(cutpoint)
        cursor = cutpoint + minimum_stage_samples
    bounds = np.asarray([0, *cutpoints, point_count], dtype=np.int64)
    if np.any(np.diff(bounds) < minimum_stage_samples):
        raise ValueError(f"Invalid fallback stage proposal: {bounds.tolist()}")
    return bounds


def propose_stage_bounds(
    features: np.ndarray,
    task_xyz: np.ndarray,
    env: BarCleanEnv,
) -> tuple[np.ndarray, str]:
    try:
        bounds, _ = motion_phase_stage_bounds(features, env)
        return bounds, "task_motion_direction"
    except ValueError:
        return (
            _fallback_stage_bounds(task_xyz, env.stage_endpoint_positions_bar),
            "ordered_endpoint_proximity_fallback",
        )


def _scene_transform(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transform = payload["optitrack_to_robot"]
    rotation = np.asarray(transform["rotation"], dtype=float)
    translation = np.asarray(transform["translation"], dtype=float)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("Invalid optitrack_to_robot transform in scene config.")
    return rotation, translation


def _review_demo(
    demo_id: int,
    begin: int,
    end: int,
    source_interval_id: int | None,
    features: np.ndarray,
    task_xyz: np.ndarray,
    env: BarCleanEnv,
    sampling_hz: float,
    diagnostics: dict | None = None,
) -> dict:
    local_bounds, proposal_method = propose_stage_bounds(
        features[begin:end], task_xyz[begin:end], env
    )
    return {
        "demo_id": int(demo_id),
        "source_interval_id": source_interval_id,
        "start_index": int(begin),
        "end_index": int(end),
        "start_s": float(begin / sampling_hz),
        "end_s": float(end / sampling_hz),
        "duration_s": float((end - begin) / sampling_hz),
        "automatic_start_index": int(begin),
        "automatic_end_index": int(end),
        "cutpoints_local_indices": local_bounds[1:-1].astype(int).tolist(),
        "automatic_cutpoints_local_indices": local_bounds[1:-1].astype(int).tolist(),
        "cutpoint_proposal_method": proposal_method,
        "diagnostics": diagnostics or {},
    }


def prepare_dataset(
    bag_path: str | Path,
    dataset_directory: str | Path,
    *,
    analysis_hz: float = 20.0,
    stationary_threshold_rad_s: float = 0.10,
    minimum_dwell_s: float = 3.0,
    stationary_bridge_s: float = 0.35,
    minimum_demo_duration_s: float = 15.0,
    endpoint_tolerance_m: float = 0.065,
    north_start_margin_m: float = 0.0,
    scene_config_path: str | Path = DEFAULT_SCENE_CONFIG,
    overwrite: bool = False,
) -> dict:
    """Read one ROS bag and write automatic proposals for human review."""
    bag = Path(bag_path).expanduser().resolve()
    destination = Path(dataset_directory).expanduser().resolve()
    scene_config = Path(scene_config_path).expanduser().resolve()
    if not bag.is_file():
        raise FileNotFoundError(f"ROS bag does not exist: {bag}")
    if not scene_config.is_file():
        raise FileNotFoundError(f"Scene config does not exist: {scene_config}")
    if destination.exists() and any(destination.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Dataset directory is not empty: {destination}; use --overwrite explicitly."
            )
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    raw_times, raw_values = read_recording(
        bag, JOINT_TOPIC, BAR_TOPIC, OBSTACLE_TOPIC
    )
    common_start = max(values[0] for values in raw_times.values())
    common_end = min(values[-1] for values in raw_times.values())
    dt = 1.0 / float(analysis_hz)
    grid_absolute = np.arange(math.ceil(common_start / dt) * dt, common_end, dt)
    if len(grid_absolute) < 2:
        raise RuntimeError("ROS bag has no usable common time interval.")
    grid_s = grid_absolute - grid_absolute[0]

    joint_indices, joint_gaps = nearest_indices(
        raw_times[JOINT_TOPIC], grid_absolute
    )
    bar_indices, bar_gaps = nearest_indices(raw_times[BAR_TOPIC], grid_absolute)
    obstacle_indices, obstacle_gaps = nearest_indices(
        raw_times[OBSTACLE_TOPIC], grid_absolute
    )
    joint_positions = raw_values[JOINT_TOPIC][joint_indices]
    smooth = gaussian_filter1d(
        joint_positions,
        sigma=max(0.5, 0.1 * analysis_hz),
        axis=0,
        mode="nearest",
    )
    joint_velocity = np.gradient(smooth, dt, axis=0)
    joint_speed = np.linalg.norm(joint_velocity, axis=1)
    flange_pose = iiwa14_fk(joint_positions)
    bar_grid = raw_values[BAR_TOPIC][bar_indices]
    obstacle_grid = raw_values[OBSTACLE_TOPIC][obstacle_indices]

    bar_lock = robust_static_pose(
        raw_values[BAR_TOPIC],
        position_floor_m=0.02,
        local_axis=np.asarray([1.0, 0.0, 0.0]),
        axis_floor_rad=np.deg2rad(8.0),
    )
    obstacle_lock = robust_static_pose(
        raw_values[OBSTACLE_TOPIC], position_floor_m=0.02
    )
    tracker_rotation, tracker_translation = _scene_transform(scene_config)
    scene_definition = json.loads(scene_config.read_text(encoding="utf-8"))
    obstacle_by_name = {
        str(value["name"]): value for value in scene_definition["obstacles"]
    }
    planning_obstacle = dict(scene_definition["planning_obstacle"])
    obstacle_kwargs = {}
    if planning_obstacle["type"] == "circle":
        obstacle_kwargs["obstacle_center"] = obstacle_by_name[
            str(planning_obstacle["obstacle"])
        ]["locked_pose_robot"][:3]
    elif planning_obstacle["type"] == "capsule":
        obstacle_kwargs["obstacle_endpoints"] = [
            obstacle_by_name[str(name)]["locked_pose_robot"][:3]
            for name in planning_obstacle["endpoint_obstacles"]
        ]
    else:
        raise ValueError("Scene planning_obstacle must be circle or capsule.")
    scene_centerline = dict(scene_definition["bar"]["lateral_centerline"])
    env = BarCleanEnv(
        dt=dt,
        optitrack_to_robot_rotation=tracker_rotation,
        optitrack_to_robot_translation=tracker_translation,
        **obstacle_kwargs,
        bar_lateral_centerline=scene_centerline,
    )
    scene = BarInspectScene(
        bar_pose_optitrack=bar_lock["pose"],
        obstacle_pose_optitrack=(
            obstacle_lock["pose"]
            if planning_obstacle["type"] == "circle"
            else None
        ),
        bar_lateral_centerline=scene_centerline,
    )
    features = env.compute_all_features_matrix(flange_pose, scene=scene)
    task_xyz = _task_coordinates(features, env)
    feature_names = np.asarray([spec["name"] for spec in env.feature_schema])

    dwell_runs = detect_stationary_runs(
        joint_speed,
        analysis_hz,
        threshold_rad_s=stationary_threshold_rad_s,
        minimum_dwell_s=minimum_dwell_s,
        bridge_s=stationary_bridge_s,
    )
    bar_outline_u = env.task_definition["scene_geometry"]["bar_outline_u"]
    bar_north_end_axial_m = float(min(bar_outline_u))
    accepted, rejected = propose_demo_intervals(
        dwell_runs,
        task_xyz,
        analysis_hz,
        env.stage_endpoint_positions_bar,
        minimum_duration_s=minimum_demo_duration_s,
        endpoint_tolerance_m=endpoint_tolerance_m,
        bar_north_end_axial_m=bar_north_end_axial_m,
        north_start_margin_m=north_start_margin_m,
    )
    if not accepted:
        raise RuntimeError(
            "Automatic segmentation found no complete BarClean demonstrations. "
            "Adjust the proposal thresholds and prepare again."
        )

    demos = [
        _review_demo(
            demo_id,
            row["start_index"],
            row["end_index"],
            row["source_interval_id"],
            features,
            task_xyz,
            env,
            analysis_hz,
            diagnostics={
                key: value
                for key, value in row.items()
                if key
                not in {
                    "start_index",
                    "end_index",
                    "source_interval_id",
                    "duration_s",
                }
            },
        )
        for demo_id, row in enumerate(accepted)
    ]

    archive_path = destination / ANALYSIS_ARCHIVE
    np.savez_compressed(
        archive_path,
        schema_version=np.asarray(REVIEW_SCHEMA_VERSION, dtype=np.int64),
        timestamps_s=grid_s,
        joint_positions=joint_positions,
        joint_velocity=joint_velocity,
        joint_speed=joint_speed,
        flange_pose=flange_pose,
        bar_pose_optitrack=bar_grid,
        obstacle_pose_optitrack=obstacle_grid,
        features=features,
        feature_names=feature_names,
        task_xyz=task_xyz,
        dwell_runs=dwell_runs,
        sampling_hz=np.asarray(analysis_hz),
        source_bag=np.asarray(_stored_path(bag)),
        optitrack_to_robot_rotation=tracker_rotation,
        optitrack_to_robot_translation=tracker_translation,
        global_bar_pose_optitrack=bar_lock["pose"],
        global_obstacle_pose_optitrack=obstacle_lock["pose"],
        scene_config_json=np.asarray(json.dumps(scene_definition, sort_keys=True)),
    )
    review = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "task_id": "BarClean",
        "dataset_id": destination.name,
        "source_bag": _stored_path(bag),
        "analysis_archive": ANALYSIS_ARCHIVE,
        "analysis_hz": float(analysis_hz),
        "settings": {
            "stationary_threshold_rad_s": float(stationary_threshold_rad_s),
            "minimum_dwell_s": float(minimum_dwell_s),
            "stationary_bridge_s": float(stationary_bridge_s),
            "minimum_demo_duration_s": float(minimum_demo_duration_s),
            "endpoint_tolerance_m": float(endpoint_tolerance_m),
            "bar_north_end_axial_m": bar_north_end_axial_m,
            "north_start_margin_m": float(north_start_margin_m),
            "scene_config": _stored_path(scene_config),
            "scene_config_sha256": _config_fingerprint(scene_config),
            "task_definition": _stored_path(env.task_definition_path),
            "task_definition_sha256": _config_fingerprint(env.task_definition_path),
        },
        "quality": {
            "max_joint_resample_gap_s": float(np.max(joint_gaps)),
            "max_bar_resample_gap_s": float(np.max(bar_gaps)),
            "max_obstacle_resample_gap_s": float(np.max(obstacle_gaps)),
            "all_analysis_values_finite": bool(
                np.isfinite(joint_positions).all()
                and np.isfinite(flange_pose).all()
                and np.isfinite(features).all()
            ),
        },
        "automatic_rejected_intervals": rejected,
        "demos": demos,
        "review": {
            "demo_boundaries_confirmed": False,
            "cutpoints_confirmed": False,
        },
        "export": {"status": "pending"},
    }
    _json_dump(destination / REVIEW_FILE, review)
    return review


def _load_analysis(dataset_directory: Path) -> dict[str, np.ndarray]:
    review = json.loads((dataset_directory / REVIEW_FILE).read_text(encoding="utf-8"))
    with np.load(dataset_directory / review["analysis_archive"], allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _validate_demo_rows(demos: list[dict], point_count: int) -> None:
    if not demos:
        raise ValueError("At least one demonstration is required.")
    previous_end = 0
    for expected_id, demo in enumerate(demos):
        begin = int(demo["start_index"])
        end = int(demo["end_index"])
        if int(demo["demo_id"]) != expected_id:
            raise ValueError("Demo IDs must be consecutive and zero based.")
        if not (0 <= begin < end <= point_count):
            raise ValueError(f"Invalid demo {expected_id} bounds {begin}:{end}.")
        if expected_id and begin < previous_end:
            raise ValueError("Demonstrations must be ordered and non-overlapping.")
        local = np.asarray(demo["cutpoints_local_indices"], dtype=int)
        bounds = np.r_[0, local, end - begin]
        if local.shape != (4,) or np.any(np.diff(bounds) <= 0):
            raise ValueError(f"Demo {expected_id} needs four ordered internal cutpoints.")
        previous_end = end


def _review_state(dataset_directory: Path) -> dict:
    review = json.loads((dataset_directory / REVIEW_FILE).read_text(encoding="utf-8"))
    analysis = _load_analysis(dataset_directory)
    _validate_demo_rows(review["demos"], len(analysis["timestamps_s"]))
    hz = float(review["analysis_hz"])
    payload = dict(review)
    payload["demos"] = []
    for demo in review["demos"]:
        item = dict(demo)
        item["start_s"] = float(item["start_index"] / hz)
        item["end_s"] = float(item["end_index"] / hz)
        item["duration_s"] = float((item["end_index"] - item["start_index"]) / hz)
        item["cutpoints_local_s"] = [
            float(value / hz) for value in item["cutpoints_local_indices"]
        ]
        payload["demos"].append(item)
    return payload


class ReviewStore:
    def __init__(self, dataset_directory: str | Path):
        self.directory = Path(dataset_directory).expanduser().resolve()
        self.path = self.directory / REVIEW_FILE
        if not self.path.is_file():
            raise FileNotFoundError(f"Review file does not exist: {self.path}")
        self._lock = threading.RLock()

    def load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def state(self) -> dict:
        with self._lock:
            return _review_state(self.directory)

    def update_demo_boundaries(self, rows: list[dict]) -> dict:
        with self._lock:
            review = self.load()
            analysis = _load_analysis(self.directory)
            hz = float(review["analysis_hz"])
            point_count = len(analysis["timestamps_s"])
            env = BarCleanEnv(
                dt=1.0 / hz,
                optitrack_to_robot_rotation=analysis["optitrack_to_robot_rotation"],
                optitrack_to_robot_translation=analysis["optitrack_to_robot_translation"],
            )
            previous_by_bounds = {
                (int(item["start_index"]), int(item["end_index"])): item
                for item in review["demos"]
            }
            demos = []
            for demo_id, row in enumerate(rows):
                begin = int(round(float(row["start_s"]) * hz))
                end = int(round(float(row["end_s"]) * hz))
                if not (0 <= begin < end <= point_count):
                    raise ValueError(f"Invalid demo {demo_id} times.")
                previous = previous_by_bounds.get((begin, end))
                proposed = _review_demo(
                    demo_id,
                    begin,
                    end,
                    None if previous is None else previous.get("source_interval_id"),
                    analysis["features"],
                    analysis["task_xyz"],
                    env,
                    hz,
                )
                if previous is not None:
                    proposed["cutpoints_local_indices"] = list(
                        previous["cutpoints_local_indices"]
                    )
                    proposed["automatic_cutpoints_local_indices"] = list(
                        previous["automatic_cutpoints_local_indices"]
                    )
                    proposed["cutpoint_proposal_method"] = previous[
                        "cutpoint_proposal_method"
                    ]
                    proposed["diagnostics"] = dict(previous.get("diagnostics", {}))
                demos.append(proposed)
            _validate_demo_rows(demos, point_count)
            review["demos"] = demos
            review["review"]["demo_boundaries_confirmed"] = False
            review["review"]["cutpoints_confirmed"] = False
            review["export"] = {"status": "pending"}
            _json_dump(self.path, review)
            return self.state()

    def update_cutpoints(self, demo_id: int, cutpoints_local_s: list[float]) -> dict:
        with self._lock:
            review = self.load()
            hz = float(review["analysis_hz"])
            if not 0 <= int(demo_id) < len(review["demos"]):
                raise ValueError(f"Unknown demo ID {demo_id}.")
            demo = review["demos"][int(demo_id)]
            cuts = np.asarray(
                [int(round(float(value) * hz)) for value in cutpoints_local_s],
                dtype=int,
            )
            length = int(demo["end_index"] - demo["start_index"])
            bounds = np.r_[0, cuts, length]
            if cuts.shape != (4,) or np.any(np.diff(bounds) <= 0):
                raise ValueError("Four ordered cutpoints strictly inside the demo are required.")
            demo["cutpoints_local_indices"] = cuts.tolist()
            review["review"]["cutpoints_confirmed"] = False
            review["export"] = {"status": "pending"}
            _json_dump(self.path, review)
            return self.state()

    def update_demo_annotations(
        self,
        demo_id: int,
        annotation_recording_s: list[float],
    ) -> dict:
        """Save start, four cutpoints, and end as one ordered annotation."""
        with self._lock:
            review = self.load()
            analysis = _load_analysis(self.directory)
            hz = float(review["analysis_hz"])
            if not 0 <= int(demo_id) < len(review["demos"]):
                raise ValueError(f"Unknown demo ID {demo_id}.")
            indices = np.asarray(
                [int(round(float(value) * hz)) for value in annotation_recording_s],
                dtype=int,
            )
            if indices.shape != (6,) or np.any(np.diff(indices) <= 0):
                raise ValueError("Start, CP1–CP4, and end must be strictly ordered.")
            begin, end = int(indices[0]), int(indices[-1])
            if not (0 <= begin < end <= len(analysis["timestamps_s"])):
                raise ValueError("Demo annotation is outside the recording.")
            demos = review["demos"]
            if demo_id and begin < int(demos[demo_id - 1]["end_index"]):
                raise ValueError("Demo start overlaps the previous demo.")
            if demo_id + 1 < len(demos) and end > int(demos[demo_id + 1]["start_index"]):
                raise ValueError("Demo end overlaps the next demo.")
            demo = demos[int(demo_id)]
            demo["start_index"] = begin
            demo["end_index"] = end
            demo["cutpoints_local_indices"] = (indices[1:-1] - begin).tolist()
            review["review"]["demo_boundaries_confirmed"] = False
            review["review"]["cutpoints_confirmed"] = False
            review["export"] = {"status": "pending"}
            _validate_demo_rows(demos, len(analysis["timestamps_s"]))
            _json_dump(self.path, review)
            return self.state()

    def confirm_demo_boundaries(self) -> dict:
        with self._lock:
            review = self.load()
            analysis = _load_analysis(self.directory)
            _validate_demo_rows(review["demos"], len(analysis["timestamps_s"]))
            review["review"]["demo_boundaries_confirmed"] = True
            review["review"]["cutpoints_confirmed"] = False
            _json_dump(self.path, review)
            return self.state()

    def confirm_cutpoints(self) -> dict:
        with self._lock:
            review = self.load()
            if not review["review"]["demo_boundaries_confirmed"]:
                raise ValueError("Confirm demonstration boundaries first.")
            analysis = _load_analysis(self.directory)
            _validate_demo_rows(review["demos"], len(analysis["timestamps_s"]))
            review["review"]["cutpoints_confirmed"] = True
            _json_dump(self.path, review)
            return self.state()

    def export(self, output_hz: float, output_name: str, activate: bool) -> dict:
        with self._lock:
            export_reviewed_dataset(
                self.directory,
                output_hz=output_hz,
                output_name=output_name,
                activate=activate,
            )
            return self.state()


def _robust_demo_scene_pose(poses: np.ndarray, *, is_bar: bool) -> np.ndarray:
    kwargs = {"position_floor_m": 0.02}
    if is_bar:
        kwargs.update(
            local_axis=np.asarray([1.0, 0.0, 0.0]),
            axis_floor_rad=np.deg2rad(8.0),
        )
    return np.asarray(robust_static_pose(poses, **kwargs)["pose"], dtype=float)


def export_reviewed_dataset(
    dataset_directory: str | Path,
    *,
    output_hz: float = DEFAULT_OUTPUT_HZ,
    output_name: str = DEFAULT_TRAINING_ARCHIVE,
    activate: bool = False,
) -> dict:
    """Export only after both human-review gates have been confirmed."""
    directory = Path(dataset_directory).expanduser().resolve()
    review_path = directory / REVIEW_FILE
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not review["review"]["demo_boundaries_confirmed"]:
        raise ValueError("Demonstration boundaries have not been confirmed.")
    if not review["review"]["cutpoints_confirmed"]:
        raise ValueError("Internal cutpoints have not been confirmed.")
    analysis = _load_analysis(directory)
    _validate_demo_rows(review["demos"], len(analysis["timestamps_s"]))
    source_hz = float(review["analysis_hz"])
    if output_hz <= 0.0:
        raise ValueError("output_hz must be positive.")

    flange_chunks = []
    time_chunks = []
    joint_chunks = []
    velocity_chunks = []
    bar_chunks = []
    obstacle_chunks = []
    demo_id_chunks = []
    source_bounds = []
    annotation_bounds = []
    bar_poses = []
    obstacle_poses = []
    offset = 0
    for demo in review["demos"]:
        begin, end = int(demo["start_index"]), int(demo["end_index"])
        count = end - begin
        local_time = np.arange(count, dtype=float) / source_hz
        flange_chunks.append(analysis["flange_pose"][begin:end])
        time_chunks.append(local_time)
        joint_chunks.append(analysis["joint_positions"][begin:end])
        velocity_chunks.append(analysis["joint_velocity"][begin:end])
        bar_chunks.append(analysis["bar_pose_optitrack"][begin:end])
        obstacle_chunks.append(analysis["obstacle_pose_optitrack"][begin:end])
        demo_id_chunks.append(np.full(count, int(demo["demo_id"]), dtype=np.int64))
        source_bounds.append([offset, offset + count])
        local_cuts = np.asarray(demo["cutpoints_local_indices"], dtype=np.int64)
        annotation_bounds.append(np.r_[0, local_cuts, count] + offset)
        bar_poses.append(
            _robust_demo_scene_pose(analysis["bar_pose_optitrack"][begin:end], is_bar=True)
        )
        obstacle_poses.append(
            _robust_demo_scene_pose(
                analysis["obstacle_pose_optitrack"][begin:end], is_bar=False
            )
        )
        offset += count

    loader_input = directory / LOADER_INPUT_ARCHIVE
    np.savez_compressed(
        loader_input,
        schema_version=np.asarray(1, dtype=np.int64),
        timestamps_s=np.concatenate(time_chunks),
        demo_time_s=np.concatenate(time_chunks),
        joint_positions=np.concatenate(joint_chunks),
        joint_velocity=np.concatenate(velocity_chunks),
        flange_pose=np.concatenate(flange_chunks),
        bar_pose_optitrack=np.concatenate(bar_chunks),
        obstacle_pose_optitrack=np.concatenate(obstacle_chunks),
        demo_id=np.concatenate(demo_id_chunks),
        demo_bounds_indices=np.asarray(source_bounds, dtype=np.int64),
        demo_bar_poses_optitrack=np.asarray(bar_poses, dtype=float),
        demo_obstacle_poses_optitrack=np.asarray(obstacle_poses, dtype=float),
        sampling_hz=np.asarray(source_hz),
        source_bag=np.asarray(review["source_bag"]),
        scene_pose_policy=np.asarray("per_demo_robust_static_lock"),
        optitrack_to_robot_rotation=analysis["optitrack_to_robot_rotation"],
        optitrack_to_robot_translation=analysis["optitrack_to_robot_translation"],
        scene_config_json=analysis["scene_config_json"],
    )

    annotation = directory / ANNOTATION_ARCHIVE
    np.savez_compressed(
        annotation,
        schema_version=np.asarray(1, dtype=np.int64),
        timestamps=np.concatenate(time_chunks),
        coarse_bounds_indices=np.asarray(annotation_bounds, dtype=np.int64),
        source_demo_ids=np.arange(len(review["demos"]), dtype=np.int64),
        downsample_hz=np.asarray(source_hz),
        cutpoint_annotation_kind=np.asarray("human_reviewed_gui_stage_boundaries"),
        source_bag=np.asarray(review["source_bag"]),
    )
    output = Path(output_name)
    if not output.is_absolute():
        output = directory / output
    summary = process_bar_clean_archive(
        loader_input,
        output,
        output_hz=float(output_hz),
        annotation_reference_path=annotation,
    )
    bundle = load_BarClean(
        n_demos=len(review["demos"]),
        processed_demo_path=output,
        source_demo_ids=list(range(len(review["demos"]))),
    )
    validation = {
        "loader": "envs.BarClean.load_BarClean",
        "training_archive": _stored_path(output),
        "demo_count": len(bundle.demos),
        "points_per_demo": [len(demo) for demo in bundle.demos],
        "feature_count": int(bundle.features[0].shape[1]),
        "feature_names": [spec["name"] for spec in bundle.feature_schema],
        "true_cutpoints": [np.asarray(value, dtype=int).tolist() for value in bundle.true_cutpoints],
        "all_finite": bool(
            all(np.isfinite(demo).all() for demo in bundle.demos)
            and all(np.isfinite(feature).all() for feature in bundle.features)
        ),
    }
    _json_dump(directory / "loader_validation.json", validation)
    if activate:
        config = json.loads(DEFAULT_ENV_CONFIG.read_text(encoding="utf-8"))
        config["n_demos"] = len(bundle.demos)
        config["source_demo_ids"] = list(range(len(bundle.demos)))
        config["processed_demo_path"] = _stored_path(output)
        _json_dump(DEFAULT_ENV_CONFIG, config)
    review["export"] = {
        "status": "complete",
        "training_archive": _stored_path(output),
        "output_hz": float(output_hz),
        "activated": bool(activate),
        "summary": summary,
        "validation": validation,
    }
    _json_dump(review_path, review)
    return review["export"]


def _archive_scalar(archive, key: str):
    return np.asarray(archive[key]).item()


def merge_processed_datasets(
    source_specs: list[tuple[str | Path, list[int]]],
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Merge selected demos from reviewed BarClean training archives."""
    if not source_specs:
        raise ValueError("At least one source archive is required.")

    destination = Path(output_path).expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Merged archive already exists: {destination}; use --overwrite explicitly."
        )

    required = {
        "trajectory",
        "features",
        "feature_names",
        "timestamps",
        "demo_id",
        "coarse_stage_labels",
        "coarse_bounds_indices",
        "demo_bar_poses",
        "demo_obstacle_poses",
        "source_demo_ids",
        "boundary_task_xyz_m",
        "axial_progress_m",
        "lateral_progress_m",
        "downsample_hz",
        "cutpoint_annotation_kind",
        "cutpoint_evaluation_role",
        "scene_pose_policy",
        "optitrack_to_robot_rotation",
        "optitrack_to_robot_translation",
    }
    reference: dict | None = None
    chunks = {
        "trajectory": [],
        "features": [],
        "timestamps": [],
        "coarse_stage_labels": [],
        "demo_bar_poses": [],
        "demo_obstacle_poses": [],
        "boundary_task_xyz_m": [],
        "axial_progress_m": [],
        "lateral_progress_m": [],
    }
    merged_bounds = []
    merged_demo_ids = []
    provenance = []
    offset = 0

    for source_path, requested_ids in source_specs:
        path = Path(source_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Processed BarClean archive does not exist: {path}")
        ids = [int(value) for value in requested_ids]
        if not ids:
            raise ValueError(f"No source demo IDs requested from {path}.")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate source demo IDs requested from {path}: {ids}")

        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(required.difference(archive.files))
            if missing:
                raise ValueError(f"Source archive {path} is missing keys: {missing}")

            feature_names = np.asarray(archive["feature_names"])
            downsample_hz = float(_archive_scalar(archive, "downsample_hz"))
            tracker_rotation = np.asarray(
                archive["optitrack_to_robot_rotation"], dtype=float
            )
            tracker_translation = np.asarray(
                archive["optitrack_to_robot_translation"], dtype=float
            )
            annotation_kind = str(_archive_scalar(archive, "cutpoint_annotation_kind"))
            evaluation_role = str(_archive_scalar(archive, "cutpoint_evaluation_role"))
            scene_pose_policy = str(_archive_scalar(archive, "scene_pose_policy"))
            source_hz = (
                float(_archive_scalar(archive, "source_hz"))
                if "source_hz" in archive.files
                else None
            )
            downsample_factor = (
                int(_archive_scalar(archive, "downsample_factor"))
                if "downsample_factor" in archive.files
                else None
            )
            annotation_source_hz = (
                float(_archive_scalar(archive, "cutpoint_annotation_source_hz"))
                if "cutpoint_annotation_source_hz" in archive.files
                else None
            )
            metadata = {
                "feature_names": feature_names,
                "downsample_hz": downsample_hz,
                "tracker_rotation": tracker_rotation,
                "tracker_translation": tracker_translation,
                "annotation_kind": annotation_kind,
                "evaluation_role": evaluation_role,
                "scene_pose_policy": scene_pose_policy,
                "source_hz": source_hz,
                "downsample_factor": downsample_factor,
                "annotation_source_hz": annotation_source_hz,
            }
            if reference is None:
                reference = metadata
            else:
                if not np.array_equal(feature_names, reference["feature_names"]):
                    raise ValueError(f"Feature schema mismatch in source archive {path}.")
                if not np.isclose(downsample_hz, reference["downsample_hz"]):
                    raise ValueError(f"Downsample rate mismatch in source archive {path}.")
                if not np.allclose(
                    tracker_rotation, reference["tracker_rotation"], rtol=0.0, atol=1e-10
                ) or not np.allclose(
                    tracker_translation,
                    reference["tracker_translation"],
                    rtol=0.0,
                    atol=1e-10,
                ):
                    raise ValueError(
                        f"OptiTrack-to-robot transform mismatch in source archive {path}."
                    )
                for key in (
                    "annotation_kind",
                    "evaluation_role",
                    "scene_pose_policy",
                    "source_hz",
                    "downsample_factor",
                    "annotation_source_hz",
                ):
                    if metadata[key] != reference[key]:
                        raise ValueError(f"{key} mismatch in source archive {path}.")

            bounds = np.asarray(archive["coarse_bounds_indices"], dtype=np.int64)
            archive_source_ids = np.asarray(archive["source_demo_ids"], dtype=np.int64)
            if bounds.ndim != 2 or bounds.shape[1] != 6:
                raise ValueError(
                    f"coarse_bounds_indices in {path} must have shape (num_demos, 6)."
                )
            if archive_source_ids.shape != (len(bounds),):
                raise ValueError(f"source_demo_ids shape does not match bounds in {path}.")
            id_to_index = {int(source_id): index for index, source_id in enumerate(archive_source_ids)}
            if len(id_to_index) != len(archive_source_ids):
                raise ValueError(f"Source archive contains duplicate source_demo_ids: {path}")
            missing_ids = [source_id for source_id in ids if source_id not in id_to_index]
            if missing_ids:
                raise ValueError(
                    f"Requested source demo IDs {missing_ids} are absent from {path}; "
                    f"available IDs are {archive_source_ids.tolist()}."
                )

            trajectory = np.asarray(archive["trajectory"], dtype=float)
            features = np.asarray(archive["features"], dtype=float)
            timestamps = np.asarray(archive["timestamps"], dtype=float)
            labels = np.asarray(archive["coarse_stage_labels"], dtype=np.int64)
            bar_poses = np.asarray(archive["demo_bar_poses"], dtype=float)
            obstacle_poses = np.asarray(archive["demo_obstacle_poses"], dtype=float)
            boundary_xyz = np.asarray(archive["boundary_task_xyz_m"], dtype=float)
            axial_progress = np.asarray(archive["axial_progress_m"], dtype=float)
            lateral_progress = np.asarray(archive["lateral_progress_m"], dtype=float)

            for source_demo_id in ids:
                archive_index = int(id_to_index[source_demo_id])
                row = np.asarray(bounds[archive_index], dtype=np.int64)
                begin, end = int(row[0]), int(row[-1])
                if not (
                    0 <= begin < end <= len(trajectory)
                    and np.all(np.diff(row) > 0)
                ):
                    raise ValueError(
                        f"Invalid bounds for source demo {source_demo_id} in {path}: {row.tolist()}"
                    )
                new_demo_id = len(provenance)
                length = end - begin
                chunks["trajectory"].append(trajectory[begin:end].copy())
                chunks["features"].append(features[begin:end].copy())
                demo_time = timestamps[begin:end].copy()
                chunks["timestamps"].append(demo_time - demo_time[0])
                chunks["coarse_stage_labels"].append(labels[begin:end].copy())
                chunks["demo_bar_poses"].append(bar_poses[archive_index].copy())
                chunks["demo_obstacle_poses"].append(obstacle_poses[archive_index].copy())
                chunks["boundary_task_xyz_m"].append(boundary_xyz[archive_index].copy())
                chunks["axial_progress_m"].append(axial_progress[archive_index].copy())
                chunks["lateral_progress_m"].append(lateral_progress[archive_index].copy())
                merged_bounds.append((row - begin + offset).astype(np.int64))
                merged_demo_ids.append(np.full(length, new_demo_id, dtype=np.int64))
                provenance.append(
                    {
                        "merged_demo_id": int(new_demo_id),
                        "source_archive": _stored_path(path),
                        "source_demo_id": int(source_demo_id),
                        "source_archive_index": int(archive_index),
                        "source_bounds_indices": row.tolist(),
                        "points": int(length),
                    }
                )
                offset += length

    assert reference is not None
    output = {
        "schema_version": np.asarray(1, dtype=np.int64),
        "timestamps": np.concatenate(chunks["timestamps"]),
        "trajectory": np.concatenate(chunks["trajectory"]),
        "features": np.concatenate(chunks["features"]),
        "feature_names": np.asarray(reference["feature_names"]),
        "demo_id": np.concatenate(merged_demo_ids),
        "coarse_stage_labels": np.concatenate(chunks["coarse_stage_labels"]),
        "coarse_bounds_indices": np.asarray(merged_bounds, dtype=np.int64),
        "demo_bar_poses": np.asarray(chunks["demo_bar_poses"], dtype=float),
        "demo_obstacle_poses": np.asarray(chunks["demo_obstacle_poses"], dtype=float),
        "source_demo_ids": np.arange(len(provenance), dtype=np.int64),
        "excluded_source_demo_ids": np.asarray([], dtype=np.int64),
        "boundary_task_xyz_m": np.asarray(chunks["boundary_task_xyz_m"], dtype=float),
        "axial_progress_m": np.asarray(chunks["axial_progress_m"], dtype=float),
        "lateral_progress_m": np.asarray(chunks["lateral_progress_m"], dtype=float),
        "downsample_hz": np.asarray(reference["downsample_hz"], dtype=float),
        "cutpoint_annotation_kind": np.asarray(reference["annotation_kind"]),
        "cutpoint_evaluation_role": np.asarray(reference["evaluation_role"]),
        "scene_pose_policy": np.asarray(reference["scene_pose_policy"]),
        "optitrack_to_robot_rotation": np.asarray(reference["tracker_rotation"], dtype=float),
        "optitrack_to_robot_translation": np.asarray(reference["tracker_translation"], dtype=float),
        "source_archive": np.asarray("merged_processed_barclean_archives"),
        "merge_origin_archives": np.asarray(
            [item["source_archive"] for item in provenance]
        ),
        "merge_origin_source_demo_ids": np.asarray(
            [item["source_demo_id"] for item in provenance], dtype=np.int64
        ),
        "merge_origin_archive_indices": np.asarray(
            [item["source_archive_index"] for item in provenance], dtype=np.int64
        ),
    }
    if reference["source_hz"] is not None:
        output["source_hz"] = np.asarray(reference["source_hz"], dtype=float)
    if reference["downsample_factor"] is not None:
        output["downsample_factor"] = np.asarray(
            reference["downsample_factor"], dtype=np.int64
        )
    if reference["annotation_source_hz"] is not None:
        output["cutpoint_annotation_source_hz"] = np.asarray(
            reference["annotation_source_hz"], dtype=float
        )
    output["cutpoint_annotation_source"] = np.asarray(
        "merged_reviewed_cutpoint_archives"
    )

    if not (
        np.isfinite(output["trajectory"]).all()
        and np.isfinite(output["features"]).all()
        and np.isfinite(output["timestamps"]).all()
    ):
        raise ValueError("Merged BarClean archive contains non-finite values.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **output)
    temporary.replace(destination)

    merged_ids = list(range(len(provenance)))
    bundle = load_BarClean(
        n_demos=len(merged_ids),
        processed_demo_path=destination,
        source_demo_ids=merged_ids,
    )
    validation = {
        "loader": "envs.BarClean.load_BarClean",
        "training_archive": _stored_path(destination),
        "demo_count": len(bundle.demos),
        "source_demo_ids": merged_ids,
        "points_per_demo": [len(demo) for demo in bundle.demos],
        "true_cutpoints": [
            np.asarray(value, dtype=int).tolist() for value in bundle.true_cutpoints
        ],
        "all_finite": bool(
            all(np.isfinite(demo).all() for demo in bundle.demos)
            and all(np.isfinite(feature).all() for feature in bundle.features)
        ),
    }
    summary = {
        "schema_version": 1,
        "artifact_type": "merged_barclean_dataset",
        "output": _stored_path(destination),
        "demo_count": len(provenance),
        "downsample_hz": float(reference["downsample_hz"]),
        "demo_mapping": provenance,
        "validation": validation,
    }
    _json_dump(destination.parent / "merge_manifest.json", summary)
    _json_dump(destination.parent / "loader_validation.json", validation)
    return summary


class MatplotlibReviewApp:
    """Single-window reviewer for demo boundaries and internal cutpoints."""

    CONTEXT_SECONDS = 5.0
    ANNOTATION_LABELS = ("Start", "CP1", "CP2", "CP3", "CP4", "End")
    ANNOTATION_COLORS = ("#2ca02c", "#d62728", "#1f77b4", "#9467bd", "#8c564b", "#e41a1c")

    def __init__(self, store: ReviewStore):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, CheckButtons, TextBox

        self.plt = plt
        self.Button = Button
        self.store = store
        self.analysis = _load_analysis(store.directory)
        self.state = store.state()
        self.hz = float(self.state["analysis_hz"])
        self.selected_demo = 0
        self.active_annotation: int | None = None
        self.annotation_seconds: list[float] = []
        self.annotation_lines: list[list[object]] = []
        self.annotation_markers: list[object] = []
        self.feature_active_lines: list[object] = []
        self.active_spans: list[object] = []

        self.figure, self.axes = plt.subplots(2, 2, figsize=(14, 9))
        self.figure.subplots_adjust(bottom=0.22, top=0.89, hspace=0.32, wspace=0.25)
        self.message = self.figure.text(0.02, 0.02, "", color="tab:orange", fontsize=10)
        self._widgets(TextBox, CheckButtons)
        self.figure.canvas.mpl_connect("button_press_event", self._stage_press)
        self.figure.canvas.mpl_connect("motion_notify_event", self._stage_motion)
        self.figure.canvas.mpl_connect("button_release_event", self._stage_release)
        self._redraw()

    @property
    def stage_figure(self):
        return self.figure

    @property
    def stage_axes(self):
        return self.axes

    def _button(self, rectangle, label, callback):
        widget = self.Button(self.figure.add_axes(rectangle), label)
        widget.on_clicked(callback)
        return widget

    def _widgets(self, TextBox, CheckButtons) -> None:
        self.previous_button = self._button([0.02, 0.10, 0.065, 0.05], "Previous", self._previous_demo)
        self.next_button = self._button([0.09, 0.10, 0.065, 0.05], "Next", self._next_demo)
        self.remove_button = self._button([0.17, 0.10, 0.09, 0.05], "Remove", self._remove_demo)
        self.confirm_button = self._button([0.275, 0.10, 0.11, 0.05], "Confirm all", self._confirm_all)
        self.output_hz_box = TextBox(
            self.figure.add_axes([0.43, 0.10, 0.06, 0.05]),
            "Hz ",
            initial=f"{DEFAULT_OUTPUT_HZ:g}",
        )
        self.output_name_box = TextBox(
            self.figure.add_axes([0.55, 0.10, 0.18, 0.05]),
            "File ",
            initial=DEFAULT_TRAINING_ARCHIVE,
        )
        self.activate_check = CheckButtons(
            self.figure.add_axes([0.76, 0.08, 0.10, 0.08]), ["Activate"], [False]
        )
        self.export_button = self._button([0.89, 0.10, 0.08, 0.05], "Export", self._export)

    def _set_message(self, message: str) -> None:
        self.message.set_text(message)
        self.figure.canvas.draw_idle()

    def _review_label(self) -> str:
        boundary_ok = self.state["review"]["demo_boundaries_confirmed"]
        cutpoint_ok = self.state["review"]["cutpoints_confirmed"]
        return "CONFIRMED" if boundary_ok and cutpoint_ok else "needs review"

    def _annotation_indices(self) -> np.ndarray:
        return np.asarray(
            [int(round(value * self.hz)) for value in self.annotation_seconds], dtype=int
        )

    def _redraw(self) -> None:
        if not self.state["demos"]:
            return
        self.selected_demo = min(self.selected_demo, len(self.state["demos"]) - 1)
        demo = self.state["demos"][self.selected_demo]
        begin, end = int(demo["start_index"]), int(demo["end_index"])
        cuts = begin + np.asarray(demo["cutpoints_local_indices"], dtype=int)
        annotation_indices = np.r_[begin, cuts, end]
        self.annotation_seconds = (annotation_indices / self.hz).astype(float).tolist()

        point_count = len(self.analysis["timestamps_s"])
        context_samples = max(1, int(round(self.CONTEXT_SECONDS * self.hz)))
        context_begin = max(0, begin - context_samples)
        context_end = min(point_count, end + context_samples)
        context_time = np.arange(context_begin, context_end, dtype=float) / self.hz
        active_time = np.arange(begin, end, dtype=float) / self.hz
        task_xyz = np.asarray(self.analysis["task_xyz"], dtype=float)
        features = np.asarray(self.analysis["features"], dtype=float)
        names = [str(value) for value in self.analysis["feature_names"].tolist()]
        feature_names = ("table_dist", "bar_lateral_offset", "bar_axial_offset")

        for axis in self.axes.flat:
            axis.clear()
        path_axis = self.axes[0, 0]
        path_axis.plot(
            task_xyz[context_begin:context_end, 0],
            task_xyz[context_begin:context_end, 1],
            color="0.75",
            linewidth=1.0,
            label=f"±{self.CONTEXT_SECONDS:g}s context",
        )
        self.active_path_line, = path_axis.plot(
            task_xyz[begin:end, 0], task_xyz[begin:end, 1], color="tab:blue", linewidth=1.5,
            label="selected demo",
        )
        marker_indices = annotation_indices.copy()
        marker_indices[-1] = max(begin, end - 1)
        self.annotation_markers = []
        for index, (label, color, sample_index) in enumerate(
            zip(self.ANNOTATION_LABELS, self.ANNOTATION_COLORS, marker_indices)
        ):
            marker = "o" if index in (0, 5) else "x"
            line, = path_axis.plot(
                task_xyz[sample_index, 0],
                task_xyz[sample_index, 1],
                marker=marker,
                markersize=8,
                markeredgewidth=2,
                linestyle="None",
                color=color,
                label=label,
            )
            self.annotation_markers.append(line)
        path_axis.set(xlabel="bar axial [m]", ylabel="bar lateral [m]", title="task-frame path")
        path_axis.axis("equal")
        path_axis.grid(alpha=0.2)
        path_axis.legend(fontsize=8, ncol=3)

        feature_axes = tuple(self.axes.flat[1:])
        self.annotation_lines = [[] for _ in self.annotation_seconds]
        self.feature_active_lines = []
        self.active_spans = []
        for feature_name, axis in zip(feature_names, feature_axes):
            column = names.index(feature_name)
            axis.plot(
                context_time,
                features[context_begin:context_end, column],
                color="0.72",
                linewidth=1.0,
            )
            active_line, = axis.plot(
                active_time, features[begin:end, column], color="black", linewidth=1.2
            )
            self.feature_active_lines.append(active_line)
            self.active_spans.append(
                axis.axvspan(self.annotation_seconds[0], self.annotation_seconds[-1], color="tab:blue", alpha=0.08)
            )
            for index, (label, value, color) in enumerate(
                zip(self.ANNOTATION_LABELS, self.annotation_seconds, self.ANNOTATION_COLORS)
            ):
                line = axis.axvline(
                    value,
                    color=color,
                    linestyle="-" if index in (0, 5) else "--",
                    linewidth=2.2 if index in (0, 5) else 1.8,
                    picker=8,
                    label=label,
                )
                self.annotation_lines[index].append(line)
            axis.set(
                xlabel="recording time [s]",
                ylabel=feature_name,
                title=f"{feature_name} (gray = context)",
                xlim=(context_begin / self.hz, context_end / self.hz),
            )
            axis.grid(alpha=0.2)
        self.figure.suptitle(
            f"Demo {self.selected_demo}/{len(self.state['demos']) - 1}: drag Start, CP1–CP4, or End · {self._review_label()}",
            fontsize=14,
        )
        self.figure.canvas.draw_idle()

    def _select_demo(self, demo_id: int) -> None:
        self.selected_demo = int(np.clip(demo_id, 0, len(self.state["demos"]) - 1))
        self._redraw()

    def _previous_demo(self, _event) -> None:
        self._select_demo(self.selected_demo - 1)

    def _next_demo(self, _event) -> None:
        self._select_demo(self.selected_demo + 1)

    def _remove_demo(self, _event) -> None:
        if len(self.state["demos"]) <= 1:
            self._set_message("At least one demo must remain.")
            return
        rows = [
            {"start_s": float(demo["start_s"]), "end_s": float(demo["end_s"])}
            for index, demo in enumerate(self.state["demos"])
            if index != self.selected_demo
        ]
        self.state = self.store.update_demo_boundaries(rows)
        self.selected_demo = min(self.selected_demo, len(rows) - 1)
        self._set_message("Selected demo removed; confirmations reset.")
        self._redraw()

    def _nearest_annotation(self, event) -> int | None:
        if event.inaxes not in tuple(self.axes.flat[1:]) or event.xdata is None:
            return None
        distances = [
            abs(float(event.x) - event.inaxes.transData.transform((value, 0.0))[0])
            for value in self.annotation_seconds
        ]
        closest = int(np.argmin(distances))
        return closest if distances[closest] <= 10.0 else None

    def _stage_press(self, event) -> None:
        if event.button == 1:
            self.active_annotation = self._nearest_annotation(event)

    def _annotation_limits(self, index: int) -> tuple[float, float]:
        sample = 1.0 / self.hz
        lower = self.annotation_seconds[index - 1] + sample if index else 0.0
        upper = (
            self.annotation_seconds[index + 1] - sample
            if index + 1 < len(self.annotation_seconds)
            else len(self.analysis["timestamps_s"]) / self.hz
        )
        demos = self.state["demos"]
        if index == 0 and self.selected_demo:
            lower = max(lower, float(demos[self.selected_demo - 1]["end_s"]))
        if index == 5 and self.selected_demo + 1 < len(demos):
            upper = min(upper, float(demos[self.selected_demo + 1]["start_s"]))
        return lower, upper

    def _update_active_artists(self, annotation_index: int) -> None:
        value = self.annotation_seconds[annotation_index]
        for line in self.annotation_lines[annotation_index]:
            line.set_xdata([value, value])
        indices = self._annotation_indices()
        point_index = indices[annotation_index]
        if annotation_index == 5:
            point_index = max(indices[0], point_index - 1)
        task_xyz = np.asarray(self.analysis["task_xyz"], dtype=float)
        point = task_xyz[point_index]
        self.annotation_markers[annotation_index].set_data([point[0]], [point[1]])
        if annotation_index not in (0, 5):
            return
        begin, end = int(indices[0]), int(indices[-1])
        self.active_path_line.set_data(task_xyz[begin:end, 0], task_xyz[begin:end, 1])
        features = np.asarray(self.analysis["features"], dtype=float)
        names = [str(name) for name in self.analysis["feature_names"].tolist()]
        active_time = np.arange(begin, end, dtype=float) / self.hz
        for feature_name, line, span in zip(
            ("table_dist", "bar_lateral_offset", "bar_axial_offset"),
            self.feature_active_lines,
            self.active_spans,
        ):
            line.set_data(active_time, features[begin:end, names.index(feature_name)])
            start, finish = self.annotation_seconds[0], self.annotation_seconds[-1]
            span.set_x(start)
            span.set_width(finish - start)

    def _stage_motion(self, event) -> None:
        if self.active_annotation is None or event.xdata is None:
            return
        if event.inaxes not in tuple(self.axes.flat[1:]):
            return
        index = self.active_annotation
        lower, upper = self._annotation_limits(index)
        value = float(np.clip(round(event.xdata * self.hz) / self.hz, lower, upper))
        self.annotation_seconds[index] = value
        self._update_active_artists(index)
        self._set_message(
            f"Demo {self.selected_demo} {self.ANNOTATION_LABELS[index]} = {value:.3f} s recording time"
        )

    def _stage_release(self, _event) -> None:
        if self.active_annotation is None:
            return
        self.active_annotation = None
        try:
            self.state = self.store.update_demo_annotations(
                self.selected_demo, self.annotation_seconds
            )
            self._set_message("Annotation saved; confirmations reset.")
        except ValueError as error:
            self._set_message(str(error))
            self.state = self.store.state()
        self._redraw()

    def _confirm_all(self, _event) -> None:
        try:
            self.state = self.store.confirm_demo_boundaries()
            self.state = self.store.confirm_cutpoints()
            self._set_message("All demo boundaries and cutpoints confirmed.")
            self._redraw()
        except ValueError as error:
            self._set_message(str(error))

    def _export(self, _event) -> None:
        try:
            output_hz = float(self.output_hz_box.text)
            output_name = str(self.output_name_box.text).strip()
            activate = bool(self.activate_check.get_status()[0])
            self.state = self.store.export(output_hz, output_name, activate)
            self._set_message(
                f"Exported and loader-validated: {self.state['export']['training_archive']}"
            )
            self._redraw()
        except (ValueError, OSError) as error:
            self._set_message(str(error))

    def show(self) -> None:
        print("Matplotlib review opened. Drag Start, CP1–CP4, and End directly.")
        self.plt.show()


def _add_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("bag", type=Path)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--analysis-hz", type=float, default=20.0)
    parser.add_argument("--stationary-threshold", type=float, default=0.10)
    parser.add_argument("--minimum-dwell", type=float, default=3.0)
    parser.add_argument("--stationary-bridge", type=float, default=0.35)
    parser.add_argument("--minimum-demo-duration", type=float, default=15.0)
    parser.add_argument("--endpoint-tolerance", type=float, default=0.065)
    parser.add_argument("--north-start-margin", type=float, default=0.0)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--overwrite", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BarClean rosbag pipeline with mandatory visual review gates."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Create automatic review proposals.")
    _add_prepare_arguments(prepare_parser)
    run_parser = subparsers.add_parser(
        "run", help="Prepare proposals and open the interactive Matplotlib reviewer."
    )
    _add_prepare_arguments(run_parser)
    review_parser = subparsers.add_parser(
        "review", help="Open an existing interactive Matplotlib review."
    )
    review_parser.add_argument("dataset_dir", type=Path)
    export_parser = subparsers.add_parser("export", help="Export an already confirmed review.")
    export_parser.add_argument("dataset_dir", type=Path)
    export_parser.add_argument("--output-hz", type=float, default=DEFAULT_OUTPUT_HZ)
    export_parser.add_argument("--output-name", default=DEFAULT_TRAINING_ARCHIVE)
    export_parser.add_argument("--activate", action="store_true")
    merge_parser = subparsers.add_parser(
        "merge", help="Merge selected demos from reviewed training archives."
    )
    merge_parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        required=True,
        metavar=("ARCHIVE", "DEMO_IDS"),
        help="Repeat as: --source training_5hz.npz 0,1,2",
    )
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.command in {"prepare", "run"}:
        review = prepare_dataset(
            args.bag,
            args.dataset_dir,
            analysis_hz=args.analysis_hz,
            stationary_threshold_rad_s=args.stationary_threshold,
            minimum_dwell_s=args.minimum_dwell,
            stationary_bridge_s=args.stationary_bridge,
            minimum_demo_duration_s=args.minimum_demo_duration,
            endpoint_tolerance_m=args.endpoint_tolerance,
            north_start_margin_m=args.north_start_margin,
            scene_config_path=args.scene_config,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "dataset_directory": str(args.dataset_dir.resolve()),
                    "automatic_demo_count": len(review["demos"]),
                    "review_required": True,
                },
                indent=2,
            )
        )
        if args.command == "prepare":
            return
        MatplotlibReviewApp(ReviewStore(args.dataset_dir)).show()
    elif args.command == "review":
        MatplotlibReviewApp(ReviewStore(args.dataset_dir)).show()
    elif args.command == "export":
        result = export_reviewed_dataset(
            args.dataset_dir,
            output_hz=args.output_hz,
            output_name=args.output_name,
            activate=args.activate,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "merge":
        source_specs = []
        for source_path, demo_ids_text in args.source:
            demo_ids = [
                int(value.strip())
                for value in str(demo_ids_text).split(",")
                if value.strip()
            ]
            source_specs.append((Path(source_path), demo_ids))
        result = merge_processed_datasets(
            source_specs,
            args.output,
            overwrite=args.overwrite,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
