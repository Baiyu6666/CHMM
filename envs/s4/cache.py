from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..base import TaskBundle


S4_DEMO_CACHE_VERSION = 3
S4_FEATURE_EXTRACTOR_VERSION = 1
_REQUIRED_FIELDS = (
    "trajectory",
    "features",
    "stage_labels",
    "cutpoints",
    "timestamps",
    "reference_trajectory",
    "planned_normal_force",
    "commanded_normal_force",
    "measured_normal_force",
)
_OPTIONAL_FIELDS = (
    "contact_slider_trajectory",
    "joint_positions",
    "joint_velocities",
    "joint_position_commands",
    "joint_position_commands_nominal",
    "joint_torque_commands",
    "execution_joint_noise",
    "preload_indent",
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def s4_demo_cache_path(
    *,
    task_name: str,
    seed: int,
    env_config: dict,
    rollout_config: dict,
    cache_dir=None,
) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / "demo_cache"
    payload = {
        "task_name": str(task_name),
        "seed": int(seed),
        "env_config": _jsonable(env_config),
        "rollout_config": _jsonable(rollout_config),
        "cache_version": int(S4_DEMO_CACHE_VERSION),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()[:16]
    return root / str(task_name) / f"seed_{int(seed)}_{digest}.npz"


def make_s4_cache_manifest(*, task_name: str, env, count: int, rollout_config: dict) -> dict:
    controller = str(rollout_config.get("execution_control", "position")).strip().lower()
    return {
        "format_name": "LearnStageConstraint.S4Demonstrations",
        "format_version": 1,
        "cache_version": int(S4_DEMO_CACHE_VERSION),
        "task_name": str(task_name),
        "num_demos": int(count),
        "execution": {
            "backend": str(env.rollout_backend),
            "controller": controller,
            "normal_force_semantics": (
                "Measured PyBullet contact force."
                if controller in {"admittance", "torque_preload", "torque"}
                else "Analytic or commanded normal-force trace."
            ),
        },
        "storage": {
            "container": "NumPy NPZ",
            "requires_pickle": False,
            "manifest_key": "dataset_manifest_json",
            "demo_key_template": "{field}_{demo_index:03d}",
        },
        "units": {
            "position": "m",
            "time": "s",
            "angle": "rad",
            "normal_force": "N",
        },
        "sampling": {"dt": float(env.dt), "timestamps": "Stored explicitly per demonstration."},
        "coordinate_frame": {"name": "s4_task_frame", "state": "[x, y, z, theta]"},
        "stage_semantics": ["approach", "contact_alignment", "sliding", "insertion"],
        "required_demo_fields": list(_REQUIRED_FIELDS),
        "optional_demo_fields": list(_OPTIONAL_FIELDS),
        "feature_schema": env.get_feature_schema(),
        "feature_extractor": {
            "name": "S4SlideInsert.compute_all_features_matrix",
            "version": int(S4_FEATURE_EXTRACTOR_VERSION),
            "materialized": True,
            "training_policy": "Use stored features by default.",
        },
        "true_constraints": env.get_true_constraints(),
        "constraint_specs": env.get_constraint_specs(),
    }


def _configure_bundle_env(env, demos, labels, cutpoints) -> None:
    env.demo_subgoals = [
        np.asarray(trajectory[int(points[1]), :4], dtype=float).copy()
        for trajectory, points in zip(demos, cutpoints)
    ]
    env.demo_goals = [np.asarray(trajectory[-1, :4], dtype=float).copy() for trajectory in demos]
    env.demo_stage_lengths = [
        np.bincount(stage_labels, minlength=env.n_segments).astype(int)
        for stage_labels in labels
    ]
    env.subgoal = np.mean(np.stack(env.demo_subgoals, axis=0), axis=0)
    env.goal = np.mean(np.stack(env.demo_goals, axis=0), axis=0)


def make_s4_bundle(
    *,
    task_name: str,
    seed: int,
    env,
    records: list,
    scene_specs: list,
    manifest: dict,
    cache_path: Path = None,
    cache_hit: bool = False,
) -> TaskBundle:
    demos = [np.asarray(record["trajectory"], dtype=float) for record in records]
    features = [np.asarray(record["features"], dtype=float) for record in records]
    labels = [np.asarray(record["stage_labels"], dtype=int) for record in records]
    cutpoints = [np.asarray(record["cutpoints"], dtype=int) for record in records]
    for trajectory, record in zip(demos, records):
        env.register_normal_load_trace(trajectory, np.asarray(record["measured_normal_force"], dtype=float))
    _configure_bundle_env(env, demos, labels, cutpoints)
    return TaskBundle(
        name=str(task_name),
        demos=demos,
        features=features,
        env=env,
        true_taus=None,
        true_cutpoints=cutpoints,
        true_labels=labels,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(),
        constraint_specs=env.get_constraint_specs(),
        meta={
            "seed": int(seed),
            "cutpoints": [points.tolist() for points in cutpoints],
            "task_name": str(task_name),
            "scene_specs": list(scene_specs),
            "demo_records": records,
            "reference_trajectories": [np.asarray(record["reference_trajectory"], dtype=float) for record in records],
            "planned_normal_force_traces": [np.asarray(record["planned_normal_force"], dtype=float) for record in records],
            "commanded_normal_force_traces": [np.asarray(record["commanded_normal_force"], dtype=float) for record in records],
            "measured_normal_force_traces": [np.asarray(record["measured_normal_force"], dtype=float) for record in records],
            "feature_dataset": {
                "source": "stored",
                "extractor_version": int(S4_FEATURE_EXTRACTOR_VERSION),
                "training_policy": "stored",
            },
            "standalone_manifest": dict(manifest),
            "observation_specs": env.get_observation_spec(),
            "render_camera_presets": env.get_render_camera_presets(),
            "asset_handles": env.get_asset_handles(),
            "demo_cache": None
            if cache_path is None
            else {
                "path": str(cache_path),
                "hit": bool(cache_hit),
                "version": int(S4_DEMO_CACHE_VERSION),
            },
        },
    )


def save_s4_demo_cache(
    *,
    cache_path: Path,
    bundle: TaskBundle,
    env_config: dict,
    rollout_config: dict,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "count": np.asarray(len(bundle.demos), dtype=np.int64),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "task_name": bundle.name,
                    "seed": bundle.meta.get("seed"),
                    "env_config": _jsonable(env_config),
                    "rollout_config": _jsonable(rollout_config),
                    "cache_version": int(S4_DEMO_CACHE_VERSION),
                },
                sort_keys=True,
            )
        ),
        "scene_specs_json": np.asarray(json.dumps(_jsonable(bundle.meta.get("scene_specs", [])), sort_keys=True)),
        "dataset_manifest_json": np.asarray(
            json.dumps(_jsonable(bundle.meta.get("standalone_manifest", {})), sort_keys=True)
        ),
    }
    for demo_index, record in enumerate(bundle.meta.get("demo_records", [])):
        for field in (*_REQUIRED_FIELDS, *_OPTIONAL_FIELDS):
            if field in record and record[field] is not None:
                arrays[f"{field}_{demo_index:03d}"] = np.asarray(record[field])
    temporary_path = cache_path.with_name(cache_path.name + ".tmp")
    np.savez_compressed(temporary_path, **arrays)
    written_path = temporary_path if temporary_path.exists() else temporary_path.with_suffix(temporary_path.suffix + ".npz")
    os.replace(str(written_path), str(cache_path))


def load_s4_demo_cache(
    *,
    cache_path: Path,
    task_name: str,
    n_demos: int,
    seed: int,
    env,
    rollout_config: dict,
):
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            manifest = json.loads(str(data["dataset_manifest_json"].item()))
            if str(manifest.get("format_name")) != "LearnStageConstraint.S4Demonstrations":
                return None
            if str(manifest.get("task_name")) != str(task_name):
                return None
            if int(manifest.get("cache_version", -1)) != int(S4_DEMO_CACHE_VERSION):
                return None
            if int(manifest.get("feature_extractor", {}).get("version", -1)) != int(S4_FEATURE_EXTRACTOR_VERSION):
                return None
            expected_controller = str(rollout_config.get("execution_control", "position")).strip().lower()
            if str(manifest.get("execution", {}).get("controller", "")).strip().lower() != expected_controller:
                return None
            count = int(np.asarray(data["count"]).item())
            if count < int(n_demos):
                return None
            records = []
            for demo_index in range(int(n_demos)):
                record = {}
                for field in (*_REQUIRED_FIELDS, *_OPTIONAL_FIELDS):
                    key = f"{field}_{demo_index:03d}"
                    if key in data:
                        record[field] = np.asarray(data[key])
                if any(field not in record for field in _REQUIRED_FIELDS):
                    return None
                trajectory = np.asarray(record["trajectory"])
                features = np.asarray(record["features"])
                if trajectory.ndim != 2 or trajectory.shape[1] < 4:
                    return None
                if features.shape != (len(trajectory), len(env.get_feature_schema())):
                    return None
                if any(len(np.asarray(record[field])) != len(trajectory) for field in ("stage_labels", "timestamps", "reference_trajectory", "planned_normal_force", "commanded_normal_force", "measured_normal_force")):
                    return None
                records.append(record)
            scene_specs = json.loads(str(data["scene_specs_json"].item()))[: int(n_demos)]
    except Exception:
        return None
    print(f"\033[31m[S4 demo cache] loaded {int(n_demos)}/{int(count)} demos from {cache_path}\033[0m", flush=True)
    return make_s4_bundle(
        task_name=task_name,
        seed=seed,
        env=env,
        records=records,
        scene_specs=scene_specs,
        manifest=manifest,
        cache_path=cache_path,
        cache_hit=True,
    )
