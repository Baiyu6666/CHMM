from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ..base import TaskBundle
from .config import (
    S5SyntheticPreset,
    S5_SYNTHETIC_V23,
    active_s5_env_kwargs,
    cache_compatible_s5_loader_config,
)
from .constants import S5_DEMO_CACHE_VERSION, S5_FEATURE_EXTRACTOR_VERSION
from .execution import _make_stage_labels

_S5_DEMO_CACHE_VERSION = S5_DEMO_CACHE_VERSION
_S5_FEATURE_EXTRACTOR_VERSION = S5_FEATURE_EXTRACTOR_VERSION

def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _s5_standalone_manifest(*, task_name: str, env: Any, n_demos: int) -> dict:
    return {
        "format_name": "LearnStageConstraint.S5Demonstrations",
        "format_version": 2,
        "cache_version": int(_S5_DEMO_CACHE_VERSION),
        "generator_preset": {
            "name": str(getattr(env, "preset_name", "s5_synthetic_v23")),
            "version": int(getattr(env, "preset_version", 22)),
        },
        "task_name": str(task_name),
        "num_demos": int(n_demos),
        "storage": {
            "container": "NumPy NPZ",
            "requires_pickle": False,
            "manifest_key": "dataset_manifest_json",
            "demo_key_template": "{field}_{demo_index:03d}",
        },
        "units": {
            "position": "m",
            "time": "s",
            "angles": "rad",
            "linear_speed": "m/s",
            "angular_speed": "rad/s",
        },
        "sampling": {
            "dt": float(env.dt),
            "timestamps": "Stored explicitly for every demonstration.",
            "reference_time_parameterization": "fixed_step_path_time_parameterization",
        },
        "coordinate_frame": {
            "name": "s5_task_frame",
            "sphere_center": np.asarray(env.sphere_center, dtype=float).tolist(),
            "sphere_radius": float(env.sphere_radius),
            "tool_axis_semantics": "Unit vector in the same frame as position; full roll is available only when quaternion is present.",
        },
        "stage_semantics": [
            "approach",
            "surface_trace",
            "surface_to_shell_transition",
            "shell_inspection",
            "departure",
        ],
        "required_demo_fields": [
            "position",
            "tool_axis",
            "goal_position",
            "timestamps",
            "features",
            "stage_labels",
            "cutpoints",
            "reference_position",
            "reference_tool_axis",
        ],
        "optional_demo_fields": [
            "quaternion",
            "linear_velocity",
            "angular_velocity",
            "contact_flags",
            "joint_positions",
            "joint_velocities",
            "joint_position_commands",
            "joint_position_commands_nominal",
            "execution_joint_noise",
        ],
        "feature_schema": env.get_feature_schema(),
        "feature_extractor": {
            "name": "S5SphereInspect.compute_all_features_matrix",
            "version": int(_S5_FEATURE_EXTRACTOR_VERSION),
            "materialized": True,
            "training_policy": "Use stored features by default; recomputation is explicit migration or verification only.",
        },
        "true_constraints": env.get_true_constraints(),
        "constraint_specs": env.get_constraint_specs(),
        "goal_dist": {
            "mode": str(env.goal_dist_mode),
            "semantics": (
                "Euclidean distance to the stored goal_position assigned to each demonstration."
                if env.goal_dist_mode == "demo_goal"
                else "Euclidean distance to one shared nominal inspection point."
            ),
        },
        "goal_position": {
            "semantics": "Per-demo assigned final reference target; it is distinct from the realized final sample.",
            "unit": "m",
            "frame": "s5_task_frame",
        },
        "nominal_goal": {
            "point": np.asarray(env.goal, dtype=float).tolist(),
            "semantics": (
                "Fixed shared anchor used by goal_dist."
                if env.goal_dist_mode == "nominal_shared"
                else "Task reference point retained as metadata; goal_dist uses each demonstration's stored goal_position."
            ),
        },
        "per_demo_metadata_key": "demo_metadata_json",
    }


def _s5_demo_cache_path(
    *,
    task_name: str,
    n_seed: int,
    env_cfg: dict,
    run_kwargs: dict,
    preset: S5SyntheticPreset | None,
    cache_dir=None,
    cache_version: int = _S5_DEMO_CACHE_VERSION,
) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parents[1] / "demo_cache"
    cache_env_cfg = dict(env_cfg)
    if int(cache_version) < 20:
        cache_env_cfg.pop("goal_dist_mode", None)
    payload = {
        "task_name": str(task_name),
        "seed": int(n_seed),
        "env_cfg": _jsonable(cache_env_cfg),
        "run_kwargs": _jsonable(run_kwargs),
        "cache_version": int(cache_version),
    }
    if preset is not None:
        payload["generator_preset"] = {
            "name": str(preset.name),
            "version": int(preset.version),
        }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return root / str(task_name) / f"seed_{int(n_seed)}_{digest}.npz"


def _make_s5_bundle_from_arrays(
    *,
    task_name: str,
    seed: int,
    env: Any,
    demos: list[np.ndarray],
    true_cutpoints: list[np.ndarray],
    scene_specs: list[dict],
    demo_metadata: list[dict],
    tool_axis_traces: list[np.ndarray | None],
    reference_trajectories: list[np.ndarray | None],
    reference_tool_axis_traces: list[np.ndarray | None],
    demo_records: list[dict[str, np.ndarray]],
    standalone_manifest: dict,
    cache_path: Path | None,
    cache_hit: bool,
    cache_version: int | None = None,
    metadata_complete: bool = True,
    features_materialized: bool = False,
) -> TaskBundle:
    feature_traces = [np.asarray(record["features"], dtype=float) for record in demo_records]
    goal_positions = [np.asarray(record["goal_position"], dtype=float) for record in demo_records]
    return TaskBundle(
        name=task_name,
        demos=demos,
        features=feature_traces,
        env=env,
        true_taus=[None for _ in demos],
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=dict(env.true_constraints),
        constraint_specs=env.get_constraint_specs(),
        meta={
            "seed": int(seed),
            "task_name": task_name,
            "scene_specs": scene_specs,
            "demo_metadata": demo_metadata,
            "tool_axis_traces": tool_axis_traces,
            "reference_trajectories": reference_trajectories,
            "reference_tool_axis_traces": reference_tool_axis_traces,
            "demo_records": demo_records,
            "goal_positions": goal_positions,
            "feature_dataset": {
                "source": "materialized_from_raw" if features_materialized else "stored",
                "extractor_version": int(_S5_FEATURE_EXTRACTOR_VERSION),
                "training_policy": "stored",
            },
            "standalone_manifest": standalone_manifest,
            "observation_specs": env.get_observation_spec(),
            "render_camera_presets": env.get_render_camera_presets(),
            "asset_handles": env.get_asset_handles(),
            "demo_cache": None
            if cache_path is None
            else {
                "path": str(cache_path),
                "hit": bool(cache_hit),
                "version": None if cache_version is None else int(cache_version),
                "metadata_complete": bool(metadata_complete),
            },
        },
    )


def _try_load_s5_demo_cache(*, cache_path: Path, task_name: str, n_demos: int, seed: int, env: Any):
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            cache_metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
            cache_version = int(cache_metadata.get("cache_version", 0))
            has_standalone_manifest = "dataset_manifest_json" in data
            stored_standalone_manifest = (
                json.loads(str(data["dataset_manifest_json"].item()))
                if has_standalone_manifest
                else _s5_standalone_manifest(task_name=task_name, env=env, n_demos=int(n_demos))
            )
            stored_preset = dict(stored_standalone_manifest.get("generator_preset", {}))
            expected_preset = {
                "name": str(getattr(env, "preset_name", "")),
                "version": int(getattr(env, "preset_version", -1)),
            }
            if has_standalone_manifest and stored_preset != expected_preset:
                return None
            if cache_version >= 21 and not has_standalone_manifest:
                return None
            count = int(data["count"])
            if count < int(n_demos):
                return None
            demos = [np.asarray(data[f"demo_{i}"], dtype=float) for i in range(int(n_demos))]
            cutpoints = [np.asarray(data[f"cutpoints_{i}"], dtype=int) for i in range(int(n_demos))]
            tool_axes = []
            reference_trajectories = []
            reference_tool_axes = []
            for i in range(int(n_demos)):
                key = f"tool_axis_{i}"
                tool_axes.append(None if key not in data else np.asarray(data[key], dtype=float))
                reference_key = f"reference_demo_{i}"
                reference_axis_key = f"reference_tool_axis_{i}"
                reference_trajectories.append(
                    None if reference_key not in data else np.asarray(data[reference_key], dtype=float)
                )
                reference_tool_axes.append(
                    None if reference_axis_key not in data else np.asarray(data[reference_axis_key], dtype=float)
                )
            scene_specs = json.loads(str(data["scene_specs_json"].item()))[: int(n_demos)]
            if "demo_metadata_json" in data:
                demo_metadata = json.loads(str(data["demo_metadata_json"].item()))[: int(n_demos)]
            else:
                demo_metadata = [{} for _ in range(int(n_demos))]
            demo_records = []
            features_materialized = False
            stored_extractor = dict(stored_standalone_manifest.get("feature_extractor", {}))
            stored_goal_dist = dict(stored_standalone_manifest.get("goal_dist", {}))
            stored_features_are_current = bool(
                int(stored_extractor.get("version", 0)) == int(_S5_FEATURE_EXTRACTOR_VERSION)
                and str(stored_goal_dist.get("mode", "")) == str(env.goal_dist_mode)
            )
            standalone_fields = list(stored_standalone_manifest.get("required_demo_fields", [])) + list(
                stored_standalone_manifest.get("optional_demo_fields", [])
            )
            for i in range(int(n_demos)):
                record = {}
                for field in standalone_fields:
                    key = f"{field}_{i:03d}"
                    if key in data:
                        record[str(field)] = np.asarray(data[key])
                position = np.asarray(demos[i], dtype=float)
                tool_axis = tool_axes[i]
                reference_position = reference_trajectories[i]
                reference_axis = reference_tool_axes[i]
                record["position"] = position
                record.setdefault("timestamps", np.arange(len(position), dtype=float) * float(env.dt))
                record.setdefault("stage_labels", _make_stage_labels(cutpoints[i], len(position)))
                record["cutpoints"] = np.asarray(cutpoints[i], dtype=int)
                if tool_axis is not None:
                    record["tool_axis"] = np.asarray(tool_axis, dtype=float)
                if reference_position is not None:
                    record["reference_position"] = np.asarray(reference_position, dtype=float)
                if reference_axis is not None:
                    record["reference_tool_axis"] = np.asarray(reference_axis, dtype=float)

                goal_position = record.get("goal_position")
                if goal_position is None:
                    if reference_position is not None and len(reference_position) > 0:
                        goal_position = np.asarray(reference_position, dtype=float)[-1]
                    else:
                        goal_position = position[-1]
                goal_position = np.asarray(goal_position, dtype=float).reshape(3)
                record["goal_position"] = goal_position

                stored_features = record.get("features")
                stored_shape_is_valid = bool(
                    stored_features is not None
                    and np.asarray(stored_features).ndim == 2
                    and len(np.asarray(stored_features)) == len(position)
                    and np.asarray(stored_features).shape[1] == len(env.get_feature_schema())
                )
                if not (stored_features_are_current and stored_shape_is_valid):
                    stored_features = env.compute_all_features_matrix(
                        position,
                        tool_axis=tool_axis,
                        goal_position=goal_position,
                        use_cached=False,
                    )
                    features_materialized = True
                record["features"] = np.asarray(stored_features, dtype=float)
                demo_records.append(record)
    except Exception:
        return None

    print(
        f"\033[31m[S5 demo cache] loaded {int(n_demos)}/{int(count)} demos from {cache_path}\033[0m",
        flush=True,
    )
    for traj, axis, record in zip(demos, tool_axes, demo_records):
        if axis is not None:
            env.register_tool_axis_trace(traj, axis)
        env.register_goal_position(traj, record["goal_position"])
        env.register_feature_trace(traj, record["features"])
    standalone_manifest = _s5_standalone_manifest(task_name=task_name, env=env, n_demos=int(n_demos))
    return _make_s5_bundle_from_arrays(
        task_name=task_name,
        seed=seed,
        env=env,
        demos=demos,
        true_cutpoints=cutpoints,
        scene_specs=scene_specs,
        demo_metadata=demo_metadata,
        tool_axis_traces=tool_axes,
        reference_trajectories=reference_trajectories,
        reference_tool_axis_traces=reference_tool_axes,
        demo_records=demo_records,
        standalone_manifest=standalone_manifest,
        cache_path=cache_path,
        cache_hit=True,
        cache_version=cache_version,
        metadata_complete=bool(
            all(bool(item) for item in demo_metadata)
            and bool(has_standalone_manifest)
            and all(value is not None for value in reference_trajectories)
            and all(value is not None for value in reference_tool_axes)
        ),
        features_materialized=features_materialized,
    )


def _save_s5_demo_cache(*, cache_path: Path, bundle: TaskBundle, tool_axis_traces: list, env_cfg: dict, run_kwargs: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "count": np.asarray(len(bundle.demos), dtype=np.int64),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "task_name": bundle.name,
                    "seed": bundle.meta.get("seed"),
                    "env_cfg": _jsonable(env_cfg),
                    "run_kwargs": _jsonable(run_kwargs),
                    "cache_version": _S5_DEMO_CACHE_VERSION,
                },
                sort_keys=True,
            )
        ),
        "scene_specs_json": np.asarray(json.dumps(_jsonable(bundle.meta.get("scene_specs", [])), sort_keys=True)),
        "demo_metadata_json": np.asarray(json.dumps(_jsonable(bundle.meta.get("demo_metadata", [])), sort_keys=True)),
        "dataset_manifest_json": np.asarray(
            json.dumps(_jsonable(bundle.meta.get("standalone_manifest", {})), sort_keys=True)
        ),
    }
    reference_trajectories = list(bundle.meta.get("reference_trajectories", []))
    reference_tool_axes = list(bundle.meta.get("reference_tool_axis_traces", []))
    demo_records = list(bundle.meta.get("demo_records", []))
    for i, demo in enumerate(bundle.demos):
        arrays[f"demo_{i}"] = np.asarray(demo, dtype=float)
        arrays[f"cutpoints_{i}"] = np.asarray(bundle.true_cutpoints[i], dtype=int)
        if i < len(tool_axis_traces) and tool_axis_traces[i] is not None:
            arrays[f"tool_axis_{i}"] = np.asarray(tool_axis_traces[i], dtype=float)
        if i < len(reference_trajectories) and reference_trajectories[i] is not None:
            arrays[f"reference_demo_{i}"] = np.asarray(reference_trajectories[i], dtype=float)
        if i < len(reference_tool_axes) and reference_tool_axes[i] is not None:
            arrays[f"reference_tool_axis_{i}"] = np.asarray(reference_tool_axes[i], dtype=float)
        if i < len(demo_records):
            for field, values in dict(demo_records[i]).items():
                if values is not None:
                    arrays[f"{str(field)}_{i:03d}"] = np.asarray(values)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    np.savez_compressed(tmp_path, **arrays)
    written = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    os.replace(written, cache_path)


def _build_sphere_inspect_bundle(
    *,
    task_name: str,
    n_demos: int,
    seed: int,
    env_kwargs=None,
    demo_kwargs=None,
    preset: S5SyntheticPreset = S5_SYNTHETIC_V23,
    **extra_env_kwargs,
) -> TaskBundle:
    env_cfg = dict(env_kwargs or {})
    env_cfg.update(extra_env_kwargs)
    cache_demos = bool(env_cfg.pop("cache_demos", False))
    cache_dir = env_cfg.pop("demo_cache_dir", None)
    run_kwargs = dict(demo_kwargs or {})
    from .task import S5SphereInspectEnv

    env = S5SphereInspectEnv.from_preset(preset, **active_s5_env_kwargs(env_cfg))
    cache_path = None
    if cache_demos:
        cache_path = _s5_demo_cache_path(
            task_name=task_name,
            n_seed=int(seed),
            env_cfg=env_cfg,
            run_kwargs=run_kwargs,
            preset=preset,
            cache_dir=cache_dir,
        )
        cache_candidates = [cache_path]
        preset_unaware_cache_path = _s5_demo_cache_path(
            task_name=task_name,
            n_seed=int(seed),
            env_cfg=env_cfg,
            run_kwargs=run_kwargs,
            preset=None,
            cache_dir=cache_dir,
        )
        if preset_unaware_cache_path not in cache_candidates:
            cache_candidates.append(preset_unaware_cache_path)
        if int(_S5_DEMO_CACHE_VERSION) <= 20:
            version19_cache_path = _s5_demo_cache_path(
                task_name=task_name,
                n_seed=int(seed),
                env_cfg=env_cfg,
                run_kwargs=run_kwargs,
                preset=None,
                cache_dir=cache_dir,
                cache_version=19,
            )
            if version19_cache_path not in cache_candidates:
                cache_candidates.append(version19_cache_path)
            legacy_env_cfg = dict(env_cfg)
            legacy_env_cfg.setdefault("transition_stage_fraction", 0.40)
            legacy_env_cfg.setdefault("repos_angle_range", (0.95, 1.18))
            legacy_cache_path = _s5_demo_cache_path(
                task_name=task_name,
                n_seed=int(seed),
                env_cfg=legacy_env_cfg,
                run_kwargs=run_kwargs,
                preset=None,
                cache_dir=cache_dir,
                cache_version=17,
            )
            if legacy_cache_path != cache_path:
                cache_candidates.append(legacy_cache_path)
        for candidate_path in cache_candidates:
            cached_bundle = _try_load_s5_demo_cache(
                cache_path=candidate_path,
                task_name=task_name,
                n_demos=int(n_demos),
                seed=int(seed),
                env=env,
            )
            if cached_bundle is not None:
                feature_dataset = dict(cached_bundle.meta.get("feature_dataset", {}))
                if candidate_path != cache_path or feature_dataset.get("source") != "stored":
                    _save_s5_demo_cache(
                        cache_path=cache_path,
                        bundle=cached_bundle,
                        tool_axis_traces=list(cached_bundle.meta.get("tool_axis_traces", [])),
                        env_cfg=env_cfg,
                        run_kwargs=run_kwargs,
                    )
                    cached_bundle.meta["demo_cache"] = {
                        "path": str(cache_path),
                        "hit": True,
                        "version": int(_S5_DEMO_CACHE_VERSION),
                        "metadata_complete": bool(
                            dict(cached_bundle.meta.get("demo_cache", {})).get("metadata_complete", False)
                        ),
                        "migrated_from": str(candidate_path),
                    }
                    cached_bundle.meta["feature_dataset"] = {
                        "source": "stored",
                        "extractor_version": int(_S5_FEATURE_EXTRACTOR_VERSION),
                        "training_policy": "stored",
                    }
                    print(
                        f"\033[31m[S5 demo cache] migrated dataset to v{int(_S5_DEMO_CACHE_VERSION)} "
                        f"at {cache_path}\033[0m",
                        flush=True,
                    )
                return cached_bundle

    demos = []
    true_cutpoints = []
    scene_specs = []
    demo_metadata = []
    tool_axis_traces = []
    reference_trajectories = []
    reference_tool_axis_traces = []
    goal_positions = []
    demo_records = []
    for demo_idx in range(int(n_demos)):
        scene = env.sample_scene()
        scene["demo_index"] = int(demo_idx)
        latent = env.rollout_demo(scene, seed=env.demo_seed_for_index(seed, demo_idx), **run_kwargs)
        observation = env.compute_observation(latent, scene)
        traj = np.asarray(observation["trajectory"], dtype=float)
        tool_axis = observation.get("tool_axis")
        if tool_axis is not None:
            tool_axis = np.asarray(tool_axis, dtype=float)
            env.register_tool_axis_trace(traj, tool_axis)
        demos.append(traj)
        true_cutpoints.append(np.asarray(observation["true_cutpoints"], dtype=int))
        scene_specs.append(dict(scene))
        demo_metadata.append(_jsonable(observation.get("generation_metadata", {})))
        tool_axis_traces.append(None if tool_axis is None else np.asarray(tool_axis, dtype=float))
        reference_trajectory = observation.get("reference_trajectory", traj)
        reference_tool_axis = observation.get("reference_tool_axis", tool_axis)
        reference_trajectories.append(
            None if reference_trajectory is None else np.asarray(reference_trajectory, dtype=float)
        )
        reference_tool_axis_traces.append(
            None if reference_tool_axis is None else np.asarray(reference_tool_axis, dtype=float)
        )
        goal_position = np.asarray(observation["goal_position"], dtype=float).reshape(3)
        goal_positions.append(goal_position)
        record = {
            "position": np.asarray(traj, dtype=float),
            "goal_position": goal_position,
            "timestamps": np.asarray(observation["timestamps"], dtype=float),
            "features": np.asarray(observation["features"], dtype=float),
            "stage_labels": _make_stage_labels(observation["true_cutpoints"], len(traj)),
            "cutpoints": np.asarray(observation["true_cutpoints"], dtype=int),
            "reference_position": np.asarray(reference_trajectory, dtype=float),
        }
        if tool_axis is not None:
            record["tool_axis"] = np.asarray(tool_axis, dtype=float)
        if reference_tool_axis is not None:
            record["reference_tool_axis"] = np.asarray(reference_tool_axis, dtype=float)
        optional_observation_fields = {
            "quaternions": "quaternion",
            "linear_velocity": "linear_velocity",
            "angular_velocity": "angular_velocity",
            "contact_flags": "contact_flags",
            "joint_positions": "joint_positions",
            "joint_velocities": "joint_velocities",
            "joint_position_commands": "joint_position_commands",
            "joint_position_commands_nominal": "joint_position_commands_nominal",
            "execution_joint_noise": "execution_joint_noise",
        }
        for observation_key, record_key in optional_observation_fields.items():
            if observation.get(observation_key) is not None:
                record[record_key] = np.asarray(observation[observation_key])
        demo_records.append(record)
    true_taus = [None for _ in demos]
    standalone_manifest = _s5_standalone_manifest(
        task_name=task_name,
        env=env,
        n_demos=len(demos),
    )
    bundle = TaskBundle(
        name=task_name,
        demos=demos,
        features=[np.asarray(record["features"], dtype=float) for record in demo_records],
        env=env,
        true_taus=true_taus,
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=dict(env.true_constraints),
        constraint_specs=env.get_constraint_specs(),
        meta={
            "seed": seed,
            "task_name": task_name,
            "scene_specs": scene_specs,
            "demo_metadata": demo_metadata,
            "tool_axis_traces": tool_axis_traces,
            "reference_trajectories": reference_trajectories,
            "reference_tool_axis_traces": reference_tool_axis_traces,
            "goal_positions": goal_positions,
            "demo_records": demo_records,
            "feature_dataset": {
                "source": "stored",
                "extractor_version": int(_S5_FEATURE_EXTRACTOR_VERSION),
                "training_policy": "stored",
            },
            "standalone_manifest": standalone_manifest,
            "observation_specs": env.get_observation_spec(),
            "render_camera_presets": env.get_render_camera_presets(),
            "asset_handles": env.get_asset_handles(),
            "demo_cache": None
            if cache_path is None
            else {
                "path": str(cache_path),
                "hit": False,
                "version": int(_S5_DEMO_CACHE_VERSION),
                "metadata_complete": True,
            },
        },
    )
    if cache_path is not None:
        _save_s5_demo_cache(
            cache_path=cache_path,
            bundle=bundle,
            tool_axis_traces=tool_axis_traces,
            env_cfg=env_cfg,
            run_kwargs=run_kwargs,
        )
    return bundle


def _apply_default_s5_loader_config(env_cfg: dict) -> dict:
    return cache_compatible_s5_loader_config(env_cfg)


def load_S5SphereInspect(
    n_demos: int = 10,
    seed: int = 0,
    env_kwargs=None,
    demo_kwargs=None,
    **extra_env_kwargs,
) -> TaskBundle:
    env_cfg = dict(env_kwargs or {})
    env_cfg.update(extra_env_kwargs)
    env_cfg = _apply_default_s5_loader_config(env_cfg)
    env_cfg.setdefault("rollout_backend", "analytic")
    env_cfg.setdefault("observation_backend", "analytic_raw")
    env_cfg.setdefault("eval_tag", "S5SphereInspect")
    return _build_sphere_inspect_bundle(
        task_name="S5SphereInspect",
        n_demos=n_demos,
        seed=seed,
        env_kwargs=env_cfg,
        demo_kwargs=demo_kwargs,
    )
