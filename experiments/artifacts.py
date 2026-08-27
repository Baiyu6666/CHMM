from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from methods import JOINT_METHODS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs"

_CONSTRAINT_KEYS = (
    "ConstraintFeatureNames",
    "ConstraintTrueActiveMask",
    "ConstraintPredictedActiveMask",
    "ConstraintTargetMatrix",
    "ConstraintLearnedSemanticsMatrix",
    "ConstraintLearnedValueMatrix",
    "ConstraintLearnedRawValueMatrix",
    "ConstraintLearnedValuePerDemo",
    "ParameterErrorMatrix",
    "ParameterErrorMatrixRaw",
    "ConstraintErrorMatrix",
    "ConstraintErrorMatrixRaw",
    "ConstraintSemanticsMatrix",
    "ConstraintFeatureScales",
    "ConstraintLearnedDemoCount",
)

_MAP_METHODS = frozenset(
    {"map", "map_pooled", "map_balanced_pooled", "map_balanced_vote"}
)
_LEARNED_MODES = frozenset(
    {"inactive", "target_value", "lower_bound", "upper_bound"}
)


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, set):
        return sorted(_jsonify(v) for v in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonify(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def dataset_fingerprint(dataset: Any) -> str:
    digest = hashlib.sha256()
    demos = list(getattr(dataset, "demos", []))
    cutpoints = list(getattr(dataset, "true_cutpoints", []))
    digest.update(np.asarray([len(demos), len(cutpoints)], dtype=np.int64).tobytes())
    for values in [*demos, *cutpoints]:
        array = np.ascontiguousarray(np.asarray(values))
        digest.update(str(array.dtype).encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def resolve_run_dir(
    method_name: str,
    dataset_name: str,
    dataset_seed: int,
    method_seed: int,
    output_root: str | Path = DEFAULT_RUN_ROOT,
) -> Path:
    root = Path(output_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    return (
        root
        / method_name
        / dataset_name
        / f"method_seed_{int(method_seed):03d}"
    )


def resolve_plot_dir(run_dir: str | Path) -> Path:
    return Path(run_dir)


def default_method_seed(method_name: str, method_kwargs: Mapping[str, Any]) -> int:
    if method_name in JOINT_METHODS or method_name == "fchmm":
        return int(method_kwargs.get("seed", 0))
    segmenter_cfg = dict(method_kwargs.get("segmenter", {}))
    return int(segmenter_cfg.get("seed", 0))


def apply_run_plot_dirs(
    method_name: str,
    method_kwargs: Mapping[str, Any],
    plot_dir: str | Path,
) -> dict[str, Any]:
    plot_dir_str = str(plot_dir)
    cfg = dict(method_kwargs)
    if method_name in JOINT_METHODS or method_name == "fchmm":
        cfg["plot_dir"] = plot_dir_str
        return cfg
    segmenter_cfg = dict(cfg.get("segmenter", {}))
    constraint_cfg = dict(cfg.get("posthoc_constraint", {}))
    segmenter_cfg["plot_dir"] = plot_dir_str
    constraint_cfg["plot_dir"] = plot_dir_str
    cfg["segmenter"] = segmenter_cfg
    cfg["posthoc_constraint"] = constraint_cfg
    return cfg


def _extract_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    if "joint_result" in result:
        return dict(result["joint_result"].get("metrics", {}))
    constraints = result.get("constraints", {})
    return dict(constraints.get("metrics", {}))


def _extract_segmentation(result: Mapping[str, Any]) -> dict[str, Any]:
    dataset = result.get("dataset")
    payload: dict[str, Any] = {
        "demo_lengths": [int(len(X)) for X in getattr(dataset, "demos", [])],
        "true_taus": getattr(dataset, "true_taus", None),
        "true_cutpoints": getattr(dataset, "true_cutpoints", None),
    }

    if "joint_result" in result:
        joint_result = dict(result["joint_result"])
        payload["predicted_taus"] = joint_result.get("taus_hat")
        payload["predicted_cutpoints"] = joint_result.get("cutpoints_hat")
        payload["predicted_stage_ends"] = joint_result.get("stage_ends_hat")
        return payload

    segmentation = result.get("segmentation")
    if segmentation is None:
        return payload

    payload["predicted_taus"] = getattr(segmentation, "taus", None)
    payload["predicted_cutpoints"] = getattr(segmentation, "cutpoints", None)
    model = getattr(segmentation, "model", None)
    stage_ends = getattr(model, "stage_ends_", None)
    if stage_ends is not None:
        payload["predicted_stage_ends"] = stage_ends
    return payload


def _extract_constraints(result: Mapping[str, Any]) -> dict[str, Any]:
    dataset = result.get("dataset")
    metrics = _extract_metrics(result)
    payload = {
        "true_constraints": getattr(dataset, "true_constraints", None),
        "constraint_specs": getattr(dataset, "constraint_specs", None),
    }
    for key in _CONSTRAINT_KEYS:
        if key in metrics:
            payload[key] = metrics[key]
    return payload


def _finite_or_none(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def _map_mode_scores(result: Mapping[str, Any], stage: int, feature: int) -> dict[str, float]:
    model = result.get("joint_result", {}).get("model")
    costs = getattr(model, "map_shared_mode_costs_", None)
    try:
        raw = dict(costs[int(stage)][int(feature)])
    except (IndexError, TypeError, ValueError):
        return {}
    aliases = {
        "inactive": "inactive",
        "eq": "target_value",
        "lb": "lower_bound",
        "ub": "upper_bound",
    }
    normalized_costs = {
        aliases[str(name)]: float(value)
        for name, value in raw.items()
        if str(name) in aliases and _finite_or_none(value) is not None
    }
    if not normalized_costs:
        return {}
    minimum = min(normalized_costs.values())
    weights = {
        name: float(np.exp(-min(max(cost - minimum, 0.0), 700.0)))
        for name, cost in normalized_costs.items()
    }
    total = sum(weights.values())
    return {name: value / total for name, value in weights.items()}


def _quaternion_to_matrix(quaternion: Any) -> np.ndarray:
    values = np.asarray(quaternion, dtype=float).reshape(4)
    norm = float(np.linalg.norm(values))
    if not np.all(np.isfinite(values)) or norm <= 1e-12:
        raise ValueError("Endpoint quaternion must be finite and nonzero")
    x, y, z, w = values / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _matrix_to_quaternion(matrix: Any) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=float).reshape(3, 3)
    eigenvalues, eigenvectors = np.linalg.eigh(
        np.asarray(
            [
                [
                    rotation[0, 0] - rotation[1, 1] - rotation[2, 2],
                    rotation[1, 0] + rotation[0, 1],
                    rotation[2, 0] + rotation[0, 2],
                    rotation[1, 2] - rotation[2, 1],
                ],
                [
                    rotation[1, 0] + rotation[0, 1],
                    rotation[1, 1] - rotation[0, 0] - rotation[2, 2],
                    rotation[2, 1] + rotation[1, 2],
                    rotation[2, 0] - rotation[0, 2],
                ],
                [
                    rotation[2, 0] + rotation[0, 2],
                    rotation[2, 1] + rotation[1, 2],
                    rotation[2, 2] - rotation[0, 0] - rotation[1, 1],
                    rotation[0, 1] - rotation[1, 0],
                ],
                [
                    rotation[1, 2] - rotation[2, 1],
                    rotation[2, 0] - rotation[0, 2],
                    rotation[0, 1] - rotation[1, 0],
                    rotation[0, 0] + rotation[1, 1] + rotation[2, 2],
                ],
            ],
            dtype=float,
        )
        / 3.0
    )
    quaternion = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[3] >= 0.0 else -quaternion


def _mean_quaternion(quaternions: Any) -> np.ndarray:
    values = np.asarray(quaternions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4 or not len(values):
        raise ValueError("Endpoint quaternion aggregation requires an N x 4 matrix")
    norms = np.linalg.norm(values, axis=1)
    if not np.all(np.isfinite(values)) or np.any(norms <= 1e-12):
        raise ValueError("Endpoint quaternions must be finite and nonzero")
    unit = values / norms[:, None]
    eigenvalues, eigenvectors = np.linalg.eigh(unit.T @ unit)
    mean = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
    mean /= np.linalg.norm(mean)
    return mean if mean[3] >= 0.0 else -mean


def _aggregate_learned_endpoint_poses(result: Mapping[str, Any]) -> tuple[list[list[float]], dict[str, Any]]:
    dataset = result.get("dataset")
    demos = list(getattr(dataset, "demos", []))
    env = getattr(dataset, "env", None)
    cutpoints = result.get("joint_result", {}).get("cutpoints_hat")
    if cutpoints is None:
        segmentation = result.get("segmentation")
        cutpoints = getattr(segmentation, "cutpoints", None)
    if not demos or cutpoints is None or len(cutpoints) != len(demos):
        raise ValueError("Learned endpoint export requires one segmentation per demo")
    required_attributes = (
        "table_surface_point",
        "table_normal",
        "_bar_geometry_trace",
        "get_demo_scene",
    )
    if env is None or any(not hasattr(env, name) for name in required_attributes):
        raise ValueError("Learned endpoint export requires a task-frame-aware environment")

    n_endpoints = len(np.asarray(cutpoints[0], dtype=int).reshape(-1))
    if n_endpoints < 1:
        raise ValueError("Learned endpoint export requires at least one stage boundary")
    poses_by_endpoint: list[list[np.ndarray]] = [[] for _ in range(n_endpoints)]
    per_demo_poses: list[list[list[float]]] = []
    table_point = np.asarray(env.table_surface_point, dtype=float).reshape(3)
    table_normal = np.asarray(env.table_normal, dtype=float).reshape(3)
    table_normal /= np.linalg.norm(table_normal)

    for demo_index, (demo, demo_cutpoints) in enumerate(zip(demos, cutpoints)):
        trajectory = np.asarray(demo, dtype=float)
        boundaries = np.asarray(demo_cutpoints, dtype=int).reshape(-1)
        if trajectory.ndim != 2 or trajectory.shape[1] < 7:
            raise ValueError("Learned endpoint export requires xyz+xyzw demo poses")
        if len(boundaries) != n_endpoints or np.any(boundaries <= 0) or np.any(boundaries >= len(trajectory)):
            raise ValueError("Learned stage boundaries are invalid for endpoint export")
        scene = env.get_demo_scene(demo_index)
        bar_reference, bar_axis, bar_lateral = env._bar_geometry_trace(
            trajectory, scene=scene
        )
        reference = np.asarray(bar_reference[0], dtype=float).reshape(3)
        rotation_world_from_task = np.column_stack(
            (
                np.asarray(bar_axis[0], dtype=float).reshape(3),
                np.asarray(bar_lateral[0], dtype=float).reshape(3),
                table_normal,
            )
        )
        origin = reference - table_normal * float((reference - table_point) @ table_normal)
        demo_poses = []
        for endpoint_index, boundary in enumerate(boundaries):
            sample = trajectory[int(boundary) - 1]
            position_task = rotation_world_from_task.T @ (sample[:3] - origin)
            rotation_task = rotation_world_from_task.T @ _quaternion_to_matrix(sample[3:7])
            quaternion_task = _matrix_to_quaternion(rotation_task)
            pose_task = np.concatenate((position_task, quaternion_task))
            poses_by_endpoint[endpoint_index].append(pose_task)
            demo_poses.append(pose_task.tolist())
        per_demo_poses.append(demo_poses)

    aggregate = []
    for endpoint_values in poses_by_endpoint:
        values = np.asarray(endpoint_values, dtype=float)
        aggregate.append(
            np.concatenate(
                (np.mean(values[:, :3], axis=0), _mean_quaternion(values[:, 3:7]))
            ).tolist()
        )
    return aggregate, {
        "position_aggregation": "arithmetic_mean",
        "orientation_aggregation": "markley_quaternion_mean",
        "sample_policy": "last_sample_before_predicted_cutpoint",
        "coordinate_frame": "bar_table_task",
        "demo_count": len(demos),
        "per_demo_endpoint_poses_bar": per_demo_poses,
    }


def _extract_learned_constraint_artifact(
    *,
    dataset_name: str,
    method_name: str,
    method_seed: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = _extract_metrics(result)
    feature_names = [str(value) for value in metrics["ConstraintFeatureNames"]]
    semantics = np.asarray(metrics["ConstraintLearnedSemanticsMatrix"], dtype=object)
    values = np.asarray(metrics["ConstraintLearnedValueMatrix"], dtype=float)
    if semantics.ndim != 2 or values.shape != semantics.shape:
        raise ValueError("Learned constraint semantics and value matrices must have equal 2-D shapes")
    if semantics.shape[1] != len(feature_names):
        raise ValueError("Learned constraint feature names do not match matrix width")

    dataset = result.get("dataset")
    schema_by_name = {
        str(spec.get("name")): dict(spec)
        for spec in (getattr(dataset, "feature_schema", None) or [])
    }
    feature_schema = [
        {
            "name": name,
            "unit": str(schema_by_name.get(name, {}).get("unit", "")),
            "frame": str(schema_by_name.get(name, {}).get("frame", "")),
        }
        for name in feature_names
    ]
    dataset_meta = getattr(dataset, "meta", {}) or {}
    observation_specs = dataset_meta.get("observation_specs", {})
    task_frame = (
        dict(observation_specs.get("task_frame", {}))
        if isinstance(observation_specs, Mapping)
        else {}
    )
    feature_definition = (
        dict(observation_specs.get("feature_definition", {}))
        if isinstance(observation_specs, Mapping)
        else {}
    )
    endpoint_poses, endpoint_aggregation = _aggregate_learned_endpoint_poses(result)
    if len(endpoint_poses) != semantics.shape[0] - 1:
        raise ValueError("Learned endpoint count does not match the learned stage count")

    pairs = []
    for stage in range(semantics.shape[0]):
        for feature, feature_name in enumerate(feature_names):
            mode = str(semantics[stage, feature]).strip() or "inactive"
            if mode not in _LEARNED_MODES:
                raise ValueError(
                    "Unsupported learned constraint mode {!r} at stage {}, feature {}".format(
                        mode, stage, feature_name
                    )
                )
            pair = {
                "stage": int(stage),
                "feature_name": feature_name,
                "mode": mode,
                "value": None if mode == "inactive" else _finite_or_none(values[stage, feature]),
                "mode_scores": _map_mode_scores(result, stage, feature),
            }
            if mode != "inactive" and pair["value"] is None:
                raise ValueError(
                    "Active learned constraint has no finite value at stage {}, feature {}".format(
                        stage, feature_name
                    )
                )
            scores = pair["mode_scores"]
            pair["confidence"] = max(scores.values()) if scores else None
            pairs.append(pair)

    return {
        "schema_version": 5,
        "artifact_type": "learned_stage_constraints",
        "task_id": str(dataset_name),
        "method_name": str(method_name),
        "method_seed": int(method_seed),
        "num_stages": int(semantics.shape[0]),
        "feature_schema": feature_schema,
        "task_frame": task_frame,
        "feature_definition": feature_definition,
        "endpoint_coordinate_frame": "bar_table_task",
        "stage_endpoint_poses_bar": endpoint_poses,
        "endpoint_aggregation": endpoint_aggregation,
        "feature_stage_modes": pairs,
        "true_constraint_specs": getattr(dataset, "constraint_specs", None),
    }


def _extract_scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if np.isscalar(value):
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                out[str(key)] = value
    return out


def _append_scalar_if_finite(out: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    if not np.isscalar(value):
        return
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(value_f):
        out[str(key)] = value_f


def _extract_objectives(method_name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    if method_name in {
        "swcl",
        "map",
        "map_pooled",
        "map_balanced_pooled",
        "map_balanced_vote",
    }:
        model = result.get("joint_result", {}).get("model", None)
        if model is not None:
            history = getattr(model, "loss_total", None)
            if history:
                _append_scalar_if_finite(out, "ModelObjectiveFinal", history[-1])
                _append_scalar_if_finite(out, "TrainingTotalCostFinal", history[-1])
        return out

    segmentation = result.get("segmentation", None)
    seg_model = getattr(segmentation, "model", None) if segmentation is not None else None
    seg_extras = getattr(segmentation, "extras", {}) if segmentation is not None else {}
    if not isinstance(seg_extras, Mapping):
        seg_extras = {}
    constraint_model = result.get("constraints", {}).get("model", None)

    if method_name == "cluster" and seg_model is not None:
        history = getattr(seg_model, "objective_history_", None)
        if history:
            _append_scalar_if_finite(out, "ModelObjectiveFinal", history[-1])
            _append_scalar_if_finite(out, "SegmentationObjectiveFinal", history[-1])
    elif method_name == "changeforest" and seg_model is not None:
        history = getattr(seg_model, "objective_history_", None)
        if history:
            _append_scalar_if_finite(out, "ModelObjectiveFinal", history[-1])
            _append_scalar_if_finite(out, "SegmentationGainFinal", history[-1])
    elif method_name == "arhsmm":
        history = (seg_extras.get("segmentation_history") or {}).get("loglik")
        if history:
            _append_scalar_if_finite(out, "ModelObjectiveFinal", history[-1])
            _append_scalar_if_finite(out, "SegmentationLogLikelihoodFinal", history[-1])
    elif method_name == "gmmhmm" and seg_model is not None:
        history = getattr(seg_model, "loss_loglik", None)
        if history:
            _append_scalar_if_finite(out, "ModelObjectiveFinal", history[-1])
            _append_scalar_if_finite(out, "SegmentationLogLikelihoodFinal", history[-1])
    elif method_name in {"fchmm", "hmm"} and constraint_model is not None:
        history = getattr(constraint_model, "loss_loglik", None)
        if history:
            _append_scalar_if_finite(out, "ModelObjectiveFinal", history[-1])
            _append_scalar_if_finite(out, "TrainingLogLikelihoodFinal", history[-1])

    if constraint_model is not None:
        _append_scalar_if_finite(out, "PosthocObjectiveFinal", getattr(constraint_model, "posthoc_total_objective_", None))
        _append_scalar_if_finite(
            out,
            "PosthocFeatureObjectiveFinal",
            getattr(constraint_model, "posthoc_feature_objective_", None),
        )
    return out


def save_run_artifacts(
    *,
    run_dir: str | Path,
    dataset_name: str,
    method_name: str,
    dataset_kwargs: Mapping[str, Any],
    method_kwargs: Mapping[str, Any],
    result: Mapping[str, Any],
    env_config_path: str | Path | None = None,
    method_config_path: str | Path | None = None,
) -> dict[str, Path]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = _extract_metrics(result)
    objectives = _extract_objectives(method_name, result)
    plot_dir = resolve_plot_dir(run_dir)
    method_seed = default_method_seed(method_name, method_kwargs)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "method_name": method_name,
        "pipeline": result.get("pipeline"),
        "dataset_seed": int(dataset_kwargs.get("seed", 0)),
        "dataset_fingerprint": dataset_fingerprint(result.get("dataset")),
        "method_seed": int(method_seed),
        "num_demos": int(len(getattr(result.get("dataset"), "demos", []))),
        "num_stages": int(
            getattr(
                getattr(result.get("segmentation"), "model", None),
                "num_stages",
                getattr(
                    result.get("joint_result", {}).get("model", None),
                    "num_stages",
                    0,
                ),
            )
        ),
        "plot_dir": str(plot_dir),
        "env_config_path": str(env_config_path) if env_config_path is not None else None,
        "method_config_path": str(method_config_path) if method_config_path is not None else None,
    }
    config_snapshot = {
        "dataset_name": dataset_name,
        "method_name": method_name,
        "dataset_kwargs": dict(dataset_kwargs),
        "method_kwargs": dict(method_kwargs),
    }

    files = {
        "metadata": write_json(run_dir / "metadata.json", metadata),
        "config": write_json(run_dir / "config_snapshot.json", config_snapshot),
        "metrics": write_json(
            run_dir / "metrics.json",
            {
                "scalar_metrics": _extract_scalar_metrics(metrics),
                "all_metrics": metrics,
                "objectives": objectives,
            },
        ),
        "objectives": write_json(
            run_dir / "objectives.json",
            objectives,
        ),
        "segmentation": write_json(
            run_dir / "segmentation.json",
            _extract_segmentation(result),
        ),
        "constraints": write_json(
            run_dir / "constraints.json",
            _extract_constraints(result),
        ),
    }
    if method_name in _MAP_METHODS:
        files["learned_constraints"] = write_json(
            run_dir / "learned_constraints.json",
            _extract_learned_constraint_artifact(
                dataset_name=dataset_name,
                method_name=method_name,
                method_seed=method_seed,
                result=result,
            ),
        )
    return files
