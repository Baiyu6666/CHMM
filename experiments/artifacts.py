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

    scales = np.asarray(
        metrics.get("ConstraintFeatureScales", np.ones(len(feature_names))), dtype=float
    ).reshape(-1)
    if scales.size != len(feature_names):
        raise ValueError("Learned constraint feature scales do not match feature names")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("Learned constraint feature scales must be positive and finite")

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
            "scale": _finite_or_none(scales[index]),
        }
        for index, name in enumerate(feature_names)
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
    planning_profile = result.get("learned_planning_profile")
    if planning_profile is None:
        planning_profile = dataset_meta.get("planning_profile")
    if not isinstance(planning_profile, Mapping):
        raise ValueError("A learned constraint artifact requires a planning profile")

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
                "scale": _finite_or_none(scales[feature]),
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
        "schema_version": 3,
        "artifact_type": "learned_stage_constraints",
        "task_id": str(dataset_name),
        "method_name": str(method_name),
        "method_seed": int(method_seed),
        "num_stages": int(semantics.shape[0]),
        "feature_schema": feature_schema,
        "task_frame": task_frame,
        "feature_definition": feature_definition,
        "planning_profile": json.loads(json.dumps(dict(planning_profile))),
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
