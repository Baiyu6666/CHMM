from __future__ import annotations

import json
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
    "ConstraintTargetMatrix",
    "ConstraintLearnedValueMatrix",
    "ConstraintLearnedValuePerDemo",
    "ConstraintErrorMatrix",
    "ConstraintLearnedDemoCount",
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
    posthoc_key = "posthoc" if method_name == "hmm" else "constraints"
    constraint_cfg = dict(cfg.get(posthoc_key, {}))
    segmenter_cfg["plot_dir"] = plot_dir_str
    constraint_cfg["plot_dir"] = plot_dir_str
    cfg["segmenter"] = segmenter_cfg
    cfg[posthoc_key] = constraint_cfg
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


def _extract_scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if np.isscalar(value):
            try:
                out[str(key)] = float(value)
            except (TypeError, ValueError):
                out[str(key)] = value
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
    plot_dir = resolve_plot_dir(run_dir)
    method_seed = default_method_seed(method_name, method_kwargs)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": dataset_name,
        "method_name": method_name,
        "pipeline": result.get("pipeline"),
        "dataset_seed": int(dataset_kwargs.get("seed", 0)),
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
            },
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
    return files
