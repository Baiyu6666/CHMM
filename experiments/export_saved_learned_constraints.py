from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import load_env  # noqa: E402
from experiments.artifacts import _aggregate_learned_endpoint_poses  # noqa: E402


def convert(constraints_path: Path, task_id: str | None = None) -> Path:
    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    metadata_path = constraints_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = [str(value) for value in constraints["ConstraintFeatureNames"]]
    semantics = np.asarray(
        constraints["ConstraintLearnedSemanticsMatrix"], dtype=object
    )
    values = np.asarray(constraints["ConstraintLearnedValueMatrix"], dtype=float)
    if semantics.ndim != 2 or values.shape != semantics.shape:
        raise ValueError("Constraint semantics and value matrices must have equal 2-D shapes")
    if semantics.shape[1] != len(names):
        raise ValueError("Constraint matrix columns do not match feature metadata")

    pairs = []
    for stage in range(semantics.shape[0]):
        for feature, name in enumerate(names):
            mode = str(semantics[stage, feature]).strip() or "inactive"
            if mode not in {
                "inactive",
                "target_value",
                "lower_bound",
                "upper_bound",
            }:
                raise ValueError("Unsupported learned constraint mode {}".format(mode))
            value = float(values[stage, feature]) if mode != "inactive" else None
            if value is not None and not np.isfinite(value):
                raise ValueError("Active learned constraint has a non-finite value")
            pairs.append(
                {
                    "stage": stage,
                    "feature_name": name,
                    "mode": mode,
                    "value": value,
                    "mode_scores": {},
                    "confidence": None,
                }
            )

    resolved_task = str(task_id or metadata["dataset_name"])
    if resolved_task != "BarClean":
        raise ValueError("Only BarClean has a planner task definition")
    definition_path = (
        PROJECT_ROOT
        / "robot/stage_cons_iiwa14/ros_ws/src/stage_constraint_planner/config/bar_clean_true.json"
    )
    definition = json.loads(definition_path.read_text(encoding="utf-8"))
    config_snapshot = json.loads(
        constraints_path.with_name("config_snapshot.json").read_text(encoding="utf-8")
    )
    segmentation = json.loads(
        constraints_path.with_name("segmentation.json").read_text(encoding="utf-8")
    )
    dataset = load_env(
        str(config_snapshot["dataset_name"]),
        **dict(config_snapshot["dataset_kwargs"]),
    )
    endpoint_poses, endpoint_aggregation = _aggregate_learned_endpoint_poses(
        {
            "dataset": dataset,
            "joint_result": {"cutpoints_hat": segmentation["predicted_cutpoints"]},
        }
    )
    if len(endpoint_poses) != int(semantics.shape[0]) - 1:
        raise ValueError("Learned endpoint count does not match the learned stage count")
    artifact = {
        "schema_version": 5,
        "artifact_type": "learned_stage_constraints",
        "task_id": resolved_task,
        "method_name": str(metadata["method_name"]),
        "method_seed": int(metadata["method_seed"]),
        "num_stages": int(semantics.shape[0]),
        "feature_schema": [
            {
                "name": name,
                "unit": str(definition["feature_units"].get(name, "")),
            }
            for name in names
        ],
        "task_frame": dict(definition["task_frame"]),
        "feature_definition": dict(definition["feature_definition"]),
        "endpoint_coordinate_frame": "bar_table_task",
        "stage_endpoint_poses_bar": endpoint_poses,
        "endpoint_aggregation": endpoint_aggregation,
        "feature_stage_modes": pairs,
        "true_constraint_specs": constraints.get("constraint_specs"),
    }
    output = constraints_path.with_name("learned_constraints.json")
    output.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a saved MAP constraints.json into the planner artifact format."
    )
    parser.add_argument("constraints_json", type=Path)
    parser.add_argument("--task-id")
    args = parser.parse_args()
    print(convert(args.constraints_json, task_id=args.task_id))


if __name__ == "__main__":
    main()
