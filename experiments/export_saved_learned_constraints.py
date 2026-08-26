from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def convert(constraints_path: Path, task_id: str | None = None) -> Path:
    constraints = json.loads(constraints_path.read_text(encoding="utf-8"))
    metadata_path = constraints_path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = [str(value) for value in constraints["ConstraintFeatureNames"]]
    semantics = np.asarray(
        constraints["ConstraintLearnedSemanticsMatrix"], dtype=object
    )
    values = np.asarray(constraints["ConstraintLearnedValueMatrix"], dtype=float)
    scales = np.asarray(
        constraints.get("ConstraintFeatureScales", np.ones(len(names))), dtype=float
    ).reshape(-1)
    if semantics.ndim != 2 or values.shape != semantics.shape:
        raise ValueError("Constraint semantics and value matrices must have equal 2-D shapes")
    if semantics.shape[1] != len(names) or scales.size != len(names):
        raise ValueError("Constraint matrix columns do not match feature metadata")
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("Constraint feature scales must be positive and finite")

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
                    "scale": float(scales[feature]),
                    "mode_scores": {},
                    "confidence": None,
                }
            )

    resolved_task = str(task_id or metadata["dataset_name"])
    artifact = {
        "schema_version": 1,
        "artifact_type": "learned_stage_constraints",
        "task_id": resolved_task,
        "method_name": str(metadata["method_name"]),
        "method_seed": int(metadata["method_seed"]),
        "num_stages": int(semantics.shape[0]),
        "feature_schema": [
            {"name": name, "unit": "", "scale": float(scales[index])}
            for index, name in enumerate(names)
        ],
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
