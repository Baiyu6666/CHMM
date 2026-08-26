from types import SimpleNamespace

from experiments.artifacts import _extract_learned_constraint_artifact


def test_map_artifact_contains_every_feature_stage_pair_and_inactive_mode():
    metrics = {
        "ConstraintFeatureNames": ["surface_dist", "tool_pitch"],
        "ConstraintLearnedSemanticsMatrix": [
            ["inactive", "target_value"],
            ["lower_bound", "upper_bound"],
        ],
        "ConstraintLearnedValueMatrix": [
            [float("nan"), 1.2],
            [0.04, 1.4],
        ],
        "ConstraintFeatureScales": [0.01, 0.1],
    }
    model = SimpleNamespace(
        map_shared_mode_costs_=[
            [
                {"inactive": 0.0, "eq": 2.0, "lb": 3.0, "ub": 4.0},
                {"inactive": 4.0, "eq": 0.0, "lb": 2.0, "ub": 3.0},
            ],
            [
                {"inactive": 4.0, "eq": 2.0, "lb": 0.0, "ub": 3.0},
                {"inactive": 4.0, "eq": 2.0, "lb": 3.0, "ub": 0.0},
            ],
        ]
    )
    dataset = SimpleNamespace(
        feature_schema=[
            {"name": "surface_dist", "unit": "m", "frame": "bar_table_task.z"},
            {"name": "tool_pitch", "unit": "rad", "frame": "bar_table_task orientation"},
        ],
        constraint_specs=[],
        meta={
            "observation_specs": {
                "task_frame": {
                    "frame_id": "bar_table_task",
                    "snapshot_policy": "frozen_per_task",
                }
            }
        },
    )
    result = {
        "joint_result": {"metrics": metrics, "model": model},
        "dataset": dataset,
    }

    artifact = _extract_learned_constraint_artifact(
        dataset_name="BarClean",
        method_name="map",
        method_seed=0,
        result=result,
    )

    assert artifact["schema_version"] == 2
    assert artifact["task_id"] == "BarClean"
    assert artifact["task_frame"]["frame_id"] == "bar_table_task"
    assert artifact["feature_schema"][0]["frame"] == "bar_table_task.z"
    assert len(artifact["feature_stage_modes"]) == 4
    inactive = artifact["feature_stage_modes"][0]
    assert inactive["mode"] == "inactive"
    assert inactive["value"] is None
    assert inactive["mode_scores"]["inactive"] > inactive["mode_scores"]["target_value"]
