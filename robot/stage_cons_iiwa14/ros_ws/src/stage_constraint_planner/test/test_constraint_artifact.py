import json

import pytest

from stage_constraint_planner.constraint_artifact import configure_task_constraints


def _task_config():
    return {
        "stage_names": ["s1", "s2"],
        "task_frame": {
            "frame_id": "bar_table_task",
            "snapshot_policy": "frozen_per_task",
        },
        "feature_units": {"surface_dist": "m", "tool_pitch": "rad"},
        "constraint_terms": [
            {
                "feature_name": "surface_dist",
                "stage": 0,
                "semantics": "target_value",
                "value": 0.02,
                "scale": 0.01,
                "weight": 2.0,
            }
        ],
    }


def _artifact():
    return {
        "schema_version": 2,
        "artifact_type": "learned_stage_constraints",
        "task_id": "BarClean",
        "task_frame": {
            "frame_id": "bar_table_task",
            "snapshot_policy": "frozen_per_task",
        },
        "num_stages": 2,
        "feature_schema": [
            {"name": "surface_dist", "unit": "m", "scale": 0.01},
            {"name": "tool_pitch", "unit": "rad", "scale": 0.1},
        ],
        "feature_stage_modes": [
            {
                "stage": 0,
                "feature_name": "surface_dist",
                "mode": "inactive",
                "value": None,
                "scale": 0.01,
            },
            {
                "stage": 0,
                "feature_name": "tool_pitch",
                "mode": "lower_bound",
                "value": 1.1,
                "scale": 0.1,
            },
            {
                "stage": 1,
                "feature_name": "surface_dist",
                "mode": "upper_bound",
                "value": 0.04,
                "scale": 0.01,
                "weight": 3.0,
            },
            {
                "stage": 1,
                "feature_name": "tool_pitch",
                "mode": "target_value",
                "value": 1.4,
                "scale": 0.1,
            },
        ],
    }


def test_true_source_preserves_true_terms(tmp_path):
    config = _task_config()

    configure_task_constraints(config, "BarClean", "true", tmp_path)

    assert config["constraint_terms"] == config["true_constraint_terms"]
    assert config["planning_constraint_source"] == "true"


def test_dense_artifact_replaces_terms_and_skips_inactive(tmp_path):
    artifact_path = tmp_path / "learned_constraints.json"
    artifact_path.write_text(json.dumps(_artifact()), encoding="utf-8")
    config = _task_config()

    configure_task_constraints(config, "BarClean", artifact_path, tmp_path)

    assert len(config["constraint_terms"]) == 3
    assert config["constraint_terms"][0]["semantics"] == "lower_bound"
    assert config["constraint_terms"][1]["semantics"] == "upper_bound"
    assert config["constraint_terms"][1]["weight"] == 3.0
    assert config["true_constraint_terms"][0]["value"] == 0.02


def test_artifact_must_contain_every_candidate_pair(tmp_path):
    artifact = _artifact()
    artifact["feature_stage_modes"].pop()
    artifact_path = tmp_path / "learned_constraints.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="missing feature-stage pairs"):
        configure_task_constraints(
            _task_config(), "BarClean", artifact_path, tmp_path
        )


def test_artifact_cannot_escape_configured_root(tmp_path):
    outside = tmp_path.parent / "outside-learned-constraints.json"
    outside.write_text(json.dumps(_artifact()), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside the configured artifact root"):
            configure_task_constraints(
                _task_config(), "BarClean", outside, tmp_path
            )
    finally:
        outside.unlink()


def test_artifact_task_frame_must_match_planner(tmp_path):
    artifact = _artifact()
    artifact["task_frame"]["frame_id"] = "raw_motive_frame"
    artifact_path = tmp_path / "learned_constraints.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="task frame does not match"):
        configure_task_constraints(
            _task_config(), "BarClean", artifact_path, tmp_path
        )
