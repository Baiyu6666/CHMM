import json

import pytest

from stage_constraint_planner.constraint_artifact import configure_planning_profile


def _task_config():
    return {
        "stage_names": ["s1", "s2"],
        "task_frame": {
            "frame_id": "bar_table_task",
            "snapshot_policy": "frozen_per_task",
        },
        "feature_definition": {"id": "test_features_v1"},
        "planning_profile_fields": [
            "stage_names",
            "endpoint_coordinate_frame",
            "stage_endpoint_positions_bar",
            "planner",
            "position_smooth_scale",
            "axis_smooth_scale",
            "yaw_smooth_scale",
            "constraint_transition",
            "constraint_settling",
            "optimizer_weights",
        ],
        "endpoint_coordinate_frame": "bar_table_task",
        "stage_endpoint_positions_bar": [[0.1, 0.0, 0.1]],
        "planner": {
            "control_spacing_m": 0.04,
            "output_spacing_m": 0.005,
            "output_axis_spacing_deg": 2.0,
            "min_control_points": 4,
            "max_control_points": 8,
            "max_nfev": 20,
            "multi_start": 1,
        },
        "position_smooth_scale": 0.02,
        "axis_smooth_scale": 0.2,
        "yaw_smooth_scale": 0.2,
        "constraint_transition": {"fraction": 0.2, "min_distance": 0.01, "max_distance": 0.05},
        "constraint_settling": {
            "control_points": 2,
            "max_progress_m": 0.004,
            "progress_weight": 0.5,
            "smoothness_scale": 0.2,
        },
        "optimizer_weights": {"constraint": 3.0},
        "feature_units": {"table_dist": "m", "tool_pitch": "rad"},
        "constraint_terms": [
            {
                "feature_name": "table_dist",
                "stage": 0,
                "semantics": "target_value",
                "value": 0.02,
                "scale": 0.01,
                "weight": 2.0,
            },
            {
                "feature_name": "tool_pitch",
                "stage": 0,
                "semantics": "target_value",
                "value": 1.2,
                "scale": 0.05,
                "weight": 4.0,
            },
        ],
    }


def _artifact():
    return {
        "schema_version": 5,
        "artifact_type": "learned_stage_constraints",
        "task_id": "BarClean",
        "task_frame": {
            "frame_id": "bar_table_task",
            "snapshot_policy": "frozen_per_task",
        },
        "feature_definition": {"id": "test_features_v1"},
        "endpoint_coordinate_frame": "bar_table_task",
        "stage_endpoint_poses_bar": [
            [0.25, -0.1, 0.08, 0.0, 0.0, 0.0, 1.0]
        ],
        "num_stages": 2,
        "feature_schema": [
            {"name": "table_dist", "unit": "m"},
            {"name": "tool_pitch", "unit": "rad"},
        ],
        "feature_stage_modes": [
            {
                "stage": 0,
                "feature_name": "table_dist",
                "mode": "inactive",
                "value": None,
            },
            {
                "stage": 0,
                "feature_name": "tool_pitch",
                "mode": "lower_bound",
                "value": 1.1,
            },
            {
                "stage": 1,
                "feature_name": "table_dist",
                "mode": "upper_bound",
                "value": 0.04,
            },
            {
                "stage": 1,
                "feature_name": "tool_pitch",
                "mode": "target_value",
                "value": 1.4,
            },
        ],
    }


def test_true_source_preserves_true_terms(tmp_path):
    config = _task_config()

    configure_planning_profile(config, "BarClean", "true", tmp_path)

    assert config["constraint_terms"] == config["true_constraint_terms"]
    assert config["planning_constraint_source"] == "true"


def test_dense_artifact_replaces_terms_and_skips_inactive(tmp_path):
    artifact_path = tmp_path / "learned_constraints.json"
    artifact_path.write_text(json.dumps(_artifact()), encoding="utf-8")
    config = _task_config()

    configure_planning_profile(config, "BarClean", artifact_path, tmp_path)

    assert len(config["constraint_terms"]) == 3
    assert config["constraint_terms"][0]["semantics"] == "lower_bound"
    assert config["constraint_terms"][1]["semantics"] == "upper_bound"
    assert config["constraint_terms"][0]["scale"] == 0.05
    assert config["constraint_terms"][0]["weight"] == 4.0
    assert config["constraint_terms"][1]["scale"] == 0.01
    assert config["constraint_terms"][1]["weight"] == 2.0
    assert config["true_constraint_terms"][0]["value"] == 0.02
    assert config["stage_names"] == ["s1", "s2"]
    assert config["stage_endpoint_positions_bar"] == [[0.25, -0.1, 0.08]]
    assert config["stage_endpoint_poses_bar"] == [
        [0.25, -0.1, 0.08, 0.0, 0.0, 0.0, 1.0]
    ]
    assert config["planner"]["control_spacing_m"] == 0.04


def test_artifact_must_contain_every_candidate_pair(tmp_path):
    artifact = _artifact()
    artifact["feature_stage_modes"].pop()
    artifact_path = tmp_path / "learned_constraints.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="missing feature-stage pairs"):
        configure_planning_profile(
            _task_config(), "BarClean", artifact_path, tmp_path
        )


def test_artifact_cannot_escape_configured_root(tmp_path):
    outside = tmp_path.parent / "outside-learned-constraints.json"
    outside.write_text(json.dumps(_artifact()), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside the configured artifact root"):
            configure_planning_profile(
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
        configure_planning_profile(
            _task_config(), "BarClean", artifact_path, tmp_path
        )
