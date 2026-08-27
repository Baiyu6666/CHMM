from types import SimpleNamespace

import numpy as np

from experiments.artifacts import _extract_learned_constraint_artifact


class _EndpointEnv:
    table_surface_point = np.zeros(3)
    table_normal = np.asarray([0.0, 0.0, 1.0])

    @staticmethod
    def get_demo_scene(_demo_index):
        return None

    @staticmethod
    def _bar_geometry_trace(trajectory, scene=None):
        del scene
        count = len(trajectory)
        return (
            np.zeros((count, 3)),
            np.repeat([[1.0, 0.0, 0.0]], count, axis=0),
            np.repeat([[0.0, 1.0, 0.0]], count, axis=0),
        )


def test_map_artifact_contains_every_feature_stage_pair_and_inactive_mode():
    metrics = {
        "ConstraintFeatureNames": ["table_dist", "tool_pitch"],
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
        demos=[
            np.asarray(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 [1.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
                 [2.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]]
            ),
            np.asarray(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                 [3.0, 0.0, 0.3, 0.0, 0.0, 0.0, -1.0],
                 [4.0, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]]
            ),
        ],
        env=_EndpointEnv(),
        feature_schema=[
            {"name": "table_dist", "unit": "m", "frame": "bar_table_task.z"},
            {"name": "tool_pitch", "unit": "rad", "frame": "bar_table_task orientation"},
        ],
        constraint_specs=[],
        meta={
            "observation_specs": {
                "task_frame": {
                    "frame_id": "bar_table_task",
                    "snapshot_policy": "frozen_per_task",
                },
                "feature_definition": {"id": "bar_table_features_v1"},
            },
            "planning_profile": {
                "stage_names": ["s1", "s2"],
                "endpoint_coordinate_frame": "bar_table_task",
                "stage_endpoint_positions_bar": [[0.2, 0.0, 0.1]],
                "planner": {"control_spacing_m": 0.04},
                "position_smooth_scale": 0.02,
                "axis_smooth_scale": 0.2,
                "yaw_smooth_scale": 0.2,
                "constraint_transition": {},
                "optimizer_weights": {},
            },
        },
    )
    result = {
        "joint_result": {
            "metrics": metrics,
            "model": model,
            "cutpoints_hat": [np.asarray([2]), np.asarray([2])],
        },
        "dataset": dataset,
    }

    artifact = _extract_learned_constraint_artifact(
        dataset_name="BarClean",
        method_name="map",
        method_seed=0,
        result=result,
    )

    assert artifact["schema_version"] == 5
    assert artifact["task_id"] == "BarClean"
    assert artifact["task_frame"]["frame_id"] == "bar_table_task"
    assert artifact["feature_definition"]["id"] == "bar_table_features_v1"
    assert artifact["endpoint_coordinate_frame"] == "bar_table_task"
    assert np.allclose(
        artifact["stage_endpoint_poses_bar"][0],
        [2.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
    )
    assert artifact["endpoint_aggregation"]["demo_count"] == 2
    assert artifact["feature_schema"][0]["frame"] == "bar_table_task.z"
    assert len(artifact["feature_stage_modes"]) == 4
    inactive = artifact["feature_stage_modes"][0]
    assert inactive["mode"] == "inactive"
    assert inactive["value"] is None
    assert "scale" not in inactive
    assert "planning_profile" not in artifact
    assert inactive["mode_scores"]["inactive"] > inactive["mode_scores"]["target_value"]
