import json
from pathlib import Path

import numpy as np

from stage_constraint_planner.optimizer import (
    StageConstraintTrajectoryOptimizer,
    build_bar_table_task_frame,
    canonicalize_bar_pose,
    quaternions_from_axes_and_yaws,
    quaternion_to_matrix,
    tool_yaw_from_quaternion,
    transform_pose,
)


def test_smoothstep5_stays_inside_unit_interval_near_upper_endpoint():
    values = np.asarray(
        [
            0.0,
            0.5,
            0.9999979999999999,
            np.nextafter(1.0, 0.0),
            1.0,
        ]
    )

    smoothed = StageConstraintTrajectoryOptimizer._smoothstep5(values)

    assert np.all(np.isfinite(smoothed))
    assert np.all(smoothed >= 0.0)
    assert np.all(smoothed <= 1.0)
    assert np.all(1.0 - smoothed >= 0.0)


def test_transform_pose_maps_optitrack_axes_into_robot_frame():
    rotation = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    pose = np.asarray([-0.153, 0.048, 0.651, 0.0, 0.0, 0.0, 1.0])

    transformed = transform_pose(pose, rotation, np.zeros(3))

    assert np.allclose(transformed[:3], [0.651, -0.153, 0.048])
    assert np.allclose(quaternion_to_matrix(transformed[3:]), rotation)


def test_canonical_bar_axis_points_away_from_obstacle():
    scene_rotation = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    bar = np.asarray([0.653, -0.172, 0.050, 0.0, 0.0, 0.0, 1.0])
    bar[3:] = transform_pose(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        scene_rotation,
        np.zeros(3),
    )[3:]
    obstacle = np.asarray([0.669, 0.094, 0.050, 0.0, 0.0, 0.0, 1.0])

    corrected, flipped = canonicalize_bar_pose(
        bar,
        obstacle,
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        "away_from_obstacle",
    )

    axis = quaternion_to_matrix(corrected[3:])[:, 0]
    toward_obstacle = obstacle[:3] - corrected[:3]
    assert flipped
    assert np.allclose(corrected[:3], bar[:3])
    assert float(axis @ toward_obstacle) < 0.0


def test_canonicalization_moves_demo_s1_to_obstacle_end_of_current_bar():
    scene_rotation = np.asarray(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    bar = transform_pose(
        [-0.172, 0.050, 0.653, 0.0, 0.0, 0.0, 1.0],
        scene_rotation,
        np.zeros(3),
    )
    obstacle = transform_pose(
        [0.094, 0.050, 0.669, 0.0, 0.0, 0.0, 1.0],
        scene_rotation,
        np.zeros(3),
    )
    s1_bar = np.asarray([-0.10104227, 0.16634768, -0.01365577])
    uncorrected_s1 = bar[:3] + quaternion_to_matrix(bar[3:]) @ s1_bar
    corrected, flipped = canonicalize_bar_pose(
        bar,
        obstacle,
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        "away_from_obstacle",
    )
    corrected_s1 = corrected[:3] + quaternion_to_matrix(corrected[3:]) @ s1_bar

    assert flipped
    assert np.linalg.norm(corrected_s1[:2] - obstacle[:2]) < np.linalg.norm(
        uncorrected_s1[:2] - obstacle[:2]
    )


def test_bar_table_endpoints_use_distance_above_table_not_marker_height():
    config_path = Path(__file__).resolve().parents[1] / "config" / "bar_clean_true.json"
    config = json.loads(config_path.read_text())
    optimizer = StageConstraintTrajectoryOptimizer(config)
    start = np.asarray([0.4, -0.2, 0.3])
    goal = np.asarray([0.5, -0.1, 0.3])
    marker_low = np.asarray([0.6, -0.2, 0.05, 0.0, 0.0, 0.0, 1.0])
    marker_high = marker_low.copy()
    marker_high[2] = 0.30
    obstacle = np.asarray([0.6, 0.2, 0.05, 0.0, 0.0, 0.0, 1.0])
    frame_low = build_bar_table_task_frame(
        marker_low,
        obstacle,
        config["table_surface_point"],
        config["table_normal"],
        config["bar_axis_local"],
        config["canonical_bar_axis"],
    )
    frame_high = build_bar_table_task_frame(
        marker_high,
        obstacle,
        config["table_surface_point"],
        config["table_normal"],
        config["bar_axis_local"],
        config["canonical_bar_axis"],
    )

    endpoints_low = optimizer._world_endpoints(
        start, goal, marker_low, task_frame=frame_low
    )[1:-1]
    endpoints_high = optimizer._world_endpoints(
        start, goal, marker_high, task_frame=frame_high
    )[1:-1]
    table_point = np.asarray(config["table_surface_point"], dtype=float)
    table_normal = np.asarray(config["table_normal"], dtype=float)
    expected = np.asarray([0.060, 0.060, 0.020, 0.020])

    assert np.allclose((endpoints_low - table_point) @ table_normal, expected)
    assert np.allclose((endpoints_high - table_point) @ table_normal, expected)
    assert np.allclose(endpoints_low, endpoints_high)


def test_bar_table_task_frame_ignores_motive_axes_other_than_bar_axis():
    roll_about_bar_axis = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )
    identity_pose = np.asarray([0.6, -0.2, 0.24, 0.0, 0.0, 0.0, 1.0])
    rolled_pose = identity_pose.copy()
    rolled_pose[3:] = transform_pose(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        roll_about_bar_axis,
        np.zeros(3),
    )[3:]
    obstacle = np.asarray([0.6, 0.2, 0.2, 0.0, 0.0, 0.0, 1.0])

    frames = [
        build_bar_table_task_frame(
            pose,
            obstacle,
            [0.0, 0.0, 0.14584],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            "tracker_frame",
        )
        for pose in (identity_pose, rolled_pose)
    ]

    assert np.allclose(frames[0]["rotation_world_from_task"], np.eye(3))
    assert np.allclose(
        frames[0]["rotation_world_from_task"],
        frames[1]["rotation_world_from_task"],
    )
    assert np.isclose(frames[0]["origin"][2], 0.14584)
    assert frames[0]["snapshot_policy"] == "frozen_per_task"


def test_endpoint_group_offset_moves_selected_subgoals_only():
    config_path = Path(__file__).resolve().parents[1] / "config" / "bar_clean_true.json"
    config = json.loads(config_path.read_text())
    base = np.asarray(config["stage_endpoint_positions_bar"], dtype=float)

    optimizer = StageConstraintTrajectoryOptimizer(config)

    expected = base.copy()
    for group in config["endpoint_group_offsets"]:
        indices = [int(value) - 1 for value in group["endpoints"]]
        expected[indices] += np.asarray(group["offset_bar"], dtype=float)
    assert np.allclose(optimizer._endpoint_positions_bar, expected)
    assert np.allclose(optimizer._endpoint_positions_bar[0], base[0])


def test_full_orientation_reconstruction_preserves_bar_relative_yaw():
    frame = {
        "axial": np.asarray([1.0, 0.0, 0.0]),
        "lateral": np.asarray([0.0, 1.0, 0.0]),
        "normal": np.asarray([0.0, 0.0, 1.0]),
    }
    axes = np.asarray([[0.0, 0.0, -1.0], [0.1, 0.0, -0.995]])
    yaws = np.asarray([-np.pi / 2.0, -np.pi / 2.0])

    quaternions = quaternions_from_axes_and_yaws(axes, yaws, frame)

    assert np.allclose(
        [tool_yaw_from_quaternion(value, frame) for value in quaternions],
        yaws,
    )
    assert np.allclose(
        [quaternion_to_matrix(value)[:, 2] for value in quaternions],
        axes / np.linalg.norm(axes, axis=1, keepdims=True),
    )


def test_bar_clean_yaw_equality_is_active_only_in_user_stages_two_and_three():
    config_path = Path(__file__).resolve().parents[1] / "config" / "bar_clean_true.json"
    config = json.loads(config_path.read_text())
    terms = [
        value for value in config["constraint_terms"]
        if value["feature_name"] == "tool_yaw"
    ]

    assert [value["stage"] for value in terms] == [1, 2]
    assert all(value["semantics"] == "target_value" for value in terms)
    assert np.isclose(terms[0]["value"], terms[1]["value"])


def test_tool_yaw_active_mask_follows_loaded_constraint_terms():
    labels = np.asarray([0, 0, 1, 1, 2, 3, 4])
    terms = [
        {"feature_name": "tool_yaw", "stage": 1},
        {"feature_name": "tool_yaw", "stage": 2},
        {"feature_name": "surface_dist", "stage": 3},
    ]

    active = StageConstraintTrajectoryOptimizer._feature_active_mask(
        labels, terms, "tool_yaw"
    )
    inactive = StageConstraintTrajectoryOptimizer._feature_active_mask(
        labels, [], "tool_yaw"
    )

    assert active.tolist() == [False, False, True, True, True, False, False]
    assert not np.any(inactive)


def test_tool_yaw_mask_waits_until_soft_stage_transition_is_complete():
    labels = np.asarray([0, 1, 1, 1, 2, 2, 3])
    weights = np.asarray(
        [
            [1.00, 0.00, 0.00, 0.00],
            [0.75, 0.25, 0.00, 0.00],
            [0.25, 0.75, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.00, 0.50, 0.50, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.50, 0.50],
        ]
    )
    terms = [
        {"feature_name": "tool_yaw", "stage": 1},
        {"feature_name": "tool_yaw", "stage": 2},
    ]

    active = StageConstraintTrajectoryOptimizer._feature_active_mask(
        labels, terms, "tool_yaw", weights
    )

    assert active.tolist() == [False, False, False, True, True, True, False]
