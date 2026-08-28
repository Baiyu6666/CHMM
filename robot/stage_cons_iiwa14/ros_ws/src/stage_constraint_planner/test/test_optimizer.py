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


def test_orientation_unpack_keeps_user_start_and_goal_as_fixed_boundaries():
    positions = np.asarray(
        [
            [0.0, 0.0, 0.2],
            [0.1, 0.0, 0.2],
            [0.2, 0.0, 0.2],
            [0.3, 0.0, 0.2],
        ]
    )
    axes = np.asarray(
        [
            [0.20, 0.0, -0.98],
            [0.10, 0.0, -0.99],
            [-0.10, 0.0, -0.99],
            [-0.25, 0.0, -0.97],
        ]
    )
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    yaws = np.asarray([0.3, 0.2, -0.1, -0.4])
    candidate_axes = axes.copy()
    candidate_axes[1:3] = np.asarray(
        [[0.0, 0.0, -1.0], [0.0, 0.2, -0.98]]
    )
    candidate_axes[1:3] /= np.linalg.norm(
        candidate_axes[1:3], axis=1, keepdims=True
    )
    candidate_yaws = yaws.copy()
    candidate_yaws[1:3] = [0.6, 0.4]
    free = np.asarray([1, 2], dtype=int)
    no_normal_positions = np.empty(0, dtype=int)
    normal = np.asarray([0.0, 0.0, 1.0])

    packed = StageConstraintTrajectoryOptimizer._pack(
        positions,
        candidate_axes,
        candidate_yaws,
        free,
        no_normal_positions,
        normal,
        free,
        free,
    )
    _, _, unpacked_axes, unpacked_yaws = (
        StageConstraintTrajectoryOptimizer._unpack(
            packed,
            positions,
            axes,
            yaws,
            free,
            no_normal_positions,
            normal,
            free,
            free,
        )
    )

    assert np.allclose(unpacked_axes[[0, -1]], axes[[0, -1]])
    assert np.allclose(unpacked_yaws[[0, -1]], yaws[[0, -1]])
    assert np.allclose(unpacked_axes[1:3], candidate_axes[1:3])
    assert np.allclose(unpacked_yaws[1:3], candidate_yaws[1:3])


def test_position_unpack_keeps_endpoint_horizontal_coordinates_and_goal_fixed():
    template = np.asarray(
        [
            [0.0, 0.0, 0.1],
            [0.1, 0.2, 0.1],
            [0.2, 0.3, 0.1],
            [0.3, 0.4, 0.1],
            [0.4, 0.5, 0.1],
            [0.5, 0.6, 0.1],
        ]
    )
    endpoint_indices = np.asarray([1, 3, 5], dtype=int)
    free = StageConstraintTrajectoryOptimizer._free_position_indices(
        len(template), endpoint_indices
    )
    normal_indices = endpoint_indices[:-1]
    normal = np.asarray([0.0, 0.0, 1.0])
    candidate = template.copy()
    candidate[free] += [0.01, -0.02, 0.03]
    candidate[normal_indices] += [0.04, -0.05, 0.06]
    axes = np.tile([0.0, 0.0, -1.0], (len(template), 1))
    yaws = np.zeros(len(template))
    no_orientation = np.empty(0, dtype=int)

    packed = StageConstraintTrajectoryOptimizer._pack(
        candidate,
        axes,
        yaws,
        free,
        normal_indices,
        normal,
        no_orientation,
        no_orientation,
    )
    positions, _, _, _ = StageConstraintTrajectoryOptimizer._unpack(
        packed,
        template,
        axes,
        yaws,
        free,
        normal_indices,
        normal,
        no_orientation,
        no_orientation,
    )

    assert free.tolist() == [2, 4]
    assert np.allclose(positions[free], candidate[free])
    assert np.allclose(positions[normal_indices, :2], template[normal_indices, :2])
    assert np.allclose(positions[normal_indices, 2], candidate[normal_indices, 2])
    assert np.allclose(positions[0], template[0])
    assert np.allclose(positions[-1], template[-1])


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
    expected = np.asarray([0.10204, 0.10204, 0.07004, 0.07004])
    expected_world_z = np.asarray([0.21141, 0.21141, 0.17941, 0.17941])

    assert np.allclose((endpoints_low - table_point) @ table_normal, expected)
    assert np.allclose((endpoints_high - table_point) @ table_normal, expected)
    assert np.allclose(endpoints_low[:, 2], expected_world_z)
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


def test_bar_clean_subgoals_are_stored_as_final_coordinates_without_offsets():
    config_path = Path(__file__).resolve().parents[1] / "config" / "bar_clean_true.json"
    config = json.loads(config_path.read_text())
    endpoints = np.asarray(config["stage_endpoint_positions_bar"], dtype=float)

    optimizer = StageConstraintTrajectoryOptimizer(config)

    assert "endpoint_group_offsets" not in config
    assert np.allclose(optimizer._endpoint_positions_bar, endpoints)
    assert np.isclose(endpoints[1, 1], 0.0)


def test_endpoint_keeps_task_xy_while_normal_coordinate_satisfies_constraint():
    config_path = Path(__file__).resolve().parents[1] / "config" / "bar_clean_true.json"
    config = json.loads(config_path.read_text())
    config["stage_endpoint_positions_bar"][3][2] = 0.20
    config["constraint_terms"] = [
        {
            "feature_name": "table_dist",
            "stage": 3,
            "semantics": "target_value",
            "value": 0.07004,
            "scale": 0.003,
            "weight": 1.5,
        }
    ]
    optimizer = StageConstraintTrajectoryOptimizer(
        config,
        control_spacing=0.06,
        output_spacing=0.02,
        min_control_points=5,
        max_control_points=6,
        max_nfev=120,
        multi_start=1,
    )
    start = np.asarray([0.6183, -0.1613, 0.26584, 0.0, 1.0, 0.0, 0.0])
    goal = np.asarray([0.45, -0.44, 0.26584, 0.0, 1.0, 0.0, 0.0])
    bar = np.asarray([0.60, -0.20, 0.24, 0.0, 0.0, 0.0, 1.0])
    obstacle = np.asarray([0.60, 0.20, 0.24, 0.0, 0.0, 0.0, 1.0])

    planned = optimizer.plan(start, goal, bar, obstacle, seed=11)

    target = planned["stage_endpoint_targets_world"][3]
    achieved = planned["stage_endpoints_world"][3]
    task_frame = build_bar_table_task_frame(
        bar,
        obstacle,
        config["table_surface_point"],
        config["table_normal"],
        config.get("bar_axis_local", [1.0, 0.0, 0.0]),
        optimizer._canonical_bar_axis,
    )
    position_error_task = (
        task_frame["rotation_world_from_task"].T @ (achieved - target)
    )
    table_z = float(config["table_surface_point"][2])
    constraint_target = float(config["constraint_terms"][0]["value"])
    target_violation = abs(target[2] - table_z - constraint_target)
    achieved_violation = abs(achieved[2] - table_z - constraint_target)
    goal_axis = quaternion_to_matrix(goal[3:7])[:, 2]
    goal_yaw = tool_yaw_from_quaternion(goal[3:7], task_frame)

    assert np.allclose(planned["positions"][0], start[:3])
    assert np.allclose(planned["positions"][-1], goal[:3])
    assert np.allclose(planned["tool_axes"][-1], goal_axis)
    assert abs(
        np.arctan2(
            np.sin(planned["tool_yaws"][-1] - goal_yaw),
            np.cos(planned["tool_yaws"][-1] - goal_yaw),
        )
    ) < 1e-9
    assert np.allclose(position_error_task[:2], 0.0, atol=1e-9)
    assert achieved_violation < 0.1 * target_violation
    assert abs(position_error_task[2]) > 0.05
    assert planned["endpoint_report"][3]["position_error"] < 1e-9
    assert planned["endpoint_report"][3]["horizontal_position_error"] < 1e-9
    assert abs(planned["endpoint_report"][3]["vertical_target_delta"]) > 0.05
    assert planned["endpoint_report"][-1]["position_error"] < 1e-9
    assert planned["endpoint_report"][-1]["axis_error"] < 1e-9
    assert planned["endpoint_report"][-1]["yaw_error"] < 1e-9
    assert planned["tool_yaw_active"][0]
    assert planned["tool_yaw_active"][-1]
    assert planned["endpoint_objective"] == 0.0


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


def test_bar_clean_yaw_equality_is_active_only_in_user_stages_two_and_four():
    config_path = Path(__file__).resolve().parents[1] / "config" / "bar_clean_true.json"
    config = json.loads(config_path.read_text())
    terms = [
        value for value in config["constraint_terms"]
        if value["feature_name"] == "tool_yaw"
    ]

    assert [value["stage"] for value in terms] == [1, 3]
    assert all(value["semantics"] == "target_value" for value in terms)
    assert np.isclose(terms[0]["value"], terms[1]["value"])


def test_bar_clean_user_stage_three_is_free_and_stage_four_has_five_equalities():
    config_path = Path(__file__).resolve().parents[1] / "config" / "bar_clean_true.json"
    config = json.loads(config_path.read_text())
    by_stage = {
        stage: [term for term in config["constraint_terms"] if term["stage"] == stage]
        for stage in range(5)
    }

    assert by_stage[2] == []
    stage_four = {term["feature_name"]: term for term in by_stage[3]}
    assert set(stage_four) == {
        "bar_axial_offset",
        "table_dist",
        "tool_pitch",
        "tool_roll",
        "tool_yaw",
    }
    assert all(term["semantics"] == "target_value" for term in stage_four.values())
    assert np.isclose(stage_four["bar_axial_offset"]["value"], -0.023)
    assert np.isclose(stage_four["table_dist"]["value"], 0.07004)
    assert np.isclose(stage_four["tool_pitch"]["value"], np.deg2rad(90.0))
    assert np.isclose(stage_four["tool_roll"]["value"], 0.0)
    stage_two_yaw = next(
        term for term in by_stage[1] if term["feature_name"] == "tool_yaw"
    )
    assert np.isclose(stage_four["tool_yaw"]["value"], stage_two_yaw["value"])


def test_constraint_settling_is_used_only_when_next_stage_adds_constraints():
    config_path = Path(__file__).resolve().parents[1] / "config" / "bar_clean_true.json"
    config = json.loads(config_path.read_text())
    optimizer = StageConstraintTrajectoryOptimizer(config)

    assert optimizer._settling_boundaries.tolist() == [True, False, True, False]


def test_constraint_settling_reserves_progress_without_changing_endpoints():
    config_path = Path(__file__).resolve().parents[1] / "config" / "bar_clean_true.json"
    config = json.loads(config_path.read_text())
    optimizer = StageConstraintTrajectoryOptimizer(config)
    endpoints = np.asarray(
        [
            [0.0, 0.0, 0.1],
            [0.1, 0.0, 0.1],
            [0.4, 0.0, 0.1],
            [0.5, 0.1, 0.1],
            [0.5, -0.2, 0.1],
            [0.6, -0.2, 0.1],
        ]
    )
    positions, _, _, _, endpoint_indices = optimizer._initial_trajectory(
        endpoints,
        [0.0, 0.0, -1.0],
        [0.0, 0.0, -1.0],
        0.0,
        0.0,
    )
    windows = optimizer._settling_windows(endpoints, endpoint_indices)

    assert [window["boundary_index"] for window in windows] == [0, 2]
    for window in windows:
        progress = (
            positions[window["indices"]] - window["origin"][None, :]
        ) @ window["direction"]
        assert np.allclose(progress, window["progress_targets"])
    assert np.allclose(positions[endpoint_indices], endpoints[1:])


def test_learned_endpoint_pose_orientations_initialize_each_stage_boundary():
    config_path = Path(__file__).resolve().parents[1] / "config" / "bar_clean_true.json"
    config = json.loads(config_path.read_text())
    optimizer = StageConstraintTrajectoryOptimizer(config)
    endpoints = np.asarray(
        [
            [0.0, 0.0, 0.2],
            [0.1, 0.0, 0.2],
            [0.2, 0.0, 0.2],
            [0.3, 0.0, 0.2],
            [0.4, 0.0, 0.2],
            [0.5, 0.0, 0.2],
        ]
    )
    endpoint_axes = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.2, 0.0, 0.98],
            [0.4, 0.0, 0.92],
            [0.6, 0.0, 0.8],
            [0.4, 0.0, 0.92],
            [0.0, 0.0, 1.0],
        ]
    )
    endpoint_axes /= np.linalg.norm(endpoint_axes, axis=1, keepdims=True)
    endpoint_yaws = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9, 1.1])

    _, axes, yaws, _, endpoint_indices = optimizer._initial_trajectory(
        endpoints,
        endpoint_axes[0],
        endpoint_axes[-1],
        endpoint_yaws[0],
        endpoint_yaws[-1],
        endpoint_axes=endpoint_axes,
        endpoint_yaws=endpoint_yaws,
    )

    assert np.allclose(axes[endpoint_indices], endpoint_axes[1:])
    assert np.allclose(yaws[endpoint_indices], endpoint_yaws[1:])


def test_tool_yaw_active_mask_follows_loaded_constraint_terms():
    labels = np.asarray([0, 0, 1, 1, 2, 3, 4])
    terms = [
        {"feature_name": "tool_yaw", "stage": 1},
        {"feature_name": "tool_yaw", "stage": 3},
        {"feature_name": "table_dist", "stage": 3},
    ]

    active = StageConstraintTrajectoryOptimizer._feature_active_mask(
        labels, terms, "tool_yaw"
    )
    inactive = StageConstraintTrajectoryOptimizer._feature_active_mask(
        labels, [], "tool_yaw"
    )

    assert active.tolist() == [False, False, True, True, False, True, False]
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
        {"feature_name": "tool_yaw", "stage": 3},
    ]

    active = StageConstraintTrajectoryOptimizer._feature_active_mask(
        labels, terms, "tool_yaw", weights
    )

    assert active.tolist() == [False, False, False, True, False, False, False]
