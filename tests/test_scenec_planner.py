import json
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PLANNER_ROOT = (
    ROOT
    / "robot/stage_cons_iiwa14/ros_ws/src/stage_constraint_planner/src"
)
sys.path.insert(0, str(PLANNER_ROOT))

from stage_constraint_planner.optimizer import (  # noqa: E402
    StageConstraintTrajectoryOptimizer,
    bar_lateral_centerline_offset,
    capsule_clearance,
)


@pytest.mark.parametrize(
    ("scene_name", "obstacle_type", "centerline_type"),
    [
        ("sceneA", "circle", "straight"),
        ("sceneB", "circle", "straight"),
        ("sceneC", "capsule", "circular_arc_chord"),
    ],
)
def test_only_scenec_enables_capsule_and_curved_centerline(
    scene_name, obstacle_type, centerline_type
):
    path = (
        ROOT
        / "robot/stage_cons_iiwa14/ros_ws/src/stage_iiwa_sim/config/scenes"
        / f"{scene_name}.json"
    )
    scene = json.loads(path.read_text(encoding="utf-8"))

    assert scene["planning_obstacle"]["type"] == obstacle_type
    assert scene["bar"]["lateral_centerline"]["type"] == centerline_type


def test_scenec_planner_uses_curved_lateral_and_capsule_obstacle_clearance():
    task_path = (
        ROOT
        / "robot/stage_cons_iiwa14/ros_ws/src/stage_constraint_planner/config/bar_clean_true.json"
    )
    scene_path = (
        ROOT
        / "robot/stage_cons_iiwa14/ros_ws/src/stage_iiwa_sim/config/scenes/sceneC.json"
    )
    task = json.loads(task_path.read_text(encoding="utf-8"))
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    planner = dict(task["planner"])
    optimizer = StageConstraintTrajectoryOptimizer(
        task,
        control_spacing=planner["control_spacing_m"],
        output_spacing=planner["output_spacing_m"],
        output_axis_spacing=np.deg2rad(planner["output_axis_spacing_deg"]),
        min_control_points=planner["min_control_points"],
        max_control_points=planner["max_control_points"],
        max_nfev=planner["max_nfev"],
        multi_start=1,
    )
    start = task["gui"]["default_start"]
    goal = task["gui"]["default_goal"]
    start_pose = [start[key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    goal_pose = [goal[key] for key in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    obstacle_endpoints = np.asarray(
        [value["locked_pose_robot"][:3] for value in scene["obstacles"]], dtype=float
    )
    obstacle = {
        "type": "capsule",
        "endpoints": obstacle_endpoints,
        "radius": float(scene["obstacles"][0]["radius"]),
    }

    result = optimizer.plan(
        start_pose,
        goal_pose,
        scene["bar"]["locked_pose_robot"],
        obstacle,
        bar_lateral_centerline=scene["bar"]["lateral_centerline"],
        seed=2026,
    )

    positions = result["positions"]
    normal = np.asarray(task["table_normal"], dtype=float)
    assert np.allclose(
        result["features"]["obs_dist"],
        capsule_clearance(positions, obstacle, normal),
    )
    assert float(
        bar_lateral_centerline_offset(
            0.0, scene["bar"]["lateral_centerline"]
        )
    ) == pytest.approx(0.05)

    task_rotation = np.asarray(
        result["task_frame"]["rotation_world_from_task"], dtype=float
    )
    task_origin = np.asarray(result["task_frame"]["origin"], dtype=float)
    endpoint_targets_task = (
        np.asarray(result["stage_endpoint_targets_world"][:-1]) - task_origin
    ) @ task_rotation
    configured_endpoints = np.asarray(task["stage_endpoint_positions_bar"], dtype=float)
    expected_endpoints = configured_endpoints.copy()
    expected_endpoints[:, 1] += bar_lateral_centerline_offset(
        expected_endpoints[:, 0], scene["bar"]["lateral_centerline"]
    )
    assert np.allclose(endpoint_targets_task, expected_endpoints, atol=1e-9)

    axial = (
        np.asarray(result["features"]["axial_offset"], dtype=float)
        + float(task["bar_axial_offset_reference"])
    )
    lateral = np.asarray(result["features"]["lateral_offset"], dtype=float)
    stage_labels = np.asarray(result["stage_labels"], dtype=int)
    inside_curved_bar = (
        (stage_labels == 1)
        & (axial >= scene["bar"]["lateral_centerline"]["axial_bounds_m"][0])
        & (axial <= scene["bar"]["lateral_centerline"]["axial_bounds_m"][1])
    )
    assert np.sqrt(np.mean(np.square(lateral[inside_curved_bar]))) < 0.003
    assert np.max(np.abs(lateral[inside_curved_bar])) < 0.0085
