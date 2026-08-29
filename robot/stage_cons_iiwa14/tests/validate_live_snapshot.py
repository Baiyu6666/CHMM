#!/usr/bin/env python3
"""Offline-compile the currently latched ROS plan and measured joints.

The script only reads ROS topics through ``docker exec``. It never calls a ROS
service, arms the driver, or publishes a command.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pybullet as bullet


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ROBOT_ROOT = PROJECT_ROOT / "robot" / "stage_cons_iiwa14"
ROS_SOURCE = ROBOT_ROOT / "ros_ws" / "src"
sys.path.insert(
    0,
    str(
        ROS_SOURCE
        / "stage_cartesian_trajectory"
        / "src"
    ),
)

from stage_cartesian_trajectory import CartesianTrajectoryCompiler  # noqa: E402


CONTAINER = "stage_cons_iiwa14"
ROS_SETUP = (
    "source /opt/ros/noetic/setup.bash && "
    "source /home/ros/ros_ws/devel/setup.bash && "
)


def container_output(command: str) -> str:
    return subprocess.check_output(
        ["docker", "exec", CONTAINER, "bash", "-lc", ROS_SETUP + command],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=15.0,
    )


def topic_csv(topic: str) -> dict[str, str]:
    output = container_output(f"rostopic echo -p -n 1 {topic}")
    rows = list(csv.DictReader(io.StringIO(output)))
    if len(rows) != 1:
        raise RuntimeError(f"Expected one CSV row from {topic}, got {len(rows)}")
    return rows[0]


def topic_string(topic: str) -> str:
    output = container_output(f"rostopic echo -p -n 1 {topic}")
    lines = [line for line in output.splitlines() if line.strip()]
    if len(lines) != 2 or "," not in lines[1]:
        raise RuntimeError(f"Expected one String row from {topic}")
    return lines[1].split(",", 1)[1]


def indexed_vectors(row: dict[str, str], fields: tuple[str, ...]) -> list[list[float]]:
    pattern = re.compile(r"field\.poses(\d+)\.pose\.position\.x$")
    indices = sorted(
        int(match.group(1))
        for key in row
        if (match := pattern.match(key)) is not None
    )
    return [
        [float(row[f"field.poses{index}.pose.{field}"]) for field in fields]
        for index in indices
    ]


def resolved_robot_description() -> bytes:
    encoded = container_output("rosparam get /iiwa14/robot_description")
    description = subprocess.check_output(
        ["ruby", "-ryaml", "-e", "print YAML.load(STDIN.read)"],
        input=encoded,
        text=True,
    )
    root = ElementTree.fromstring(
        description
    )
    package_roots = {
        path.name: path
        for path in ROS_SOURCE.iterdir()
        if path.is_dir()
    }
    for element in root.iter():
        uri = element.get("filename", "")
        if not uri.startswith("package://"):
            continue
        package, relative = uri[len("package://") :].split("/", 1)
        element.set("filename", str(package_roots[package] / relative))
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ("gazebo", "transmission", "self_collision_checking"):
                parent.remove(child)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def main() -> None:
    global CONTAINER
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", default=CONTAINER)
    parser.add_argument(
        "--tool-z-only",
        action="store_true",
        help="diagnose the spin selected by the legacy Tool-Z-only IK search",
    )
    parser.add_argument(
        "--home",
        action="store_true",
        help="compile the external Robot Home pose instead of the latched planner path",
    )
    parser.add_argument(
        "--minimum-approach-z",
        type=float,
        default=0.20,
        help="offline TCP transit floor in metres",
    )
    parser.add_argument(
        "--home-obstacle-clearance",
        type=float,
        help="offline-only override for the Home obstacle clearance in metres",
    )
    parser.add_argument(
        "--start-joints",
        nargs=7,
        type=float,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="offline start joints in radians; avoids reading /iiwa14/joint_states",
    )
    parser.add_argument(
        "--first-yaw-active-index",
        type=int,
        help=(
            "offline diagnostic override for a latched path produced before "
            "transition-aware yaw masks"
        ),
    )
    parser.add_argument(
        "--yaw-offset-deg",
        type=float,
        default=0.0,
        help="offline diagnostic rotation applied to every planned Tool-X axis",
    )
    arguments = parser.parse_args()
    CONTAINER = arguments.container
    if arguments.start_joints is None:
        joint_row = topic_csv("/iiwa14/joint_states")
        start_q = np.asarray(
            [float(joint_row[f"field.position{index}"]) for index in range(7)],
            dtype=float,
        )
    else:
        start_q = np.asarray(arguments.start_joints, dtype=float)
    if arguments.home:
        home_path = (
            ROBOT_ROOT
            / "ros_ws"
            / "src"
            / "stage_constraint_planner"
            / "config"
            / "robot_home.json"
        )
        definition = json.loads(home_path.read_text(encoding="utf-8"))
        home_execution = definition["execution"]
        home_joints = np.asarray(definition["joint_position_reference"], dtype=float)
        pose = definition["pose"]
        position = [pose[key] for key in ("x", "y", "z")]
        quaternion = [pose[key] for key in ("qx", "qy", "qz", "qw")]
        positions = np.repeat(np.asarray(position, dtype=float)[None, :], 2, axis=0)
        quaternions = [quaternion, quaternion]
        task_definition = json.loads(
            (
                ROS_SOURCE
                / "stage_constraint_planner"
                / "config"
                / "bar_clean_true.json"
            ).read_text(encoding="utf-8")
        )
        scene_definition = json.loads(
            (
                ROS_SOURCE
                / "stage_iiwa_sim"
                / "config"
                / "demo_scene.json"
            ).read_text(encoding="utf-8")
        )
        clearance = next(
            float(term["value"])
            for term in task_definition["constraint_terms"]
            if term["feature_name"] == "obstacle_clearance"
            and int(term["stage"]) == 0
            and term["semantics"] == "lower_bound"
        )
        if arguments.home_obstacle_clearance is not None:
            clearance = float(arguments.home_obstacle_clearance)
            if not math.isfinite(clearance) or clearance < 0.0:
                raise ValueError("Home obstacle clearance override must be non-negative")
        scene_obstacles = {
            str(obstacle["name"]): obstacle
            for obstacle in scene_definition["obstacles"]
        }
        planning_obstacle = scene_definition["planning_obstacle"]
        if planning_obstacle["type"] == "circle":
            geometry = scene_obstacles[str(planning_obstacle["obstacle"])]
            center = np.asarray(geometry["locked_pose_robot"][:3], dtype=float)
            radius = float(geometry["radius"])
            approach_geometry = {"type": "circle", "center": center.tolist()}
        elif planning_obstacle["type"] == "capsule":
            endpoint_geometry = [
                scene_obstacles[str(name)]
                for name in planning_obstacle["endpoint_obstacles"]
            ]
            endpoints = np.asarray(
                [value["locked_pose_robot"][:3] for value in endpoint_geometry],
                dtype=float,
            )
            radius = float(endpoint_geometry[0]["radius"])
            approach_geometry = {
                "type": "capsule",
                "endpoints": endpoints.tolist(),
            }
        else:
            raise ValueError("Scene planning_obstacle must be circle or capsule")
        approach_obstacle = {
            **approach_geometry,
            "table_normal": task_definition["table_normal"],
            "radius": radius,
            "clearance": clearance,
            "margin": 0.0,
        }
    else:
        plan_row = topic_csv("/stage_cons/plan")
        positions = np.asarray(
            indexed_vectors(
                plan_row,
                ("position.x", "position.y", "position.z"),
            ),
            dtype=float,
        )
        quaternions = indexed_vectors(
            plan_row,
            (
                "orientation.x",
                "orientation.y",
                "orientation.z",
                "orientation.w",
            ),
        )
        orientation_metadata = json.loads(
            topic_string("/stage_cons/plan_orientation_constraints")
        )
        tool_yaw_active = np.asarray(
            orientation_metadata["tool_yaw_active"], dtype=bool
        )
        approach_obstacle = orientation_metadata["approach_obstacle"]
        stage_timing = orientation_metadata["stage_timing"]
        if tool_yaw_active.shape != (len(positions),):
            raise RuntimeError(
                "Planner yaw-mask metadata does not match the latched path"
            )
        if arguments.first_yaw_active_index is not None:
            index = int(arguments.first_yaw_active_index)
            if index < 0 or index >= len(tool_yaw_active):
                raise RuntimeError("Yaw activation override is outside the path")
            first_active = np.flatnonzero(tool_yaw_active)
            if len(first_active) == 0 or index < int(first_active[0]):
                raise RuntimeError("Yaw activation override cannot move earlier")
            tool_yaw_active[int(first_active[0]):index] = False

    physics = bullet.connect(bullet.DIRECT)
    handle = tempfile.NamedTemporaryFile(mode="wb", suffix=".urdf", delete=False)
    try:
        handle.write(resolved_robot_description())
        handle.close()
        robot = bullet.loadURDF(
            handle.name,
            useFixedBase=True,
            flags=(
                bullet.URDF_USE_INERTIA_FROM_FILE
                | bullet.URDF_MAINTAIN_LINK_ORDER
                | bullet.URDF_USE_SELF_COLLISION
                | bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
            ),
            physicsClientId=physics,
        )
    finally:
        try:
            os.unlink(handle.name)
        except OSError:
            pass

    by_name = {}
    tip_index = None
    for index in range(bullet.getNumJoints(robot, physicsClientId=physics)):
        info = bullet.getJointInfo(robot, index, physicsClientId=physics)
        by_name[info[1].decode()] = (index, info)
        if info[12].decode() == "iiwa14_link_7":
            tip_index = index
    joint_names = [f"iiwa14_joint_{index}" for index in range(1, 8)]
    joint_info = [by_name[name] for name in joint_names]
    joint_indices = [value[0] for value in joint_info]
    lower = [float(value[1][8]) for value in joint_info]
    upper = [float(value[1][9]) for value in joint_info]
    velocity = [float(value[1][11]) for value in joint_info]
    task_definition = json.loads(
        (
            ROS_SOURCE
            / "stage_constraint_planner"
            / "config"
            / "bar_clean_true.json"
        ).read_text(encoding="utf-8")
    )
    execution = task_definition["execution"]
    if arguments.home:
        execution = home_execution
    compiler = CartesianTrajectoryCompiler(
        bullet,
        physics,
        robot,
        joint_indices,
        tip_index,
        lower,
        upper,
        velocity,
        max_joint_step=0.15,
        velocity_scale=0.25,
        acceleration_limit=1.00,
        approach_speed=float(execution["approach_speed_mps"]),
        task_speed=float(
            execution.get(
                "task_speed_mps",
                task_definition["execution"]["task_speed_mps"],
            )
        ),
        position_tolerance=0.003 if arguments.tool_z_only else 0.002,
        approach_position_tolerance=float(
            execution.get("approach_position_tolerance_m", 0.01)
        ),
        approach_joint_bridge_limit=float(
            execution.get("approach_joint_bridge_limit_rad", 3.0)
        ),
        minimum_approach_z=arguments.minimum_approach_z,
        approach_clearance_z=0.33,
    )
    bases = [compiler.tool_basis_from_quaternion(value) for value in quaternions]
    x_axes = np.asarray([value[0] for value in bases])
    axes = np.asarray([value[1] for value in bases])
    if arguments.yaw_offset_deg:
        angle = math.radians(arguments.yaw_offset_deg)
        x_axes = (
            math.cos(angle) * x_axes
            + math.sin(angle) * np.cross(axes, x_axes)
        )
    if arguments.home:
        result = compiler.compile_joint_home(
            start_q,
            home_joints,
            approach_obstacle=approach_obstacle,
        )
    else:
        result = compiler.compile(
            positions,
            axes,
            start_q,
            tool_x_axes=None if arguments.tool_z_only else x_axes,
            tool_x_active=None if arguments.tool_z_only else tool_yaw_active,
            approach_obstacle=approach_obstacle,
            stage_timing=stage_timing,
        )
    selected_spin = None
    if arguments.tool_z_only:
        actual_x = np.asarray(
            [compiler.tip_state(q)[1][:, 0] for q in result["task"]["position"]]
        )
        spin = np.arctan2(
            np.einsum("ij,ij->i", axes, np.cross(x_axes, actual_x)),
            np.einsum("ij,ij->i", x_axes, actual_x),
        )
        selected_spin = {
            "minimum_deg": float(np.degrees(np.min(spin))),
            "median_deg": float(np.degrees(np.median(spin))),
            "maximum_deg": float(np.degrees(np.max(spin))),
        }
    print(
        json.dumps(
            {
                "mode": "home" if arguments.home else "planner_path",
                "start_joints_rad": start_q.tolist(),
                "prepared_task_start_joints_rad": (
                    result["approach"]["position"][-1].tolist()
                ),
                "metrics": result["metrics"],
                "selected_spin_from_planned_tool_x": selected_spin,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
