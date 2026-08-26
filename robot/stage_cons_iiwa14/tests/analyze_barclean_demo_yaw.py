#!/usr/bin/env python3
"""Estimate executed BarClean Tool-X yaw by stage from a recorded real task."""

import argparse
import csv
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pybullet as bullet


ROBOT_ROOT = Path(__file__).resolve().parents[1]
ROS_SOURCE = ROBOT_ROOT / "ros_ws" / "src"
sys.path.insert(0, str(ROS_SOURCE / "stage_constraint_planner" / "src"))

from stage_constraint_planner.optimizer import (  # noqa: E402
    build_bar_table_task_frame,
)


def bag_csv(container, bag, topic):
    output = subprocess.check_output(
        [
            "docker", "exec", container, "/entrypoint.sh", "rostopic", "echo",
            "-b", bag, "-p", topic,
        ],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=60.0,
    )
    return list(csv.DictReader(io.StringIO(output)))


def pose(row):
    return np.asarray(
        [
            float(row["field.pose.position.x"]),
            float(row["field.pose.position.y"]),
            float(row["field.pose.position.z"]),
            float(row["field.pose.orientation.x"]),
            float(row["field.pose.orientation.y"]),
            float(row["field.pose.orientation.z"]),
            float(row["field.pose.orientation.w"]),
        ]
    )


def robot_description(container):
    encoded = subprocess.check_output(
        [
            "docker", "exec", container, "bash", "-lc",
            "source /opt/ros/noetic/setup.bash && "
            "source /home/ros/ros_ws/devel/setup.bash && "
            "rosparam get /iiwa14/robot_description",
        ],
        text=True,
        stderr=subprocess.DEVNULL,
        timeout=15.0,
    )
    description = subprocess.check_output(
        ["ruby", "-ryaml", "-e", "print YAML.load(STDIN.read)"],
        input=encoded,
        text=True,
    )
    root = ElementTree.fromstring(description)
    package_roots = {path.name: path for path in ROS_SOURCE.iterdir() if path.is_dir()}
    for element in root.iter():
        uri = element.get("filename", "")
        if uri.startswith("package://"):
            package, relative = uri[len("package://"):].split("/", 1)
            element.set("filename", str(package_roots[package] / relative))
    for parent in root.iter():
        for child in list(parent):
            if child.tag in ("gazebo", "transmission", "self_collision_checking"):
                parent.remove(child)
    return ElementTree.tostring(root, encoding="utf-8", xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--container", default="stage_cons_iiwa14")
    arguments = parser.parse_args()
    bag = str(arguments.bag)
    container = arguments.container

    config = json.loads(
        (ROS_SOURCE / "stage_constraint_planner/config/bar_clean_true.json").read_text()
    )
    joint_rows = bag_csv(container, bag, "/iiwa14/joint_states")
    bar_rows = bag_csv(container, bag, "/vrpn_client_node/baiyu_bar/pose_from_iiwa14")
    obstacle_rows = bag_csv(
        container, bag, "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14"
    )
    plan_rows = bag_csv(container, bag, "/stage_cons/plan")
    boundary_rows = bag_csv(container, bag, "/stage_cons/plan_stage_boundaries")
    if len(plan_rows) != 1 or len(boundary_rows) != 1:
        raise RuntimeError("Expected one recorded plan and one boundary message")

    task_frame = build_bar_table_task_frame(
        pose(bar_rows[len(bar_rows) // 2]),
        pose(obstacle_rows[len(obstacle_rows) // 2]),
        config["table_surface_point"],
        config["table_normal"],
        config["bar_axis_local"],
        config["canonical_bar_axis"],
    )
    plan_row = plan_rows[0]
    indices = sorted(
        int(key[len("field.poses"):].split(".", 1)[0])
        for key in plan_row
        if key.startswith("field.poses") and key.endswith(".pose.position.x")
    )
    positions = np.asarray(
        [
            [
                float(plan_row[f"field.poses{index}.pose.position.{axis}"])
                for axis in "xyz"
            ]
            for index in indices
        ]
    )
    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
    )
    boundaries = [
        int(value)
        for key, value in boundary_rows[0].items()
        if key.startswith("field.data") and value not in (None, "")
    ]
    fractions = distances[np.asarray(boundaries, dtype=int)] / distances[-1]

    physics = bullet.connect(bullet.DIRECT)
    handle = tempfile.NamedTemporaryFile(mode="wb", suffix=".urdf", delete=False)
    try:
        handle.write(robot_description(container))
        handle.close()
        robot = bullet.loadURDF(handle.name, useFixedBase=True, physicsClientId=physics)
    finally:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
    joint_indices = []
    tip_index = None
    for index in range(bullet.getNumJoints(robot, physicsClientId=physics)):
        info = bullet.getJointInfo(robot, index, physicsClientId=physics)
        if info[1].decode().startswith("iiwa14_joint_"):
            joint_indices.append(index)
        if info[12].decode() == "iiwa14_link_7":
            tip_index = index

    first_time = int(joint_rows[0]["%time"])
    last_time = int(joint_rows[-1]["%time"])
    yaws_by_stage = {1: [], 2: []}
    normal = np.asarray(task_frame["normal"])
    axial = np.asarray(task_frame["axial"])
    lateral = np.asarray(task_frame["lateral"])
    for row in joint_rows[::10]:
        fraction = (int(row["%time"]) - first_time) / float(last_time - first_time)
        stage = None
        if fractions[0] <= fraction < fractions[1]:
            stage = 1
        elif fractions[1] <= fraction < fractions[2]:
            stage = 2
        if stage is None:
            continue
        q = [float(row[f"field.position{index}"]) for index in range(7)]
        for joint_index, value in zip(joint_indices, q):
            bullet.resetJointState(robot, joint_index, value, physicsClientId=physics)
        state = bullet.getLinkState(
            robot, tip_index, computeForwardKinematics=True, physicsClientId=physics
        )
        rotation = np.asarray(bullet.getMatrixFromQuaternion(state[5])).reshape(3, 3)
        tool_x = rotation[:, 0]
        horizontal = tool_x - normal * float(tool_x @ normal)
        horizontal /= np.linalg.norm(horizontal)
        yaws_by_stage[stage].append(
            math.atan2(float(horizontal @ lateral), float(horizontal @ axial))
        )

    report = {}
    for stage, values in yaws_by_stage.items():
        values = np.unwrap(np.asarray(values))
        report[str(stage)] = {
            "samples": len(values),
            "median_deg": float(np.degrees(np.median(values))),
            "p10_deg": float(np.degrees(np.percentile(values, 10))),
            "p90_deg": float(np.degrees(np.percentile(values, 90))),
        }
    print(json.dumps({"stage_yaw": report, "time_fractions": fractions.tolist()}, indent=2))


if __name__ == "__main__":
    main()
