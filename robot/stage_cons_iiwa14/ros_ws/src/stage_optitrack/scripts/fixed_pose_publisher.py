#!/usr/bin/env python3
"""Publish fixed robot-frame scene poses on the legacy relative-pose topics."""

import json
import math

import rospy
from geometry_msgs.msg import PoseStamped


def _matrix_multiply(left, right):
    return [
        [sum(left[row][k] * right[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]


def _matrix_vector(matrix, vector):
    return [sum(matrix[row][k] * vector[k] for k in range(3)) for row in range(3)]


def _transpose(matrix):
    return [[matrix[column][row] for column in range(3)] for row in range(3)]


def _quaternion_to_matrix(quaternion):
    x, y, z, w = quaternion
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("orientation_robot_xyzw must be a finite nonzero quaternion")
    x, y, z, w = (value / norm for value in (x, y, z, w))
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def _matrix_to_quaternion(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
            0.25 * scale,
        ]
    else:
        index = max(range(3), key=lambda item: matrix[item][item])
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
            quaternion = [
                0.25 * scale,
                (matrix[0][1] + matrix[1][0]) / scale,
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[2][1] - matrix[1][2]) / scale,
            ]
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
            quaternion = [
                (matrix[0][1] + matrix[1][0]) / scale,
                0.25 * scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                (matrix[0][2] - matrix[2][0]) / scale,
            ]
        else:
            scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
            quaternion = [
                (matrix[0][2] + matrix[2][0]) / scale,
                (matrix[1][2] + matrix[2][1]) / scale,
                0.25 * scale,
                (matrix[1][0] - matrix[0][1]) / scale,
            ]
    norm = math.sqrt(sum(value * value for value in quaternion))
    return [value / norm for value in quaternion]


class FixedPosePublisher:
    def __init__(self):
        self._base_name = str(rospy.get_param("~base_name", "iiwa14"))
        self._object_name = str(rospy.get_param("~object_name", "baiyu_bar"))
        self._object_key = str(rospy.get_param("~object_key"))
        self._object_index = int(rospy.get_param("~object_index", 0))
        self._output_frame = str(rospy.get_param("~output_frame", "base"))
        config_path = str(
            rospy.get_param("~scene_config", "/workcell_definition/demo_scene.json")
        )
        with open(config_path, "r", encoding="utf-8") as stream:
            scene = json.load(stream)
        if self._object_key not in {"bar", "obstacles"}:
            raise ValueError("object_key must be bar or obstacles")
        object_config = scene[self._object_key]
        if self._object_key == "obstacles":
            object_config = object_config[self._object_index]
        pose_robot = [float(value) for value in object_config["locked_pose_robot"]]
        self._rate_hz = float(scene.get("fixed_pose_publish_rate", 20.0))
        if self._rate_hz <= 0.0:
            raise ValueError("publish_rate must be positive")

        if len(pose_robot) != 7:
            raise ValueError("locked_pose_robot must contain xyz+xyzw")
        position_robot = pose_robot[:3]
        orientation_robot = pose_robot[3:]
        if not all(math.isfinite(value) for value in position_robot + orientation_robot):
            raise ValueError("Fixed pose values must be finite")

        robot_from_output = scene["optitrack_to_robot"]["rotation"]
        robot_from_output = [[float(value) for value in row] for row in robot_from_output]
        if len(robot_from_output) != 3 or any(len(row) != 3 for row in robot_from_output):
            raise ValueError("robot_from_output_rotation must be 3x3")
        for row in range(3):
            for column in range(3):
                dot = sum(
                    robot_from_output[row][index]
                    * robot_from_output[column][index]
                    for index in range(3)
                )
                expected = 1.0 if row == column else 0.0
                if not math.isclose(dot, expected, abs_tol=1e-9):
                    raise ValueError("optitrack_to_robot rotation must be orthonormal")
        translation = [
            float(value)
            for value in scene["optitrack_to_robot"].get(
                "translation", [0.0, 0.0, 0.0]
            )
        ]
        if len(translation) != 3 or not all(math.isfinite(value) for value in translation):
            raise ValueError("optitrack_to_robot translation must contain three finite values")
        output_from_robot = _transpose(robot_from_output)
        position_output = _matrix_vector(
            output_from_robot,
            [position_robot[index] - translation[index] for index in range(3)],
        )
        rotation_output = _matrix_multiply(
            output_from_robot, _quaternion_to_matrix(orientation_robot)
        )
        orientation_output = _matrix_to_quaternion(rotation_output)

        self._message = PoseStamped()
        self._message.header.frame_id = self._output_frame
        self._message.pose.position.x, self._message.pose.position.y, self._message.pose.position.z = position_output
        (
            self._message.pose.orientation.x,
            self._message.pose.orientation.y,
            self._message.pose.orientation.z,
            self._message.pose.orientation.w,
        ) = orientation_output
        topic = "/vrpn_client_node/{}/pose_from_{}".format(
            self._object_name, self._base_name
        )
        self._publisher = rospy.Publisher(topic, PoseStamped, queue_size=1, latch=True)
        rospy.loginfo(
            "Fixed scene pose %s in robot frame: p=%s q_xyzw=%s -> %s",
            self._object_name,
            position_robot,
            orientation_robot,
            topic,
        )

    def run(self):
        rate = rospy.Rate(self._rate_hz)
        while not rospy.is_shutdown():
            self._message.header.stamp = rospy.Time.now()
            self._publisher.publish(self._message)
            rate.sleep()


if __name__ == "__main__":
    rospy.init_node("fixed_scene_pose_publisher")
    FixedPosePublisher().run()
