#!/usr/bin/env python3
"""Publish a Cartesian straight-line Path; deliberately never commands hardware."""

import copy
import math

import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Trigger, TriggerResponse


def normalized_quaternion(quaternion):
    values = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm < 1e-9:
        raise ValueError("pose contains an invalid quaternion")
    return [value / norm for value in values]


def slerp(first, second, amount):
    dot = sum(a * b for a, b in zip(first, second))
    if dot < 0.0:
        second = [-value for value in second]
        dot = -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        values = [(1.0 - amount) * a + amount * b for a, b in zip(first, second)]
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]
    angle = math.acos(dot)
    denominator = math.sin(angle)
    return [
        (math.sin((1.0 - amount) * angle) * a + math.sin(amount * angle) * b)
        / denominator
        for a, b in zip(first, second)
    ]


class StraightLinePlanner:
    def __init__(self):
        self._frame_id = rospy.get_param("~frame_id", "iiwa14_link_0")
        self._spacing = float(rospy.get_param("~sample_spacing", 0.01))
        if self._spacing <= 0.0:
            raise ValueError("~sample_spacing must be positive")
        self._start = None
        self._goal = None
        self._publisher = rospy.Publisher(
            "/stage_cons/plan", Path, queue_size=1, latch=True
        )
        self._start_subscriber = rospy.Subscriber(
            "/stage_cons/planner/start", PoseStamped, self._start_callback, queue_size=1
        )
        self._goal_subscriber = rospy.Subscriber(
            "/stage_cons/planner/goal", PoseStamped, self._goal_callback, queue_size=1
        )
        self._service = rospy.Service("~plan", Trigger, self._plan)

    def _start_callback(self, message):
        self._start = copy.deepcopy(message)

    def _goal_callback(self, message):
        self._goal = copy.deepcopy(message)

    def _plan(self, _request):
        if self._start is None or self._goal is None:
            return TriggerResponse(False, "Publish both start and goal poses first")
        start_frame = self._start.header.frame_id or self._frame_id
        goal_frame = self._goal.header.frame_id or self._frame_id
        if start_frame != goal_frame:
            return TriggerResponse(False, "Start and goal frames differ; no TF is applied")

        start_position = self._start.pose.position
        goal_position = self._goal.pose.position
        delta = [
            goal_position.x - start_position.x,
            goal_position.y - start_position.y,
            goal_position.z - start_position.z,
        ]
        distance = math.sqrt(sum(value * value for value in delta))
        count = max(2, int(math.ceil(distance / self._spacing)) + 1)
        try:
            start_quaternion = normalized_quaternion(self._start.pose.orientation)
            goal_quaternion = normalized_quaternion(self._goal.pose.orientation)
        except ValueError as error:
            return TriggerResponse(False, str(error))

        stamp = rospy.Time.now()
        path = Path()
        path.header.stamp = stamp
        path.header.frame_id = start_frame
        for index in range(count):
            amount = index / float(count - 1)
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = start_frame
            pose.pose.position.x = start_position.x + amount * delta[0]
            pose.pose.position.y = start_position.y + amount * delta[1]
            pose.pose.position.z = start_position.z + amount * delta[2]
            quaternion = slerp(start_quaternion, goal_quaternion, amount)
            pose.pose.orientation.x = quaternion[0]
            pose.pose.orientation.y = quaternion[1]
            pose.pose.orientation.z = quaternion[2]
            pose.pose.orientation.w = quaternion[3]
            path.poses.append(pose)
        self._publisher.publish(path)
        return TriggerResponse(True, "Published {} Cartesian samples".format(count))


if __name__ == "__main__":
    rospy.init_node("straight_line_planner")
    StraightLinePlanner()
    rospy.spin()
