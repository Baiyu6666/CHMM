#!/usr/bin/env python3
"""Submit one real-task planner request without slow temporary ROS publishers."""

import json
import sys
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import Trigger


TASK_IDS = {"BarInspect", "BarClean"}


def pose_message(values):
    message = PoseStamped()
    message.header.frame_id = "iiwa14_link_0"
    message.pose.position.x = float(values["x"])
    message.pose.position.y = float(values["y"])
    message.pose.position.z = float(values["z"])
    message.pose.orientation.x = float(values["qx"])
    message.pose.orientation.y = float(values["qy"])
    message.pose.orientation.z = float(values["qz"])
    message.pose.orientation.w = float(values["qw"])
    return message


def main():
    if len(sys.argv) != 2:
        raise ValueError("Expected one JSON task argument")
    task = json.loads(sys.argv[1])
    task_id = str(task["task_id"])
    if task_id not in TASK_IDS:
        raise ValueError("Unknown task_id {}".format(task_id))

    rospy.init_node("stage_real_plan_submitter", anonymous=True)
    planner_selected = threading.Event()
    executor_selected = threading.Event()

    def selected(event):
        def callback(message):
            try:
                status = json.loads(message.data)
            except (TypeError, ValueError):
                return
            if str(status.get("task_id", "")) == task_id:
                event.set()
        return callback

    subscribers = [
        rospy.Subscriber(
            "/stage_cons/planner/status", String, selected(planner_selected), queue_size=2
        ),
        rospy.Subscriber(
            "/iiwa14/real_executor/status", String, selected(executor_selected), queue_size=2
        ),
    ]
    task_publisher = rospy.Publisher(
        "/stage_cons/planner/task", String, queue_size=1, latch=True
    )
    start_publisher = rospy.Publisher(
        "/stage_cons/planner/start", PoseStamped, queue_size=1, latch=True
    )
    goal_publisher = rospy.Publisher(
        "/stage_cons/planner/goal", PoseStamped, queue_size=1, latch=True
    )
    rospy.wait_for_service("/stage_constraint_planner/plan", timeout=5.0)
    rospy.set_param(
        "/stage_constraint_planner/constraint_source",
        str(task.get("constraint_source", "true")),
    )

    publishers = (task_publisher, start_publisher, goal_publisher)
    deadline = time.monotonic() + 3.0
    while not rospy.is_shutdown() and any(
        publisher.get_num_connections() < 1 for publisher in publishers
    ):
        if time.monotonic() >= deadline:
            raise RuntimeError("Planner input subscribers did not connect")
        rospy.sleep(0.02)

    for _ in range(3):
        task_publisher.publish(String(data=task_id))
        rospy.sleep(0.02)
    deadline = time.monotonic() + 2.0
    while not rospy.is_shutdown() and not (
        planner_selected.is_set() and executor_selected.is_set()
    ):
        if time.monotonic() >= deadline:
            raise RuntimeError("Planner or real executor did not confirm task selection")
        rospy.sleep(0.02)

    start = pose_message(task["start"])
    goal = pose_message(task["goal"])
    for _ in range(3):
        stamp = rospy.Time.now()
        start.header.stamp = stamp
        goal.header.stamp = stamp
        start_publisher.publish(start)
        goal_publisher.publish(goal)
        rospy.sleep(0.02)

    response = rospy.ServiceProxy("/stage_constraint_planner/plan", Trigger)()
    for subscriber in subscribers:
        subscriber.unregister()
    print(json.dumps(
        {"success": bool(response.success), "message": response.message},
        separators=(",", ":"),
    ))
    return 0 if response.success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps(
            {"success": False, "message": str(error)}, separators=(",", ":")
        ))
        sys.exit(1)
