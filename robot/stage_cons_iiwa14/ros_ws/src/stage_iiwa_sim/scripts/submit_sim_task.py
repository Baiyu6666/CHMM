#!/usr/bin/env python3

import json
import sys
import threading
import time

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger


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
    if task_id not in {"BarInspect", "BarClean"}:
        raise ValueError("Unknown task_id {}".format(task_id))
    rospy.init_node("stage_sim_task_submitter", anonymous=True)
    planner_selected = threading.Event()
    simulator_selected = threading.Event()

    def planner_status_callback(message):
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if str(status.get("task_id", "")) == task_id:
            planner_selected.set()

    def simulator_status_callback(message):
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        if str(status.get("task_id", "")) == task_id:
            simulator_selected.set()

    planner_status_subscriber = rospy.Subscriber(
        "/stage_cons/planner/status", String, planner_status_callback, queue_size=2
    )
    simulator_status_subscriber = rospy.Subscriber(
        "/iiwa14/sim/status", String, simulator_status_callback, queue_size=2
    )
    start_publisher = rospy.Publisher(
        "/stage_cons/planner/start", PoseStamped, queue_size=1, latch=True
    )
    goal_publisher = rospy.Publisher(
        "/stage_cons/planner/goal", PoseStamped, queue_size=1, latch=True
    )
    task_publisher = rospy.Publisher(
        "/stage_cons/planner/task", String, queue_size=1, latch=True
    )
    rospy.wait_for_service("/iiwa14/sim/set_task_recording", timeout=5.0)
    rospy.wait_for_service("/stage_constraint_planner/plan", timeout=5.0)

    connection_deadline = time.monotonic() + 3.0
    while not rospy.is_shutdown() and (
        start_publisher.get_num_connections() < 1
        or goal_publisher.get_num_connections() < 1
        or task_publisher.get_num_connections() < 1
    ):
        if time.monotonic() >= connection_deadline:
            raise RuntimeError("Planner pose subscribers did not connect")
        rospy.sleep(0.02)

    recording_response = rospy.ServiceProxy(
        "/iiwa14/sim/set_task_recording", SetBool
    )(bool(task.get("record", True)))
    if not recording_response.success:
        raise RuntimeError(recording_response.message)

    for _ in range(3):
        task_publisher.publish(String(data=task_id))
        rospy.sleep(0.02)
    confirmation_deadline = time.monotonic() + 3.0
    while not rospy.is_shutdown() and not (
        planner_selected.is_set() and simulator_selected.is_set()
    ):
        if time.monotonic() >= confirmation_deadline:
            missing = []
            if not planner_selected.is_set():
                missing.append("planner")
            if not simulator_selected.is_set():
                missing.append("simulator")
            raise RuntimeError(
                "Task selection was not confirmed by {}".format(" and ".join(missing))
            )
        rospy.sleep(0.02)

    planner_status_subscriber.unregister()
    simulator_status_subscriber.unregister()
    start = pose_message(task["start"])
    goal = pose_message(task["goal"])
    for _ in range(3):
        stamp = rospy.Time.now()
        start.header.stamp = stamp
        goal.header.stamp = stamp
        start_publisher.publish(start)
        goal_publisher.publish(goal)
        rospy.sleep(0.02)

    plan_response = rospy.ServiceProxy("/stage_constraint_planner/plan", Trigger)()
    print(
        json.dumps(
            {"success": bool(plan_response.success), "message": plan_response.message},
            separators=(",", ":"),
        )
    )
    return 0 if plan_response.success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({"success": False, "message": str(error)}, separators=(",", ":")))
        sys.exit(1)
