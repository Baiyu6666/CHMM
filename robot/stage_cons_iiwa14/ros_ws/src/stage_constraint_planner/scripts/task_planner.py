#!/usr/bin/env python3

import base64
import copy
import json
import math
import threading

import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from stage_constraint_planner import (
    StageConstraintTrajectoryOptimizer,
    continuous_quaternions_from_axes,
)
from std_msgs.msg import Int32MultiArray, String
from std_srvs.srv import Trigger, TriggerResponse


def _pose_array(message):
    pose = message.pose
    values = np.asarray(
        [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.linalg.norm(values[3:]) <= 1e-9:
        raise ValueError("Pose contains invalid position or quaternion values")
    values[3:] /= np.linalg.norm(values[3:])
    return values


class TaskPlannerNode:
    def __init__(self):
        package_path = rospkg.RosPack().get_path("stage_constraint_planner")
        config_paths = {
            "BarInspect": rospy.get_param(
                "~bar_inspect_config",
                package_path + "/config/bar_inspect_true.json",
            ),
            "BarClean": rospy.get_param(
                "~bar_clean_config",
                package_path + "/config/bar_clean_true.json",
            ),
        }
        self._configs = {}
        self._optimizers = {}
        optimizer_kwargs = {
            "control_spacing": float(rospy.get_param("~control_spacing", 0.045)),
            "output_spacing": float(rospy.get_param("~output_spacing", 0.005)),
            "output_axis_spacing": math.radians(
                float(rospy.get_param("~output_axis_spacing_deg", 2.0))
            ),
            "min_control_points": int(rospy.get_param("~min_control_points", 6)),
            "max_control_points": int(rospy.get_param("~max_control_points", 10)),
            "max_nfev": int(rospy.get_param("~max_nfev", 80)),
            "multi_start": int(rospy.get_param("~multi_start", 2)),
        }
        for task_id, config_path in config_paths.items():
            with open(config_path, "r") as handle:
                config = json.load(handle)
            if str(config.get("task_id")) != task_id:
                raise ValueError(
                    "Planner config {} declares task_id {!r}, expected {!r}".format(
                        config_path, config.get("task_id"), task_id
                    )
                )
            self._configs[task_id] = config
            self._optimizers[task_id] = StageConstraintTrajectoryOptimizer(
                config, **optimizer_kwargs
            )

        self._frame_id = rospy.get_param("~frame_id", "iiwa14_link_0")
        self._scene_max_age = float(rospy.get_param("~scene_max_age", 1.0))
        self._lock = threading.RLock()
        self._planning = False
        self._task_id = str(rospy.get_param("~default_task", "BarInspect"))
        if self._task_id not in self._optimizers:
            raise ValueError("Unknown default planner task {}".format(self._task_id))
        self._start = None
        self._goal = None
        self._bar = None
        self._bar_received = 0.0
        self._obstacle = None
        self._obstacle_received = 0.0

        self._path_publisher = rospy.Publisher(
            "/stage_cons/plan", Path, queue_size=1, latch=True
        )
        self._boundary_publisher = rospy.Publisher(
            "/stage_cons/plan_stage_boundaries",
            Int32MultiArray,
            queue_size=1,
            latch=True,
        )
        self._status_publisher = rospy.Publisher(
            "/stage_cons/planner/status", String, queue_size=1, latch=True
        )
        self._visualization_publisher = rospy.Publisher(
            "/stage_cons/plan_visualization", String, queue_size=1, latch=True
        )
        rospy.Subscriber(
            "/stage_cons/planner/task", String, self._task_callback, queue_size=1
        )
        rospy.Subscriber(
            "/stage_cons/planner/start", PoseStamped, self._start_callback, queue_size=1
        )
        rospy.Subscriber(
            "/stage_cons/planner/goal", PoseStamped, self._goal_callback, queue_size=1
        )
        rospy.Subscriber(
            rospy.get_param("~bar_topic", "/vrpn_client_node/baiyu_bar/pose_from_iiwa14"),
            PoseStamped,
            self._bar_callback,
            queue_size=2,
        )
        obstacle_topics = rospy.get_param(
            "~obstacle_topics",
            [
                "/vrpn_client_node/obstacle/pose_from_iiwa14",
                "/vrpn_client_node/baiyu_obs_ball/pose_from_iiwa14",
            ],
        )
        self._obstacle_subscribers = [
            rospy.Subscriber(topic, PoseStamped, self._obstacle_callback, queue_size=2)
            for topic in obstacle_topics
        ]
        self._service = rospy.Service("~plan", Trigger, self._plan)
        self._publish_status(
            {
                "state": "ready",
                "task_id": self._task_id,
                "available_tasks": sorted(self._optimizers),
            }
        )

    def _publish_status(self, payload):
        self._status_publisher.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _task_callback(self, message):
        task_id = str(message.data).strip()
        if task_id not in self._optimizers:
            self._publish_status(
                {
                    "state": "rejected_task",
                    "task_id": task_id,
                    "available_tasks": sorted(self._optimizers),
                }
            )
            rospy.logerr("Unknown planner task %s", task_id)
            return
        with self._lock:
            if self._planning:
                self._publish_status(
                    {
                        "state": "rejected_task_switch",
                        "task_id": self._task_id,
                        "requested_task_id": task_id,
                    }
                )
                rospy.logerr("Cannot switch planner tasks while optimization is active")
                return
            self._task_id = task_id
        self._publish_status({"state": "task_selected", "task_id": task_id})

    def _start_callback(self, message):
        with self._lock:
            self._start = copy.deepcopy(message)

    def _goal_callback(self, message):
        with self._lock:
            self._goal = copy.deepcopy(message)

    def _bar_callback(self, message):
        with self._lock:
            self._bar = copy.deepcopy(message)
            self._bar_received = rospy.get_time()

    def _obstacle_callback(self, message):
        with self._lock:
            self._obstacle = copy.deepcopy(message)
            self._obstacle_received = rospy.get_time()

    def _snapshot(self):
        with self._lock:
            values = (
                self._task_id,
                copy.deepcopy(self._start),
                copy.deepcopy(self._goal),
                copy.deepcopy(self._bar),
                float(self._bar_received),
                copy.deepcopy(self._obstacle),
                float(self._obstacle_received),
            )
        task_id, start, goal, bar, bar_received, obstacle, obstacle_received = values
        if start is None or goal is None:
            raise ValueError("Publish both start and goal poses first")
        if bar is None or obstacle is None:
            raise ValueError("Current bar and obstacle poses have not been received")
        start_frame = start.header.frame_id or self._frame_id
        goal_frame = goal.header.frame_id or self._frame_id
        if start_frame != goal_frame:
            raise ValueError("Start and goal frames differ; no TF conversion is applied")
        now = rospy.get_time()
        if self._scene_max_age > 0.0:
            if now - bar_received > self._scene_max_age:
                raise ValueError("Current bar pose is stale")
            if now - obstacle_received > self._scene_max_age:
                raise ValueError("Current obstacle pose is stale")
        return task_id, start, goal, bar, obstacle, start_frame

    def _plan(self, _request):
        with self._lock:
            if self._planning:
                return TriggerResponse(False, "A planning request is already active")
            self._planning = True
        try:
            task_id, start, goal, bar, obstacle, frame = self._snapshot()
            config = self._configs[task_id]
            optimizer = self._optimizers[task_id]
            self._publish_status({"state": "optimizing", "task_id": task_id})
            planned = optimizer.plan(
                _pose_array(start),
                _pose_array(goal),
                _pose_array(bar),
                _pose_array(obstacle),
                seed=int(rospy.get_param("~seed", 2026)),
            )
            quaternions = continuous_quaternions_from_axes(
                planned["tool_axes"], _pose_array(start)[3:]
            )
            stamp = rospy.Time.now()
            path = Path()
            path.header.stamp = stamp
            path.header.frame_id = frame
            for position, quaternion in zip(planned["positions"], quaternions):
                pose = PoseStamped()
                pose.header.stamp = stamp
                pose.header.frame_id = frame
                pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = position.tolist()
                (
                    pose.pose.orientation.x,
                    pose.pose.orientation.y,
                    pose.pose.orientation.z,
                    pose.pose.orientation.w,
                ) = quaternion.tolist()
                path.poses.append(pose)
            boundaries = Int32MultiArray(
                data=planned["stage_boundaries"].astype(int).tolist()
            )
            positions = np.asarray(planned["positions"], dtype=float)
            distances = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
            )
            nominal_speed = float(rospy.get_param("~visualization_speed", 0.03))
            if not math.isfinite(nominal_speed) or nominal_speed <= 0.0:
                raise ValueError("~visualization_speed must be positive and finite")
            times = distances / nominal_speed
            feature_names = [str(name) for name in config["visualization_features"]]
            feature_units = dict(config.get("feature_units", {}))
            feature_samples = np.column_stack(
                [times] + [planned["features"][name] for name in feature_names]
            )
            visualization = {
                "task_id": task_id,
                "stage_names": [str(name) for name in config["stage_names"]],
                "trace": positions[:, :2].tolist(),
                "feature_names": feature_names,
                "feature_schema": [
                    {"name": name, "unit": str(feature_units.get(name, ""))}
                    for name in feature_names
                ],
                "constraint_specs": [
                    {
                        "feature_name": str(term["feature_name"]),
                        "stage": int(term["stage"]),
                        "semantics": str(term["semantics"]),
                        "value": float(term["value"]),
                    }
                    for term in config["constraint_terms"]
                ],
                "feature_samples": feature_samples.tolist(),
                "stage_boundaries": boundaries.data,
                "stage_boundary_times": [
                    float(times[index]) for index in boundaries.data[:-1]
                ],
                "stage_transition_end_times": [
                    float(times[window["end_index"]])
                    for window in planned["stage_transition_windows"]
                ],
                "nominal_speed": nominal_speed,
            }
            self._boundary_publisher.publish(boundaries)
            self._visualization_publisher.publish(
                String(
                    data=base64.b64encode(
                        json.dumps(visualization, separators=(",", ":")).encode("utf-8")
                    ).decode("ascii")
                )
            )
            self._path_publisher.publish(path)
            report = {
                "state": "published",
                "task_id": task_id,
                "points": len(path.poses),
                "stage_boundaries": boundaries.data,
                "stage_transition_windows": planned["stage_transition_windows"],
                "objective": float(planned["objective"]),
                "solver_success": bool(planned["solver_success"]),
                "solver_evaluations": int(planned["solver_evaluations"]),
                "constraint_report": planned["constraint_report"],
            }
            self._publish_status(report)
            rospy.loginfo(
                "Published optimized %d-point, %d-stage %s path",
                len(path.poses),
                len(boundaries.data),
                task_id,
            )
            return TriggerResponse(True, json.dumps(report, separators=(",", ":")))
        except (KeyError, ValueError, RuntimeError) as error:
            self._publish_status({"state": "failed", "message": str(error)})
            rospy.logerr("Stage constraint planning failed: %s", error)
            return TriggerResponse(False, str(error))
        finally:
            with self._lock:
                self._planning = False


if __name__ == "__main__":
    rospy.init_node("stage_constraint_planner")
    TaskPlannerNode()
    rospy.spin()
