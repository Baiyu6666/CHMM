#!/usr/bin/env python3

import base64
import copy
import json
import math
import os
import threading
from collections import deque

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from stage_constraint_planner import (
    StageConstraintTrajectoryOptimizer,
    configure_task_constraints,
    transform_pose,
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
        self._config_paths = {
            "BarClean": rospy.get_param(
                "~bar_clean_config",
                "/task_definitions/bar_clean_true.json",
            ),
        }
        self._configs = {}
        self._optimizers = {}
        self._learned_constraint_root = os.path.realpath(
            rospy.get_param("~learned_constraint_root", "/learned_constraints")
        )
        for task_id in self._config_paths:
            self._load_task_definition(task_id)

        self._frame_id = rospy.get_param("~frame_id", "iiwa14_link_0")
        self._scene_pose_rotation = np.asarray(
            rospy.get_param(
                "~scene_pose_rotation",
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ),
            dtype=float,
        ).reshape(3, 3)
        self._scene_pose_translation = np.asarray(
            rospy.get_param("~scene_pose_translation", [0.0, 0.0, 0.0]),
            dtype=float,
        ).reshape(3)
        # Validate configuration at startup instead of failing on the first plan.
        transform_pose(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            self._scene_pose_rotation,
            self._scene_pose_translation,
        )
        self._scene_max_age = float(rospy.get_param("~scene_max_age", 1.0))
        self._scene_future_tolerance = float(
            rospy.get_param("~scene_future_tolerance", 0.1)
        )
        self._scene_stability_window = float(
            rospy.get_param("~scene_stability_window", 0.15)
        )
        self._scene_max_speed = float(rospy.get_param("~scene_max_speed", 2.0))
        self._scene_max_jump = float(rospy.get_param("~scene_max_jump", 0.10))
        self._lock = threading.RLock()
        self._planning = False
        self._task_id = str(rospy.get_param("~default_task", "BarClean"))
        if self._task_id not in self._optimizers:
            raise ValueError("Unknown default planner task {}".format(self._task_id))
        self._start = None
        self._goal = None
        self._bar = None
        self._bar_received = 0.0
        self._bar_history = deque(maxlen=256)
        self._obstacle = None
        self._obstacle_received = 0.0
        self._obstacle_history = deque(maxlen=256)

        self._path_publisher = rospy.Publisher(
            "/stage_cons/plan", Path, queue_size=1, latch=True
        )
        self._boundary_publisher = rospy.Publisher(
            "/stage_cons/plan_stage_boundaries",
            Int32MultiArray,
            queue_size=1,
            latch=True,
        )
        self._orientation_constraint_publisher = rospy.Publisher(
            "/stage_cons/plan_orientation_constraints",
            String,
            queue_size=1,
            latch=True,
        )
        self._status_publisher = rospy.Publisher(
            "/stage_cons/planner/status", String, queue_size=1, latch=True
        )
        self._visualization_publisher = rospy.Publisher(
            "/stage_cons/plan_visualization", String, queue_size=1, latch=True
        )
        self._tracking_status_publisher = rospy.Publisher(
            "/stage_cons/planner/tracking_status", String, queue_size=1, latch=True
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
        self._obstacle_subscriber = rospy.Subscriber(
            rospy.get_param(
                "~obstacle_topic",
                "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14",
            ),
            PoseStamped,
            self._obstacle_callback,
            queue_size=2,
        )
        self._service = rospy.Service("~plan", Trigger, self._plan)
        self._tracking_status_timer = rospy.Timer(
            rospy.Duration(0.1), self._publish_tracking_status
        )
        self._publish_status(
            {
                "state": "ready",
                "task_id": self._task_id,
                "available_tasks": sorted(self._optimizers),
            }
        )

    def _load_planning_constraints(self, task_id, config, source):
        configure_task_constraints(
            config,
            task_id,
            source,
            self._learned_constraint_root,
        )

    def _load_task_definition(self, task_id, constraint_source="true"):
        config_path = self._config_paths[task_id]
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        if str(config.get("task_id")) != task_id:
            raise ValueError(
                "Task definition {} declares task_id {!r}, expected {!r}".format(
                    config_path, config.get("task_id"), task_id
                )
            )
        self._load_planning_constraints(task_id, config, constraint_source)
        planner = dict(config["planner"])
        optimizer = StageConstraintTrajectoryOptimizer(
            config,
            control_spacing=float(planner["control_spacing_m"]),
            output_spacing=float(planner["output_spacing_m"]),
            output_axis_spacing=math.radians(float(planner["output_axis_spacing_deg"])),
            min_control_points=int(planner["min_control_points"]),
            max_control_points=int(planner["max_control_points"]),
            max_nfev=int(planner["max_nfev"]),
            multi_start=int(planner["multi_start"]),
        )
        self._configs[task_id] = config
        self._optimizers[task_id] = optimizer
        return config, optimizer

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
        try:
            position = _pose_array(message)[:3]
        except ValueError:
            return
        received = rospy.get_time()
        with self._lock:
            self._bar = copy.deepcopy(message)
            self._bar_received = received
            self._bar_history.append((received, position.tolist()))

    def _obstacle_callback(self, message):
        try:
            position = _pose_array(message)[:3]
        except ValueError:
            return
        received = rospy.get_time()
        with self._lock:
            self._obstacle = copy.deepcopy(message)
            self._obstacle_received = received
            self._obstacle_history.append((received, position.tolist()))

    def _tracking_reason(self, name, message, received, history, now):
        if message is None:
            return "{} pose has not been received".format(name)
        receipt_age = now - received
        if receipt_age < -self._scene_future_tolerance:
            return "{} receipt time is in the future".format(name)
        if self._scene_max_age > 0.0 and receipt_age > self._scene_max_age:
            return "{} pose callback is stale".format(name)
        stamp = message.header.stamp.to_sec()
        if stamp <= 0.0:
            return "{} pose has no source timestamp".format(name)
        source_age = now - stamp
        if source_age < -self._scene_future_tolerance:
            return "{} pose source timestamp is in the future".format(name)
        if self._scene_max_age > 0.0 and source_age > self._scene_max_age:
            return "{} pose source timestamp is stale".format(name)

        recent = [
            (sample_time, np.asarray(position, dtype=float))
            for sample_time, position in history
            if received - sample_time <= self._scene_stability_window
        ]
        for (time_a, position_a), (time_b, position_b) in zip(recent, recent[1:]):
            distance = float(np.linalg.norm(position_b - position_a))
            if distance > self._scene_max_jump:
                return "{} tracking jumped {:.3f} m".format(name, distance)
            elapsed = time_b - time_a
            if elapsed > 1e-4 and distance / elapsed > self._scene_max_speed:
                return "{} tracking velocity is implausible ({:.2f} m/s)".format(
                    name, distance / elapsed
                )
        return None

    def _tracking_reasons(self):
        now = rospy.get_time()
        with self._lock:
            bar = copy.deepcopy(self._bar)
            bar_received = float(self._bar_received)
            bar_history = list(self._bar_history)
            obstacle = copy.deepcopy(self._obstacle)
            obstacle_received = float(self._obstacle_received)
            obstacle_history = list(self._obstacle_history)
        return {
            "bar": self._tracking_reason(
                "bar", bar, bar_received, bar_history, now
            ),
            "obstacle": self._tracking_reason(
                "obstacle",
                obstacle,
                obstacle_received,
                obstacle_history,
                now,
            ),
        }

    def _publish_tracking_status(self, _event):
        reasons = self._tracking_reasons()
        self._tracking_status_publisher.publish(
            String(
                data=json.dumps(
                    {
                        "valid": all(reason is None for reason in reasons.values()),
                        "bar": reasons["bar"],
                        "obstacle": reasons["obstacle"],
                        "stamp": rospy.get_time(),
                    },
                    sort_keys=True,
                )
            )
        )

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
                list(self._bar_history),
                list(self._obstacle_history),
            )
        (
            task_id,
            start,
            goal,
            bar,
            bar_received,
            obstacle,
            obstacle_received,
            bar_history,
            obstacle_history,
        ) = values
        if start is None or goal is None:
            raise ValueError("Publish both start and goal poses first")
        if bar is None or obstacle is None:
            raise ValueError("Current bar and obstacle poses have not been received")
        start_frame = start.header.frame_id or self._frame_id
        goal_frame = goal.header.frame_id or self._frame_id
        if start_frame != goal_frame:
            raise ValueError("Start and goal frames differ; no TF conversion is applied")
        now = rospy.get_time()
        bar_reason = self._tracking_reason(
            "bar", bar, bar_received, bar_history, now
        )
        obstacle_reason = self._tracking_reason(
            "obstacle", obstacle, obstacle_received, obstacle_history, now
        )
        if bar_reason is not None or obstacle_reason is not None:
            raise ValueError(
                "OptiTrack scene is invalid: "
                + "; ".join(
                    reason
                    for reason in (bar_reason, obstacle_reason)
                    if reason is not None
                )
            )
        return task_id, start, goal, bar, obstacle, start_frame

    def _plan(self, _request):
        with self._lock:
            if self._planning:
                return TriggerResponse(False, "A planning request is already active")
            self._planning = True
        try:
            task_id, start, goal, bar, obstacle, frame = self._snapshot()
            # The host task definition is intentionally re-read for every plan.
            # Parameter-only edits therefore take effect on the next click.
            constraint_source = rospy.get_param(
                "/stage_constraint_planner/constraint_source", "true"
            )
            config, optimizer = self._load_task_definition(
                task_id, constraint_source=constraint_source
            )
            bar_pose = transform_pose(
                _pose_array(bar),
                self._scene_pose_rotation,
                self._scene_pose_translation,
            )
            obstacle_pose = transform_pose(
                _pose_array(obstacle),
                self._scene_pose_rotation,
                self._scene_pose_translation,
            )
            self._publish_status({"state": "optimizing", "task_id": task_id})
            planned = optimizer.plan(
                _pose_array(start),
                _pose_array(goal),
                bar_pose,
                obstacle_pose,
                seed=int(rospy.get_param("~seed", 2026)),
            )
            quaternions = planned["tool_quaternions"]
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
            tool_yaw_active = np.asarray(
                planned["tool_yaw_active"], dtype=bool
            )
            if tool_yaw_active.shape != (len(path.poses),):
                raise ValueError(
                    "Planner tool-yaw mask does not match the path length"
                )
            orientation_constraints = {
                "schema_version": 1,
                "stamp_ns": int(stamp.to_nsec()),
                "task_id": task_id,
                "point_count": len(path.poses),
                "tool_yaw_active": tool_yaw_active.astype(int).tolist(),
            }
            positions = np.asarray(planned["positions"], dtype=float)
            distances = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
            )
            nominal_speed = float(config["execution"]["task_speed_mps"])
            if not math.isfinite(nominal_speed) or nominal_speed <= 0.0:
                raise ValueError("~visualization_speed must be positive and finite")
            times = distances / nominal_speed
            feature_names = [str(name) for name in config["visualization_features"]]
            feature_units = dict(config.get("feature_units", {}))
            feature_samples = np.column_stack(
                [times] + [planned["features"][name] for name in feature_names]
            )
            task_frame = planned["task_frame"]
            bar_axis = np.asarray(task_frame["axial"], dtype=float)
            bar_reference = np.asarray(task_frame["bar_reference"], dtype=float)
            frame_origin = np.asarray(task_frame["origin"], dtype=float)
            visualization = {
                "task_id": task_id,
                "bar_axis_flipped": bool(planned["bar_axis_flipped"]),
                "task_frame": {
                    "frame_id": str(task_frame["frame_id"]),
                    "origin": frame_origin.tolist(),
                    "axial": bar_axis.tolist(),
                    "lateral": np.asarray(task_frame["lateral"], dtype=float).tolist(),
                    "normal": np.asarray(task_frame["normal"], dtype=float).tolist(),
                    "snapshot_policy": str(task_frame["snapshot_policy"]),
                },
                "scene_geometry": {
                    "bar": {
                        "pivot": bar_reference[:2].tolist(),
                        "axis": bar_axis[:2].tolist(),
                    },
                    "obstacle": {"center": obstacle_pose[:2].tolist()},
                },
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
                    for term in config["true_constraint_terms"]
                ],
                "planning_constraint_specs": [
                    {
                        "feature_name": str(term["feature_name"]),
                        "stage": int(term["stage"]),
                        "semantics": str(term["semantics"]),
                        "value": float(term["value"]),
                    }
                    for term in config["constraint_terms"]
                ],
                "planning_constraint_source": config["planning_constraint_source"],
                "tool_yaw_active": tool_yaw_active.astype(int).tolist(),
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
            self._orientation_constraint_publisher.publish(
                String(
                    data=json.dumps(
                        orientation_constraints, separators=(",", ":")
                    )
                )
            )
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
                "bar_axis_flipped": bool(planned["bar_axis_flipped"]),
                "points": len(path.poses),
                "stage_boundaries": boundaries.data,
                "stage_transition_windows": planned["stage_transition_windows"],
                "objective": float(planned["objective"]),
                "solver_success": bool(planned["solver_success"]),
                "solver_evaluations": int(planned["solver_evaluations"]),
                "constraint_report": planned["constraint_report"],
                "planning_constraint_source": config["planning_constraint_source"],
            }
            self._publish_status(report)
            rospy.loginfo(
                "Published optimized %d-point, %d-stage %s path",
                len(path.poses),
                len(boundaries.data),
                task_id,
            )
            return TriggerResponse(True, json.dumps(report, separators=(",", ":")))
        except (KeyError, OSError, ValueError, RuntimeError) as error:
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
