#!/usr/bin/env python3
"""Prepare and explicitly execute a validated iiwa14 position trajectory.

Receiving /stage_cons/plan never commands hardware. A caller must first invoke
~prepare and then ~execute while FRI reports POSITION + COMMANDING_ACTIVE.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import signal
import subprocess
import tempfile
import threading
import time
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import actionlib
import numpy as np
import pybullet as bullet
import rospkg
import rospy
from actionlib_msgs.msg import GoalStatus
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    JointTrajectoryControllerState,
)
from iiwa_driver.msg import AdditionalOutputs
from nav_msgs.msg import Path as RosPath
from sensor_msgs.msg import JointState
from stage_cartesian_trajectory import (
    CartesianTrajectoryCompiler,
    TrajectoryValidationError,
)
from std_msgs.msg import Bool, Empty, Int32, String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse
from trajectory_msgs.msg import JointTrajectoryPoint


ValidationError = TrajectoryValidationError


class RealExecutor:
    _TASK_IDS = {"BarInspect", "BarClean"}

    def __init__(self):
        self._lock = threading.RLock()
        self._joint_names = ["iiwa14_joint_{}".format(index) for index in range(1, 8)]
        self._base_frame = rospy.get_param("~base_frame", "iiwa14_link_0")
        self._tip_link = rospy.get_param("~tip_link", "iiwa14_link_7")
        self._action_name = rospy.get_param(
            "~controller_action",
            "/iiwa14/PositionTrajectoryController/follow_joint_trajectory",
        )
        self._output_root = Path(rospy.get_param("~output_root", "/data/demos"))
        self._position_mode = int(rospy.get_param("~position_command_mode", 1))
        self._joint_state_timeout = float(rospy.get_param("~joint_state_timeout", 0.5))
        self._fri_status_timeout = float(rospy.get_param("~fri_status_timeout", 0.5))
        self._start_drift_limit = float(rospy.get_param("~start_drift_limit_rad", math.radians(0.5)))
        self._task_start_settle_tolerance = float(
            rospy.get_param("~task_start_settle_tolerance_rad", math.radians(1.0))
        )
        self._task_start_settle_timeout = float(
            rospy.get_param("~task_start_settle_timeout", 2.0)
        )
        self._task_start_settle_samples = int(
            rospy.get_param("~task_start_settle_samples", 3)
        )
        if (
            not math.isfinite(self._task_start_settle_tolerance)
            or self._task_start_settle_tolerance <= 0.0
            or not math.isfinite(self._task_start_settle_timeout)
            or self._task_start_settle_timeout <= 0.0
            or self._task_start_settle_samples < 1
        ):
            raise ValueError("Task-start settling parameters must be positive")
        self._max_joint_step = float(rospy.get_param("~max_joint_step_rad", 0.15))
        self._velocity_scale = float(rospy.get_param("~velocity_scale", 0.20))
        self._acceleration_limit = float(rospy.get_param("~acceleration_limit_rad_s2", 1.00))
        task_definition_dir = rospy.get_param("~task_definition_dir", "/task_definitions")
        self._home_config_path = rospy.get_param(
            "~home_config", os.path.join(task_definition_dir, "robot_home.json")
        )
        self._task_config_paths = {
            "BarInspect": os.path.join(task_definition_dir, "bar_inspect_true.json"),
            "BarClean": os.path.join(task_definition_dir, "bar_clean_true.json"),
        }
        with open(self._task_config_paths["BarInspect"], "r", encoding="utf-8") as stream:
            initial_execution = json.load(stream)["execution"]
        self._approach_speed = float(initial_execution["approach_speed_mps"])
        self._task_speed = float(initial_execution["task_speed_mps"])
        self._approach_position_tolerance = float(
            initial_execution["approach_position_tolerance_m"]
        )
        self._approach_joint_bridge_limit = float(
            initial_execution["approach_joint_bridge_limit_rad"]
        )
        self._minimum_approach_z = float(rospy.get_param("~minimum_approach_z", 0.20))
        self._approach_clearance_z = float(rospy.get_param("~approach_clearance_z", 0.33))
        self._torque_thresholds = np.asarray(
            rospy.get_param("~external_torque_thresholds_nm", [20, 20, 15, 15, 8, 8, 8]),
            dtype=float,
        )
        if self._torque_thresholds.shape != (7,):
            raise ValueError("~external_torque_thresholds_nm must contain seven values")
        self._torque_timeout = float(
            rospy.get_param("~external_torque_timeout", 0.25)
        )
        if not math.isfinite(self._torque_timeout) or self._torque_timeout <= 0.0:
            raise ValueError("~external_torque_timeout must be positive and finite")

        self._path = None
        self._path_tool_yaw_active = None
        self._path_approach_obstacle = None
        self._path_stage_timing = None
        self._pending_path = None
        self._orientation_constraints = {}
        self._operation = "idle"
        self._task_id = "BarInspect"
        self._constraint_source = "true"
        self._path_serial = 0
        self._joint_position = None
        self._joint_received = 0.0
        self._controller_desired_position = None
        self._controller_state_received = 0.0
        self._commanding = False
        self._commanding_received = 0.0
        self._fri_mode = 0
        self._fri_mode_received = 0.0
        self._prepared = None
        self._worker = None
        self._abort = threading.Event()
        self._record_requested = True
        self._record_process = None
        self._run_directory = None
        self._torque_trip_count = 0
        self._torque_received = 0.0
        self._protective_stop = False
        self._position_armed = False
        self._holding_final_position = False
        self._execution_active = False
        self._execution_failure_reason = None

        self._client = actionlib.SimpleActionClient(
            self._action_name, FollowJointTrajectoryAction
        )
        self._position_gate = rospy.ServiceProxy(
            "/iiwa14/iiwa_driver/set_position_commanding", SetBool
        )
        self._position_heartbeat_pub = rospy.Publisher(
            "/iiwa14/iiwa_driver/position_command_heartbeat", Empty, queue_size=1
        )
        self._fri_ready_pub = rospy.Publisher(
            "~fri_ready_status", Bool, queue_size=1, latch=True
        )
        self._fri_ready_pub.publish(Bool(data=False))
        self._position_heartbeat_timer = rospy.Timer(
            rospy.Duration(0.05), self._publish_position_heartbeat
        )
        self._fri_ready_timer = rospy.Timer(
            rospy.Duration(0.1), self._publish_fri_ready_status
        )
        self._status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        self._last_status_message = None
        self._status_heartbeat_timer = rospy.Timer(
            rospy.Duration(0.25), self._publish_status_heartbeat
        )
        rospy.Subscriber(
            "/stage_cons/planner/task", String, self._task_callback, queue_size=1
        )
        rospy.Subscriber(
            "/stage_cons/plan_orientation_constraints",
            String,
            self._orientation_constraints_callback,
            queue_size=1,
        )
        rospy.Subscriber("/stage_cons/plan", RosPath, self._path_callback, queue_size=1)
        rospy.Subscriber("joint_states", JointState, self._joint_callback, queue_size=5)
        rospy.Subscriber(
            "PositionTrajectoryController/state",
            JointTrajectoryControllerState,
            self._controller_state_callback,
            queue_size=5,
        )
        rospy.Subscriber("commanding_status", Bool, self._commanding_callback, queue_size=2)
        rospy.Subscriber("fri_command_mode", Int32, self._mode_callback, queue_size=2)
        rospy.Subscriber("additional_outputs", AdditionalOutputs, self._torque_callback, queue_size=5)
        rospy.Service("~prepare", Trigger, self._prepare)
        rospy.Service("~validate", Trigger, self._validate_only)
        rospy.Service("~execute", Trigger, self._execute)
        rospy.Service("~return_home", Trigger, self._return_home)
        rospy.Service("~abort", Trigger, self._abort_execution)
        rospy.Service("~set_recording", SetBool, self._set_recording)

        self._physics = bullet.connect(bullet.DIRECT)
        if self._physics < 0:
            raise RuntimeError("Could not start the PyBullet kinematics client")
        self._robot = self._load_robot()
        self._joint_indices, self._lower, self._upper, self._velocity_limits = self._joint_model()
        self._tip_index = self._find_link(self._tip_link)
        self._trajectory_compiler = CartesianTrajectoryCompiler(
            bullet,
            self._physics,
            self._robot,
            self._joint_indices,
            self._tip_index,
            self._lower,
            self._upper,
            self._velocity_limits,
            max_joint_step=self._max_joint_step,
            velocity_scale=self._velocity_scale,
            acceleration_limit=self._acceleration_limit,
            approach_speed=self._approach_speed,
            task_speed=self._task_speed,
            approach_position_tolerance=self._approach_position_tolerance,
            approach_joint_bridge_limit=self._approach_joint_bridge_limit,
            minimum_approach_z=self._minimum_approach_z,
            approach_clearance_z=self._approach_clearance_z,
        )
        self._publish("idle", message="Real executor ready; no trajectory prepared")
        rospy.on_shutdown(self._shutdown)

    def _publish(self, phase, **fields):
        with self._lock:
            payload = {
                "phase": phase,
                "operation": self._operation,
                "task_id": self._task_id,
                "constraint_source": self._constraint_source,
                "path_serial": self._path_serial,
                "stamp": rospy.Time.now().to_sec(),
                "record": self._record_requested,
                "run_directory": str(self._run_directory) if self._run_directory else None,
                "protective_stop": self._protective_stop,
                "holding_final_position": self._holding_final_position,
                "execution_active": self._execution_active,
                "fri_failure_latched": self._execution_failure_reason,
            }
            payload.update(fields)
            message = String(data=json.dumps(payload, sort_keys=True))
            self._last_status_message = message
        self._status_pub.publish(message)

    def _publish_status_heartbeat(self, _event):
        with self._lock:
            message = self._last_status_message
        if message is not None:
            self._status_pub.publish(message)

    def _publish_position_heartbeat(self, _event):
        with self._lock:
            position_armed = self._position_armed
        if position_armed:
            self._position_heartbeat_pub.publish(Empty())

    def _publish_fri_ready_status(self, _event):
        reason = self._fri_failure_reason()
        self._fri_ready_pub.publish(Bool(data=reason is None))
        with self._lock:
            execution_active = self._execution_active
            holding_final_position = self._holding_final_position
        if reason is not None and execution_active:
            self._latch_execution_failure(reason)
        elif reason is not None and holding_final_position:
            self._release_final_hold(
                "FRI readiness was lost; final position hold released ({})".format(reason)
            )

    @staticmethod
    def _parse_orientation_constraints(message):
        try:
            payload = json.loads(message.data)
            if int(payload.get("schema_version", 0)) != 3:
                raise ValueError("unsupported schema_version")
            stamp_ns = int(payload["stamp_ns"])
            point_count = int(payload["point_count"])
            task_id = str(payload["task_id"])
            active = np.asarray(payload["tool_yaw_active"], dtype=int)
            raw_obstacle = payload["approach_obstacle"]
            center = np.asarray(raw_obstacle["center"], dtype=float)
            table_normal = np.asarray(raw_obstacle["table_normal"], dtype=float)
            radius = float(raw_obstacle["radius"])
            clearance = float(raw_obstacle["clearance"])
            margin = float(raw_obstacle["margin"])
            raw_timing = payload["stage_timing"]
            boundaries = np.asarray(raw_timing["boundaries"], dtype=int)
            transition_windows = np.asarray(
                raw_timing["transition_windows"], dtype=int
            )
            if transition_windows.size == 0:
                transition_windows = np.empty((0, 2), dtype=int)
            speed_scale = float(raw_timing["speed_scale"])
            ramp_before_m = float(raw_timing["ramp_before_m"])
            task_start_ramp_m = float(raw_timing["task_start_ramp_m"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Invalid planner path metadata: {}".format(error)
            ) from error
        if stamp_ns <= 0 or point_count < 2 or active.shape != (point_count,):
            raise ValueError(
                "Planner orientation-constraint metadata has invalid dimensions"
            )
        if np.any((active != 0) & (active != 1)):
            raise ValueError("Planner tool-yaw mask must contain only 0 or 1")
        if (
            center.shape != (3,)
            or table_normal.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(table_normal))
            or np.linalg.norm(table_normal) <= 1e-12
            or not all(math.isfinite(value) for value in (radius, clearance, margin))
            or radius <= 0.0
            or clearance < 0.0
            or margin < 0.0
        ):
            raise ValueError("Planner Stage-0 obstacle metadata is invalid")
        approach_obstacle = {
            "center": center.tolist(),
            "table_normal": table_normal.tolist(),
            "radius": radius,
            "clearance": clearance,
            "margin": margin,
        }
        if (
            boundaries.ndim != 1
            or len(boundaries) < 1
            or boundaries[0] <= 0
            or boundaries[-1] != point_count - 1
            or np.any(np.diff(boundaries) <= 0)
            or transition_windows.shape != (max(len(boundaries) - 1, 0), 2)
        ):
            raise ValueError("Planner stage-timing metadata has invalid dimensions")
        for index, window in enumerate(transition_windows):
            if (
                int(window[0]) != int(boundaries[index])
                or int(window[1]) < int(window[0])
                or int(window[1]) > int(boundaries[index + 1])
            ):
                raise ValueError(
                    "Planner stage transition window {} is invalid".format(index)
                )
        if (
            not all(
                math.isfinite(value)
                for value in (speed_scale, ramp_before_m, task_start_ramp_m)
            )
            or not 0.0 < speed_scale <= 1.0
            or ramp_before_m < 0.0
            or task_start_ramp_m < 0.0
        ):
            raise ValueError("Planner stage-timing values are invalid")
        stage_timing = {
            "boundaries": boundaries.tolist(),
            "transition_windows": transition_windows.tolist(),
            "speed_scale": speed_scale,
            "ramp_before_m": ramp_before_m,
            "task_start_ramp_m": task_start_ramp_m,
        }
        return (
            stamp_ns,
            task_id,
            active.astype(bool),
            approach_obstacle,
            stage_timing,
        )

    def _accept_path(
        self,
        message,
        task_id,
        tool_yaw_active,
        approach_obstacle,
        stage_timing,
    ):
        with self._lock:
            if self._execution_active or self._worker is not None:
                rospy.logerr("Ignoring a new planner path during real execution")
                return
            if task_id != self._task_id:
                rospy.logerr(
                    "Ignoring planner path metadata for %s while %s is selected",
                    task_id,
                    self._task_id,
                )
                return
            if len(tool_yaw_active) != len(message.poses):
                rospy.logerr(
                    "Ignoring planner path: %d poses but %d yaw-mask entries",
                    len(message.poses),
                    len(tool_yaw_active),
                )
                return
            self._path = message
            self._path_tool_yaw_active = np.asarray(
                tool_yaw_active, dtype=bool
            ).copy()
            self._path_approach_obstacle = dict(approach_obstacle)
            self._path_stage_timing = {
                "boundaries": list(stage_timing["boundaries"]),
                "transition_windows": [
                    list(window) for window in stage_timing["transition_windows"]
                ],
                "speed_scale": float(stage_timing["speed_scale"]),
                "ramp_before_m": float(stage_timing["ramp_before_m"]),
                "task_start_ramp_m": float(stage_timing["task_start_ramp_m"]),
            }
            self._pending_path = None
            self._operation = "task"
            self._path_serial += 1
            path_serial = self._path_serial
            self._prepared = None
        self._publish(
            "path_received",
            path_serial=path_serial,
            points=len(message.poses),
            message="Path received; prepare is required",
        )

    def _orientation_constraints_callback(self, message):
        try:
            stamp_ns, task_id, active, approach_obstacle, stage_timing = (
                self._parse_orientation_constraints(message)
            )
        except ValueError as error:
            rospy.logerr("%s", error)
            return
        pending = None
        with self._lock:
            self._orientation_constraints[stamp_ns] = (
                task_id,
                active,
                approach_obstacle,
                stage_timing,
            )
            for stale_stamp in sorted(self._orientation_constraints)[:-4]:
                self._orientation_constraints.pop(stale_stamp, None)
            if (
                self._pending_path is not None
                and int(self._pending_path.header.stamp.to_nsec()) == stamp_ns
            ):
                pending = self._pending_path
        if pending is not None:
            self._accept_path(
                pending, task_id, active, approach_obstacle, stage_timing
            )

    def _path_callback(self, message):
        stamp_ns = int(message.header.stamp.to_nsec())
        with self._lock:
            if self._execution_active or self._worker is not None:
                rospy.logerr("Ignoring a new planner path during real execution")
                return
            metadata = self._orientation_constraints.get(stamp_ns)
            if metadata is None:
                self._pending_path = message
                rospy.loginfo(
                    "Waiting for yaw-mask metadata matching planner path stamp %d",
                    stamp_ns,
                )
                return
        self._accept_path(
            message, metadata[0], metadata[1], metadata[2], metadata[3]
        )

    def _task_callback(self, message):
        task_id = str(message.data).strip()
        if task_id not in self._TASK_IDS:
            rospy.logerr("Ignoring unknown task id %s", task_id)
            return
        with self._lock:
            if self._worker is not None:
                rospy.logerr("Ignoring task switch to %s during real execution", task_id)
                return
            self._task_id = task_id
            self._operation = "task"
            self._constraint_source = str(
                rospy.get_param("/stage_constraint_planner/constraint_source", "true")
            )
            self._path = None
            self._path_tool_yaw_active = None
            self._path_approach_obstacle = None
            self._path_stage_timing = None
            self._pending_path = None
            self._prepared = None
            with open(self._task_config_paths[task_id], "r", encoding="utf-8") as stream:
                execution = json.load(stream)["execution"]
            self._approach_speed = float(execution["approach_speed_mps"])
            self._task_speed = float(execution["task_speed_mps"])
            self._approach_position_tolerance = float(
                execution["approach_position_tolerance_m"]
            )
            self._approach_joint_bridge_limit = float(
                execution["approach_joint_bridge_limit_rad"]
            )
            self._trajectory_compiler.set_task_speeds(
                self._approach_speed, self._task_speed
            )
            self._trajectory_compiler.set_approach_position_tolerance(
                self._approach_position_tolerance
            )
            self._trajectory_compiler.set_approach_joint_bridge_limit(
                self._approach_joint_bridge_limit
            )
        self._publish("task_selected", message="{} selected".format(task_id))

    def _joint_callback(self, message):
        positions = dict(zip(message.name, message.position))
        if not all(name in positions for name in self._joint_names):
            return
        values = np.asarray([positions[name] for name in self._joint_names], dtype=float)
        if not np.all(np.isfinite(values)):
            return
        with self._lock:
            self._joint_position = values
            self._joint_received = time.monotonic()

    def _controller_state_callback(self, message):
        positions = dict(zip(message.joint_names, message.desired.positions))
        if not all(name in positions for name in self._joint_names):
            return
        values = np.asarray([positions[name] for name in self._joint_names], dtype=float)
        if not np.all(np.isfinite(values)):
            return
        with self._lock:
            self._controller_desired_position = values
            self._controller_state_received = time.monotonic()

    def _controller_command_error(self, current):
        with self._lock:
            desired = (
                None
                if self._controller_desired_position is None
                else self._controller_desired_position.copy()
            )
            age = time.monotonic() - self._controller_state_received
        if desired is None or age > self._joint_state_timeout:
            return None
        return float(np.max(np.abs(desired - current)))

    def _commanding_callback(self, message):
        commanding = bool(message.data)
        with self._lock:
            self._commanding = commanding
            self._commanding_received = time.monotonic()
            execution_active = self._execution_active
            holding_final_position = self._holding_final_position
        if execution_active and not commanding:
            self._latch_execution_failure("FRI left COMMANDING_ACTIVE during execution")
        elif holding_final_position and not commanding:
            self._release_final_hold("FRI Overlay stopped; final position hold released")

    def _mode_callback(self, message):
        mode = int(message.data)
        with self._lock:
            self._fri_mode = mode
            self._fri_mode_received = time.monotonic()
            execution_active = self._execution_active
            holding_final_position = self._holding_final_position
        if execution_active and mode != self._position_mode:
            self._latch_execution_failure(
                "FRI left POSITION command mode during execution (mode {})".format(mode)
            )
        elif holding_final_position and mode != self._position_mode:
            self._release_final_hold(
                "FRI left POSITION mode; final position hold released"
            )

    def _release_final_hold(self, reason):
        with self._lock:
            if not self._holding_final_position:
                return False
            self._holding_final_position = False
        self._disarm_position_commands()
        self._publish("hold_released", message=str(reason))
        rospy.loginfo("Real executor final position hold released: %s", reason)
        return True

    def _latch_execution_failure(self, reason):
        with self._lock:
            if not self._execution_active:
                return False
            if self._execution_failure_reason is not None:
                return False
            self._execution_failure_reason = str(reason)
            self._prepared = None
        self._abort.set()
        self._disarm_position_commands()
        self._client.cancel_all_goals()
        rospy.logerr("Real execution failed closed: %s", reason)
        return True

    def _fri_failure_reason(self):
        now = time.monotonic()
        with self._lock:
            failure = self._execution_failure_reason
            execution_active = self._execution_active
            commanding = self._commanding
            commanding_age = now - self._commanding_received
            mode = self._fri_mode
            mode_age = now - self._fri_mode_received
            torque_age = now - self._torque_received
        if failure is not None and execution_active:
            return failure
        if commanding_age > self._fri_status_timeout or mode_age > self._fri_status_timeout:
            return "FRI status became stale during execution"
        if not commanding:
            return "FRI is not in COMMANDING_ACTIVE"
        if mode != self._position_mode:
            return "FRI command mode changed to {}; POSITION mode {} is required".format(
                mode, self._position_mode
            )
        if torque_age < 0.0 or torque_age > self._torque_timeout:
            return "External torque feedback is missing or stale"
        return None

    def _torque_callback(self, message):
        values = np.asarray(message.external_torques.data, dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            return
        with self._lock:
            self._torque_received = time.monotonic()
        if np.any(np.abs(values) > self._torque_thresholds):
            self._torque_trip_count += 1
        else:
            self._torque_trip_count = 0
        with self._lock:
            position_control_active = (
                self._execution_active or self._holding_final_position
            )
            protective_stop = self._protective_stop
        if self._torque_trip_count >= 5 and position_control_active and not protective_stop:
            with self._lock:
                self._protective_stop = True
            self._abort.set()
            self._disarm_position_commands()
            self._client.cancel_all_goals()
            self._publish("protective_stop", message="External joint torque threshold exceeded")

    def _fresh_joint_position(self):
        with self._lock:
            position = None if self._joint_position is None else self._joint_position.copy()
            age = time.monotonic() - self._joint_received
        if position is None or age > self._joint_state_timeout:
            raise ValidationError("/iiwa14/joint_states is missing or stale")
        return position

    def _require_hardware_ready(self):
        reason = self._fri_failure_reason()
        if reason is not None:
            raise ValidationError(reason)
        if not self._client.wait_for_server(rospy.Duration(1.0)):
            raise ValidationError("PositionTrajectoryController action server is unavailable")

    @staticmethod
    def _resolve_package_meshes(root):
        packages = rospkg.RosPack()
        cache = {}
        for element in root.iter():
            uri = element.get("filename", "")
            if not uri.startswith("package://"):
                continue
            package_name, relative = uri[len("package://"):].split("/", 1)
            if package_name not in cache:
                cache[package_name] = packages.get_path(package_name)
            element.set("filename", os.path.join(cache[package_name], relative))

    def _load_robot(self):
        description = rospy.get_param("robot_description")
        root = ElementTree.fromstring(description)
        self._resolve_package_meshes(root)
        for parent in root.iter():
            for child in list(parent):
                if child.tag in ("gazebo", "transmission", "self_collision_checking"):
                    parent.remove(child)
        handle = tempfile.NamedTemporaryFile(mode="wb", suffix=".urdf", delete=False)
        try:
            handle.write(ElementTree.tostring(root, encoding="utf-8", xml_declaration=True))
            handle.close()
            robot = bullet.loadURDF(
                handle.name,
                useFixedBase=True,
                flags=(bullet.URDF_USE_INERTIA_FROM_FILE |
                       bullet.URDF_MAINTAIN_LINK_ORDER |
                       bullet.URDF_USE_SELF_COLLISION |
                       bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT),
                physicsClientId=self._physics,
            )
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
        if robot < 0:
            raise RuntimeError("Could not load iiwa14 robot_description")
        return robot

    def _joint_model(self):
        by_name = {}
        for index in range(bullet.getNumJoints(self._robot, physicsClientId=self._physics)):
            info = bullet.getJointInfo(self._robot, index, physicsClientId=self._physics)
            by_name[info[1].decode()] = (index, info)
        missing = [name for name in self._joint_names if name not in by_name]
        if missing:
            raise RuntimeError("URDF is missing joints: {}".format(", ".join(missing)))
        indices, lower, upper, velocity = [], [], [], []
        for name in self._joint_names:
            index, info = by_name[name]
            indices.append(index)
            lower.append(float(info[8]))
            upper.append(float(info[9]))
            velocity.append(float(info[11]))
        return indices, np.asarray(lower), np.asarray(upper), np.asarray(velocity)

    def _find_link(self, name):
        for index in range(bullet.getNumJoints(self._robot, physicsClientId=self._physics)):
            info = bullet.getJointInfo(self._robot, index, physicsClientId=self._physics)
            if info[12].decode() == name:
                return index
        raise RuntimeError("URDF is missing tip link {}".format(name))

    def _time_parameterize(self, q_path, minimum_duration):
        return self._trajectory_compiler.time_parameterize(q_path, minimum_duration)

    def _build_plan(self, path, q_current):
        if not path.poses or len(path.poses) < 2:
            raise ValidationError("Planner path must contain at least two poses")
        frame = path.header.frame_id or path.poses[0].header.frame_id
        if frame != self._base_frame:
            raise ValidationError("Path frame {} does not match {}".format(frame, self._base_frame))
        positions = np.asarray([
            [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
            for pose in path.poses
        ], dtype=float)
        bases = [
            self._trajectory_compiler.tool_basis_from_quaternion([
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ])
            for pose in path.poses
        ]
        x_axes = np.asarray([basis[0] for basis in bases])
        axes = np.asarray([basis[1] for basis in bases])
        with self._lock:
            tool_yaw_active = (
                None
                if self._path_tool_yaw_active is None
                else self._path_tool_yaw_active.copy()
            )
            approach_obstacle = (
                None
                if self._path_approach_obstacle is None
                else dict(self._path_approach_obstacle)
            )
            stage_timing = (
                None
                if self._path_stage_timing is None
                else {
                    "boundaries": list(self._path_stage_timing["boundaries"]),
                    "transition_windows": [
                        list(window)
                        for window in self._path_stage_timing["transition_windows"]
                    ],
                    "speed_scale": float(self._path_stage_timing["speed_scale"]),
                    "ramp_before_m": float(
                        self._path_stage_timing["ramp_before_m"]
                    ),
                    "task_start_ramp_m": float(
                        self._path_stage_timing["task_start_ramp_m"]
                    ),
                }
            )
        if tool_yaw_active is None or tool_yaw_active.shape != (len(path.poses),):
            raise ValidationError(
                "Planner path is missing matching tool-yaw activation metadata"
            )
        if approach_obstacle is None:
            raise ValidationError(
                "Planner path is missing matching Stage-0 obstacle metadata"
            )
        if stage_timing is None:
            raise ValidationError(
                "Planner path is missing matching stage-timing metadata"
            )
        plan = self._trajectory_compiler.compile(
            positions,
            axes,
            q_current,
            abort_requested=self._abort.is_set,
            tool_x_axes=x_axes,
            tool_x_active=tool_yaw_active,
            approach_obstacle=approach_obstacle,
            stage_timing=stage_timing,
        )
        plan["path_serial"] = self._path_serial
        plan["task_id"] = self._task_id
        plan["operation"] = "task"
        return plan

    def _load_home_definition(self):
        with open(self._home_config_path, "r", encoding="utf-8") as stream:
            definition = json.load(stream)
        if str(definition.get("frame_id", "")) != self._base_frame:
            raise ValidationError(
                "Home pose frame {} does not match {}".format(
                    definition.get("frame_id"), self._base_frame
                )
            )
        try:
            joint_position = np.asarray(
                definition["joint_position_reference"], dtype=float
            )
            execution = definition["execution"]
            approach_speed = float(execution["approach_speed_mps"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValidationError("Home configuration is incomplete or invalid") from error
        if joint_position.shape != (len(self._joint_names),):
            raise ValidationError("Home configuration must contain seven joint positions")
        if not np.all(np.isfinite(joint_position)):
            raise ValidationError("Home joint positions contain non-finite values")
        return joint_position, approach_speed

    def _build_home_plan(self, q_current):
        joint_position, approach_speed = self._load_home_definition()
        self._trajectory_compiler.set_task_speeds(
            approach_speed, self._task_speed
        )
        plan = self._trajectory_compiler.compile_joint_home(
            q_current,
            abort_requested=self._abort.is_set,
            target_q=joint_position,
        )
        plan["path_serial"] = self._path_serial
        plan["task_id"] = "RobotHome"
        plan["operation"] = "home"
        plan["home_joint_position"] = joint_position.tolist()
        return plan

    def _return_home(self, _request):
        try:
            self._require_hardware_ready()
            if self._protective_stop:
                raise ValidationError(
                    "Protective stop is latched; stop and restart the real station after inspection"
                )
            current = self._fresh_joint_position()
            with self._lock:
                if self._worker is not None or self._execution_active:
                    raise ValidationError("A real robot motion is already running")
                self._operation = "home"
                self._prepared = None
            self._abort.clear()
            self._publish(
                "home_preparing",
                message="Building a joint-posture trajectory to Robot Home",
            )
            prepared = self._build_home_plan(current)
            latest = self._fresh_joint_position()
            drift = float(np.max(np.abs(latest - prepared["start"])))
            if drift > self._start_drift_limit:
                self._publish(
                    "home_repreparing",
                    message="Robot moved while preparing Home; rebuilding from current joints",
                )
                prepared = self._build_home_plan(latest)
                latest = self._fresh_joint_position()
                residual = float(np.max(np.abs(latest - prepared["start"])))
                if residual > self._start_drift_limit:
                    raise ValidationError(
                        "Robot continued moving while preparing Home ({:.4f} rad)".format(
                            residual
                        )
                    )
            if self._abort.is_set():
                raise ValidationError("Return Home was cancelled during preparation")
            self._publish(
                "home_prepared",
                message="Robot Home trajectory validated; synchronizing controller",
                metrics=prepared["metrics"],
            )
            with self._lock:
                self._prepared = prepared
                self._execution_failure_reason = None
                self._execution_active = True
            self._synchronize_and_arm(latest)
            if self._abort.is_set():
                raise ValidationError("Return Home was cancelled before motion")
            failure = self._fri_failure_reason()
            if failure is not None:
                self._latch_execution_failure(failure)
                raise ValidationError(failure)
            self._torque_trip_count = 0
            worker = threading.Thread(
                target=self._execution_worker, args=(prepared,), daemon=True
            )
            with self._lock:
                self._worker = worker
            worker.start()
            return TriggerResponse(True, "Validated Return Home motion started")
        except Exception as error:
            self._disarm_position_commands()
            self._client.cancel_all_goals()
            with self._lock:
                fri_failure = self._execution_failure_reason
                self._execution_active = False
                self._prepared = None
            phase = "failed" if fri_failure is not None else "rejected"
            self._publish(phase, message=fri_failure or str(error))
            return TriggerResponse(False, str(error))

    def _prepare(self, _request):
        try:
            self._require_hardware_ready()
            if self._protective_stop:
                raise ValidationError(
                    "Protective stop is latched; stop and restart the real station after inspection"
                )
            q_current = self._fresh_joint_position()
            with self._lock:
                if self._worker is not None:
                    raise ValidationError("A real task is already running")
                path = self._path
            if path is None:
                raise ValidationError("No /stage_cons/plan has been received")
            self._abort.clear()
            self._publish("preparing", message="Solving continuous IK and validating trajectory")
            prepared = self._build_plan(path, q_current)
            if self._abort.is_set():
                raise ValidationError("Preparation aborted")
            with self._lock:
                self._prepared = prepared
            self._publish("prepared", message="Trajectory validated; explicit execute is required", metrics=prepared["metrics"])
            return TriggerResponse(True, json.dumps(prepared["metrics"], sort_keys=True))
        except (ValidationError, ValueError, RuntimeError) as error:
            self._publish("rejected", message=str(error))
            return TriggerResponse(False, str(error))

    def _validate_only(self, _request):
        """Run the complete kinematic checks without arming or sending a goal."""
        try:
            q_current = self._fresh_joint_position()
            with self._lock:
                path = self._path
            if path is None:
                raise ValidationError("No /stage_cons/plan has been received")
            self._abort.clear()
            self._publish("validating", message="Running offline trajectory validation")
            validated = self._build_plan(path, q_current)
            self._publish(
                "validated", message="Offline validation passed; hardware is not armed",
                metrics=validated["metrics"],
            )
            return TriggerResponse(True, json.dumps(validated["metrics"], sort_keys=True))
        except (ValidationError, ValueError, RuntimeError) as error:
            self._publish("validation_rejected", message=str(error))
            return TriggerResponse(False, str(error))

    def _refresh_prepared_start(self, prepared, current):
        drift = float(np.max(np.abs(current - prepared["start"])))
        if drift <= self._start_drift_limit:
            return prepared, current
        with self._lock:
            path = self._path
            path_serial = self._path_serial
        if path is None or prepared["path_serial"] != path_serial:
            raise ValidationError("Planner path changed before start synchronization")
        self._publish(
            "repreparing",
            message=(
                "Robot moved {:.4f} rad during preparation; rebuilding the approach "
                "from the latest measured joints"
            ).format(drift),
        )
        refreshed = self._build_plan(path, current)
        latest = self._fresh_joint_position()
        residual = float(np.max(np.abs(latest - refreshed["start"])))
        with self._lock:
            path_unchanged = self._path_serial == path_serial
        if not path_unchanged or refreshed["path_serial"] != path_serial:
            raise ValidationError("Planner path changed during automatic preparation refresh")
        if residual > self._start_drift_limit:
            raise ValidationError(
                (
                    "Robot continued moving during automatic preparation refresh "
                    "({:.4f} rad); refusing to arm"
                ).format(residual)
            )
        with self._lock:
            self._prepared = refreshed
        self._publish(
            "prepared",
            message="Approach rebuilt from the latest measured joint position",
            metrics=refreshed["metrics"],
        )
        return refreshed, latest

    def _goal(self, segment, start_delay=0.3):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = (
            rospy.Time.now() + rospy.Duration(float(start_delay))
            if start_delay > 0.0
            else rospy.Time(0)
        )
        goal.trajectory.joint_names = list(self._joint_names)
        for q, velocity, acceleration, stamp in zip(
            segment["position"], segment["velocity"], segment["acceleration"], segment["time"]
        ):
            point = JointTrajectoryPoint()
            point.positions = q.tolist()
            point.velocities = velocity.tolist()
            point.accelerations = acceleration.tolist()
            point.time_from_start = rospy.Duration(float(stamp))
            goal.trajectory.points.append(point)
        return goal

    def _disarm_position_commands(self):
        with self._lock:
            self._position_armed = False
            self._holding_final_position = False
        try:
            self._position_gate.wait_for_service(timeout=1.0)
            response = self._position_gate(False)
            if not response.success:
                rospy.logerr("Driver did not confirm fixed Stop hold: %s", response.message)
                return False, str(response.message)
            return True, str(response.message)
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Could not close the position gate: %s", error)
            return False, str(error)

    def _synchronize_and_arm(self, current):
        with self._lock:
            continuing_final_hold = (
                self._holding_final_position and self._position_armed
            )
            if not continuing_final_hold:
                self._position_armed = False
        self._position_gate.wait_for_service(timeout=2.0)
        if not continuing_final_hold:
            disabled = self._position_gate(False)
            if not disabled.success:
                raise ValidationError("Could not close the driver position-command gate")
        hold_path = np.vstack((current, current))
        # Keep the synchronization goal active while the driver gate is opened.
        # Waiting until after this goal completed allowed the trajectory
        # controller to expose a stale pre-FRI command between the two steps.
        hold = self._time_parameterize(hold_path, 1.0)
        # A normal task goal is deliberately scheduled 0.3 s in the future and
        # its first point has another trajectory delay.  That is appropriate for
        # motion, but it used to make the driver's 0.25 s arm handshake expire
        # before this synchronization command reached the hardware interface.
        # Start the stationary hold immediately and give the controller 50 ms to
        # replace its pre-FRI command while the driver gate remains closed.
        hold["time"] = np.asarray([0.05, max(1.0, float(hold["duration"]))])
        hold["duration"] = float(hold["time"][-1])
        self._client.send_goal(self._goal(hold, start_delay=0.0))
        if not continuing_final_hold:
            deadline = time.monotonic() + 2.0
            command_error = None
            while time.monotonic() < deadline:
                if self._abort.is_set():
                    self._client.cancel_goal()
                    raise ValidationError(
                        "Execution was cancelled during controller synchronization"
                    )
                command_error = self._controller_command_error(current)
                if (
                    command_error is not None
                    and command_error <= self._start_drift_limit
                ):
                    break
                time.sleep(0.02)
            else:
                self._client.cancel_goal()
                detail = (
                    "controller state is missing or stale"
                    if command_error is None
                    else "maximum desired-joint error is {:.4f} rad".format(command_error)
                )
                raise ValidationError(
                    "Position controller did not synchronize to measured joints: " + detail
                )
        # Start the watchdog heartbeat before asking the real-time driver to
        # open its gate, so there is no unmonitored arm interval.
        with self._lock:
            self._position_armed = True
        armed = self._position_gate(True)
        if not armed.success:
            self._disarm_position_commands()
            raise ValidationError("Driver refused position-command arming: " + armed.message)
        if self._abort.is_set():
            self._disarm_position_commands()
            raise ValidationError("Execution was cancelled while arming position commands")
        if not self._client.wait_for_result(rospy.Duration(3.0)):
            self._disarm_position_commands()
            self._client.cancel_goal()
            raise ValidationError("Position controller synchronization timed out")
        if self._client.get_state() != GoalStatus.SUCCEEDED:
            state = self._client.get_state()
            self._disarm_position_commands()
            raise ValidationError(
                "Position controller synchronization failed with action state {}".format(
                    state
                )
            )
        with self._lock:
            self._holding_final_position = False

    def _run_segment(self, name, segment):
        goal = self._goal(segment)
        status_fields = {
            "message": name.replace("_", " "),
            "duration_s": segment["duration"],
        }
        if name == "executing":
            trajectory_start = goal.trajectory.header.stamp
            first_motion = goal.trajectory.points[0].time_from_start
            final_motion = goal.trajectory.points[-1].time_from_start
            status_fields.update(
                motion_start_unix_ns=int(
                    (trajectory_start + first_motion).to_nsec()
                ),
                motion_end_unix_ns=int(
                    (trajectory_start + final_motion).to_nsec()
                ),
            )
        self._publish(name, **status_fields)
        self._client.send_goal(goal)
        deadline = time.monotonic() + segment["duration"] + 10.0
        while time.monotonic() < deadline:
            failure = self._fri_failure_reason()
            if failure is not None:
                self._latch_execution_failure(failure)
                raise ValidationError(failure)
            if self._abort.is_set():
                self._client.cancel_goal()
                raise ValidationError("Execution aborted")
            if self._client.wait_for_result(rospy.Duration(0.1)):
                failure = self._fri_failure_reason()
                if failure is not None:
                    self._latch_execution_failure(failure)
                    raise ValidationError(failure)
                state = self._client.get_state()
                if state != GoalStatus.SUCCEEDED:
                    raise ValidationError("{} failed with action state {}".format(name, state))
                return
        self._client.cancel_goal()
        raise ValidationError("{} timed out".format(name))

    def _wait_for_task_start(self, target):
        """Wait briefly for physical tracking to settle at the compiled start.

        The trajectory action can report success as soon as its time and velocity
        conditions are satisfied.  Keep the final approach target held and retain
        the original one-degree position criterion instead of rejecting one
        transient joint-state sample immediately after action completion.
        """
        target = np.asarray(target, dtype=float)
        deadline = time.monotonic() + self._task_start_settle_timeout
        settled_samples = 0
        last_error = math.inf
        while True:
            failure = self._fri_failure_reason()
            if failure is not None:
                self._latch_execution_failure(failure)
                raise ValidationError(failure)
            if self._abort.is_set():
                raise ValidationError("Execution aborted while settling at task start")
            actual = self._fresh_joint_position()
            last_error = float(np.max(np.abs(actual - target)))
            if last_error <= self._task_start_settle_tolerance:
                settled_samples += 1
                if settled_samples >= self._task_start_settle_samples:
                    return last_error
            else:
                settled_samples = 0
            if time.monotonic() >= deadline:
                raise ValidationError(
                    "Robot did not settle at task start ({:.3f} rad after {:.1f} s)".format(
                        last_error, self._task_start_settle_timeout
                    )
                )
            time.sleep(0.05)

    def _start_recording(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        name = now.strftime("%Y%m%dT%H%M%S_%fZ_real_task")
        self._run_directory = self._output_root / self._task_id / name
        self._run_directory.parent.mkdir(parents=True, exist_ok=True)
        self._run_directory.mkdir(parents=True, exist_ok=False)
        bag = self._run_directory / "real_task.bag"
        topics = [
            "/iiwa14/joint_states", "/iiwa14/additional_outputs",
            "/iiwa14/PositionTrajectoryController/state", "/iiwa14/real_executor/status",
            "/iiwa14/commanding_status", "/iiwa14/fri_command_mode",
            "/iiwa14/fri_diagnostics",
            "/stage_cons/plan", "/tf", "/tf_static",
            "/stage_cons/planner/task", "/stage_cons/plan_stage_boundaries",
            "/stage_cons/plan_orientation_constraints",
            "/stage_cons/planner/tracking_status",
            "/vrpn_client_node/baiyu_bar/pose_from_iiwa14",
            "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14",
        ]
        metadata = {
            "schema_version": 1,
            "mode": "real",
            "task_id": self._task_id,
            "constraint_source": self._constraint_source,
            "robot": "iiwa14",
            "tip_link": self._tip_link,
            "orientation_control": "position_plus_full_orientation",
            "started_at_utc": now.isoformat(),
            "bag_file": bag.name,
            "topics": topics,
        }
        (self._run_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self._record_process = subprocess.Popen(
            ["rosbag", "record", "--lz4", "--output-name", str(bag)] + topics,
            start_new_session=True,
        )
        rospy.sleep(0.3)

    def _stop_recording(self):
        process = self._record_process
        self._record_process = None
        if process is None:
            return
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5.0)

    def _execution_worker(self, prepared):
        final_phase = "failed"
        message = "Execution failed"
        operation = str(prepared.get("operation", "task"))
        try:
            if operation == "home":
                recovery = prepared.get("recovery")
                if recovery is not None:
                    self._run_segment("home_recovering", recovery)
                    actual = self._fresh_joint_position()
                    recovery_error = float(
                        np.max(
                            np.abs(
                                actual - prepared["approach"]["position"][0]
                            )
                        )
                    )
                    if recovery_error > math.radians(1.0):
                        raise ValidationError(
                            "Robot did not settle after vertical Home recovery "
                            "({:.3f} rad)".format(recovery_error)
                        )
                    if self._abort.is_set():
                        raise ValidationError(
                            "Return Home was cancelled after vertical recovery"
                        )
                    failure = self._fri_failure_reason()
                    if failure is not None:
                        self._latch_execution_failure(failure)
                        raise ValidationError(failure)
                self._run_segment("returning_home", prepared["approach"])
                message = "Robot returned Home"
            else:
                self._run_segment("moving_to_start", prepared["approach"])
                self._wait_for_task_start(prepared["task"]["position"][0])
                if self._record_requested:
                    self._start_recording()
                self._run_segment("executing", prepared["task"])
                message = "Real task completed"
            final_phase = "complete"
        except Exception as error:
            message = str(error)
            with self._lock:
                fri_failure = self._execution_failure_reason
            if fri_failure is not None:
                final_phase = "failed"
                message = fri_failure
            elif self._abort.is_set() and not self._protective_stop:
                final_phase = "aborted"
        finally:
            self._stop_recording()

            # A successful Position Overlay task must keep the controller's
            # fixed final target, gate, and heartbeat alive. Disabling the gate
            # here would make the driver use the freshly measured joints as a
            # moving reference and allows the arm to drift while FRI remains
            # COMMANDING_ACTIVE.
            completion_failure = self._fri_failure_reason()
            now = time.monotonic()
            with self._lock:
                commanding = self._commanding
                mode = self._fri_mode
                commanding_fresh = (
                    now - self._commanding_received <= self._fri_status_timeout
                )
                mode_fresh = now - self._fri_mode_received <= self._fri_status_timeout
                hold_final = (
                    final_phase == "complete"
                    and completion_failure is None
                    and self._execution_failure_reason is None
                    and not self._abort.is_set()
                    and self._position_armed
                    and commanding
                    and commanding_fresh
                    and mode == self._position_mode
                    and mode_fresh
                )
                if hold_final:
                    self._holding_final_position = True
                    self._execution_active = False
                    self._worker = None
                    self._prepared = None
                    message = (
                        "Robot returned Home; holding position until FRI Overlay stops"
                        if operation == "home"
                        else "Real task completed; holding final position until FRI Overlay stops"
                    )
                    self._publish(
                        "complete", message=message, metrics=prepared["metrics"]
                    )

            if not hold_final:
                with self._lock:
                    fri_failure = self._execution_failure_reason
                if final_phase == "complete":
                    final_phase = "failed"
                    if fri_failure is not None:
                        message = fri_failure
                    elif self._abort.is_set():
                        message = "Execution was cancelled before final position hold"
                    elif not commanding_fresh or not mode_fresh:
                        message = "FRI status became stale before final position hold"
                    elif not commanding:
                        message = "FRI left COMMANDING_ACTIVE before final position hold"
                    elif mode != self._position_mode:
                        message = "FRI left POSITION mode before final position hold"
                    elif completion_failure is not None:
                        message = completion_failure
                    else:
                        message = "Position command gate was not armed at task completion"
                self._disarm_position_commands()
                self._client.cancel_all_goals()
                with self._lock:
                    self._worker = None
                    self._prepared = None
                    self._execution_active = False
                self._publish(
                    final_phase, message=message, metrics=prepared["metrics"]
                )

    def _execute(self, _request):
        try:
            self._require_hardware_ready()
            current = self._fresh_joint_position()
            with self._lock:
                prepared = self._prepared
                if self._worker is not None:
                    raise ValidationError("A real task is already running")
            if prepared is None:
                raise ValidationError("No validated trajectory is prepared")
            if str(prepared.get("operation", "task")) != "task":
                raise ValidationError("Prepared motion is not a planner task")
            if prepared["path_serial"] != self._path_serial:
                raise ValidationError("Planner path changed after preparation")
            prepared, current = self._refresh_prepared_start(prepared, current)
            self._abort.clear()
            with self._lock:
                self._execution_failure_reason = None
                self._execution_active = True
            self._synchronize_and_arm(current)
            if self._abort.is_set():
                raise ValidationError("Execution was cancelled before trajectory start")
            failure = self._fri_failure_reason()
            if failure is not None:
                self._latch_execution_failure(failure)
                raise ValidationError(failure)
            self._torque_trip_count = 0
            worker = threading.Thread(target=self._execution_worker, args=(prepared,), daemon=True)
            with self._lock:
                self._worker = worker
            worker.start()
            return TriggerResponse(True, "Validated real execution started")
        except Exception as error:
            self._disarm_position_commands()
            self._client.cancel_all_goals()
            with self._lock:
                fri_failure = self._execution_failure_reason
                self._execution_active = False
                if fri_failure is not None:
                    self._prepared = None
            phase = "failed" if fri_failure is not None else "rejected"
            self._publish(phase, message=fri_failure or str(error))
            return TriggerResponse(False, str(error))

    def _abort_execution(self, _request):
        self._abort.set()
        # Close the hardware authority boundary and latch the current fixed
        # hold before cancelling the ROS action.  Cancelling first lets the
        # controller expose its own post-cancel command while the gate is still
        # open, which can produce motion after the user pressed Stop.
        disarmed, detail = self._disarm_position_commands()
        self._client.cancel_all_goals()
        with self._lock:
            if self._worker is None:
                self._prepared = None
        if not disarmed:
            self._publish(
                "control_status_unknown",
                message="Stop requested, but the driver did not confirm its fixed hold: " + detail,
            )
            return TriggerResponse(
                False,
                "Driver did not confirm the fixed Stop hold: " + detail,
            )
        self._publish("aborted", message="Fixed Stop hold confirmed; task terminated")
        return TriggerResponse(True, "Fixed Stop hold confirmed; task terminated")

    def _set_recording(self, request):
        with self._lock:
            if self._worker is not None:
                return SetBoolResponse(False, "Cannot change recording while executing")
            self._record_requested = bool(request.data)
        self._publish("recording_configured", message="Recording {}".format("enabled" if request.data else "disabled"))
        return SetBoolResponse(True, "Recording preference updated")

    def _shutdown(self):
        self._abort.set()
        self._disarm_position_commands()
        self._client.cancel_all_goals()
        self._stop_recording()
        if self._physics >= 0:
            bullet.disconnect(self._physics)
            self._physics = -1


if __name__ == "__main__":
    rospy.init_node("real_executor")
    RealExecutor()
    rospy.spin()
