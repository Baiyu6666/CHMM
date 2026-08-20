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
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
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
        self._start_drift_limit = float(rospy.get_param("~start_drift_limit_rad", math.radians(0.5)))
        self._max_joint_step = float(rospy.get_param("~max_joint_step_rad", 0.15))
        self._velocity_scale = float(rospy.get_param("~velocity_scale", 0.10))
        self._acceleration_limit = float(rospy.get_param("~acceleration_limit_rad_s2", 0.25))
        self._approach_speed = float(rospy.get_param("~approach_speed_mps", 0.04))
        self._task_speed = float(rospy.get_param("~task_speed_mps", 0.025))
        self._torque_thresholds = np.asarray(
            rospy.get_param("~external_torque_thresholds_nm", [20, 20, 15, 15, 8, 8, 8]),
            dtype=float,
        )
        if self._torque_thresholds.shape != (7,):
            raise ValueError("~external_torque_thresholds_nm must contain seven values")

        self._path = None
        self._path_serial = 0
        self._joint_position = None
        self._joint_received = 0.0
        self._commanding = False
        self._fri_mode = 0
        self._prepared = None
        self._worker = None
        self._abort = threading.Event()
        self._record_requested = True
        self._record_process = None
        self._run_directory = None
        self._torque_trip_count = 0
        self._protective_stop = False
        self._position_armed = False

        self._client = actionlib.SimpleActionClient(
            self._action_name, FollowJointTrajectoryAction
        )
        self._position_gate = rospy.ServiceProxy(
            "/iiwa14/iiwa_driver/set_position_commanding", SetBool
        )
        self._position_heartbeat_pub = rospy.Publisher(
            "/iiwa14/iiwa_driver/position_command_heartbeat", Empty, queue_size=1
        )
        self._position_heartbeat_timer = rospy.Timer(
            rospy.Duration(0.05), self._publish_position_heartbeat
        )
        self._status_pub = rospy.Publisher("~status", String, queue_size=1, latch=True)
        rospy.Subscriber("/stage_cons/plan", RosPath, self._path_callback, queue_size=1)
        rospy.Subscriber("joint_states", JointState, self._joint_callback, queue_size=5)
        rospy.Subscriber("commanding_status", Bool, self._commanding_callback, queue_size=2)
        rospy.Subscriber("fri_command_mode", Int32, self._mode_callback, queue_size=2)
        rospy.Subscriber("additional_outputs", AdditionalOutputs, self._torque_callback, queue_size=5)
        rospy.Service("~prepare", Trigger, self._prepare)
        rospy.Service("~validate", Trigger, self._validate_only)
        rospy.Service("~execute", Trigger, self._execute)
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
        )
        self._publish("idle", message="Real executor ready; no trajectory prepared")
        rospy.on_shutdown(self._shutdown)

    def _publish(self, phase, **fields):
        payload = {
            "phase": phase,
            "stamp": rospy.Time.now().to_sec(),
            "record": self._record_requested,
            "run_directory": str(self._run_directory) if self._run_directory else None,
            "protective_stop": self._protective_stop,
        }
        payload.update(fields)
        self._status_pub.publish(String(data=json.dumps(payload, sort_keys=True)))

    def _publish_position_heartbeat(self, _event):
        if self._position_armed:
            self._position_heartbeat_pub.publish(Empty())

    def _path_callback(self, message):
        with self._lock:
            self._path = message
            self._path_serial += 1
            self._prepared = None
        self._publish("path_received", points=len(message.poses), message="Path received; prepare is required")

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

    def _commanding_callback(self, message):
        self._commanding = bool(message.data)

    def _mode_callback(self, message):
        self._fri_mode = int(message.data)

    def _torque_callback(self, message):
        values = np.asarray(message.external_torques.data, dtype=float)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            return
        if np.any(np.abs(values) > self._torque_thresholds):
            self._torque_trip_count += 1
        else:
            self._torque_trip_count = 0
        if self._torque_trip_count >= 5 and self._worker is not None:
            self._protective_stop = True
            self._abort.set()
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
        if not self._commanding:
            raise ValidationError("FRI is not in COMMANDING_ACTIVE")
        if self._fri_mode != self._position_mode:
            raise ValidationError(
                "FRI command mode is {}; POSITION mode {} is required".format(
                    self._fri_mode, self._position_mode
                )
            )
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
        axes = np.asarray([
            self._trajectory_compiler.tool_z_from_quaternion([
                pose.pose.orientation.x,
                pose.pose.orientation.y,
                pose.pose.orientation.z,
                pose.pose.orientation.w,
            ])
            for pose in path.poses
        ])
        plan = self._trajectory_compiler.compile(
            positions,
            axes,
            q_current,
            abort_requested=self._abort.is_set,
        )
        plan["path_serial"] = self._path_serial
        return plan

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

    def _goal(self, segment):
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.3)
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
        self._position_armed = False
        try:
            self._position_gate.wait_for_service(timeout=1.0)
            self._position_gate(False)
        except (rospy.ROSException, rospy.ServiceException):
            pass

    def _synchronize_and_arm(self, current):
        self._position_armed = False
        self._position_gate.wait_for_service(timeout=2.0)
        disabled = self._position_gate(False)
        if not disabled.success:
            raise ValidationError("Could not close the driver position-command gate")
        hold_path = np.vstack((current, current))
        hold = self._time_parameterize(hold_path, 0.5)
        self._client.send_goal(self._goal(hold))
        if not self._client.wait_for_result(rospy.Duration(3.0)):
            self._client.cancel_goal()
            raise ValidationError("Position controller synchronization timed out")
        if self._client.get_state() != GoalStatus.SUCCEEDED:
            raise ValidationError(
                "Position controller synchronization failed with action state {}".format(
                    self._client.get_state()
                )
            )
        # Start the watchdog heartbeat before asking the real-time driver to
        # open its gate, so there is no unmonitored arm interval.
        self._position_armed = True
        armed = self._position_gate(True)
        if not armed.success:
            self._position_armed = False
            raise ValidationError("Driver refused position-command arming: " + armed.message)

    def _run_segment(self, name, segment):
        self._publish(name, message=name.replace("_", " "), duration_s=segment["duration"])
        self._client.send_goal(self._goal(segment))
        deadline = time.monotonic() + segment["duration"] + 10.0
        while time.monotonic() < deadline:
            if self._abort.is_set():
                self._client.cancel_goal()
                raise ValidationError("Execution aborted")
            if self._client.wait_for_result(rospy.Duration(0.1)):
                state = self._client.get_state()
                if state != GoalStatus.SUCCEEDED:
                    raise ValidationError("{} failed with action state {}".format(name, state))
                return
        self._client.cancel_goal()
        raise ValidationError("{} timed out".format(name))

    def _start_recording(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        name = now.strftime("%Y%m%dT%H%M%S_%fZ_real_task")
        self._run_directory = self._output_root / name
        self._run_directory.mkdir(parents=True, exist_ok=False)
        bag = self._run_directory / "real_task.bag"
        topics = [
            "/iiwa14/joint_states", "/iiwa14/additional_outputs",
            "/iiwa14/PositionTrajectoryController/state", "/iiwa14/real_executor/status",
            "/iiwa14/commanding_status", "/iiwa14/fri_command_mode",
            "/stage_cons/plan", "/tf", "/tf_static",
        ]
        metadata = {
            "schema_version": 1,
            "mode": "real",
            "robot": "iiwa14",
            "tip_link": self._tip_link,
            "orientation_control": "position_plus_tool_z",
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
        try:
            self._run_segment("moving_to_start", prepared["approach"])
            actual = self._fresh_joint_position()
            start_error = float(np.max(np.abs(actual - prepared["task"]["position"][0])))
            if start_error > math.radians(1.0):
                raise ValidationError("Robot did not settle at task start ({:.3f} rad)".format(start_error))
            if self._record_requested:
                self._start_recording()
            self._run_segment("executing", prepared["task"])
            final_phase = "complete"
            message = "Real task completed"
        except Exception as error:
            message = str(error)
            if self._abort.is_set() and not self._protective_stop:
                final_phase = "aborted"
        finally:
            self._client.cancel_all_goals()
            self._stop_recording()
            if final_phase != "complete":
                self._disarm_position_commands()
            self._publish(final_phase, message=message, metrics=prepared["metrics"])
            with self._lock:
                self._worker = None
                self._prepared = None

    def _execute(self, _request):
        armed = False
        try:
            self._require_hardware_ready()
            current = self._fresh_joint_position()
            with self._lock:
                prepared = self._prepared
                if self._worker is not None:
                    raise ValidationError("A real task is already running")
            if prepared is None:
                raise ValidationError("No validated trajectory is prepared")
            if prepared["path_serial"] != self._path_serial:
                raise ValidationError("Planner path changed after preparation")
            drift = float(np.max(np.abs(current - prepared["start"])))
            if drift > self._start_drift_limit:
                raise ValidationError("Robot moved {:.4f} rad after planning; prepare again".format(drift))
            self._synchronize_and_arm(current)
            armed = True
            self._abort.clear()
            self._torque_trip_count = 0
            worker = threading.Thread(target=self._execution_worker, args=(prepared,), daemon=True)
            with self._lock:
                self._worker = worker
            worker.start()
            return TriggerResponse(True, "Validated real execution started")
        except (ValidationError, ValueError, RuntimeError) as error:
            if armed:
                self._disarm_position_commands()
            self._publish("rejected", message=str(error))
            return TriggerResponse(False, str(error))

    def _abort_execution(self, _request):
        self._abort.set()
        self._client.cancel_all_goals()
        self._disarm_position_commands()
        with self._lock:
            if self._worker is None:
                self._prepared = None
        self._publish("aborted", message="Abort requested")
        return TriggerResponse(True, "Abort requested")

    def _set_recording(self, request):
        with self._lock:
            if self._worker is not None:
                return SetBoolResponse(False, "Cannot change recording while executing")
            self._record_requested = bool(request.data)
        self._publish("recording_configured", message="Recording {}".format("enabled" if request.data else "disabled"))
        return SetBoolResponse(True, "Recording preference updated")

    def _shutdown(self):
        self._abort.set()
        self._client.cancel_all_goals()
        self._disarm_position_commands()
        self._stop_recording()
        if self._physics >= 0:
            bullet.disconnect(self._physics)
            self._physics = -1


if __name__ == "__main__":
    rospy.init_node("real_executor")
    RealExecutor()
    rospy.spin()
