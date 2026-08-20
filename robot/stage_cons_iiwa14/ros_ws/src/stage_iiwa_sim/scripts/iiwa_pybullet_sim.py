#!/usr/bin/env python3
"""Run the iiwa14 and the measured workcell geometry in PyBullet.

The planner publishes a Cartesian ``nav_msgs/Path``.  The shared trajectory
compiler used by the real executor turns the complete path into continuous,
time-parameterized joint segments.  This node only executes those segments in
PyBullet, publishes the usual ROS state topics, and logs the simulation run.
"""

import csv
import json
import math
import os
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime, timezone
from xml.etree import ElementTree

import pybullet as bullet
import numpy as np
import rospkg
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from stage_cartesian_trajectory import (
    CartesianTrajectoryCompiler,
    TrajectoryValidationError,
)
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger


class IiwaPyBulletSim:
    JOINT_NAMES = ["iiwa14_joint_{}".format(index) for index in range(1, 8)]
    LOG_FIELDS = (
        ["wall_time_utc", "ros_time", "sim_time", "controller_state", "path_index"]
        + ["q{}".format(index) for index in range(1, 8)]
        + ["dq{}".format(index) for index in range(1, 8)]
        + ["q_target{}".format(index) for index in range(1, 8)]
        + ["ee_x", "ee_y", "ee_z", "ee_qx", "ee_qy", "ee_qz", "ee_qw"]
        + ["target_x", "target_y", "target_z", "target_qx", "target_qy", "target_qz", "target_qw"]
        + [
            "contact_count",
            "table_contact_count",
            "bar_contact_count",
            "obstacle_contact_count",
        ]
    )

    def __init__(self):
        self.frame_id = rospy.get_param("~frame_id", "iiwa14_link_0")
        self.physics_hz = float(rospy.get_param("~physics_hz", 240.0))
        self.publish_hz = float(rospy.get_param("~publish_hz", 60.0))
        self.max_joint_force = float(rospy.get_param("~max_joint_force", 250.0))
        self.joint_settle_tolerance = float(
            rospy.get_param("~joint_settle_tolerance_rad", 0.01)
        )
        self.segment_settle_timeout = float(
            rospy.get_param("~segment_settle_timeout", 3.0)
        )
        self.max_joint_step = float(rospy.get_param("~max_joint_step_rad", 0.15))
        self.velocity_scale = float(rospy.get_param("~velocity_scale", 0.10))
        self.acceleration_limit = float(
            rospy.get_param("~acceleration_limit_rad_s2", 0.25)
        )
        self.approach_speed = float(rospy.get_param("~approach_speed_mps", 0.04))
        self.task_speed = float(rospy.get_param("~task_speed_mps", 0.025))
        if self.joint_settle_tolerance <= 0.0 or self.segment_settle_timeout <= 0.0:
            raise ValueError("joint settle tolerance and timeout must be positive")
        self.gui = bool(rospy.get_param("~gui", False))
        self.auto_plan = bool(rospy.get_param("~auto_plan", True))
        self.render_video = bool(rospy.get_param("~render_video", True))
        self.video_fps = int(rospy.get_param("~video_fps", 30))
        self.video_size = [int(value) for value in rospy.get_param("~video_size", [640, 480])]
        self.video_initial_hold = float(rospy.get_param("~video_initial_hold", 1.0))
        self.video_post_roll = float(rospy.get_param("~video_post_roll", 0.25))
        if self.video_fps <= 0 or len(self.video_size) != 2 or min(self.video_size) <= 0:
            raise ValueError("video_fps and video_size must be positive")

        self.table_top_z = float(rospy.get_param("~table_top_z", 0.14584))
        self.table_size = self._vector_param("~table_size", [0.8, 0.8, 0.05], 3)
        self.table_center_xy = self._vector_param("~table_center_xy", [-0.60, 0.0], 2)
        self.bar_size = self._vector_param("~bar_size", [0.29806, 0.06239, 0.06], 3)
        self.bar_size[2] = float(rospy.get_param("~bar_height", self.bar_size[2]))
        self.bar_center_xy = self._vector_param("~bar_center_xy", [-0.50, 0.0], 2)
        self.bar_yaw = float(rospy.get_param("~bar_yaw", 0.0))
        self.obstacle_center = self._vector_param(
            "~obstacle_center", [-0.285, -0.090, self.table_top_z + 0.05], 3
        )
        self.obstacle_radius = float(rospy.get_param("~obstacle_radius", 0.05))
        self.scan_standoff = float(rospy.get_param("~scan_standoff", 0.08))
        self.placeholder_distance = float(rospy.get_param("~placeholder_distance", 0.15))

        self._path_lock = threading.Lock()
        self._pending_path = None
        self._task_path = []
        self._path_index = -1
        self._prepared_plan = None
        self._active_segment = None
        self._segment_started = 0.0
        self._controller_state = "idle"
        self._status_message = "Simulator ready"
        self._target_pose = None
        self._sim_time = 0.0
        self._last_publish = -math.inf
        self._last_status = -math.inf
        self._closed = False
        self._goal_marker_id = None

        connection_mode = bullet.GUI if self.gui else bullet.DIRECT
        self.client = bullet.connect(connection_mode)
        if self.client < 0:
            raise RuntimeError("PyBullet connection failed")
        bullet.resetSimulation(physicsClientId=self.client)
        bullet.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client)
        bullet.setTimeStep(1.0 / self.physics_hz, physicsClientId=self.client)
        bullet.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / self.physics_hz,
            numSolverIterations=100,
            physicsClientId=self.client,
        )

        self.robot_id = self._load_robot()
        (
            self.joint_indices,
            self.lower_limits,
            self.upper_limits,
            self.joint_ranges,
            self.velocity_limits,
        ) = self._joint_model()
        self.ee_link_index = self._find_link("iiwa14_link_ee")
        self.trajectory_compiler = CartesianTrajectoryCompiler(
            bullet,
            self.client,
            self.robot_id,
            self.joint_indices,
            self.ee_link_index,
            self.lower_limits,
            self.upper_limits,
            self.velocity_limits,
            max_joint_step=self.max_joint_step,
            velocity_scale=self.velocity_scale,
            acceleration_limit=self.acceleration_limit,
            approach_speed=self.approach_speed,
            task_speed=self.task_speed,
        )
        self.table_id, self.bar_id, self.obstacle_id = self._create_workcell()
        self._move_to_scan_start()

        self.joint_publisher = rospy.Publisher("joint_states", JointState, queue_size=5)
        self.status_publisher = rospy.Publisher("sim/status", String, queue_size=2, latch=True)
        self.bar_publisher = rospy.Publisher(
            "/vrpn_client_node/baiyu_bar/pose_from_iiwa14", PoseStamped, queue_size=2, latch=True
        )
        self.obstacle_publisher = rospy.Publisher(
            "/vrpn_client_node/obstacle/pose_from_iiwa14", PoseStamped, queue_size=2, latch=True
        )
        self.start_publisher = rospy.Publisher(
            "/stage_cons/planner/start", PoseStamped, queue_size=1, latch=True
        )
        self.goal_publisher = rospy.Publisher(
            "/stage_cons/planner/goal", PoseStamped, queue_size=1, latch=True
        )
        self.path_subscriber = rospy.Subscriber(
            "/stage_cons/plan", Path, self._path_callback, queue_size=1
        )
        self.recording_service = rospy.Service("sim/set_recording", SetBool, self._set_recording)
        self.task_recording_service = rospy.Service(
            "sim/set_task_recording", SetBool, self._set_task_recording
        )

        self._log_file = None
        self._log_writer = None
        self._run_directory = None
        self._recording = False
        self._video_process = None
        self._video_log_file = None
        self._video_path = None
        self._video_active = False
        self._video_finished = False
        self._video_complete_time = None
        self._task_record_requested = bool(rospy.get_param("~record", True))

        rospy.on_shutdown(self.close)
        if self.auto_plan:
            self._auto_plan_timer = rospy.Timer(
                rospy.Duration(2.0), self._publish_placeholder_request, oneshot=True
            )

        rospy.loginfo(
            "PyBullet iiwa14 ready (%s); table top %.5f m, bar %.3f x %.3f x %.3f m",
            "GUI" if self.gui else "DIRECT",
            self.table_top_z,
            self.bar_size[0],
            self.bar_size[1],
            self.bar_size[2],
        )

    @staticmethod
    def _vector_param(name, default, length):
        values = [float(value) for value in rospy.get_param(name, default)]
        if len(values) != length:
            raise ValueError("{} must contain {} values".format(name, length))
        return values

    def _load_robot(self):
        description = rospy.get_param("robot_description")
        root = ElementTree.fromstring(description)
        package_paths = {}
        ros_packages = rospkg.RosPack()
        for element in root.iter():
            if element.tag == "mesh" and element.get("filename", "").startswith("package://"):
                uri = element.get("filename")
                package_name, relative_path = uri[len("package://"):].split("/", 1)
                if package_name not in package_paths:
                    package_paths[package_name] = ros_packages.get_path(package_name)
                element.set("filename", os.path.join(package_paths[package_name], relative_path))
        for parent in root.iter():
            for child in list(parent):
                if child.tag in ("gazebo", "transmission", "self_collision_checking"):
                    parent.remove(child)

        handle = tempfile.NamedTemporaryFile(mode="wb", suffix=".urdf", delete=False)
        try:
            handle.write(ElementTree.tostring(root, encoding="utf-8", xml_declaration=True))
            handle.close()
            robot_id = bullet.loadURDF(
                handle.name,
                useFixedBase=True,
                flags=(
                    bullet.URDF_USE_INERTIA_FROM_FILE
                    | bullet.URDF_MAINTAIN_LINK_ORDER
                    | bullet.URDF_USE_SELF_COLLISION
                    | bullet.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
                ),
                physicsClientId=self.client,
            )
        finally:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
        if robot_id < 0:
            raise RuntimeError("Could not load the iiwa14 URDF")
        return robot_id

    def _joint_model(self):
        joint_by_name = {}
        for index in range(bullet.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = bullet.getJointInfo(self.robot_id, index, physicsClientId=self.client)
            joint_by_name[info[1].decode("utf-8")] = (index, info)
        missing = [name for name in self.JOINT_NAMES if name not in joint_by_name]
        if missing:
            raise RuntimeError("URDF is missing joints: {}".format(", ".join(missing)))
        indices = [joint_by_name[name][0] for name in self.JOINT_NAMES]
        lowers = [joint_by_name[name][1][8] for name in self.JOINT_NAMES]
        uppers = [joint_by_name[name][1][9] for name in self.JOINT_NAMES]
        ranges = [upper - lower for lower, upper in zip(lowers, uppers)]
        velocities = [joint_by_name[name][1][11] for name in self.JOINT_NAMES]
        return indices, lowers, uppers, ranges, velocities

    def _find_link(self, link_name):
        for index in range(bullet.getNumJoints(self.robot_id, physicsClientId=self.client)):
            info = bullet.getJointInfo(self.robot_id, index, physicsClientId=self.client)
            if info[12].decode("utf-8") == link_name:
                return index
        raise RuntimeError("URDF is missing link {}".format(link_name))

    def _create_box(self, size, position, rgba, orientation=(0.0, 0.0, 0.0, 1.0)):
        half = [value / 2.0 for value in size]
        collision = bullet.createCollisionShape(
            bullet.GEOM_BOX, halfExtents=half, physicsClientId=self.client
        )
        visual = bullet.createVisualShape(
            bullet.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=self.client
        )
        return bullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            baseOrientation=orientation,
            physicsClientId=self.client,
        )

    def _create_workcell(self):
        table_position = [
            self.table_center_xy[0],
            self.table_center_xy[1],
            self.table_top_z - self.table_size[2] / 2.0,
        ]
        table_id = self._create_box(self.table_size, table_position, [0.55, 0.42, 0.28, 1.0])

        bar_position = [
            self.bar_center_xy[0],
            self.bar_center_xy[1],
            self.table_top_z + self.bar_size[2] / 2.0,
        ]
        bar_orientation = bullet.getQuaternionFromEuler([0.0, 0.0, self.bar_yaw])
        bar_id = self._create_box(
            self.bar_size, bar_position, [0.30, 0.32, 0.35, 1.0], bar_orientation
        )

        collision = bullet.createCollisionShape(
            bullet.GEOM_SPHERE, radius=self.obstacle_radius, physicsClientId=self.client
        )
        visual = bullet.createVisualShape(
            bullet.GEOM_SPHERE,
            radius=self.obstacle_radius,
            rgbaColor=[0.80, 0.10, 0.10, 1.0],
            physicsClientId=self.client,
        )
        obstacle_id = bullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=self.obstacle_center,
            physicsClientId=self.client,
        )
        return table_id, bar_id, obstacle_id

    def _show_goal_marker(self, position):
        if self._goal_marker_id is not None:
            bullet.removeBody(self._goal_marker_id, physicsClientId=self.client)
        visual = bullet.createVisualShape(
            bullet.GEOM_SPHERE,
            radius=0.025,
            rgbaColor=[0.10, 0.90, 0.20, 0.85],
            physicsClientId=self.client,
        )
        self._goal_marker_id = bullet.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self.client,
        )

    def _scan_pose(self, at_end=False):
        half_length = self.bar_size[0] / 2.0
        local_x = half_length - 0.01 if at_end else -half_length + 0.01
        cosine = math.cos(self.bar_yaw)
        sine = math.sin(self.bar_yaw)
        position = [
            self.bar_center_xy[0] + cosine * local_x,
            self.bar_center_xy[1] + sine * local_x,
            self.table_top_z + self.bar_size[2] + self.scan_standoff,
        ]
        orientation = bullet.getQuaternionFromEuler([0.0, math.pi, self.bar_yaw])
        return position, orientation

    def _ik(self, position, orientation):
        solution = bullet.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            position,
            orientation,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges,
            restPoses=self._joint_positions(),
            maxNumIterations=200,
            residualThreshold=1e-6,
            physicsClientId=self.client,
        )
        return [solution[index] for index in range(len(self.joint_indices))]

    def _move_to_scan_start(self):
        position, orientation = self._scan_pose(at_end=False)
        # A non-singular seed makes the elbow side deterministic across computers.
        seed = [0.0, -0.65, 0.0, 1.35, 0.0, -0.75, 0.0]
        for joint_index, value in zip(self.joint_indices, seed):
            bullet.resetJointState(self.robot_id, joint_index, value, physicsClientId=self.client)
        target = self._ik(position, orientation)
        for joint_index, value in zip(self.joint_indices, target):
            bullet.resetJointState(self.robot_id, joint_index, value, physicsClientId=self.client)
        self._hold(target)

    def _hold(self, targets):
        self._last_joint_target = list(targets)
        bullet.setJointMotorControlArray(
            self.robot_id,
            self.joint_indices,
            bullet.POSITION_CONTROL,
            targetPositions=targets,
            forces=[self.max_joint_force] * len(self.joint_indices),
            positionGains=[0.50] * len(self.joint_indices),
            velocityGains=[1.0] * len(self.joint_indices),
            physicsClientId=self.client,
        )

    def _joint_positions(self):
        return [
            state[0]
            for state in bullet.getJointStates(
                self.robot_id, self.joint_indices, physicsClientId=self.client
            )
        ]

    def _path_callback(self, message):
        if not message.poses:
            rospy.logwarn("Ignoring an empty planner path")
            return
        if message.header.frame_id and message.header.frame_id != self.frame_id:
            rospy.logerr(
                "Ignoring path in %s; simulator expects %s (no TF conversion is applied)",
                message.header.frame_id,
                self.frame_id,
            )
            return
        with self._path_lock:
            if self._controller_state in ("planning", "moving_to_start", "executing"):
                rospy.logwarn("Ignoring a new path while a simulation task is active")
                return
            self._pending_path = message
            self._controller_state = "planning"
            self._status_message = "Compiling shared joint trajectory"
        rospy.loginfo("Queued %d Cartesian planner waypoints", len(message.poses))

    def _compile_pending_path(self, message):
        task_path = [pose.pose for pose in message.poses]
        try:
            positions = np.asarray([
                [pose.position.x, pose.position.y, pose.position.z]
                for pose in task_path
            ], dtype=float)
            axes = np.asarray([
                self.trajectory_compiler.tool_z_from_quaternion([
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ])
                for pose in task_path
            ])
            current = np.asarray(self._joint_positions(), dtype=float)
            prepared = self.trajectory_compiler.compile(positions, axes, current)
        except (TrajectoryValidationError, ValueError, RuntimeError) as error:
            self._prepared_plan = None
            self._active_segment = None
            self._controller_state = "failed"
            self._status_message = str(error)
            rospy.logerr("Shared trajectory compilation failed: %s", error)
            return

        self._task_path = task_path
        self._prepared_plan = prepared
        self._active_segment = prepared["approach"]
        self._segment_started = self._sim_time
        self._path_index = 0
        self._controller_state = "moving_to_start"
        self._status_message = "Executing shared approach joint trajectory"
        self._target_pose = task_path[0]
        self._show_goal_marker(positions[-1].tolist())
        rospy.loginfo(
            "Shared compiler prepared %d approach and %d task joint samples",
            len(prepared["approach"]["position"]),
            len(prepared["task"]["position"]),
        )

    def _publish_placeholder_request(self, _event):
        if rospy.is_shutdown():
            return
        start_position, start_orientation = self._end_effector_pose()
        goal_position = [
            start_position[0] + math.cos(self.bar_yaw) * self.placeholder_distance,
            start_position[1] + math.sin(self.bar_yaw) * self.placeholder_distance,
            start_position[2],
        ]
        # Use the pose actually reached by the initialized robot.  This keeps
        # the placeholder test focused on translation and avoids asking the IK
        # solver to rotate into a joint-limit singularity at the same time.
        goal_orientation = start_orientation
        self._show_goal_marker(goal_position)
        start = self._pose_message(start_position, start_orientation)
        goal = self._pose_message(goal_position, goal_orientation)
        self.start_publisher.publish(start)
        self.goal_publisher.publish(goal)
        try:
            rospy.wait_for_service("/straight_line_planner/plan", timeout=5.0)
            response = rospy.ServiceProxy("/straight_line_planner/plan", Trigger)()
            if not response.success:
                rospy.logerr("Placeholder planner rejected the request: %s", response.message)
            else:
                rospy.loginfo("Placeholder planner: %s", response.message)
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Could not run placeholder planner: %s", error)

    def _pose_message(self, position, orientation):
        message = PoseStamped()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = self.frame_id
        message.pose.position.x, message.pose.position.y, message.pose.position.z = position
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = orientation
        return message

    def _end_effector_pose(self):
        state = bullet.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        return list(state[4]), list(state[5])

    def _control_step(self):
        with self._path_lock:
            pending = self._pending_path
            self._pending_path = None
        if pending is not None:
            self._compile_pending_path(pending)

        with self._path_lock:
            if (
                self._controller_state not in ("moving_to_start", "executing")
                or self._active_segment is None
            ):
                self._hold(self._last_joint_target)
                return
            elapsed = self._sim_time - self._segment_started
            target = self.trajectory_compiler.sample_position(
                self._active_segment, elapsed
            )
            self._hold(target)

            if self._controller_state == "executing":
                times = np.asarray(self._active_segment["time"], dtype=float)
                self._path_index = int(
                    np.clip(np.searchsorted(times, elapsed, side="right") - 1, 0, len(times) - 1)
                )
                self._target_pose = self._task_path[self._path_index]

            duration = self._active_segment["duration"]
            if elapsed < duration:
                return
            actual = np.asarray(self._joint_positions(), dtype=float)
            final_target = np.asarray(self._active_segment["position"][-1], dtype=float)
            joint_error = float(np.max(np.abs(actual - final_target)))
            if joint_error > self.joint_settle_tolerance:
                if elapsed - duration <= self.segment_settle_timeout:
                    return
                failed_phase = self._controller_state
                self._controller_state = "failed"
                self._status_message = (
                    "{} joint trajectory did not settle (max error {:.4f} rad)".format(
                        "Approach" if failed_phase == "moving_to_start" else "Task",
                        joint_error,
                    )
                )
                self._stop_recording(status="failed")
                rospy.logerr(self._status_message)
                return

            if self._controller_state == "moving_to_start":
                if self._task_record_requested:
                    self._start_recording()
                self._active_segment = self._prepared_plan["task"]
                self._segment_started = self._sim_time
                self._path_index = 0
                self._controller_state = "executing"
                self._status_message = "Executing shared task joint trajectory"
                self._target_pose = self._task_path[0]
                rospy.loginfo("Task start reached; executing shared joint trajectory")
            else:
                self._path_index = len(self._task_path) - 1
                self._controller_state = "complete"
                self._status_message = "Task complete"
                self._stop_recording()
                rospy.loginfo("Shared joint trajectory execution complete")

    def _publish_state(self):
        now = rospy.Time.now()
        states = bullet.getJointStates(
            self.robot_id, self.joint_indices, physicsClientId=self.client
        )
        joint_state = JointState()
        joint_state.header.stamp = now
        joint_state.name = list(self.JOINT_NAMES)
        joint_state.position = [state[0] for state in states]
        joint_state.velocity = [state[1] for state in states]
        joint_state.effort = [state[3] for state in states]
        self.joint_publisher.publish(joint_state)

        bar_position, bar_orientation = bullet.getBasePositionAndOrientation(
            self.bar_id, physicsClientId=self.client
        )
        obstacle_position, obstacle_orientation = bullet.getBasePositionAndOrientation(
            self.obstacle_id, physicsClientId=self.client
        )
        bar_message = self._pose_message(bar_position, bar_orientation)
        bar_message.header.stamp = now
        obstacle_message = self._pose_message(obstacle_position, obstacle_orientation)
        obstacle_message.header.stamp = now
        self.bar_publisher.publish(bar_message)
        self.obstacle_publisher.publish(obstacle_message)

        if self._sim_time - self._last_status >= 0.1:
            status = {
                "mode": "pybullet",
                "controller": self._controller_state,
                "message": self._status_message,
                "path_index": self._path_index,
                "path_length": len(self._task_path),
                "trajectory_metrics": (
                    self._prepared_plan["metrics"] if self._prepared_plan else None
                ),
                "recording": self._recording,
                "run_directory": self._run_directory,
                "video_path": self._video_path,
                "sim_time": self._sim_time,
            }
            self.status_publisher.publish(String(data=json.dumps(status, sort_keys=True)))
            self._last_status = self._sim_time

        if self._recording:
            self._write_log_row(now, states)

    def _start_recording(self):
        if self._recording:
            return
        output_root = os.path.abspath(os.path.expanduser(rospy.get_param("~output_root", "/data/sim_runs")))
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self._run_directory = os.path.join(output_root, run_name)
        os.makedirs(self._run_directory, exist_ok=False)
        self._log_file = open(os.path.join(self._run_directory, "trajectory.csv"), "w", newline="")
        self._log_writer = csv.DictWriter(self._log_file, fieldnames=self.LOG_FIELDS)
        self._log_writer.writeheader()
        self._log_file.flush()
        self._recording = True
        self._video_finished = False
        self._video_complete_time = None
        self._video_path = None
        self._write_metadata("running")
        rospy.loginfo("Recording simulation data in %s", self._run_directory)

    def _camera_rgb(self):
        width, height = self.video_size
        view = bullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[-0.42, 0.0, 0.42],
            distance=1.25,
            yaw=38.0,
            pitch=-24.0,
            roll=0.0,
            upAxisIndex=2,
        )
        projection = bullet.computeProjectionMatrixFOV(
            fov=55.0,
            aspect=float(width) / float(height),
            nearVal=0.05,
            farVal=3.0,
        )
        image = bullet.getCameraImage(
            width,
            height,
            viewMatrix=view,
            projectionMatrix=projection,
            renderer=bullet.ER_TINY_RENDERER,
            shadow=1,
            physicsClientId=self.client,
        )
        return image[2][:, :, :3].tobytes()

    def _begin_video(self):
        if not self.render_video or not self._recording or self._video_finished:
            return False
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            rospy.logerr("render_video is enabled but ffmpeg is not installed")
            self._video_finished = True
            return False
        width, height = self.video_size
        self._video_path = os.path.join(self._run_directory, "goal_reaching.mp4")
        self._video_log_file = open(
            os.path.join(self._run_directory, "video_ffmpeg.log"), "w"
        )
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            "{}x{}".format(width, height),
            "-framerate",
            str(self.video_fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            self._video_path,
        ]
        self._video_process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._video_log_file,
        )
        self._video_active = True
        first_frame = self._camera_rgb()
        for _ in range(max(1, int(round(self.video_initial_hold * self.video_fps)))):
            self._video_process.stdin.write(first_frame)
        rospy.loginfo("Rendering goal-reaching video to %s", self._video_path)
        return True

    def _capture_video(self):
        if not self.render_video or not self._recording or self._video_finished:
            return
        if self._controller_state == "executing" and not self._video_active:
            if not self._begin_video():
                return
        if not self._video_active:
            return
        try:
            self._video_process.stdin.write(self._camera_rgb())
        except (BrokenPipeError, OSError) as error:
            rospy.logerr("Video encoder stopped unexpectedly: %s", error)
            self._finish_video()
            return
        if self._controller_state == "complete":
            if self._video_complete_time is None:
                self._video_complete_time = self._sim_time
            elif self._sim_time - self._video_complete_time >= self.video_post_roll:
                self._finish_video()

    def _finish_video(self):
        if self._video_process is None:
            return
        process = self._video_process
        self._video_process = None
        self._video_active = False
        self._video_finished = True
        try:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            return_code = process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            return_code = process.wait(timeout=2.0)
        finally:
            self._video_log_file.close()
            self._video_log_file = None
        if return_code == 0:
            rospy.loginfo("Goal-reaching video complete: %s", self._video_path)
        else:
            rospy.logerr("ffmpeg exited with code %d; see video_ffmpeg.log", return_code)

    def _stop_recording(self, status="complete"):
        if not self._recording:
            return
        self._finish_video()
        self._recording = False
        self._write_metadata(status)
        self._log_file.flush()
        self._log_file.close()
        self._log_file = None
        self._log_writer = None

    def _set_recording(self, request):
        try:
            if request.data:
                self._start_recording()
                return SetBoolResponse(True, "Recording to {}".format(self._run_directory))
            self._stop_recording()
            return SetBoolResponse(True, "Recording stopped")
        except (OSError, ValueError) as error:
            return SetBoolResponse(False, str(error))

    def _set_task_recording(self, request):
        if self._controller_state in ("planning", "moving_to_start", "executing"):
            return SetBoolResponse(False, "Cannot change recording during a task")
        self._task_record_requested = bool(request.data)
        return SetBoolResponse(
            True,
            "Task recording enabled" if self._task_record_requested else "Task recording disabled",
        )

    def _write_metadata(self, status):
        metadata = {
            "status": status,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "simulator": "pybullet",
            "pybullet_api_version": bullet.getAPIVersion(),
            "frame_id": self.frame_id,
            "physics_hz": self.physics_hz,
            "publish_hz": self.publish_hz,
            "render_video": self.render_video,
            "video_fps": self.video_fps,
            "video_size": self.video_size,
            "video_slowdown": self.physics_hz / float(self.video_fps),
            "video_path": self._video_path,
            "table_top_z": self.table_top_z,
            "table_size": self.table_size,
            "bar_size": self.bar_size,
            "bar_center_xy": self.bar_center_xy,
            "bar_yaw": self.bar_yaw,
            "obstacle_center": self.obstacle_center,
            "obstacle_radius": self.obstacle_radius,
            "placeholder_distance": self.placeholder_distance,
            "planner": "stage_placeholder_planner/straight_line_planner",
            "trajectory_compiler": "stage_cartesian_trajectory/CartesianTrajectoryCompiler",
            "orientation_control": "position_plus_tool_z",
            "trajectory_metrics": (
                self._prepared_plan["metrics"] if self._prepared_plan else None
            ),
        }
        with open(os.path.join(self._run_directory, "metadata.json"), "w") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _write_log_row(self, now, states):
        position, orientation = self._end_effector_pose()
        if self._target_pose is None:
            target_position = [float("nan")] * 3
            target_orientation = [float("nan")] * 4
        else:
            target_position = [
                self._target_pose.position.x,
                self._target_pose.position.y,
                self._target_pose.position.z,
            ]
            target_orientation = [
                self._target_pose.orientation.x,
                self._target_pose.orientation.y,
                self._target_pose.orientation.z,
                self._target_pose.orientation.w,
            ]
        contacts = bullet.getContactPoints(bodyA=self.robot_id, physicsClientId=self.client)
        table_contacts = bullet.getContactPoints(
            bodyA=self.robot_id, bodyB=self.table_id, physicsClientId=self.client
        )
        bar_contacts = bullet.getContactPoints(
            bodyA=self.robot_id, bodyB=self.bar_id, physicsClientId=self.client
        )
        obstacle_contacts = bullet.getContactPoints(
            bodyA=self.robot_id, bodyB=self.obstacle_id, physicsClientId=self.client
        )
        values = (
            [
                datetime.now(timezone.utc).isoformat(),
                now.to_sec(),
                self._sim_time,
                self._controller_state,
                self._path_index,
            ]
            + [state[0] for state in states]
            + [state[1] for state in states]
            + list(self._last_joint_target)
            + position
            + orientation
            + target_position
            + target_orientation
            + [len(contacts), len(table_contacts), len(bar_contacts), len(obstacle_contacts)]
        )
        self._log_writer.writerow(dict(zip(self.LOG_FIELDS, values)))
        if int(self._sim_time * self.publish_hz) % int(max(1.0, self.publish_hz)) == 0:
            self._log_file.flush()

    def run(self):
        rate = rospy.Rate(self.physics_hz)
        while not rospy.is_shutdown():
            self._control_step()
            bullet.stepSimulation(physicsClientId=self.client)
            self._sim_time += 1.0 / self.physics_hz
            self._capture_video()
            if self._sim_time - self._last_publish >= 1.0 / self.publish_hz:
                self._publish_state()
                self._last_publish = self._sim_time
            rate.sleep()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop_recording()
        if bullet.isConnected(self.client):
            bullet.disconnect(physicsClientId=self.client)


if __name__ == "__main__":
    rospy.init_node("iiwa_pybullet_sim")
    try:
        IiwaPyBulletSim().run()
    except rospy.ROSInterruptException:
        pass
