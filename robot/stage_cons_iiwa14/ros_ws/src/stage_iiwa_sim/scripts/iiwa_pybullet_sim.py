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
from collections import deque
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
from stage_constraint_planner.optimizer import (
    bar_lateral_centerline_offset,
    obstacle_clearance as evaluate_obstacle_clearance,
)
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse


class IiwaPyBulletSim:
    TASK_IDS = {"BarClean"}
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
        self._task_id = "BarClean"
        self._constraint_source = "true"
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
        self.motion_speed_scale = float(rospy.get_param("~motion_speed_scale", 1.5))
        if not math.isfinite(self.motion_speed_scale) or self.motion_speed_scale <= 0.0:
            raise ValueError("motion_speed_scale must be positive and finite")
        self.max_joint_step = float(rospy.get_param("~max_joint_step_rad", 0.15))
        self.velocity_scale = self.motion_speed_scale * float(
            rospy.get_param("~velocity_scale", 0.20)
        )
        self.acceleration_limit = self.motion_speed_scale ** 2 * float(
            rospy.get_param("~acceleration_limit_rad_s2", 1.00)
        )
        self.approach_speed = None
        self.approach_position_tolerance = None
        self.approach_joint_bridge_limit = None
        self.minimum_approach_z = float(rospy.get_param("~minimum_approach_z", 0.20))
        self.task_speed = None
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

        self.scene_config_path = rospy.get_param(
            "~scene_config",
            os.path.join(
                rospkg.RosPack().get_path("stage_iiwa_sim"),
                "config",
                "demo_scene.json",
            ),
        )
        self._load_demo_scene(self.scene_config_path)
        self.scan_standoff = float(rospy.get_param("~scan_standoff", 0.08))
        self.auto_goal_distance = float(rospy.get_param("~auto_goal_distance", 0.15))

        self._path_lock = threading.Lock()
        self._scene_request_lock = threading.Lock()
        self._scene_apply_event = threading.Event()
        self._pending_scene_snapshot = None
        self._scene_apply_error = None
        self._pending_path = None
        self._awaiting_path_metadata = None
        self._orientation_constraints = {}
        self._task_path = []
        self._path_index = -1
        self._prepared_plan = None
        self._active_segment = None
        self._segment_started = 0.0
        self._controller_state = "idle"
        self._task_sequence = 0
        self._abort_requested = False
        self._status_message = "Simulator ready"
        self._target_pose = None
        self._sim_time = 0.0
        self._last_publish = -math.inf
        self._last_status = -math.inf
        self._closed = False
        self._goal_marker_id = None
        self._visualization_trace = deque(maxlen=4000)
        self._feature_trace = deque(maxlen=2400)
        self._feature_started = None
        self._last_feature_sample = -math.inf
        self._feature_terminal_captured = False

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
            approach_position_tolerance=self.approach_position_tolerance,
            approach_joint_bridge_limit=self.approach_joint_bridge_limit,
            minimum_approach_z=self.minimum_approach_z,
            approach_clearance_z=float(
                rospy.get_param(
                    "~approach_clearance_z",
                    self.table_top_z + 2.0 * max(self.obstacle_radii) + 0.08,
                )
            ),
        )
        self.table_id, self.bar_id, self.obstacle_ids = self._create_workcell()
        self._move_to_scan_start()

        self.joint_publisher = rospy.Publisher("joint_states", JointState, queue_size=5)
        self.status_publisher = rospy.Publisher("sim/status", String, queue_size=2, latch=True)
        self.bar_publisher = rospy.Publisher(
            "/vrpn_client_node/baiyu_bar/pose_from_iiwa14", PoseStamped, queue_size=2, latch=True
        )
        self.obstacle_publishers = [
            rospy.Publisher(topic, PoseStamped, queue_size=2, latch=True)
            for topic in (
                "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14",
                "/vrpn_client_node/baiyu_obs_bar_b/pose_from_iiwa14",
            )
        ]
        self.start_publisher = rospy.Publisher(
            "/stage_cons/planner/start", PoseStamped, queue_size=1, latch=True
        )
        self.goal_publisher = rospy.Publisher(
            "/stage_cons/planner/goal", PoseStamped, queue_size=1, latch=True
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
        self.path_subscriber = rospy.Subscriber(
            "/stage_cons/plan", Path, self._path_callback, queue_size=1
        )
        self.recording_service = rospy.Service("sim/set_recording", SetBool, self._set_recording)
        self.task_video_service = rospy.Service(
            "sim/set_task_video", SetBool, self._set_task_video
        )
        self.abort_service = rospy.Service("sim/abort", Trigger, self._abort_task)
        self.scene_snapshot_service = rospy.Service(
            "sim/apply_scene_snapshot", Trigger, self._apply_scene_snapshot
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
        self._video_next_frame_time = None
        self._task_video_requested = False

        rospy.on_shutdown(self.close)
        if self.auto_plan:
            self._auto_plan_timer = rospy.Timer(
                rospy.Duration(2.0), self._publish_auto_plan_request, oneshot=True
            )

        rospy.loginfo(
            "PyBullet iiwa14 ready (%s); Demo scene bar=(%.3f, %.3f) yaw=%.1f deg, obstacle=(%.3f, %.3f)",
            "GUI" if self.gui else "DIRECT",
            self.bar_reference_xy[0],
            self.bar_reference_xy[1],
            math.degrees(self.bar_yaw),
            self.obstacle_centers[0][0],
            self.obstacle_centers[0][1],
        )

    @staticmethod
    def _rotation_from_quaternion(quaternion):
        values = np.asarray(quaternion, dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("Scene quaternion must contain four finite values")
        norm = float(np.linalg.norm(values))
        if norm <= 1e-12:
            raise ValueError("Scene quaternion cannot be zero")
        x, y, z, w = values / norm
        return np.asarray([
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ])

    @staticmethod
    def _quaternion_from_rotation(matrix):
        matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            values = [
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            ]
        else:
            index = int(np.argmax(np.diag(matrix)))
            if index == 0:
                scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
                values = [
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                ]
            elif index == 1:
                scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
                values = [
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                ]
            else:
                scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
                values = [
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                ]
        values = np.asarray(values, dtype=float)
        return (values / np.linalg.norm(values)).tolist()

    def _load_demo_scene(self, path):
        with open(path, "r") as handle:
            config = json.load(handle)

        table = config["table"]
        bar = config["bar"]
        obstacles = list(config["obstacles"])
        planning_obstacle = dict(config["planning_obstacle"])
        bar_pose = np.asarray(bar["locked_pose_robot"], dtype=float)
        obstacle_poses = np.asarray(
            [value["locked_pose_robot"] for value in obstacles], dtype=float
        )
        if bar_pose.shape != (7,) or obstacle_poses.ndim != 2 or obstacle_poses.shape[1] != 7:
            raise ValueError("Demo scene object poses must contain seven values")
        if len(obstacle_poses) != 2:
            raise ValueError("The current station expects exactly two scene obstacles")
        obstacle_index_by_name = {
            str(value["name"]): index for index, value in enumerate(obstacles)
        }
        self.planning_obstacle_type = str(planning_obstacle.get("type"))
        if self.planning_obstacle_type == "circle":
            obstacle_name = str(planning_obstacle.get("obstacle"))
            if obstacle_name not in obstacle_index_by_name:
                raise ValueError("Demo scene circle obstacle is not published")
            self.planning_obstacle_indices = [
                obstacle_index_by_name[obstacle_name]
            ]
        elif self.planning_obstacle_type == "capsule":
            endpoint_names = [
                str(value) for value in planning_obstacle["endpoint_obstacles"]
            ]
            if (
                len(endpoint_names) != 2
                or len(set(endpoint_names)) != 2
                or any(name not in obstacle_index_by_name for name in endpoint_names)
            ):
                raise ValueError("Demo scene capsule endpoints are invalid")
            self.planning_obstacle_indices = [
                obstacle_index_by_name[name] for name in endpoint_names
            ]
        else:
            raise ValueError("Demo scene planning_obstacle must be circle or capsule")

        bar_rotation = self._rotation_from_quaternion(bar_pose[3:])
        self.bar_reference_position = bar_pose[:3].tolist()
        self.bar_reference_orientation = self._quaternion_from_rotation(bar_rotation)
        self.obstacle_reference_positions = obstacle_poses[:, :3].tolist()
        self.obstacle_reference_orientations = [
            self._quaternion_from_rotation(
                self._rotation_from_quaternion(obstacle_pose[3:])
            )
            for obstacle_pose in obstacle_poses
        ]
        axis_local = np.asarray(bar["axis_local"], dtype=float)
        bar_axis = bar_rotation @ axis_local
        bar_axis[2] = 0.0
        axis_norm = float(np.linalg.norm(bar_axis[:2]))
        if axis_norm <= 1e-12:
            raise ValueError("Demo bar axis is vertical in the robot frame")
        bar_axis /= axis_norm

        self.scene_source = dict(config["source"])
        self.scene_snapshot_source = "fixed_demo"
        self.scene_bar_live = False
        self.scene_obstacles_live = [False] * len(obstacles)
        self.scene_locked_bar_pose = bar_pose.tolist()
        self.scene_locked_obstacle_poses = obstacle_poses.tolist()
        self.table_top_z = float(table["top_z"])
        self.table_size = [float(value) for value in table["size"]]
        self.bar_reference_xy = bar_pose[:2].tolist()
        self.bar_axis_xy = bar_axis[:2].tolist()
        self.bar_lateral_xy = [-self.bar_axis_xy[1], self.bar_axis_xy[0]]
        self.bar_outline_u = [float(value) for value in bar["outline_u"]]
        self.bar_outline_v = [float(value) for value in bar["outline_v"]]
        self.bar_lateral_centerline = dict(bar["lateral_centerline"])
        middle_u = 0.5 * sum(self.bar_outline_u)
        middle_v = 0.5 * sum(self.bar_outline_v)
        self.bar_center_xy = [
            self.bar_reference_xy[index]
            + middle_u * self.bar_axis_xy[index]
            + middle_v * self.bar_lateral_xy[index]
            for index in range(2)
        ]
        self.bar_yaw = math.atan2(self.bar_axis_xy[1], self.bar_axis_xy[0])
        self.bar_size = [
            self.bar_outline_u[1] - self.bar_outline_u[0],
            self.bar_outline_v[1] - self.bar_outline_v[0],
            float(bar["height"]),
        ]
        self.obstacle_radii = [float(value["radius"]) for value in obstacles]
        planning_radii = np.asarray(
            [self.obstacle_radii[index] for index in self.planning_obstacle_indices]
        )
        if (
            np.any(planning_radii <= 0.0)
            or not np.allclose(planning_radii, planning_radii[0], atol=1e-9)
        ):
            raise ValueError("Planning obstacle must have one positive radius")
        self.planning_obstacle_radius = float(planning_radii[0])
        self.obstacle_centers = [
            [
                float(obstacle_pose[0]),
                float(obstacle_pose[1]),
                self.table_top_z + obstacle_radius,
            ]
            for obstacle_pose, obstacle_radius in zip(
                obstacle_poses, self.obstacle_radii
            )
        ]
        obstacle_midpoint = np.mean(np.asarray(self.obstacle_centers)[:, :2], axis=0)
        self.table_center_xy = [
            0.5 * (self.bar_center_xy[index] + obstacle_midpoint[index])
            for index in range(2)
        ]
        self.initial_ik_seed = [float(value) for value in config["initial_ik_seed"]]
        if len(self.initial_ik_seed) != 7:
            raise ValueError("Demo scene initial IK seed must contain seven joints")

        feature_runtime = config["feature_runtime"]
        planner_config_dir = rospy.get_param("~task_definition_dir", "/task_definitions")
        self._task_config_paths = {}
        self._feature_definitions = {}
        for task_id, filename in (("BarClean", "bar_clean_true.json"),):
            self._task_config_paths[task_id] = os.path.join(
                planner_config_dir, filename
            )
        self.feature_sample_hz = float(feature_runtime["sample_hz"])
        self.feature_tool_axis_local = np.asarray(
            feature_runtime["tool_axis_local"], dtype=float
        )
        if (
            self.feature_tool_axis_local.shape != (3,)
            or self.feature_sample_hz <= 0.0
        ):
            raise ValueError("Demo feature definition has invalid dimensions")
        self.feature_tool_axis_local /= np.linalg.norm(self.feature_tool_axis_local)
        self._apply_task_feature_definition(self._task_id)
        self.bar_axis_local = axis_local

    def _apply_task_feature_definition(self, task_id):
        config_path = self._task_config_paths[task_id]
        with open(config_path, "r", encoding="utf-8") as stream:
            task_config = json.load(stream)
        feature_names = [str(value) for value in task_config["visualization_features"]]
        units = task_config["feature_units"]
        self._feature_definitions[task_id] = {
            "source": "task_definitions/{}".format(os.path.basename(config_path)),
            "schema": [{"name": name, "unit": str(units[name])} for name in feature_names],
            "true_constraints": {
                "bar_axial_offset_reference": float(task_config.get("bar_axial_offset_reference", 0.0))
            },
            "constraint_specs": [dict(value) for value in task_config["constraint_terms"]],
            "table_surface_point": task_config["table_surface_point"],
            "table_normal": task_config["table_normal"],
            "execution": dict(task_config["execution"]),
        }
        definition = self._feature_definitions[task_id]
        self.feature_definition_source = str(definition["source"])
        self.feature_schema = [dict(value) for value in definition["schema"]]
        self.feature_true_constraints = dict(definition["true_constraints"])
        self.feature_constraint_specs = [
            dict(value) for value in definition["constraint_specs"]
        ]
        self.feature_table_surface_point = np.asarray(
            definition["table_surface_point"], dtype=float
        )
        self.feature_table_normal = np.asarray(
            definition["table_normal"], dtype=float
        )
        if (
            self.feature_table_surface_point.shape != (3,)
            or self.feature_table_normal.shape != (3,)
            or np.linalg.norm(self.feature_table_normal) <= 1e-12
        ):
            raise ValueError("Planner feature definition has invalid table geometry")
        self.feature_table_normal /= np.linalg.norm(self.feature_table_normal)
        execution = definition["execution"]
        self.approach_speed = self.motion_speed_scale * float(execution["approach_speed_mps"])
        self.task_speed = self.motion_speed_scale * float(execution["task_speed_mps"])
        self.approach_position_tolerance = float(
            execution["approach_position_tolerance_m"]
        )
        self.approach_joint_bridge_limit = float(
            execution["approach_joint_bridge_limit_rad"]
        )
        if hasattr(self, "trajectory_compiler"):
            self.trajectory_compiler.set_task_speeds(self.approach_speed, self.task_speed)
            self.trajectory_compiler.set_approach_position_tolerance(
                self.approach_position_tolerance
            )
            self.trajectory_compiler.set_approach_joint_bridge_limit(
                self.approach_joint_bridge_limit
            )

    def _task_callback(self, message):
        task_id = str(message.data).strip()
        if task_id not in self.TASK_IDS:
            rospy.logerr("Ignoring unknown task id %s", task_id)
            return
        with self._path_lock:
            if self._controller_state in ("planning", "moving_to_start", "executing"):
                rospy.logerr("Ignoring task switch to %s during simulation execution", task_id)
                return
            self._task_id = task_id
            self._constraint_source = str(
                rospy.get_param("/stage_constraint_planner/constraint_source", "true")
            )
            self._apply_task_feature_definition(task_id)
            self._pending_path = None
            self._awaiting_path_metadata = None
            self._feature_trace.clear()
        rospy.loginfo("Simulation task selected: %s", task_id)

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

    @staticmethod
    def _validated_scene_snapshot(payload):
        if not isinstance(payload, dict):
            raise ValueError("Simulation scene snapshot must be an object")
        bar = payload.get("bar")
        obstacles = payload.get("obstacles")
        if not isinstance(bar, dict) or not isinstance(obstacles, list) or not obstacles:
            raise ValueError("Simulation scene snapshot is missing bar or obstacles")
        try:
            pivot = np.asarray(bar["pivot"], dtype=float)
            axis = np.asarray(bar["axis"], dtype=float)
            centers = np.asarray(
                [obstacle["center"] for obstacle in obstacles], dtype=float
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Simulation scene snapshot has invalid geometry") from error
        if (
            pivot.shape != (2,)
            or axis.shape != (2,)
            or centers.shape != (2, 2)
            or not np.all(np.isfinite(np.concatenate((pivot, axis, centers.reshape(-1)))))
            or np.max(np.abs(np.concatenate((pivot, centers.reshape(-1))))) > 2.0
        ):
            raise ValueError("Simulation scene snapshot has invalid geometry")
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-12:
            raise ValueError("Simulation scene snapshot has a zero bar axis")
        axis /= axis_norm
        return {
            "source": str(payload.get("source", "unknown")),
            "bar": {
                "pivot": pivot.tolist(),
                "axis": axis.tolist(),
                "lateral_centerline": dict(
                    bar.get("lateral_centerline", {"type": "straight"})
                ),
                "live": bar.get("live") is True,
            },
            "obstacles": [
                {
                    "center": center.tolist(),
                    "live": obstacle.get("live") is True,
                }
                for center, obstacle in zip(centers, obstacles)
            ],
        }

    def _apply_scene_snapshot(self, _request):
        with self._scene_request_lock:
            with self._path_lock:
                if self._controller_state in (
                    "planning", "moving_to_start", "executing"
                ):
                    return TriggerResponse(
                        False, "Cannot replace the scene during simulation execution"
                    )
            parameter = "~pending_scene_snapshot"
            try:
                snapshot = self._validated_scene_snapshot(
                    rospy.get_param(parameter)
                )
            except (KeyError, ValueError) as error:
                return TriggerResponse(False, str(error))
            finally:
                if rospy.has_param(parameter):
                    rospy.delete_param(parameter)

            self._scene_apply_error = None
            self._scene_apply_event.clear()
            with self._path_lock:
                self._pending_scene_snapshot = snapshot
            if not self._scene_apply_event.wait(timeout=3.0):
                return TriggerResponse(False, "PyBullet scene update timed out")
            if self._scene_apply_error:
                return TriggerResponse(False, self._scene_apply_error)
            return TriggerResponse(
                True,
                "Applied {} simulation scene snapshot".format(
                    snapshot["source"]
                ),
            )

    def _publish_scene_poses(self, stamp):
        bar_message = self._pose_message(
            self.bar_reference_position, self.bar_reference_orientation
        )
        bar_message.header.stamp = stamp
        self.bar_publisher.publish(bar_message)
        for publisher, position, orientation in zip(
            self.obstacle_publishers,
            self.obstacle_reference_positions,
            self.obstacle_reference_orientations,
        ):
            message = self._pose_message(position, orientation)
            message.header.stamp = stamp
            publisher.publish(message)

    def _apply_scene_snapshot_now(self, snapshot):
        pivot = list(snapshot["bar"]["pivot"])
        axis = list(snapshot["bar"]["axis"])
        centers = [list(value["center"]) for value in snapshot["obstacles"]]
        self.bar_reference_xy = pivot
        self.bar_axis_xy = axis
        self.bar_lateral_xy = [-axis[1], axis[0]]
        self.bar_yaw = math.atan2(axis[1], axis[0])
        self.bar_reference_position[:2] = pivot
        self.bar_reference_orientation = list(
            bullet.getQuaternionFromEuler([0.0, 0.0, self.bar_yaw])
        )
        self.bar_lateral_centerline = dict(snapshot["bar"]["lateral_centerline"])
        middle_u = 0.5 * sum(self.bar_outline_u)
        middle_v = 0.5 * sum(self.bar_outline_v)
        self.bar_center_xy = [
            pivot[index]
            + middle_u * self.bar_axis_xy[index]
            + middle_v * self.bar_lateral_xy[index]
            for index in range(2)
        ]
        for index, center in enumerate(centers):
            self.obstacle_reference_positions[index][:2] = center
            self.obstacle_centers[index][:2] = center
        bar_body_position = [
            self.bar_center_xy[0],
            self.bar_center_xy[1],
            self.table_top_z + self.bar_size[2] / 2.0,
        ]
        bar_body_orientation = bullet.getQuaternionFromEuler(
            [0.0, 0.0, self.bar_yaw]
        )
        bullet.resetBasePositionAndOrientation(
            self.bar_id,
            bar_body_position,
            bar_body_orientation,
            physicsClientId=self.client,
        )
        for obstacle_id, center, orientation in zip(
            self.obstacle_ids,
            self.obstacle_centers,
            self.obstacle_reference_orientations,
        ):
            bullet.resetBasePositionAndOrientation(
                obstacle_id,
                center,
                orientation,
                physicsClientId=self.client,
            )
        if self._goal_marker_id is not None:
            bullet.removeBody(self._goal_marker_id, physicsClientId=self.client)
            self._goal_marker_id = None
        self.scene_snapshot_source = str(snapshot["source"])
        self.scene_bar_live = bool(snapshot["bar"]["live"])
        self.scene_obstacles_live = [
            bool(value["live"]) for value in snapshot["obstacles"]
        ]
        self._pending_path = None
        self._awaiting_path_metadata = None
        self._prepared_plan = None
        self._active_segment = None
        self._task_path = []
        self._path_index = -1
        self._target_pose = None
        self._visualization_trace.clear()
        self._feature_trace.clear()
        self._feature_started = None
        self._feature_terminal_captured = False
        self._controller_state = "idle"
        self._status_message = "Ready with frozen {} scene".format(
            self.scene_snapshot_source
        )
        self._publish_scene_poses(rospy.Time.now())
        rospy.loginfo(
            "Applied simulation scene snapshot (%s): bar=(%.3f, %.3f), "
            "obstacle=(%.3f, %.3f)",
            self.scene_snapshot_source,
            pivot[0], pivot[1], center[0], center[1],
        )

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

        obstacle_ids = []
        for center, radius in zip(self.obstacle_centers, self.obstacle_radii):
            collision = bullet.createCollisionShape(
                bullet.GEOM_SPHERE, radius=radius, physicsClientId=self.client
            )
            visual = bullet.createVisualShape(
                bullet.GEOM_SPHERE,
                radius=radius,
                rgbaColor=[0.80, 0.10, 0.10, 1.0],
                physicsClientId=self.client,
            )
            obstacle_ids.append(
                bullet.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=collision,
                    baseVisualShapeIndex=visual,
                    basePosition=center,
                    physicsClientId=self.client,
                )
            )
        return table_id, bar_id, obstacle_ids

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
        local_u = self.bar_outline_u[1] - 0.01 if at_end else self.bar_outline_u[0] + 0.01
        local_v = 0.5 * sum(self.bar_outline_v)
        position = [
            self.bar_reference_xy[0]
            + self.bar_axis_xy[0] * local_u
            + self.bar_lateral_xy[0] * local_v,
            self.bar_reference_xy[1]
            + self.bar_axis_xy[1] * local_u
            + self.bar_lateral_xy[1] * local_v,
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
        for joint_index, value in zip(self.joint_indices, self.initial_ik_seed):
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

    @staticmethod
    def _parse_orientation_constraints(message):
        try:
            payload = json.loads(message.data)
            if int(payload.get("schema_version", 0)) != 5:
                raise ValueError("unsupported schema_version")
            stamp_ns = int(payload["stamp_ns"])
            point_count = int(payload["point_count"])
            task_id = str(payload["task_id"])
            active = np.asarray(payload["tool_yaw_active"], dtype=int)
            raw_obstacle = dict(payload["approach_obstacle"])
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
        try:
            center = np.asarray(raw_obstacle["center"], dtype=float)
            table_normal = np.asarray(raw_obstacle["table_normal"], dtype=float)
            radius = float(raw_obstacle["radius"])
            clearance = float(raw_obstacle["clearance"])
            margin = float(raw_obstacle["margin"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Planner Stage-0 circle metadata is incomplete") from error
        if (
            str(raw_obstacle.get("type")) != "circle"
            or center.shape != (3,)
            or table_normal.shape != (3,)
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(table_normal))
            or np.linalg.norm(table_normal) <= 1e-12
            or not all(math.isfinite(value) for value in (radius, clearance, margin))
            or radius <= 0.0
            or clearance < 0.0
            or margin < 0.0
        ):
            raise ValueError("Planner Stage-0 circle metadata is invalid")
        approach_obstacle = {
            "type": "circle",
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

    def _queue_path(
        self,
        message,
        task_id,
        tool_yaw_active,
        approach_obstacle,
        stage_timing,
    ):
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
        with self._path_lock:
            if self._controller_state in ("planning", "moving_to_start", "executing"):
                rospy.logwarn("Ignoring a new path while a simulation task is active")
                return
            self._pending_path = (
                message,
                np.asarray(tool_yaw_active, dtype=bool).copy(),
                {
                    **approach_obstacle,
                    "center": list(approach_obstacle["center"]),
                    "table_normal": list(approach_obstacle["table_normal"]),
                },
                {
                    "boundaries": list(stage_timing["boundaries"]),
                    "transition_windows": [
                        list(window)
                        for window in stage_timing["transition_windows"]
                    ],
                    "speed_scale": float(stage_timing["speed_scale"]),
                    "ramp_before_m": float(stage_timing["ramp_before_m"]),
                    "task_start_ramp_m": float(stage_timing["task_start_ramp_m"]),
                },
            )
            self._awaiting_path_metadata = None
            self._visualization_trace.clear()
            self._feature_trace.clear()
            self._feature_started = None
            self._last_feature_sample = -math.inf
            self._feature_terminal_captured = False
            self._task_sequence += 1
            self._controller_state = "planning"
            self._status_message = "Compiling shared joint trajectory"
        rospy.loginfo("Queued %d Cartesian planner waypoints", len(message.poses))

    def _orientation_constraints_callback(self, message):
        try:
            stamp_ns, task_id, active, approach_obstacle, stage_timing = (
                self._parse_orientation_constraints(message)
            )
        except ValueError as error:
            rospy.logerr("%s", error)
            return
        waiting = None
        with self._path_lock:
            self._orientation_constraints[stamp_ns] = (
                task_id,
                active,
                approach_obstacle,
                stage_timing,
            )
            for stale_stamp in sorted(self._orientation_constraints)[:-4]:
                self._orientation_constraints.pop(stale_stamp, None)
            if (
                self._awaiting_path_metadata is not None
                and int(self._awaiting_path_metadata.header.stamp.to_nsec()) == stamp_ns
            ):
                waiting = self._awaiting_path_metadata
        if waiting is not None:
            self._queue_path(
                waiting, task_id, active, approach_obstacle, stage_timing
            )

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
            metadata = self._orientation_constraints.get(
                int(message.header.stamp.to_nsec())
            )
            if metadata is None:
                self._awaiting_path_metadata = message
                return
        self._queue_path(
            message, metadata[0], metadata[1], metadata[2], metadata[3]
        )

    def _compile_pending_path(
        self, message, tool_yaw_active, approach_obstacle, stage_timing
    ):
        task_path = [pose.pose for pose in message.poses]
        try:
            positions = np.asarray([
                [pose.position.x, pose.position.y, pose.position.z]
                for pose in task_path
            ], dtype=float)
            bases = [
                self.trajectory_compiler.tool_basis_from_quaternion([
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ])
                for pose in task_path
            ]
            x_axes = np.asarray([basis[0] for basis in bases])
            axes = np.asarray([basis[1] for basis in bases])
            current = np.asarray(self._joint_positions(), dtype=float)
            prepared = self.trajectory_compiler.compile(
                positions,
                axes,
                current,
                tool_x_axes=x_axes,
                tool_x_active=tool_yaw_active,
                approach_obstacle=approach_obstacle,
                stage_timing=stage_timing,
            )
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

    def _publish_auto_plan_request(self, _event):
        if rospy.is_shutdown():
            return
        start_position, start_orientation = self._end_effector_pose()
        goal_position = [
            start_position[0] + math.cos(self.bar_yaw) * self.auto_goal_distance,
            start_position[1] + math.sin(self.bar_yaw) * self.auto_goal_distance,
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
            rospy.wait_for_service("/stage_constraint_planner/plan", timeout=10.0)
            response = rospy.ServiceProxy("/stage_constraint_planner/plan", Trigger)()
            if not response.success:
                rospy.logerr("Stage constraint planner rejected the request: %s", response.message)
            else:
                rospy.loginfo("Stage constraint planner: %s", response.message)
        except (rospy.ROSException, rospy.ServiceException) as error:
            rospy.logerr("Could not run stage constraint planner: %s", error)

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
            pending_scene = self._pending_scene_snapshot
            self._pending_scene_snapshot = None
            abort_requested = self._abort_requested
            self._abort_requested = False
            if abort_requested:
                self._pending_path = None
            pending = self._pending_path
            self._pending_path = None
        if pending_scene is not None:
            try:
                self._apply_scene_snapshot_now(pending_scene)
                self._scene_apply_error = None
            except (KeyError, TypeError, ValueError, RuntimeError) as error:
                self._scene_apply_error = str(error)
                self._controller_state = "failed"
                self._status_message = "Scene snapshot failed: {}".format(error)
                rospy.logerr(self._status_message)
            finally:
                self._scene_apply_event.set()
        if abort_requested:
            current = self._joint_positions()
            self._prepared_plan = None
            self._active_segment = None
            self._path_index = -1
            self._target_pose = None
            self._controller_state = "aborted"
            self._status_message = "Task aborted; holding current joint position"
            self._hold(current)
            self._stop_recording(status="aborted")
            rospy.logwarn(self._status_message)
            return
        if pending is not None:
            self._compile_pending_path(
                pending[0], pending[1], pending[2], pending[3]
            )

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
                self._start_recording()
                self._active_segment = self._prepared_plan["task"]
                self._segment_started = self._sim_time
                self._path_index = 0
                self._visualization_trace.clear()
                self._feature_trace.clear()
                self._feature_started = self._sim_time
                self._last_feature_sample = -math.inf
                self._feature_terminal_captured = False
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

        self._publish_scene_poses(now)
        bar_body_position, bar_body_orientation = bullet.getBasePositionAndOrientation(
            self.bar_id, physicsClientId=self.client
        )
        obstacle_body_positions = [
            bullet.getBasePositionAndOrientation(
                obstacle_id, physicsClientId=self.client
            )[0]
            for obstacle_id in self.obstacle_ids
        ]

        ee_state = bullet.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client,
        )
        ee_position = ee_state[4]
        ee_orientation = ee_state[5]
        if self._controller_state == "executing":
            xy = [float(ee_position[0]), float(ee_position[1])]
            if not self._visualization_trace or math.dist(self._visualization_trace[-1], xy) >= 1e-4:
                self._visualization_trace.append(xy)

        active_feature_trace = self._controller_state == "executing"
        terminal_feature_sample = (
            self._controller_state in ("complete", "failed", "aborted")
            and not self._feature_terminal_captured
            and self._feature_started is not None
        )
        feature_due = (
            self._sim_time - self._last_feature_sample + 0.5 / self.physics_hz
            >= 1.0 / self.feature_sample_hz
        )
        if (active_feature_trace and feature_due) or terminal_feature_sample:
            bar_rotation_matrix = np.asarray(
                bullet.getMatrixFromQuaternion(self.bar_reference_orientation), dtype=float
            ).reshape(3, 3)
            ee_rotation_matrix = np.asarray(
                bullet.getMatrixFromQuaternion(ee_orientation), dtype=float
            ).reshape(3, 3)
            bar_axis_3d = bar_rotation_matrix @ self.bar_axis_local
            bar_axis_3d -= self.feature_table_normal * float(
                bar_axis_3d @ self.feature_table_normal
            )
            bar_axis_3d /= np.linalg.norm(bar_axis_3d)
            bar_lateral_3d = np.cross(self.feature_table_normal, bar_axis_3d)
            bar_lateral_3d /= np.linalg.norm(bar_lateral_3d)
            tool_axis = ee_rotation_matrix @ self.feature_tool_axis_local
            tool_x = ee_rotation_matrix[:, 0]
            tool_x_horizontal = tool_x - self.feature_table_normal * float(
                tool_x @ self.feature_table_normal
            )
            tool_x_horizontal /= np.linalg.norm(tool_x_horizontal)
            planning_positions = np.asarray(obstacle_body_positions, dtype=float)[
                self.planning_obstacle_indices
            ]
            planning_geometry = {
                "type": self.planning_obstacle_type,
                "radius": self.planning_obstacle_radius,
            }
            if self.planning_obstacle_type == "circle":
                planning_geometry["center"] = planning_positions[0]
            else:
                planning_geometry["endpoints"] = planning_positions
            obstacle_clearance = float(
                evaluate_obstacle_clearance(
                    np.asarray(ee_position, dtype=float)[None, :],
                    planning_geometry,
                    self.feature_table_normal,
                )[0]
            )
            table_dist = float(
                (np.asarray(ee_position) - self.feature_table_surface_point)
                @ self.feature_table_normal
            )
            relative_bar = np.asarray(ee_position) - np.asarray(
                self.bar_reference_position
            )
            raw_bar_axial = float(relative_bar @ bar_axis_3d)
            bar_lateral_offset = float(relative_bar @ bar_lateral_3d)
            bar_lateral_offset -= float(
                bar_lateral_centerline_offset(
                    raw_bar_axial, self.bar_lateral_centerline
                )
            )
            down_component = -float(tool_axis @ self.feature_table_normal)
            forward_component = float(tool_axis @ bar_axis_3d)
            tool_pitch = math.atan2(down_component, forward_component)
            tool_roll = math.asin(
                float(np.clip(tool_axis @ bar_lateral_3d, -1.0, 1.0))
            )
            tool_yaw = math.atan2(
                float(tool_x_horizontal @ bar_lateral_3d),
                float(tool_x_horizontal @ bar_axis_3d),
            )
            bar_axial_offset = float(
                raw_bar_axial
                - float(
                    self.feature_true_constraints.get(
                        "bar_axial_offset_reference", 0.0
                    )
                )
            )
            feature_values = {
                "obstacle_clearance": obstacle_clearance,
                "table_dist": table_dist,
                "bar_lateral_offset": bar_lateral_offset,
                "tool_pitch": tool_pitch,
                "tool_roll": tool_roll,
                "tool_yaw": tool_yaw,
                "bar_axial_offset": bar_axial_offset,
            }
            self._feature_trace.append(
                [
                    float(self._sim_time - self._feature_started),
                    *[
                        feature_values[str(spec["name"])]
                        for spec in self.feature_schema
                    ],
                ]
            )
            self._last_feature_sample = self._sim_time
            if terminal_feature_sample:
                self._feature_terminal_captured = True

        if self._sim_time - self._last_status >= 0.1:
            bar_rotation = bullet.getMatrixFromQuaternion(bar_body_orientation)
            bar_axis_norm = math.hypot(bar_rotation[0], bar_rotation[3])
            bar_axis = (
                [
                    float(bar_rotation[0]) / bar_axis_norm,
                    float(bar_rotation[3]) / bar_axis_norm,
                ]
                if bar_axis_norm > 1e-12
                else [1.0, 0.0]
            )
            bar_lateral = [-bar_axis[1], bar_axis[0]]
            middle_u = 0.5 * sum(self.bar_outline_u)
            middle_v = 0.5 * sum(self.bar_outline_v)
            bar_pivot = [
                float(bar_body_position[index])
                - middle_u * bar_axis[index]
                - middle_v * bar_lateral[index]
                for index in range(2)
            ]
            planning_geometry = {
                "type": self.planning_obstacle_type,
                "radius": self.planning_obstacle_radius,
                "live": all(
                    self.scene_obstacles_live[index]
                    for index in self.planning_obstacle_indices
                ),
            }
            if self.planning_obstacle_type == "circle":
                center = obstacle_body_positions[self.planning_obstacle_indices[0]]
                planning_geometry["center"] = [
                    float(center[0]), float(center[1])
                ]
            else:
                planning_geometry["endpoints"] = [
                    [
                        float(obstacle_body_positions[index][0]),
                        float(obstacle_body_positions[index][1]),
                    ]
                    for index in self.planning_obstacle_indices
                ]
            status = {
                "mode": "pybullet",
                "task_id": self._task_id,
                "constraint_source": self._constraint_source,
                "controller": self._controller_state,
                "task_sequence": self._task_sequence,
                "scene_snapshot_source": self.scene_snapshot_source,
                "message": self._status_message,
                "path_index": self._path_index,
                "path_length": len(self._task_path),
                "trajectory_metrics": (
                    self._prepared_plan["metrics"] if self._prepared_plan else None
                ),
                "recording": self._recording,
                "video_capable": self.render_video,
                "render_video": self._task_video_requested,
                "run_directory": self._run_directory,
                "video_path": self._video_path,
                "sim_time": self._sim_time,
                "trace": list(self._visualization_trace),
                "feature_series": {
                    "source": self.feature_definition_source,
                    "schema": self.feature_schema,
                    "true_constraints": self.feature_true_constraints,
                    "constraint_specs": self.feature_constraint_specs,
                    "samples": list(self._feature_trace),
                },
                "current_ee": {
                    "x": float(ee_position[0]),
                    "y": float(ee_position[1]),
                    "z": float(ee_position[2]),
                    "qx": float(ee_orientation[0]),
                    "qy": float(ee_orientation[1]),
                    "qz": float(ee_orientation[2]),
                    "qw": float(ee_orientation[3]),
                },
                "scene_geometry": {
                    "bar": {
                        "pivot": bar_pivot,
                        "axis": bar_axis,
                        "outline_u": list(self.bar_outline_u),
                        "outline_v": list(self.bar_outline_v),
                        "lateral_centerline": dict(self.bar_lateral_centerline),
                        "live": self.scene_bar_live,
                    },
                    "obstacles": [
                        {
                            "center": [float(center[0]), float(center[1])],
                            "radius": float(radius),
                            "live": bool(live),
                        }
                        for center, radius, live in zip(
                            obstacle_body_positions,
                            self.obstacle_radii,
                            self.scene_obstacles_live,
                        )
                    ],
                    "obstacle": planning_geometry,
                },
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
        self._run_directory = os.path.join(output_root, self._task_id, run_name)
        os.makedirs(self._run_directory, exist_ok=False)
        self._log_file = open(os.path.join(self._run_directory, "trajectory.csv"), "w", newline="")
        self._log_writer = csv.DictWriter(self._log_file, fieldnames=self.LOG_FIELDS)
        self._log_writer.writeheader()
        self._log_file.flush()
        self._recording = True
        self._video_finished = False
        self._video_complete_time = None
        self._video_next_frame_time = None
        self._video_path = None
        self._write_metadata("running")
        rospy.loginfo("Recording simulation data in %s", self._run_directory)

    def _camera_rgb(self):
        width, height = self.video_size
        view = bullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[self.table_center_xy[0], self.table_center_xy[1], 0.42],
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
        if (
            not self.render_video
            or not self._task_video_requested
            or not self._recording
            or self._video_finished
        ):
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
        self._video_next_frame_time = self._sim_time
        first_frame = self._camera_rgb()
        for _ in range(max(1, int(round(self.video_initial_hold * self.video_fps)))):
            self._video_process.stdin.write(first_frame)
        rospy.loginfo("Rendering goal-reaching video to %s", self._video_path)
        return True

    def _capture_video(self):
        if (
            not self.render_video
            or not self._task_video_requested
            or not self._recording
            or self._video_finished
        ):
            return
        if self._controller_state == "executing" and not self._video_active:
            if not self._begin_video():
                return
        if not self._video_active:
            return
        if self._sim_time + 0.5 / self.physics_hz < self._video_next_frame_time:
            return
        try:
            self._video_process.stdin.write(self._camera_rgb())
            self._video_next_frame_time += 1.0 / self.video_fps
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
        self._video_next_frame_time = None
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

    def _set_task_video(self, request):
        if self._controller_state in ("planning", "moving_to_start", "executing"):
            return SetBoolResponse(False, "Cannot change video rendering during a task")
        with self._path_lock:
            self._task_video_requested = bool(request.data)
            self._pending_path = None
            self._awaiting_path_metadata = None
            self._prepared_plan = None
            self._active_segment = None
            self._task_path = []
            self._path_index = -1
            self._target_pose = None
            self._visualization_trace.clear()
            self._controller_state = "idle"
            self._status_message = "Ready for the next task"
        self._run_directory = None
        self._video_path = None
        return SetBoolResponse(
            True,
            "Task video enabled" if self._task_video_requested else "Task video disabled",
        )

    def _abort_task(self, _request):
        with self._path_lock:
            if self._controller_state not in ("planning", "moving_to_start", "executing"):
                return TriggerResponse(True, "No simulation task is active")
            self._abort_requested = True
        return TriggerResponse(True, "Simulation task abort requested")

    def _write_metadata(self, status):
        metadata = {
            "status": status,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "simulator": "pybullet",
            "task_id": self._task_id,
            "constraint_source": self._constraint_source,
            "pybullet_api_version": bullet.getAPIVersion(),
            "frame_id": self.frame_id,
            "physics_hz": self.physics_hz,
            "publish_hz": self.publish_hz,
            "motion_speed_scale": self.motion_speed_scale,
            "velocity_scale": self.velocity_scale,
            "acceleration_limit_rad_s2": self.acceleration_limit,
            "approach_speed_mps": self.approach_speed,
            "task_speed_mps": self.task_speed,
            "render_video": self._task_video_requested,
            "video_fps": self.video_fps,
            "video_size": self.video_size,
            "video_slowdown": 1.0,
            "video_path": self._video_path,
            "table_top_z": self.table_top_z,
            "table_size": self.table_size,
            "table_center_xy": self.table_center_xy,
            "scene_config": self.scene_config_path,
            "scene_source": self.scene_source,
            "scene_snapshot_source": self.scene_snapshot_source,
            "scene_bar_live": self.scene_bar_live,
            "scene_obstacles_live": self.scene_obstacles_live,
            "scene_locked_bar_pose": self.scene_locked_bar_pose,
            "scene_locked_obstacle_poses": self.scene_locked_obstacle_poses,
            "bar_size": self.bar_size,
            "bar_reference_xy": self.bar_reference_xy,
            "bar_center_xy": self.bar_center_xy,
            "bar_yaw": self.bar_yaw,
            "obstacle_centers": self.obstacle_centers,
            "obstacle_radii": self.obstacle_radii,
            "auto_goal_distance": self.auto_goal_distance,
            "planner": "stage_constraint_planner/task_planner",
            "trajectory_compiler": "stage_cartesian_trajectory/CartesianTrajectoryCompiler",
            "orientation_control": "position_plus_full_orientation",
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
        obstacle_contacts = [
            contact
            for obstacle_id in self.obstacle_ids
            for contact in bullet.getContactPoints(
                bodyA=self.robot_id,
                bodyB=obstacle_id,
                physicsClientId=self.client,
            )
        ]
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
            if (
                self._sim_time - self._last_publish + 0.5 / self.physics_hz
                >= 1.0 / self.publish_hz
            ):
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
