#!/usr/bin/env python3
"""ROS-aware HTTP console for safe demonstration collection."""

import json
import math
import os
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float64MultiArray, Int32, String
from std_srvs.srv import SetBool, Trigger


class DemoGui:
    TASK_IDS = {"BarClean"}

    def __init__(self):
        self._lock = threading.RLock()
        self._service_lock = threading.Lock()
        self._mode_requested = False
        self._driver_demo_active = False
        self._mode_loss_cleanup_running = False
        self._fri_commanding = False
        self._fri_command_mode = 0
        self._orientation_requested = False
        self._vertical_requested = False
        self._assistance_active = False
        self._recorder_state = "unknown"
        self._recorder_session = None
        self._trace = deque(maxlen=int(rospy.get_param("~max_trace_points", 5000)))
        self._current_ee = None
        self._current_bar = None
        self._current_obstacle = None
        self._bar_pose_window = deque(maxlen=21)
        self._obstacle_pose_window = deque(maxlen=21)
        self._last_joint = 0.0
        self._last_bar = 0.0
        self._last_obstacle = 0.0
        self._last_fixture = 0.0
        self._last_recorder = 0.0
        self._last_commanding = 0.0
        self._last_command_mode = 0.0
        self._last_motion_gate = 0.0
        self._last_ee = 0.0
        self._last_trace_sample = 0.0
        self._message = "Waiting for robot data"
        self._task_id = str(rospy.get_param("~task_id", "BarClean"))
        if self._task_id not in self.TASK_IDS:
            raise ValueError("Unknown initial task id {}".format(self._task_id))
        rospy.set_param("/demo_recorder/task_id", self._task_id)

        self._root_frame = rospy.get_param("~root_frame", "iiwa14_link_0")
        self._ee_frame = rospy.get_param("~ee_frame", "iiwa14_link_ee")
        self._gui_file = rospy.get_param("~gui_file")
        self._host = rospy.get_param("~host", "127.0.0.1")
        self._port = int(rospy.get_param("~port", 8080))
        self._tracker_to_robot_rotation = rospy.get_param(
            "~optitrack_to_robot_rotation",
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )
        self._tracker_to_robot_translation = rospy.get_param(
            "~optitrack_to_robot_translation", [0.0, 0.0, 0.0]
        )
        self._bar_outline_u = [float(value) for value in rospy.get_param(
            "~bar_outline_u", [-0.15, 0.15]
        )]
        self._bar_outline_v = [float(value) for value in rospy.get_param(
            "~bar_outline_v", [-0.03, 0.03]
        )]
        self._obstacle_radius = float(rospy.get_param("~obstacle_radius", 0.025))
        # Mechanical position hold was removed from the driver after unsafe
        # hardware behavior. Do not make it recoverable through ROS params.
        self._position_hold_enabled = False

        self._orientation_service = rospy.ServiceProxy(
            "/iiwa14/demo_virtual_fixture/enable_orientation", SetBool
        )
        self._vertical_service = rospy.ServiceProxy(
            "/iiwa14/demo_virtual_fixture/enable_vertical_damping", SetBool
        )
        self._all_service = rospy.ServiceProxy(
            "/iiwa14/demo_virtual_fixture/enable_all", SetBool
        )
        self._record_start_service = rospy.ServiceProxy(
            "/demo_recorder/start", Trigger
        )
        self._record_stop_service = rospy.ServiceProxy(
            "/demo_recorder/stop", Trigger
        )
        self._demo_mode_service = rospy.ServiceProxy(
            "/iiwa14/iiwa_driver/set_demo_mode", SetBool
        )
        self._demo_heartbeat_publisher = rospy.Publisher(
            "/iiwa14/iiwa_driver/demo_mode_heartbeat", Empty, queue_size=1
        )

        rospy.Subscriber(
            "/iiwa14/joint_states", JointState, self._on_joint_state, queue_size=1
        )
        rospy.Subscriber(
            "/vrpn_client_node/baiyu_bar/pose_from_iiwa14",
            PoseStamped,
            self._on_bar_pose,
            queue_size=1,
        )
        rospy.Subscriber(
            "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14",
            PoseStamped,
            self._on_obstacle_pose,
            queue_size=1,
        )
        rospy.Subscriber(
            "/iiwa14/demo_virtual_fixture/status",
            Float64MultiArray,
            self._on_fixture_status,
            queue_size=1,
        )
        rospy.Subscriber(
            "/stage_cons/demo_recorder/status",
            String,
            self._on_recorder_status,
            queue_size=1,
        )
        rospy.Subscriber(
            "/iiwa14/commanding_status", Bool, self._on_commanding, queue_size=1
        )
        rospy.Subscriber(
            "/iiwa14/fri_command_mode", Int32, self._on_command_mode, queue_size=1
        )
        rospy.Subscriber(
            "/iiwa14/demo_mode_active", Bool, self._on_motion_gate, queue_size=1
        )

        self._tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
        self._tf_timer = rospy.Timer(rospy.Duration(0.05), self._sample_ee)
        self._heartbeat_timer = rospy.Timer(rospy.Duration(0.1), self._heartbeat)
        self._http_server = None
        rospy.on_shutdown(self.shutdown)

    @staticmethod
    def _age(stamp):
        return None if stamp <= 0.0 else max(0.0, time.monotonic() - stamp)

    @staticmethod
    def _fresh(stamp, timeout):
        age = DemoGui._age(stamp)
        return age is not None and age <= timeout

    def _on_joint_state(self, _message):
        with self._lock:
            self._last_joint = time.monotonic()

    @staticmethod
    def _median(values):
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    def _tracker_vector_to_robot(self, vector):
        return [
            sum(float(row[index]) * float(vector[index]) for index in range(3))
            for row in self._tracker_to_robot_rotation
        ]

    def _tracker_point_to_robot(self, point):
        rotated = self._tracker_vector_to_robot(point)
        return [
            rotated[index] + float(self._tracker_to_robot_translation[index])
            for index in range(3)
        ]

    def _on_bar_pose(self, message):
        pose = message.pose
        position = self._tracker_point_to_robot(
            [pose.position.x, pose.position.y, pose.position.z]
        )
        x = float(pose.orientation.x)
        y = float(pose.orientation.y)
        z = float(pose.orientation.z)
        w = float(pose.orientation.w)
        quaternion_norm = math.sqrt(x * x + y * y + z * z + w * w)
        if quaternion_norm <= 1e-12 or not math.isfinite(quaternion_norm):
            return
        x /= quaternion_norm
        y /= quaternion_norm
        z /= quaternion_norm
        w /= quaternion_norm
        tracker_axis = [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ]
        axis = self._tracker_vector_to_robot(tracker_axis)
        axis_norm = math.hypot(axis[0], axis[1])
        if axis_norm <= 1e-12 or not all(math.isfinite(value) for value in position + axis):
            return
        sample = (position[0], position[1], axis[0] / axis_norm, axis[1] / axis_norm)
        with self._lock:
            self._bar_pose_window.append(sample)
            values = list(self._bar_pose_window)
            axis_x = self._median([item[2] for item in values])
            axis_y = self._median([item[3] for item in values])
            filtered_norm = math.hypot(axis_x, axis_y)
            if filtered_norm > 1e-12:
                self._current_bar = {
                    "pivot": [
                        self._median([item[0] for item in values]),
                        self._median([item[1] for item in values]),
                    ],
                    "axis": [axis_x / filtered_norm, axis_y / filtered_norm],
                    "outline_u": list(self._bar_outline_u),
                    "outline_v": list(self._bar_outline_v),
                }
            self._last_bar = time.monotonic()

    def _on_obstacle_pose(self, message):
        pose = message.pose
        position = self._tracker_point_to_robot(
            [pose.position.x, pose.position.y, pose.position.z]
        )
        if not all(math.isfinite(value) for value in position):
            return
        with self._lock:
            self._obstacle_pose_window.append((position[0], position[1]))
            values = list(self._obstacle_pose_window)
            self._current_obstacle = {
                "center": [
                    self._median([item[0] for item in values]),
                    self._median([item[1] for item in values]),
                ],
                "radius": self._obstacle_radius,
            }
            self._last_obstacle = time.monotonic()

    def _on_fixture_status(self, message):
        if len(message.data) < 3:
            return
        with self._lock:
            self._assistance_active = bool(message.data[0] > 0.5)
            self._orientation_requested = bool(message.data[1] > 0.5)
            self._vertical_requested = bool(message.data[2] > 0.5)
            self._last_fixture = time.monotonic()

    def _on_recorder_status(self, message):
        try:
            status = json.loads(message.data)
        except (TypeError, ValueError):
            return
        with self._lock:
            self._recorder_state = status.get("state", "unknown")
            self._recorder_session = status.get("session_id")
            self._last_recorder = time.monotonic()

    def _on_commanding(self, message):
        with self._lock:
            self._fri_commanding = bool(message.data)
            self._last_commanding = time.monotonic()

    def _on_command_mode(self, message):
        with self._lock:
            self._fri_command_mode = int(message.data)
            self._last_command_mode = time.monotonic()

    def _on_motion_gate(self, message):
        lost_mode = False
        with self._lock:
            active = bool(message.data)
            lost_mode = self._driver_demo_active and not active
            self._driver_demo_active = active
            if not active:
                # Driver mode/session changes define a new command epoch. Do not
                # let the GUI heartbeat silently reactivate an old Demo request.
                self._mode_requested = False
            self._last_motion_gate = time.monotonic()
            if lost_mode and not self._mode_loss_cleanup_running:
                self._mode_loss_cleanup_running = True
                threading.Thread(
                    target=self._cleanup_after_mode_loss, daemon=True
                ).start()

    def _cleanup_after_mode_loss(self):
        try:
            with self._service_lock:
                self._call_set_bool(self._all_service, False)
                with self._lock:
                    recording = self._recorder_state == "recording"
                if recording:
                    try:
                        self._record_stop_service()
                    except (rospy.ServiceException, rospy.ROSException):
                        pass
                with self._lock:
                    self._orientation_requested = False
                    self._vertical_requested = False
                    self._message = (
                        "FRI/control mode changed; Demo stopped and must be enabled again"
                    )
        finally:
            with self._lock:
                self._mode_loss_cleanup_running = False

    def _heartbeat(self, _event):
        with self._lock:
            requested = self._mode_requested
        if requested:
            self._demo_heartbeat_publisher.publish(Empty())

    def _sample_ee(self, _event):
        try:
            transform = self._tf_buffer.lookup_transform(
                self._root_frame,
                self._ee_frame,
                rospy.Time(0),
                rospy.Duration(0.01),
            )
        except Exception:
            return
        point = {
            "x": float(transform.transform.translation.x),
            "y": float(transform.transform.translation.y),
            "z": float(transform.transform.translation.z),
            "qx": float(transform.transform.rotation.x),
            "qy": float(transform.transform.rotation.y),
            "qz": float(transform.transform.rotation.z),
            "qw": float(transform.transform.rotation.w),
            "stamp": transform.header.stamp.to_sec(),
        }
        if not all(
            math.isfinite(point[key])
            for key in ("x", "y", "z", "qx", "qy", "qz", "qw")
        ):
            return
        now = time.monotonic()
        with self._lock:
            self._current_ee = point
            self._last_ee = now
            if self._driver_demo_active and now - self._last_trace_sample >= 0.04:
                self._trace.append([point["x"], point["y"]])
                self._last_trace_sample = now

    def _dependencies(self):
        with self._lock:
            return {
                "joint_state": self._fresh(self._last_joint, 0.5),
                "ee_tf": self._fresh(self._last_ee, 0.5),
                "optitrack_bar": self._fresh(self._last_bar, 0.5),
                "optitrack_obstacle": self._fresh(self._last_obstacle, 0.5),
                "fixture": self._fresh(self._last_fixture, 0.5),
                # The recorder status topic is latched and intentionally only
                # changes on state transitions, so its age is not a heartbeat.
                "recorder": self._last_recorder > 0.0,
                "fri_commanding": self._fresh(self._last_commanding, 0.5)
                and self._fri_commanding,
                "fri_torque_mode": self._fresh(self._last_command_mode, 0.5)
                and self._fri_command_mode == 3,
                "motion_gate": self._fresh(self._last_motion_gate, 0.5),
            }

    def state(self):
        dependencies = self._dependencies()
        with self._lock:
            return {
                "ok": True,
                "task_id": self._task_id,
                "available_tasks": sorted(self.TASK_IDS),
                "mode_active": self._driver_demo_active,
                "mode_requested": self._mode_requested,
                "ready": all(dependencies.values()),
                "dependencies": dependencies,
                "scene_geometry": {
                    "bar": self._current_bar,
                    "obstacle": self._current_obstacle,
                },
                "orientation_requested": self._orientation_requested,
                "vertical_requested": self._vertical_requested,
                "assistance_active": self._assistance_active,
                "fri_commanding": self._fri_commanding,
                "fri_command_mode": self._fri_command_mode,
                "position_hold_enabled": self._position_hold_enabled,
                "recorder_state": self._recorder_state,
                "recorder_session": self._recorder_session,
                "recording": self._recorder_state == "recording",
                "current_ee": self._current_ee,
                "trace": list(self._trace),
                "root_frame": self._root_frame,
                "ee_frame": self._ee_frame,
                "message": self._message,
            }

    def _call_set_bool(self, proxy, enabled):
        try:
            response = proxy(bool(enabled))
        except (rospy.ServiceException, rospy.ROSException) as error:
            return False, str(error)
        return bool(response.success), response.message

    def set_mode(self, enabled):
        with self._service_lock:
            if enabled:
                missing = [name for name, ready in self._dependencies().items() if not ready]
                if missing:
                    return False, "Not ready: " + ", ".join(missing)
                ok, message = self._call_set_bool(self._all_service, False)
                if not ok:
                    return False, "Could not disable assistance: " + message
                ok, message = self._call_set_bool(self._demo_mode_service, True)
                if not ok:
                    return False, "Could not release motion gate: " + message
                with self._lock:
                    self._trace.clear()
                    self._last_trace_sample = 0.0
                    self._mode_requested = True
                    self._driver_demo_active = True
                    self._message = "Demo motion active; assistance remains off"
                return True, self._message

            errors = []
            ok, message = self._call_set_bool(self._all_service, False)
            if not ok:
                errors.append("assistance: " + message)
            ok, message = self._call_set_bool(self._demo_mode_service, False)
            if not ok:
                errors.append("motion gate: " + message)
            if self.state()["recording"]:
                try:
                    response = self._record_stop_service()
                    if not response.success:
                        errors.append("recording: " + response.message)
                except (rospy.ServiceException, rospy.ROSException) as error:
                    errors.append("recording: " + str(error))
            with self._lock:
                self._mode_requested = False
                self._driver_demo_active = False
                self._message = (
                    "Hold mode active at the current joint pose"
                    if self._position_hold_enabled
                    else "Demo mode inactive; experimental position hold is disabled"
                )
            if errors:
                return False, "; ".join(errors)
            return True, self._message

    def set_assistance(self, channel, enabled):
        with self._service_lock:
            with self._lock:
                if not self._driver_demo_active:
                    return False, "Activate demo mode first"
            proxy = {
                "orientation": self._orientation_service,
                "vertical": self._vertical_service,
            }.get(channel)
            if proxy is None:
                return False, "Unknown assistance channel"
            ok, message = self._call_set_bool(proxy, enabled)
            with self._lock:
                self._message = message
            return ok, message

    def set_recording(self, enabled, payload):
        with self._service_lock:
            with self._lock:
                if enabled and not self._driver_demo_active:
                    return False, "Activate demo mode first"
            try:
                if enabled:
                    label = str(payload.get("label", "demo")).strip() or "demo"
                    notes = str(payload.get("notes", "")).strip()
                    rospy.set_param("/demo_recorder/task_id", self._task_id)
                    rospy.set_param("/demo_recorder/label", label)
                    rospy.set_param("/demo_recorder/operator_notes", notes)
                    response = self._record_start_service()
                else:
                    response = self._record_stop_service()
            except (rospy.ServiceException, rospy.ROSException) as error:
                return False, str(error)
            with self._lock:
                self._message = response.message
            return bool(response.success), response.message

    def set_task(self, task_id):
        task_id = str(task_id).strip()
        if task_id not in self.TASK_IDS:
            return False, "Unknown task id {}".format(task_id)
        with self._service_lock:
            with self._lock:
                if self._recorder_state == "recording":
                    return False, "Stop the current recording before switching tasks"
                self._task_id = task_id
                self._message = "{} selected for demonstration collection".format(task_id)
            rospy.set_param("/demo_recorder/task_id", task_id)
        return True, self._message

    def reset_trace(self):
        with self._lock:
            self._trace.clear()
            self._last_trace_sample = 0.0
        return True, "XY trace cleared"

    def start_http(self):
        gui = self

        class Handler(BaseHTTPRequestHandler):
            def _json(self, status, payload):
                body = json.dumps(payload, allow_nan=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def _body(self):
                length = int(self.headers.get("Content-Length", "0"))
                return json.loads(self.rfile.read(length).decode("utf-8") or "{}")

            def do_GET(self):
                path = self.path.split("?", 1)[0]
                if path == "/api/state":
                    self._json(200, gui.state())
                    return
                if path not in ("/", "/index.html"):
                    self._json(404, {"ok": False, "message": "Not found"})
                    return
                try:
                    with open(gui._gui_file, "rb") as stream:
                        body = stream.read()
                except OSError as error:
                    self._json(500, {"ok": False, "message": str(error)})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                try:
                    payload = self._body()
                    if self.path == "/api/demo_mode":
                        ok, message = gui.set_mode(bool(payload["enabled"]))
                    elif self.path == "/api/assistance/orientation":
                        ok, message = gui.set_assistance(
                            "orientation", bool(payload["enabled"])
                        )
                    elif self.path == "/api/assistance/vertical":
                        ok, message = gui.set_assistance(
                            "vertical", bool(payload["enabled"])
                        )
                    elif self.path == "/api/record":
                        ok, message = gui.set_recording(
                            bool(payload["enabled"]), payload
                        )
                    elif self.path == "/api/task":
                        ok, message = gui.set_task(payload["task_id"])
                    elif self.path == "/api/trace/reset":
                        ok, message = gui.reset_trace()
                    else:
                        self._json(404, {"ok": False, "message": "Not found"})
                        return
                    self._json(
                        200 if ok else 409,
                        {"ok": ok, "message": message},
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    self._json(400, {"ok": False, "message": str(error)})
                except Exception as error:
                    rospy.logerr("Demo GUI request failed: %s", error)
                    self._json(500, {"ok": False, "message": str(error)})

            def log_message(self, format_string, *args):
                rospy.logdebug("demo GUI: " + format_string, *args)

        if not os.path.isfile(self._gui_file):
            raise RuntimeError("GUI file is unavailable: " + self._gui_file)
        self._http_server = ThreadingHTTPServer((self._host, self._port), Handler)
        threading.Thread(target=self._http_server.serve_forever, daemon=True).start()
        rospy.loginfo("Stage demo GUI available at http://%s:%d", self._host, self._port)

    def shutdown(self):
        with self._lock:
            self._mode_requested = False
        try:
            self._call_set_bool(self._all_service, False)
        except Exception:
            pass
        try:
            self._call_set_bool(self._demo_mode_service, False)
        except Exception:
            pass
        if self._http_server is not None:
            self._http_server.shutdown()
            self._http_server.server_close()
            self._http_server = None


if __name__ == "__main__":
    rospy.init_node("stage_demo_gui")
    application = DemoGui()
    application.start_http()
    rospy.spin()
