#!/usr/bin/env python3
"""Local-only supervisor for the stage-constraint robot workstation.

The HTTP API exposes a fixed allow-list of lifecycle actions. It deliberately
does not provide a shell endpoint and never starts the robot driver on login.
"""

from __future__ import annotations

import base64
import json
import csv
import http.client
import io
import math
import os
import secrets
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent / "web"
CONTAINER = "stage_cons_iiwa14"
SIM_CONTAINER = "stage_cons_iiwa14_sim"
HOST = "127.0.0.1"
PORT = 8080
DEMO_PORT = 8081
TASK_TRACE_POINTS = 4000
SCENE_CONFIG = (
    PROJECT_ROOT
    / "ros_ws"
    / "src"
    / "stage_iiwa_sim"
    / "config"
    / "demo_scene.json"
)
TASK_PROFILES = {
    "BarInspect": {
        "display_name": "Bar Inspect",
        "n_stages": 4,
        "stage_names": ["Approach", "Vertical scan", "Oblique scan", "Depart"],
    },
    "BarClean": {
        "display_name": "Bar Clean",
        "n_stages": 5,
        "stage_names": [
            "Approach",
            "Longitudinal clean",
            "Free reposition",
            "Right-to-left discharge",
            "Depart",
        ],
    },
}


class Supervisor:
    def __init__(self) -> None:
        self.token = secrets.token_urlsafe(32)
        self._lock = threading.RLock()
        self._job: Optional[threading.Thread] = None
        self._job_name = ""
        self._job_ok: Optional[bool] = None
        self._job_message = "Ready"
        self._logs: deque[str] = deque(maxlen=500)
        self._children: Dict[str, subprocess.Popen[str]] = {}
        self._task_abort = threading.Event()
        self._task_state: Dict[str, object] = {
            "task_id": "BarInspect",
            "mode": "simulator",
            "phase": "idle",
            "record": True,
            "message": "No task has been started",
            "run_directory": None,
            "video_available": False,
        }
        self._fixed_scene_geometry = self._load_fixed_scene_geometry()
        self._fixed_feature_series = self._load_fixed_feature_series()
        self._task_trace: deque[List[float]] = deque(maxlen=TASK_TRACE_POINTS)
        self._task_current_ee: Optional[Dict[str, float]] = None
        self._task_scene_geometry = self._fallback_scene_geometry()
        self._task_scene_source = "fallback"
        self._task_feature_series = self._empty_feature_series()
        self._task_planned_trace: List[List[float]] = []
        self._task_planned_feature_series = self._empty_feature_series()
        self._task_stage_boundary_indices: List[int] = []
        self._task_stage_boundary_times: List[float] = []
        self._task_stage_transition_end_times: List[float] = []

    def log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            for line in str(text).rstrip().splitlines() or [""]:
                self._logs.append(f"[{stamp}] {line}")

    @staticmethod
    def _env() -> Dict[str, str]:
        env = os.environ.copy()
        env["USER_UID"] = str(os.getuid())
        return env

    def _run(
        self,
        command: List[str],
        *,
        check: bool = True,
        timeout: Optional[float] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> subprocess.CompletedProcess[str]:
        self.log("$ " + " ".join(command))
        env = self._env()
        if env_overrides:
            env.update(env_overrides)
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        if result.stdout:
            self.log(result.stdout)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(command)}"
            )
        return result

    def _container_running(self) -> bool:
        return self._named_container_running(CONTAINER)

    @staticmethod
    def _named_container_running(container: str) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _ros_nodes(self) -> List[str]:
        if not self._container_running():
            return []
        result = subprocess.run(
            [
                "docker",
                "exec",
                CONTAINER,
                "bash",
                "-lc",
                "timeout 2 rosnode list 2>/dev/null",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.stdout.splitlines() if result.returncode == 0 else []

    def _wait_for_ros_master(self, timeout: float = 15.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._container_running():
                result = subprocess.run(
                    ["docker", "exec", CONTAINER, "rosnode", "list"],
                    text=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if result.returncode == 0:
                    return
            time.sleep(0.5)
        raise RuntimeError("Container started, but ROS master was not ready after 15 s")

    def _driver_process_containers(self) -> List[str]:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        conflicts: List[str] = []
        for container in result.stdout.splitlines():
            top = subprocess.run(
                ["docker", "top", container],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if (
                "roslaunch iiwa_driver" in top.stdout
                or "iiwa14_bringup.launch" in top.stdout
                or "/iiwa_driver/iiwa_driver" in top.stdout
            ):
                conflicts.append(container)
        return conflicts

    def _driver_binary_running(self) -> bool:
        if not self._container_running():
            return False
        top = subprocess.run(
            ["docker", "top", CONTAINER],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return (
            top.returncode == 0
            and "/iiwa_driver/iiwa_driver" in top.stdout
        )

    def _robot_iface_state(self) -> Dict[str, object]:
        env_file = PROJECT_ROOT / ".env"
        iface = ""
        host_ip = ""
        if env_file.exists():
            for raw in env_file.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key == "IIWA14_IFACE":
                    iface = value.strip().strip("'\"")
                elif key == "FRI_HOST_IP":
                    host_ip = value.strip().strip("'\"")
        configured = False
        if iface and host_ip:
            result = subprocess.run(
                ["ip", "-4", "addr", "show", "dev", iface],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            configured = result.returncode == 0 and f"{host_ip}/24" in result.stdout
        return {"interface": iface, "host_ip": host_ip, "configured": configured}

    @staticmethod
    def _transform_point(
        rotation: List[List[float]], translation: List[float], point: List[float]
    ) -> List[float]:
        return [
            sum(rotation[row][column] * point[column] for column in range(3))
            + translation[row]
            for row in range(3)
        ]

    @staticmethod
    def _rotate_vector(rotation: List[List[float]], vector: List[float]) -> List[float]:
        return [
            sum(rotation[row][column] * vector[column] for column in range(3))
            for row in range(3)
        ]

    @classmethod
    def _load_fixed_scene_geometry(cls) -> Dict[str, object]:
        config = json.loads(SCENE_CONFIG.read_text(encoding="utf-8"))
        transform = config["optitrack_to_robot"]
        rotation = [[float(value) for value in row] for row in transform["rotation"]]
        translation = [float(value) for value in transform["translation"]]
        bar = config["bar"]
        obstacle = config["obstacle"]
        bar_pose = [float(value) for value in bar["locked_pose_optitrack"]]
        obstacle_pose = [float(value) for value in obstacle["locked_pose_optitrack"]]
        if len(bar_pose) != 7 or len(obstacle_pose) != 7:
            raise ValueError("Demo scene object poses must contain seven values")

        x, y, z, w = bar_pose[3:]
        quaternion_norm = math.sqrt(x * x + y * y + z * z + w * w)
        if quaternion_norm <= 1e-12:
            raise ValueError("Demo scene bar quaternion cannot be zero")
        x, y, z, w = (
            value / quaternion_norm for value in (x, y, z, w)
        )
        tracker_rotation = [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
        tracker_axis = cls._rotate_vector(
            tracker_rotation, [float(value) for value in bar["axis_local"]]
        )
        robot_axis = cls._rotate_vector(rotation, tracker_axis)
        axis_norm = math.hypot(robot_axis[0], robot_axis[1])
        if axis_norm <= 1e-12:
            raise ValueError("Demo scene bar axis is vertical in the robot frame")
        bar_reference = cls._transform_point(rotation, translation, bar_pose[:3])
        obstacle_reference = cls._transform_point(
            rotation, translation, obstacle_pose[:3]
        )
        return {
            "bar": {
                "pivot": bar_reference[:2],
                "axis": [robot_axis[0] / axis_norm, robot_axis[1] / axis_norm],
                "outline_u": [float(value) for value in bar["outline_u"]],
                "outline_v": [float(value) for value in bar["outline_v"]],
                "live": False,
            },
            "obstacle": {
                "center": obstacle_reference[:2],
                "radius": float(obstacle["radius"]),
                "live": False,
            },
        }

    def _fallback_scene_geometry(self) -> Dict[str, object]:
        return json.loads(json.dumps(self._fixed_scene_geometry))

    @staticmethod
    def _load_fixed_feature_series() -> Dict[str, object]:
        config = json.loads(SCENE_CONFIG.read_text(encoding="utf-8"))
        definition = config["feature_definition"]
        return {
            "source": str(definition["source"]),
            "schema": [dict(value) for value in definition["schema"]],
            "true_constraints": dict(definition["true_constraints"]),
            "constraint_specs": [
                dict(value) for value in definition["constraint_specs"]
            ],
        }

    def _empty_feature_series(self) -> Dict[str, object]:
        series = json.loads(json.dumps(self._fixed_feature_series))
        series["samples"] = []
        return series

    def _demo_visualization(self) -> Optional[Dict[str, object]]:
        try:
            with urlopen(f"http://127.0.0.1:{DEMO_PORT}/api/state", timeout=0.15) as response:
                demo_state = json.load(response)
        except (OSError, URLError, ValueError):
            return None

        dependencies = demo_state.get("dependencies", {})
        demo_scene = demo_state.get("scene_geometry", {})
        scene = self._fallback_scene_geometry()
        live_objects = []
        for name, dependency in (
            ("bar", "optitrack_bar"),
            ("obstacle", "optitrack_obstacle"),
        ):
            geometry = demo_scene.get(name) if isinstance(demo_scene, dict) else None
            if dependencies.get(dependency) and isinstance(geometry, dict):
                scene[name] = {**geometry, "live": True}
                live_objects.append(name)
        current_ee = demo_state.get("current_ee") if dependencies.get("ee_tf") else None
        return {
            "current_ee": current_ee,
            "scene_geometry": scene,
            "source": "optitrack" if live_objects else "fallback",
        }

    def task_visualization(self) -> Dict[str, object]:
        with self._lock:
            mode = str(self._task_state.get("mode", "simulator"))
            phase = str(self._task_state.get("phase", "idle"))
            use_simulation = (
                mode == "simulator"
                and phase != "idle"
                and self._task_scene_source == "simulation"
            )
            trace = [list(point) for point in self._task_trace]
            current_ee = dict(self._task_current_ee) if self._task_current_ee else None
            scene = json.loads(json.dumps(self._task_scene_geometry))
            source = self._task_scene_source
            feature_series = json.loads(json.dumps(self._task_feature_series))
            planned_trace = [list(point) for point in self._task_planned_trace]
            planned_feature_series = json.loads(
                json.dumps(self._task_planned_feature_series)
            )
            stage_boundary_indices = list(self._task_stage_boundary_indices)
            stage_boundary_times = list(self._task_stage_boundary_times)
            stage_transition_end_times = list(self._task_stage_transition_end_times)

        if not use_simulation:
            demo_visualization = self._demo_visualization()
            if demo_visualization is not None:
                current_ee = demo_visualization["current_ee"]
                scene = demo_visualization["scene_geometry"]
                source = demo_visualization["source"]
            elif source != "simulation":
                current_ee = None
                scene = self._fallback_scene_geometry()
                source = "fallback"

        return {
            "ok": True,
            "task_id": str(self._task_state.get("task_id", "BarInspect")),
            "mode": mode,
            "phase": phase,
            "current_ee": current_ee,
            "trace": trace,
            "planned_trace": planned_trace,
            "scene_geometry": scene,
            "source": source,
            "feature_series": feature_series,
            "planned_feature_series": planned_feature_series,
            "stage_boundary_indices": stage_boundary_indices,
            "stage_boundary_times": stage_boundary_times,
            "stage_transition_end_times": stage_transition_end_times,
        }

    def state(self) -> Dict[str, object]:
        nodes = self._ros_nodes()
        with self._lock:
            busy = self._job is not None and self._job.is_alive()
            return {
                "token": self.token,
                "project_root": str(PROJECT_ROOT),
                "available_tasks": [
                    {"task_id": task_id, **profile}
                    for task_id, profile in TASK_PROFILES.items()
                ],
                "container_running": self._container_running(),
                "driver_running": "/iiwa14/iiwa_driver" in nodes,
                "demo_running": "/stage_demo_gui" in nodes,
                "simulator_running": self._named_container_running(SIM_CONTAINER),
                "task": {
                    **self._task_state,
                    "video_available": self._task_state.get("phase") == "complete"
                    and self.task_video_path() is not None,
                },
                "job": {
                    "busy": busy,
                    "name": self._job_name,
                    "ok": self._job_ok,
                    "message": self._job_message,
                },
                "robot_network": self._robot_iface_state(),
                "demo_url": "/demo",
                "logs": list(self._logs)[-120:],
            }

    @staticmethod
    def _validated_pose(name: str, value: object) -> Dict[str, float]:
        if not isinstance(value, dict):
            raise ValueError(f"{name} pose must be an object")
        keys = ("x", "y", "z", "qx", "qy", "qz", "qw")
        try:
            pose = {key: float(value[key]) for key in keys}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{name} pose must contain seven numeric values") from error
        if not all(math.isfinite(number) for number in pose.values()):
            raise ValueError(f"{name} pose contains a non-finite value")
        norm = math.sqrt(sum(pose[key] ** 2 for key in ("qx", "qy", "qz", "qw")))
        if norm < 1e-8:
            raise ValueError(f"{name} quaternion cannot be zero")
        for key in ("qx", "qy", "qz", "qw"):
            pose[key] /= norm
        return pose

    @staticmethod
    def _pose_message(pose: Dict[str, float]) -> str:
        return json.dumps(
            {
                "header": {"frame_id": "iiwa14_link_0"},
                "pose": {
                    "position": {key: pose[key] for key in ("x", "y", "z")},
                    "orientation": {
                        axis: pose["q" + axis] for axis in ("x", "y", "z", "w")
                    },
                },
            },
            separators=(",", ":"),
        )

    def _sim_compose(self, *arguments: str) -> List[str]:
        return [
            "docker",
            "compose",
            "-p",
            "stage_cons_iiwa14_sim",
            "-f",
            "compose.yaml",
            "-f",
            "compose.sim.yaml",
            *arguments,
        ]

    def _wait_for_simulator(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._named_container_running(SIM_CONTAINER):
                result = subprocess.run(
                    ["docker", "exec", SIM_CONTAINER, "/entrypoint.sh", "rosnode", "list"],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if result.returncode == 0 and "/iiwa14/pybullet_sim" in result.stdout.splitlines():
                    return
            time.sleep(0.5)
        raise RuntimeError("PyBullet simulator did not become ready within 30 s")

    def _ensure_simulator(
        self, require_video: bool
    ) -> Tuple[bool, Dict[str, object]]:
        environment = {
            "SIM_AUTO_PLAN": "false",
            "SIM_RECORD": "false",
            "SIM_RENDER_VIDEO": "true",
        }
        recreate = False
        if self._named_container_running(SIM_CONTAINER):
            try:
                self._wait_for_simulator(timeout=3.0)
                status = self._read_sim_status()
            except (RuntimeError, subprocess.TimeoutExpired):
                self.log("Simulator is unhealthy; recreating its container")
                recreate = True
            else:
                if "task_sequence" not in status or "task_id" not in status:
                    raise RuntimeError(
                        "Simulator image is outdated; rebuild the workstation image once"
                    )
                if require_video and status.get("render_video") is not True:
                    self.log("Restarting simulator once to enable task video rendering")
                    recreate = True
                else:
                    self.log("Reusing the running PyBullet simulator and robot state")
                    return True, status

        arguments = ["up", "-d"]
        if recreate:
            arguments.append("--force-recreate")
        self._run(self._sim_compose(*arguments), env_overrides=environment)
        self._wait_for_simulator()
        status = self._read_sim_status()
        if "task_sequence" not in status or "task_id" not in status:
            raise RuntimeError(
                "Simulator image is outdated; rebuild the workstation image once"
            )
        if require_video and status.get("render_video") is not True:
            raise RuntimeError("Simulator started without video-rendering capability")
        return False, status

    def _sim_ros(self, *arguments: str, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
        return self._run(
            ["docker", "exec", SIM_CONTAINER, "/entrypoint.sh", *arguments],
            timeout=timeout,
        )

    def _submit_sim_task(
        self,
        task_id: str,
        start: Dict[str, float],
        goal: Dict[str, float],
        record: bool,
    ) -> None:
        payload = json.dumps(
            {"task_id": task_id, "start": start, "goal": goal, "record": record},
            separators=(",", ":"),
        )
        result = self._sim_ros(
            "rosrun",
            "stage_iiwa_sim",
            "submit_sim_task.py",
            payload,
            timeout=15.0,
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        try:
            response = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise RuntimeError("Simulation task submitter returned invalid output") from error
        if response.get("success") is not True:
            raise RuntimeError(str(response.get("message", "Planner rejected the task")))

    def _read_sim_status(self) -> Dict[str, object]:
        result = subprocess.run(
            [
                "docker",
                "exec",
                SIM_CONTAINER,
                "/entrypoint.sh",
                "rostopic",
                "echo",
                "-n",
                "1",
                "-p",
                "/iiwa14/sim/status",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Could not read simulator status")
        data_line = self._string_message_from_csv(result.stdout)
        try:
            return json.loads(data_line)
        except json.JSONDecodeError as error:
            raise RuntimeError("Simulator returned an invalid status message") from error

    def _read_plan_visualization(self, container: str) -> Dict[str, object]:
        result = subprocess.run(
            [
                "docker", "exec", container, "/entrypoint.sh",
                "rostopic", "echo", "-n", "1", "-p",
                "/stage_cons/plan_visualization",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Could not read planner visualization")
        try:
            encoded = self._string_message_from_csv(result.stdout)
            return json.loads(base64.b64decode(encoded, validate=True).decode("utf-8"))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError("Planner returned invalid visualization data") from error

    def _wait_for_sim_task(
        self, previous_sequence: int, timeout: float = 15.0
    ) -> Tuple[int, Dict[str, object]]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = self._read_sim_status()
            try:
                task_sequence = int(status["task_sequence"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("Simulator status has no valid task sequence") from error
            if task_sequence > previous_sequence:
                self._update_sim_visualization(status)
                return task_sequence, status
            time.sleep(0.05)
        raise RuntimeError("Simulator did not accept the new planner path within 15 s")

    @staticmethod
    def _string_message_from_csv(output: str) -> str:
        rows = list(csv.reader(io.StringIO(output)))
        for row in rows:
            if row and not row[0].startswith("%") and len(row) >= 2:
                return ",".join(row[1:]).strip()
        return ""

    def _real_ros(
        self, *arguments: str, timeout: float = 10.0
    ) -> subprocess.CompletedProcess[str]:
        return self._run(
            ["docker", "exec", CONTAINER, "/entrypoint.sh", *arguments],
            timeout=timeout,
        )

    def _read_real_status(self) -> Dict[str, object]:
        result = subprocess.run(
            [
                "docker", "exec", CONTAINER, "/entrypoint.sh",
                "rostopic", "echo", "-n", "1", "-p",
                "/iiwa14/real_executor/status",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("Could not read real-executor status")
        try:
            return json.loads(self._string_message_from_csv(result.stdout))
        except json.JSONDecodeError as error:
            raise RuntimeError("Real executor returned an invalid status message") from error

    def _wait_for_real_station(
        self, child: subprocess.Popen[str], timeout: float = 20.0
    ) -> None:
        required = {
            "/iiwa14/iiwa_driver",
            "/iiwa14/real_executor",
            "/stage_constraint_planner",
        }
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError(
                    "Real station exited during startup; inspect the supervisor log"
                )
            if required.issubset(set(self._ros_nodes())):
                return
            time.sleep(0.5)
        raise RuntimeError("Real station did not become ready within 20 s")

    def _start_real_station(self) -> None:
        nodes = set(self._ros_nodes())
        if "/iiwa14/real_executor" in nodes:
            return
        if "/iiwa14/iiwa_driver" in nodes:
            raise RuntimeError(
                "A non-real-task iiwa_driver is already running. Stop it before switching "
                "to PositionTrajectoryController."
            )
        conflicts = self._driver_process_containers()
        if conflicts:
            raise RuntimeError(
                "Another iiwa driver is already running in: " + ", ".join(conflicts)
            )
        child = self._spawn(
            "real",
            [
                "docker", "exec", CONTAINER, "/entrypoint.sh",
                "roslaunch", "stage_real_executor", "real_station.launch",
            ],
        )
        self._wait_for_real_station(child)

    def _execute_real_task(
        self,
        task_id: str,
        start: Dict[str, float],
        goal: Dict[str, float],
        record: bool,
    ) -> None:
        if not self._container_running():
            raise RuntimeError("Start the workstation container first")
        if not self._robot_iface_state()["configured"]:
            raise RuntimeError("Robot network interface is not configured")
        if self._named_container_running(SIM_CONTAINER):
            raise RuntimeError("Stop the PyBullet simulator before real-robot execution")
        self._task_abort.clear()
        self._reset_task_visualization()
        self._set_task_state(
            task_id=task_id, mode="real", phase="starting", record=record,
            start=start, goal=goal,
            message="Starting the iiwa14 position-control station",
            run_directory=None, video_available=False,
        )
        self._start_real_station()
        self._real_ros(
            "rosservice", "call", "/iiwa14/real_executor/set_recording",
            "data: {}".format("true" if record else "false"),
        )
        self._real_ros(
            "rostopic", "pub", "-1", "/stage_cons/planner/task",
            "std_msgs/String", "data: '{}'".format(task_id),
        )
        for topic, pose in (
            ("/stage_cons/planner/start", start),
            ("/stage_cons/planner/goal", goal),
        ):
            self._real_ros(
                "rostopic", "pub", "-1", topic, "geometry_msgs/PoseStamped",
                self._pose_message(pose),
            )
        response = self._real_ros(
            "rosservice", "call", "/stage_constraint_planner/plan", timeout=30.0
        )
        if "success: True" not in response.stdout:
            raise RuntimeError("Planner rejected the real task")
        self._update_plan_visualization(self._read_plan_visualization(CONTAINER))
        self._set_task_state(
            phase="preparing", message="Solving continuous IK and validating the joint trajectory"
        )
        response = self._real_ros(
            "rosservice", "call", "/iiwa14/real_executor/prepare", timeout=60.0
        )
        if "success: True" not in response.stdout:
            raise RuntimeError("Real executor rejected the trajectory: " + response.stdout.strip())
        if self._task_abort.is_set():
            self._set_task_state(phase="aborted", message="Real task aborted before execution")
            return
        response = self._real_ros(
            "rosservice", "call", "/iiwa14/real_executor/execute", timeout=10.0
        )
        if "success: True" not in response.stdout:
            raise RuntimeError("Real executor refused to start: " + response.stdout.strip())

        deadline = time.monotonic() + 600.0
        last_phase = ""
        while time.monotonic() < deadline:
            if self._task_abort.is_set():
                self._set_task_state(phase="aborted", message="Real task aborted by user")
                return
            status = self._read_real_status()
            status_task_id = str(status.get("task_id", ""))
            if status_task_id != task_id:
                raise RuntimeError(
                    "Real executor reports task {}, but GUI selected {}".format(
                        status_task_id, task_id
                    )
                )
            phase = str(status.get("phase", "unknown"))
            if phase != last_phase:
                self.log("Real task phase: " + phase)
                last_phase = phase
            message = str(status.get("message", phase))
            self._set_task_state(
                phase=phase,
                message=message,
                run_directory=status.get("run_directory"),
                video_available=False,
            )
            if phase == "complete":
                return
            if phase in ("failed", "rejected", "protective_stop"):
                raise RuntimeError(message)
            if phase == "aborted":
                return
            time.sleep(0.25)
        raise RuntimeError("Real task exceeded the 600 s timeout")

    def _set_task_state(self, **values: object) -> None:
        with self._lock:
            self._task_state.update(values)

    def _reset_task_visualization(self) -> None:
        with self._lock:
            self._task_trace.clear()
            self._task_current_ee = None
            self._task_scene_geometry = self._fallback_scene_geometry()
            self._task_scene_source = "fallback"
            self._task_feature_series = self._empty_feature_series()
            self._task_planned_trace = []
            self._task_planned_feature_series = self._empty_feature_series()
            self._task_stage_boundary_indices = []
            self._task_stage_boundary_times = []
            self._task_stage_transition_end_times = []

    def _update_plan_visualization(self, payload: Dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        trace = payload.get("trace")
        feature_names = payload.get("feature_names")
        feature_schema = payload.get("feature_schema")
        constraint_specs = payload.get("constraint_specs")
        feature_samples = payload.get("feature_samples")
        boundary_indices = payload.get("stage_boundaries")
        boundary_times = payload.get("stage_boundary_times")
        transition_end_times = payload.get("stage_transition_end_times")
        with self._lock:
            selected_task = str(self._task_state.get("task_id", "BarInspect"))
        if task_id != selected_task:
            raise RuntimeError(
                "Planner returned task {}, but GUI selected {}".format(
                    task_id, selected_task
                )
            )
        if (
            not isinstance(trace, list)
            or not isinstance(feature_names, list)
            or not isinstance(feature_schema, list)
            or not isinstance(constraint_specs, list)
        ):
            raise RuntimeError("Planner visualization has an invalid trace or feature schema")
        schema_names = [str(value.get("name", "")) for value in feature_schema]
        if [str(value) for value in feature_names] != schema_names or not all(schema_names):
            raise RuntimeError("Planner visualization feature names do not match its schema")
        if (
            not isinstance(feature_samples, list)
            or not isinstance(boundary_indices, list)
            or not isinstance(boundary_times, list)
            or not isinstance(transition_end_times, list)
        ):
            raise RuntimeError("Planner visualization has invalid feature samples")

        valid_trace = []
        for point in trace:
            if (
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(value, (int, float)) for value in point)
                and all(math.isfinite(float(value)) for value in point)
            ):
                valid_trace.append([float(value) for value in point])
        width = len(schema_names) + 1
        valid_samples = []
        for sample in feature_samples:
            if (
                isinstance(sample, list)
                and len(sample) == width
                and all(isinstance(value, (int, float)) for value in sample)
                and all(math.isfinite(float(value)) for value in sample)
            ):
                valid_samples.append([float(value) for value in sample])
        valid_boundary_indices = [
            int(value)
            for value in boundary_indices
            if isinstance(value, int) and not isinstance(value, bool)
        ]
        valid_boundaries = [
            float(value)
            for value in boundary_times
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        valid_transition_ends = [
            float(value)
            for value in transition_end_times
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        ]
        expected_transitions = int(TASK_PROFILES[task_id]["n_stages"]) - 1
        expected_stages = expected_transitions + 1
        if (
            len(valid_trace) < 2
            or len(valid_samples) < 2
            or len(valid_boundary_indices) != expected_stages
            or len(valid_boundaries) != expected_transitions
            or len(valid_transition_ends) != expected_transitions
            or any(
                current <= previous
                for previous, current in zip(
                    valid_boundary_indices, valid_boundary_indices[1:]
                )
            )
            or valid_boundary_indices[-1] != len(valid_trace) - 1
            or any(end < start for start, end in zip(valid_boundaries, valid_transition_ends))
        ):
            raise RuntimeError("Planner visualization is incomplete")

        series = {
            "source": "stage_constraint_planner/{}".format(task_id),
            "schema": json.loads(json.dumps(feature_schema)),
            "true_constraints": {},
            "constraint_specs": json.loads(json.dumps(constraint_specs)),
            "samples": valid_samples,
        }
        with self._lock:
            self._task_planned_trace = valid_trace[-TASK_TRACE_POINTS:]
            self._task_planned_feature_series = series
            self._task_stage_boundary_indices = valid_boundary_indices
            self._task_stage_boundary_times = valid_boundaries
            self._task_stage_transition_end_times = valid_transition_ends

    def _update_sim_visualization(self, status: Dict[str, object]) -> None:
        status_task_id = str(status.get("task_id", ""))
        with self._lock:
            selected_task = str(self._task_state.get("task_id", "BarInspect"))
        if status_task_id != selected_task:
            raise RuntimeError(
                "Simulator reports task {}, but GUI selected {}".format(
                    status_task_id, selected_task
                )
            )
        current_ee = status.get("current_ee")
        scene = status.get("scene_geometry")
        trace = status.get("trace")
        feature_series = status.get("feature_series")
        phase = str(status.get("controller", "unknown"))
        with self._lock:
            if isinstance(trace, list):
                valid_trace = []
                for value in trace[-TASK_TRACE_POINTS:]:
                    if (
                        isinstance(value, list)
                        and len(value) == 2
                        and all(isinstance(number, (int, float)) for number in value)
                        and all(math.isfinite(float(number)) for number in value)
                    ):
                        valid_trace.append([float(value[0]), float(value[1])])
                self._task_trace.clear()
                self._task_trace.extend(valid_trace)
            if isinstance(current_ee, dict):
                try:
                    point = {
                        axis: float(current_ee[axis]) for axis in ("x", "y", "z")
                    }
                except (KeyError, TypeError, ValueError):
                    point = None
                if point is not None and all(math.isfinite(value) for value in point.values()):
                    try:
                        orientation = {
                            key: float(current_ee[key])
                            for key in ("qx", "qy", "qz", "qw")
                        }
                    except (KeyError, TypeError, ValueError):
                        orientation = {}
                    if orientation and all(
                        math.isfinite(value) for value in orientation.values()
                    ):
                        point.update(orientation)
                    self._task_current_ee = point
                    if not isinstance(trace, list) and phase in ("moving_to_start", "executing"):
                        xy = [point["x"], point["y"]]
                        if not self._task_trace or math.dist(self._task_trace[-1], xy) >= 1e-4:
                            self._task_trace.append(xy)
            if isinstance(scene, dict) and scene.get("bar") and scene.get("obstacle"):
                self._task_scene_geometry = json.loads(json.dumps(scene))
                self._task_scene_source = "simulation"
            if isinstance(feature_series, dict):
                schema = feature_series.get("schema")
                samples = feature_series.get("samples")
                if isinstance(schema, list) and isinstance(samples, list):
                    width = len(schema) + 1
                    valid_samples = []
                    for sample in samples[-2400:]:
                        if (
                            isinstance(sample, list)
                            and len(sample) == width
                            and all(isinstance(value, (int, float)) for value in sample)
                            and all(math.isfinite(float(value)) for value in sample)
                        ):
                            valid_samples.append([float(value) for value in sample])
                    self._task_feature_series = {
                        "source": str(feature_series.get("source", "unknown")),
                        "schema": json.loads(json.dumps(schema)),
                        "true_constraints": json.loads(
                            json.dumps(feature_series.get("true_constraints", {}))
                        ),
                        "constraint_specs": json.loads(
                            json.dumps(feature_series.get("constraint_specs", []))
                        ),
                        "samples": valid_samples,
                    }

    def execute_task(self, payload: Dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        if task_id not in TASK_PROFILES:
            raise ValueError("Unknown task_id {}".format(task_id))
        with self._lock:
            selected_task = str(self._task_state.get("task_id", "BarInspect"))
        if task_id != selected_task:
            raise ValueError("Select {} in the GUI before execution".format(task_id))
        mode = str(payload.get("mode", "simulator"))
        if mode not in ("simulator", "real"):
            raise ValueError("mode must be simulator or real")
        start = self._validated_pose("start", payload.get("start"))
        goal = self._validated_pose("goal", payload.get("goal"))
        record = payload.get("record", True) is True

        if mode == "real":
            if payload.get("confirmed") is not True:
                raise RuntimeError("Real-robot execution requires explicit confirmation")
            self._start_job(
                "Execute real task",
                lambda: self._execute_real_task(task_id, start, goal, record),
            )
            return

        def task() -> None:
            if self._driver_process_containers():
                raise RuntimeError("Stop every real-robot iiwa driver before starting simulation")
            self._task_abort.clear()
            self._reset_task_visualization()
            self._set_task_state(
                task_id=task_id,
                mode=mode,
                phase="starting",
                record=record,
                start=start,
                goal=goal,
                message="Starting or reusing the persistent simulator",
                run_directory=None,
                video_available=False,
            )
            reused, initial_status = self._ensure_simulator(require_video=record)
            controller = str(initial_status.get("controller", "unknown"))
            if controller in ("planning", "moving_to_start", "executing"):
                raise RuntimeError("The persistent simulator is still executing another task")
            try:
                previous_sequence = int(initial_status["task_sequence"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("Simulator status has no valid task sequence") from error
            self._submit_sim_task(task_id, start, goal, record)
            self._update_plan_visualization(
                self._read_plan_visualization(SIM_CONTAINER)
            )

            task_sequence, status = self._wait_for_sim_task(previous_sequence)
            self.log(
                "Simulation task {} accepted ({})".format(
                    task_sequence, "reused simulator" if reused else "started simulator"
                )
            )

            deadline = time.monotonic() + 300.0
            last_phase = ""
            while time.monotonic() < deadline:
                if self._task_abort.is_set():
                    self._set_task_state(phase="aborted", message="Task aborted by user")
                    return
                try:
                    status_sequence = int(status["task_sequence"])
                except (KeyError, TypeError, ValueError) as error:
                    raise RuntimeError("Simulator status has no valid task sequence") from error
                if status_sequence != task_sequence:
                    raise RuntimeError("Simulator task sequence changed unexpectedly")
                phase = str(status.get("controller", "unknown"))
                self._update_sim_visualization(status)
                if phase != last_phase:
                    self.log(f"Simulation task phase: {phase}")
                    last_phase = phase
                run_directory = status.get("run_directory")
                self._set_task_state(
                    phase=phase,
                    message={
                        "moving_to_start": "Moving robot to the task start",
                        "executing": "Executing planner path",
                        "complete": "Task completed and data finalized",
                    }.get(phase, phase),
                    run_directory=run_directory,
                    video_available=bool(run_directory and record and phase == "complete"),
                )
                if phase == "complete":
                    return
                if phase == "failed":
                    raise RuntimeError(str(status.get("message", "Simulator failed to reach task start")))
                time.sleep(0.1)
                status = self._read_sim_status()
            raise RuntimeError("Simulation task exceeded the 300 s timeout")

        self._start_job("Execute simulation task", task)

    def select_task(self, payload: Dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        if task_id not in TASK_PROFILES:
            raise ValueError("Unknown task_id {}".format(task_id))
        with self._lock:
            busy = self._job is not None and self._job.is_alive()
            phase = str(self._task_state.get("phase", "idle"))
        if busy or phase in ("starting", "preparing", "moving_to_start", "executing"):
            raise RuntimeError("Cannot switch tasks while execution is active")
        request = Request(
            "http://127.0.0.1:{}/api/task".format(DEMO_PORT),
            data=json.dumps({"task_id": task_id}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=0.5) as response:
                result = json.load(response)
            if result.get("ok") is not True:
                raise RuntimeError(
                    str(result.get("message", "Demo GUI rejected task selection"))
                )
        except HTTPError as error:
            try:
                result = json.loads(error.read().decode("utf-8"))
                message = str(result.get("message", error))
            except (OSError, ValueError):
                message = str(error)
            raise RuntimeError(message) from error
        except (OSError, URLError, ValueError) as error:
            self.log("Demo GUI task sync deferred: {}".format(error))
        self._reset_task_visualization()
        self._set_task_state(
            task_id=task_id,
            phase="idle",
            message="{} selected".format(TASK_PROFILES[task_id]["display_name"]),
            run_directory=None,
            video_available=False,
        )

    def abort_task(self) -> None:
        self._task_abort.set()
        with self._lock:
            mode = self._task_state.get("mode")
        if mode == "real" and self._container_running():
            self._run(
                [
                    "docker", "exec", CONTAINER, "/entrypoint.sh", "rosservice",
                    "call", "/iiwa14/real_executor/abort",
                ],
                check=False,
                timeout=5,
            )
            self._set_task_state(
                phase="aborted", message="Real trajectory cancelled; position controller remains active"
            )
        else:
            if not self._named_container_running(SIM_CONTAINER):
                self._set_task_state(phase="aborted", message="Task aborted; simulator was not running")
                return
            try:
                response = self._sim_ros(
                    "rosservice", "call", "/iiwa14/sim/abort", timeout=5.0
                )
                if "success: True" not in response.stdout:
                    raise RuntimeError(response.stdout.strip())
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                self.log(f"Graceful simulator abort failed: {error}")
                self._run(self._sim_compose("stop"), check=False, timeout=20)
                self._set_task_state(
                    phase="aborted",
                    message="Task aborted; unhealthy simulator was stopped",
                )
                return
            self._set_task_state(
                phase="aborted",
                message="Task aborted; simulator is holding the current position",
            )

    def task_video_path(self) -> Optional[Path]:
        with self._lock:
            run_directory = self._task_state.get("run_directory")
        if not isinstance(run_directory, str):
            return None
        try:
            relative = Path(run_directory).relative_to("/data/sim_runs")
        except ValueError:
            return None
        if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
            return None
        path = PROJECT_ROOT / "data" / "sim_runs" / relative / "goal_reaching.mp4"
        return path if path.is_file() else None

    def _start_job(self, name: str, target) -> None:
        with self._lock:
            if self._job is not None and self._job.is_alive():
                raise RuntimeError(f"Another task is running: {self._job_name}")
            self._job_name = name
            self._job_ok = None
            self._job_message = f"{name} started"

            def runner() -> None:
                try:
                    target()
                except Exception as error:  # shown in the local supervisor log
                    self.log(f"ERROR: {error}")
                    with self._lock:
                        self._job_ok = False
                        self._job_message = str(error)
                        if name in ("Execute simulation task", "Execute real task") and not self._task_abort.is_set():
                            self._task_state.update(phase="failed", message=str(error))
                else:
                    with self._lock:
                        self._job_ok = True
                        self._job_message = f"{name} completed"

            self._job = threading.Thread(target=runner, daemon=True)
            self._job.start()

    def start_workstation(self) -> None:
        def task() -> None:
            self._run(["docker", "compose", "up", "-d"])
            self._wait_for_ros_master()
            if "/stage_demo_gui" not in self._ros_nodes():
                self._signal_child("demo")
            self._start_demo_process()

        self._start_job("Start workstation", task)

    def rebuild_image(self) -> None:
        def task() -> None:
            conflicts = self._driver_process_containers()
            if conflicts:
                raise RuntimeError(
                    "Stop every iiwa driver before rebuilding the image: "
                    + ", ".join(conflicts)
                )
            self._run(["docker", "compose", "build"])

        self._start_job("Rebuild workstation image", task)

    def _read_env_value(self, key: str, default: str) -> str:
        env_file = PROJECT_ROOT / ".env"
        if not env_file.exists():
            return default
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            current_key, value = line.split("=", 1)
            if current_key == key:
                return value.strip().strip("'\"")
        return default

    def _spawn(self, name: str, command: List[str]) -> subprocess.Popen[str]:
        with self._lock:
            child = self._children.get(name)
            if child is not None and child.poll() is None:
                self.log(f"{name} is already running")
                return child
        self.log("$ " + " ".join(command))
        child = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=self._env(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        with self._lock:
            self._children[name] = child

        def collect() -> None:
            assert child.stdout is not None
            for line in child.stdout:
                self.log(f"{name}: {line.rstrip()}")
            code = child.wait()
            self.log(f"{name} exited with code {code}")

        threading.Thread(target=collect, daemon=True).start()
        return child

    def _wait_for_demo_ready(
        self, child: subprocess.Popen[str], timeout: float = 12.0
    ) -> None:
        required_nodes = {
            "/demo_recorder",
            "/iiwa14/demo_virtual_fixture",
            "/optitrack_base_transform",
            "/stage_demo_gui",
            "/vrpn_client_node",
        }
        deadline = time.monotonic() + timeout
        stable_since: Optional[float] = None
        while time.monotonic() < deadline:
            code = child.poll()
            if code is not None:
                raise RuntimeError(
                    f"Demo station exited during startup (code {code}); see logs above"
                )
            nodes_ready = required_nodes.issubset(set(self._ros_nodes()))
            port_ready = False
            if nodes_ready:
                try:
                    with socket.create_connection((HOST, DEMO_PORT), timeout=0.25):
                        port_ready = True
                except OSError:
                    pass
            if nodes_ready and port_ready:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= 1.5:
                    return
            else:
                stable_since = None
            time.sleep(0.5)
        raise RuntimeError("Demo station did not become healthy within 12 s")

    def _start_demo_process(self) -> None:
        if not self._container_running():
            raise RuntimeError("Container is not running")
        if "/stage_demo_gui" in self._ros_nodes():
            self.log("Demo station is already running")
            return
        server = self._read_env_value("OPTITRACK_SERVER", "128.178.145.104")
        base = self._read_env_value("OPTITRACK_BASE", "iiwa14")
        obj = self._read_env_value("OPTITRACK_OBJECT", "baiyu_bar")
        obstacle = self._read_env_value("OPTITRACK_OBSTACLE", "baiyu_obs_ball")
        child = self._spawn(
            "demo",
            [
                "docker",
                "exec",
                CONTAINER,
                "/entrypoint.sh",
                "roslaunch",
                "stage_demo_gui",
                "demo_station.launch",
                f"optitrack_server:={server}",
                f"optitrack_base:={base}",
                f"optitrack_object:={obj}",
                f"optitrack_obstacle:={obstacle}",
                f"gui_port:={DEMO_PORT}",
            ],
        )
        self._wait_for_demo_ready(child)

    def start_demo_if_available(self) -> None:
        if not self._container_running():
            return

        def task() -> None:
            self._wait_for_ros_master()
            self._start_demo_process()

        self._start_job("Start Demo station automatically", task)

    def start_driver(self) -> None:
        def task() -> None:
            if not self._container_running():
                raise RuntimeError("Start the workstation container first")
            if not self._robot_iface_state()["configured"]:
                raise RuntimeError("Robot network interface is not configured")
            if "/iiwa14/iiwa_driver" in self._ros_nodes():
                self.log("iiwa_driver is already running")
                return
            with self._lock:
                previous_child = self._children.get("driver")
            if previous_child is not None and previous_child.poll() is None:
                self.log(
                    "Stopping stale iiwa_driver roslaunch wrapper before retrying"
                )
                self._signal_child("driver")
            conflicts = self._driver_process_containers()
            if conflicts:
                raise RuntimeError(
                    "An iiwa driver process already exists in container(s): "
                    + ", ".join(conflicts)
                )
            # Ensure assistance cannot be left active by a previous ROS graph.
            self._run(
                [
                    "docker",
                    "exec",
                    CONTAINER,
                    "bash",
                    "-lc",
                    "rosservice call /iiwa14/demo_virtual_fixture/enable_all 'data: false' >/dev/null 2>&1 || true",
                ],
                check=False,
                timeout=3,
            )
            child = self._spawn(
                "driver",
                [
                    "docker",
                    "exec",
                    CONTAINER,
                    "/entrypoint.sh",
                    "roslaunch",
                    "iiwa_driver",
                    "iiwa14_bringup.launch",
                ],
            )
            try:
                self._wait_for_driver_ready(child)
            except Exception:
                self._signal_child("driver")
                raise

        self._start_job("Start iiwa_driver", task)

    def _wait_for_driver_ready(
        self, child: subprocess.Popen[str], timeout: float = 15.0
    ) -> None:
        started_at = time.monotonic()
        deadline = started_at + timeout
        stable_since: Optional[float] = None
        while time.monotonic() < deadline:
            code = child.poll()
            if code is not None:
                raise RuntimeError(
                    f"iiwa_driver launch exited during startup (code {code}); see logs above"
                )
            if "/iiwa14/iiwa_driver" in self._ros_nodes():
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= 2.0:
                    return
            else:
                stable_since = None
                if (
                    time.monotonic() - started_at >= 2.0
                    and not self._driver_binary_running()
                ):
                    raise RuntimeError(
                        "iiwa_driver executable exited inside roslaunch; "
                        "the stale launch wrapper was cleaned up and the start can be retried"
                    )
            time.sleep(0.5)
        raise RuntimeError("iiwa_driver ROS node did not become healthy within 15 s")

    def _signal_child(self, name: str) -> None:
        with self._lock:
            child = self._children.get(name)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)

    def _stop_driver_process(self) -> None:
        self._signal_child("real")
        self._signal_child("driver")
        if self._container_running():
            self._run(
                [
                    "docker",
                    "exec",
                    CONTAINER,
                    "bash",
                    "-lc",
                    "pkill -INT -f '[r]oslaunch stage_real_executor real_station.launch' || true; "
                    "pkill -INT -f '[r]oslaunch iiwa_driver .*iiwa14_bringup.launch' || true",
                ],
                check=False,
                timeout=5,
            )
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                driver_node_running = "/iiwa14/iiwa_driver" in self._ros_nodes()
                launch_running = CONTAINER in self._driver_process_containers()
                if not driver_node_running and not launch_running:
                    return
                time.sleep(0.5)
            raise RuntimeError(
                "iiwa_driver or its roslaunch wrapper did not stop; "
                "use the SmartPAD/E-stop if motion persists"
            )

    def stop_driver(self) -> None:
        def task() -> None:
            if self._container_running():
                self._run(
                    [
                        "docker",
                        "exec",
                        CONTAINER,
                        "bash",
                        "-lc",
                        "rosservice call /iiwa14/demo_virtual_fixture/enable_all 'data: false' >/dev/null 2>&1 || true",
                    ],
                    check=False,
                    timeout=3,
                )
            self._stop_driver_process()

        self._start_job("Stop iiwa_driver", task)

    def stop_all(self) -> None:
        def task() -> None:
            self._task_abort.set()
            self._run(self._sim_compose("stop"), check=False, timeout=20)
            if self._container_running():
                self._run(
                    [
                        "docker",
                        "exec",
                        CONTAINER,
                        "bash",
                        "-lc",
                        "rosservice call /iiwa14/demo_virtual_fixture/enable_all 'data: false' >/dev/null 2>&1 || true",
                    ],
                    check=False,
                    timeout=3,
                )
            self._stop_driver_process()
            self._signal_child("demo")
            self._run(["docker", "compose", "stop"], check=False)

        self._start_job("Stop workstation", task)


SUPERVISOR = Supervisor()


class Handler(BaseHTTPRequestHandler):
    server_version = "StageSupervisor/1.0"

    def _json(self, status: HTTPStatus, payload: Dict[str, object]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type: str) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _proxy_demo(self) -> None:
        parsed = urlsplit(self.path)
        upstream_path = parsed.path[len("/demo"):]
        if not upstream_path:
            upstream_path = "/"
        if parsed.query:
            upstream_path += "?" + parsed.query
        body = None
        if self.command in ("POST", "PUT", "PATCH"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._json(
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "message": "Invalid request length"},
                )
                return
            if length < 0 or length > 65536:
                self._json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"ok": False, "message": "Demo request is too large"},
                )
                return
            body = self.rfile.read(length)
        headers = {
            name: self.headers[name]
            for name in ("Accept", "Content-Type")
            if name in self.headers
        }
        connection = http.client.HTTPConnection(HOST, DEMO_PORT, timeout=3.0)
        try:
            connection.request(self.command, upstream_path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self.send_response(response.status, response.reason)
            content_type = response.getheader("Content-Type")
            if content_type:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except (OSError, http.client.HTTPException) as error:
            self._json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"ok": False, "message": "Demo station is unavailable: {}".format(error)},
            )
        finally:
            connection.close()

    def do_GET(self) -> None:  # noqa: N802
        request_path = self.path.partition("?")[0]
        if request_path == "/demo":
            self.send_response(HTTPStatus.TEMPORARY_REDIRECT)
            self.send_header("Location", "/demo/")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif request_path.startswith("/demo/"):
            self._proxy_demo()
        elif request_path == "/" or request_path == "/index.html":
            self._file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
        elif request_path == "/api/state":
            self._json(HTTPStatus.OK, SUPERVISOR.state())
        elif request_path == "/api/task/visualization":
            self._json(HTTPStatus.OK, SUPERVISOR.task_visualization())
        elif request_path == "/api/task/video":
            video = SUPERVISOR.task_video_path()
            if video is None:
                self.send_error(HTTPStatus.NOT_FOUND, "No completed task video")
            else:
                self._file(video, "video/mp4")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        request_path = self.path.partition("?")[0]
        if request_path.startswith("/demo/"):
            self._proxy_demo()
            return
        if self.headers.get("X-Stage-Token") != SUPERVISOR.token:
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "message": "Invalid token"})
            return
        length = min(int(self.headers.get("Content-Length", "0")), 4096)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/start-workstation":
                SUPERVISOR.start_workstation()
            elif self.path == "/api/rebuild-image":
                SUPERVISOR.rebuild_image()
            elif self.path == "/api/start-driver":
                if payload.get("confirmed") is not True:
                    raise RuntimeError("启动 iiwa_driver 前必须在界面中二次确认")
                SUPERVISOR.start_driver()
            elif self.path == "/api/stop-driver":
                SUPERVISOR.stop_driver()
            elif self.path == "/api/stop-all":
                SUPERVISOR.stop_all()
            elif self.path == "/api/task/execute":
                SUPERVISOR.execute_task(payload)
            elif self.path == "/api/task/select":
                SUPERVISOR.select_task(payload)
            elif self.path == "/api/task/abort":
                SUPERVISOR.abort_task()
            else:
                self._json(HTTPStatus.NOT_FOUND, {"ok": False, "message": "Not found"})
                return
            self._json(HTTPStatus.ACCEPTED, {"ok": True, "message": "Accepted"})
        except (ValueError, RuntimeError) as error:
            self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "message": str(error)})

    def log_message(self, format_string: str, *args: object) -> None:
        if any(
            isinstance(argument, str)
            and (
                argument.startswith("GET /api/state ")
                or argument.startswith("GET /api/task/visualization ")
            )
            for argument in args
        ):
            return
        SUPERVISOR.log(format_string % args)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    SUPERVISOR.log(f"Host supervisor available at http://{HOST}:{PORT}")
    SUPERVISOR.start_demo_if_available()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
