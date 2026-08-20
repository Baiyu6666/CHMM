#!/usr/bin/env python3
"""Local-only supervisor for the stage-constraint robot workstation.

The HTTP API exposes a fixed allow-list of lifecycle actions. It deliberately
does not provide a shell endpoint and never starts the robot driver on login.
"""

from __future__ import annotations

import json
import csv
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
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = Path(__file__).resolve().parent / "web"
CONTAINER = "stage_cons_iiwa14"
SIM_CONTAINER = "stage_cons_iiwa14_sim"
HOST = "127.0.0.1"
PORT = 8080
DEMO_PORT = 8081


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
            "mode": "simulator",
            "phase": "idle",
            "record": True,
            "message": "No task has been started",
            "run_directory": None,
            "video_available": False,
        }

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

    def state(self) -> Dict[str, object]:
        nodes = self._ros_nodes()
        with self._lock:
            busy = self._job is not None and self._job.is_alive()
            return {
                "token": self.token,
                "project_root": str(PROJECT_ROOT),
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
                "demo_url": f"http://127.0.0.1:{DEMO_PORT}",
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

    def _sim_ros(self, *arguments: str, timeout: float = 10.0) -> subprocess.CompletedProcess[str]:
        return self._run(
            ["docker", "exec", SIM_CONTAINER, "/entrypoint.sh", *arguments],
            timeout=timeout,
        )

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
        required = {"/iiwa14/iiwa_driver", "/iiwa14/real_executor", "/straight_line_planner"}
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
        self, start: Dict[str, float], goal: Dict[str, float], record: bool
    ) -> None:
        if not self._container_running():
            raise RuntimeError("Start the workstation container first")
        if not self._robot_iface_state()["configured"]:
            raise RuntimeError("Robot network interface is not configured")
        if self._named_container_running(SIM_CONTAINER):
            raise RuntimeError("Stop the PyBullet simulator before real-robot execution")
        self._task_abort.clear()
        self._set_task_state(
            mode="real", phase="starting", record=record, start=start, goal=goal,
            message="Starting the iiwa14 position-control station",
            run_directory=None, video_available=False,
        )
        self._start_real_station()
        self._real_ros(
            "rosservice", "call", "/iiwa14/real_executor/set_recording",
            "data: {}".format("true" if record else "false"),
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
            "rosservice", "call", "/straight_line_planner/plan", timeout=10.0
        )
        if "success: True" not in response.stdout:
            raise RuntimeError("Planner rejected the real task")
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

    def execute_task(self, payload: Dict[str, object]) -> None:
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
                "Execute real task", lambda: self._execute_real_task(start, goal, record)
            )
            return

        def task() -> None:
            if self._driver_process_containers():
                raise RuntimeError("Stop every real-robot iiwa driver before starting simulation")
            self._task_abort.clear()
            self._set_task_state(
                mode=mode,
                phase="starting",
                record=record,
                start=start,
                goal=goal,
                message="Building and starting the isolated simulator",
                run_directory=None,
                video_available=False,
            )
            environment = {
                "SIM_AUTO_PLAN": "false",
                "SIM_RECORD": "false",
                "SIM_RENDER_VIDEO": "true" if record else "false",
            }
            self._run(self._sim_compose("build"), env_overrides=environment)
            self._run(
                self._sim_compose("up", "-d", "--force-recreate"),
                env_overrides=environment,
            )
            self._wait_for_simulator()
            self._sim_ros(
                "rosservice",
                "call",
                "/iiwa14/sim/set_task_recording",
                "data: {}".format("true" if record else "false"),
            )
            for topic, pose in (
                ("/stage_cons/planner/start", start),
                ("/stage_cons/planner/goal", goal),
            ):
                self._sim_ros(
                    "rostopic",
                    "pub",
                    "-1",
                    topic,
                    "geometry_msgs/PoseStamped",
                    self._pose_message(pose),
                )
            response = self._sim_ros(
                "rosservice", "call", "/straight_line_planner/plan", timeout=10.0
            )
            if "success: True" not in response.stdout:
                raise RuntimeError("Planner rejected the task")

            deadline = time.monotonic() + 300.0
            last_phase = ""
            while time.monotonic() < deadline:
                if self._task_abort.is_set():
                    self._set_task_state(phase="aborted", message="Task aborted by user")
                    return
                status = self._read_sim_status()
                phase = str(status.get("controller", "unknown"))
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
                time.sleep(0.5)
            raise RuntimeError("Simulation task exceeded the 300 s timeout")

        self._start_job("Execute simulation task", task)

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
            self._run(self._sim_compose("stop"), check=False, timeout=20)
            self._set_task_state(phase="aborted", message="Task aborted; simulator stopped")

    def task_video_path(self) -> Optional[Path]:
        with self._lock:
            run_directory = self._task_state.get("run_directory")
        if not isinstance(run_directory, str):
            return None
        run_name = Path(run_directory).name
        if not run_name or run_name in (".", ".."):
            return None
        path = PROJECT_ROOT / "data" / "sim_runs" / run_name / "goal_reaching.mp4"
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

    def build_and_start(self) -> None:
        def task() -> None:
            conflicts = self._driver_process_containers()
            if conflicts:
                raise RuntimeError(
                    "Stop every iiwa driver before rebuilding containers: "
                    + ", ".join(conflicts)
                )
            self._run(["docker", "compose", "build"])
            self._signal_child("demo")
            self._run(["docker", "compose", "up", "-d"])
            self._wait_for_ros_master()
            self._start_demo_process()

        self._start_job("Build and start workstation", task)

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

    def start_demo(self) -> None:
        def task() -> None:
            self._wait_for_ros_master()
            self._start_demo_process()

        self._start_job("Start Demo station", task)

    def configure_network(self) -> None:
        def task() -> None:
            if not self._container_running():
                raise RuntimeError("Start the workstation container first")
            self._run(
                ["pkexec", str(PROJECT_ROOT / "scripts" / "connect_robot_network.sh")],
                timeout=60,
            )

        self._start_job("Configure robot network", task)

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

    def do_GET(self) -> None:  # noqa: N802
        request_path = self.path.partition("?")[0]
        if request_path == "/" or request_path == "/index.html":
            self._file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
        elif request_path == "/api/state":
            self._json(HTTPStatus.OK, SUPERVISOR.state())
        elif request_path == "/api/task/video":
            video = SUPERVISOR.task_video_path()
            if video is None:
                self.send_error(HTTPStatus.NOT_FOUND, "No completed task video")
            else:
                self._file(video, "video/mp4")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.headers.get("X-Stage-Token") != SUPERVISOR.token:
            self._json(HTTPStatus.FORBIDDEN, {"ok": False, "message": "Invalid token"})
            return
        length = min(int(self.headers.get("Content-Length", "0")), 4096)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/build-start":
                SUPERVISOR.build_and_start()
            elif self.path == "/api/start-demo":
                SUPERVISOR.start_demo()
            elif self.path == "/api/configure-network":
                SUPERVISOR.configure_network()
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
            and argument.startswith("GET /api/state ")
            for argument in args
        ):
            return
        SUPERVISOR.log(format_string % args)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    SUPERVISOR.log(f"Host supervisor available at http://{HOST}:{PORT}")
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
