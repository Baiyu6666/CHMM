#!/usr/bin/env python3
"""Local-only supervisor for the stage-constraint robot workstation.

The HTTP API exposes a fixed allow-list of lifecycle actions. It deliberately
does not provide a shell endpoint and never starts the robot driver on login.
"""

from __future__ import annotations

import json
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
    ) -> subprocess.CompletedProcess[str]:
        self.log("$ " + " ".join(command))
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=self._env(),
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
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER],
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
            if "roslaunch iiwa_driver" in top.stdout or "iiwa14_bringup.launch" in top.stdout:
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
        self._signal_child("driver")
        if self._container_running():
            self._run(
                [
                    "docker",
                    "exec",
                    CONTAINER,
                    "bash",
                    "-lc",
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
        if self.path == "/" or self.path == "/index.html":
            self._file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
        elif self.path == "/api/state":
            self._json(HTTPStatus.OK, SUPERVISOR.state())
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
