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
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
FINAL_VIDEO_RUN_ROOT = PROJECT_ROOT.parent / "final_video_runs"
WEB_ROOT = Path(__file__).resolve().parent / "web"
GUI_SETTINGS_PATH = PROJECT_ROOT / "data" / "gui_settings.json"
LEARNED_CONSTRAINT_ROOT = REPOSITORY_ROOT / "outputs"
MAP_METHODS = frozenset(
    {"map", "map_pooled", "map_balanced_pooled", "map_balanced_vote"}
)
CONTAINER = "stage_cons_iiwa14"
SIM_CONTAINER = "stage_cons_iiwa14_sim"
HOST = "127.0.0.1"
PORT = 8080
DEMO_PORT = 8081
TASK_TRACE_POINTS = 4000
TASK_ACTIVE_PHASES = frozenset(
    {
        "starting",
        "planning",
        "waiting_for_fri",
        "preparing",
        "prepared",
        "repreparing",
        "moving_to_start",
        "home_preparing",
        "home_repreparing",
        "home_prepared",
        "home_recovering",
        "returning_home",
        "executing",
    }
)
SCENE_CONFIG = (
    PROJECT_ROOT
    / "ros_ws"
    / "src"
    / "stage_iiwa_sim"
    / "config"
    / "demo_scene.json"
)
SCENE_CONFIG_DIR = SCENE_CONFIG.parent / "scenes"
TASK_CONFIGS = {
    "BarClean": (
        PROJECT_ROOT / "ros_ws" / "src" / "stage_constraint_planner"
        / "config" / "bar_clean_true.json"
    ),
}
FEATURE_SAMPLE_HZ = 20.0
CAMERA_PREVIEW_MARKER = "Stage Camera Preview"
CAMERA_SOURCE_LAUNCHER = Path.home() / "apps" / "android-camera-v4l2"
TASK_OUTCOME_NAMES = ("obs_avoid", "bar_clean", "table_clean")


def _bar_centerline_lateral_offset(axial: float, specification: object) -> float:
    spec = dict(specification) if isinstance(specification, dict) else {"type": "straight"}
    if str(spec.get("type", "straight")) == "straight":
        return 0.0
    if str(spec.get("type")) != "circular_arc_chord":
        raise ValueError("Unknown bar lateral centerline type")
    radius = float(spec["radius_m"])
    lower, upper = (float(value) for value in spec["axial_bounds_m"])
    sign = float(spec["bulge_sign"])
    if not lower <= axial <= upper:
        return 0.0
    midpoint = 0.5 * (lower + upper)
    half_chord = 0.5 * (upper - lower)
    return sign * (
        math.sqrt(max(radius * radius - (axial - midpoint) ** 2, 0.0))
        - math.sqrt(radius * radius - half_chord * half_chord)
    )


class RosTopicStream:
    """Keep one docker/rostopic subscriber alive and cache its newest value."""

    def __init__(self, container: str, topic: str) -> None:
        self._container = container
        self._topic = topic
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._value: Optional[str] = None
        self._received = 0.0
        self._last_error = "subscriber has not started"
        self._node_name = "stage_host_topic_{}_{}".format(
            os.getpid(), secrets.token_hex(4)
        )

    def _kill_ros_node(self) -> None:
        try:
            subprocess.run(
                [
                    "docker", "exec", self._container, "/entrypoint.sh",
                    "rosnode", "kill", "/" + self._node_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _ensure_started(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            process: Optional[subprocess.Popen[str]] = None
            try:
                process = subprocess.Popen(
                    [
                        "docker", "exec", self._container, "/entrypoint.sh",
                        "rostopic", "echo", "-p", self._topic,
                        "__name:=" + self._node_name,
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=1,
                )
                with self._condition:
                    self._process = process
                    self._last_error = "waiting for the first topic message"
                    self._condition.notify_all()
                if process.stdout is None:
                    raise RuntimeError("subscriber stdout was not created")
                for raw_line in process.stdout:
                    if self._stop.is_set():
                        break
                    line = raw_line.strip()
                    if not line or line.startswith("%"):
                        continue
                    _stamp, separator, value = line.partition(",")
                    if not separator:
                        continue
                    with self._condition:
                        self._value = value.strip()
                        self._received = time.monotonic()
                        self._last_error = ""
                        self._condition.notify_all()
            except (OSError, RuntimeError) as error:
                with self._condition:
                    self._last_error = str(error)
                    self._condition.notify_all()
            finally:
                self._kill_ros_node()
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                with self._condition:
                    if self._process is process:
                        self._process = None
                    if not self._stop.is_set() and not self._last_error:
                        self._last_error = "topic subscriber exited"
                    self._condition.notify_all()
            self._stop.wait(0.5)

    def read(self, timeout: float, max_age: float) -> str:
        self._ensure_started()
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                now = time.monotonic()
                if (
                    self._value is not None
                    and now - self._received <= max_age
                ):
                    return self._value
                remaining = deadline - now
                if remaining <= 0.0:
                    detail = self._last_error or "cached topic value is stale"
                    raise RuntimeError(
                        "Could not read fresh {} from {}: {}".format(
                            self._topic, self._container, detail
                        )
                    )
                self._condition.wait(timeout=remaining)

    def stop(self) -> None:
        self._stop.set()
        self._kill_ros_node()
        with self._condition:
            process = self._process
            self._condition.notify_all()
        if process is not None and process.poll() is None:
            process.terminate()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


class RosTfPoseStream:
    """Keep one full-precision ROS TF stream alive and cache its newest pose."""

    def __init__(self, container: str, base_frame: str, tip_frame: str) -> None:
        self._container = container
        self._base_frame = base_frame
        self._tip_frame = tip_frame
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[subprocess.Popen[str]] = None
        self._pose: Optional[Dict[str, float]] = None
        self._received = 0.0
        self._last_error = "TF subscriber has not started"
        self._node_name = "stage_host_tf_{}_{}".format(
            os.getpid(), secrets.token_hex(4)
        )

    @staticmethod
    def _parse_pose(line: str) -> Optional[Dict[str, float]]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        keys = ("x", "y", "z", "qx", "qy", "qz", "qw")
        if not isinstance(payload, dict) or set(payload) != set(keys):
            return None
        try:
            pose = {key: float(payload[key]) for key in keys}
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(value) for value in pose.values()):
            return None
        return pose

    def _kill_ros_node(self) -> None:
        try:
            subprocess.run(
                [
                    "docker", "exec", self._container, "/entrypoint.sh",
                    "rosnode", "kill", "/" + self._node_name,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _ensure_started(self) -> None:
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            process: Optional[subprocess.Popen[str]] = None
            try:
                process = subprocess.Popen(
                    [
                        "docker", "exec", self._container, "/entrypoint.sh",
                        "rosrun", "stage_real_executor", "stream_tf_pose.py",
                        self._base_frame,
                        self._tip_frame, str(int(FEATURE_SAMPLE_HZ)),
                        "__name:=" + self._node_name,
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    bufsize=1,
                )
                with self._condition:
                    self._process = process
                    self._last_error = "waiting for the first TF pose"
                    self._condition.notify_all()
                if process.stdout is None:
                    raise RuntimeError("TF subscriber stdout was not created")
                for raw_line in process.stdout:
                    if self._stop.is_set():
                        break
                    line = raw_line.strip()
                    pose = self._parse_pose(line)
                    if pose is None:
                        continue
                    with self._condition:
                        self._pose = pose
                        self._received = time.monotonic()
                        self._last_error = ""
                        self._condition.notify_all()
            except (OSError, RuntimeError) as error:
                with self._condition:
                    self._last_error = str(error)
                    self._condition.notify_all()
            finally:
                self._kill_ros_node()
                if process is not None and process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                with self._condition:
                    if self._process is process:
                        self._process = None
                    if not self._stop.is_set() and not self._last_error:
                        self._last_error = "TF subscriber exited"
                    self._condition.notify_all()
            self._stop.wait(0.5)

    def read(self, timeout: float, max_age: float) -> Dict[str, float]:
        self._ensure_started()
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                now = time.monotonic()
                if self._pose is not None and now - self._received <= max_age:
                    return dict(self._pose)
                remaining = deadline - now
                if remaining <= 0.0:
                    detail = self._last_error or "cached TF pose is stale"
                    raise RuntimeError(
                        "Could not read fresh {} -> {} from {}: {}".format(
                            self._base_frame, self._tip_frame, self._container, detail
                        )
                    )
                self._condition.wait(timeout=remaining)

    def stop(self) -> None:
        self._stop.set()
        self._kill_ros_node()
        with self._condition:
            process = self._process
            self._condition.notify_all()
        if process is not None and process.poll() is None:
            process.terminate()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


class TaskVideoRecorder:
    """Record one V4L2 stream inside an absolute robot-motion time window."""

    def __init__(
        self,
        device: str = "/dev/video10",
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
    ) -> None:
        self._device = device
        self._width = int(width)
        self._height = int(height)
        self._fps = float(fps)
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._log_stream = None
        self._final_path: Optional[Path] = None
        self._partial_path: Optional[Path] = None
        self._metadata_path: Optional[Path] = None
        self._last_error: Optional[str] = None

    @property
    def active(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def started(self) -> bool:
        return self._process is not None

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def preflight(self) -> None:
        if not Path(self._device).exists():
            raise RuntimeError(
                "Video recording requested, but {} does not exist; start "
                "android-camera-v4l2 first".format(self._device)
            )
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "v4l2", "-framerate", str(int(round(self._fps))),
            "-video_size", "{}x{}".format(self._width, self._height),
            # Read several frames so a stale v4l2loopback frame cannot make a
            # disconnected phone camera look healthy.
            "-i", self._device, "-frames:v", "3", "-f", "null", "-",
        ]
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                timeout=3.0,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError("Video recording requires ffmpeg on the host") from error
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                "{} exists but did not deliver fresh {}x{} frames within 3 s; "
                "start or reconnect the phone camera".format(
                    self._device, self._width, self._height
                )
            ) from error
        if result.returncode != 0:
            detail = result.stderr.strip() or "frame capture failed"
            raise RuntimeError(
                "Could not read {}x{} video from {}: {}".format(
                    self._width, self._height, self._device, detail
                )
            )

    def start(
        self,
        output_directory: Path,
        motion_start_unix_ns: int,
        motion_end_unix_ns: int,
    ) -> Path:
        if self.active:
            raise RuntimeError("Task video recorder is already active")
        self._last_error = None
        if motion_end_unix_ns <= motion_start_unix_ns:
            raise RuntimeError("Real executor returned an invalid video time window")
        output_directory.mkdir(parents=True, exist_ok=True)
        final_path = output_directory / "execution.mp4"
        partial_path = output_directory / "execution.partial.mp4"
        metadata_path = output_directory / "execution_video_metadata.json"
        log_path = output_directory / "execution_video_ffmpeg.log"
        for path in (final_path, partial_path, metadata_path, log_path):
            if path.exists():
                raise RuntimeError("Refusing to overwrite existing task video file " + str(path))

        # V4L2 timestamps use CLOCK_MONOTONIC, while ROS reports Unix time.
        # Capture the clock offset once and let FFmpeg discard every frame
        # outside the executor's scheduled task-motion interval.
        clock_offset_ns = time.time_ns() - time.monotonic_ns()
        motion_start_monotonic_s = (
            motion_start_unix_ns - clock_offset_ns
        ) / 1e9
        motion_end_monotonic_s = (
            motion_end_unix_ns - clock_offset_ns
        ) / 1e9
        select_filter = (
            "select=between(t\\,{:.9f}\\,{:.9f}),setpts=PTS-STARTPTS".format(
                motion_start_monotonic_s, motion_end_monotonic_s
            )
        )
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-nostdin",
            "-copyts",
            "-thread_queue_size", "1024",
            "-f", "v4l2", "-framerate", str(int(round(self._fps))),
            "-video_size", "{}x{}".format(self._width, self._height),
            "-i", self._device,
            "-an", "-vf", select_filter, "-vsync", "vfr",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "15",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(partial_path),
        ]
        metadata = {
            "schema_version": 1,
            "device": self._device,
            "width": self._width,
            "height": self._height,
            "fps": self._fps,
            "codec": "libx264-crf15-veryfast",
            "motion_start_unix_ns": int(motion_start_unix_ns),
            "motion_end_unix_ns": int(motion_end_unix_ns),
            "motion_start_monotonic_s": motion_start_monotonic_s,
            "motion_end_monotonic_s": motion_end_monotonic_s,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        log_stream = log_path.open("wb")
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=log_stream,
                start_new_session=True,
            )
        except Exception:
            log_stream.close()
            raise
        self._process = process
        self._log_stream = log_stream
        self._final_path = final_path
        self._partial_path = partial_path
        self._metadata_path = metadata_path
        time.sleep(0.15)
        if process.poll() is not None:
            self.stop(completed=False)
            detail = log_path.read_text(encoding="utf-8", errors="replace").strip()
            raise RuntimeError("FFmpeg video recorder exited during startup: " + detail)
        if time.time_ns() >= motion_start_unix_ns:
            self.stop(completed=False)
            raise RuntimeError(
                "FFmpeg did not arm before the scheduled robot-motion start"
            )
        return final_path

    def stop(self, completed: bool) -> Optional[Path]:
        process = self._process
        final_path = self._final_path
        partial_path = self._partial_path
        metadata_path = self._metadata_path
        self._process = None
        self._final_path = None
        self._partial_path = None
        self._metadata_path = None
        if process is not None and process.poll() is None:
            if completed:
                drain_delay_s = self._completion_drain_delay(metadata_path)
                if drain_delay_s > 0.0:
                    time.sleep(drain_delay_s)
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=2.0)
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None
        if partial_path is None:
            self._last_error = "video output path was not initialized"
            return None
        packet_count, recorded_duration_s = self._video_stats(partial_path)
        if packet_count <= 0:
            self._last_error = "recording contains no video packets"
            self._delete_incomplete_video(
                partial_path, metadata_path, self._last_error
            )
            return None
        coverage_error = None
        if completed and metadata_path is not None and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected_duration_s = (
                int(metadata["motion_end_unix_ns"])
                - int(metadata["motion_start_unix_ns"])
            ) / 1e9
            coverage_tolerance_s = max(0.2, 2.0 / self._fps)
            if recorded_duration_s + coverage_tolerance_s < expected_duration_s:
                completed = False
                coverage_error = (
                    "recorded {:.3f} s of the scheduled {:.3f} s motion window"
                ).format(recorded_duration_s, expected_duration_s)
                self._last_error = coverage_error
        if completed and final_path is not None:
            self._last_error = None
            partial_path.replace(final_path)
            if metadata_path is not None and metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["completed"] = True
                metadata["video_file"] = final_path.name
                metadata_path.write_text(
                    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
            return final_path
        failure_reason = coverage_error or "recording interrupted before completion"
        self._last_error = failure_reason
        self._delete_incomplete_video(partial_path, metadata_path, failure_reason)
        return None

    @staticmethod
    def _delete_incomplete_video(
        video_path: Path,
        metadata_path: Optional[Path],
        error: str,
    ) -> None:
        try:
            video_path.unlink()
        except FileNotFoundError:
            pass
        if metadata_path is None or not metadata_path.is_file():
            return
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["completed"] = False
        metadata["video_file"] = None
        metadata["video_deleted"] = True
        metadata["error"] = error
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _completion_drain_delay(metadata_path: Optional[Path]) -> float:
        if metadata_path is None or not metadata_path.is_file():
            return 0.0
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            motion_end_unix_ns = int(metadata["motion_end_unix_ns"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return 0.0
        # The completion status and the final wireless camera frame travel on
        # independent queues. Keep FFmpeg alive briefly after the scheduled
        # endpoint so frames already captured by the phone can drain. The
        # select filter still discards every timestamp beyond motion_end.
        drain_until_unix_ns = motion_end_unix_ns + 500_000_000
        return min(1.0, max(0.0, (drain_until_unix_ns - time.time_ns()) / 1e9))

    @staticmethod
    def _video_stats(path: Path) -> Tuple[int, float]:
        if not path.is_file() or path.stat().st_size == 0:
            return 0, 0.0
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-count_packets", "-show_entries", "stream=nb_read_packets",
                    "-show_entries", "format=duration", "-of", "json", str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return 0, 0.0
        if result.returncode != 0:
            return 0, 0.0
        try:
            payload = json.loads(result.stdout)
            packet_count = int(payload["streams"][0]["nb_read_packets"])
            duration_s = float(payload["format"]["duration"])
            return packet_count, duration_s
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
            return 0, 0.0


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
        self._topic_streams: Dict[Tuple[str, str], RosTopicStream] = {}
        self._tf_pose_streams: Dict[Tuple[str, str, str], RosTfPoseStream] = {}
        self._real_station_verified = False
        self._task_abort = threading.Event()
        self._task_video_recorder = TaskVideoRecorder()
        self._camera_source_process: Optional[subprocess.Popen[bytes]] = None
        self._camera_source_lock = threading.Lock()
        self._camera_preview_process: Optional[subprocess.Popen[bytes]] = None
        self._task_state: Dict[str, object] = {
            "task_id": "BarClean",
            "mode": "simulator",
            "phase": "idle",
            "data_saved": True,
            "video": False,
            "message": "No task has been started",
            "run_directory": None,
            "video_available": False,
            "review_pending": False,
            "review_status": None,
            "final_result_directory": None,
        }
        self._fixed_scene_geometry = self._load_fixed_scene_geometry()
        scene_config = json.loads(SCENE_CONFIG.read_text(encoding="utf-8"))
        scene_transform = scene_config["optitrack_to_robot"]
        self._optitrack_to_robot_rotation = [
            [float(value) for value in row]
            for row in scene_transform["rotation"]
        ]
        self._optitrack_to_robot_translation = [
            float(value) for value in scene_transform["translation"]
        ]
        self._task_profiles = self._load_task_profiles()
        self._task_feature_definitions = self._load_task_feature_definitions()
        self._constraint_sources = self._discover_constraint_sources()
        self._gui_settings = self._load_gui_settings()
        self._save_gui_settings()
        restored_task = str(self._gui_settings["task_id"])
        self._task_state["task_id"] = restored_task
        self._task_state["constraint_source_id"] = self._gui_settings[
            "constraint_source_by_task"
        ][restored_task]
        restored_source = self._resolve_constraint_source(
            restored_task, self._task_state["constraint_source_id"]
        )
        self._task_state["constraint_source_label"] = restored_source["label"]
        self._task_trace: deque[List[float]] = deque(maxlen=TASK_TRACE_POINTS)
        self._task_current_ee: Optional[Dict[str, float]] = None
        self._task_scene_geometry = self._fallback_scene_geometry()
        self._task_scene_source = "fallback"
        self._task_feature_series = self._empty_feature_series(restored_task)
        self._task_planned_trace: List[List[float]] = []
        self._task_planned_feature_series = self._empty_feature_series(restored_task)
        self._task_stage_boundary_indices: List[int] = []
        self._task_stage_boundary_times: List[float] = []
        self._task_stage_transition_end_times: List[float] = []
        self._task_execution_started: Optional[float] = None
        self._task_last_feature_sample = -math.inf

    def _topic_value(
        self,
        container: str,
        topic: str,
        *,
        timeout: float = 3.0,
        max_age: float = 1.0,
    ) -> str:
        key = (container, topic)
        with self._lock:
            stream = self._topic_streams.get(key)
            if stream is None:
                stream = RosTopicStream(container, topic)
                self._topic_streams[key] = stream
        return stream.read(timeout=timeout, max_age=max_age)

    def shutdown(self) -> None:
        self._task_video_recorder.stop(completed=False)
        with self._lock:
            streams = [
                *self._topic_streams.values(),
                *self._tf_pose_streams.values(),
            ]
            self._topic_streams.clear()
            self._tf_pose_streams.clear()
        for stream in streams:
            stream.stop()

    def _robot_ee_pose(
        self, *, timeout: float = 0.15, max_age: float = 0.25
    ) -> Optional[Dict[str, float]]:
        key = (CONTAINER, "iiwa14_link_0", "iiwa14_link_7")
        with self._lock:
            stream = self._tf_pose_streams.get(key)
            if stream is None:
                stream = RosTfPoseStream(*key)
                self._tf_pose_streams[key] = stream
        try:
            return stream.read(timeout=timeout, max_age=max_age)
        except RuntimeError:
            return None

    def cleanup_stale_topic_streams(self) -> None:
        if not self._container_running():
            return
        try:
            result = subprocess.run(
                [
                    "docker", "exec", CONTAINER, "timeout", "3s", "/entrypoint.sh",
                    "rosnode", "list",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=4.0,
                check=False,
            )
            names = [
                name
                for name in result.stdout.splitlines()
                if name.startswith(("/stage_host_topic_", "/stage_host_tf_"))
            ]
            if names:
                subprocess.run(
                    [
                        "docker", "exec", CONTAINER, "timeout", "3s", "/entrypoint.sh",
                        "rosnode", "kill", *names,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=4.0,
                    check=False,
                )
                self.log(
                    "Cleaned {} stale host topic subscriber(s)".format(len(names))
                )
        except (OSError, subprocess.TimeoutExpired):
            pass

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

    @staticmethod
    def _ensure_data_directories() -> None:
        for relative_path in (
            "data/demos",
            "data/models",
            "data/real_runs",
            "data/sim_runs",
        ):
            (PROJECT_ROOT / relative_path).mkdir(parents=True, exist_ok=True)
        FINAL_VIDEO_RUN_ROOT.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _assert_container_storage_access(
        container: str, writable_paths: Tuple[str, ...]
    ) -> None:
        uid_result = subprocess.run(
            ["docker", "exec", container, "id", "-u"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if uid_result.returncode != 0:
            raise RuntimeError(f"Could not verify the runtime UID in {container}")
        container_uid = uid_result.stdout.strip()
        host_uid = str(os.getuid())
        if container_uid != host_uid:
            raise RuntimeError(
                f"{container} uses UID {container_uid}, but the host uses UID {host_uid}; "
                "rebuild the image from the GUI before starting a task"
            )
        for writable_path in writable_paths:
            access_result = subprocess.run(
                ["docker", "exec", container, "test", "-w", writable_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if access_result.returncode != 0:
                raise RuntimeError(
                    f"{container} cannot write {writable_path}; check the host data-directory ownership"
                )

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

    def _running_iiwa_controllers(self) -> Optional[List[str]]:
        if not self._container_running():
            return []
        try:
            result = subprocess.run(
                [
                    "docker", "exec", CONTAINER,
                    "timeout", "--kill-after=0.25s", "0.75s", "/entrypoint.sh",
                    "rosservice", "call", "/iiwa14/controller_manager/list_controllers",
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=1.25,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None
        if result.returncode != 0:
            return None
        running: List[str] = []
        controller_name: Optional[str] = None
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if line.startswith("name:"):
                controller_name = line.split(":", 1)[1].strip().strip("'\"")
            elif line.startswith("state:") and controller_name is not None:
                state = line.split(":", 1)[1].strip().strip("'\"")
                if state == "running":
                    running.append(controller_name)
                controller_name = None
        return running

    def _real_station_interfaces_ready(self) -> bool:
        # The controller-manager switch blocks until the SmartPAD FRI session
        # starts. Planning is allowed before that; the later FRI gate still
        # prevents prepare/execute without a running position controller.
        commands = ("rosservice", "rostopic")
        discovered: Dict[str, set[str]] = {}
        for command in commands:
            try:
                result = subprocess.run(
                    [
                        "docker", "exec", CONTAINER, "timeout", "2s",
                        "/entrypoint.sh", command, "list",
                    ],
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return False
            if result.returncode != 0:
                return False
            discovered[command] = set(result.stdout.splitlines())
        required_services = {
            "/iiwa14/iiwa_driver/set_position_commanding",
            "/iiwa14/real_executor/prepare",
            "/iiwa14/real_executor/execute",
            "/iiwa14/real_executor/return_home",
            "/iiwa14/real_executor/abort",
        }
        required_topics = {
            "/iiwa14/PositionTrajectoryController/follow_joint_trajectory/status",
            "/iiwa14/PositionTrajectoryController/state",
            "/iiwa14/real_executor/status",
            "/iiwa14/real_executor/fri_ready_status",
        }
        return (
            required_services.issubset(discovered["rosservice"])
            and required_topics.issubset(discovered["rostopic"])
        )

    @staticmethod
    def _control_mode_from_graph(
        nodes: List[str], running_controllers: Optional[List[str]]
    ) -> Dict[str, object]:
        driver_running = "/iiwa14/iiwa_driver" in nodes
        real_executor_running = "/iiwa14/real_executor" in nodes
        if not driver_running:
            return {"mode": "idle", "label": "Idle / no robot commands", "healthy": True}
        if running_controllers is None:
            return {
                "mode": "planner_waiting" if real_executor_running else "driver_waiting",
                "label": (
                    "Planner / waiting for FRI or controller"
                    if real_executor_running
                    else "Driver / waiting for FRI or controller"
                ),
                "healthy": False,
            }
        torque_running = "SafeTorqueController" in running_controllers
        position_running = "PositionTrajectoryController" in running_controllers
        if torque_running and position_running:
            return {
                "mode": "conflict",
                "label": "Control ownership conflict",
                "healthy": False,
            }
        if torque_running:
            return {"mode": "demo", "label": "Demo / Torque", "healthy": True}
        if position_running and real_executor_running:
            return {"mode": "planner", "label": "Planner / Position", "healthy": True}
        if position_running:
            return {
                "mode": "incomplete",
                "label": "Position controller has no executor",
                "healthy": False,
            }
        if real_executor_running:
            return {
                "mode": "planner_waiting",
                "label": "Planner / waiting for FRI",
                "healthy": True,
            }
        return {
            "mode": "driver_waiting",
            "label": "Driver / waiting for FRI",
            "healthy": True,
        }

    def _quiesce_demo_control(self) -> None:
        """Remove every Demo-side command source before a driver transition."""
        if not self._container_running():
            return
        commands = (
            "rosservice call /iiwa14/demo_virtual_fixture/enable_all 'data: false'",
            "rosservice call /iiwa14/iiwa_driver/set_demo_mode 'data: false'",
            "rosservice call /demo_recorder/stop",
        )
        for command in commands:
            try:
                self._run(
                    ["docker", "exec", CONTAINER, "bash", "-lc", command],
                    check=False,
                    timeout=3,
                )
            except subprocess.TimeoutExpired:
                # A controller switch can hold the driver's ROS service queue
                # while FRI is absent.  Do not let that prevent the subsequent
                # real-executor abort and process-level driver shutdown.
                self.log("Quiesce command timed out; continuing fail-closed shutdown: " + command)

    def _release_robot_control(self, reason: str) -> None:
        """Fail closed through Idle before another controller may be started."""
        if not self._container_running():
            return
        nodes = set(self._ros_nodes())
        self.log("Releasing robot control before " + reason)
        self._quiesce_demo_control()
        if "/iiwa14/real_executor" in nodes:
            try:
                self._abort_real_and_confirm(reason)
            except RuntimeError as error:
                # A dead executor must not prevent fail-closed driver shutdown.
                # The next mode is allowed only after the old driver and launch
                # wrappers are confirmed absent below.
                self.log("Real executor did not confirm abort; forcing driver shutdown: " + str(error))
        self._stop_driver_process()

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
        bar = config["bar"]
        obstacles = list(config["obstacles"])
        planning_obstacle = dict(config["planning_obstacle"])
        bar_pose = [float(value) for value in bar["locked_pose_robot"]]
        obstacle_poses = [
            [float(value) for value in obstacle["locked_pose_robot"]]
            for obstacle in obstacles
        ]
        if len(bar_pose) != 7 or not obstacle_poses or any(
            len(pose) != 7 for pose in obstacle_poses
        ):
            raise ValueError("Demo scene object poses must contain seven values")
        obstacle_index_by_name = {
            str(value["name"]): index for index, value in enumerate(obstacles)
        }
        planning_type = str(planning_obstacle.get("type"))
        if planning_type == "circle":
            obstacle_name = str(planning_obstacle.get("obstacle", ""))
            if obstacle_name not in obstacle_index_by_name:
                raise ValueError("Circle planning_obstacle references an unknown obstacle")
            source_indices = [obstacle_index_by_name[obstacle_name]]
        elif planning_type == "capsule":
            endpoint_names = [
                str(value)
                for value in planning_obstacle.get("endpoint_obstacles", [])
            ]
            if (
                len(endpoint_names) != 2
                or len(set(endpoint_names)) != 2
                or any(name not in obstacle_index_by_name for name in endpoint_names)
            ):
                raise ValueError("Capsule planning_obstacle requires two obstacles")
            source_indices = [
                obstacle_index_by_name[name] for name in endpoint_names
            ]
        else:
            raise ValueError("Demo scene planning_obstacle must be circle or capsule")
        planning_radii = [
            float(obstacles[index]["radius"]) for index in source_indices
        ]
        if any(radius <= 0.0 for radius in planning_radii) or (
            planning_type == "capsule"
            and not math.isclose(planning_radii[0], planning_radii[1], abs_tol=1e-9)
        ):
            raise ValueError("Planning obstacle radii must be positive and consistent")

        x, y, z, w = bar_pose[3:]
        quaternion_norm = math.sqrt(x * x + y * y + z * z + w * w)
        if quaternion_norm <= 1e-12:
            raise ValueError("Demo scene bar quaternion cannot be zero")
        x, y, z, w = (
            value / quaternion_norm for value in (x, y, z, w)
        )
        bar_rotation = [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
        robot_axis = cls._rotate_vector(
            bar_rotation, [float(value) for value in bar["axis_local"]]
        )
        axis_norm = math.hypot(robot_axis[0], robot_axis[1])
        if axis_norm <= 1e-12:
            raise ValueError("Demo scene bar axis is vertical in the robot frame")
        bar_reference = bar_pose[:3]
        return {
            "bar": {
                "pivot": bar_reference[:2],
                "axis": [robot_axis[0] / axis_norm, robot_axis[1] / axis_norm],
                "outline_u": [float(value) for value in bar["outline_u"]],
                "outline_v": [float(value) for value in bar["outline_v"]],
                "lateral_centerline": dict(bar["lateral_centerline"]),
                "live": False,
            },
            "obstacles": [
                {
                    "center": pose[:2],
                    "radius": float(obstacle["radius"]),
                    "live": False,
                }
                for pose, obstacle in zip(obstacle_poses, obstacles)
            ],
            "obstacle": {
                "type": planning_type,
                **(
                    {"center": list(obstacle_poses[source_indices[0]][:2])}
                    if planning_type == "circle"
                    else {
                        "endpoints": [
                            list(obstacle_poses[index][:2])
                            for index in source_indices
                        ]
                    }
                ),
                "radius": planning_radii[0],
                "source_indices": source_indices,
                "live": False,
            },
        }

    def _fallback_scene_geometry(self) -> Dict[str, object]:
        return json.loads(json.dumps(self._fixed_scene_geometry))

    @staticmethod
    def _load_task_profiles() -> Dict[str, Dict[str, object]]:
        profiles: Dict[str, Dict[str, object]] = {}
        for task_id, path in TASK_CONFIGS.items():
            config = json.loads(path.read_text(encoding="utf-8"))
            gui = dict(config["gui"])
            stage_names = [str(value) for value in gui["stage_names"]]
            profiles[task_id] = {
                "display_name": str(config["display_name"]),
                "n_stages": len(stage_names),
                "stage_names": stage_names,
                "default_start": dict(gui["default_start"]),
                "default_goal": dict(gui["default_goal"]),
            }
        return profiles

    @staticmethod
    def _stage_one_obstacle_clearance(
        constraint_specs: object,
    ) -> Optional[float]:
        if not isinstance(constraint_specs, list):
            return None
        for raw_spec in constraint_specs:
            if not isinstance(raw_spec, dict):
                continue
            try:
                if (
                    int(raw_spec.get("stage", -1)) != 0
                    or str(raw_spec.get("feature_name")) != "obstacle_clearance"
                    or str(raw_spec.get("semantics", raw_spec.get("mode")))
                    != "lower_bound"
                ):
                    continue
                value = float(raw_spec["value"])
            except (KeyError, TypeError, ValueError):
                continue
            return value if math.isfinite(value) and value >= 0.0 else None
        return None

    def _discover_constraint_sources(self) -> Dict[str, List[Dict[str, object]]]:
        sources: Dict[str, List[Dict[str, object]]] = {}
        for task_id in self._task_profiles:
            task_definition = json.loads(
                TASK_CONFIGS[task_id].read_text(encoding="utf-8")
            )
            sources[task_id] = [
                {
                    "id": "true",
                    "label": "True constraints",
                    "task_id": task_id,
                    "container_path": "true",
                    "compatible": True,
                    "stage1_obstacle_clearance_m": (
                        self._stage_one_obstacle_clearance(
                            task_definition.get("constraint_terms")
                        )
                    ),
                }
            ]
        for method_name in sorted(MAP_METHODS):
            pattern = "{}/*/method_seed_*/learned_constraints.json".format(
                method_name
            )
            for path in sorted(LEARNED_CONSTRAINT_ROOT.glob(pattern)):
                try:
                    artifact = json.loads(path.read_text(encoding="utf-8"))
                    task_id = str(artifact.get("task_id"))
                    if (
                        artifact.get("artifact_type") != "learned_stage_constraints"
                        or task_id not in self._task_profiles
                    ):
                        continue
                    compatible, reason = self._constraint_artifact_compatibility(
                        task_id, artifact
                    )
                    relative = path.relative_to(LEARNED_CONSTRAINT_ROOT)
                    metadata_path = path.with_name("metadata.json")
                    metadata = (
                        json.loads(metadata_path.read_text(encoding="utf-8"))
                        if metadata_path.exists()
                        else {}
                    )
                    artifact_method = str(artifact.get("method_name", method_name))
                    method_seed = int(artifact.get("method_seed", 0))
                    sources[task_id].append(
                        {
                            "id": "learned:" + relative.as_posix(),
                            "label": "{} · seed {}".format(
                                artifact_method, method_seed
                            ),
                            "task_id": task_id,
                            "container_path": "/learned_constraints/"
                            + relative.as_posix(),
                            "compatible": compatible,
                            "reason": reason,
                            "created_at_utc": metadata.get("created_at_utc"),
                            "stage1_obstacle_clearance_m": (
                                self._stage_one_obstacle_clearance(
                                    artifact.get("feature_stage_modes")
                                )
                            ),
                        }
                    )
                except (KeyError, OSError, TypeError, ValueError):
                    continue
        return sources

    def _constraint_artifact_compatibility(
        self, task_id: str, artifact: Dict[str, object]
    ) -> Tuple[bool, str]:
        try:
            if int(artifact.get("schema_version", -1)) != 5:
                return False, "unsupported schema version"
            task_definition = json.loads(
                TASK_CONFIGS[task_id].read_text(encoding="utf-8")
            )
            expected_frame = dict(task_definition.get("task_frame", {}))
            artifact_frame = artifact.get("task_frame")
            if expected_frame:
                if not isinstance(artifact_frame, dict):
                    return False, "task-frame definition is missing"
                if str(artifact_frame.get("frame_id")) != str(
                    expected_frame.get("frame_id")
                ):
                    return False, "task frame does not match"
                if str(artifact_frame.get("snapshot_policy")) != str(
                    expected_frame.get("snapshot_policy")
                ):
                    return False, "task-frame snapshot policy does not match"
            if dict(artifact.get("feature_definition", {})) != dict(
                task_definition.get("feature_definition", {})
            ):
                return False, "feature definition does not match"
            n_stages = int(artifact["num_stages"])
            expected_stages = int(self._task_profiles[task_id]["n_stages"])
            if n_stages != expected_stages:
                return False, "{} stages; task requires {}".format(
                    n_stages, expected_stages
                )
            if str(artifact.get("endpoint_coordinate_frame", "")) != str(
                task_definition.get("endpoint_coordinate_frame", "")
            ):
                return False, "endpoint coordinate frame does not match"
            endpoint_poses = artifact.get("stage_endpoint_poses_bar")
            if not isinstance(endpoint_poses, list) or len(endpoint_poses) != n_stages - 1:
                return False, "aggregated endpoint poses do not match the stages"
            for pose in endpoint_poses:
                if not isinstance(pose, list) or len(pose) != 7:
                    return False, "endpoint poses must contain xyz+xyzw"
                values = [float(value) for value in pose]
                if not all(math.isfinite(value) for value in values):
                    return False, "endpoint poses must be finite"
                if math.sqrt(sum(value * value for value in values[3:])) <= 1e-9:
                    return False, "endpoint quaternion must be nonzero"
            schema = artifact["feature_schema"]
            pairs = artifact["feature_stage_modes"]
            if not isinstance(schema, list) or not schema or not isinstance(pairs, list):
                return False, "feature schema or feature-stage matrix is missing"
            names = [str(spec["name"]) for spec in schema]
            if any(not name for name in names) or len(set(names)) != len(names):
                return False, "feature schema contains empty or duplicate names"
            expected_pairs = {
                (stage, name) for stage in range(n_stages) for name in names
            }
            seen_pairs = set()
            supported = {
                str(spec["name"])
                for spec in self._task_feature_definitions[task_id]["schema"]
            }
            true_terms = [dict(term) for term in task_definition["constraint_terms"]]
            unsupported = set()
            for pair in pairs:
                stage = int(pair["stage"])
                name = str(pair["feature_name"])
                key = (stage, name)
                if key in seen_pairs or key not in expected_pairs:
                    return False, "feature-stage matrix has duplicate or invalid pairs"
                seen_pairs.add(key)
                mode = str(pair["mode"])
                if mode not in (
                    "inactive",
                    "target_value",
                    "lower_bound",
                    "upper_bound",
                ):
                    return False, "unsupported mode {}".format(mode)
                if mode != "inactive":
                    if name not in supported:
                        unsupported.add(name)
                    value = float(pair["value"])
                    if not math.isfinite(value):
                        return False, "active constraint value is not finite"
                    exact = [
                        term for term in true_terms
                        if int(term["stage"]) == stage
                        and str(term["feature_name"]) == name
                    ]
                    candidates = exact or [
                        term for term in true_terms
                        if str(term["feature_name"]) == name
                    ]
                    parameters = {
                        (float(term["scale"]), float(term.get("weight", 1.0)))
                        for term in candidates
                    }
                    if len(parameters) != 1:
                        return False, "task-fixed scale/weight is missing or ambiguous"
                    scale, weight = next(iter(parameters))
                    if not math.isfinite(scale) or scale <= 0.0:
                        return False, "task-fixed constraint scale is invalid"
                    if not math.isfinite(weight) or weight < 0.0:
                        return False, "task-fixed constraint weight is invalid"
            if seen_pairs != expected_pairs:
                return False, "feature-stage matrix is incomplete"
            if unsupported:
                return False, "unsupported active features: " + ", ".join(
                    sorted(unsupported)
                )
            return True, ""
        except (KeyError, TypeError, ValueError):
            return False, "invalid learned constraint artifact"

    def _refresh_constraint_sources(self) -> None:
        sources = self._discover_constraint_sources()
        changed = False
        with self._lock:
            self._constraint_sources = sources
            for task_id, options in sources.items():
                valid_ids = {
                    str(option["id"])
                    for option in options
                    if option["compatible"] is True
                }
                selected = str(
                    self._gui_settings["constraint_source_by_task"].get(
                        task_id, "true"
                    )
                )
                if selected not in valid_ids:
                    self._gui_settings["constraint_source_by_task"][task_id] = "true"
                    changed = True
                    if str(self._task_state.get("task_id")) == task_id:
                        self._task_state["constraint_source_id"] = "true"
                        self._task_state["constraint_source_label"] = "True constraints"
            if changed:
                self._save_gui_settings()

    def _default_gui_settings(self) -> Dict[str, object]:
        task_id = "BarClean" if "BarClean" in self._task_profiles else next(iter(self._task_profiles))
        return {
            "schema_version": 1,
            "task_id": task_id,
            "constraint_source_by_task": {
                name: "true" for name in self._task_profiles
            },
            "start_by_task": {
                name: dict(profile["default_start"])
                for name, profile in self._task_profiles.items()
            },
            "goal_by_task": {
                name: dict(profile["default_goal"])
                for name, profile in self._task_profiles.items()
            },
            "record_video": False,
        }

    @staticmethod
    def _settings_pose(value: object) -> Optional[Dict[str, float]]:
        if not isinstance(value, dict):
            return None
        keys = ("x", "y", "z", "qx", "qy", "qz", "qw")
        try:
            pose = {key: float(value[key]) for key in keys}
        except (KeyError, TypeError, ValueError):
            return None
        if not all(math.isfinite(number) for number in pose.values()):
            return None
        norm = math.sqrt(sum(pose[key] ** 2 for key in ("qx", "qy", "qz", "qw")))
        return pose if norm > 1e-9 else None

    def _load_gui_settings(self) -> Dict[str, object]:
        settings = self._default_gui_settings()
        try:
            saved = json.loads(GUI_SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return settings
        task_id = str(saved.get("task_id", settings["task_id"]))
        if task_id in self._task_profiles:
            settings["task_id"] = task_id
        for map_name in ("start_by_task", "goal_by_task"):
            saved_poses = saved.get(map_name, {})
            if isinstance(saved_poses, dict):
                for name in self._task_profiles:
                    pose = self._settings_pose(saved_poses.get(name))
                    if pose is not None:
                        settings[map_name][name] = pose
        saved_sources = saved.get("constraint_source_by_task", {})
        if isinstance(saved_sources, dict):
            for name, options in self._constraint_sources.items():
                source_id = str(saved_sources.get(name, "true"))
                if any(option["id"] == source_id and option["compatible"] for option in options):
                    settings["constraint_source_by_task"][name] = source_id
        settings["record_video"] = saved.get("record_video", False) is True
        return settings

    def _save_gui_settings(self) -> None:
        GUI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = GUI_SETTINGS_PATH.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._gui_settings, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(GUI_SETTINGS_PATH)

    def _resolve_constraint_source(self, task_id: str, source_id: str) -> Dict[str, object]:
        self._constraint_sources = self._discover_constraint_sources()
        for source in self._constraint_sources[task_id]:
            if source["id"] == source_id:
                if source["compatible"] is not True:
                    raise ValueError(
                        "Learned constraints are incompatible: {}".format(source.get("reason", "unknown reason"))
                    )
                return source
        raise ValueError("Unknown constraint source {} for {}".format(source_id, task_id))

    def update_gui_settings(self, payload: Dict[str, object]) -> None:
        task_id = str(payload.get("task_id", self._gui_settings["task_id"]))
        if task_id not in self._task_profiles:
            raise ValueError("Unknown task_id {}".format(task_id))
        source_id = str(
            payload.get(
                "constraint_source_id",
                self._gui_settings["constraint_source_by_task"][task_id],
            )
        )
        source = self._resolve_constraint_source(task_id, source_id)
        start = self._settings_pose(payload.get("start"))
        goal = self._settings_pose(payload.get("goal"))
        with self._lock:
            self._gui_settings["task_id"] = task_id
            self._gui_settings["constraint_source_by_task"][task_id] = source_id
            if start is not None:
                self._gui_settings["start_by_task"][task_id] = start
            if goal is not None:
                self._gui_settings["goal_by_task"][task_id] = goal
            if "record_video" in payload:
                self._gui_settings["record_video"] = payload["record_video"] is True
            self._save_gui_settings()
            if str(self._task_state.get("task_id")) == task_id:
                self._task_state["constraint_source_id"] = source_id
                self._task_state["constraint_source_label"] = source["label"]

    @staticmethod
    def _load_task_feature_definitions() -> Dict[str, Dict[str, object]]:
        definitions: Dict[str, Dict[str, object]] = {}
        for task_id, path in TASK_CONFIGS.items():
            config = json.loads(path.read_text(encoding="utf-8"))
            feature_names = [str(value) for value in config["visualization_features"]]
            units = config["feature_units"]
            definitions[task_id] = {
                "source": "stage_constraint_planner/config/{}".format(path.name),
                "schema": [
                    {"name": name, "unit": str(units[name])}
                    for name in feature_names
                ],
                "true_constraints": {
                    "bar_axial_offset_reference": float(
                        config.get("bar_axial_offset_reference", 0.0)
                    )
                },
                "constraint_specs": [dict(value) for value in config["constraint_terms"]],
                "table_surface_point": [
                    float(value) for value in config["table_surface_point"]
                ],
                "table_normal": [float(value) for value in config["table_normal"]],
                "obstacle_radius": float(config["obstacle_radius"]),
            }
        return definitions

    def _reload_task_definitions(self) -> None:
        profiles = self._load_task_profiles()
        definitions = self._load_task_feature_definitions()
        with self._lock:
            self._task_profiles = profiles
            self._task_feature_definitions = definitions

    def _empty_feature_series(self, task_id: str) -> Dict[str, object]:
        definition = self._task_feature_definitions[task_id]
        series = {
            key: json.loads(json.dumps(definition[key]))
            for key in ("source", "schema", "true_constraints", "constraint_specs")
        }
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
        for name, dependency in (("bar", "optitrack_bar"),):
            geometry = demo_scene.get(name) if isinstance(demo_scene, dict) else None
            if dependencies.get(dependency) and isinstance(geometry, dict):
                scene[name] = {**geometry, "live": True}
                live_objects.append(name)
        demo_obstacles = demo_scene.get("obstacles") if isinstance(demo_scene, dict) else None
        if isinstance(demo_obstacles, list) and len(demo_obstacles) == len(scene["obstacles"]):
            scene["obstacles"] = [
                {**geometry, "live": True} for geometry in demo_obstacles
            ]
            source_indices = scene["obstacle"]["source_indices"]
            if scene["obstacle"]["type"] == "circle":
                scene["obstacle"].update(
                    center=list(scene["obstacles"][source_indices[0]]["center"]),
                    live=True,
                )
            else:
                scene["obstacle"].update(
                    endpoints=[
                        list(scene["obstacles"][index]["center"])
                        for index in source_indices
                    ],
                    live=True,
                )
            live_objects.append("obstacles")
        current_ee = demo_state.get("current_ee") if dependencies.get("ee_tf") else None
        return {
            "current_ee": current_ee,
            "scene_geometry": scene,
            "source": self._scene_pose_source() if live_objects else "fallback",
        }

    @staticmethod
    def _pose_from_topic_csv(payload: str) -> Dict[str, float]:
        try:
            row = next(csv.reader([payload]))
        except (csv.Error, StopIteration) as error:
            raise ValueError("Pose topic returned invalid CSV") from error
        # RosTopicStream removes the leading %time field. PoseStamped then has
        # seq, stamp, frame_id, position xyz, and quaternion xyzw.
        if len(row) != 10:
            raise ValueError(
                "Pose topic returned {} fields instead of 10".format(len(row))
            )
        try:
            values = [float(value) for value in row[3:10]]
        except ValueError as error:
            raise ValueError("Pose topic returned a non-numeric pose") from error
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Pose topic returned a non-finite pose")
        quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
        if quaternion_norm <= 1e-12:
            raise ValueError("Pose topic returned a zero quaternion")
        values[3:] = [value / quaternion_norm for value in values[3:]]
        return dict(zip(("x", "y", "z", "qx", "qy", "qz", "qw"), values))

    def _direct_optitrack_visualization(self) -> Optional[Dict[str, object]]:
        try:
            bar = self._pose_from_topic_csv(
                self._topic_value(
                    CONTAINER,
                    "/vrpn_client_node/baiyu_bar/pose_from_iiwa14",
                    timeout=0.2,
                    max_age=0.5,
                )
            )
            obstacles = [
                self._pose_from_topic_csv(
                    self._topic_value(
                        CONTAINER, topic, timeout=0.2, max_age=0.5
                    )
                )
                for topic in (
                    "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14",
                    "/vrpn_client_node/baiyu_obs_bar_b/pose_from_iiwa14",
                )
            ]
        except (RuntimeError, ValueError):
            return None

        rotation = self._optitrack_to_robot_rotation
        translation = self._optitrack_to_robot_translation
        bar_position = self._transform_point(
            rotation, translation, [bar["x"], bar["y"], bar["z"]]
        )
        obstacle_positions = [
            self._transform_point(
                rotation,
                translation,
                [obstacle["x"], obstacle["y"], obstacle["z"]],
            )
            for obstacle in obstacles
        ]
        x, y, z, w = (bar[key] for key in ("qx", "qy", "qz", "qw"))
        tracker_axis = [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ]
        bar_axis = self._rotate_vector(rotation, tracker_axis)
        axis_norm = math.hypot(bar_axis[0], bar_axis[1])
        if axis_norm <= 1e-12:
            return None

        scene_obstacle = json.loads(
            json.dumps(self._fixed_scene_geometry["obstacle"])
        )
        source_indices = scene_obstacle["source_indices"]
        if scene_obstacle["type"] == "circle":
            scene_obstacle.update(
                center=obstacle_positions[source_indices[0]][:2], live=True
            )
        else:
            scene_obstacle.update(
                endpoints=[
                    obstacle_positions[index][:2] for index in source_indices
                ],
                live=True,
            )
        return {
            "current_ee": None,
            "scene_geometry": {
                "bar": {
                    "pivot": bar_position[:2],
                    "axis": [bar_axis[0] / axis_norm, bar_axis[1] / axis_norm],
                    "outline_u": list(self._fixed_scene_geometry["bar"]["outline_u"]),
                    "outline_v": list(self._fixed_scene_geometry["bar"]["outline_v"]),
                    "lateral_centerline": dict(
                        self._fixed_scene_geometry["bar"]["lateral_centerline"]
                    ),
                    "live": True,
                },
                "obstacles": [
                    {
                        "center": position[:2],
                        "radius": float(reference["radius"]),
                        "live": True,
                    }
                    for position, reference in zip(
                        obstacle_positions,
                        self._fixed_scene_geometry["obstacles"],
                    )
                ],
                "obstacle": scene_obstacle,
            },
            "source": self._scene_pose_source(),
        }

    def _simulation_scene_snapshot(self) -> Dict[str, object]:
        visualization = (
            self._direct_optitrack_visualization() or self._demo_visualization()
        )
        if visualization is None:
            scene = self._fallback_scene_geometry()
            source = "fallback"
        else:
            scene = visualization.get("scene_geometry")
            source = str(visualization.get("source", "fallback"))
        if not isinstance(scene, dict):
            raise RuntimeError("Task scene snapshot is unavailable")
        bar = scene.get("bar")
        obstacles = scene.get("obstacles")
        if not isinstance(bar, dict) or not isinstance(obstacles, list) or not obstacles:
            raise RuntimeError("Task scene snapshot is missing bar or obstacle geometry")
        try:
            pivot = [float(value) for value in bar["pivot"]]
            axis = [float(value) for value in bar["axis"]]
            centers = [
                [float(value) for value in obstacle["center"]]
                for obstacle in obstacles
            ]
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Task scene snapshot contains invalid geometry") from error
        if (
            len(pivot) != 2
            or len(axis) != 2
            or any(len(center) != 2 for center in centers)
            or not all(
                math.isfinite(value)
                for value in pivot + axis + [item for center in centers for item in center]
            )
        ):
            raise RuntimeError("Task scene snapshot contains invalid geometry")
        axis_norm = math.hypot(axis[0], axis[1])
        if axis_norm <= 1e-12:
            raise RuntimeError("Task scene snapshot contains a zero bar axis")
        snapshot = {
            "source": source,
            "bar": {
                "pivot": pivot,
                "axis": [axis[0] / axis_norm, axis[1] / axis_norm],
                "lateral_centerline": dict(
                    bar.get("lateral_centerline", {"type": "straight"})
                ),
                "live": bar.get("live") is True,
            },
            "obstacles": [
                {
                    "center": center,
                    "live": obstacle.get("live") is True,
                }
                for center, obstacle in zip(centers, obstacles)
            ],
        }
        self.log(
            "Frozen simulation scene from {}: bar=({:.3f}, {:.3f}), "
            "obstacles={}".format(
                source, pivot[0], pivot[1], centers
            )
        )
        return snapshot

    def task_visualization(self) -> Dict[str, object]:
        with self._lock:
            mode = str(self._task_state.get("mode", "simulator"))
            phase = str(self._task_state.get("phase", "idle"))
            use_simulation = (
                mode == "simulator"
                and phase == "executing"
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
            keep_planner_scene = (
                source == "planner_scene" and phase in TASK_ACTIVE_PHASES
            )
            if not keep_planner_scene:
                live_visualization = (
                    self._direct_optitrack_visualization() or demo_visualization
                )
                if live_visualization is not None:
                    scene = live_visualization["scene_geometry"]
                    source = live_visualization["source"]
                else:
                    current_ee = None
                    scene = self._fallback_scene_geometry()
                    source = "fallback"
            elif demo_visualization is None:
                current_ee = None

        return {
            "ok": True,
            "task_id": str(self._task_state.get("task_id", "BarClean")),
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
        self._reload_task_definitions()
        self._refresh_constraint_sources()
        nodes = self._ros_nodes()
        driver_node_running = "/iiwa14/iiwa_driver" in nodes
        driver_process_running = self._driver_binary_running()
        controllers = (
            self._running_iiwa_controllers()
            if driver_node_running
            else []
        )
        if driver_process_running and not driver_node_running:
            robot_control = {
                "mode": "incomplete",
                "label": "Driver process exists without a ROS node",
                "healthy": False,
            }
        else:
            robot_control = self._control_mode_from_graph(nodes, controllers)
        with self._lock:
            busy = self._job is not None and self._job.is_alive()
            return {
                "token": self.token,
                "project_root": str(PROJECT_ROOT),
                "available_tasks": [
                    {"task_id": task_id, **profile}
                    for task_id, profile in self._task_profiles.items()
                ],
                "constraint_sources": json.loads(json.dumps(self._constraint_sources)),
                "gui_settings": json.loads(json.dumps(self._gui_settings)),
                "scene_pose_source": self._scene_pose_source(),
                "scene_name": self._scene_name(),
                "available_scenes": self._available_scene_names(),
                "container_running": self._container_running(),
                "driver_running": driver_node_running or driver_process_running,
                "robot_control": robot_control,
                "camera_source_running": self.camera_source_running(),
                "camera_preview_running": self.camera_preview_running(),
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
        self._ensure_data_directories()
        environment = {
            "SIM_AUTO_PLAN": "false",
            "SIM_RENDER_VIDEO": "true",
        }
        recreate = False
        if self._named_container_running(SIM_CONTAINER):
            self._assert_container_storage_access(SIM_CONTAINER, ("/data/sim_runs",))
            try:
                self._wait_for_simulator(timeout=3.0)
                status = self._read_sim_status()
            except (RuntimeError, subprocess.TimeoutExpired):
                self.log("Simulator is unhealthy; recreating its container")
                recreate = True
            else:
                if any(
                    key not in status
                    for key in (
                        "task_sequence", "task_id", "video_capable",
                        "scene_snapshot_source",
                    )
                ):
                    self.log("Simulator container is outdated; recreating it from the current image")
                    recreate = True
                elif require_video and status.get("video_capable") is not True:
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
        self._assert_container_storage_access(SIM_CONTAINER, ("/data/sim_runs",))
        status = self._read_sim_status()
        if any(
            key not in status
            for key in (
                "task_sequence", "task_id", "video_capable",
                "scene_snapshot_source",
            )
        ):
            raise RuntimeError(
                "Simulator image is outdated; rebuild the workstation image once"
            )
        if require_video and status.get("video_capable") is not True:
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
        render_video: bool,
        scene_snapshot: Dict[str, object],
        constraint_source: str = "true",
    ) -> None:
        payload = json.dumps(
            {
                "task_id": task_id,
                "start": start,
                "goal": goal,
                "render_video": render_video,
                "scene_snapshot": scene_snapshot,
                "constraint_source": constraint_source,
            },
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
        data_line = self._topic_value(
            SIM_CONTAINER,
            "/iiwa14/sim/status",
            timeout=3.0,
            max_age=0.75,
        )
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
        # rostopic -p stores std_msgs/String.data in one CSV field. Planner
        # visualizations can legitimately exceed Python's 128 KiB default.
        csv.field_size_limit(16 * 1024 * 1024)
        for row in csv.reader(io.StringIO(output)):
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
        try:
            return json.loads(
                self._topic_value(
                    CONTAINER,
                    "/iiwa14/real_executor/status",
                    timeout=3.0,
                    max_age=0.75,
                )
            )
        except json.JSONDecodeError as error:
            raise RuntimeError("Real executor returned an invalid status message") from error

    def _wait_for_real_plan(
        self, previous_serial: int, task_id: str, timeout: float = 10.0
    ) -> bool:
        deadline = time.monotonic() + timeout
        last_error = ""
        while time.monotonic() < deadline:
            if self._task_abort.is_set():
                return False
            try:
                status = self._read_real_status()
            except RuntimeError as error:
                last_error = str(error)
                time.sleep(0.1)
                continue
            try:
                path_serial = int(status.get("path_serial", -1))
            except (TypeError, ValueError):
                path_serial = -1
            if (
                path_serial > previous_serial
                and str(status.get("task_id", "")) == task_id
                and str(status.get("phase", "")) == "path_received"
            ):
                self.log(
                    "Real executor acknowledged planner path serial {}".format(
                        path_serial
                    )
                )
                return True
            time.sleep(0.1)
        detail = ": " + last_error if last_error else ""
        raise RuntimeError(
            "Real executor did not acknowledge this task's new planner path" + detail
        )

    def _read_real_fri_ready(self) -> Tuple[bool, str]:
        try:
            ready_value = self._topic_value(
                CONTAINER,
                "/iiwa14/real_executor/fri_ready_status",
                timeout=3.0,
                max_age=0.5,
            )
        except RuntimeError:
            return False, "waiting for FRI readiness status"
        # ``rostopic echo -p`` serializes std_msgs/Bool as numeric CSV (1/0),
        # while some ROS versions/tools use True/False.  Accept both forms.
        ready = ready_value.strip().lower() in {"1", "true"}
        return ready, (
            "FRI POSITION mode is COMMANDING_ACTIVE"
            if ready
            else "waiting for COMMANDING_ACTIVE + POSITION"
        )

    def _wait_for_fri_position(self, timeout: float = 180.0) -> bool:
        self._set_task_state(
            phase="waiting_for_fri",
            message="SmartPAD: start FRIOverlayGripper, then select Position",
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._task_abort.is_set():
                return False
            ready, reason = self._read_real_fri_ready()
            if ready:
                self.log("FRI POSITION mode is COMMANDING_ACTIVE")
                return True
            self._set_task_state(
                phase="waiting_for_fri",
                message="Start FRIOverlayGripper on SmartPAD, then select Position — "
                + reason,
            )
            time.sleep(0.5)
        raise RuntimeError(
            "Timed out waiting for FRIOverlayGripper POSITION mode; retry from the GUI"
        )

    def _wait_for_real_station(
        self, child: Optional[subprocess.Popen[str]], timeout: float = 20.0
    ) -> None:
        required = {
            "/iiwa14/iiwa_driver",
            "/iiwa14/real_executor",
            "/stage_constraint_planner",
        }
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if child is not None and child.poll() is not None:
                raise RuntimeError(
                    "Real station exited during startup; inspect the supervisor log"
                )
            if (
                required.issubset(set(self._ros_nodes()))
                and self._real_station_interfaces_ready()
            ):
                with self._lock:
                    self._real_station_verified = True
                return
            time.sleep(0.5)
        raise RuntimeError(
            "Real station interfaces did not become ready within {:.0f} s".format(
                timeout
            )
        )

    def _optitrack_settings(self) -> Tuple[str, str, str, str, str]:
        return (
            self._read_env_value("OPTITRACK_SERVER", "128.178.145.104"),
            self._read_env_value("OPTITRACK_BASE", "iiwa14"),
            self._read_env_value("OPTITRACK_OBJECT", "baiyu_bar"),
            self._read_env_value("OPTITRACK_OBSTACLE", "baiyu_obs_bar"),
            self._read_env_value("OPTITRACK_OBSTACLE_B", "baiyu_obs_bar_b"),
        )

    def _scene_pose_source(self) -> str:
        source = self._read_env_value("SCENE_POSE_SOURCE", "fixed").lower()
        if source not in {"fixed", "optitrack"}:
            raise RuntimeError("SCENE_POSE_SOURCE must be fixed or optitrack")
        return source

    @staticmethod
    def _available_scene_names() -> List[str]:
        return sorted(path.stem for path in SCENE_CONFIG_DIR.glob("scene*.json"))

    @staticmethod
    def _scene_name() -> str:
        config = json.loads(SCENE_CONFIG.read_text(encoding="utf-8"))
        return str(config["source"]["scene_name"])

    @staticmethod
    def _write_env_value(key: str, value: str) -> None:
        env_file = PROJECT_ROOT / ".env"
        lines = (
            env_file.read_text(encoding="utf-8").splitlines()
            if env_file.exists()
            else []
        )
        replacement = "{}={}".format(key, value)
        updated: List[str] = []
        replaced = False
        for raw in lines:
            stripped = raw.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                current_key = stripped.split("=", 1)[0].strip()
                if current_key == key:
                    if not replaced:
                        updated.append(replacement)
                        replaced = True
                    continue
            updated.append(raw)
        if not replaced:
            updated.append(replacement)
        temporary = env_file.with_name(env_file.name + ".tmp")
        temporary.write_text("\n".join(updated) + "\n", encoding="utf-8")
        if env_file.exists():
            os.chmod(temporary, env_file.stat().st_mode)
        temporary.replace(env_file)

    def _scene_tracking_nodes(self) -> set:
        if self._scene_pose_source() == "fixed":
            return {
                "/fixed_scene_bar_publisher",
                "/fixed_scene_obstacle_a_publisher",
                "/fixed_scene_obstacle_b_publisher",
            }
        return {"/vrpn_client_node", "/optitrack_base_transform"}

    def _inactive_scene_tracking_nodes(self) -> set:
        if self._scene_pose_source() == "fixed":
            return {"/vrpn_client_node", "/optitrack_base_transform"}
        return {
            "/fixed_scene_bar_publisher",
            "/fixed_scene_obstacle_a_publisher",
            "/fixed_scene_obstacle_b_publisher",
        }

    @staticmethod
    def _topic_has_fresh_message(
        topic: str, timeout: float = 3.0, max_age: float = 1.0
    ) -> bool:
        try:
            result = subprocess.run(
                [
                    "docker", "exec", CONTAINER,
                    "timeout", "{:.2f}s".format(timeout),
                    "/entrypoint.sh", "rostopic", "echo", "-n", "1", "-p", topic,
                ],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                timeout=timeout + 1.0,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False
        if result.returncode != 0:
            return False
        rows = list(csv.reader(io.StringIO(result.stdout)))
        if len(rows) < 2:
            return False
        header = rows[0]
        try:
            stamp_index = header.index("field.header.stamp")
            raw_stamp = float(rows[-1][stamp_index])
        except (ValueError, IndexError):
            return False
        stamp_seconds = raw_stamp / 1e9 if raw_stamp > 1e12 else raw_stamp
        age = time.time() - stamp_seconds
        return -0.1 <= age <= max_age

    def _wait_for_optitrack_ready(
        self,
        child: Optional[subprocess.Popen[str]],
        base: str,
        obj: str,
        obstacle_a: str,
        obstacle_b: str,
        timeout: float = 20.0,
    ) -> None:
        source = self._scene_pose_source()
        required_nodes = self._scene_tracking_nodes()
        raw_topics = () if source == "fixed" else (
            ("base", base, "/vrpn_client_node/{}/pose".format(base)),
            ("object", obj, "/vrpn_client_node/{}/pose".format(obj)),
            ("obstacle A", obstacle_a, "/vrpn_client_node/{}/pose".format(obstacle_a)),
            ("obstacle B", obstacle_b, "/vrpn_client_node/{}/pose".format(obstacle_b)),
        )
        transformed_topics = (
            "/vrpn_client_node/{}/pose_from_{}".format(obj, base),
            "/vrpn_client_node/{}/pose_from_{}".format(obstacle_a, base),
            "/vrpn_client_node/{}/pose_from_{}".format(obstacle_b, base),
        )
        deadline = time.monotonic() + timeout
        missing_raw = list(raw_topics)
        missing_transformed = list(transformed_topics)
        while time.monotonic() < deadline:
            if child is not None and child.poll() is not None:
                raise RuntimeError(
                    "OptiTrack launch exited during startup; inspect the supervisor log"
                )
            if required_nodes.issubset(set(self._ros_nodes())):
                missing_raw = [
                    item
                    for item in raw_topics
                    if not self._topic_has_fresh_message(item[2])
                ]
                if missing_raw:
                    missing_transformed = list(transformed_topics)
                else:
                    missing_transformed = [
                        topic
                        for topic in transformed_topics
                        if not self._topic_has_fresh_message(topic)
                    ]
                if not missing_raw and not missing_transformed:
                    self.log(
                        "Scene input is ready from {}: {}, {}, and {} are publishing fresh poses".format(
                            source, obj, obstacle_a, obstacle_b
                        )
                    )
                    return
            time.sleep(0.25)
        if missing_raw:
            missing_roles = {role for role, _name, _topic in missing_raw}
            if missing_roles == {"base"}:
                raise RuntimeError(
                    "OptiTrack objects are available, but base rigid body '{}' did not "
                    "deliver a fresh raw pose on {}. Base-relative object poses cannot "
                    "be computed. In Motive, enable and track exactly '{}' and disable "
                    "any duplicate robot-base rigid body.".format(
                        base, missing_raw[0][2], base
                    )
                )
            raise RuntimeError(
                "OptiTrack did not deliver fresh raw poses for: {}. Confirm these exact "
                "rigid-body names are enabled and tracked in Motive.".format(
                    ", ".join(
                        "{} ({})".format(name, topic)
                        for _role, name, topic in missing_raw
                    )
                )
            )
        if source == "fixed":
            raise RuntimeError(
                "Fixed scene publishers did not deliver fresh poses: {}".format(
                    ", ".join(missing_transformed)
                )
            )
        raise RuntimeError(
            "OptiTrack raw poses are fresh, but these base-relative topics did not "
            "deliver a new pose: {}. Inspect the base-transform node.".format(
                ", ".join(missing_transformed)
            )
        )

    def _start_optitrack_process(self) -> None:
        if not self._container_running():
            raise RuntimeError("Container is not running")
        server, base, obj, obstacle_a, obstacle_b = self._optitrack_settings()
        source = self._scene_pose_source()
        required_nodes = self._scene_tracking_nodes()
        nodes = set(self._ros_nodes())
        conflicts = nodes.intersection(self._inactive_scene_tracking_nodes())
        if conflicts:
            raise RuntimeError(
                "Scene source changed to '{}', but nodes from the other source are still running: {}. "
                "Stop the workstation once, then restart it.".format(
                    source, ", ".join(sorted(conflicts))
                )
            )
        if required_nodes.issubset(nodes):
            self.log("Reusing the running OptiTrack ROS chain")
            self._wait_for_optitrack_ready(None, base, obj, obstacle_a, obstacle_b)
            return
        partial = required_nodes.intersection(nodes)
        if partial:
            raise RuntimeError(
                "OptiTrack ROS chain is only partially running ({}); stop the "
                "workstation once, then retry".format(", ".join(sorted(partial)))
            )
        child = self._spawn(
            "tracking",
            [
                "docker", "exec", CONTAINER, "/entrypoint.sh",
                "roslaunch", "stage_optitrack", "optitrack.launch",
                "server:={}".format(server),
                "base_name:={}".format(base),
                "object_name:={}".format(obj),
                "obstacle_a_name:={}".format(obstacle_a),
                "obstacle_b_name:={}".format(obstacle_b),
                "use_fixed_scene:={}".format("true" if source == "fixed" else "false"),
            ],
        )
        try:
            self._wait_for_optitrack_ready(child, base, obj, obstacle_a, obstacle_b)
        except Exception:
            self._signal_child("tracking")
            raise

    def _start_real_station(self, require_optitrack: bool = True) -> None:
        # Planner tasks consume live object poses. Return Home is deliberately
        # independent of OptiTrack and starts the same position station without
        # waiting for any tracked rigid body.
        station_nodes = {
            "/iiwa14/iiwa_driver",
            "/iiwa14/real_executor",
            "/stage_constraint_planner",
        }
        tracking_nodes = self._scene_tracking_nodes()
        nodes = set(self._ros_nodes())
        station_running = station_nodes.issubset(nodes)
        tracking_running = tracking_nodes.issubset(nodes)
        if station_running and (not require_optitrack or tracking_running):
            with self._lock:
                already_verified = self._real_station_verified
            if already_verified:
                self.log("Reusing the verified Position station (fast path)")
                return
            # A supervisor restart forgets its in-memory readiness result even
            # though the ROS station may still be healthy. Verify the service
            # contract once, then use the node-only fast path on later tasks.
            if self._real_station_interfaces_ready():
                with self._lock:
                    self._real_station_verified = True
                self.log("Reusing the running Position station after one interface check")
                return
        if require_optitrack:
            self._start_optitrack_process()
        nodes = set(self._ros_nodes())
        if "/iiwa14/real_executor" in nodes:
            self._wait_for_real_station(None, timeout=5.0)
            return
        if "/iiwa14/iiwa_driver" in nodes:
            self._release_robot_control("switching to Planner / Position control")
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
        try:
            self._wait_for_real_station(child)
        except Exception:
            self.log("Real station startup failed; removing the partial control chain")
            self._stop_driver_process()
            raise

    def _abort_real_and_confirm(self, reason: str) -> None:
        self.log("Requesting fail-closed real-task abort: " + str(reason))
        try:
            response = self._run(
                [
                    "docker", "exec", CONTAINER, "/entrypoint.sh", "rosservice",
                    "call", "/iiwa14/real_executor/abort",
                ],
                check=False,
                timeout=5.0,
            )
        except subprocess.TimeoutExpired as error:
            self._set_task_state(
                phase="control_status_unknown",
                message=(
                    "GUI lost execution supervision and abort timed out. "
                    "Use SmartPAD stop or emergency stop."
                ),
            )
            raise RuntimeError("real-executor abort timed out; control status is unknown") from error
        if response.returncode != 0 or "success: True" not in response.stdout:
            self._set_task_state(
                phase="control_status_unknown",
                message=(
                    "GUI lost execution supervision and abort was not accepted. "
                    "Use SmartPAD stop or emergency stop."
                ),
            )
            raise RuntimeError("real-executor abort was not accepted; control status is unknown")

        deadline = time.monotonic() + 5.0
        last_error = ""
        while time.monotonic() < deadline:
            try:
                status = self._read_real_status()
            except RuntimeError as error:
                last_error = str(error)
                time.sleep(0.1)
                continue
            if (
                status.get("execution_active") is False
                and status.get("holding_final_position") is not True
                and str(status.get("phase", ""))
                in {"aborted", "failed", "protective_stop", "rejected"}
            ):
                self.log("Real executor confirmed the robot command path is stopped")
                return
            time.sleep(0.1)

        self._set_task_state(
            phase="control_status_unknown",
            message=(
                "Abort was requested but the stopped state could not be confirmed. "
                "Use SmartPAD stop or emergency stop."
            ),
        )
        detail = ": " + last_error if last_error else ""
        raise RuntimeError(
            "could not confirm real-executor abort; control status is unknown" + detail
        )

    def _execute_real_task(
        self,
        task_id: str,
        start: Dict[str, float],
        goal: Dict[str, float],
        constraint_source: str = "true",
        record_video: bool = False,
    ) -> None:
        if not self._container_running():
            raise RuntimeError("Start the workstation container first")
        self._assert_container_storage_access(
            CONTAINER, ("/data/demos", "/data/real_runs")
        )
        if not self._robot_iface_state()["configured"]:
            raise RuntimeError("Robot network interface is not configured")
        # The PyBullet station has its own bridge-network container and ROS graph.
        # It cannot command the FRI driver, so it may remain available while a real
        # task is prepared or executed.
        self._reset_task_visualization(task_id)
        self._set_task_state(
            task_id=task_id, mode="real", phase="starting",
            data_saved=True, video=record_video,
            start=start, goal=goal,
            message="Checking or reusing the iiwa14 Position station",
            run_directory=None, video_available=False,
            review_pending=False, review_status=None,
            final_result_directory=None,
            constraint_source=constraint_source,
        )
        if record_video:
            self._set_task_state(
                message="Checking the 1920x1080 camera stream before robot preparation"
            )
            self._task_video_recorder.preflight()
            self.log("Task video source /dev/video10 is ready at 1920x1080")
        self._start_real_station()
        if self._task_abort.is_set():
            self._abort_real_and_confirm("user stopped the task during station startup")
            self._set_task_state(
                phase="aborted", message="Real task stopped before planning"
            )
            return
        initial_status = self._read_real_status()
        initial_phase = str(initial_status.get("phase", "unknown"))
        if (
            initial_status.get("execution_active") is True
            or initial_phase
            in {"preparing", "repreparing", "moving_to_start", "executing"}
        ):
            self._set_task_state(
                phase=initial_phase,
                message="The real executor is already running a task",
            )
            raise RuntimeError(
                "The real executor is already active; abort or wait for it before submitting a new plan"
            )
        if initial_status.get("protective_stop") is True:
            raise RuntimeError(
                "The real executor has a latched protective stop; inspect and restart the real station"
            )
        try:
            previous_path_serial = int(initial_status["path_serial"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(
                "Real executor status has no valid path serial; rebuild the workstation image"
            ) from error
        self._set_task_state(
            phase="planning",
            message="Planning the Cartesian task; the robot will not move yet",
        )
        plan_request = json.dumps(
            {
                "task_id": task_id,
                "start": start,
                "goal": goal,
                "constraint_source": constraint_source,
            },
            separators=(",", ":"),
        )
        response = self._real_ros(
            "rosrun", "stage_real_executor", "submit_real_plan.py", plan_request,
            timeout=30.0,
        )
        lines = [line for line in response.stdout.splitlines() if line.strip()]
        try:
            plan_result = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError) as error:
            raise RuntimeError("Real plan submitter returned invalid output") from error
        if plan_result.get("success") is not True:
            raise RuntimeError(
                "Planner rejected the real task: "
                + str(plan_result.get("message", "no reason returned"))
            )
        if not self._wait_for_real_plan(previous_path_serial, task_id):
            self._set_task_state(
                phase="aborted", message="Real task aborted after planning"
            )
            return
        self._update_plan_visualization(self._read_plan_visualization(CONTAINER))
        if not self._wait_for_fri_position():
            self._set_task_state(
                phase="aborted", message="Real task aborted while waiting for FRI"
            )
            return
        self._real_ros(
            "rosservice", "call", "/iiwa14/real_executor/set_recording",
            "data: true",
        )
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
        # Once this request is sent, a timeout is ambiguous: the executor may
        # already have armed and started. Any subsequent supervisor failure must
        # therefore explicitly abort and confirm the stopped state.
        execution_may_have_started = True
        video_started = False
        preview_suspended = False
        try:
            if record_video:
                # Keep the preview available while the operator checks the
                # framing and while the task is prepared. Close it before the
                # execute request so FFmpeg is the only /dev/video10 reader
                # during the synchronized motion window.
                self._sync_camera_preview(False)
                preview_suspended = True
                self.log(
                    "Closed camera preview before robot execution to reserve "
                    "the video stream for recording"
                )
            response = self._real_ros(
                "rosservice", "call", "/iiwa14/real_executor/execute", timeout=10.0
            )
            if "success: True" not in response.stdout:
                raise RuntimeError(
                    "Real executor refused to start: " + response.stdout.strip()
                )

            deadline = time.monotonic() + 600.0
            last_phase = ""
            while time.monotonic() < deadline:
                if self._task_abort.is_set():
                    self._set_task_state(
                        phase="aborted", message="Real task aborted by user"
                    )
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
                if phase == "executing":
                    if record_video and not video_started:
                        run_directory = self._real_run_host_directory(
                            status.get("run_directory")
                        )
                        try:
                            motion_start_unix_ns = int(
                                status["motion_start_unix_ns"]
                            )
                            motion_end_unix_ns = int(
                                status["motion_end_unix_ns"]
                            )
                        except (KeyError, TypeError, ValueError) as error:
                            raise RuntimeError(
                                "Real executor status has no valid task-motion video window; "
                                "rebuild and restart the workstation"
                            ) from error
                        if time.time_ns() >= motion_start_unix_ns:
                            raise RuntimeError(
                                "Video recorder received the task-motion start too late; "
                                "aborting instead of saving an unsynchronized experiment"
                            )
                        self._task_video_recorder.start(
                            run_directory,
                            motion_start_unix_ns,
                            motion_end_unix_ns,
                        )
                        video_started = True
                        self.log(
                            "Task video armed for the exact executing interval "
                            "({:.3f} s)".format(
                                (motion_end_unix_ns - motion_start_unix_ns) / 1e9
                            )
                        )
                    self._update_real_visualization()
                if phase == "complete":
                    self._update_real_visualization()
                    execution_may_have_started = False
                    if record_video:
                        video_path = self._task_video_recorder.stop(completed=True)
                        if video_path is None:
                            detail = getattr(
                                self._task_video_recorder,
                                "last_error",
                                None,
                            )
                            raise RuntimeError(
                                "Robot completed, but the synchronized camera recording "
                                "is incomplete: " + (detail or "unknown video error")
                            )
                        self.log("Task video saved: " + str(video_path))
                        self._set_task_state(
                            video_available=True,
                            review_pending=True,
                            review_status="pending",
                            final_result_directory=None,
                            message=(
                                "Task and video completed; review the playback, then "
                                "accept or reject this run as a final result"
                            ),
                        )
                    return
                if phase in ("failed", "rejected", "protective_stop"):
                    raise RuntimeError(message)
                if phase == "aborted":
                    execution_may_have_started = False
                    return
                time.sleep(0.1)
            raise RuntimeError("Real task exceeded the 600 s timeout")
        except Exception as error:
            if execution_may_have_started:
                self._abort_real_and_confirm(str(error))
            raise
        finally:
            if self._task_video_recorder.started:
                self._task_video_recorder.stop(completed=False)
                self.log("Incomplete task video deleted after interruption")
            if preview_suspended:
                try:
                    self._sync_camera_preview(True)
                    self.log("Restored camera preview after task recording")
                except Exception as error:
                    self.log(
                        "Could not restore camera preview after task recording: "
                        + str(error)
                    )

    def _return_robot_home(self) -> None:
        if not self._container_running():
            raise RuntimeError("Start the workstation container first")
        if not self._robot_iface_state()["configured"]:
            raise RuntimeError("Robot network interface is not configured")
        with self._lock:
            task_id = str(self._task_state.get("task_id", "BarClean"))
        self._set_task_state(
            mode="home",
            phase="starting",
            data_saved=False,
            video=False,
            message="Starting Position control for Return Home (OptiTrack not required)",
            run_directory=None,
            video_available=False,
        )
        self._start_real_station(require_optitrack=False)
        if self._task_abort.is_set():
            self._abort_real_and_confirm("user stopped Return Home during station startup")
            self._set_task_state(
                phase="aborted", message="Return Home stopped before FRI"
            )
            return
        status = self._read_real_status()
        if status.get("execution_active") is True or str(status.get("phase", "")) in {
            "moving_to_start", "executing", "home_recovering", "returning_home"
        }:
            raise RuntimeError("The real executor is already running a robot motion")
        if status.get("protective_stop") is True:
            raise RuntimeError(
                "The real executor has a latched protective stop; inspect and restart the real station"
            )
        if not self._wait_for_fri_position():
            self._set_task_state(
                phase="aborted", message="Return Home stopped while waiting for FRI"
            )
            return
        self._set_task_state(
            phase="home_preparing",
            message="Validating the joint-posture trajectory to Robot Home",
        )
        execution_may_have_started = True
        try:
            response = self._real_ros(
                "rosservice",
                "call",
                "/iiwa14/real_executor/return_home",
                timeout=70.0,
            )
            if "success: True" not in response.stdout:
                raise RuntimeError(
                    "Real executor refused Return Home: " + response.stdout.strip()
                )
            deadline = time.monotonic() + 300.0
            last_phase = ""
            while time.monotonic() < deadline:
                if self._task_abort.is_set():
                    self._set_task_state(
                        phase="aborted", message="Return Home stopped by user"
                    )
                    return
                status = self._read_real_status()
                if str(status.get("operation", "")) != "home":
                    raise RuntimeError(
                        "Real executor switched away from the Return Home operation"
                    )
                phase = str(status.get("phase", "unknown"))
                if phase != last_phase:
                    self.log("Return Home phase: " + phase)
                    last_phase = phase
                message = str(status.get("message", phase))
                self._set_task_state(
                    task_id=task_id,
                    phase=phase,
                    message=message,
                    run_directory=None,
                    video_available=False,
                )
                if phase == "complete":
                    execution_may_have_started = False
                    return
                if phase in ("failed", "rejected", "protective_stop"):
                    raise RuntimeError(message)
                if phase == "aborted":
                    execution_may_have_started = False
                    return
                time.sleep(0.1)
            raise RuntimeError("Return Home exceeded the 300 s timeout")
        except Exception as error:
            if execution_may_have_started:
                self._abort_real_and_confirm(str(error))
            raise

    def return_robot_home(self) -> None:
        with self._lock:
            if self._video_review_pending(self._task_state):
                raise RuntimeError(
                    "Review the completed task video before starting Return Home"
                )
        self._start_job(
            "Return robot Home", self._return_robot_home, reset_task_abort=True
        )

    def _set_task_state(self, **values: object) -> None:
        with self._lock:
            self._task_state.update(values)

    @staticmethod
    def _video_review_pending(task_state: Dict[str, object]) -> bool:
        return (
            task_state.get("review_pending") is True
            and task_state.get("video") is True
            and task_state.get("video_available") is True
        )

    def _reset_task_visualization(self, task_id: Optional[str] = None) -> None:
        self._reload_task_definitions()
        if task_id is None:
            with self._lock:
                task_id = str(self._task_state.get("task_id", "BarClean"))
        with self._lock:
            self._task_trace.clear()
            self._task_current_ee = None
            self._task_scene_geometry = self._fallback_scene_geometry()
            self._task_scene_source = "fallback"
            self._task_feature_series = self._empty_feature_series(task_id)
            self._task_planned_trace = []
            self._task_planned_feature_series = self._empty_feature_series(task_id)
            self._task_stage_boundary_indices = []
            self._task_stage_boundary_times = []
            self._task_stage_transition_end_times = []
            self._task_execution_started = None
            self._task_last_feature_sample = -math.inf

    @staticmethod
    def _tool_axis_from_pose(pose: Dict[str, float]) -> List[float]:
        x, y, z, w = (pose[key] for key in ("qx", "qy", "qz", "qw"))
        return [
            2.0 * (x * z + y * w),
            2.0 * (y * z - x * w),
            1.0 - 2.0 * (x * x + y * y),
        ]

    @staticmethod
    def _tool_x_from_pose(pose: Dict[str, float]) -> List[float]:
        x, y, z, w = (pose[key] for key in ("qx", "qy", "qz", "qw"))
        return [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        ]

    def _real_feature_values(
        self,
        task_id: str,
        pose: Dict[str, float],
        scene: Dict[str, object],
    ) -> Dict[str, float]:
        definition = self._task_feature_definitions[task_id]
        bar = scene["bar"]
        obstacle = scene["obstacle"]
        axis = [float(value) for value in bar["axis"]]
        axis_norm = math.hypot(axis[0], axis[1])
        if axis_norm <= 1e-12:
            raise ValueError("Planner scene contains a zero bar axis")
        axis = [axis[0] / axis_norm, axis[1] / axis_norm, 0.0]
        lateral = [-axis[1], axis[0], 0.0]
        normal = [float(value) for value in definition["table_normal"]]
        normal_norm = math.sqrt(sum(value * value for value in normal))
        if normal_norm <= 1e-12:
            raise ValueError("Task config contains a zero table normal")
        normal = [value / normal_norm for value in normal]
        position = [pose["x"], pose["y"], pose["z"]]
        table_point = [float(value) for value in definition["table_surface_point"]]
        pivot = [float(value) for value in bar["pivot"]]
        obstacle_type = str(obstacle.get("type"))
        if obstacle_type == "circle":
            center = [float(value) for value in obstacle["center"]]
            if len(center) != 2:
                raise ValueError("Planner scene contains invalid circle geometry")
            obstacle_clearance = math.hypot(
                position[0] - center[0], position[1] - center[1]
            ) - float(obstacle["radius"])
        elif obstacle_type == "capsule":
            obstacle_endpoints = [
                [float(value) for value in endpoint]
                for endpoint in obstacle["endpoints"]
            ]
            if len(obstacle_endpoints) != 2 or any(
                len(endpoint) != 2 for endpoint in obstacle_endpoints
            ):
                raise ValueError("Planner scene contains invalid capsule geometry")
            segment = [
                obstacle_endpoints[1][index] - obstacle_endpoints[0][index]
                for index in range(2)
            ]
            denominator = sum(value * value for value in segment)
            if denominator <= 1e-12:
                raise ValueError("Planner capsule endpoints are coincident")
            relative_obstacle = [
                position[index] - obstacle_endpoints[0][index]
                for index in range(2)
            ]
            phase = max(
                0.0,
                min(
                    1.0,
                    sum(
                        relative_obstacle[index] * segment[index]
                        for index in range(2)
                    )
                    / denominator,
                ),
            )
            closest_obstacle = [
                obstacle_endpoints[0][index] + phase * segment[index]
                for index in range(2)
            ]
            obstacle_clearance = math.hypot(
                position[0] - closest_obstacle[0],
                position[1] - closest_obstacle[1],
            ) - float(obstacle["radius"])
        else:
            raise ValueError("Planner scene contains unknown obstacle geometry")
        relative_bar = [position[0] - pivot[0], position[1] - pivot[1], 0.0]
        tool_axis = self._tool_axis_from_pose(pose)
        tool_x = self._tool_x_from_pose(pose)
        tool_x_normal = sum(tool_x[index] * normal[index] for index in range(3))
        tool_x_horizontal = [
            tool_x[index] - tool_x_normal * normal[index]
            for index in range(3)
        ]
        tool_x_horizontal_norm = math.sqrt(
            sum(value * value for value in tool_x_horizontal)
        )
        if tool_x_horizontal_norm <= 1e-12:
            raise ValueError("Tool-X cannot be projected into the table plane")
        tool_x_horizontal = [
            value / tool_x_horizontal_norm for value in tool_x_horizontal
        ]
        down_component = -sum(tool_axis[index] * normal[index] for index in range(3))
        forward_component = sum(tool_axis[index] * axis[index] for index in range(3))
        plane_component = sum(tool_axis[index] * lateral[index] for index in range(3))
        axial_reference = float(
            definition["true_constraints"].get("bar_axial_offset_reference", 0.0)
        )
        raw_bar_axial = sum(
            relative_bar[index] * axis[index] for index in range(3)
        )
        raw_bar_lateral = sum(
            relative_bar[index] * lateral[index] for index in range(3)
        )
        return {
            "obstacle_clearance": obstacle_clearance,
            "table_dist": sum(
                (position[index] - table_point[index]) * normal[index]
                for index in range(3)
            ),
            "bar_lateral_offset": raw_bar_lateral
            - _bar_centerline_lateral_offset(
                raw_bar_axial, bar.get("lateral_centerline")
            ),
            "tool_pitch": math.atan2(down_component, forward_component),
            "tool_roll": math.asin(max(-1.0, min(1.0, plane_component))),
            "tool_yaw": math.atan2(
                sum(
                    tool_x_horizontal[index] * lateral[index]
                    for index in range(3)
                ),
                sum(
                    tool_x_horizontal[index] * axis[index]
                    for index in range(3)
                ),
            ),
            "bar_axial_offset": raw_bar_axial - axial_reference,
        }

    def _update_real_visualization(self, sample_time: Optional[float] = None) -> None:
        pose = self._robot_ee_pose()
        if pose is None:
            return
        now = time.monotonic() if sample_time is None else float(sample_time)
        with self._lock:
            task_id = str(self._task_state.get("task_id", "BarClean"))
            if self._task_execution_started is None:
                self._task_execution_started = now
            if now - self._task_last_feature_sample < 1.0 / FEATURE_SAMPLE_HZ:
                return
            scene = json.loads(json.dumps(self._task_scene_geometry))
            elapsed = now - self._task_execution_started
        try:
            values = self._real_feature_values(task_id, pose, scene)
        except (KeyError, TypeError, ValueError):
            return
        definition = self._task_feature_definitions[task_id]
        names = [str(spec["name"]) for spec in definition["schema"]]
        sample = [elapsed, *[values[name] for name in names]]
        if not all(math.isfinite(value) for value in sample):
            return
        with self._lock:
            self._task_current_ee = dict(pose)
            xy = [pose["x"], pose["y"]]
            if not self._task_trace or math.dist(self._task_trace[-1], xy) >= 1e-4:
                self._task_trace.append(xy)
            self._task_feature_series["source"] = "real_tf/{}".format(task_id)
            self._task_feature_series["samples"].append(sample)
            self._task_feature_series["samples"] = self._task_feature_series["samples"][-2400:]
            self._task_last_feature_sample = now

    def _update_plan_visualization(self, payload: Dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        trace = payload.get("trace")
        feature_names = payload.get("feature_names")
        feature_schema = payload.get("feature_schema")
        constraint_specs = payload.get("constraint_specs")
        planning_constraint_specs = payload.get(
            "planning_constraint_specs", constraint_specs
        )
        planning_constraint_source = str(
            payload.get("planning_constraint_source", "true")
        )
        feature_samples = payload.get("feature_samples")
        boundary_indices = payload.get("stage_boundaries")
        boundary_times = payload.get("stage_boundary_times")
        transition_end_times = payload.get("stage_transition_end_times")
        planner_scene = payload.get("scene_geometry")
        with self._lock:
            selected_task = str(self._task_state.get("task_id", "BarClean"))
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
            or not isinstance(planning_constraint_specs, list)
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
        valid_scene = None
        if isinstance(planner_scene, dict):
            bar = planner_scene.get("bar")
            obstacle = planner_scene.get("obstacle")
            if isinstance(bar, dict) and isinstance(obstacle, dict):
                try:
                    pivot = [float(value) for value in bar["pivot"]]
                    axis = [float(value) for value in bar["axis"]]
                    obstacle_radius = float(obstacle["radius"])
                except (KeyError, TypeError, ValueError):
                    pivot, axis, obstacle_radius = [], [], math.nan
                obstacle_type = str(obstacle.get("type"))
                obstacle_geometry = None
                try:
                    if obstacle_type == "circle":
                        center = [float(value) for value in obstacle["center"]]
                        if len(center) == 2 and all(map(math.isfinite, center)):
                            obstacle_geometry = {"center": center}
                    elif obstacle_type == "capsule":
                        endpoints = [
                            [float(value) for value in endpoint]
                            for endpoint in obstacle["endpoints"]
                        ]
                        if len(endpoints) == 2 and all(
                            len(endpoint) == 2
                            and all(map(math.isfinite, endpoint))
                            for endpoint in endpoints
                        ):
                            obstacle_geometry = {"endpoints": endpoints}
                except (KeyError, TypeError, ValueError):
                    obstacle_geometry = None
                if (
                    len(pivot) == 2
                    and len(axis) == 2
                    and obstacle_geometry is not None
                    and math.isfinite(obstacle_radius)
                    and obstacle_radius > 0.0
                    and all(map(math.isfinite, pivot + axis))
                ):
                    axis_norm = math.hypot(axis[0], axis[1])
                    if axis_norm > 1e-12:
                        valid_scene = self._fallback_scene_geometry()
                        valid_scene["bar"].update(
                            pivot=pivot,
                            axis=[axis[0] / axis_norm, axis[1] / axis_norm],
                            lateral_centerline=dict(
                                bar.get("lateral_centerline", {"type": "straight"})
                            ),
                            live=True,
                        )
                        valid_scene["obstacle"].update(
                            type=obstacle_type,
                            **obstacle_geometry,
                            radius=obstacle_radius,
                            live=True,
                        )
                        source_indices = valid_scene["obstacle"]["source_indices"]
                        source_centers = (
                            [obstacle_geometry["center"]]
                            if obstacle_type == "circle"
                            else obstacle_geometry["endpoints"]
                        )
                        if len(source_indices) == len(source_centers):
                            for index, center in zip(source_indices, source_centers):
                                valid_scene["obstacles"][index].update(
                                    center=center, live=True
                                )
        expected_transitions = int(self._task_profiles[task_id]["n_stages"]) - 1
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
            "planning_constraint_specs": json.loads(
                json.dumps(planning_constraint_specs)
            ),
            "planning_constraint_source": planning_constraint_source,
            "samples": valid_samples,
        }
        with self._lock:
            self._task_planned_trace = valid_trace[-TASK_TRACE_POINTS:]
            self._task_planned_feature_series = series
            self._task_stage_boundary_indices = valid_boundary_indices
            self._task_stage_boundary_times = valid_boundaries
            self._task_stage_transition_end_times = valid_transition_ends
            if valid_scene is not None:
                self._task_scene_geometry = valid_scene
                self._task_scene_source = "planner_scene"

    def _update_sim_visualization(self, status: Dict[str, object]) -> None:
        status_task_id = str(status.get("task_id", ""))
        with self._lock:
            selected_task = str(self._task_state.get("task_id", "BarClean"))
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
            if (
                isinstance(scene, dict)
                and scene.get("bar")
                and scene.get("obstacles")
                and scene.get("obstacle")
            ):
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
        self._reload_task_definitions()
        if task_id not in self._task_profiles:
            raise ValueError("Unknown task_id {}".format(task_id))
        with self._lock:
            selected_task = str(self._task_state.get("task_id", "BarClean"))
            review_pending = self._video_review_pending(self._task_state)
        if review_pending:
            raise RuntimeError(
                "Review the completed task video before starting another run"
            )
        if task_id != selected_task:
            raise ValueError("Select {} in the GUI before execution".format(task_id))
        mode = str(payload.get("mode", "simulator"))
        if mode not in ("simulator", "real"):
            raise ValueError("mode must be simulator or real")
        start = self._validated_pose("start", payload.get("start"))
        goal = self._validated_pose("goal", payload.get("goal"))
        record_video = payload.get("record_video", False) is True
        render_video = record_video and mode == "simulator"
        source_id = str(
            payload.get(
                "constraint_source_id",
                self._gui_settings["constraint_source_by_task"][task_id],
            )
        )
        constraint_source = self._resolve_constraint_source(task_id, source_id)
        self.update_gui_settings(
            {
                "task_id": task_id,
                "constraint_source_id": source_id,
                "start": start,
                "goal": goal,
                "record_video": record_video,
            }
        )
        self._set_task_state(
            constraint_source_id=source_id,
            constraint_source_label=constraint_source["label"],
        )

        if mode == "real":
            def real_task() -> None:
                if record_video:
                    self._sync_camera_source(True)
                    self._sync_camera_preview(True)
                else:
                    self._sync_camera_preview(False)
                    self._sync_camera_source(False)
                self._execute_real_task(
                    task_id,
                    start,
                    goal,
                    str(constraint_source["container_path"]),
                    record_video,
                )

            self._start_job(
                "Execute real task",
                real_task,
                reset_task_abort=True,
            )
            return

        scene_snapshot = self._simulation_scene_snapshot()

        def task() -> None:
            # The simulator runs in a separate bridge-network container and does
            # not share the real robot's FRI control path or ROS graph.
            self._reset_task_visualization(task_id)
            self._set_task_state(
                task_id=task_id,
                mode=mode,
                phase="starting",
                data_saved=True,
                video=render_video,
                start=start,
                goal=goal,
                message="Starting or reusing the persistent simulator",
                run_directory=None,
                video_available=False,
                review_pending=False,
                review_status=None,
                final_result_directory=None,
                constraint_source_id=source_id,
                constraint_source_label=constraint_source["label"],
            )
            reused, initial_status = self._ensure_simulator(
                require_video=render_video
            )
            controller = str(initial_status.get("controller", "unknown"))
            if controller in ("planning", "moving_to_start", "executing"):
                raise RuntimeError("The persistent simulator is still executing another task")
            try:
                previous_sequence = int(initial_status["task_sequence"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("Simulator status has no valid task sequence") from error
            self._submit_sim_task(
                task_id,
                start,
                goal,
                render_video,
                scene_snapshot,
                str(constraint_source["container_path"]),
            )
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
                    video_available=bool(
                        run_directory and render_video and phase == "complete"
                    ),
                )
                if phase == "complete":
                    return
                if phase == "failed":
                    raise RuntimeError(str(status.get("message", "Simulator failed to reach task start")))
                time.sleep(0.1)
                status = self._read_sim_status()
            raise RuntimeError("Simulation task exceeded the 300 s timeout")

        self._start_job("Execute simulation task", task, reset_task_abort=True)

    def select_task(self, payload: Dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        self._reload_task_definitions()
        if task_id not in self._task_profiles:
            raise ValueError("Unknown task_id {}".format(task_id))
        with self._lock:
            busy = self._job is not None and self._job.is_alive()
            phase = str(self._task_state.get("phase", "idle"))
            review_pending = self._video_review_pending(self._task_state)
        if review_pending:
            raise RuntimeError("Review the completed task video before switching tasks")
        if busy or phase in TASK_ACTIVE_PHASES:
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
        self._reset_task_visualization(task_id)
        source_id = self._gui_settings["constraint_source_by_task"][task_id]
        source = self._resolve_constraint_source(task_id, source_id)
        with self._lock:
            self._gui_settings["task_id"] = task_id
            self._save_gui_settings()
        self._set_task_state(
            task_id=task_id,
            constraint_source_id=source_id,
            constraint_source_label=source["label"],
            phase="idle",
            message="{} selected".format(self._task_profiles[task_id]["display_name"]),
            run_directory=None,
            video_available=False,
            review_pending=False,
            review_status=None,
            final_result_directory=None,
        )

    def abort_task(self) -> None:
        self._task_abort.set()
        with self._lock:
            mode = str(self._task_state.get("mode", ""))
            phase = str(self._task_state.get("phase", "idle"))
        if phase not in TASK_ACTIVE_PHASES:
            raise RuntimeError("No active task to stop")
        if mode in ("real", "home"):
            if not self._container_running():
                message = "Real task terminated; the robot-control container is already stopped"
            elif "/iiwa14/real_executor" in set(self._ros_nodes()):
                self._abort_real_and_confirm("user pressed Stop Execution")
                message = "Real trajectory stopped; command gate disarmed and task terminated"
            else:
                # The execution thread observes _task_abort before it can plan or arm.
                message = "Real task stopped before the robot executor started"
            self._set_task_state(
                phase="aborted", message=message
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
                deadline = time.monotonic() + 5.0
                while time.monotonic() < deadline:
                    status = self._read_sim_status()
                    if str(status.get("controller", "")) == "aborted":
                        self._update_sim_visualization(status)
                        self._set_task_state(
                            phase="aborted",
                            message="Simulation stopped at the current position; task terminated",
                            run_directory=status.get("run_directory"),
                        )
                        return
                    time.sleep(0.05)
                raise RuntimeError("simulator did not confirm the stopped state")
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                self.log(f"Graceful simulator abort failed: {error}")
                self._run(self._sim_compose("stop"), check=False, timeout=20)
                self._set_task_state(
                    phase="aborted",
                    message="Task aborted; unhealthy simulator was stopped",
                )
                return

    @staticmethod
    def _real_run_host_directory(run_directory: object) -> Path:
        if not isinstance(run_directory, str):
            raise RuntimeError("Real executor did not provide an output directory for video")
        try:
            relative = Path(run_directory).relative_to("/data/demos")
        except ValueError as error:
            raise RuntimeError("Unsafe real-run video output path") from error
        if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
            raise RuntimeError("Unsafe real-run video output path")
        return PROJECT_ROOT / "data" / "demos" / relative

    def _camera_preview_pids(self) -> List[int]:
        process = self._camera_preview_process
        if process is not None and process.poll() is not None:
            self._camera_preview_process = None
        pids = []
        proc_root = Path("/proc")
        try:
            entries = list(proc_root.iterdir())
        except OSError:
            return pids
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                arguments = [
                    value.decode("utf-8", errors="replace")
                    for value in (entry / "cmdline").read_bytes().split(b"\0")
                    if value
                ]
            except OSError:
                continue
            if (
                any(Path(argument).name == "ffplay" for argument in arguments[:3])
                and "/dev/video10" in arguments
                and any(CAMERA_PREVIEW_MARKER in argument for argument in arguments)
            ):
                pids.append(int(entry.name))
        return sorted(pids)

    def _camera_source_pids(self) -> List[int]:
        process = self._camera_source_process
        pids = set()
        if process is not None:
            if process.poll() is None:
                pids.add(process.pid)
            else:
                self._camera_source_process = None
        try:
            entries = list(Path("/proc").iterdir())
        except OSError:
            return sorted(pids)
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                arguments = [
                    value.decode("utf-8", errors="replace")
                    for value in (entry / "cmdline").read_bytes().split(b"\0")
                    if value
                ]
            except OSError:
                continue
            if (
                any(Path(argument).name == "scrcpy" for argument in arguments[:5])
                and "--video-source=camera" in arguments
                and "--v4l2-sink=/dev/video10" in arguments
            ):
                pids.add(int(entry.name))
        return sorted(pids)

    def camera_source_running(self) -> bool:
        return bool(self._camera_source_pids())

    @staticmethod
    def _camera_source_log_detail(log_path: Path) -> str:
        try:
            return log_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip()[-1600:]
        except OSError:
            return ""

    def _wait_for_camera_source(
        self,
        process: Optional[subprocess.Popen[bytes]],
        log_path: Path,
        timeout: float = 12.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        last_error = "waiting for /dev/video10 capture capability"
        while time.monotonic() < deadline:
            if process is not None and process.poll() is not None:
                detail = self._camera_source_log_detail(log_path)
                raise RuntimeError(
                    "Phone camera source exited during startup: "
                    + (detail or "scrcpy could not connect to the wireless Android device")
                )
            try:
                probe = subprocess.run(
                    ["v4l2-ctl", "--device", "/dev/video10", "--all"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=1.0,
                    check=False,
                )
            except FileNotFoundError as error:
                raise RuntimeError(
                    "Automatic phone-camera startup requires v4l2-ctl on the host"
                ) from error
            except subprocess.TimeoutExpired:
                probe = None
            if (
                probe is not None
                and probe.returncode == 0
                and "Video Capture" in probe.stdout
            ):
                try:
                    self._task_video_recorder.preflight()
                    return
                except RuntimeError as error:
                    last_error = str(error)
            elif probe is not None and probe.stderr.strip():
                last_error = probe.stderr.strip()
            time.sleep(0.2)
        detail = self._camera_source_log_detail(log_path)
        raise RuntimeError(
            "Phone camera did not become ready within {:.0f} s: {}{}".format(
                timeout,
                last_error,
                "\n" + detail if detail else "",
            )
        )

    def _sync_camera_source(self, enabled: bool) -> None:
        with self._camera_source_lock:
            pids = self._camera_source_pids()
            log_path = PROJECT_ROOT / "data" / "camera_source_scrcpy.log"
            if enabled:
                if pids:
                    self._wait_for_camera_source(
                        self._camera_source_process, log_path
                    )
                    self.log(
                        "Reusing phone camera source on /dev/video10 (PID {})".format(
                            pids[0]
                        )
                    )
                    return
                if not CAMERA_SOURCE_LAUNCHER.is_file() or not os.access(
                    str(CAMERA_SOURCE_LAUNCHER), os.X_OK
                ):
                    raise RuntimeError(
                        "Phone camera launcher is missing or not executable: "
                        + str(CAMERA_SOURCE_LAUNCHER)
                    )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                command = [str(CAMERA_SOURCE_LAUNCHER), "-e", "--no-window"]
                try:
                    with log_path.open("ab") as log_stream:
                        process = subprocess.Popen(
                            command,
                            stdin=subprocess.DEVNULL,
                            stdout=log_stream,
                            stderr=log_stream,
                            start_new_session=True,
                        )
                except FileNotFoundError as error:
                    raise RuntimeError(
                        "Could not start the phone camera launcher"
                    ) from error
                self._camera_source_process = process
                try:
                    self._wait_for_camera_source(process, log_path)
                except Exception:
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    self._camera_source_process = None
                    raise
                self.log(
                    "Started phone camera source at 1920x1080 / 30 fps on /dev/video10"
                )
                return

            if not pids:
                self.log("Phone camera source is already stopped")
                return
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and self._camera_source_pids():
                time.sleep(0.05)
            for pid in self._camera_source_pids():
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self._camera_source_process = None
            self.log("Stopped phone camera source because video was unchecked")

    def set_camera_source(self, payload: Dict[str, object]) -> None:
        if payload.get("enabled") is not True:
            raise ValueError("Camera source endpoint only accepts enabled=true")
        self._sync_camera_source(True)

    def prepare_demo_video_recording(self, payload: Dict[str, object]) -> None:
        enabled = payload.get("enabled") is True
        if enabled:
            self._sync_camera_source(True)
            self._sync_camera_preview(False)
            self.log("Camera prepared for Demo video recording")
            return
        if self.camera_source_running():
            self._sync_camera_preview(True)
            self.log("Camera preview restored after Demo video recording")

    def camera_preview_running(self) -> bool:
        return bool(self._camera_preview_pids())

    def _sync_camera_preview(self, enabled: bool) -> None:
        pids = self._camera_preview_pids()
        if enabled:
            if pids:
                self.log(
                    "Reusing the existing low-bandwidth camera preview (PID {})".format(
                        pids[0]
                    )
                )
                return
            if not Path("/dev/video10").exists():
                raise RuntimeError(
                    "Cannot open the camera preview because /dev/video10 does not exist"
                )
            log_path = PROJECT_ROOT / "data" / "camera_preview_ffplay.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [
                "ffplay",
                "-hide_banner",
                "-loglevel", "warning",
                "-nostats",
                "-f", "v4l2",
                "-framerate", "30",
                "-video_size", "1920x1080",
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-i", "/dev/video10",
                "-an",
                "-vf", "fps=5,scale=640:-2",
                "-framedrop",
                "-window_title", CAMERA_PREVIEW_MARKER + " (640px · 5 fps)",
            ]
            try:
                with log_path.open("ab") as log_stream:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=log_stream,
                        start_new_session=True,
                    )
            except FileNotFoundError as error:
                raise RuntimeError("Camera preview requires ffplay on the host") from error
            self._camera_preview_process = process
            time.sleep(0.25)
            if process.poll() is not None:
                self._camera_preview_process = None
                detail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
                raise RuntimeError(
                    "Camera preview exited during startup: "
                    + (detail[-1200:] or "unknown ffplay error")
                )
            self.log("Opened low-bandwidth camera preview at 640 px / 5 fps")
            return

        if not pids:
            self.log("Camera preview is already closed")
            return
        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and self._camera_preview_pids():
            time.sleep(0.05)
        remaining = self._camera_preview_pids()
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process = self._camera_preview_process
        if process is not None:
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                pass
            self._camera_preview_process = None
        self.log("Closed the low-bandwidth camera preview because video was unchecked")

    @staticmethod
    def _final_video_run_name(value: object, default_name: str) -> str:
        name = default_name if value is None else str(value).strip()
        if not name:
            raise ValueError("Final run name cannot be empty")
        if len(name) > 80:
            raise ValueError("Final run name cannot exceed 80 characters")
        if (
            name in (".", "..")
            or name.startswith(".")
            or "/" in name
            or "\\" in name
            or any(ord(character) < 32 or ord(character) == 127 for character in name)
        ):
            raise ValueError(
                "Final run name must be one visible directory name without slashes"
            )
        return name

    @staticmethod
    def _archive_final_video_run(
        task_state: Dict[str, object],
        visualization: Dict[str, object],
        final_name: str,
        outcomes: Dict[str, str],
    ) -> Path:
        task_id = str(task_state.get("task_id", ""))
        if not task_id or Path(task_id).name != task_id or task_id in (".", ".."):
            raise RuntimeError("Unsafe task name for the final video archive")
        source_directory = Supervisor._real_run_host_directory(
            task_state.get("run_directory")
        )
        if not source_directory.is_dir():
            raise RuntimeError("The completed real-run directory no longer exists")
        required_names = (
            "execution.mp4",
            "real_task.bag",
            "metadata.json",
            "execution_video_metadata.json",
        )
        source_root = source_directory.resolve()
        for name in required_names:
            source = source_directory / name
            try:
                source.resolve().relative_to(source_root)
            except ValueError as error:
                raise RuntimeError("Unsafe file in the completed real run") from error
            if not source.is_file() or source.stat().st_size == 0:
                raise RuntimeError(
                    "Cannot accept this run: required artifact {} is missing or empty".format(
                        name
                    )
                )

        destination_parent = FINAL_VIDEO_RUN_ROOT / task_id
        destination_parent.mkdir(parents=True, exist_ok=True)
        destination = destination_parent / final_name
        if destination.exists():
            raise RuntimeError("This run already exists in the final video archive")
        visualization_temporary = source_directory / ".visualization.json.tmp"
        result_temporary = source_directory / ".result.json.tmp"
        visualization_temporary.write_text(
            json.dumps(visualization, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        visualization_temporary.replace(source_directory / "visualization.json")
        files = {
            path.name: {"bytes": path.stat().st_size}
            for path in source_directory.iterdir()
            if path.is_file()
            and path.name != "result.json"
            and path != result_temporary
        }
        manifest = {
            "schema_version": 2,
            "status": "accepted_final",
            "accepted_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
            ),
            "task_id": task_id,
            "source_run_directory": str(task_state.get("run_directory")),
            "final_run_name": final_name,
            "outcomes": outcomes,
            "task": task_state,
            "files": files,
        }
        result_temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result_temporary.replace(source_directory / "result.json")
        source_directory.replace(destination)
        return destination

    @staticmethod
    def _validated_task_outcomes(value: object) -> Dict[str, str]:
        if not isinstance(value, dict) or set(value) != set(TASK_OUTCOME_NAMES):
            raise ValueError(
                "Final run outcomes must contain exactly obs_avoid, bar_clean, and table_clean"
            )
        outcomes = {name: str(value[name]).strip().lower() for name in TASK_OUTCOME_NAMES}
        invalid = [name for name, result in outcomes.items() if result not in ("success", "fail")]
        if invalid:
            raise ValueError(
                "Final run outcomes must be success or fail: " + ", ".join(invalid)
            )
        return outcomes

    def review_task_video(self, payload: Dict[str, object]) -> None:
        decision = str(payload.get("decision", "")).strip().lower()
        if decision not in ("accept", "reject"):
            raise ValueError("Video review decision must be accept or reject")
        with self._lock:
            if self._job is not None and self._job.is_alive():
                raise RuntimeError("Wait for the current GUI job to finish")
            task_state = json.loads(json.dumps(self._task_state))
        if task_state.get("review_pending") is not True:
            raise RuntimeError("There is no completed video waiting for review")
        if (
            task_state.get("phase") != "complete"
            or task_state.get("mode") != "real"
            or task_state.get("video") is not True
            or self.task_video_path() is None
        ):
            raise RuntimeError("The pending review has no complete real-task video")
        if decision == "reject":
            video_path = self.task_video_path()
            if video_path is None:
                raise RuntimeError("The rejected task video no longer exists")
            video_path.unlink()
            self._set_task_state(
                review_pending=False,
                review_status="rejected",
                final_result_directory=None,
                video_available=False,
                message=(
                    "Run was not selected as final; its video was permanently deleted"
                ),
            )
            self.log("Rejected task video permanently deleted: " + str(video_path))
            return

        visualization = self.task_visualization()
        run_directory = task_state.get("run_directory")
        source_directory = self._real_run_host_directory(run_directory)
        final_name = self._final_video_run_name(
            payload.get("final_name"), source_directory.name
        )
        outcomes = self._validated_task_outcomes(payload.get("outcomes"))

        def archive() -> None:
            destination = self._archive_final_video_run(
                task_state, visualization, final_name, outcomes
            )
            with self._lock:
                if self._task_state.get("run_directory") != run_directory:
                    raise RuntimeError("The reviewed run changed while it was archived")
                self._task_state.update(
                    review_pending=False,
                    review_status="accepted",
                    final_result_directory=str(destination),
                    message="Run accepted as final and moved to " + str(destination),
                )
            self.log("Final video run moved without duplication: " + str(destination))

        self._start_job("Archive final video run", archive)

    def task_video_path(self) -> Optional[Path]:
        with self._lock:
            run_directory = self._task_state.get("run_directory")
            mode = str(self._task_state.get("mode", ""))
            review_status = str(self._task_state.get("review_status", ""))
            final_result_directory = self._task_state.get("final_result_directory")
        if not isinstance(run_directory, str):
            return None
        if mode == "real":
            if review_status == "accepted" and isinstance(
                final_result_directory, str
            ):
                final_root = FINAL_VIDEO_RUN_ROOT.resolve()
                final_directory = Path(final_result_directory)
                try:
                    final_directory.resolve().relative_to(final_root)
                except ValueError:
                    return None
                path = final_directory / "execution.mp4"
                return path if path.is_file() else None
            try:
                path = self._real_run_host_directory(run_directory) / "execution.mp4"
            except RuntimeError:
                return None
            return path if path.is_file() else None
        try:
            relative = Path(run_directory).relative_to("/data/sim_runs")
        except ValueError:
            return None
        if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
            return None
        path = PROJECT_ROOT / "data" / "sim_runs" / relative / "goal_reaching.mp4"
        return path if path.is_file() else None

    def _start_job(
        self, name: str, target, *, reset_task_abort: bool = False
    ) -> None:
        with self._lock:
            if self._job is not None and self._job.is_alive():
                raise RuntimeError(f"Another task is running: {self._job_name}")
            if reset_task_abort:
                # Clear before the worker becomes visible. Clearing inside the worker
                # can erase a Stop click that arrives immediately after task submission.
                self._task_abort.clear()
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
                        if (
                            name in (
                                "Execute simulation task",
                                "Execute real task",
                                "Return robot Home",
                            )
                            and not self._task_abort.is_set()
                            and self._task_state.get("phase") != "control_status_unknown"
                        ):
                            self._task_state.update(phase="failed", message=str(error))
                else:
                    with self._lock:
                        self._job_ok = True
                        self._job_message = f"{name} completed"

            self._job = threading.Thread(target=runner, daemon=True)
            self._job.start()

    def start_workstation(self) -> None:
        def task() -> None:
            self._ensure_data_directories()
            self._run(["docker", "compose", "up", "-d"])
            self._assert_container_storage_access(
                CONTAINER, ("/data/demos", "/data/real_runs")
            )
            self._wait_for_ros_master()
            if not self._robot_iface_state()["configured"]:
                self.log("Configuring the robot network before starting the station")
                self._run(
                    ["pkexec", str(PROJECT_ROOT / "scripts" / "connect_robot_network.sh")],
                    timeout=60,
                )
                if not self._robot_iface_state()["configured"]:
                    raise RuntimeError("Robot network configuration did not complete")
            if "/stage_demo_gui" not in self._ros_nodes():
                self._signal_child("demo")
            self._start_demo_process()

        self._start_job("Start workstation", task)

    def set_scene_pose_source(self, payload: Dict[str, object]) -> None:
        source = str(payload.get("source", "")).strip().lower()
        if source not in {"fixed", "optitrack"}:
            raise ValueError("Scene pose source must be fixed or optitrack")

        def task() -> None:
            current = self._scene_pose_source()
            if source == current:
                self.log("Scene pose source is already " + source)
                return
            nodes = set(self._ros_nodes())
            if (
                "/iiwa14/iiwa_driver" in nodes
                or self._driver_binary_running()
                or self._task_state.get("phase") in TASK_ACTIVE_PHASES
            ):
                raise RuntimeError(
                    "Exit Demo / release robot control and stop the active task before changing the scene source"
                )

            demo_was_running = "/stage_demo_gui" in nodes
            tracking_was_running = bool(
                nodes.intersection(
                    self._scene_tracking_nodes()
                    | self._inactive_scene_tracking_nodes()
                )
            )
            self._signal_child("demo")
            self._signal_child("tracking")
            if self._container_running():
                self._run(
                    [
                        "docker",
                        "exec",
                        CONTAINER,
                        "bash",
                        "-lc",
                        "pkill -INT -f '[r]oslaunch stage_demo_gui demo_station.launch' || true; "
                        "pkill -INT -f '[r]oslaunch stage_optitrack optitrack.launch' || true",
                    ],
                    check=False,
                    timeout=8,
                )
                scene_nodes = (
                    self._scene_tracking_nodes()
                    | self._inactive_scene_tracking_nodes()
                    | {"/stage_demo_gui"}
                )
                deadline = time.monotonic() + 8.0
                while time.monotonic() < deadline:
                    if not set(self._ros_nodes()).intersection(scene_nodes):
                        break
                    time.sleep(0.25)
                else:
                    raise RuntimeError(
                        "Scene publishers did not stop; stop the workstation before changing the source"
                    )

            with self._lock:
                self._write_env_value("SCENE_POSE_SOURCE", source)
            self.log("Scene pose source changed from {} to {}".format(current, source))
            if demo_was_running:
                self._start_demo_process()
            elif tracking_was_running:
                self._start_optitrack_process()

        self._start_job("Switch scene pose source", task)

    def set_scene(self, payload: Dict[str, object]) -> None:
        scene_name = str(payload.get("scene", "")).strip()
        available = self._available_scene_names()
        if scene_name not in available:
            raise ValueError(
                "Unknown scene {}; choose one of {}".format(
                    scene_name, ", ".join(available)
                )
            )
        source_path = SCENE_CONFIG_DIR / (scene_name + ".json")

        def task() -> None:
            current = self._scene_name()
            if scene_name == current:
                self.log("Scene is already " + scene_name)
                return
            nodes = set(self._ros_nodes())
            if (
                "/iiwa14/iiwa_driver" in nodes
                or self._driver_binary_running()
                or self._task_state.get("phase") in TASK_ACTIVE_PHASES
            ):
                raise RuntimeError(
                    "Exit Demo / release robot control and stop the active task before changing scenes"
                )
            demo_was_running = "/stage_demo_gui" in nodes
            tracking_was_running = bool(
                nodes.intersection(
                    self._scene_tracking_nodes()
                    | self._inactive_scene_tracking_nodes()
                )
            )
            self._signal_child("demo")
            self._signal_child("tracking")
            if self._container_running():
                self._run(
                    [
                        "docker", "exec", CONTAINER, "bash", "-lc",
                        "pkill -INT -f '[r]oslaunch stage_demo_gui demo_station.launch' || true; "
                        "pkill -INT -f '[r]oslaunch stage_optitrack optitrack.launch' || true",
                    ],
                    check=False,
                    timeout=8,
                )
            temporary = SCENE_CONFIG.with_name(SCENE_CONFIG.name + ".tmp")
            temporary.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
            temporary.replace(SCENE_CONFIG)
            fixed_geometry = self._load_fixed_scene_geometry()
            with self._lock:
                self._fixed_scene_geometry = fixed_geometry
                self._task_scene_geometry = self._fallback_scene_geometry()
                self._task_scene_source = "fallback"
            self.log("Scene changed from {} to {}".format(current, scene_name))
            if demo_was_running:
                self._start_demo_process()
            elif tracking_was_running:
                self._start_optitrack_process()

        self._start_job("Switch scene", task)

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
            "/stage_demo_gui",
        } | self._scene_tracking_nodes()
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
        server, base, obj, obstacle_a, obstacle_b = self._optitrack_settings()
        source = self._scene_pose_source()
        tracking_nodes = self._scene_tracking_nodes()
        conflicts = set(self._ros_nodes()).intersection(
            self._inactive_scene_tracking_nodes()
        )
        if conflicts:
            raise RuntimeError(
                "Scene source changed to '{}', but nodes from the other source are still running: {}. "
                "Stop the workstation once, then restart it.".format(
                    source, ", ".join(sorted(conflicts))
                )
            )
        start_optitrack = not tracking_nodes.issubset(set(self._ros_nodes()))
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
                "start_optitrack:={}".format(
                    "true" if start_optitrack else "false"
                ),
                "use_fixed_scene:={}".format("true" if source == "fixed" else "false"),
                f"optitrack_server:={server}",
                f"optitrack_base:={base}",
                f"optitrack_object:={obj}",
                f"optitrack_obstacle_a:={obstacle_a}",
                f"optitrack_obstacle_b:={obstacle_b}",
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

    def prepare_demo_control(self) -> None:
        def task() -> None:
            if not self._container_running():
                raise RuntimeError("Start the workstation container first")
            if not self._robot_iface_state()["configured"]:
                raise RuntimeError("Robot network interface is not configured")
            if "/iiwa14/iiwa_driver" in self._ros_nodes():
                controller_snapshot = self._running_iiwa_controllers()
                running_controllers = set(controller_snapshot or [])
                if "SafeTorqueController" in running_controllers:
                    self._start_demo_process()
                    self.log("Demo / Torque driver is already running")
                    return
                self.log(
                    "Switching robot control through Idle to SafeTorqueController"
                )
                self._release_robot_control("switching to Demo / Torque control")
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
            # The Demo station is a separate ROS launch from the torque driver.
            # A container rebuild/recreate terminates it while the host GUI
            # remains alive, so every Demo transition must restore and verify
            # the recorder/UI chain instead of assuming it is persistent.
            self._start_demo_process()
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

        self._start_job("Prepare Demo / Torque control", task)

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
        with self._lock:
            self._real_station_verified = False
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
            if self._wait_for_driver_stopped(8.0):
                return
            self.log("Driver ignored SIGINT; escalating shutdown to SIGTERM")
            self._signal_driver_processes("TERM")
            if self._wait_for_driver_stopped(3.0):
                return
            self.log("Driver ignored SIGTERM; forcing final process cleanup")
            self._signal_driver_processes("KILL")
            if self._wait_for_driver_stopped(2.0):
                return
            raise RuntimeError(
                "iiwa_driver or its roslaunch wrapper did not stop; "
                "use the SmartPAD/E-stop if motion persists"
            )

    def _signal_driver_processes(self, signal_name: str) -> None:
        self._run(
            [
                "docker",
                "exec",
                CONTAINER,
                "bash",
                "-lc",
                "pkill -{signal_name} -f '[r]oslaunch stage_real_executor real_station.launch' || true; "
                "pkill -{signal_name} -f '[r]oslaunch iiwa_driver .*iiwa14_bringup.launch' || true; "
                "pkill -{signal_name} -f '[/]iiwa_driver/iiwa_driver' || true; "
                "pkill -{signal_name} -f '[/]stage_real_executor/real_executor.py' || true".format(
                    signal_name=signal_name
                ),
            ],
            check=False,
            timeout=5,
        )

    def _wait_for_driver_stopped(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        stopped_since: Optional[float] = None
        while time.monotonic() < deadline:
            driver_node_running = "/iiwa14/iiwa_driver" in self._ros_nodes()
            launch_running = CONTAINER in self._driver_process_containers()
            if not driver_node_running and not launch_running:
                if stopped_since is None:
                    stopped_since = time.monotonic()
                elif time.monotonic() - stopped_since >= 0.5:
                    return True
            else:
                stopped_since = None
            time.sleep(0.5)
        return False

    def release_robot_control(self) -> None:
        def task() -> None:
            self._release_robot_control("returning to Idle")

        self._start_job("Release robot control", task)

    def stop_all(self) -> None:
        def task() -> None:
            self._task_abort.set()
            self._run(self._sim_compose("stop"), check=False, timeout=20)
            self._release_robot_control("stopping the workstation")
            self._signal_child("demo")
            self._signal_child("tracking")
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
            elif self.path == "/api/control/demo":
                SUPERVISOR.prepare_demo_control()
            elif self.path == "/api/control/idle":
                SUPERVISOR.release_robot_control()
            elif self.path == "/api/control/home":
                SUPERVISOR.return_robot_home()
            elif self.path == "/api/stop-all":
                SUPERVISOR.stop_all()
            elif self.path == "/api/task/execute":
                SUPERVISOR.execute_task(payload)
            elif self.path == "/api/task/video-review":
                SUPERVISOR.review_task_video(payload)
            elif self.path == "/api/task/select":
                SUPERVISOR.select_task(payload)
            elif self.path == "/api/settings":
                SUPERVISOR.update_gui_settings(payload)
            elif self.path == "/api/camera/source":
                SUPERVISOR.set_camera_source(payload)
            elif self.path == "/api/camera/demo-recording":
                SUPERVISOR.prepare_demo_video_recording(payload)
            elif self.path == "/api/scene/source":
                SUPERVISOR.set_scene_pose_source(payload)
            elif self.path == "/api/scene/select":
                SUPERVISOR.set_scene(payload)
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
    shutdown_started = threading.Event()

    def request_shutdown(_signum: int, _frame: object) -> None:
        if shutdown_started.is_set():
            return
        shutdown_started.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, request_shutdown)
    signal.signal(signal.SIGTERM, request_shutdown)
    SUPERVISOR.log(f"Host supervisor available at http://{HOST}:{PORT}")
    SUPERVISOR.cleanup_stale_topic_streams()
    SUPERVISOR.start_demo_if_available()
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        SUPERVISOR.shutdown()


if __name__ == "__main__":
    main()
