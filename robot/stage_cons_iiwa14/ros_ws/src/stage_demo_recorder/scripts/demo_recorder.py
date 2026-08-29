#!/usr/bin/env python3
"""Start and stop validated per-demonstration ROS1 bag recordings."""

import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import threading
from pathlib import Path

import rospy
from std_msgs.msg import String
from std_srvs.srv import Trigger, TriggerResponse


class DemoRecorder:
    def __init__(self):
        self._lock = threading.RLock()
        self._process = None
        self._session_dir = None
        self._metadata = None
        self._stopping = False
        self._video_process = None
        self._video_log_stream = None
        self._video_partial_path = None
        self._video_final_path = None
        self._video_requested = False
        self._video_error = None

        self._output_root = Path(rospy.get_param("~output_root", "/data/demos"))
        self._required_topics = self._topic_list("~required_topics")
        self._optional_topics = self._topic_list("~optional_topics")
        self._marker_topic = rospy.get_param("~marker_topic", "/stage_cons/demo_marker")
        status_topic = rospy.get_param(
            "~status_topic", "/stage_cons/demo_recorder/status"
        )
        self._minimum_free_bytes = int(
            float(rospy.get_param("~minimum_free_gb", 2.0)) * 1024**3
        )

        self._status_publisher = rospy.Publisher(
            status_topic, String, queue_size=1, latch=True
        )
        self._marker_publisher = rospy.Publisher(
            self._marker_topic, String, queue_size=20
        )
        self._start_service = rospy.Service("~start", Trigger, self._start)
        self._stop_service = rospy.Service("~stop", Trigger, self._stop)
        self._timer = rospy.Timer(rospy.Duration(0.5), self._poll_process)
        rospy.on_shutdown(self._shutdown)
        self._publish_status("idle")

    @staticmethod
    def _topic_list(parameter):
        values = rospy.get_param(parameter, [])
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError("{} must be a list of topic names".format(parameter))
        return list(dict.fromkeys(values))

    @staticmethod
    def _safe_label(value):
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
        return cleaned or "demo"

    @staticmethod
    def _utc_now():
        return datetime.datetime.now(datetime.timezone.utc)

    @staticmethod
    def _write_json(path, content):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(content, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(str(temporary), str(path))

    def _published_topics(self):
        return dict(rospy.get_published_topics())

    def _publish_status(self, state, **fields):
        message = {
            "state": state,
            "stamp": self._utc_now().isoformat(),
            "video_requested": self._video_requested,
            "video_state": (
                "recording"
                if self._video_process is not None
                and self._video_process.poll() is None
                else "failed" if self._video_error else "idle"
            ),
            "video_file": (
                self._video_final_path.name
                if self._video_final_path is not None
                and self._video_final_path.is_file()
                else None
            ),
            "video_error": self._video_error,
        }
        message.update(fields)
        self._status_publisher.publish(String(data=json.dumps(message, sort_keys=True)))

    def _start_video(self):
        device = str(rospy.get_param("~video_device", "/dev/video10"))
        width = int(rospy.get_param("~video_width", 1920))
        height = int(rospy.get_param("~video_height", 1080))
        fps = float(rospy.get_param("~video_fps", 30.0))
        if not Path(device).exists():
            raise RuntimeError(
                "Video recording requested, but {} does not exist".format(device)
            )
        self._video_partial_path = self._session_dir / "demo.partial.mp4"
        self._video_final_path = self._session_dir / "demo.mp4"
        log_path = self._session_dir / "demo_video_ffmpeg.log"
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-nostdin",
            "-thread_queue_size", "1024",
            "-f", "v4l2", "-framerate", str(int(round(fps))),
            "-video_size", "{}x{}".format(width, height),
            "-i", device,
            "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "15",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(self._video_partial_path),
        ]
        self._video_log_stream = log_path.open("wb")
        try:
            self._video_process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self._video_log_stream,
                start_new_session=True,
            )
        except Exception:
            self._video_log_stream.close()
            self._video_log_stream = None
            raise
        rospy.sleep(0.15)
        if self._video_process.poll() is not None:
            self._finish_video(save=False)
            try:
                detail = log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
            except OSError:
                detail = ""
            raise RuntimeError(
                "FFmpeg video recorder exited during startup: "
                + (detail or "unknown video error")
            )

    @staticmethod
    def _video_packet_count(path):
        if path is None or not path.is_file() or path.stat().st_size == 0:
            return 0
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-count_packets", "-show_entries", "stream=nb_read_packets",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5.0,
                check=False,
            )
            return int(result.stdout.strip()) if result.returncode == 0 else 0
        except (OSError, ValueError, subprocess.TimeoutExpired):
            return 0

    def _finish_video(self, save):
        process = self._video_process
        partial_path = self._video_partial_path
        final_path = self._video_final_path
        self._video_process = None
        if process is not None and process.poll() is None:
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
                process.wait(timeout=5.0)
        if self._video_log_stream is not None:
            self._video_log_stream.close()
            self._video_log_stream = None
        packet_count = self._video_packet_count(partial_path)
        if save and packet_count > 0 and partial_path is not None and final_path is not None:
            partial_path.replace(final_path)
            self._video_error = None
            return final_path
        if partial_path is not None:
            try:
                partial_path.unlink()
            except FileNotFoundError:
                pass
        if save:
            self._video_error = "recording contains no complete video packets"
        return None

    def _start(self, _request):
        with self._lock:
            if self._process is not None:
                return TriggerResponse(False, "A demonstration is already being recorded")

            published = self._published_topics()
            missing = [topic for topic in self._required_topics if topic not in published]
            if missing:
                message = "Required topics are missing: {}".format(", ".join(missing))
                self._publish_status("blocked", reason=message)
                return TriggerResponse(False, message)

            try:
                self._output_root.mkdir(parents=True, exist_ok=True)
                free_bytes = shutil.disk_usage(str(self._output_root)).free
            except OSError as error:
                message = "Output directory is unavailable: {}".format(error)
                self._publish_status("blocked", reason=message)
                return TriggerResponse(False, message)
            if free_bytes < self._minimum_free_bytes:
                message = "Only {:.2f} GiB free under {}".format(
                    free_bytes / 1024**3, self._output_root
                )
                self._publish_status("blocked", reason=message)
                return TriggerResponse(False, message)

            now = self._utc_now()
            task_id = self._safe_label(rospy.get_param("~task_id", "BarInspect"))
            label = self._safe_label(rospy.get_param("~label", "demo"))
            session_name = "{}_{}".format(now.strftime("%Y%m%dT%H%M%S_%fZ"), label)
            self._session_dir = self._output_root / task_id / session_name
            self._video_requested = rospy.get_param("~record_video", False) is True
            self._video_error = None
            self._video_process = None
            self._video_partial_path = None
            self._video_final_path = None
            try:
                self._session_dir.mkdir(parents=True, exist_ok=False)
            except OSError as error:
                message = "Could not create session directory: {}".format(error)
                self._publish_status("blocked", reason=message)
                return TriggerResponse(False, message)
            bag_path = self._session_dir / "demo.bag"
            all_topics = list(
                dict.fromkeys(
                    self._required_topics + self._optional_topics + [self._marker_topic]
                )
            )
            self._metadata = {
                "schema_version": 1,
                "session_id": session_name,
                "experiment": rospy.get_param("~experiment", "stage_constraint"),
                "task_id": task_id,
                "label": label,
                "operator_notes": rospy.get_param("~operator_notes", ""),
                "started_at_utc": now.isoformat(),
                "finished_at_utc": None,
                "recording_state": "starting",
                "bag_file": bag_path.name,
                "video_requested": self._video_requested,
                "video_file": "demo.mp4" if self._video_requested else None,
                "required_topics": self._required_topics,
                "optional_topics": self._optional_topics,
                "recorded_topics": all_topics,
                "topic_types_at_start": {
                    topic: published.get(topic) for topic in all_topics
                },
                "robot": {"name": rospy.get_param("~robot_name", "iiwa14")},
                "scene_pose": {
                    "source": rospy.get_param("~scene_pose_source", "fixed"),
                    "base_name": rospy.get_param("~optitrack_base", "iiwa14"),
                    "bar_name": rospy.get_param("~optitrack_object", "baiyu_bar"),
                    "obstacle_name": rospy.get_param(
                        "~optitrack_obstacle", "baiyu_obs_bar"
                    ),
                },
            }
            self._write_json(self._session_dir / "metadata.json", self._metadata)

            if self._video_requested:
                try:
                    self._start_video()
                except Exception as error:
                    self._video_error = str(error)
                    self._metadata["recording_state"] = "start_failed"
                    self._metadata["video_error"] = self._video_error
                    self._metadata["finished_at_utc"] = self._utc_now().isoformat()
                    self._write_json(
                        self._session_dir / "metadata.json", self._metadata
                    )
                    self._publish_status("blocked", reason=self._video_error)
                    return TriggerResponse(
                        False, "Could not start demo video: {}".format(error)
                    )

            command = [
                "rosbag",
                "record",
                "--lz4",
                "--output-name",
                str(bag_path),
            ] + all_topics
            try:
                self._process = subprocess.Popen(command, start_new_session=True)
            except Exception as error:
                if self._video_requested:
                    self._finish_video(save=False)
                self._metadata["recording_state"] = "start_failed"
                self._metadata["error"] = str(error)
                self._metadata["finished_at_utc"] = self._utc_now().isoformat()
                self._write_json(self._session_dir / "metadata.json", self._metadata)
                self._process = None
                return TriggerResponse(False, "Could not start rosbag: {}".format(error))

            self._metadata["recording_state"] = "recording"
            self._metadata["rosbag_pid"] = self._process.pid
            self._write_json(self._session_dir / "metadata.json", self._metadata)
            self._publish_status("recording", session_id=session_name)
            rospy.sleep(0.25)
            self._marker_publisher.publish(String(data="recording_started"))
            return TriggerResponse(True, str(self._session_dir))

    def _stop(self, _request):
        with self._lock:
            if self._process is None:
                return TriggerResponse(False, "No demonstration is being recorded")
            session_dir = str(self._session_dir)
            self._finish_process("completed")
            return TriggerResponse(True, session_dir)

    def _finish_process(self, final_state):
        process = self._process
        if process is None:
            return
        self._stopping = True
        video_path = self._finish_video(save=self._video_requested)
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5.0)
        return_code = process.returncode
        if self._metadata is not None:
            self._metadata["recording_state"] = final_state
            self._metadata["finished_at_utc"] = self._utc_now().isoformat()
            self._metadata["rosbag_return_code"] = return_code
            self._metadata["video_file"] = (
                video_path.name if video_path is not None else None
            )
            self._metadata["video_error"] = self._video_error
            self._write_json(self._session_dir / "metadata.json", self._metadata)
        self._process = None
        self._publish_status(final_state, session_id=self._session_dir.name)
        self._stopping = False

    def _poll_process(self, _event):
        with self._lock:
            if self._process is None or self._stopping:
                return
            if (
                self._video_process is not None
                and self._video_process.poll() is not None
                and self._video_error is None
            ):
                self._video_error = "FFmpeg exited before the demonstration stopped"
                rospy.logerr(self._video_error)
            return_code = self._process.poll()
            if return_code is not None:
                rospy.logerr("rosbag exited unexpectedly with code %s", return_code)
                self._finish_process("failed")

    def _shutdown(self):
        with self._lock:
            if self._process is not None:
                self._finish_process("interrupted")


if __name__ == "__main__":
    rospy.init_node("demo_recorder")
    DemoRecorder()
    rospy.spin()
