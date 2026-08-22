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
        message = {"state": state, "stamp": self._utc_now().isoformat()}
        message.update(fields)
        self._status_publisher.publish(String(data=json.dumps(message, sort_keys=True)))

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
                "required_topics": self._required_topics,
                "optional_topics": self._optional_topics,
                "recorded_topics": all_topics,
                "topic_types_at_start": {
                    topic: published.get(topic) for topic in all_topics
                },
                "robot": {"name": rospy.get_param("~robot_name", "iiwa14")},
                "optitrack": {
                    "server": rospy.get_param("~optitrack_server", ""),
                    "base_rigid_body": rospy.get_param("~optitrack_base", "iiwa14"),
                    "object_rigid_body": rospy.get_param(
                        "~optitrack_object", "baiyu_bar"
                    ),
                    "obstacle_rigid_body": rospy.get_param(
                        "~optitrack_obstacle", "baiyu_obs_ball"
                    ),
                },
            }
            self._write_json(self._session_dir / "metadata.json", self._metadata)

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
            self._write_json(self._session_dir / "metadata.json", self._metadata)
        self._process = None
        self._publish_status(final_state, session_id=self._session_dir.name)
        self._stopping = False

    def _poll_process(self, _event):
        with self._lock:
            if self._process is None or self._stopping:
                return
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
