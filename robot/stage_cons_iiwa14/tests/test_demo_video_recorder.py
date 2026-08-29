import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ros_ws/src/stage_demo_recorder/scripts/demo_recorder.py"
)


def load_recorder_module():
    rospy = types.ModuleType("rospy")
    sys.modules.update(
        {
            "rospy": rospy,
            "std_msgs": types.ModuleType("std_msgs"),
            "std_msgs.msg": types.SimpleNamespace(String=object),
            "std_srvs": types.ModuleType("std_srvs"),
            "std_srvs.srv": types.SimpleNamespace(
                Trigger=object,
                TriggerResponse=object,
            ),
        }
    )
    spec = importlib.util.spec_from_file_location("demo_recorder_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FinishedProcess:
    pid = 1234
    returncode = 0

    @staticmethod
    def poll():
        return 0


class DemoVideoRecorderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_recorder_module()

    def test_interrupted_demo_keeps_the_recorded_video(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Path(directory)
            partial = session / "demo.partial.mp4"
            final = session / "demo.mp4"
            partial.write_bytes(b"video")
            subject = self.module.DemoRecorder.__new__(self.module.DemoRecorder)
            subject._process = FinishedProcess()
            subject._session_dir = session
            subject._metadata = {}
            subject._stopping = False
            subject._video_requested = True
            subject._video_process = FinishedProcess()
            subject._video_log_stream = None
            subject._video_partial_path = partial
            subject._video_final_path = final
            subject._video_error = None
            subject._publish_status = mock.Mock()

            with mock.patch.object(
                self.module.DemoRecorder,
                "_video_packet_count",
                return_value=12,
            ):
                subject._finish_process("interrupted")

            self.assertTrue(final.is_file())
            self.assertFalse(partial.exists())
            metadata = json.loads((session / "metadata.json").read_text())
            self.assertEqual(metadata["recording_state"], "interrupted")
            self.assertEqual(metadata["video_file"], "demo.mp4")
            self.assertIsNone(metadata["video_error"])
            subject._publish_status.assert_called_once_with(
                "interrupted", session_id=session.name
            )


if __name__ == "__main__":
    unittest.main()
