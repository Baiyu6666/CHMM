import importlib.util
import sys
import threading
import time
import types
import unittest
from collections import deque
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ros_ws/src/stage_demo_gui/scripts/demo_gui.py"
)


def _message_module(name, **classes):
    module = types.ModuleType(name)
    for class_name, value in classes.items():
        setattr(module, class_name, value)
    return module


def load_demo_gui_module():
    class DummyMessage:
        pass

    rospy = types.ModuleType("rospy")
    rospy.ServiceException = type("ServiceException", (Exception,), {})
    rospy.ROSException = type("ROSException", (Exception,), {})
    rospy.params = {}
    rospy.set_param = lambda name, value: rospy.params.__setitem__(name, value)
    sys.modules.update(
        {
            "rospy": rospy,
            "tf2_ros": types.ModuleType("tf2_ros"),
            "geometry_msgs": types.ModuleType("geometry_msgs"),
            "geometry_msgs.msg": _message_module(
                "geometry_msgs.msg", PoseStamped=DummyMessage
            ),
            "sensor_msgs": types.ModuleType("sensor_msgs"),
            "sensor_msgs.msg": _message_module(
                "sensor_msgs.msg", JointState=DummyMessage
            ),
            "std_msgs": types.ModuleType("std_msgs"),
            "std_msgs.msg": _message_module(
                "std_msgs.msg",
                Bool=DummyMessage,
                Empty=DummyMessage,
                Float64MultiArray=DummyMessage,
                Int32=DummyMessage,
                String=DummyMessage,
            ),
            "std_srvs": types.ModuleType("std_srvs"),
            "std_srvs.srv": _message_module(
                "std_srvs.srv", SetBool=DummyMessage, Trigger=DummyMessage
            ),
        }
    )
    spec = importlib.util.spec_from_file_location("stage_demo_gui_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class DemoGuiStateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_demo_gui_module()

    def setUp(self):
        subject = self.module.DemoGui.__new__(self.module.DemoGui)
        now = time.monotonic()
        subject._lock = threading.RLock()
        subject._service_lock = threading.RLock()
        subject._last_joint = now
        subject._last_ee = now
        subject._last_bar = 0.0
        subject._last_obstacle = 0.0
        subject._last_fixture = now
        subject._last_recorder = now
        subject._last_commanding = now
        subject._last_command_mode = now
        subject._last_motion_gate = now
        subject._fri_commanding = True
        subject._fri_command_mode = 3
        subject._driver_demo_active = False
        subject._mode_requested = False
        subject._mode_loss_cleanup_running = False
        subject._trace = deque()
        subject._last_trace_sample = 1.0
        subject._all_service = object()
        subject._demo_mode_service = object()
        subject._record_start_service = lambda: types.SimpleNamespace(
            success=True, message="recording"
        )
        subject._call_set_bool = lambda _proxy, _enabled: (True, "ok")
        subject._task_id = "BarClean"
        subject._recorder_state = "idle"
        subject._recorder_session = None
        subject._recorder_video_requested = False
        subject._recorder_video_state = "idle"
        subject._recorder_video_file = None
        subject._recorder_video_error = None
        subject._message = ""
        self.subject = subject

    def test_torque_can_arm_without_tracking_or_an_open_motion_gate(self):
        dependencies = self.subject._dependencies()

        self.assertFalse(dependencies["optitrack_bar"])
        self.assertFalse(dependencies["optitrack_obstacle"])
        self.assertFalse(dependencies["motion_gate"])
        self.assertTrue(all(self.subject._mode_dependencies().values()))

        ok, _message = self.subject.set_mode(True)

        self.assertTrue(ok)
        self.assertTrue(self.subject._mode_requested)
        self.assertTrue(self.subject._driver_demo_active)

    def test_motion_gate_lamp_requires_the_gate_value_not_only_heartbeat(self):
        self.assertFalse(self.subject._dependencies()["motion_gate"])

        self.subject._driver_demo_active = True

        self.assertTrue(self.subject._dependencies()["motion_gate"])

    def test_gate_loss_invalidates_the_previous_demo_request(self):
        self.subject._driver_demo_active = True
        self.subject._mode_requested = True
        self.subject._mode_loss_cleanup_running = True

        self.subject._on_motion_gate(types.SimpleNamespace(data=False))

        self.assertFalse(self.subject._driver_demo_active)
        self.assertFalse(self.subject._mode_requested)

    def test_start_recording_arms_the_motion_gate_automatically(self):
        ok, message = self.subject.set_recording(
            True, {"label": "demo_00", "notes": "", "record_video": True}
        )

        self.assertTrue(ok)
        self.assertEqual(message, "recording")
        self.assertTrue(self.subject._driver_demo_active)
        self.assertTrue(self.subject._mode_requested)
        self.assertTrue(self.module.rospy.params["/demo_recorder/record_video"])

    def test_recorder_status_exposes_saved_video(self):
        self.subject._on_recorder_status(
            types.SimpleNamespace(
                data=(
                    '{"state":"completed","session_id":"demo_00",'
                    '"video_requested":true,"video_state":"idle",'
                    '"video_file":"demo.mp4","video_error":null}'
                )
            )
        )

        self.assertTrue(self.subject._recorder_video_requested)
        self.assertEqual(self.subject._recorder_video_file, "demo.mp4")
        self.assertIsNone(self.subject._recorder_video_error)


if __name__ == "__main__":
    unittest.main()
