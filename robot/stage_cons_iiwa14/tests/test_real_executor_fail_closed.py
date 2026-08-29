import ast
import json
import math
import threading
import time
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ros_ws/src/stage_real_executor/scripts/real_executor.py"
)
DRIVER_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ros_ws/src/iiwa_driver/src/iiwa.cpp"
)
METHODS = {
    "_parse_orientation_constraints",
    "_accept_path",
    "_orientation_constraints_callback",
    "_path_callback",
    "_controller_state_callback",
    "_controller_command_error",
    "_publish_position_heartbeat",
    "_publish_fri_ready_status",
    "_commanding_callback",
    "_mode_callback",
    "_torque_callback",
    "_release_final_hold",
    "_latch_execution_failure",
    "_fri_failure_reason",
    "_refresh_prepared_start",
    "_synchronize_and_arm",
    "_run_segment",
    "_wait_for_task_start",
    "_execution_worker",
    "_disarm_position_commands",
    "_abort_execution",
    "_load_home_obstacle",
}


def load_fail_closed_subject():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    real_executor = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RealExecutor"
    )
    selected = [
        node
        for node in real_executor.body
        if isinstance(node, ast.FunctionDef) and node.name in METHODS
    ]
    if {node.name for node in selected} != METHODS:
        raise AssertionError("Could not find every fail-closed method in RealExecutor")
    subject_class = ast.ClassDef(
        name="FailClosedSubject",
        bases=[],
        keywords=[],
        body=selected,
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[subject_class], type_ignores=[]))
    namespace = {
        "Empty": types.SimpleNamespace,
        "Bool": types.SimpleNamespace,
        "GoalStatus": types.SimpleNamespace(SUCCEEDED=3),
        "TriggerResponse": lambda success, message: types.SimpleNamespace(
            success=success, message=message
        ),
        "ValidationError": RuntimeError,
        "math": math,
        "json": json,
        "np": np,
        "time": time,
        "rospy": types.SimpleNamespace(
            Duration=lambda seconds: seconds,
            ROSException=RuntimeError,
            ServiceException=RuntimeError,
            logerr=lambda *_args: None,
            loginfo=lambda *_args: None,
        ),
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["FailClosedSubject"]


class FakeActionClient:
    def __init__(self):
        self.cancel_count = 0
        self.cancel_goal_count = 0
        self.goals = []

    def cancel_all_goals(self):
        self.cancel_count += 1

    def cancel_goal(self):
        self.cancel_goal_count += 1

    def send_goal(self, goal):
        self.goals.append(goal)

    @staticmethod
    def wait_for_result(_timeout):
        return True

    @staticmethod
    def get_state():
        return 3


class FakePositionGate:
    def __init__(self):
        self.calls = []

    @staticmethod
    def wait_for_service(timeout):
        del timeout

    def __call__(self, enabled):
        self.calls.append(enabled)
        return types.SimpleNamespace(success=True, message="ok")


class RealExecutorFailClosedTest(unittest.TestCase):
    def setUp(self):
        subject_class = load_fail_closed_subject()
        self.subject = subject_class()
        self.subject._lock = threading.RLock()
        self.subject._position_mode = 1
        self.subject._fri_status_timeout = 0.5
        self.subject._commanding = True
        self.subject._commanding_received = time.monotonic()
        self.subject._fri_mode = 1
        self.subject._fri_mode_received = time.monotonic()
        self.subject._execution_active = True
        self.subject._execution_failure_reason = None
        self.subject._holding_final_position = False
        self.subject._position_armed = True
        self.subject._joint_names = ["iiwa14_joint_{}".format(i) for i in range(1, 8)]
        self.subject._joint_state_timeout = 0.5
        self.subject._start_drift_limit = math.radians(0.5)
        self.subject._task_start_settle_tolerance = math.radians(2.0)
        self.subject._task_start_settle_timeout = 2.0
        self.subject._task_start_settle_samples = 1
        self.subject._controller_desired_position = None
        self.subject._controller_state_received = 0.0
        self.subject._protective_stop = False
        self.subject._record_requested = False
        self.subject._worker = object()
        self.subject._prepared = object()
        self.subject._task_id = "BarClean"
        self.subject._path = None
        self.subject._path_tool_yaw_active = None
        self.subject._path_approach_obstacle = None
        self.subject._path_stage_timing = None
        self.subject._pending_path = None
        self.subject._orientation_constraints = {}
        self.subject._torque_thresholds = np.ones(7)
        self.subject._torque_timeout = 0.25
        self.subject._torque_received = time.monotonic()
        self.subject._torque_trip_count = 0
        self.subject._abort = threading.Event()
        self.subject._client = FakeActionClient()
        self.subject.disarm_count = 0

        def disarm():
            self.subject.disarm_count += 1
            self.subject._position_armed = False
            self.subject._holding_final_position = False
            return True, "fixed hold confirmed"

        self.subject._disarm_position_commands = disarm

    def test_home_encloses_scenec_capsule_in_one_approach_circle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_path = root / "scene.json"
            task_path = root / "task.json"
            scene_path.write_text(
                json.dumps(
                    {
                        "obstacles": [
                            {
                                "name": "first",
                                "locked_pose_robot": [0.6, 0.1, 0.13, 0, 0, 0, 1],
                                "radius": 0.025,
                            },
                            {
                                "name": "second",
                                "locked_pose_robot": [0.5, 0.2, 0.13, 0, 0, 0, 1],
                                "radius": 0.025,
                            },
                        ],
                        "planning_obstacle": {
                            "type": "capsule",
                            "endpoint_obstacles": ["first", "second"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            task_path.write_text(
                json.dumps(
                    {
                        "table_normal": [0.0, 0.0, 1.0],
                        "constraint_terms": [
                            {
                                "feature_name": "obstacle_clearance",
                                "stage": 0,
                                "semantics": "lower_bound",
                                "value": 0.082,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            self.subject._scene_config_path = str(scene_path)
            self.subject._task_config_paths = {"BarClean": str(task_path)}

            obstacle = self.subject._load_home_obstacle()

        self.assertEqual(obstacle["type"], "circle")
        np.testing.assert_allclose(obstacle["center"], [0.55, 0.15, 0.13])
        self.assertAlmostEqual(
            obstacle["radius"], 0.025 + 0.5 * math.sqrt(0.02)
        )
        self.assertEqual(obstacle["clearance"], 0.082)

    def test_driver_closed_gate_uses_a_fixed_hold_not_moving_measurements(self):
        source = DRIVER_SOURCE.read_text(encoding="utf-8")

        self.assertIn("_position_hold_command[i] = _joint_position[i];", source)
        self.assertIn(
            "else if (_position_hold_valid.load())\n"
            "                _robot_command.setJointPosition(_position_hold_command.data());",
            source,
        )

    def test_user_abort_confirms_fixed_hold_before_cancelling_action(self):
        events = []
        self.subject._disarm_position_commands = lambda: (
            events.append("fixed_hold") or (True, "confirmed")
        )
        self.subject._client.cancel_all_goals = lambda: events.append("cancel")
        self.subject._publish = lambda phase, **_fields: events.append(phase)

        response = self.subject._abort_execution(None)

        self.assertTrue(response.success)
        self.assertEqual(events, ["fixed_hold", "cancel", "aborted"])
        self.assertTrue(self.subject._abort.is_set())

    def test_user_abort_fails_if_driver_cannot_confirm_fixed_hold(self):
        self.subject._disarm_position_commands = lambda: (False, "timeout")
        self.subject._publish = mock.Mock()

        response = self.subject._abort_execution(None)

        self.assertFalse(response.success)
        self.assertIn("did not confirm", response.message)
        self.subject._publish.assert_called_once()
        self.assertEqual(
            self.subject._publish.call_args.args[0], "control_status_unknown"
        )

    def test_real_task_bag_records_fri_diagnostics(self):
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
        real_executor = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RealExecutor"
        )
        start_recording = next(
            node
            for node in real_executor.body
            if isinstance(node, ast.FunctionDef) and node.name == "_start_recording"
        )
        topics_assignment = next(
            node
            for node in start_recording.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "topics"
                for target in node.targets
            )
        )

        topics = ast.literal_eval(topics_assignment.value)

        self.assertIn("/iiwa14/fri_diagnostics", topics)

    def test_next_execution_reuses_active_final_hold_without_gate_gap(self):
        self.subject._holding_final_position = True
        self.subject._position_gate = FakePositionGate()
        self.subject._time_parameterize = lambda path, duration: {
            "position": path,
            "velocity": np.zeros_like(path),
            "acceleration": np.zeros_like(path),
            "time": np.asarray([0.5, 1.5]),
            "duration": 1.5,
        }
        self.subject._goal = lambda segment, start_delay=0.3: (
            segment, start_delay
        )

        self.subject._synchronize_and_arm(np.zeros(7))

        self.assertEqual(self.subject._position_gate.calls, [True])
        self.assertTrue(self.subject._position_armed)
        self.assertFalse(self.subject._holding_final_position)
        self.assertEqual(len(self.subject._client.goals), 1)

    def test_new_execution_arms_only_after_controller_command_is_synchronized(self):
        self.subject._holding_final_position = False
        self.subject._position_armed = False
        self.subject._position_gate = FakePositionGate()
        self.subject._time_parameterize = lambda path, duration: {
            "position": path,
            "velocity": np.zeros_like(path),
            "acceleration": np.zeros_like(path),
            "time": np.asarray([0.5, 1.5]),
            "duration": 1.5,
        }
        goal_calls = []
        self.subject._goal = lambda segment, start_delay=0.3: (
            goal_calls.append((segment, start_delay)) or segment
        )
        errors = iter((1.2338, 0.001))
        self.subject._controller_command_error = lambda _current: next(errors)

        self.subject._synchronize_and_arm(np.zeros(7))

        self.assertEqual(self.subject._position_gate.calls, [False, True])
        self.assertTrue(self.subject._position_armed)
        self.assertEqual(len(self.subject._client.goals), 1)
        self.assertEqual(goal_calls[0][1], 0.0)
        np.testing.assert_allclose(goal_calls[0][0]["time"], [0.05, 1.5])

    def test_controller_command_error_uses_joint_names_not_message_order(self):
        message = types.SimpleNamespace(
            joint_names=list(reversed(self.subject._joint_names)),
            desired=types.SimpleNamespace(positions=list(reversed(range(7)))),
        )

        self.subject._controller_state_callback(message)

        error = self.subject._controller_command_error(np.arange(7, dtype=float))
        self.assertEqual(error, 0.0)

    def test_executing_status_exposes_scheduled_motion_video_window(self):
        class Nanoseconds:
            def __init__(self, value):
                self.value = int(value)

            def __add__(self, other):
                return Nanoseconds(self.value + other.value)

            def to_nsec(self):
                return self.value

        goal = types.SimpleNamespace(
            trajectory=types.SimpleNamespace(
                header=types.SimpleNamespace(stamp=Nanoseconds(10_000_000_000)),
                points=[
                    types.SimpleNamespace(time_from_start=Nanoseconds(300_000_000)),
                    types.SimpleNamespace(time_from_start=Nanoseconds(2_300_000_000)),
                ],
            )
        )
        published = []
        self.subject._goal = lambda _segment: goal
        self.subject._publish = lambda phase, **fields: published.append(
            (phase, fields)
        )
        self.subject._fri_failure_reason = lambda: None
        self.subject._abort.clear()

        self.subject._run_segment("executing", {"duration": 2.0})

        self.assertEqual(len(self.subject._client.goals), 1)
        self.assertEqual(published[0][0], "executing")
        self.assertEqual(
            published[0][1]["motion_start_unix_ns"], 10_300_000_000
        )
        self.assertEqual(
            published[0][1]["motion_end_unix_ns"], 12_300_000_000
        )

    def test_success_keeps_final_goal_armed_and_heartbeat_alive(self):
        events = []
        self.subject._run_segment = lambda name, _segment: events.append(name)
        self.subject._wait_for_task_start = lambda _target: events.append("settled")
        self.subject._start_recording = lambda: self.fail("recording was disabled")
        self.subject._stop_recording = lambda: events.append("recording_stopped")
        self.subject._publish = lambda phase, **fields: events.append(
            (phase, fields.get("message"))
        )
        prepared = {
            "approach": object(),
            "task": {"position": np.zeros((2, 7))},
            "metrics": {"duration_s": 1.0},
        }

        self.subject._execution_worker(prepared)

        self.assertTrue(self.subject._holding_final_position)
        self.assertTrue(self.subject._position_armed)
        self.assertFalse(self.subject._execution_active)
        self.assertIsNone(self.subject._worker)
        self.assertIsNone(self.subject._prepared)
        self.assertEqual(self.subject._client.cancel_count, 0)
        self.assertEqual(self.subject.disarm_count, 0)
        self.assertIn("moving_to_start", events)
        self.assertLess(events.index("moving_to_start"), events.index("settled"))
        self.assertLess(events.index("settled"), events.index("executing"))
        self.assertIn("executing", events)
        self.assertTrue(
            any(
                isinstance(event, tuple)
                and event[0] == "complete"
                and "holding final position" in event[1]
                for event in events
            )
        )

        published = []
        self.subject._position_heartbeat_pub = types.SimpleNamespace(
            publish=lambda message: published.append(message)
        )
        self.subject._publish_position_heartbeat(None)
        self.assertEqual(len(published), 1)

    def test_task_start_wait_accepts_tracking_after_three_settled_samples(self):
        errors = iter([0.040, 0.036, 0.030, 0.028, 0.026])
        self.subject._task_start_settle_samples = 3
        self.subject._fresh_joint_position = mock.Mock(
            side_effect=lambda: np.asarray([next(errors)] + [0.0] * 6)
        )
        self.subject._fri_failure_reason = lambda: None

        with mock.patch.object(time, "sleep", return_value=None):
            final_error = self.subject._wait_for_task_start(np.zeros(7))

        self.assertAlmostEqual(final_error, 0.026)
        self.assertEqual(self.subject._fresh_joint_position.call_count, 5)

    def test_task_start_wait_times_out_above_configured_two_degree_limit(self):
        self.subject._task_start_settle_timeout = 0.1
        self.subject._fresh_joint_position = lambda: np.asarray(
            [0.040] + [0.0] * 6
        )
        self.subject._fri_failure_reason = lambda: None

        with mock.patch.object(time, "monotonic", side_effect=[0.0, 0.2]):
            with self.assertRaisesRegex(RuntimeError, "0.040 rad after 0.1 s"):
                self.subject._wait_for_task_start(np.zeros(7))

    def test_low_home_runs_vertical_recovery_before_joint_home(self):
        events = []
        self.subject._run_segment = lambda name, _segment: events.append(name)
        self.subject._fresh_joint_position = lambda: np.zeros(7)
        self.subject._stop_recording = lambda: events.append("recording_stopped")
        self.subject._publish = lambda phase, **_fields: events.append(phase)
        prepared = {
            "operation": "home",
            "recovery": object(),
            "approach": {"position": np.zeros((2, 7))},
            "metrics": {},
        }

        self.subject._execution_worker(prepared)

        self.assertLess(
            events.index("home_recovering"), events.index("returning_home")
        )
        self.assertTrue(self.subject._holding_final_position)
        self.assertTrue(self.subject._position_armed)

    def test_home_abort_after_recovery_does_not_send_joint_home(self):
        events = []

        def run_segment(name, _segment):
            events.append(name)
            if name == "home_recovering":
                self.subject._abort.set()

        self.subject._run_segment = run_segment
        self.subject._fresh_joint_position = lambda: np.zeros(7)
        self.subject._stop_recording = lambda: None
        self.subject._publish = lambda phase, **_fields: events.append(phase)
        prepared = {
            "operation": "home",
            "recovery": object(),
            "approach": {"position": np.zeros((2, 7))},
            "metrics": {},
        }

        self.subject._execution_worker(prepared)

        self.assertIn("home_recovering", events)
        self.assertNotIn("returning_home", events)
        self.assertIn("aborted", events)
        self.assertFalse(self.subject._position_armed)

    def test_failed_task_cancels_goal_and_disarms(self):
        def run_segment(name, _segment):
            if name == "executing":
                raise RuntimeError("controller failed")

        statuses = []
        self.subject._run_segment = run_segment
        self.subject._fresh_joint_position = lambda: np.zeros(7)
        self.subject._stop_recording = lambda: None
        self.subject._publish = lambda phase, **fields: statuses.append(
            (phase, fields.get("message"))
        )
        prepared = {
            "approach": object(),
            "task": {"position": np.zeros((2, 7))},
            "metrics": {},
        }

        self.subject._execution_worker(prepared)

        self.assertFalse(self.subject._holding_final_position)
        self.assertFalse(self.subject._execution_active)
        self.assertEqual(self.subject._client.cancel_count, 1)
        self.assertEqual(self.subject.disarm_count, 1)
        self.assertEqual(statuses[-1], ("failed", "controller failed"))

    def test_commanding_loss_latches_failure_and_cannot_auto_resume(self):
        self.subject._commanding_callback(types.SimpleNamespace(data=False))

        self.assertEqual(
            self.subject._execution_failure_reason,
            "FRI left COMMANDING_ACTIVE during execution",
        )
        self.assertTrue(self.subject._abort.is_set())
        self.assertIsNone(self.subject._prepared)
        self.assertEqual(self.subject._client.cancel_count, 1)
        self.assertEqual(self.subject.disarm_count, 1)

        self.subject._commanding_callback(types.SimpleNamespace(data=True))
        self.subject._mode_callback(types.SimpleNamespace(data=1))
        self.assertEqual(
            self.subject._execution_failure_reason,
            "FRI left COMMANDING_ACTIVE during execution",
        )
        self.assertEqual(self.subject._client.cancel_count, 1)
        self.assertEqual(self.subject.disarm_count, 1)

        self.subject._execution_active = False
        self.assertIsNone(self.subject._fri_failure_reason())

    def test_position_mode_loss_latches_failure(self):
        self.subject._mode_callback(types.SimpleNamespace(data=0))

        self.assertIn("left POSITION", self.subject._execution_failure_reason)
        self.assertTrue(self.subject._abort.is_set())
        self.assertEqual(self.subject._client.cancel_count, 1)
        self.assertEqual(self.subject.disarm_count, 1)

    def test_stale_status_is_a_failure(self):
        self.subject._commanding_received = time.monotonic() - 0.6
        self.assertEqual(
            self.subject._fri_failure_reason(),
            "FRI status became stale during execution",
        )

    def test_stale_torque_feedback_is_a_failure(self):
        self.subject._torque_received = time.monotonic() - 0.3

        self.assertEqual(
            self.subject._fri_failure_reason(),
            "External torque feedback is missing or stale",
        )

    def test_loss_while_idle_does_not_create_a_task_failure(self):
        self.subject._execution_active = False
        self.subject._commanding_callback(types.SimpleNamespace(data=False))

        self.assertIsNone(self.subject._execution_failure_reason)
        self.assertFalse(self.subject._abort.is_set())
        self.assertEqual(self.subject._client.cancel_count, 0)
        self.assertEqual(self.subject.disarm_count, 0)

    def test_commanding_loss_releases_completed_hold(self):
        statuses = []
        self.subject._execution_active = False
        self.subject._holding_final_position = True
        self.subject._publish = lambda phase, **fields: statuses.append(
            (phase, fields.get("message"))
        )

        self.subject._commanding_callback(types.SimpleNamespace(data=False))

        self.assertFalse(self.subject._holding_final_position)
        self.assertIsNone(self.subject._execution_failure_reason)
        self.assertFalse(self.subject._abort.is_set())
        self.assertEqual(self.subject._client.cancel_count, 0)
        self.assertEqual(self.subject.disarm_count, 1)
        self.assertEqual(statuses[-1][0], "hold_released")

    def test_mode_loss_releases_completed_hold(self):
        self.subject._execution_active = False
        self.subject._holding_final_position = True
        self.subject._publish = lambda *_args, **_fields: None

        self.subject._mode_callback(types.SimpleNamespace(data=0))

        self.assertFalse(self.subject._holding_final_position)
        self.assertEqual(self.subject.disarm_count, 1)

    def test_stale_fri_status_releases_completed_hold(self):
        ready = []
        self.subject._execution_active = False
        self.subject._holding_final_position = True
        self.subject._commanding_received = time.monotonic() - 0.6
        self.subject._fri_mode_received = time.monotonic() - 0.6
        self.subject._fri_ready_pub = types.SimpleNamespace(
            publish=lambda message: ready.append(message.data)
        )
        self.subject._publish = lambda *_args, **_fields: None

        self.subject._publish_fri_ready_status(None)

        self.assertEqual(ready, [False])
        self.assertFalse(self.subject._holding_final_position)
        self.assertEqual(self.subject.disarm_count, 1)

    def test_torque_trip_during_completed_hold_disarms_immediately(self):
        statuses = []
        self.subject._execution_active = False
        self.subject._holding_final_position = True
        self.subject._torque_trip_count = 4
        self.subject._publish = lambda phase, **fields: statuses.append(
            (phase, fields.get("message"))
        )
        message = types.SimpleNamespace(
            external_torques=types.SimpleNamespace(data=np.full(7, 2.0))
        )

        self.subject._torque_callback(message)

        self.assertTrue(self.subject._protective_stop)
        self.assertTrue(self.subject._abort.is_set())
        self.assertFalse(self.subject._holding_final_position)
        self.assertEqual(self.subject._client.cancel_count, 1)
        self.assertEqual(self.subject.disarm_count, 1)
        self.assertEqual(statuses[-1][0], "protective_stop")

    def test_new_path_increments_serial_and_invalidates_prepared_trajectory(self):
        self.subject._execution_active = False
        self.subject._worker = None
        self.subject._path_serial = 4
        published = {}

        def publish(phase, **fields):
            published.update(phase=phase, **fields)

        self.subject._publish = publish
        stamp = types.SimpleNamespace(to_nsec=lambda: 123456)
        message = types.SimpleNamespace(
            header=types.SimpleNamespace(stamp=stamp),
            poses=[object(), object()],
        )

        self.subject._path_callback(message)

        self.assertIs(self.subject._pending_path, message)
        self.assertIsNone(self.subject._path)
        self.assertEqual(self.subject._path_serial, 4)

        self.subject._orientation_constraints_callback(
            types.SimpleNamespace(
                data=json.dumps(
                    {
                        "schema_version": 5,
                        "stamp_ns": 123456,
                        "task_id": "BarClean",
                        "point_count": 2,
                        "tool_yaw_active": [0, 1],
                        "approach_obstacle": {
                            "type": "circle",
                            "center": [0.55, -0.2, 0.1],
                            "table_normal": [0.0, 0.0, 1.0],
                            "radius": 0.075,
                            "clearance": 0.085,
                            "margin": 0.005,
                        },
                        "stage_timing": {
                            "boundaries": [1],
                            "transition_windows": [],
                            "speed_scale": 0.5,
                            "ramp_before_m": 0.02,
                            "task_start_ramp_m": 0.03,
                        },
                    }
                )
            )
        )

        self.assertIs(self.subject._path, message)
        np.testing.assert_array_equal(
            self.subject._path_tool_yaw_active, [False, True]
        )
        self.assertEqual(
            self.subject._path_approach_obstacle["center"],
            [0.55, -0.2, 0.1],
        )
        self.assertAlmostEqual(
            self.subject._path_approach_obstacle["clearance"], 0.085
        )
        self.assertEqual(self.subject._path_stage_timing["boundaries"], [1])
        self.assertAlmostEqual(
            self.subject._path_stage_timing["speed_scale"], 0.5
        )
        self.assertEqual(self.subject._path_serial, 5)
        self.assertIsNone(self.subject._prepared)
        self.assertEqual(published["phase"], "path_received")
        self.assertEqual(published["path_serial"], 5)

    def test_stage_zero_metadata_accepts_capsule_geometry(self):
        message = types.SimpleNamespace(
            data=json.dumps(
                {
                    "schema_version": 5,
                    "stamp_ns": 123456,
                    "task_id": "BarClean",
                    "point_count": 2,
                    "tool_yaw_active": [0, 1],
                    "approach_obstacle": {
                        "type": "capsule",
                        "endpoints": [
                            [0.6482, 0.1373, 0.13437],
                            [0.585671, 0.208104, 0.13437],
                        ],
                        "table_normal": [0.0, 0.0, 1.0],
                        "radius": 0.025,
                        "clearance": 0.085,
                        "margin": 0.005,
                    },
                    "stage_timing": {
                        "boundaries": [1],
                        "transition_windows": [],
                        "speed_scale": 0.5,
                        "ramp_before_m": 0.02,
                        "task_start_ramp_m": 0.03,
                    },
                }
            )
        )

        _stamp, _task, _yaw, obstacle, _timing = (
            self.subject._parse_orientation_constraints(message)
        )

        self.assertEqual(obstacle["type"], "capsule")
        self.assertEqual(len(obstacle["endpoints"]), 2)
        self.assertAlmostEqual(obstacle["radius"], 0.025)

    def test_new_path_is_ignored_during_execution(self):
        self.subject._path = "current"
        self.subject._path_serial = 4
        self.subject._publish = mock.Mock()

        self.subject._path_callback(
            types.SimpleNamespace(
                header=types.SimpleNamespace(
                    stamp=types.SimpleNamespace(to_nsec=lambda: 123456)
                ),
                poses=[object(), object()],
            )
        )

        self.assertEqual(self.subject._path, "current")
        self.assertEqual(self.subject._path_serial, 4)
        self.subject._publish.assert_not_called()

    def test_robot_motion_after_planning_rebuilds_approach_from_latest_state(self):
        current = np.full(7, 0.2, dtype=float)
        original = {
            "start": np.zeros(7, dtype=float),
            "path_serial": 7,
        }
        rebuilt = {
            "start": current.copy(),
            "path_serial": 7,
            "metrics": {"joint_points": 12},
        }
        self.subject._path = object()
        self.subject._path_serial = 7
        self.subject._build_plan = mock.Mock(return_value=rebuilt)
        self.subject._fresh_joint_position = mock.Mock(
            return_value=current.copy()
        )
        self.subject._publish = mock.Mock()

        prepared, latest = self.subject._refresh_prepared_start(original, current)

        self.assertIs(prepared, rebuilt)
        np.testing.assert_allclose(latest, current)
        self.assertIs(self.subject._prepared, rebuilt)
        self.subject._build_plan.assert_called_once_with(self.subject._path, current)
        self.assertEqual(
            [call.args[0] for call in self.subject._publish.call_args_list],
            ["repreparing", "prepared"],
        )

    def test_robot_continuing_to_move_during_rebuild_is_rejected(self):
        current = np.full(7, 0.2, dtype=float)
        self.subject._path = object()
        self.subject._path_serial = 7
        self.subject._build_plan = mock.Mock(
            return_value={
                "start": current.copy(),
                "path_serial": 7,
                "metrics": {},
            }
        )
        self.subject._fresh_joint_position = mock.Mock(
            return_value=current + np.full(7, 0.1, dtype=float)
        )
        self.subject._publish = mock.Mock()

        with self.assertRaisesRegex(RuntimeError, "continued moving"):
            self.subject._refresh_prepared_start(
                {"start": np.zeros(7, dtype=float), "path_serial": 7},
                current,
            )


if __name__ == "__main__":
    unittest.main()
