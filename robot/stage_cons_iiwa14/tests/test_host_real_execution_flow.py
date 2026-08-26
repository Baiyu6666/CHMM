import importlib.util
import math
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


SOURCE = Path(__file__).resolve().parents[1] / "host_gui/supervisor.py"


def load_supervisor_module():
    spec = importlib.util.spec_from_file_location("stage_host_supervisor_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HostRealExecutionFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_supervisor_module()

    def setUp(self):
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.module.GUI_SETTINGS_PATH = (
            Path(self._temporary_directory.name) / "gui_settings.json"
        )
        self.subject = self.module.Supervisor()

    def tearDown(self):
        self.subject.shutdown()
        self._temporary_directory.cleanup()

    @staticmethod
    def _pose(x, y):
        return {
            "x": x,
            "y": y,
            "z": 0.26584,
            "qx": 0.0,
            "qy": 1.0,
            "qz": 0.0,
            "qw": 0.0,
        }

    def test_barinspect_is_hidden_from_active_gui_tasks(self):
        self.assertEqual(set(self.subject._task_profiles), {"BarClean"})

    def test_real_trace_uses_position_station_tf_and_survives_completion(self):
        pose = self._pose(0.61, -0.17)
        self.subject._robot_ee_pose = lambda: dict(pose)
        self.subject._demo_visualization = lambda: self.fail(
            "real execution visualization must not depend on the Demo station"
        )
        names = [
            spec["name"]
            for spec in self.subject._task_feature_definitions["BarClean"]["schema"]
        ]
        self.subject._real_feature_values = lambda *_args: {
            name: float(index) for index, name in enumerate(names)
        }
        self.subject._task_state.update({"mode": "real", "phase": "executing"})

        self.subject._update_real_visualization(sample_time=10.0)
        pose["x"] += 0.01
        self.subject._update_real_visualization(sample_time=10.1)
        self.subject._task_state["phase"] = "complete"
        self.subject._demo_visualization = lambda: None
        self.subject._direct_optitrack_visualization = lambda: None

        visualization = self.subject.task_visualization()
        self.assertEqual(visualization["phase"], "complete")
        self.assertEqual(
            visualization["trace"], [[0.61, -0.17], [0.62, -0.17]]
        )
        self.assertEqual(len(visualization["feature_series"]["samples"]), 2)

    def test_tf_pose_parser_keeps_precision_and_rejects_nonfinite_values(self):
        parser = self.module.RosTfPoseStream._parse_pose
        payload = (
            '{"x":0.123456789,"y":-0.2,"z":0.3,'
            '"qx":0.0,"qy":1.0,"qz":0.0,"qw":0.0}'
        )

        self.assertEqual(parser(payload)["x"], 0.123456789)
        self.assertIsNone(parser('{"x":0.1}'))
        self.assertIsNone(parser(payload.replace("0.3", "NaN")))

    def test_real_execution_plans_and_acknowledges_new_path_before_fri(self):
        events = []
        status_reads = iter(
            [
                {"path_serial": 12, "task_id": "BarClean", "phase": "idle"},
                {
                    "path_serial": 13,
                    "task_id": "BarClean",
                    "phase": "complete",
                    "message": "done",
                    "run_directory": None,
                },
            ]
        )
        self.subject._container_running = lambda: True
        self.subject._assert_container_storage_access = lambda *_args: None
        self.subject._robot_iface_state = lambda: {"configured": True}
        self.subject._start_real_station = lambda: events.append("station")
        self.subject._read_real_status = lambda: next(status_reads)
        self.subject._read_plan_visualization = lambda _container: {}
        self.subject._update_plan_visualization = (
            lambda _payload: events.append("visualization")
        )

        def wait_for_plan(previous_serial, task_id):
            self.assertEqual(previous_serial, 12)
            self.assertEqual(task_id, "BarClean")
            events.append("path_ack")
            return True

        self.subject._wait_for_real_plan = wait_for_plan
        self.subject._wait_for_fri_position = lambda: events.append("fri") or True

        def real_ros(*arguments, timeout=10.0):
            del timeout
            if "submit_real_plan.py" in arguments:
                events.append("plan")
                return subprocess.CompletedProcess(
                    arguments, 0, '{"success":true,"message":"planned"}\n', ""
                )
            elif "/iiwa14/real_executor/set_recording" in arguments:
                events.append("recording")
            elif "/iiwa14/real_executor/prepare" in arguments:
                events.append("prepare")
            elif "/iiwa14/real_executor/execute" in arguments:
                events.append("execute")
            return subprocess.CompletedProcess(arguments, 0, "success: True\n", "")

        self.subject._real_ros = real_ros
        self.subject._execute_real_task(
            "BarClean", self._pose(0.635, -0.0362), self._pose(0.5196, -0.2868)
        )

        self.assertLess(events.index("plan"), events.index("path_ack"))
        self.assertLess(events.index("path_ack"), events.index("visualization"))
        self.assertLess(events.index("visualization"), events.index("fri"))
        self.assertLess(events.index("fri"), events.index("prepare"))
        self.assertLess(events.index("prepare"), events.index("execute"))

    def test_gui_settings_persist_task_source_poses_and_video(self):
        start = self._pose(0.51, -0.12)
        goal = self._pose(0.63, -0.31)

        self.subject.update_gui_settings(
            {
                "task_id": "BarClean",
                "constraint_source_id": "true",
                "start": start,
                "goal": goal,
                "render_video": True,
            }
        )
        restored = self.module.Supervisor()
        try:
            self.assertEqual(restored._gui_settings["task_id"], "BarClean")
            self.assertEqual(restored._gui_settings["start_by_task"]["BarClean"], start)
            self.assertEqual(restored._gui_settings["goal_by_task"]["BarClean"], goal)
            self.assertTrue(restored._gui_settings["render_video"])
        finally:
            restored.shutdown()

    def test_real_station_restores_tracking_even_when_executor_is_running(self):
        events = []
        self.subject._start_optitrack_process = lambda: events.append("tracking")
        self.subject._ros_nodes = lambda: [
            "/iiwa14/iiwa_driver",
            "/iiwa14/real_executor",
            "/stage_constraint_planner",
        ]
        self.subject._wait_for_real_station = (
            lambda child, timeout=20.0: events.append(("ready", child, timeout))
        )

        self.subject._start_real_station()

        self.assertEqual(events, ["tracking", ("ready", None, 5.0)])

    def test_verified_running_station_uses_node_only_fast_path(self):
        self.subject._real_station_verified = True
        self.subject._ros_nodes = lambda: [
            "/iiwa14/iiwa_driver",
            "/iiwa14/real_executor",
            "/stage_constraint_planner",
            "/vrpn_client_node",
            "/optitrack_base_transform",
        ]
        self.subject._start_optitrack_process = lambda: self.fail(
            "fast path must not probe OptiTrack topics"
        )
        self.subject._real_station_interfaces_ready = lambda: self.fail(
            "verified fast path must not relist ROS interfaces"
        )
        self.subject._wait_for_real_station = lambda *_args, **_kwargs: self.fail(
            "verified fast path must not enter the startup wait loop"
        )

        self.subject._start_real_station()

    def test_running_station_is_verified_once_after_supervisor_restart(self):
        checks = []
        self.subject._ros_nodes = lambda: [
            "/iiwa14/iiwa_driver",
            "/iiwa14/real_executor",
            "/stage_constraint_planner",
            "/vrpn_client_node",
            "/optitrack_base_transform",
        ]
        self.subject._real_station_interfaces_ready = (
            lambda: checks.append("interfaces") or True
        )
        self.subject._start_optitrack_process = lambda: self.fail(
            "healthy running tracking must be reused"
        )

        self.subject._start_real_station()
        self.subject._start_real_station()

        self.assertEqual(checks, ["interfaces"])
        self.assertTrue(self.subject._real_station_verified)

    def test_home_station_does_not_require_optitrack(self):
        self.subject._real_station_verified = True
        self.subject._start_optitrack_process = lambda: self.fail(
            "Return Home must not touch OptiTrack"
        )
        self.subject._ros_nodes = lambda: [
            "/iiwa14/iiwa_driver",
            "/iiwa14/real_executor",
            "/stage_constraint_planner",
        ]
        self.subject._wait_for_real_station = lambda *_args, **_kwargs: self.fail(
            "verified Home station must use the fast path"
        )

        self.subject._start_real_station(require_optitrack=False)

    def test_stopping_driver_invalidates_station_fast_path(self):
        self.subject._real_station_verified = True
        self.subject._signal_child = lambda _name: None
        self.subject._container_running = lambda: False

        self.subject._stop_driver_process()

        self.assertFalse(self.subject._real_station_verified)

    def test_return_home_waits_for_fri_then_uses_dedicated_service(self):
        events = []
        self.subject._task_trace.extend([[0.61, -0.17], [0.62, -0.18]])
        self.subject._task_planned_trace = [[0.60, -0.16], [0.63, -0.19]]
        statuses = iter(
            [
                {
                    "phase": "idle",
                    "execution_active": False,
                    "protective_stop": False,
                },
                {
                    "phase": "complete",
                    "operation": "home",
                    "message": "Robot returned Home",
                },
            ]
        )
        self.subject._container_running = lambda: True
        self.subject._robot_iface_state = lambda: {"configured": True}
        self.subject._start_real_station = (
            lambda require_optitrack=True: events.append(
                ("station", require_optitrack)
            )
        )
        self.subject._read_real_status = lambda: next(statuses)
        self.subject._wait_for_fri_position = lambda: events.append("fri") or True

        def real_ros(*arguments, timeout=10.0):
            events.append((arguments, timeout))
            return subprocess.CompletedProcess(arguments, 0, "success: True\n", "")

        self.subject._real_ros = real_ros

        self.subject._return_robot_home()

        self.assertEqual(events[0], ("station", False))
        self.assertEqual(events[1], "fri")
        self.assertIn("/iiwa14/real_executor/return_home", events[2][0])
        self.assertEqual(self.subject._task_state["phase"], "complete")
        self.assertEqual(self.subject._task_state["mode"], "home")
        self.assertEqual(
            list(self.subject._task_trace), [[0.61, -0.17], [0.62, -0.18]]
        )
        self.assertEqual(
            self.subject._task_planned_trace, [[0.60, -0.16], [0.63, -0.19]]
        )

    def test_stopping_return_home_uses_real_executor_abort(self):
        aborts = []
        self.subject._task_state.update(
            {"mode": "home", "phase": "returning_home"}
        )
        self.subject._container_running = lambda: True
        self.subject._ros_nodes = lambda: ["/iiwa14/real_executor"]
        self.subject._abort_real_and_confirm = aborts.append

        self.subject.abort_task()

        self.assertEqual(aborts, ["user pressed Stop Execution"])
        self.assertTrue(self.subject._task_abort.is_set())
        self.assertEqual(self.subject._task_state["phase"], "aborted")

    def test_supervisor_failure_requests_and_confirms_real_abort(self):
        calls = []
        self.subject._run = lambda *args, **kwargs: (
            calls.append((args, kwargs))
            or subprocess.CompletedProcess(args, 0, "success: True\n", "")
        )
        statuses = iter(
            [
                {
                    "phase": "executing",
                    "execution_active": True,
                    "holding_final_position": False,
                },
                {
                    "phase": "aborted",
                    "execution_active": False,
                    "holding_final_position": False,
                },
            ]
        )
        self.subject._read_real_status = lambda: next(statuses)

        self.subject._abort_real_and_confirm("status stream failed")

        self.assertIn("/iiwa14/real_executor/abort", calls[0][0][0])
        self.assertFalse(
            self.subject._task_state.get("phase") == "control_status_unknown"
        )

    def test_user_real_abort_waits_for_executor_to_confirm_stopped(self):
        calls = []
        self.subject._task_state.update(
            {"mode": "real", "phase": "executing"}
        )
        self.subject._container_running = lambda: True
        self.subject._ros_nodes = lambda: ["/iiwa14/real_executor"]
        self.subject._run = lambda *args, **kwargs: (
            calls.append((args, kwargs))
            or subprocess.CompletedProcess(args, 0, "success: True\n", "")
        )
        statuses = iter(
            [
                {
                    "phase": "executing",
                    "execution_active": True,
                    "holding_final_position": False,
                },
                {
                    "phase": "aborted",
                    "execution_active": False,
                    "holding_final_position": False,
                },
            ]
        )
        self.subject._read_real_status = lambda: next(statuses)

        self.subject.abort_task()

        self.assertTrue(self.subject._task_abort.is_set())
        self.assertIn("/iiwa14/real_executor/abort", calls[0][0][0])
        self.assertEqual(self.subject._task_state["phase"], "aborted")

    def test_task_job_clears_stale_abort_before_worker_starts(self):
        observed = []
        self.subject._task_abort.set()

        self.subject._start_job(
            "Execute simulation task",
            lambda: observed.append(self.subject._task_abort.is_set()),
            reset_task_abort=True,
        )
        self.subject._job.join(timeout=1.0)

        self.assertEqual(observed, [False])

    def test_demo_prepare_reuses_existing_safe_torque_driver(self):
        events = []
        self.subject._start_job = lambda _name, target: target()
        self.subject._container_running = lambda: True
        self.subject._robot_iface_state = lambda: {"configured": True}
        self.subject._ros_nodes = lambda: ["/iiwa14/iiwa_driver"]
        self.subject._running_iiwa_controllers = lambda: ["SafeTorqueController"]
        self.subject._stop_driver_process = lambda: events.append("stop")
        self.subject._spawn = lambda *_args: events.append("spawn")

        self.subject.prepare_demo_control()

        self.assertEqual(events, [])

    def test_demo_prepare_releases_position_station_before_torque_driver(self):
        events = []
        child = types.SimpleNamespace(poll=lambda: None)
        self.subject._start_job = lambda _name, target: target()
        self.subject._container_running = lambda: True
        self.subject._robot_iface_state = lambda: {"configured": True}
        self.subject._ros_nodes = lambda: ["/iiwa14/iiwa_driver"]
        self.subject._running_iiwa_controllers = lambda: [
            "PositionTrajectoryController"
        ]
        self.subject._release_robot_control = (
            lambda reason: events.append(("release_position", reason))
        )
        self.subject._driver_process_containers = lambda: []
        self.subject._run = lambda *_args, **_kwargs: None
        self.subject._spawn = lambda name, command: (
            events.append((name, command)) or child
        )
        self.subject._wait_for_driver_ready = lambda current: events.append(
            ("ready", current)
        )

        self.subject.prepare_demo_control()

        self.assertEqual(events[0][0], "release_position")
        self.assertEqual(events[1][0], "driver")
        self.assertIn("iiwa14_bringup.launch", events[1][1])
        self.assertEqual(events[2], ("ready", child))

    def test_demo_prepare_routes_even_final_hold_through_release(self):
        events = []
        self.subject._start_job = lambda _name, target: target()
        self.subject._container_running = lambda: True
        self.subject._robot_iface_state = lambda: {"configured": True}
        self.subject._ros_nodes = lambda: ["/iiwa14/iiwa_driver"]
        self.subject._running_iiwa_controllers = lambda: [
            "PositionTrajectoryController"
        ]
        self.subject._release_robot_control = lambda reason: events.append(reason)
        self.subject._driver_process_containers = lambda: []
        child = types.SimpleNamespace(poll=lambda: None)
        self.subject._run = lambda *_args, **_kwargs: None
        self.subject._spawn = lambda *_args: child
        self.subject._wait_for_driver_ready = lambda _child: None

        self.subject.prepare_demo_control()

        self.assertEqual(len(events), 1)
        self.assertIn("Demo / Torque", events[0])

    def test_control_mode_requires_one_unambiguous_owner(self):
        classify = self.subject._control_mode_from_graph
        self.assertEqual(classify([], [])["mode"], "idle")
        self.assertEqual(
            classify(["/iiwa14/iiwa_driver"], ["SafeTorqueController"])["mode"],
            "demo",
        )
        self.assertEqual(
            classify(
                ["/iiwa14/iiwa_driver", "/iiwa14/real_executor"],
                ["PositionTrajectoryController"],
            )["mode"],
            "planner",
        )
        self.assertEqual(
            classify(
                ["/iiwa14/iiwa_driver", "/iiwa14/real_executor"],
                ["SafeTorqueController", "PositionTrajectoryController"],
            )["mode"],
            "conflict",
        )
        self.assertEqual(
            classify(
                ["/iiwa14/iiwa_driver", "/iiwa14/real_executor"], []
            )["mode"],
            "planner_waiting",
        )
        self.assertEqual(
            classify(
                ["/iiwa14/iiwa_driver", "/iiwa14/real_executor"], None
            )["mode"],
            "planner_waiting",
        )

    def test_new_real_station_failure_cleans_up_partial_driver(self):
        events = []
        child = types.SimpleNamespace(poll=lambda: None)
        self.subject._start_optitrack_process = lambda: events.append("tracking")
        self.subject._ros_nodes = lambda: []
        self.subject._driver_process_containers = lambda: []
        self.subject._spawn = lambda *_args: child

        def fail_start(_child):
            raise RuntimeError("station unavailable")

        self.subject._wait_for_real_station = fail_start
        self.subject._stop_driver_process = lambda: events.append("cleanup")

        with self.assertRaisesRegex(RuntimeError, "station unavailable"):
            self.subject._start_real_station()

        self.assertEqual(events, ["tracking", "cleanup"])

    def test_real_station_automatically_releases_demo_control(self):
        events = []
        child = types.SimpleNamespace(poll=lambda: None)
        self.subject._start_optitrack_process = lambda: events.append("tracking")
        self.subject._ros_nodes = lambda: ["/iiwa14/iiwa_driver"]
        self.subject._release_robot_control = lambda reason: events.append(
            ("release", reason)
        )
        self.subject._driver_process_containers = lambda: []
        self.subject._spawn = lambda *_args: child
        self.subject._wait_for_real_station = lambda current: events.append(
            ("ready", current)
        )

        self.subject._start_real_station()

        self.assertEqual(events[0], "tracking")
        self.assertEqual(events[1][0], "release")
        self.assertEqual(events[2], ("ready", child))

    def test_release_quiesces_demo_then_aborts_real_then_stops_driver(self):
        events = []
        self.subject._container_running = lambda: True
        self.subject._ros_nodes = lambda: ["/iiwa14/real_executor", "/iiwa14/iiwa_driver"]
        self.subject._quiesce_demo_control = lambda: events.append("quiesce_demo")
        self.subject._abort_real_and_confirm = lambda reason: events.append(
            ("abort_real", reason)
        )
        self.subject._stop_driver_process = lambda: events.append("stop_driver")

        self.subject._release_robot_control("test transition")

        self.assertEqual(
            events,
            ["quiesce_demo", ("abort_real", "test transition"), "stop_driver"],
        )

    def test_demo_quiesce_timeout_does_not_block_remaining_shutdown_commands(self):
        calls = []

        def run(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 0))
            return subprocess.CompletedProcess(args[0], 0, "", "")

        self.subject._run = run
        self.subject._container_running = lambda: True

        self.subject._quiesce_demo_control()

        self.assertEqual(len(calls), 3)

    def test_optitrack_start_uses_configured_rigid_body_names(self):
        calls = []
        child = types.SimpleNamespace(poll=lambda: None)
        self.subject._container_running = lambda: True
        self.subject._optitrack_settings = lambda: (
            "10.0.0.8", "robot_base", "tracked_bar", "tracked_ball"
        )
        self.subject._ros_nodes = lambda: []
        self.subject._spawn = lambda name, command: (
            calls.append((name, command)) or child
        )
        self.subject._wait_for_optitrack_ready = (
            lambda current_child, base, obj, obstacle: calls.append(
                (current_child, base, obj, obstacle)
            )
        )

        self.subject._start_optitrack_process()

        name, command = calls[0]
        self.assertEqual(name, "tracking")
        self.assertIn("server:=10.0.0.8", command)
        self.assertIn("base_name:=robot_base", command)
        self.assertIn("object_name:=tracked_bar", command)
        self.assertIn("obstacle_name:=tracked_ball", command)
        self.assertEqual(calls[1], (child, "robot_base", "tracked_bar", "tracked_ball"))

    def test_existing_tracking_chain_checks_raw_then_transformed_poses(self):
        checked = []
        self.subject._ros_nodes = lambda: [
            "/vrpn_client_node", "/optitrack_base_transform"
        ]
        self.subject._topic_has_fresh_message = (
            lambda topic: checked.append(topic) or True
        )

        self.subject._wait_for_optitrack_ready(
            None, "iiwa14", "baiyu_bar", "baiyu_obs_bar", timeout=0.1
        )

        self.assertEqual(
            checked,
            [
                "/vrpn_client_node/iiwa14/pose",
                "/vrpn_client_node/baiyu_bar/pose",
                "/vrpn_client_node/baiyu_obs_bar/pose",
                "/vrpn_client_node/baiyu_bar/pose_from_iiwa14",
                "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14",
            ],
        )

    def test_optitrack_diagnostic_identifies_missing_base_pose(self):
        self.subject._ros_nodes = lambda: [
            "/vrpn_client_node", "/optitrack_base_transform"
        ]
        self.subject._topic_has_fresh_message = (
            lambda topic: topic != "/vrpn_client_node/iiwa14/pose"
        )

        with mock.patch.object(
            self.module.time, "monotonic", side_effect=[0.0, 0.0, 2.0]
        ), mock.patch.object(self.module.time, "sleep"):
            with self.assertRaisesRegex(
                RuntimeError, "base rigid body 'iiwa14'.*iiwa14/pose"
            ):
                self.subject._wait_for_optitrack_ready(
                    None, "iiwa14", "baiyu_bar", "baiyu_obs_bar", timeout=1.0
                )

    def test_demo_station_reuses_standalone_tracking_chain(self):
        spawned = []
        child = types.SimpleNamespace(poll=lambda: None)
        self.subject._container_running = lambda: True
        self.subject._ros_nodes = lambda: [
            "/vrpn_client_node", "/optitrack_base_transform"
        ]
        self.subject._optitrack_settings = lambda: (
            "128.178.145.104", "iiwa14", "baiyu_bar", "baiyu_obs_bar"
        )
        self.subject._spawn = lambda name, command: (
            spawned.append((name, command)) or child
        )
        self.subject._wait_for_demo_ready = lambda *_args, **_kwargs: None

        self.subject._start_demo_process()

        self.assertEqual(spawned[0][0], "demo")
        self.assertIn("start_optitrack:=false", spawned[0][1])

    def test_wait_for_fri_accepts_an_already_active_position_session(self):
        reads = []
        self.subject._read_real_fri_ready = (
            lambda: reads.append(True)
            or (True, "FRI POSITION mode is COMMANDING_ACTIVE")
        )

        self.assertTrue(self.subject._wait_for_fri_position(timeout=0.1))
        self.assertEqual(len(reads), 1)

    def test_path_acknowledgement_rejects_stale_serial_and_wrong_task(self):
        statuses = iter(
            [
                {
                    "path_serial": 8,
                    "task_id": "BarClean",
                    "phase": "path_received",
                },
                {
                    "path_serial": 9,
                    "task_id": "LegacyTask",
                    "phase": "path_received",
                },
                {
                    "path_serial": 9,
                    "task_id": "BarClean",
                    "phase": "path_received",
                },
            ]
        )
        self.subject._read_real_status = lambda: next(statuses)

        self.assertTrue(
            self.subject._wait_for_real_plan(8, "BarClean", timeout=1.0)
        )

    def test_real_execute_request_no_longer_requires_confirmed_field(self):
        started = []
        self.subject._task_state["task_id"] = "BarClean"
        self.subject._start_job = lambda name, target, **_kwargs: started.append(
            (name, target)
        )
        payload = {
            "task_id": "BarClean",
            "mode": "real",
            "start": self._pose(0.635, -0.0362),
            "goal": self._pose(0.5196, -0.2868),
        }

        self.subject.execute_task(payload)

        self.assertEqual(started[0][0], "Execute real task")

    def test_ros_csv_reader_accepts_large_planner_visualization(self):
        payload = "A" * (131072 + 4096)
        output = '%time,field.data\n1.0,"{}"\n'.format(payload)

        self.assertEqual(self.subject._string_message_from_csv(output), payload)

    def test_simulation_scene_snapshot_uses_current_optitrack_geometry(self):
        self.subject._direct_optitrack_visualization = mock.Mock(return_value=None)
        self.subject._demo_visualization = mock.Mock(
            return_value={
                "source": "optitrack",
                "current_ee": None,
                "scene_geometry": {
                    "bar": {
                        "pivot": [0.6267, 0.0824],
                        "axis": [-0.084, -1.982],
                        "live": True,
                    },
                    "obstacle": {
                        "center": [0.6271, 0.2706],
                        "live": True,
                    },
                },
            }
        )

        snapshot = self.subject._simulation_scene_snapshot()

        self.assertEqual(snapshot["source"], "optitrack")
        self.assertEqual(snapshot["bar"]["pivot"], [0.6267, 0.0824])
        self.assertAlmostEqual(
            sum(value * value for value in snapshot["bar"]["axis"]), 1.0
        )
        self.assertTrue(snapshot["bar"]["live"])
        self.assertEqual(snapshot["obstacle"]["center"], [0.6271, 0.2706])
        self.assertTrue(snapshot["obstacle"]["live"])

    def test_direct_optitrack_visualization_uses_real_station_topics(self):
        topic_values = {
            "/vrpn_client_node/baiyu_bar/pose_from_iiwa14": (
                "12,1787589991727310453,base,-0.034,0.060,0.616,0.0,1.0,0.0,0.0"
            ),
            "/vrpn_client_node/baiyu_obs_bar/pose_from_iiwa14": (
                "13,1787589991727310453,base,0.219,0.022,0.578,0.0,0.0,0.0,1.0"
            ),
        }
        self.subject._topic_value = mock.Mock(
            side_effect=lambda _container, topic, **_kwargs: topic_values[topic]
        )

        visualization = self.subject._direct_optitrack_visualization()

        self.assertEqual(visualization["source"], "optitrack")
        self.assertEqual(
            visualization["scene_geometry"]["bar"]["pivot"], [0.616, -0.034]
        )
        self.assertEqual(
            visualization["scene_geometry"]["bar"]["axis"], [0.0, -1.0]
        )
        self.assertEqual(
            visualization["scene_geometry"]["obstacle"]["center"],
            [0.578, 0.219],
        )
        self.assertTrue(visualization["scene_geometry"]["bar"]["live"])
        self.assertTrue(visualization["scene_geometry"]["obstacle"]["live"])

    def test_submit_sim_task_forwards_frozen_scene_snapshot(self):
        captured = {}
        snapshot = {
            "source": "optitrack",
            "bar": {
                "pivot": [0.6267, 0.0824],
                "axis": [-0.04, -0.999],
                "live": True,
            },
            "obstacle": {"center": [0.6271, 0.2706], "live": True},
        }

        def sim_ros(*arguments, timeout=10.0):
            captured["arguments"] = arguments
            captured["timeout"] = timeout
            return subprocess.CompletedProcess(
                arguments, 0, '{"success":true,"message":"planned"}\n', ""
            )

        self.subject._sim_ros = sim_ros
        self.subject._submit_sim_task(
            "BarClean",
            self._pose(0.6183, -0.1613),
            self._pose(0.45, -0.44),
            False,
            snapshot,
        )

        payload = self.module.json.loads(captured["arguments"][-1])
        self.assertEqual(payload["scene_snapshot"], snapshot)
        self.assertEqual(payload["task_id"], "BarClean")
        self.assertFalse(payload["render_video"])
        self.assertEqual(captured["timeout"], 15.0)

    def test_inactive_task_without_live_topics_uses_dashed_fallback(self):
        self.subject._task_state.update(
            {"task_id": "BarClean", "mode": "real", "phase": "failed"}
        )
        self.subject._direct_optitrack_visualization = mock.Mock(return_value=None)
        self.subject._demo_visualization = mock.Mock(return_value=None)
        self.subject._update_plan_visualization(
            {
                "task_id": "BarClean",
                "trace": [[float(index), 0.0] for index in range(6)],
                "feature_names": ["surface_dist"],
                "feature_schema": [{"name": "surface_dist", "unit": "m"}],
                "constraint_specs": [],
                "feature_samples": [[float(index), 0.02] for index in range(6)],
                "stage_boundaries": [1, 2, 3, 4, 5],
                "stage_boundary_times": [1.0, 2.0, 3.0, 4.0],
                "stage_transition_end_times": [1.0, 2.0, 3.0, 4.0],
                "scene_geometry": {
                    "bar": {"pivot": [0.64, -0.16], "axis": [2.0, 0.0]},
                    "obstacle": {"center": [0.67, 0.09]},
                },
            }
        )

        visualization = self.subject.task_visualization()

        self.assertEqual(visualization["source"], "fallback")
        self.assertFalse(visualization["scene_geometry"]["bar"]["live"])
        self.assertFalse(visualization["scene_geometry"]["obstacle"]["live"])

    def test_control_status_unknown_returns_to_live_optitrack_scene(self):
        self.subject._task_state.update(
            {
                "task_id": "BarClean",
                "mode": "real",
                "phase": "control_status_unknown",
            }
        )
        self.subject._task_scene_geometry = {
            "bar": {"pivot": [-4.2, 1.2], "axis": [1.0, 0.0]},
            "obstacle": {"center": [0.63, 0.32]},
        }
        self.subject._task_scene_source = "planner_scene"
        self.subject._direct_optitrack_visualization = mock.Mock(
            return_value={
                "current_ee": None,
                "scene_geometry": {
                    "bar": {"pivot": [0.61, 0.03], "axis": [0.0, 1.0]},
                    "obstacle": {"center": [0.63, 0.32]},
                },
                "source": "optitrack",
            }
        )
        self.subject._demo_visualization = mock.Mock(
            return_value={
                "current_ee": None,
                "scene_geometry": {
                    "bar": {"pivot": [9.0, 9.0], "axis": [1.0, 0.0]},
                    "obstacle": {"center": [9.0, 9.0]},
                },
                "source": "optitrack",
            }
        )

        visualization = self.subject.task_visualization()

        self.assertEqual(visualization["source"], "optitrack")
        self.assertEqual(
            visualization["scene_geometry"]["bar"]["pivot"], [0.61, 0.03]
        )

    def test_waiting_for_fri_keeps_the_planner_task_frame_frozen(self):
        self.subject._task_state.update(
            {"task_id": "BarClean", "mode": "real", "phase": "waiting_for_fri"}
        )
        self.subject._task_scene_geometry = {
            "bar": {"pivot": [0.50, -0.20], "axis": [1.0, 0.0]},
            "obstacle": {"center": [0.70, 0.10]},
        }
        self.subject._task_scene_source = "planner_scene"
        self.subject._direct_optitrack_visualization = mock.Mock(
            side_effect=AssertionError("active task must keep its frozen scene")
        )
        self.subject._demo_visualization = mock.Mock(
            return_value={
                "current_ee": None,
                "scene_geometry": {
                    "bar": {"pivot": [9.0, 9.0], "axis": [0.0, 1.0]},
                    "obstacle": {"center": [9.0, 9.0]},
                },
                "source": "optitrack",
            }
        )

        visualization = self.subject.task_visualization()

        self.assertEqual(visualization["source"], "planner_scene")
        self.assertEqual(
            visualization["scene_geometry"]["bar"]["pivot"], [0.50, -0.20]
        )

    def test_real_execution_features_use_actual_tf_and_planner_scene(self):
        self.subject._task_state.update(
            {"task_id": "BarClean", "mode": "real", "phase": "executing"}
        )
        self.subject._reset_task_visualization("BarClean")
        self.subject._task_scene_geometry = {
            "bar": {"pivot": [0.50, -0.20], "axis": [1.0, 0.0]},
            "obstacle": {"center": [0.70, 0.10], "radius": 0.05},
        }
        self.subject._task_scene_source = "planner_scene"
        actual_pose = {
            "x": 0.60,
            "y": -0.18,
            "z": 0.19584,
            "qx": 0.0,
            "qy": 1.0,
            "qz": 0.0,
            "qw": 0.0,
        }
        self.subject._robot_ee_pose = mock.Mock(return_value=actual_pose)
        self.subject._demo_visualization = mock.Mock(
            return_value={
                "current_ee": actual_pose,
                "scene_geometry": {
                    "bar": {"pivot": [9.0, 9.0], "axis": [0.0, 1.0]},
                    "obstacle": {"center": [9.0, 9.0], "radius": 0.05},
                },
                "source": "optitrack",
            }
        )

        self.subject._update_real_visualization(sample_time=10.0)
        self.subject._update_real_visualization(sample_time=10.1)
        visualization = self.subject.task_visualization()

        series = visualization["feature_series"]
        self.assertEqual(series["source"], "real_tf/BarClean")
        self.assertEqual(len(series["samples"]), 2)
        self.assertEqual(len(series["schema"]), 7)
        definition = self.subject._task_feature_definitions["BarClean"]
        expected_surface = sum(
            (actual_pose[axis] - definition["table_surface_point"][index])
            * definition["table_normal"][index]
            for index, axis in enumerate(("x", "y", "z"))
        )
        self.assertAlmostEqual(series["samples"][0][2], expected_surface)
        self.assertAlmostEqual(series["samples"][0][3], 0.02)
        self.assertAlmostEqual(abs(series["samples"][0][6]), math.pi)
        self.assertAlmostEqual(
            series["samples"][0][7],
            0.10 - definition["true_constraints"]["bar_axial_offset_reference"],
        )
        self.assertEqual(visualization["source"], "planner_scene")
        self.assertEqual(visualization["scene_geometry"]["bar"]["pivot"], [0.50, -0.20])

    def test_bar_clean_feature_targets_come_from_planner_config(self):
        self.subject._reset_task_visualization("BarClean")

        surface_specs = [
            spec
            for spec in self.subject._task_feature_series["constraint_specs"]
            if spec["feature_name"] == "surface_dist" and spec["stage"] == 1
        ]

        self.assertEqual(len(surface_specs), 1)
        self.assertEqual(surface_specs[0]["value"], 0.10204)


if __name__ == "__main__":
    unittest.main()
