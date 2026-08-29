import math
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROS_SOURCE = Path(__file__).resolve().parents[1] / "ros_ws" / "src"
sys.path.insert(
    0,
    str(ROS_SOURCE / "stage_cartesian_trajectory" / "src"),
)

from stage_cartesian_trajectory import (  # noqa: E402
    CartesianTrajectoryCompiler,
    TrajectoryValidationError,
)


class StageZeroVerticalRecoveryTest(unittest.TestCase):
    def setUp(self):
        self.compiler = object.__new__(CartesianTrajectoryCompiler)
        self.compiler._minimum_approach_z = 0.20
        self.compiler._approach_clearance_z = 0.33
        self.compiler._approach_spacing = 0.01
        self.compiler._approach_axis_spacing = math.radians(2.0)
        self.compiler._position_tolerance = 0.002
        self.compiler._approach_position_tolerance = 0.005
        self.compiler._max_joint_step = 0.15
        self.compiler._approach_speed = 0.06

    @staticmethod
    def obstacle():
        return {
            "type": "circle",
            "center": [0.0, 0.0, 0.10],
            "table_normal": [0.0, 0.0, 1.0],
            "radius": 0.025,
            "clearance": 0.085,
            "margin": 0.005,
        }

    @staticmethod
    def capsule_obstacle():
        return {
            "type": "capsule",
            "endpoints": [[-0.10, 0.0, 0.10], [0.10, 0.0, 0.10]],
            "table_normal": [0.0, 0.0, 1.0],
            "radius": 0.025,
            "clearance": 0.085,
            "margin": 0.005,
        }

    def test_capsule_allows_target_inside_old_enclosing_circle(self):
        current = np.asarray([0.0, 0.30, 0.20])
        target = np.asarray([0.0, 0.15, 0.25])
        tool_z = np.asarray([0.0, 0.0, -1.0])

        positions, _axes, _x_axes, _distance = self.compiler._approach_samples(
            current,
            tool_z,
            target,
            tool_z,
            approach_obstacle=self.capsule_obstacle(),
        )

        np.testing.assert_allclose(positions[-1], target, atol=1e-12)
        distances = self.compiler._point_segment_distances(
            positions[:, :2], np.asarray([-0.10, 0.0]), np.asarray([0.10, 0.0])
        )
        self.assertGreaterEqual(float(np.min(distances)), 0.110 - 1e-6)
        # The former enclosing-circle approximation required 0.210 m here.
        self.assertLess(float(np.linalg.norm(target[:2])), 0.210)

    def test_capsule_detour_stays_outside_exact_clearance(self):
        current = np.asarray([0.0, -0.30, 0.20])
        target = np.asarray([0.0, 0.15, 0.25])
        tool_z = np.asarray([0.0, 0.0, -1.0])

        positions, _axes, _x_axes, _distance = self.compiler._approach_samples(
            current,
            tool_z,
            target,
            tool_z,
            approach_obstacle=self.capsule_obstacle(),
        )

        full_path = np.vstack((current, positions))
        distances = self.compiler._point_segment_distances(
            full_path[:, :2], np.asarray([-0.10, 0.0]), np.asarray([0.10, 0.0])
        )
        self.assertGreaterEqual(float(np.min(distances)), 0.110 - 1e-6)
        self.assertGreater(float(np.max(np.abs(full_path[:, 0]))), 0.20)

    def test_capsule_rejects_target_inside_exact_clearance(self):
        tool_z = np.asarray([0.0, 0.0, -1.0])
        with self.assertRaisesRegex(
            TrajectoryValidationError,
            "Task start violates Stage-0 obstacle clearance",
        ):
            self.compiler._approach_samples(
                np.asarray([0.0, 0.30, 0.20]),
                tool_z,
                np.asarray([0.0, 0.08, 0.25]),
                tool_z,
                approach_obstacle=self.capsule_obstacle(),
            )

    def test_task_approach_detours_around_obstacle_without_lifting(self):
        current = np.asarray([-0.30, 0.0, 0.18])
        target = np.asarray([0.30, 0.0, 0.25])
        tool_z = np.asarray([0.0, 0.0, -1.0])

        positions, _axes, _x_axes, _distance = self.compiler._approach_samples(
            current,
            tool_z,
            target,
            tool_z,
            approach_obstacle=self.obstacle(),
        )

        radial = np.linalg.norm(positions[:, :2], axis=1)
        self.assertGreaterEqual(float(np.min(radial)), 0.110 - 1e-6)
        self.assertGreater(float(np.max(np.abs(positions[:, 1]))), 0.10)
        self.assertLessEqual(float(np.max(positions[:, 2])), target[2] + 1e-9)
        self.assertGreaterEqual(float(np.min(positions[:, 2])), current[2] - 1e-9)

    def test_task_obstacle_approach_preserves_safe_boundary_and_spacing(self):
        current = np.asarray([-0.30, 0.0, 0.18])
        target = np.asarray([0.30, 0.0, 0.25])
        tool_z = np.asarray([0.0, 0.0, -1.0])

        positions, _axes, _x_axes, distance = self.compiler._approach_samples(
            current,
            tool_z,
            target,
            tool_z,
            approach_obstacle=self.obstacle(),
        )

        edge_lengths = np.linalg.norm(
            np.diff(np.vstack((current, positions)), axis=0), axis=1
        )
        self.assertGreaterEqual(len(positions), int(math.ceil(distance / 0.01)))
        self.assertLessEqual(float(np.max(edge_lengths)), 0.01 + 1e-9)

    def test_task_approach_keeps_safe_straight_path_and_does_not_lift(self):
        current = np.asarray([-0.30, 0.30, 0.18])
        target = np.asarray([0.30, 0.30, 0.25])
        tool_z = np.asarray([0.0, 0.0, -1.0])

        positions, _axes, _x_axes, _distance = self.compiler._approach_samples(
            current,
            tool_z,
            target,
            tool_z,
            approach_obstacle=self.obstacle(),
        )

        np.testing.assert_allclose(positions[:, 1], 0.30, atol=1e-12)
        self.assertLessEqual(float(np.max(positions[:, 2])), target[2] + 1e-9)

    def test_task_approach_detours_around_circle(self):
        current = np.asarray([-0.35, 0.0, 0.20])
        target = np.asarray([0.35, 0.0, 0.25])
        tool_z = np.asarray([0.0, 0.0, -1.0])
        obstacle = self.obstacle()

        positions, _axes, _x_axes, _distance = self.compiler._approach_samples(
            current,
            tool_z,
            target,
            tool_z,
            approach_obstacle=obstacle,
        )

        radial = np.linalg.norm(positions[:, :2], axis=1)
        self.assertGreaterEqual(float(np.min(radial)), 0.110 - 1e-6)

    def test_task_approach_allows_only_monotone_egress_from_clearance_envelope(self):
        current = np.asarray([0.0, 0.08, 0.20])
        target = np.asarray([0.0, 0.30, 0.25])
        tool_z = np.asarray([0.0, 0.0, -1.0])

        positions, _axes, _x_axes, _distance = self.compiler._approach_samples(
            current,
            tool_z,
            target,
            tool_z,
            approach_obstacle=self.obstacle(),
        )

        radial = np.linalg.norm(np.vstack((current, positions))[:, :2], axis=1)
        first_safe = int(np.flatnonzero(radial >= 0.110 - 1e-6)[0])
        self.assertTrue(np.all(np.diff(radial[: first_safe + 1]) >= -1e-6))
        self.assertGreaterEqual(float(np.min(radial[first_safe:])), 0.110 - 1e-6)

    def test_task_approach_rejects_start_target_inside_clearance(self):
        tool_z = np.asarray([0.0, 0.0, -1.0])
        with self.assertRaisesRegex(
            TrajectoryValidationError, "Task start violates Stage-0 obstacle clearance"
        ):
            self.compiler._approach_samples(
                np.asarray([-0.30, 0.0, 0.18]),
                tool_z,
                np.asarray([0.0, 0.05, 0.25]),
                tool_z,
                approach_obstacle=self.obstacle(),
            )

    def test_task_approach_candidates_include_phased_orientation_profiles(self):
        current = np.asarray([-0.30, 0.30, 0.18])
        target = np.asarray([0.30, 0.30, 0.25])
        current_z = np.asarray([0.0, 0.0, 1.0])
        target_z = np.asarray([0.0, 1.0, 0.0])
        tool_x = np.asarray([1.0, 0.0, 0.0])

        candidates = self.compiler._approach_sample_candidates(
            current,
            current_z,
            target,
            target_z,
            current_x=tool_x,
            target_x=tool_x,
            approach_obstacle=self.obstacle(),
        )

        labels = [candidate[0] for candidate in candidates]
        self.assertEqual(labels[:3], [
            "blended",
            "translate_then_orient",
            "orient_then_translate",
        ])
        for _label, positions, axes, x_axes, _distance in candidates:
            np.testing.assert_allclose(positions[-1], target, atol=1e-12)
            np.testing.assert_allclose(axes[-1], target_z, atol=1e-12)
            np.testing.assert_allclose(x_axes[-1], tool_x, atol=1e-12)

    def test_task_approach_candidates_include_opposite_obstacle_arc(self):
        current = np.asarray([-0.30, 0.0, 0.18])
        target = np.asarray([0.30, 0.0, 0.25])
        tool_z = np.asarray([0.0, 0.0, -1.0])

        candidates = self.compiler._approach_sample_candidates(
            current,
            tool_z,
            target,
            tool_z,
            approach_obstacle=self.obstacle(),
        )

        labels = [candidate[0] for candidate in candidates]
        self.assertIn("alternate_obstacle_arc", labels)
        primary = next(item for item in candidates if item[0] == "blended")
        alternate = next(
            item for item in candidates
            if item[0] == "alternate_obstacle_arc"
        )
        self.assertFalse(np.allclose(primary[1], alternate[1]))
        primary_peak = primary[1][np.argmax(np.abs(primary[1][:, 1])), 1]
        alternate_peak = alternate[1][np.argmax(np.abs(alternate[1][:, 1])), 1]
        self.assertLess(float(primary_peak * alternate_peak), 0.0)

    def test_compile_retries_next_safe_approach_candidate(self):
        compiler = object.__new__(CartesianTrajectoryCompiler)
        compiler._dof = 2
        compiler.tip_state = lambda _q: (np.zeros(3), np.eye(3))
        sample = (
            np.asarray([[0.0, 0.0, 0.3]]),
            np.asarray([[0.0, 0.0, 1.0]]),
            None,
            0.1,
        )
        compiler._approach_sample_candidates = mock.Mock(
            return_value=[
                ("blended", *sample),
                ("translate_then_orient", *sample),
            ]
        )
        expected = {"metrics": {"approach_route": "translate_then_orient"}}
        compiler._compile_cartesian_candidate = mock.Mock(
            side_effect=[
                TrajectoryValidationError("primary approach failed"),
                expected,
            ]
        )
        compiler._set_q = mock.Mock()

        positions = np.asarray(
            [[0.0, 0.0, 0.3], [0.1, 0.0, 0.3]]
        )
        axes = np.repeat(
            np.asarray([[0.0, 0.0, 1.0]]), 2, axis=0
        )
        result = compiler.compile(positions, axes, np.zeros(2))

        self.assertIs(result, expected)
        self.assertEqual(compiler._compile_cartesian_candidate.call_count, 2)

    def test_continuous_ik_uses_deterministic_redundancy_multistart(self):
        compiler = object.__new__(CartesianTrajectoryCompiler)
        compiler._dof = 2
        compiler._lower = np.full(2, -2.0)
        compiler._upper = np.full(2, 2.0)
        compiler._position_tolerance = 0.002
        compiler._tool_z_tolerance = math.radians(2.0)
        compiler._tool_x_tolerance = math.radians(2.0)
        compiler._max_joint_step = 0.15
        compiler._robot = 1
        compiler._tip_index = 2
        compiler._physics = 3
        compiler._set_q = lambda _q: None
        compiler._self_collision_pair = lambda: None

        bullet = mock.Mock()
        bullet.calculateInverseKinematics.side_effect = (
            lambda *_args, **_kwargs: (
                [1.0, 1.0]
                if bullet.calculateInverseKinematics.call_count <= 8
                else [0.0, 0.0]
            )
        )
        compiler._bullet = bullet
        compiler.tip_state = lambda q: (
            (
                np.zeros(3)
                if np.allclose(q, 0.0)
                else np.asarray([1.0, 0.0, 0.0])
            ),
            np.eye(3),
        )

        trajectory, *_errors = compiler._continuous_ik(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 1.0]]),
            np.zeros(2),
            lambda: False,
            "Approach",
        )

        np.testing.assert_allclose(trajectory, [[0.0, 0.0]])
        self.assertEqual(bullet.calculateInverseKinematics.call_count, 9)

    def test_task_obstacle_approach_does_not_enforce_legacy_transit_floor(self):
        self.compiler.tip_state = lambda q: (np.asarray(q), np.eye(3))
        path = np.asarray(
            [
                [-0.30, 0.0, 0.18],
                [-0.20, 0.12, 0.18],
                [0.20, 0.12, 0.19],
            ]
        )

        minimum = self.compiler._approach_workspace_checks(
            path, enforce_transit_floor=False
        )

        self.assertAlmostEqual(minimum, 0.18)

    def test_approach_joint_interpolation_keeps_obstacle_clearance(self):
        self.compiler.tip_state = lambda q: (np.asarray(q), np.eye(3))
        safe_path = np.asarray(
            [
                [-0.30, 0.20, 0.18],
                [0.00, 0.20, 0.20],
                [0.30, 0.20, 0.25],
            ]
        )
        unsafe_path = safe_path.copy()
        unsafe_path[1, 1] = 0.05

        minimum = self.compiler._approach_obstacle_checks(
            safe_path, self.obstacle()
        )
        self.assertAlmostEqual(minimum, 0.175)
        with self.assertRaisesRegex(
            TrajectoryValidationError,
            "joint interpolation violates obstacle clearance",
        ):
            self.compiler._approach_obstacle_checks(
                unsafe_path, self.obstacle()
            )

    def test_low_start_is_lifted_before_any_lateral_motion(self):
        current = np.asarray([0.77, -0.26, 0.1962])
        target = np.asarray([0.73, 0.22, 0.2658])
        tool_z = np.asarray([0.0, 0.0, -1.0])

        positions, _axes, _x_axes, _distance = self.compiler._approach_samples(
            current, tool_z, target, tool_z
        )

        lateral = np.linalg.norm(positions[:, :2] - current[None, :2], axis=1)
        first_lateral = int(np.flatnonzero(lateral > 1e-9)[0])
        np.testing.assert_allclose(
            positions[:first_lateral, :2],
            np.repeat(current[None, :2], first_lateral, axis=0),
            atol=1e-12,
        )
        self.assertAlmostEqual(float(positions[first_lateral - 1, 2]), 0.33)

    def test_home_vertical_recovery_preserves_xy_and_full_orientation(self):
        start_q = np.zeros(7)
        rotation = np.asarray(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        self.compiler.tip_state = lambda _q: (
            np.asarray([0.70, -0.15, 0.10]),
            rotation,
        )

        def continuous_ik(positions, _axes, *_args, **_kwargs):
            return (
                np.repeat(start_q[None, :], len(positions), axis=0),
                [0.0] * len(positions),
                [0.0] * len(positions),
                [0.0] * len(positions),
            )

        self.compiler._continuous_ik = mock.Mock(side_effect=continuous_ik)
        self.compiler._densify_joint_path = lambda path, _step: path
        self.compiler.time_parameterize = mock.Mock(
            side_effect=lambda path, duration: {
                "position": path,
                "duration": duration,
            }
        )

        recovery, recovered_q, distance, *_errors = (
            self.compiler._home_vertical_recovery(start_q, lambda: False)
        )

        call = self.compiler._continuous_ik.call_args
        positions, axes = call.args[:2]
        np.testing.assert_allclose(
            positions[:, :2],
            np.repeat([[0.70, -0.15]], len(positions), axis=0),
        )
        self.assertAlmostEqual(float(positions[-1, 2]), 0.202)
        np.testing.assert_allclose(
            axes, np.repeat(rotation[None, :, 2], len(positions), axis=0)
        )
        np.testing.assert_allclose(
            call.kwargs["x_axes"],
            np.repeat(rotation[None, :, 0], len(positions), axis=0),
        )
        self.assertTrue(np.all(call.kwargs["x_active"]))
        self.assertAlmostEqual(distance, 0.102)
        np.testing.assert_allclose(recovered_q, start_q)
        self.assertIsNotNone(recovery)

    def test_home_vertical_recovery_is_skipped_at_safe_height(self):
        start_q = np.zeros(7)
        self.compiler.tip_state = lambda _q: (
            np.asarray([0.70, -0.15, 0.20]),
            np.eye(3),
        )
        self.compiler._continuous_ik = mock.Mock()

        recovery, recovered_q, distance, *_errors = (
            self.compiler._home_vertical_recovery(start_q, lambda: False)
        )

        self.assertIsNone(recovery)
        np.testing.assert_allclose(recovered_q, start_q)
        self.assertEqual(distance, 0.0)
        self.compiler._continuous_ik.assert_not_called()

    def test_joint_home_checks_the_complete_path_against_circle_obstacle(self):
        self.compiler._dof = 7
        self.compiler._lower = np.full(7, -2.0)
        self.compiler._upper = np.full(7, 2.0)
        self.compiler._set_q = mock.Mock()
        self.compiler._home_vertical_recovery = mock.Mock(
            return_value=(None, np.zeros(7), 0.0, [], [], [])
        )
        self.compiler._densify_joint_path = lambda path, _step: np.asarray(path)
        self.compiler.tip_state = lambda q: (
            np.asarray([float(q[0]), 0.0, 0.25]),
            np.eye(3),
        )
        self.compiler.time_parameterize = lambda path, _duration: {
            "position": np.asarray(path),
            "_validation_position": np.asarray(path),
            "duration": 1.0,
            "maximum_interpolated_velocity_ratio": 0.1,
            "maximum_interpolated_acceleration_ratio": 0.1,
            "maximum_interpolated_velocity_rad_s": 0.1,
            "maximum_interpolated_acceleration_rad_s2": 0.1,
        }
        self.compiler._approach_workspace_checks = mock.Mock(return_value=0.25)
        self.compiler._collision_checks = mock.Mock()
        self.compiler._approach_obstacle_checks = mock.Mock(return_value=0.12)
        obstacle = self.obstacle()

        result = self.compiler.compile_joint_home(
            np.zeros(7),
            np.full(7, 0.1),
            approach_obstacle=obstacle,
        )

        checked_path, checked_obstacle = self.compiler._approach_obstacle_checks.call_args.args
        self.assertEqual(checked_path.shape, (2, 7))
        self.assertIs(checked_obstacle, obstacle)
        self.assertEqual(result["metrics"]["minimum_home_obstacle_clearance_m"], 0.12)

    def test_workspace_check_accepts_vertical_recovery_from_below_floor(self):
        self.compiler.tip_state = lambda q: (np.asarray(q), np.eye(3))
        path = np.asarray(
            [
                [0.0, 0.0, 0.1962],
                [0.0, 0.0, 0.2050],
                [0.0, 0.0, 0.3300],
                [0.10, 0.0, 0.3300],
            ]
        )

        minimum = self.compiler._approach_workspace_checks(path)

        self.assertAlmostEqual(minimum, 0.1962)

    def test_workspace_check_rejects_low_lateral_sweep(self):
        self.compiler.tip_state = lambda q: (np.asarray(q), np.eye(3))
        path = np.asarray(
            [
                [0.0, 0.0, 0.1962],
                [0.02, 0.0, 0.1980],
                [0.02, 0.0, 0.2050],
            ]
        )

        with self.assertRaisesRegex(TrajectoryValidationError, "moves TCP laterally"):
            self.compiler._approach_workspace_checks(path)

    def test_compile_enforces_full_6d_start_and_goal_while_masking_interior_yaw(self):
        compiler = object.__new__(CartesianTrajectoryCompiler)
        compiler._dof = 2
        compiler._lower = np.full(2, -2.0)
        compiler._upper = np.full(2, 2.0)
        compiler._max_joint_step = 0.15
        compiler._position_tolerance = 0.002
        compiler._approach_position_tolerance = 0.005
        compiler._approach_joint_bridge_limit = 3.0
        compiler._tool_z_tolerance = math.radians(2.0)
        compiler._tool_x_tolerance = math.radians(2.0)
        compiler._approach_speed = 0.06
        compiler._task_speed = 0.04
        compiler.tip_state = lambda _q: (np.asarray([0.0, 0.0, 0.3]), np.eye(3))
        compiler._approach_samples = mock.Mock(
            return_value=(
                np.asarray([[0.0, 0.0, 0.3]]),
                np.asarray([[0.0, 0.0, 1.0]]),
                np.asarray([[1.0, 0.0, 0.0]]),
                0.1,
            )
        )
        ik_calls = []

        def continuous_ik(positions, _axes, seed, _abort, _phase, **kwargs):
            ik_calls.append(kwargs)
            return (
                np.repeat(np.asarray(seed)[None, :], len(positions), axis=0),
                [0.0] * len(positions),
                [0.0] * len(positions),
                [0.0] * len(positions),
            )

        compiler._continuous_ik = continuous_ik
        compiler._approach_workspace_checks = lambda _path, **_kwargs: 0.3
        compiler._collision_and_singularity_checks = lambda _path, _name: 1.0
        compiler.time_parameterize = lambda path, _duration, **_kwargs: {
            "position": path,
            "duration": 1.0,
            "_validation_position": path,
            "maximum_interpolated_velocity_ratio": 0.5,
            "maximum_interpolated_acceleration_ratio": 0.5,
            "maximum_interpolated_velocity_rad_s": 0.1,
            "maximum_interpolated_acceleration_rad_s2": 0.2,
            "timing_overhead_s": 0.0,
            "timing_iterations": 0,
        }
        compiler._set_q = lambda _q: None

        positions = np.asarray(
            [[0.0, 0.0, 0.3], [0.1, 0.0, 0.3], [0.2, 0.0, 0.3]]
        )
        axes = np.repeat(np.asarray([[0.0, 0.0, 1.0]]), 3, axis=0)
        x_axes = np.repeat(np.asarray([[1.0, 0.0, 0.0]]), 3, axis=0)
        mask = np.asarray([False, True, False])

        compiler.compile(
            positions,
            axes,
            np.zeros(2),
            tool_x_axes=x_axes,
            tool_x_active=mask,
        )

        np.testing.assert_array_equal(
            compiler._approach_samples.call_args.kwargs["current_x"],
            [1.0, 0.0, 0.0],
        )
        np.testing.assert_array_equal(
            compiler._approach_samples.call_args.kwargs["target_x"],
            [1.0, 0.0, 0.0],
        )
        np.testing.assert_array_equal(ik_calls[0]["x_active"], [True])
        np.testing.assert_array_equal(ik_calls[1]["x_active"], [True, True])


if __name__ == "__main__":
    unittest.main()
