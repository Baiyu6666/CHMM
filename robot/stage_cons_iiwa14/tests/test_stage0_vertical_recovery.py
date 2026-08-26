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
        self.compiler._approach_position_tolerance = 0.005

    @staticmethod
    def obstacle():
        return {
            "center": [0.0, 0.0, 0.10],
            "table_normal": [0.0, 0.0, 1.0],
            "radius": 0.025,
            "clearance": 0.085,
            "margin": 0.005,
        }

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

    def test_task_obstacle_approach_uses_one_uniform_sample_grid(self):
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

        self.assertEqual(len(positions), int(math.ceil(distance / 0.01)))
        edge_lengths = np.linalg.norm(
            np.diff(np.vstack((current, positions)), axis=0), axis=1
        )
        self.assertGreater(float(np.min(edge_lengths)), 0.5 * float(np.max(edge_lengths)))

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

    def test_task_approach_rejects_start_target_inside_clearance(self):
        tool_z = np.asarray([0.0, 0.0, -1.0])
        with self.assertRaisesRegex(
            TrajectoryValidationError, "Task start violates Stage-0 obstacle clearance"
        ):
            self.compiler._approach_samples(
                np.asarray([-0.30, 0.0, 0.18]),
                tool_z,
                np.asarray([0.05, 0.0, 0.25]),
                tool_z,
                approach_obstacle=self.obstacle(),
            )

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

    def test_compile_keeps_stage_zero_yaw_free_and_masks_task_yaw(self):
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
        np.testing.assert_array_equal(ik_calls[0]["x_active"], [False])
        np.testing.assert_array_equal(ik_calls[1]["x_active"], [True, False])


if __name__ == "__main__":
    unittest.main()
