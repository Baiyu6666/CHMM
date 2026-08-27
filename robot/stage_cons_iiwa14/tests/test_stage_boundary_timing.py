import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROS_SOURCE = Path(__file__).resolve().parents[1] / "ros_ws" / "src"
sys.path.insert(
    0,
    str(ROS_SOURCE / "stage_cartesian_trajectory" / "src"),
)

from stage_cartesian_trajectory import CartesianTrajectoryCompiler  # noqa: E402


class StageBoundaryTimingTest(unittest.TestCase):
    @staticmethod
    def compiler():
        compiler = object.__new__(CartesianTrajectoryCompiler)
        compiler._dof = 2
        compiler._velocity_limits = np.asarray([2.0, 1.5])
        compiler._velocity_scale = 0.25
        compiler._acceleration_limit = 1.0
        compiler._first_point_delay = 0.5
        compiler._lower = np.full(2, -2.0)
        compiler._upper = np.full(2, 2.0)
        return compiler

    @staticmethod
    def stage_timing():
        return {
            "boundaries": [4, 10],
            "transition_windows": [[4, 7]],
            "speed_scale": 0.5,
            "ramp_before_m": 0.02,
            "task_start_ramp_m": 0.03,
        }

    def test_speed_envelope_slows_task_start_and_stage_boundary(self):
        positions = np.column_stack(
            (np.linspace(0.0, 0.10, 11), np.zeros((11, 2)))
        )

        scales = self.compiler().task_segment_speed_scales(
            positions, self.stage_timing()
        )

        self.assertEqual(scales.shape, (10,))
        self.assertAlmostEqual(scales[0], 0.5)
        self.assertAlmostEqual(scales[3], 0.5)
        self.assertAlmostEqual(scales[4], 0.5)
        self.assertGreater(scales[5], scales[4])
        self.assertGreater(scales[6], scales[5])
        self.assertAlmostEqual(scales[-1], 1.0)

    def test_time_parameterization_matches_controller_quintic(self):
        compiler = self.compiler()
        q_path = np.asarray(
            [
                [0.00, 0.00],
                [0.10, -0.04],
                [0.03, 0.08],
                [0.16, 0.02],
                [0.20, 0.10],
            ]
        )
        prepared = compiler.time_parameterize(
            q_path,
            minimum_duration=0.5,
            segment_speed_scales=np.asarray([1.0, 0.5, 0.5, 1.0]),
        )

        self.assertLessEqual(
            prepared["maximum_interpolated_velocity_ratio"], 1.001
        )
        self.assertLessEqual(
            prepared["maximum_interpolated_acceleration_ratio"], 1.001
        )
        np.testing.assert_allclose(prepared["velocity"][[0, -1]], 0.0, atol=1e-10)
        np.testing.assert_allclose(
            prepared["acceleration"][[0, -1]], 0.0, atol=1e-9
        )

        spline = prepared["_position_spline"]
        for index, (start, end) in enumerate(
            zip(prepared["time"][:-1], prepared["time"][1:])
        ):
            duration = float(end - start)
            matrix = np.asarray(
                [
                    [duration ** 3, duration ** 4, duration ** 5],
                    [3 * duration ** 2, 4 * duration ** 3, 5 * duration ** 4],
                    [6 * duration, 12 * duration ** 2, 20 * duration ** 3],
                ]
            )
            p0, p1 = prepared["position"][[index, index + 1]]
            v0, v1 = prepared["velocity"][[index, index + 1]]
            a0, a1 = prepared["acceleration"][[index, index + 1]]
            low_coefficients = np.vstack((p0, v0, 0.5 * a0))
            residual = np.vstack(
                (
                    p1 - (p0 + v0 * duration + 0.5 * a0 * duration ** 2),
                    v1 - (v0 + a0 * duration),
                    a1 - a0,
                )
            )
            high_coefficients = np.linalg.solve(matrix, residual)
            local_times = np.linspace(0.0, duration, 9)
            powers = np.vstack([local_times ** order for order in range(6)]).T
            controller_values = powers @ np.vstack(
                (low_coefficients, high_coefficients)
            )
            np.testing.assert_allclose(
                controller_values,
                spline(start + local_times),
                atol=1e-9,
            )

    def test_local_violation_does_not_dilate_the_whole_path(self):
        compiler = self.compiler()
        progress = np.linspace(0.0, 1.0, 41)
        q_path = np.column_stack(
            (0.20 * progress, 0.08 * np.sin(math.pi * progress))
        )
        q_path[20, 1] += 0.08

        prepared = compiler.time_parameterize(q_path, minimum_duration=4.0)
        segment_dt = np.asarray(prepared["segment_duration_s"])

        self.assertLessEqual(
            prepared["maximum_interpolated_acceleration_ratio"], 1.001
        )
        self.assertGreater(float(np.max(segment_dt[16:24])), 1.5 * float(
            np.median(np.concatenate((segment_dt[:10], segment_dt[-10:])))
        ))
        self.assertLess(prepared["duration"], 10.0)

    def test_stage_timing_must_match_path(self):
        positions = np.column_stack(
            (np.linspace(0.0, 0.10, 11), np.zeros((11, 2)))
        )
        invalid = self.stage_timing()
        invalid["boundaries"] = [4, 9]

        with self.assertRaisesRegex(Exception, "boundaries do not match"):
            self.compiler().task_segment_speed_scales(positions, invalid)


if __name__ == "__main__":
    unittest.main()
