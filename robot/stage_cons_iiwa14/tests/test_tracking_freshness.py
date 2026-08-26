import ast
import types
import unittest
from pathlib import Path

import numpy as np


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ros_ws/src/stage_constraint_planner/scripts/task_planner.py"
)


def load_subject():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    planner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TaskPlannerNode"
    )
    method = next(
        node
        for node in planner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_tracking_reason"
    )
    subject_class = ast.ClassDef(
        name="TrackingSubject",
        bases=[],
        keywords=[],
        body=[method],
        decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[subject_class], type_ignores=[]))
    namespace = {"np": np}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["TrackingSubject"]


class TrackingFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.subject = load_subject()()
        self.subject._scene_future_tolerance = 0.1
        self.subject._scene_max_age = 1.0
        self.subject._scene_stability_window = 0.15
        self.subject._scene_max_speed = 2.0
        self.subject._scene_max_jump = 0.10

    @staticmethod
    def _message(stamp):
        return types.SimpleNamespace(
            header=types.SimpleNamespace(
                stamp=types.SimpleNamespace(to_sec=lambda: stamp)
            )
        )

    def test_replayed_source_timestamp_is_stale_even_with_fresh_callback(self):
        reason = self.subject._tracking_reason(
            "bar",
            self._message(8.5),
            10.0,
            [(10.0, [0.6, 0.0, 0.2])],
            10.0,
        )

        self.assertEqual(reason, "bar pose source timestamp is stale")

    def test_large_tracking_jump_is_rejected(self):
        reason = self.subject._tracking_reason(
            "obstacle",
            self._message(10.0),
            10.0,
            [
                (9.95, [0.6, 0.0, 0.2]),
                (10.0, [0.75, 0.0, 0.2]),
            ],
            10.0,
        )

        self.assertEqual(reason, "obstacle tracking jumped 0.150 m")

    def test_stable_fresh_tracking_is_accepted(self):
        reason = self.subject._tracking_reason(
            "bar",
            self._message(10.0),
            10.0,
            [
                (9.95, [0.6, 0.0, 0.2]),
                (10.0, [0.601, 0.0, 0.2]),
            ],
            10.0,
        )

        self.assertIsNone(reason)


if __name__ == "__main__":
    unittest.main()
