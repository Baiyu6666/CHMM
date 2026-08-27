import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ros_ws/src/stage_optitrack/scripts/fixed_pose_publisher.py"
)


def load_module():
    rospy = types.ModuleType("rospy")
    geometry_msgs = types.ModuleType("geometry_msgs")
    geometry_msgs_msg = types.ModuleType("geometry_msgs.msg")
    geometry_msgs_msg.PoseStamped = type("PoseStamped", (), {})
    sys.modules.update(
        {
            "rospy": rospy,
            "geometry_msgs": geometry_msgs,
            "geometry_msgs.msg": geometry_msgs_msg,
        }
    )
    spec = importlib.util.spec_from_file_location("fixed_pose_publisher_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FixedScenePublisherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_legacy_topic_transform_round_trips_robot_frame_pose(self):
        tracker_to_robot = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        robot_to_tracker = self.module._transpose(tracker_to_robot)
        position_robot = [0.64223939, -0.08751724, 0.12537]
        yaw = math.radians(82.2)
        quaternion_robot = [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]
        rotation_robot = self.module._quaternion_to_matrix(quaternion_robot)

        position_topic = self.module._matrix_vector(
            robot_to_tracker, position_robot
        )
        rotation_topic = self.module._matrix_multiply(
            robot_to_tracker, rotation_robot
        )
        quaternion_topic = self.module._matrix_to_quaternion(rotation_topic)
        round_trip_position = self.module._matrix_vector(
            tracker_to_robot, position_topic
        )
        round_trip_rotation = self.module._matrix_multiply(
            tracker_to_robot,
            self.module._quaternion_to_matrix(quaternion_topic),
        )

        for actual, expected in zip(round_trip_position, position_robot):
            self.assertAlmostEqual(actual, expected, places=9)
        for actual_row, expected_row in zip(round_trip_rotation, rotation_robot):
            for actual, expected in zip(actual_row, expected_row):
                self.assertAlmostEqual(actual, expected, places=9)

    def test_matrix_to_quaternion_handles_identity_obstacle_orientation(self):
        tracker_to_robot = [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        rotation_topic = self.module._transpose(tracker_to_robot)
        quaternion_topic = self.module._matrix_to_quaternion(rotation_topic)
        restored = self.module._matrix_multiply(
            tracker_to_robot,
            self.module._quaternion_to_matrix(quaternion_topic),
        )

        for row, expected_row in zip(restored, self.module._quaternion_to_matrix([0, 0, 0, 1])):
            for actual, expected in zip(row, expected_row):
                self.assertAlmostEqual(actual, expected, places=9)


if __name__ == "__main__":
    unittest.main()
