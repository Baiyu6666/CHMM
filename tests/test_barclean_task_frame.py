import numpy as np

from envs.BarClean import BarCleanEnv


def test_barclean_features_freeze_bar_and_obstacle_scene_per_task():
    env = BarCleanEnv(
        optitrack_to_robot_rotation=np.eye(3),
        optitrack_to_robot_translation=np.zeros(3),
    )
    trajectory = np.asarray(
        [
            [0.10, 0.02, 0.20, 0.0, 0.0, 0.0, 1.0],
            [0.10, 0.02, 0.20, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    bar_trace = np.asarray(
        [
            [0.00, 0.00, 0.18, 0.0, 0.0, 0.0, 1.0],
            [0.50, 0.50, 0.18, 0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)],
        ]
    )
    obstacle_trace = np.asarray(
        [
            [0.30, 0.00, 0.18, 0.0, 0.0, 0.0, 1.0],
            [1.00, 1.00, 0.18, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    features = env.compute_all_features_matrix(
        trajectory,
        scene={
            "bar_pose_optitrack": bar_trace,
            "obstacle_pose_optitrack": obstacle_trace,
        },
    )

    assert np.allclose(features[0], features[1])
    assert env.get_observation_spec()["task_frame"]["snapshot_policy"] == "frozen_per_task"
