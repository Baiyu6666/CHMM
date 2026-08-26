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


def test_barclean_learning_definition_keeps_stage_three_free_and_constrains_stage_four():
    env = BarCleanEnv()
    by_stage = {
        stage: [
            spec for spec in env.get_constraint_specs() if spec["stage"] == stage
        ]
        for stage in range(5)
    }

    assert by_stage[2] == []
    assert {spec["feature_name"] for spec in by_stage[3]} == {
        "bar_axial_offset",
        "surface_dist",
        "tool_pitch",
        "tool_plane_err",
        "tool_yaw",
    }
    assert all(spec["semantics"] == "target_value" for spec in by_stage[3])


def test_barclean_generated_stage_four_matches_its_orientation_targets():
    env = BarCleanEnv()
    trajectory, labels, _ = env.generate_demo(seed=7)
    stage_four_features = env.compute_all_features_matrix(
        trajectory[labels == 3],
        feat_ids=["tool_pitch", "tool_plane_err", "tool_yaw"],
    )

    expected = np.asarray([np.deg2rad(90.0), 0.0, np.deg2rad(-45.0)])
    assert np.allclose(np.mean(stage_four_features, axis=0), expected, atol=0.03)
