import numpy as np
import pytest

from envs.BarClean import BarCleanEnv, load_BarClean


def _write_processed_barclean_archive(path):
    env = BarCleanEnv()
    num_demos = 3
    samples_per_demo = 5
    num_samples = num_demos * samples_per_demo
    bounds = np.asarray(
        [
            np.arange(start, start + samples_per_demo + 1)
            for start in range(0, num_samples, samples_per_demo)
        ],
        dtype=int,
    )
    poses = np.tile(
        np.asarray([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
        (num_demos, 1),
    )
    np.savez(
        path,
        trajectory=np.arange(num_samples * 7, dtype=float).reshape(num_samples, 7),
        features=np.zeros((num_samples, len(env.get_feature_schema())), dtype=float),
        feature_names=np.asarray([spec["name"] for spec in env.get_feature_schema()]),
        timestamps=np.arange(num_samples, dtype=float) * 0.2,
        coarse_bounds_indices=bounds,
        demo_bar_poses=poses,
        demo_obstacle_poses=poses,
        source_demo_ids=np.asarray([5, 7, 12], dtype=int),
        downsample_hz=np.asarray(5.0),
        optitrack_to_robot_rotation=np.eye(3),
        optitrack_to_robot_translation=np.zeros(3),
    )


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
    obstacle_column = next(
        spec["column_idx"]
        for spec in env.feature_schema
        if spec["name"] == "obstacle_clearance"
    )
    assert features[0, obstacle_column] == pytest.approx(
        np.hypot(0.10 - 0.30, 0.02) - env.obstacle_radius
    )
    assert env.get_observation_spec()["task_frame"]["snapshot_policy"] == "frozen_per_task"


def test_scenec_curved_centerline_changes_only_lateral_feature():
    centerline = {
        "type": "circular_arc_chord",
        "radius_m": 0.25,
        "axial_bounds_m": [-0.15, 0.15],
        "bulge_sign": 1.0,
    }
    env = BarCleanEnv(
        optitrack_to_robot_rotation=np.eye(3),
        optitrack_to_robot_translation=np.zeros(3),
        obstacle_endpoints=[[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]],
        bar_lateral_centerline=centerline,
    )
    trajectory = np.asarray(
        [
            [-0.15, 0.00, 0.20, 0.0, 0.0, 0.0, 1.0],
            [0.00, 0.05, 0.20, 0.0, 0.0, 0.0, 1.0],
            [0.15, 0.00, 0.20, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    features = env.compute_all_features_matrix(
        trajectory,
        scene={
            "bar_pose_optitrack": [0.0, 0.0, 0.18, 0.0, 0.0, 0.0, 1.0],
            "bar_lateral_centerline": centerline,
        },
    )
    columns = {spec["name"]: spec["column_idx"] for spec in env.feature_schema}

    assert np.allclose(features[:, columns["bar_lateral_offset"]], 0.0, atol=1e-12)
    axial_reference = float(env.task_definition["bar_axial_offset_reference"])
    assert np.allclose(
        features[:, columns["bar_axial_offset"]] + axial_reference,
        [-0.15, 0.0, 0.15],
    )


def test_barclean_obstacle_feature_is_clearance_to_capsule_boundary():
    env = BarCleanEnv(
        optitrack_to_robot_rotation=np.eye(3),
        optitrack_to_robot_translation=np.zeros(3),
        obstacle_endpoints=[[0.0, 0.0, 0.0], [0.30, 0.0, 0.0]],
    )
    trajectory = np.asarray(
        [[0.20, 0.10, 0.20, 0.0, 0.0, 0.0, 1.0]]
    )
    features = env.compute_all_features_matrix(trajectory)
    column = next(
        spec["column_idx"]
        for spec in env.feature_schema
        if spec["name"] == "obstacle_clearance"
    )

    assert features[0, column] == pytest.approx(0.075)


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
        "table_dist",
        "tool_pitch",
        "tool_roll",
        "tool_yaw",
    }
    assert all(spec["semantics"] == "target_value" for spec in by_stage[3])


def test_barclean_generated_stage_four_matches_its_orientation_targets():
    env = BarCleanEnv()
    trajectory, labels, _ = env.generate_demo(seed=7)
    stage_four_features = env.compute_all_features_matrix(
        trajectory[labels == 3],
        feat_ids=["tool_pitch", "tool_roll", "tool_yaw"],
    )

    expected = np.asarray([np.deg2rad(90.0), 0.0, env.discharge_yaw])
    assert np.allclose(np.mean(stage_four_features, axis=0), expected, atol=0.03)


def test_barclean_processed_loader_selects_requested_source_demo_ids(tmp_path):
    archive_path = tmp_path / "barclean_processed.npz"
    _write_processed_barclean_archive(archive_path)

    bundle = load_BarClean(
        processed_demo_path=archive_path,
        n_demos=99,
        source_demo_ids=[12, 7],
    )

    assert bundle.meta["source_demo_ids"] == [12, 7]
    assert [scene["archive_demo_index"] for scene in bundle.meta["scene_specs"]] == [2, 1]
    assert [scene["source_demo_id"] for scene in bundle.meta["scene_specs"]] == [12, 7]
    assert np.array_equal(bundle.demos[0], np.arange(70, 105, dtype=float).reshape(5, 7))
    assert np.array_equal(bundle.demos[1], np.arange(35, 70, dtype=float).reshape(5, 7))


def test_barclean_processed_loader_defaults_to_first_n_demos(tmp_path):
    archive_path = tmp_path / "barclean_processed.npz"
    _write_processed_barclean_archive(archive_path)

    bundle = load_BarClean(processed_demo_path=archive_path, n_demos=2)

    assert bundle.meta["source_demo_ids"] == [5, 7]


def test_barclean_processed_loader_rejects_missing_source_demo_id(tmp_path):
    archive_path = tmp_path / "barclean_processed.npz"
    _write_processed_barclean_archive(archive_path)

    with pytest.raises(ValueError, match=r"available IDs are \[5, 7, 12\]"):
        load_BarClean(processed_demo_path=archive_path, source_demo_ids=[8])


def test_barclean_loader_requires_processed_real_data():
    with pytest.raises(ValueError, match="requires processed_demo_path"):
        load_BarClean()


def test_barclean_loader_uses_archive_frame_and_rate(tmp_path):
    archive_path = tmp_path / "barclean_processed.npz"
    _write_processed_barclean_archive(archive_path)
    conflicting_rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )

    bundle = load_BarClean(
        processed_demo_path=archive_path,
        n_demos=1,
        env_kwargs={
            "dt": 9.0,
            "optitrack_to_robot_rotation": conflicting_rotation,
            "optitrack_to_robot_translation": np.ones(3),
        },
    )

    assert bundle.env.dt == pytest.approx(0.2)
    assert np.array_equal(bundle.env.optitrack_to_robot_rotation, np.eye(3))
    assert np.array_equal(bundle.env.optitrack_to_robot_translation, np.zeros(3))
