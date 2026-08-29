import json

import numpy as np

from experiments.process_bar_clean_demos import process_bar_clean_archive


def test_process_barclean_maps_reference_cutpoint_times_to_new_rate(tmp_path):
    source_path = tmp_path / "source_20hz.npz"
    reference_path = tmp_path / "reference_5hz.npz"
    output_path = tmp_path / "output_10hz.npz"
    samples_per_demo = 21
    demo_count = 2
    source_bounds = np.asarray([[0, 21], [21, 42]], dtype=np.int64)
    local_time = np.arange(samples_per_demo, dtype=float) / 20.0
    trajectory = np.tile(
        np.asarray([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]),
        (samples_per_demo * demo_count, 1),
    )
    poses = np.tile(
        np.asarray([0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
        (demo_count, 1),
    )
    np.savez(
        source_path,
        flange_pose=trajectory,
        demo_time_s=np.tile(local_time, demo_count),
        demo_bounds_indices=source_bounds,
        demo_bar_poses_optitrack=poses,
        demo_obstacle_poses_optitrack=poses,
        sampling_hz=np.asarray(20.0),
        optitrack_to_robot_rotation=np.eye(3),
        optitrack_to_robot_translation=np.zeros(3),
        scene_config_json=np.asarray(
            json.dumps(
                {
                    "bar": {"lateral_centerline": {"type": "straight"}},
                    "obstacles": [
                        {
                            "name": "sceneA_obstacle",
                            "locked_pose_robot": [0.0, 0.0, 0.13, 0, 0, 0, 1],
                        },
                        {
                            "name": "sceneB_obstacle",
                            "locked_pose_robot": [0.3, 0.0, 0.13, 0, 0, 0, 1],
                        },
                    ],
                    "planning_obstacle": {
                        "type": "circle",
                        "obstacle": "sceneA_obstacle",
                    },
                }
            )
        ),
    )
    reference_timestamps = np.tile(np.arange(6, dtype=float) / 5.0, demo_count)
    reference_bounds = np.asarray(
        [[0, 1, 2, 3, 4, 6], [6, 7, 8, 9, 10, 12]],
        dtype=np.int64,
    )
    np.savez(
        reference_path,
        timestamps=reference_timestamps,
        coarse_bounds_indices=reference_bounds,
        source_demo_ids=np.arange(demo_count, dtype=np.int64),
        downsample_hz=np.asarray(5.0),
        cutpoint_annotation_kind=np.asarray("5hz_human_reference"),
    )

    summary = process_bar_clean_archive(
        source_path,
        output_path,
        output_hz=10.0,
        annotation_reference_path=reference_path,
    )

    assert summary["points_per_demo"] == [11, 11]
    assert summary["stage_lengths_per_demo"] == [[2, 2, 2, 2, 3]] * 2
    with np.load(output_path, allow_pickle=False) as output:
        assert float(output["downsample_hz"]) == 10.0
        assert output["coarse_bounds_indices"].tolist() == [
            [0, 2, 4, 6, 8, 11],
            [11, 13, 15, 17, 19, 22],
        ]
        assert str(output["cutpoint_annotation_kind"]) == (
            "10hz_time_mapped_from_5hz_human_reference"
        )
        assert str(output["cutpoint_evaluation_role"]) == (
            "external_annotation_reference"
        )
        for row in output["coarse_bounds_indices"]:
            expected = np.repeat(np.arange(5), np.diff(row))
            assert np.array_equal(output["coarse_stage_labels"][row[0] : row[-1]], expected)
