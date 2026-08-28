import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.bar_clean_data_pipeline import (
    MatplotlibReviewApp,
    ReviewStore,
    _validate_demo_rows,
    detect_stationary_runs,
    export_reviewed_dataset,
    merge_processed_datasets,
    propose_demo_intervals,
)


def test_detect_stationary_runs_bridges_short_motion_gap():
    speed = np.full(100, 0.5)
    speed[10:35] = 0.01
    speed[20:22] = 0.4
    speed[60:90] = 0.01
    runs = detect_stationary_runs(
        speed,
        10.0,
        threshold_rad_s=0.1,
        minimum_dwell_s=2.0,
        bridge_s=0.3,
    )
    assert runs.tolist() == [[10, 35], [60, 90]]


def test_propose_demo_intervals_rejects_reset_motion():
    hz = 10.0
    dwells = np.asarray([[0, 20], [140, 160], [220, 240]], dtype=int)
    task_xyz = np.zeros((240, 3), dtype=float)
    endpoints = np.asarray(
        [[0.0, 0.0, 0.1], [0.3, 0.0, 0.1], [0.3, 0.1, 0.07], [0.3, -0.1, 0.07]]
    )
    anchors = [25, 55, 85, 115]
    for index, anchor in enumerate(anchors):
        task_xyz[anchor] = endpoints[index]
    task_xyz[20:23, 0] = -0.2
    task_xyz[160:220] = 2.0
    accepted, rejected = propose_demo_intervals(
        dwells,
        task_xyz,
        hz,
        endpoints,
        minimum_duration_s=5.0,
        endpoint_tolerance_m=0.02,
        bar_north_end_axial_m=-0.15,
    )
    assert [row["source_interval_id"] for row in accepted] == [0]
    assert [row["source_interval_id"] for row in rejected] == [1]


def test_propose_demo_intervals_rejects_start_south_of_bar():
    hz = 10.0
    dwells = np.asarray([[0, 20], [100, 120]], dtype=int)
    task_xyz = np.zeros((120, 3), dtype=float)
    endpoints = np.asarray(
        [[0.0, 0.0, 0.1], [0.3, 0.0, 0.1], [0.3, 0.1, 0.07], [0.3, -0.1, 0.07]]
    )
    for index, anchor in enumerate([30, 50, 70, 90]):
        task_xyz[anchor] = endpoints[index]
    task_xyz[20:23, 0] = 0.2

    accepted, rejected = propose_demo_intervals(
        dwells,
        task_xyz,
        hz,
        endpoints,
        minimum_duration_s=5.0,
        endpoint_tolerance_m=0.02,
        bar_north_end_axial_m=-0.15,
    )

    assert accepted == []
    assert rejected[0]["start_north_of_bar"] is False
    assert "start is not north of the bar's north end" in rejected[0]["rejection_reasons"]


def test_review_requires_four_ordered_cutpoints():
    valid = [
        {
            "demo_id": 0,
            "start_index": 10,
            "end_index": 100,
            "cutpoints_local_indices": [10, 30, 50, 70],
        }
    ]
    _validate_demo_rows(valid, 120)
    invalid = [dict(valid[0], cutpoints_local_indices=[10, 30, 30, 70])]
    with pytest.raises(ValueError, match="four ordered"):
        _validate_demo_rows(invalid, 120)


def test_export_requires_both_human_review_gates(tmp_path):
    review = {
        "review": {
            "demo_boundaries_confirmed": True,
            "cutpoints_confirmed": False,
        }
    }
    (tmp_path / "review.json").write_text(json.dumps(review))
    with pytest.raises(ValueError, match="Internal cutpoints"):
        export_reviewed_dataset(tmp_path)


def _write_processed_archive(path, source_ids, marker, tracker_translation=None):
    feature_names = np.asarray(
        [
            "obstacle_clearance",
            "table_dist",
            "bar_lateral_offset",
            "tool_pitch",
            "tool_roll",
            "motion_axis_err",
            "speed",
            "angular_speed",
            "bar_axial_offset",
            "tool_yaw",
        ]
    )
    bounds = []
    trajectories = []
    features = []
    timestamps = []
    labels = []
    demo_ids = []
    offset = 0
    for demo_index, source_id in enumerate(source_ids):
        length = 10
        local_bounds = np.asarray([0, 2, 4, 6, 8, 10], dtype=np.int64)
        bounds.append(local_bounds + offset)
        trajectory = np.zeros((length, 7), dtype=float)
        trajectory[:, 0] = marker + source_id
        trajectory[:, 6] = 1.0
        trajectories.append(trajectory)
        features.append(np.full((length, len(feature_names)), marker + source_id))
        timestamps.append(np.arange(length, dtype=float) / 5.0)
        labels.append(np.repeat(np.arange(5, dtype=np.int64), 2))
        demo_ids.append(np.full(length, demo_index, dtype=np.int64))
        offset += length
    pose = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        timestamps=np.concatenate(timestamps),
        trajectory=np.concatenate(trajectories),
        features=np.concatenate(features),
        feature_names=feature_names,
        demo_id=np.concatenate(demo_ids),
        coarse_stage_labels=np.concatenate(labels),
        coarse_bounds_indices=np.asarray(bounds),
        demo_bar_poses=np.repeat(pose[None, :], len(source_ids), axis=0),
        demo_obstacle_poses=np.repeat(pose[None, :], len(source_ids), axis=0),
        source_demo_ids=np.asarray(source_ids, dtype=np.int64),
        boundary_task_xyz_m=np.zeros((len(source_ids), 6, 3)),
        axial_progress_m=np.zeros((len(source_ids), 5)),
        lateral_progress_m=np.zeros((len(source_ids), 5)),
        source_hz=np.asarray(20.0),
        downsample_hz=np.asarray(5.0),
        downsample_factor=np.asarray(4, dtype=np.int64),
        cutpoint_annotation_kind=np.asarray(
            "5hz_time_mapped_from_human_reviewed_gui_stage_boundaries"
        ),
        cutpoint_evaluation_role=np.asarray("human_reviewed_reference"),
        scene_pose_policy=np.asarray("per_demo_robust_static_lock"),
        optitrack_to_robot_rotation=np.eye(3),
        optitrack_to_robot_translation=np.asarray(
            [0.0, 0.0, 0.0]
            if tracker_translation is None
            else tracker_translation,
            dtype=float,
        ),
        cutpoint_annotation_source_hz=np.asarray(20.0),
    )


def test_merge_processed_datasets_reindexes_bounds_and_preserves_provenance(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    output = tmp_path / "merged" / "training_5hz.npz"
    _write_processed_archive(first, [10, 11], marker=100.0)
    _write_processed_archive(second, [20, 21], marker=200.0)

    summary = merge_processed_datasets(
        [(first, [11]), (second, [20, 21])],
        output,
    )

    assert summary["demo_count"] == 3
    assert [row["source_demo_id"] for row in summary["demo_mapping"]] == [11, 20, 21]
    with np.load(output, allow_pickle=False) as archive:
        assert archive["source_demo_ids"].tolist() == [0, 1, 2]
        assert archive["merge_origin_source_demo_ids"].tolist() == [11, 20, 21]
        assert archive["coarse_bounds_indices"].tolist() == [
            [0, 2, 4, 6, 8, 10],
            [10, 12, 14, 16, 18, 20],
            [20, 22, 24, 26, 28, 30],
        ]
        assert archive["demo_id"].tolist() == [0] * 10 + [1] * 10 + [2] * 10
        assert archive["trajectory"][:, 0].tolist() == (
            [111.0] * 10 + [220.0] * 10 + [221.0] * 10
        )


def test_merge_processed_datasets_rejects_transform_mismatch(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_processed_archive(first, [0], marker=1.0)
    _write_processed_archive(
        second,
        [0],
        marker=2.0,
        tracker_translation=[0.1, 0.0, 0.0],
    )

    with pytest.raises(ValueError, match="transform mismatch"):
        merge_processed_datasets(
            [(first, [0]), (second, [0])],
            tmp_path / "merged.npz",
        )


def test_matplotlib_cutpoint_drag_updates_lines_and_path_marker_live(tmp_path):
    import matplotlib

    matplotlib.use("Agg", force=True)
    point_count = 140
    hz = 10.0
    task_xyz = np.column_stack(
        [np.arange(point_count, dtype=float), np.arange(point_count, dtype=float) * 2, np.zeros(point_count)]
    )
    feature_names = np.asarray(
        [
            "obstacle_clearance",
            "table_dist",
            "bar_lateral_offset",
            "tool_pitch",
            "tool_roll",
            "motion_axis_err",
            "speed",
            "angular_speed",
            "bar_axial_offset",
            "tool_yaw",
        ]
    )
    np.savez_compressed(
        tmp_path / "analysis.npz",
        timestamps_s=np.arange(point_count, dtype=float) / hz,
        joint_speed=np.linspace(0.0, 1.0, point_count),
        dwell_runs=np.asarray([[0, 5], [135, 140]], dtype=int),
        task_xyz=task_xyz,
        features=np.zeros((point_count, len(feature_names))),
        feature_names=feature_names,
    )
    review = {
        "schema_version": 1,
        "task_id": "BarClean",
        "dataset_id": "test",
        "source_bag": "test.bag",
        "analysis_archive": "analysis.npz",
        "analysis_hz": hz,
        "settings": {"stationary_threshold_rad_s": 0.1},
        "quality": {},
        "automatic_rejected_intervals": [],
        "demos": [
            {
                "demo_id": 0,
                "source_interval_id": 0,
                "start_index": 20,
                "end_index": 120,
                "cutpoints_local_indices": [20, 40, 60, 80],
                "automatic_cutpoints_local_indices": [20, 40, 60, 80],
                "cutpoint_proposal_method": "test",
                "diagnostics": {},
            }
        ],
        "review": {
            "demo_boundaries_confirmed": True,
            "cutpoints_confirmed": False,
        },
        "export": {"status": "pending"},
    }
    (tmp_path / "review.json").write_text(json.dumps(review))
    store = ReviewStore(tmp_path)
    state = store.state()
    assert "overview_plot" not in state
    assert all("plot" not in demo for demo in state["demos"])
    app = MatplotlibReviewApp(store)
    assert app.output_hz_box.text == "5"
    assert app.output_name_box.text == "training_5hz.npz"
    feature_axis = app.stage_axes[0, 1]
    assert float(feature_axis.lines[0].get_xdata()[0]) == 0.0
    assert float(feature_axis.lines[0].get_xdata()[-1]) == 13.9
    press_x = feature_axis.transData.transform((6.0, 0.0))[0]
    app._stage_press(
        SimpleNamespace(button=1, inaxes=feature_axis, xdata=6.0, x=press_x)
    )
    app._stage_motion(
        SimpleNamespace(inaxes=feature_axis, xdata=6.5)
    )
    assert app.annotation_seconds[2] == 6.5
    assert all(float(line.get_xdata()[0]) == 6.5 for line in app.annotation_lines[2])
    assert float(app.annotation_markers[2].get_xdata()[0]) == task_xyz[65, 0]
    assert float(app.annotation_markers[2].get_ydata()[0]) == task_xyz[65, 1]
    app._stage_release(SimpleNamespace())

    press_x = feature_axis.transData.transform((2.0, 0.0))[0]
    app._stage_press(
        SimpleNamespace(button=1, inaxes=feature_axis, xdata=2.0, x=press_x)
    )
    app._stage_motion(SimpleNamespace(inaxes=feature_axis, xdata=1.5))
    assert app.annotation_seconds[0] == 1.5
    assert float(app.active_path_line.get_xdata()[0]) == task_xyz[15, 0]
    app._stage_release(SimpleNamespace())
    assert ReviewStore(tmp_path).state()["demos"][0]["start_index"] == 15
    app.plt.close(app.figure)
