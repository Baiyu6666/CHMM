from __future__ import annotations

import argparse
import json

import pytest

from robot.video_process.concat_trajectory_videos import build_filter
from robot.video_process import process_recording
from robot.video_process.process_recording import resolve_selected_runs
from robot.video_process.render_experiment_profiles import (
    _axis_limits,
    _semantics_kind,
    constraint_references,
    stage_timeline,
)


def test_stage_timeline_scales_plan_to_recorded_duration() -> None:
    visualization = {
        "planned_feature_series": {
            "schema": [{"name": "distance", "unit": "m"}],
            "samples": [[0.0, 0.0], [10.0, 1.0]],
        },
        "stage_boundary_times": [2.0, 6.0],
        "stage_transition_end_times": [2.5, 6.5],
    }
    windows, transitions, scale = stage_timeline(visualization, 20.0)

    assert scale == pytest.approx(2.0)
    assert transitions == pytest.approx([(4.0, 5.0), (12.0, 13.0)])
    assert [(item.start_s, item.end_s) for item in windows] == pytest.approx(
        [(0.0, 4.0), (5.0, 12.0), (13.0, 20.0)]
    )


def test_constraint_references_keep_stage_specific_values() -> None:
    series = {
        "true_constraints": {"oracle_second": 0.24},
        "constraint_specs": [
            {
                "feature_name": "distance",
                "stage": 0,
                "semantics": "lower_bound",
                "value": 0.10,
            },
            {
                "feature_name": "distance",
                "stage": 2,
                "semantics": "target_value",
                "oracle_key": "oracle_second",
                "value": 9.0,
            },
            {
                "feature_name": "other",
                "stage": 1,
                "semantics": "target_value",
                "value": 1.0,
            },
        ],
    }

    references = constraint_references(series, "distance", "constraint_specs")

    assert [(item.stage, item.value, item.semantics) for item in references] == [
        (0, 0.10, "lower_bound"),
        (2, 0.24, "target_value"),
    ]


def test_constraint_semantics_distinguish_equality_and_inequality() -> None:
    assert _semantics_kind("target_value") == "target"
    assert _semantics_kind("lower_bound") == "lower"
    assert _semantics_kind("upper_bound") == "upper"


def test_angle_axes_can_share_yaw_span_without_sharing_center() -> None:
    yaw_min, yaw_max = _axis_limits([-45.0, 10.0])
    yaw_span = yaw_max - yaw_min

    pitch_min, pitch_max = _axis_limits([87.0, 91.0], yaw_span)
    roll_min, roll_max = _axis_limits([-2.5, 0.5], yaw_span)

    assert pitch_max - pitch_min == pytest.approx(yaw_span)
    assert roll_max - roll_min == pytest.approx(yaw_span)
    assert 0.5 * (pitch_min + pitch_max) == pytest.approx(89.0)
    assert 0.5 * (roll_min + roll_max) == pytest.approx(-1.0)


def test_concat_filter_pauses_at_start_and_end_of_every_run() -> None:
    graph = build_filter(2, 1920, 1080, 30.0, 1.0, 1.0)

    assert graph.count("tpad=start_mode=clone:stop_mode=clone") == 2
    assert graph.count("start_duration=1.000000") == 2
    assert graph.count("stop_duration=1.000000") == 2
    assert "[v0][v1]concat=n=2:v=1:a=0[outv]" in graph


def test_selected_run_prefixes_preserve_requested_order() -> None:
    available = [
        {"id": "20260829T065033_535014Z_real_task"},
        {"id": "20260829T071031_477167Z_real_task"},
    ]

    selected = resolve_selected_runs(available, ["20260829T071031", "20260829T065033"])

    assert [item["id"] for item in selected] == [
        "20260829T071031_477167Z_real_task",
        "20260829T065033_535014Z_real_task",
    ]


def test_ambiguous_run_prefix_is_rejected() -> None:
    available = [{"id": "run_001_a"}, {"id": "run_001_b"}]

    with pytest.raises(RuntimeError, match="ambiguous"):
        resolve_selected_runs(available, ["run_001"])


def test_render_outputs_live_with_final_runs(tmp_path, monkeypatch) -> None:
    final_root = tmp_path / "final_video_runs"
    run_directory = final_root / "BarClean" / "run_001"
    run_directory.mkdir(parents=True)
    for name in ("execution.mp4", "visualization.json", "result.json"):
        (run_directory / name).write_bytes(b"data")
    (run_directory / "metadata.json").write_text(
        json.dumps({"constraint_source": "true"}), encoding="utf-8"
    )
    commands = []

    def record_command(command):
        commands.append(command)
        return {"ok": True}

    monkeypatch.setattr(process_recording, "_run_command", record_command)
    options = argparse.Namespace(
        task="BarClean",
        final_run_root=final_root,
        runs=["run_001"],
        output=None,
        start_pause_seconds=1.0,
        end_pause_seconds=1.0,
        fps=30.0,
        crf=15,
        panel_width_ratio=0.28,
        max_seconds=None,
    )

    result = process_recording.render_selected(options)

    clip = run_directory / "execution_profiles.mp4"
    assert result["clips"] == [str(clip)]
    assert result["output"] == str(final_root / "selected_runs.mp4")
    assert result["manifest"] == str(final_root / "render_manifest.json")
    assert str(clip) in commands[0]
    assert str(clip) in commands[1]
