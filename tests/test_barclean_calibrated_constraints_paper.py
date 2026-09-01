from copy import deepcopy

import numpy as np

from experiments.plot_barclean_calibrated_constraints_paper import (
    BARCLEAN_SPONGE_YAW_OFFSET_DEG,
    prepare_barclean_paper_payload,
)


def test_barclean_paper_payload_uses_sponge_and_error_coordinates():
    feature_names = [
        "obs_dist",
        "table_dist",
        "lateral_offset",
        "axial_offset",
        "tool_pitch",
        "tool_roll",
        "tool_yaw",
    ]
    payload = {
        "ConstraintFeatureNames": feature_names,
        "ConstraintLearnedValueMatrix": [
            [0.0642, 0.0932, 0.0041, -0.0319, 1.53, -0.0314, -0.688]
        ],
    }
    original = deepcopy(payload)

    transformed = prepare_barclean_paper_payload(payload)

    assert payload == original
    assert transformed["ConstraintFeatureNames"] == [
        "obs_dist [mm]",
        "table_dist [mm]",
        "lateral_offset [mm]",
        "axial_offset [mm]",
        "tool_pitch [deg]",
        "tool_roll [deg]",
        "tool_yaw [deg]",
    ]
    values = np.asarray(transformed["ConstraintLearnedValueMatrix"])[0]
    np.testing.assert_allclose(values[:4], [64.2, 93.2, 4.1, 344.16532])
    np.testing.assert_allclose(values[4], np.rad2deg(1.53) - 90.0)
    np.testing.assert_allclose(values[5], np.rad2deg(-0.0314))
    np.testing.assert_allclose(
        values[6],
        np.rad2deg(-0.688) + BARCLEAN_SPONGE_YAW_OFFSET_DEG,
    )
    assert transformed["BarCleanPaperCalibration"]["axial_offset_zero_task_x_m"] == -0.15
    assert transformed["BarCleanPaperCalibration"]["axial_offset_display_shift_mm"] == 376.06532


def test_barclean_paper_payload_wraps_calibrated_sponge_yaw():
    payload = {
        "ConstraintFeatureNames": [
            "obs_dist",
            "table_dist",
            "lateral_offset",
            "axial_offset",
            "tool_pitch",
            "tool_roll",
            "tool_yaw",
        ],
        "ConstraintLearnedValueMatrix": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, np.deg2rad(170.0)]],
    }

    transformed = prepare_barclean_paper_payload(
        payload,
        sponge_yaw_offset_deg=30.0,
    )

    assert np.asarray(transformed["ConstraintLearnedValueMatrix"])[0, 6] == -160.0
