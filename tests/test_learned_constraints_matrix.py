from pathlib import Path

from experiments.plot_learned_constraints_matrix import _infer_dataset_name
from visualization.learned_constraints_matrix import _prepare_constraints_matrix


def test_barclean_paper_constraint_units_and_scaling():
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
        "values": [[0.0642, 0.0932, 0.0041, -0.0319, 1.53, -0.0314, -0.688]],
        "semantics": [["target_value"] * len(feature_names)],
    }

    _, text, _, labels = _prepare_constraints_matrix(
        payload,
        value_key="values",
        semantics_key="semantics",
        dataset_name="BarClean",
    )

    assert labels == [
        "obs_dist [mm]",
        "table_dist [mm]",
        "lateral_offset [mm]",
        "axial_offset [mm]",
        "tool_pitch [deg]",
        "tool_roll [deg]",
        "tool_yaw [deg]",
    ]
    assert text[:, 0].tolist() == [
        "=\n64.2",
        "=\n93.2",
        "=\n4.1",
        "=\n-31.9",
        "=\n87.7",
        "=\n-1.8",
        "=\n-39.4",
    ]


def test_barclean_constraints_path_identifies_dataset():
    path = Path("outputs/map_balanced_pooled/BarClean/method_seed_000/constraints.json")

    assert _infer_dataset_name(path) == "BarClean"
