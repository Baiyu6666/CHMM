from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import ListedColormap

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None


def _kind_from_semantic(text: object) -> str:
    value = str(text).strip().lower()
    if value in {"target_value", "target", "equality", "eq", "equal"}:
        return "eq"
    if value in {"upper_bound", "upper", "le", "<=", "max"}:
        return "upper"
    if value in {"lower_bound", "lower", "ge", ">=", "min"}:
        return "lower"
    return "none"


def _format_value(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return ""
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if abs_value < 1e-3:
        return f"{value:.2e}"
    if abs_value < 0.1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.3g}"


def _dataset_name_from_payload(payload: dict, dataset_name: str | None) -> str:
    if dataset_name:
        return str(dataset_name)
    for key in ("dataset_name", "dataset", "env_name", "task_name", "Task"):
        value = payload.get(key)
        if value:
            return str(value)
    return ""


def _display_scale_and_unit(dataset_name: str, feature_name: str) -> tuple[float, str]:
    dataset = str(dataset_name).lower()
    name = str(feature_name)
    if "s3obsavoid" in dataset:
        if name in {"obs_dist", "arc_dist", "line_dist"}:
            return 100.0, "cm"
        if name == "speed":
            return 100.0, "cm/s"
        if name == "heading":
            return 1.0, "rad"
        if name == "noise":
            return 1.0, ""
    if "s4slideinsert" in dataset:
        if name in {"surf_dist", "center_dist", "start_dist", "insert_err"}:
            return 1000.0, "mm"
        if name == "orient_err":
            return 180.0 / np.pi, "deg"
        if name == "speed":
            return 1000.0, "mm/s"
        if name == "angular_speed":
            return 180.0 / np.pi, "deg/s"
        if name == "normal_force":
            return 1.0, "N"
    if "s5sphereinspect" in dataset:
        if name in {"surf_dist", "start_dist", "goal_dist"}:
            return 1000.0, "mm"
        if name == "normal_err":
            return 180.0 / np.pi, "deg"
        if name == "speed":
            return 1000.0, "mm/s"
        if name == "ang_speed":
            return 180.0 / np.pi, "deg/s"
    if "barclean" in dataset:
        if name in {"obs_dist", "table_dist", "lateral_offset", "axial_offset"}:
            return 1000.0, "mm"
        if name in {"tool_pitch", "tool_roll", "tool_yaw"}:
            return 180.0 / np.pi, "deg"
    return 1.0, ""


def _feature_label(feature_name: str, unit: str) -> str:
    unit = str(unit).strip()
    return str(feature_name) if not unit else f"{feature_name} [{unit}]"


def _constraint_active_mask(values: np.ndarray, semantics: np.ndarray, active: np.ndarray | None) -> np.ndarray:
    if active is not None and active.shape == values.shape:
        return np.asarray(active, dtype=bool)
    semantic_active = np.vectorize(lambda value: _kind_from_semantic(value) != "none", otypes=[bool])(semantics)
    return np.isfinite(values) & semantic_active


def _matrix_layout(display_labels: list[str], num_stages: int) -> tuple[float, float, float, float]:
    matrix_height_in = 4.74
    cell_width_in = 1.27
    right_margin_in = 0.18
    max_label_chars = max((len(label) for label in display_labels), default=12)
    label_width_in = float(np.clip(0.62 + 0.125 * float(max_label_chars), 2.25, 2.95))
    fig_width_in = label_width_in + cell_width_in * float(num_stages) + right_margin_in
    return fig_width_in, matrix_height_in, label_width_in, right_margin_in


def _prepare_constraints_matrix(
    payload: dict,
    *,
    value_key: str,
    semantics_key: str,
    active_key: str | None = None,
    dataset_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    feature_names = [str(v) for v in payload.get("ConstraintFeatureNames", [])]
    dataset_name = _dataset_name_from_payload(payload, dataset_name)
    values = np.asarray(payload.get(value_key, []), dtype=float)
    semantics = np.asarray(payload.get(semantics_key, []), dtype=object)
    active = None
    if active_key is not None and active_key in payload:
        active = np.asarray(payload.get(active_key, []), dtype=float) > 0.5
    if values.ndim != 2 or semantics.shape != values.shape:
        raise ValueError(f"{value_key} and {semantics_key} must have the same 2D shape.")
    active = _constraint_active_mask(values, semantics, active)
    if not feature_names:
        feature_names = [f"feature_{idx}" for idx in range(values.shape[1])]
    display_info = [_display_scale_and_unit(dataset_name, name) for name in feature_names]
    display_labels = [_feature_label(name, unit) for name, (_, unit) in zip(feature_names, display_info)]

    codes = np.zeros_like(values, dtype=int)
    text = np.empty(values.shape, dtype=object)
    for stage_idx in range(values.shape[0]):
        for feat_idx in range(values.shape[1]):
            kind = _kind_from_semantic(semantics[stage_idx, feat_idx]) if active[stage_idx, feat_idx] else "none"
            value = float(values[stage_idx, feat_idx]) * float(display_info[feat_idx][0])
            if kind == "eq" and np.isfinite(value):
                codes[stage_idx, feat_idx] = 1
                text[stage_idx, feat_idx] = f"=\n{_format_value(value)}"
            elif kind == "upper" and np.isfinite(value):
                codes[stage_idx, feat_idx] = 2
                text[stage_idx, feat_idx] = f"≤\n{_format_value(value)}"
            elif kind == "lower" and np.isfinite(value):
                codes[stage_idx, feat_idx] = 3
                text[stage_idx, feat_idx] = f"≥\n{_format_value(value)}"
            else:
                text[stage_idx, feat_idx] = "Inactive"

    matrix = codes.T
    text_matrix = text.T
    stage_labels = [f"stage {idx + 1}" for idx in range(values.shape[0])]
    return matrix, text_matrix, stage_labels, display_labels


def _draw_constraints_matrix_panel(
    ax,
    matrix: np.ndarray,
    text_matrix: np.ndarray,
    stage_labels: list[str],
    display_labels: list[str],
    *,
    title: str | None = None,
) -> None:
    cmap = ListedColormap(["#f3f3f3", "#54a24b", "#e45756", "#4c78a8"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(stage_labels)))
    ax.set_xticklabels(stage_labels, fontsize=18)
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_yticklabels(display_labels, fontsize=18)
    ax.tick_params(axis="both", which="major", length=9, width=2)
    if title:
        ax.set_title(title, fontsize=20, fontweight="bold", pad=8)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            code = int(matrix[row, col])
            color = "white" if code in {1, 2, 3} else "black"
            ax.text(
                col,
                row,
                str(text_matrix[row, col]),
                ha="center",
                va="center",
                fontsize=17,
                color=color,
                linespacing=0.9,
            )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(2.5)

    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)


def _plot_constraints_matrix_paper(
    payload: dict,
    save_path: str | Path,
    *,
    value_key: str,
    semantics_key: str,
    active_key: str | None = None,
    dataset_name: str | None = None,
) -> Path | None:
    if plt is None:
        return None

    matrix, text_matrix, stage_labels, display_labels = _prepare_constraints_matrix(
        payload,
        value_key=value_key,
        semantics_key=semantics_key,
        active_key=active_key,
        dataset_name=dataset_name,
    )

    # Keep matrix columns comparable across datasets: same figure height and
    # same per-stage column width. The label area is dataset-specific to avoid
    # excessive whitespace for shorter feature names.
    fig_width_in, matrix_height_in, label_width_in, right_margin_in = _matrix_layout(
        display_labels,
        len(stage_labels),
    )
    fig, ax = plt.subplots(figsize=(fig_width_in, matrix_height_in), dpi=100)
    _draw_constraints_matrix_panel(ax, matrix, text_matrix, stage_labels, display_labels)
    fig.subplots_adjust(
        left=label_width_in / fig_width_in,
        right=1.0 - right_margin_in / fig_width_in,
        bottom=0.16,
        top=0.99,
    )
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def plot_learned_constraints_matrix_paper(payload: dict, save_path: str | Path, dataset_name: str | None = None) -> Path | None:
    return _plot_constraints_matrix_paper(
        payload,
        save_path,
        value_key="ConstraintLearnedValueMatrix",
        semantics_key="ConstraintLearnedSemanticsMatrix",
        active_key="ConstraintPredictedActiveMask",
        dataset_name=dataset_name,
    )


def plot_true_constraints_matrix_paper(payload: dict, save_path: str | Path, dataset_name: str | None = None) -> Path | None:
    return _plot_constraints_matrix_paper(
        payload,
        save_path,
        value_key="ConstraintTargetMatrix",
        semantics_key="ConstraintSemanticsMatrix",
        active_key=None,
        dataset_name=dataset_name,
    )


def plot_true_vs_learned_constraints_matrix_paper(
    payload: dict,
    save_path: str | Path,
    dataset_name: str | None = None,
) -> Path | None:
    if plt is None:
        return None

    true_matrix, true_text, true_stages, true_labels = _prepare_constraints_matrix(
        payload,
        value_key="ConstraintTargetMatrix",
        semantics_key="ConstraintSemanticsMatrix",
        active_key="ConstraintTrueActiveMask",
        dataset_name=dataset_name,
    )
    learned_matrix, learned_text, learned_stages, learned_labels = _prepare_constraints_matrix(
        payload,
        value_key="ConstraintLearnedValueMatrix",
        semantics_key="ConstraintLearnedSemanticsMatrix",
        active_key="ConstraintPredictedActiveMask",
        dataset_name=dataset_name,
    )

    if true_stages != learned_stages:
        raise ValueError("True and learned constraint matrices must have matching stage labels.")
    if true_labels != learned_labels:
        raise ValueError("True and learned constraint matrices must have matching feature labels.")

    fig_width_in, matrix_height_in, label_width_in, right_margin_in = _matrix_layout(
        true_labels,
        len(true_stages),
    )
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(fig_width_in, matrix_height_in * 2.0 + 0.6),
        dpi=100,
    )
    _draw_constraints_matrix_panel(
        axes[0],
        true_matrix,
        true_text,
        true_stages,
        true_labels,
        title="True constraint",
    )
    _draw_constraints_matrix_panel(
        axes[1],
        learned_matrix,
        learned_text,
        learned_stages,
        learned_labels,
        title="Learned constraint",
    )
    fig.subplots_adjust(
        left=label_width_in / fig_width_in,
        right=1.0 - right_margin_in / fig_width_in,
        bottom=0.08,
        top=0.96,
        hspace=0.32,
    )
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out
