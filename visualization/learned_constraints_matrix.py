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


def plot_learned_constraints_matrix_paper(payload: dict, save_path: str | Path) -> Path | None:
    if plt is None:
        return None

    feature_names = [str(v) for v in payload.get("ConstraintFeatureNames", [])]
    values = np.asarray(payload.get("ConstraintLearnedValueMatrix", []), dtype=float)
    semantics = np.asarray(payload.get("ConstraintLearnedSemanticsMatrix", []), dtype=object)
    active = np.asarray(payload.get("ConstraintPredictedActiveMask", []), dtype=float) > 0.5
    if values.ndim != 2 or semantics.shape != values.shape:
        raise ValueError("ConstraintLearnedValueMatrix and ConstraintLearnedSemanticsMatrix must have the same 2D shape.")
    if active.shape != values.shape:
        active = np.isfinite(values)
    if not feature_names:
        feature_names = [f"feature_{idx}" for idx in range(values.shape[1])]

    codes = np.zeros_like(values, dtype=int)
    text = np.empty(values.shape, dtype=object)
    for stage_idx in range(values.shape[0]):
        for feat_idx in range(values.shape[1]):
            kind = _kind_from_semantic(semantics[stage_idx, feat_idx]) if active[stage_idx, feat_idx] else "none"
            value = float(values[stage_idx, feat_idx])
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
                text[stage_idx, feat_idx] = "None"

    matrix = codes.T
    text_matrix = text.T
    stage_labels = [f"stage {idx + 1}" for idx in range(values.shape[0])]

    fig, ax = plt.subplots(figsize=(8.2, 4.74), dpi=100)
    cmap = ListedColormap(["#f3f3f3", "#54a24b", "#e45756", "#4c78a8"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")

    ax.set_xticks(np.arange(len(stage_labels)))
    ax.set_xticklabels(stage_labels, fontsize=22)
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=22)
    ax.tick_params(axis="both", which="major", length=9, width=2)

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
    fig.subplots_adjust(left=0.36, right=0.99, bottom=0.16, top=0.99)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out
