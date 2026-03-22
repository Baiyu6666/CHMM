from __future__ import annotations
from pathlib import Path

import numpy as np

from .io import learner_plot_dir, save_figure

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def _stage_labels(num_states: int):
    return [f"s{i + 1}" for i in range(int(num_states))]


def _feature_names(metrics, learner):
    names = metrics.get("ConstraintFeatureNames")
    if isinstance(names, list) and names:
        return [str(x) for x in names]
    out = []
    for local_idx in range(int(getattr(learner, "num_features", 0))):
        out.append(f"f{local_idx}")
    return out


def _scalar_metrics(metrics):
    out = {}
    for key, value in metrics.items():
        if np.isscalar(value):
            value_f = float(value)
            if np.isfinite(value_f):
                out[str(key)] = value_f
    return out


def _draw_heatmap(ax, matrix, title, *, feature_names, stage_labels, cmap="viridis", fmt=".2f", vmin=None, vmax=None):
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        ax.axis("off")
        return

    arr_plot = np.array(arr, dtype=float)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#d9d9d9")
    masked = np.ma.masked_invalid(arr_plot)
    if vmin is None:
        finite = arr_plot[np.isfinite(arr_plot)]
        vmin = float(np.min(finite)) if finite.size > 0 else 0.0
    if vmax is None:
        finite = arr_plot[np.isfinite(arr_plot)]
        vmax = float(np.max(finite)) if finite.size > 0 else 1.0
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(arr.shape[1]))
    ax.set_xticklabels(stage_labels, fontsize=7)
    ax.set_yticks(range(arr.shape[0]))
    ax.set_yticklabels(feature_names, fontsize=7)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            text = "nan" if not np.isfinite(value) else format(float(value), fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color="black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_mask(ax, matrix, title, *, feature_names, stage_labels):
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        ax.axis("off")
        return
    _draw_heatmap(
        ax,
        arr,
        title,
        feature_names=feature_names,
        stage_labels=stage_labels,
        cmap="Greys",
        fmt=".0f",
        vmin=0.0,
        vmax=1.0,
    )


def plot_evaluation_summary(learner, metrics, *, method_name: str, filename: str = "evaluation_summary.png", plot_dir: object = None):
    if plt is None or not isinstance(metrics, dict):
        return None

    scalar_metrics = _scalar_metrics(metrics)
    feature_names = _feature_names(metrics, learner)
    num_states = int(getattr(learner, "num_states", 0))
    stage_labels = _stage_labels(num_states)

    error_matrix = np.asarray(metrics.get("ConstraintErrorMatrix", []), dtype=float)
    target_matrix = np.asarray(metrics.get("ConstraintTargetMatrix", []), dtype=float)
    true_active = np.asarray(metrics.get("ConstraintTrueActiveMask", []), dtype=float)
    pred_active = np.asarray(metrics.get("ConstraintPredictedActiveMask", []), dtype=float)
    activation_match = None
    if true_active.shape == pred_active.shape and true_active.size > 0:
        activation_match = (true_active == pred_active).astype(float)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.6))
    axes = axes.reshape(2, 3)
    fig.suptitle(f"{method_name} evaluation summary", fontsize=11)

    ax_text = axes[0, 0]
    ax_text.axis("off")
    lines = []
    for key in sorted(scalar_metrics.keys()):
        lines.append(f"{key}: {scalar_metrics[key]:.4f}")
    if not lines:
        lines.append("No scalar metrics.")
    ax_text.text(
        0.02,
        0.98,
        "\n".join(lines),
        ha="left",
        va="top",
        fontsize=8,
        family="monospace",
        transform=ax_text.transAxes,
    )

    _draw_heatmap(
        axes[0, 1],
        np.asarray(error_matrix, dtype=float).T if error_matrix.ndim == 2 else error_matrix,
        "constraint error",
        feature_names=feature_names,
        stage_labels=stage_labels,
        cmap="magma_r",
        fmt=".3f",
    )
    _draw_heatmap(
        axes[0, 2],
        np.asarray(target_matrix, dtype=float).T if target_matrix.ndim == 2 else target_matrix,
        "constraint target",
        feature_names=feature_names,
        stage_labels=stage_labels,
        cmap="viridis",
        fmt=".3f",
    )
    _draw_mask(
        axes[1, 0],
        np.asarray(true_active, dtype=float).T if true_active.ndim == 2 else true_active,
        "true active",
        feature_names=feature_names,
        stage_labels=stage_labels,
    )
    _draw_mask(
        axes[1, 1],
        np.asarray(pred_active, dtype=float).T if pred_active.ndim == 2 else pred_active,
        "pred active",
        feature_names=feature_names,
        stage_labels=stage_labels,
    )
    if activation_match is not None:
        _draw_mask(
            axes[1, 2],
            np.asarray(activation_match, dtype=float).T,
            "activation match",
            feature_names=feature_names,
            stage_labels=stage_labels,
        )
    else:
        axes[1, 2].axis("off")

    fig.subplots_adjust(wspace=0.55, hspace=0.35)
    out_path = learner_plot_dir(learner, plot_dir=plot_dir) / filename
    return save_figure(fig, Path(out_path), close=True, dpi=170)
