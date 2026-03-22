from __future__ import annotations

import math

import numpy as np

from .io import learner_plot_dir, save_figure

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _feature_names(learner):
    names = []
    for local_idx in range(learner.num_features):
        selected_col = int(learner.selected_feature_columns[local_idx])
        name = None
        for i, spec in enumerate(learner.raw_feature_specs):
            if int(spec.get("column_idx", i)) == selected_col:
                name = str(spec.get("name", f"f{local_idx}"))
                break
        names.append(name or f"f{local_idx}")
    return names


def _score_thresholds(learner):
    eq_thr = float(getattr(learner, "equality_dispersion_ratio_threshold", 0.1))
    ineq_thr = float(getattr(learner, "inequality_score_activation_threshold", -0.5))
    return np.asarray(
        [eq_thr if hasattr(learner, "_is_equality_feature") and learner._is_equality_feature(i) else ineq_thr for i in range(learner.num_features)],
        dtype=float,
    )


def _matrix_text_color(value, vmax):
    if not np.isfinite(value):
        return "black"
    threshold = 0.55 * float(max(vmax, 1e-6))
    return "white" if abs(float(value)) >= threshold else "black"


def plot_scdp_activation_masks(learner, it):
    if plt is None:
        return None
    if getattr(learner, "feature_activation_mode", "fixed_mask") == "score":
        raw_matrices = [np.asarray(m, dtype=float) for m in getattr(learner, "demo_feature_score_matrices_", [])]
        thresholds = _score_thresholds(learner)
        matrices = [thresholds[None, :] - m for m in raw_matrices]
        title = "SCDP constraint margins"
        cmap = "RdBu_r"
        value_formatter = lambda x: f"{float(x):.2f}"
    else:
        matrices = [np.asarray(r, dtype=float) for r in getattr(learner, "demo_r_matrices_", [])]
        title = "SCDP activation masks"
        cmap = "Greys"
        value_formatter = lambda x: str(int(x))
    if not matrices:
        return None

    feature_names = _feature_names(learner)
    is_score_mode = getattr(learner, "feature_activation_mode", "fixed_mask") == "score"
    show_rate_panel = is_score_mode and getattr(learner, "posthoc_activation_summary_", None) is not None
    show_std_panel = is_score_mode and bool(getattr(learner, "demo_feature_score_matrices_", []))
    total_panels = len(matrices) + (1 if show_rate_panel else 0) + (1 if show_std_panel else 0)
    ncols = min(4, max(1, total_panels))
    nrows = int(math.ceil(float(total_panels) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows), squeeze=False)
    stage_labels = [f"s{i + 1}" for i in range(matrices[0].shape[0])]

    for panel_idx, ax in enumerate(axes.flat):
        if panel_idx >= total_panels:
            ax.axis("off")
            continue
        rate_panel_idx = len(matrices) if show_rate_panel else None
        std_panel_idx = len(matrices) + (1 if show_rate_panel else 0) if show_std_panel else None
        if show_rate_panel and panel_idx == rate_panel_idx:
            matrix = np.asarray(learner.posthoc_activation_summary_["activation_rate_matrix"], dtype=float).T
            ax.set_title("activation rate")
            vmin, vmax = 0.0, 1.0
        elif show_std_panel and panel_idx == std_panel_idx:
            matrix = np.std(np.asarray(matrices, dtype=float), axis=0).T
            ax.set_title("score std")
            vmin = float(np.nanmin(matrix))
            vmax = float(np.nanmax(matrix))
        else:
            demo_idx = panel_idx
            matrix = matrices[demo_idx].T
            ax.set_title(f"demo {demo_idx}")
            if getattr(learner, "feature_activation_mode", "fixed_mask") != "score":
                vmin, vmax = 0.0, 1.0
            else:
                vmax = float(np.nanmax(np.abs(matrix)))
                if not np.isfinite(vmax) or vmax <= 0.0:
                    vmax = 1.0
                vmin = -vmax
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6
        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(stage_labels)
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(feature_names)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = float(matrix[i, j]) if np.isscalar(matrix[i, j]) else np.nan
                ax.text(
                    j, i, value_formatter(matrix[i, j]),
                    ha="center", va="center",
                    color=_matrix_text_color(value, max(abs(vmin), abs(vmax))),
                    fontsize=8,
                )

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.10, top=0.88, wspace=0.35, hspace=0.40)
    out = learner_plot_dir(learner) / f"activation_masks_iter_{int(it):04d}.png"
    save_figure(fig, out, dpi=220)
    return out
