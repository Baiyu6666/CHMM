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


def plot_scdp_activation_masks(learner, it):
    if plt is None:
        return None
    if getattr(learner, "feature_activation_mode", "fixed_mask") == "score":
        matrices = [np.asarray(m, dtype=float) for m in getattr(learner, "demo_feature_score_matrices_", [])]
        title = "SCDP feature scores"
        cmap = "viridis_r"
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
            matrix = np.std(np.asarray(learner.demo_feature_score_matrices_, dtype=float), axis=0).T
            ax.set_title("score std")
            vmin = float(np.nanmin(matrix))
            vmax = float(np.nanmax(matrix))
        else:
            demo_idx = panel_idx
            matrix = matrices[demo_idx].T
            ax.set_title(f"demo {demo_idx}")
            vmin = 0.0 if getattr(learner, "feature_activation_mode", "fixed_mask") != "score" else float(np.nanmin(matrix))
            vmax = 1.0 if getattr(learner, "feature_activation_mode", "fixed_mask") != "score" else float(np.nanmax(matrix))
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
                ax.text(j, i, value_formatter(matrix[i, j]), ha="center", va="center", color="white", fontsize=8)

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.10, top=0.88, wspace=0.35, hspace=0.40)
    out = learner_plot_dir(learner) / f"activation_masks_iter_{int(it):04d}.png"
    save_figure(fig, out, dpi=220)
    return out


def plot_scdp_demo_avg_costs(learner, it):
    if plt is None or not getattr(learner, "current_demo_cost_breakdown", None):
        return None

    labels = ["constraint", "progress", "subgoal_consensus", "param_consensus", "feature_score_consensus", "total"]
    rows = []
    for demo_idx, costs in enumerate(learner.current_demo_cost_breakdown):
        T = max(len(learner.demos[demo_idx]), 1)
        rows.append([float(costs.get(label, 0.0)) / float(T) for label in labels])
    values = np.asarray(rows, dtype=float)

    fig_w = max(5.2, 1.0 + 0.85 * len(labels))
    fig_h = max(3.4, 1.0 + 0.45 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, cmap="magma", aspect="auto")
    ax.set_title("Per-demo average step costs")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"demo {i}" for i in range(len(rows))])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = learner_plot_dir(learner) / f"demo_avg_costs_iter_{int(it):04d}.png"
    save_figure(fig, out, dpi=220)
    return out
