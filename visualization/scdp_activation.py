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
    if plt is None or not getattr(learner, "demo_r_matrices_", None):
        return None

    masks = [np.asarray(r, dtype=float) for r in learner.demo_r_matrices_]
    feature_names = _feature_names(learner)
    n = len(masks)
    ncols = min(4, max(1, n))
    nrows = int(math.ceil(float(n) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.2 * nrows), squeeze=False)
    stage_labels = [f"s{i + 1}" for i in range(masks[0].shape[0])]

    for demo_idx, ax in enumerate(axes.flat):
        if demo_idx >= n:
            ax.axis("off")
            continue
        mask = masks[demo_idx].T
        ax.imshow(mask, cmap="Greys", vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(f"demo {demo_idx}")
        ax.set_xticks(range(mask.shape[1]))
        ax.set_xticklabels(stage_labels)
        ax.set_yticks(range(mask.shape[0]))
        ax.set_yticklabels(feature_names)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                ax.text(j, i, str(int(mask[i, j])), ha="center", va="center", color="tab:red", fontsize=9)

    fig.suptitle("SCDP activation masks", fontsize=12)
    fig.tight_layout()
    out = learner_plot_dir(learner) / f"activation_masks_iter_{int(it):04d}.png"
    save_figure(fig, out, dpi=220)
    return out


def plot_scdp_demo_avg_costs(learner, it):
    if plt is None or not getattr(learner, "current_demo_cost_breakdown", None):
        return None

    labels = ["constraint", "progress", "consensus", "r_consensus", "total"]
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
