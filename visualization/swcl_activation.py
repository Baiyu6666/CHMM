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
    if hasattr(learner, "_equality_score_threshold"):
        eq_thr = float(learner._equality_score_threshold())
    else:
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


def _activation_line_specs(learner):
    feature_names = _feature_names(learner)
    cmap = plt.cm.get_cmap("tab10", max(int(learner.num_features), 1))
    stage_linestyles = ["-", "--", "-.", ":"]
    specs = []
    for stage_idx in range(int(learner.num_stages)):
        for feat_idx in range(int(learner.num_features)):
            specs.append(
                {
                    "stage_idx": int(stage_idx),
                    "feat_idx": int(feat_idx),
                    "label": f"s{stage_idx + 1}:{feature_names[feat_idx]}",
                    "color": cmap(feat_idx % max(int(learner.num_features), 1)),
                    "linestyle": stage_linestyles[stage_idx % len(stage_linestyles)],
                }
            )
    return specs


def plot_swcl_activation_dynamics(learner, it):
    if plt is None:
        return None
    demo_history = getattr(learner, "demo_activation_history", None) or []
    proto_history = getattr(learner, "activation_proto_history", None) or []
    if not demo_history or not proto_history:
        return None
    demo_hist = np.asarray(demo_history, dtype=float)
    proto_hist = np.asarray(proto_history, dtype=float)
    if demo_hist.ndim != 4 or proto_hist.ndim != 3:
        return None

    num_iters, num_demos, _, _ = demo_hist.shape
    n_panels = num_demos + 1
    ncols = min(3, max(1, n_panels))
    nrows = int(math.ceil(float(n_panels) / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 3.2 * nrows), squeeze=False)
    line_specs = _activation_line_specs(learner)
    x = np.arange(1, num_iters + 1, dtype=int)

    for panel_idx, ax in enumerate(axes.flat):
        if panel_idx >= n_panels:
            ax.axis("off")
            continue
        if panel_idx < num_demos:
            for spec in line_specs:
                y = demo_hist[:, panel_idx, spec["stage_idx"], spec["feat_idx"]]
                ax.plot(
                    x,
                    y,
                    color=spec["color"],
                    linestyle=spec["linestyle"],
                    linewidth=1.8,
                    marker="o",
                    markersize=3.5,
                    alpha=0.95,
                    label=spec["label"],
                )
            if x.size == 1:
                ax.set_xlim(0.5, 1.5)
            else:
                ax.set_xlim(float(x[0]), float(x[-1]))
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("iteration")
            ax.set_ylabel("activation")
            ax.set_title(f"demo {panel_idx} activation history")
            ax.grid(alpha=0.15, linewidth=0.5)
            if x.size <= 12:
                ax.set_xticks(x)
        else:
            for spec in line_specs:
                y = proto_hist[:, spec["stage_idx"], spec["feat_idx"]]
                ax.plot(
                    x,
                    y,
                    color=spec["color"],
                    linestyle=spec["linestyle"],
                    linewidth=1.8,
                    marker="o",
                    markersize=3.5,
                    alpha=0.95,
                    label=spec["label"],
                )
            if x.size == 1:
                ax.set_xlim(0.5, 1.5)
            else:
                ax.set_xlim(float(x[0]), float(x[-1]))
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel("iteration")
            ax.set_ylabel("activation rate")
            ax.set_title("shared activation prototype")
            ax.grid(alpha=0.15, linewidth=0.5)
            if x.size <= 12:
                ax.set_xticks(x)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, max(1, int(math.ceil(len(labels) / 2.0)))), fontsize=7.5, frameon=False)
    fig.suptitle("SWCL activation dynamics", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.18, top=0.90, wspace=0.28, hspace=0.35)
    out = learner_plot_dir(learner) / f"activation_dynamics_iter_{int(it):04d}.png"
    save_figure(fig, out, dpi=220)
    return out


def plot_swcl_activation_masks(learner, it):
    if plt is None:
        return None
    is_score_mode = getattr(learner, "feature_activation_mode", "fixed_mask") in {"score", "joint_mask_search"}
    is_joint_mask_search = getattr(learner, "feature_activation_mode", "fixed_mask") == "joint_mask_search"
    if is_score_mode:
        raw_matrices = [np.asarray(m, dtype=float) for m in getattr(learner, "demo_feature_score_matrices_", [])]
        thresholds = _score_thresholds(learner)
        matrices = [thresholds[None, :] - m for m in raw_matrices]
        title = "SWCL constraint margins"
        cmap = "RdBu_r"
        value_formatter = lambda x: f"{float(x):.2f}"
    else:
        matrices = [np.asarray(r, dtype=float) for r in getattr(learner, "demo_r_matrices_", [])]
        title = "SWCL activation masks"
        cmap = "Greys"
        value_formatter = lambda x: str(int(x))
    if not matrices:
        return None

    feature_names = _feature_names(learner)
    show_rate_panel = is_score_mode and getattr(learner, "posthoc_activation_summary_", None) is not None
    show_std_panel = is_score_mode and bool(getattr(learner, "demo_feature_score_matrices_", []))
    show_proto_panel = is_score_mode and getattr(learner, "shared_activation_proto", None) is not None
    total_panels = len(matrices) + (1 if show_rate_panel else 0) + (1 if show_std_panel else 0) + (1 if show_proto_panel else 0)
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
        proto_panel_idx = len(matrices) + (1 if show_rate_panel else 0) + (1 if show_std_panel else 0) if show_proto_panel else None
        if show_rate_panel and panel_idx == rate_panel_idx:
            matrix = np.asarray(learner.posthoc_activation_summary_["activation_rate_matrix"], dtype=float).T
            ax.set_title("activation rate")
            vmin, vmax = 0.0, 1.0
            panel_cmap = "Greys"
            panel_formatter = lambda x: f"{float(x):.2f}"
        elif show_std_panel and panel_idx == std_panel_idx:
            matrix = np.std(np.asarray(matrices, dtype=float), axis=0).T
            ax.set_title("score std")
            vmin = float(np.nanmin(matrix))
            vmax = float(np.nanmax(matrix))
            panel_cmap = "magma"
            panel_formatter = lambda x: f"{float(x):.2f}"
        elif show_proto_panel and panel_idx == proto_panel_idx:
            matrix = np.asarray(learner.shared_activation_proto, dtype=float).T
            ax.set_title("shared R" if is_joint_mask_search else "activation proto")
            vmin, vmax = 0.0, 1.0
            panel_cmap = "Greys"
            panel_formatter = (lambda x: str(int(round(float(x))))) if is_joint_mask_search else (lambda x: f"{float(x):.2f}")
        else:
            demo_idx = panel_idx
            matrix = matrices[demo_idx].T
            ax.set_title(f"demo {demo_idx}")
            if not is_score_mode:
                vmin, vmax = 0.0, 1.0
                panel_cmap = cmap
                panel_formatter = value_formatter
            else:
                vmax = float(np.nanmax(np.abs(matrix)))
                if not np.isfinite(vmax) or vmax <= 0.0:
                    vmax = 1.0
                vmin = -vmax
                panel_cmap = cmap
                panel_formatter = value_formatter
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6
        im = ax.imshow(matrix, cmap=panel_cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(matrix.shape[1]))
        ax.set_xticklabels(stage_labels)
        ax.set_yticks(range(matrix.shape[0]))
        ax.set_yticklabels(feature_names)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = float(matrix[i, j]) if np.isscalar(matrix[i, j]) else np.nan
                ax.text(
                    j, i, panel_formatter(matrix[i, j]),
                    ha="center", va="center",
                    color=_matrix_text_color(value, max(abs(vmin), abs(vmax))),
                    fontsize=8,
                )

    fig.suptitle(title, fontsize=12)
    fig.subplots_adjust(left=0.12, right=0.92, bottom=0.10, top=0.88, wspace=0.35, hspace=0.40)
    out = learner_plot_dir(learner) / f"activation_masks_iter_{int(it):04d}.png"
    save_figure(fig, out, dpi=220)
    return out
