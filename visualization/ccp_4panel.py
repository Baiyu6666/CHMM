from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from .io import learner_plot_dir, save_figure

PAPER_FIGSIZE = (8.4, 6.0)
PAPER_TITLE_SIZE = 9
PAPER_LABEL_SIZE = 8
PAPER_TICK_SIZE = 7
PAPER_LEGEND_SIZE = 6.5


def _env_has_xy_obstacle(env) -> bool:
    return hasattr(env, "obs_center") and hasattr(env, "obs_radius")


def _env_has_3d_obstacle(env) -> bool:
    return hasattr(env, "obs_center_xy") and hasattr(env, "obs_radius")


def _is_pickplace(env) -> bool:
    return getattr(env, "eval_tag", "") == "PickPlace"


def _xy_point(point):
    arr = np.asarray(point, dtype=float).reshape(-1)
    return arr[:2]


def _legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l is None:
            continue
        text = str(l).strip()
        if text and not text.startswith("_") and text not in by_label:
            by_label[text] = h
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=PAPER_LEGEND_SIZE, frameon=False, loc="best")


def _feature_name(learner, local_idx):
    schema = getattr(learner, "raw_feature_specs", None) or []
    selected_col = int(learner.selected_feature_columns[local_idx])
    for i, spec in enumerate(schema):
        if int(spec.get("column_idx", i)) == selected_col:
            return str(spec.get("name", f"f{local_idx}"))
    return f"f{local_idx}"


def _segment_bounds(stage_ends):
    starts = []
    ends = []
    prev = -1
    for end in stage_ends:
        starts.append(prev + 1)
        ends.append(int(end))
        prev = int(end)
    return starts, ends


def _draw_feature_bands(ax, learner, tau_hat0):
    X0 = learner.demos[0]
    T0 = len(X0)
    t_axis = np.arange(T0)
    Fz = learner.standardized_features[0]
    raw_values = (
        Fz * learner.feat_std[learner.selected_feature_columns][None, :]
        + learner.feat_mean[learner.selected_feature_columns][None, :]
    )
    max_features = min(learner.num_features, 6)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max_features))
    stage_ends = learner.stage_ends_[0]
    starts, ends = _segment_bounds(stage_ends)
    for k, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(s, e, color=("orange" if k == 0 else "red"), alpha=0.035)
    for m in range(max_features):
        label = _feature_name(learner, m)
        ax.plot(t_axis, raw_values[:, m], lw=1.0, color=colors[m], label=label)
        for k in range(min(learner.num_states, len(starts))):
            if learner.r[k, m] != 1:
                continue
            summary = learner.feature_models[k][m].get_summary()
            low = summary.get("L")
            high = summary.get("U")
            if low is None or high is None:
                continue
            fid = learner.selected_feature_columns[m]
            low_raw = low * learner.feat_std[fid] + learner.feat_mean[fid]
            high_raw = high * learner.feat_std[fid] + learner.feat_mean[fid]
            color = colors[m]
            s = starts[k]
            e = ends[k]
            ax.fill_between(
                t_axis[s : e + 1],
                low_raw,
                high_raw,
                color=color,
                alpha=0.12 if k == 0 else 0.08,
            )
    ax.axvline(int(tau_hat0), color="black", linestyle="--", lw=1.0, label="pred boundary")
    if learner.true_taus[0] is not None:
        ax.axvline(int(learner.true_taus[0]), color="green", linestyle=":", lw=1.0, label="true boundary")
    ax.set_title("Demo0 feature constraints", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("time", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("raw feature value", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_completion_ellipse_2d(ax, center_xy, precision_diag, color, label, n_std=2.0):
    center_xy = np.asarray(center_xy, dtype=float).reshape(-1)[:2]
    precision_diag = np.asarray(precision_diag, dtype=float).reshape(-1)[:2]
    if np.any(precision_diag <= 0):
        return
    std_xy = 1.0 / np.sqrt(np.maximum(precision_diag, 1e-12))
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    ellipse = np.stack(
        [
            center_xy[0] + n_std * std_xy[0] * np.cos(theta),
            center_xy[1] + n_std * std_xy[1] * np.sin(theta),
        ],
        axis=1,
    )
    ax.plot(ellipse[:, 0], ellipse[:, 1], color=color, lw=1.0, alpha=0.85, label=label)


def _draw_boundary_cost_profile(ax, learner, demo_idx, stage_idx=0):
    X = learner.demos[demo_idx]
    T = len(X)
    if learner.num_states != 2:
        ax.text(0.5, 0.5, "Boundary profile is shown for 2-stage CCP.", ha="center", va="center")
        ax.axis("off")
        return
    candidate_taus = np.arange(max(1, learner.duration_min[0] - 1), min(T - 2, T - learner.duration_min[1]) + 1)
    if len(candidate_taus) == 0:
        ax.text(0.5, 0.5, "No feasible boundaries under duration bounds.", ha="center", va="center")
        ax.axis("off")
        return
    total = []
    constraint = []
    end = []
    progress = []
    for tau in candidate_taus:
        p1 = learner._segment_cost_parts(demo_idx, 0, 0, int(tau))
        p2 = learner._segment_cost_parts(demo_idx, 1, int(tau) + 1, T - 1)
        raw_constraint = p1["constraint"] + p2["constraint"]
        raw_end = p1["end"] + p2["end"]
        raw_progress = p1["progress"] + p2["progress"]
        weighted_constraint = learner.lambda_constraint * raw_constraint
        weighted_end = learner.lambda_end * raw_end
        weighted_progress = learner.lambda_progress * raw_progress
        total.append(weighted_constraint + weighted_end + weighted_progress)
        constraint.append(weighted_constraint)
        end.append(weighted_end)
        progress.append(weighted_progress)
    ax.plot(candidate_taus, total, color="black", lw=1.4, label="total")
    ax.plot(candidate_taus, constraint, color="tab:red", lw=1.0, label="constraint")
    ax.plot(candidate_taus, end, color="tab:blue", lw=1.0, label="completion")
    ax.plot(candidate_taus, progress, color="tab:orange", lw=1.2, label="progress", linestyle="-")

    left_max = max(
        [0.0]
        + [float(np.max(series)) for series in (total, constraint, end, progress) if len(series) > 0]
    )
    ax.set_ylim(top=min(left_max * 1.05 if left_max > 0.0 else 1.0, 250.0))

    pred_tau = int(learner.stage_ends_[demo_idx][0])
    ax.axvline(pred_tau, color="black", linestyle="--", lw=1.0, label="pred boundary")
    prev_stage_ends = getattr(learner, "segmentation_history", None)
    if isinstance(prev_stage_ends, list) and len(prev_stage_ends) >= 2:
        prev_demo_stage_ends = prev_stage_ends[-2][demo_idx]
        prev_tau = int(prev_demo_stage_ends[0])
        ax.axvline(prev_tau, color="dimgray", linestyle="-.", lw=1.0, label="prev pred boundary")
    if learner.true_taus[demo_idx] is not None:
        ax.axvline(int(learner.true_taus[demo_idx]), color="green", linestyle=":", lw=1.0, label="true boundary")
    ax.set_title("Demo0 boundary cost profile", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("boundary index", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("cost", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)

    by_label = {}
    handles, labels = ax.get_legend_handles_labels()
    for h, l in zip(handles, labels):
        if l is None:
            continue
        text = str(l).strip()
        if text and not text.startswith("_") and text not in by_label:
            by_label[text] = h
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=PAPER_LEGEND_SIZE, frameon=False, loc="best")


def plot_ccp_results_4panel(learner, it):
    if plt is None:
        return
    fig = plt.figure(figsize=PAPER_FIGSIZE)
    X_dim = learner.demos[0].shape[1]

    if X_dim == 3:
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    else:
        ax1 = fig.add_subplot(2, 2, 1)

    taus = [int(ends[0]) for ends in learner.stage_ends_] if learner.num_states == 2 else None
    if X_dim == 3:
        for i, X in enumerate(learner.demos):
            X = np.asarray(X, dtype=float)
            stage_ends = learner.stage_ends_[i]
            starts, ends = _segment_bounds(stage_ends)
            stage_colors = ["orange", "red", "purple", "brown"]
            for k, (s, e) in enumerate(zip(starts, ends)):
                color = stage_colors[k % len(stage_colors)]
                ax1.scatter(X[s : e + 1, 0], X[s : e + 1, 1], X[s : e + 1, 2], c=color, s=3, alpha=0.35)
                ax1.scatter(X[e, 0], X[e, 1], X[e, 2], c="blue", marker="x", s=22, linewidths=0.9,
                            label="pred stage end" if i == 0 and k == 0 else "")
            if learner.true_taus[i] is not None:
                tt = int(learner.true_taus[i])
                ax1.scatter(X[tt, 0], X[tt, 1], X[tt, 2], c="green", marker="x", s=22, linewidths=0.9,
                            label="true boundary" if i == 0 else "")
        if hasattr(learner.env, "subgoal"):
            sg = np.asarray(learner.env.subgoal, dtype=float)
            ax1.scatter(sg[0], sg[1], sg[2], c="green", marker="*", s=28, label="true subgoal")
        if hasattr(learner.env, "goal"):
            gg = np.asarray(learner.env.goal, dtype=float)
            ax1.scatter(gg[0], gg[1], gg[2], c="green", marker="P", s=28, label="true goal")
        if getattr(learner, "g1", None) is not None:
            ax1.scatter(learner.g1[0], learner.g1[1], learner.g1[2], c="blue", marker="D", s=24, label="mu1 / g1")
        if getattr(learner, "g2", None) is not None:
            ax1.scatter(learner.g2[0], learner.g2[1], learner.g2[2], c="navy", marker="P", s=24, label="mu2 / g2")
        if _env_has_3d_obstacle(learner.env):
            pass
        ax1.set_xlabel("x", fontsize=PAPER_LABEL_SIZE)
        ax1.set_ylabel("y", fontsize=PAPER_LABEL_SIZE)
        ax1.set_zlabel("z", fontsize=PAPER_LABEL_SIZE)
    else:
        for i, X in enumerate(learner.demos):
            X = np.asarray(X, dtype=float)
            stage_ends = learner.stage_ends_[i]
            starts, ends = _segment_bounds(stage_ends)
            stage_colors = ["orange", "red", "purple", "brown"]
            for k, (s, e) in enumerate(zip(starts, ends)):
                color = stage_colors[k % len(stage_colors)]
                ax1.scatter(X[s : e + 1, 0], X[s : e + 1, 1], c=color, s=4, alpha=0.35)
                ax1.scatter(X[e, 0], X[e, 1], c="blue", marker="x", s=18, linewidths=0.9,
                            label="pred stage end" if i == 0 and k == 0 else "")
            if learner.true_taus[i] is not None:
                tt = int(learner.true_taus[i])
                ax1.scatter(X[tt, 0], X[tt, 1], c="green", marker="x", s=18, linewidths=0.9,
                            label="true boundary" if i == 0 else "")
        if _env_has_xy_obstacle(learner.env):
            cx, cy = learner.env.obs_center
            r = learner.env.obs_radius
            ax1.add_patch(plt.Circle((cx, cy), r, color="gray", fill=False, linestyle="-", label="obstacle"))
        if hasattr(learner.env, "subgoal") and not _is_pickplace(learner.env):
            sg = _xy_point(learner.env.subgoal)
            ax1.scatter(sg[0], sg[1], c="green", marker="*", s=28, label="true subgoal")
        if hasattr(learner.env, "goal") and not _is_pickplace(learner.env):
            gg = _xy_point(learner.env.goal)
            ax1.scatter(gg[0], gg[1], c="green", marker="P", s=28, label="true goal")
        if getattr(learner, "g1", None) is not None:
            g1 = _xy_point(learner.g1)
            ax1.scatter(g1[0], g1[1], c="blue", marker="D", s=24, label="mu1 / g1")
            _draw_completion_ellipse_2d(ax1, learner.g1, learner.end_precision[0], "blue", "stage1 completion")
        if getattr(learner, "g2", None) is not None:
            g2 = _xy_point(learner.g2)
            ax1.scatter(g2[0], g2[1], c="navy", marker="P", s=24, label="mu2 / g2")
            if learner.num_states > 1:
                _draw_completion_ellipse_2d(ax1, learner.g2, learner.end_precision[1], "navy", "stage2 completion")
        if hasattr(learner, "g1_hist") and len(learner.g1_hist) > 1:
            G1 = np.asarray(learner.g1_hist, dtype=float)
            ax1.plot(G1[:, 0], G1[:, 1], color="blue", alpha=0.35, lw=1.0, label="g1 history")
        if hasattr(learner, "g2_hist") and len(learner.g2_hist) > 1:
            G2 = np.asarray(learner.g2_hist, dtype=float)
            ax1.plot(G2[:, 0], G2[:, 1], color="navy", alpha=0.35, lw=1.0, label="g2 history")
        ax1.set_xlabel("x", fontsize=PAPER_LABEL_SIZE)
        ax1.set_ylabel("z" if _is_pickplace(learner.env) else "y", fontsize=PAPER_LABEL_SIZE)
        ax1.set_aspect("equal", adjustable="box")
    ax1.set_title(f"Iter {int(it)}: trajectories & completion centers", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax1.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax1)

    ax2 = fig.add_subplot(2, 2, 2)
    iters = np.arange(len(learner.loss_total))
    ax2.plot(iters, learner.loss_total, color="black", lw=1.3, label="total")
    ax2.plot(iters, learner.loss_constraint, color="tab:red", lw=1.0, label="constraint")
    ax2.plot(iters, learner.loss_end, color="tab:blue", lw=1.0, label="completion")
    ax2.plot(iters, learner.loss_progress, color="tab:orange", lw=1.0, label="progress")
    ax2.set_title("CCP learning curves", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax2.set_xlabel("iteration", fontsize=PAPER_LABEL_SIZE)
    ax2.set_ylabel("objective", fontsize=PAPER_LABEL_SIZE)
    ax2.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    tau_hat0 = int(learner.stage_ends_[0][0]) if learner.num_states >= 2 else int(learner.stage_ends_[0][0])
    _draw_feature_bands(ax3, learner, tau_hat0)

    ax4 = fig.add_subplot(2, 2, 4)
    _draw_boundary_cost_profile(ax4, learner, demo_idx=0)

    save_figure(fig, learner_plot_dir(learner) / f"plot4panel_iter_{int(it):04d}.png", dpi=220)
