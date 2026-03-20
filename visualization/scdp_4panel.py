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


def _segment_bounds(stage_ends):
    starts = []
    ends = []
    prev = -1
    for end in stage_ends:
        starts.append(prev + 1)
        ends.append(int(end))
        prev = int(end)
    return starts, ends


def _feature_name(learner, local_idx):
    schema = getattr(learner, "raw_feature_specs", None) or []
    selected_col = int(learner.selected_feature_columns[local_idx])
    for i, spec in enumerate(schema):
        if int(spec.get("column_idx", i)) == selected_col:
            return str(spec.get("name", f"f{local_idx}"))
    return f"f{local_idx}"


def _xy_point(point):
    arr = np.asarray(point, dtype=float).reshape(-1)
    return arr[:2]


def _draw_trajectories(ax, learner, it):
    for i, X in enumerate(learner.demos):
        X = np.asarray(X, dtype=float)
        stage_ends = learner.stage_ends_[i]
        starts, ends = _segment_bounds(stage_ends)
        colors = ["orange", "red"]
        for k, (s, e) in enumerate(zip(starts, ends)):
            ax.scatter(X[s : e + 1, 0], X[s : e + 1, 1], c=colors[k], s=4, alpha=0.35)
            ax.scatter(
                X[e, 0],
                X[e, 1],
                c="blue" if k == 0 else "navy",
                marker="x",
                s=20,
                linewidths=0.9,
                label="pred stage end" if i == 0 and k == 0 else "",
            )
        if learner.true_taus[i] is not None:
            tt = int(learner.true_taus[i])
            ax.scatter(X[tt, 0], X[tt, 1], c="green", marker="x", s=18, linewidths=0.9, label="true boundary" if i == 0 else "")

    if hasattr(learner.env, "obs_radius") and (hasattr(learner.env, "obs_center") or hasattr(learner.env, "obs_center_xy")):
        center = getattr(learner.env, "obs_center", None)
        if center is None:
            center = getattr(learner.env, "obs_center_xy")
        cx, cy = np.asarray(center, dtype=float).reshape(-1)[:2]
        r = learner.env.obs_radius
        ax.add_patch(plt.Circle((cx, cy), r, color="gray", fill=False, linestyle="-", label="obstacle"))
    if hasattr(learner.env, "subgoal"):
        sg = _xy_point(learner.env.subgoal)
        ax.scatter(sg[0], sg[1], c="green", marker="*", s=28, label="true subgoal")
    if hasattr(learner.env, "goal"):
        gg = _xy_point(learner.env.goal)
        ax.scatter(gg[0], gg[1], c="green", marker="P", s=28, label="true goal")
    if getattr(learner, "g1", None) is not None:
        g1 = _xy_point(learner.g1)
        ax.scatter(g1[0], g1[1], c="blue", marker="D", s=24, label="shared g1")
    if getattr(learner, "g2", None) is not None:
        g2 = _xy_point(learner.g2)
        ax.scatter(g2[0], g2[1], c="navy", marker="P", s=24, label="shared g2")
    if getattr(learner, "current_stage_params_per_demo", None):
        local_g1 = _xy_point(learner.current_stage_params_per_demo[0][0].subgoal)
        ax.scatter(local_g1[0], local_g1[1], c="purple", marker="D", s=22, label="demo0 local g1")

    ax.set_title(f"Iter {int(it)}: trajectories & SCDP goals", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("x", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("y", fontsize=PAPER_LABEL_SIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_learning_curves(ax, learner):
    iters = np.arange(len(learner.loss_total))
    weighted_constraint = learner.lambda_constraint * np.asarray(learner.loss_constraint, dtype=float)
    weighted_progress = learner.lambda_progress * np.asarray(learner.loss_progress, dtype=float)
    weighted_consensus = np.asarray(learner.consensus_lambda_hist[: len(learner.loss_consensus)], dtype=float) * np.asarray(
        learner.loss_consensus,
        dtype=float,
    )
    ax.plot(iters, learner.loss_total, color="black", lw=1.3, label="total")
    ax.plot(iters, weighted_constraint, color="tab:red", lw=1.0, label="constraint")
    ax.plot(iters, weighted_progress, color="tab:orange", lw=1.0, label="progress")
    ax.plot(iters, weighted_consensus, color="tab:purple", lw=1.0, label="consensus")
    ax.set_title("SCDP learning curves", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("iteration", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("objective", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_feature_bands(ax, learner):
    if not getattr(learner, "current_stage_params_per_demo", None):
        ax.axis("off")
        return
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
    local_stage_params = learner.current_stage_params_per_demo[0]
    for k, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(s, e, color=("orange" if k == 0 else "red"), alpha=0.035)
    for m in range(max_features):
        label = _feature_name(learner, m)
        ax.plot(t_axis, raw_values[:, m], lw=1.0, color=colors[m], label=label)
        for k in range(learner.num_states):
            if learner.r[k, m] != 1:
                continue
            summary = local_stage_params[k].model_summaries[m]
            low = summary.get("L")
            high = summary.get("U")
            if low is None or high is None:
                continue
            fid = learner.selected_feature_columns[m]
            low_raw = low * learner.feat_std[fid] + learner.feat_mean[fid]
            high_raw = high * learner.feat_std[fid] + learner.feat_mean[fid]
            s = starts[k]
            e = ends[k]
            ax.fill_between(
                t_axis[s : e + 1],
                low_raw,
                high_raw,
                color=colors[m],
                alpha=0.12 if k == 0 else 0.08,
            )
    tau_hat0 = int(learner.stage_ends_[0][0])
    ax.axvline(tau_hat0, color="black", linestyle="--", lw=1.0, label="pred boundary")
    if learner.true_taus[0] is not None:
        ax.axvline(int(learner.true_taus[0]), color="green", linestyle=":", lw=1.0, label="true boundary")
    ax.set_title("Demo0 local feature constraints", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("time", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("raw feature value", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_cost_profile(ax, learner, demo_idx=0):
    X = learner.demos[demo_idx]
    T = len(X)
    candidate_taus = np.arange(max(1, learner.duration_min[0] - 1), min(T - 2, T - learner.duration_min[1] - 1) + 1)
    if len(candidate_taus) == 0:
        ax.text(0.5, 0.5, "No feasible boundaries under duration bounds.", ha="center", va="center")
        ax.axis("off")
        return

    total = []
    constraint = []
    progress = []
    consensus = []
    r_consensus = []
    for tau in candidate_taus:
        lam_consensus = float(getattr(learner, "current_consensus_lambda", 0.0))
        lam_r_consensus = lam_consensus * float(getattr(learner, "lambda_r_consensus", 0.0))
        info = learner._candidate_cost(
            demo_idx=demo_idx,
            tau=int(tau),
            lam_consensus=lam_consensus,
            lam_r_consensus=lam_r_consensus,
            shared_stage_subgoals=learner.shared_stage_subgoals,
            shared_param_vectors=learner.shared_param_vectors,
            shared_r_mean=getattr(learner, "shared_r_mean", None),
        )
        if info is None:
            total.append(np.nan)
            constraint.append(np.nan)
            progress.append(np.nan)
            consensus.append(np.nan)
            r_consensus.append(np.nan)
            continue
        constraint.append(learner.lambda_constraint * float(info["constraint"]))
        progress.append(learner.lambda_progress * float(info["progress"]))
        consensus.append(lam_consensus * float(info["consensus"]))
        r_consensus.append(lam_r_consensus * float(info.get("r_consensus", 0.0)))
        total.append(float(info["total"]))

    ax.plot(candidate_taus, total, color="black", lw=1.4, label="total")
    ax.plot(candidate_taus, constraint, color="tab:red", lw=1.0, label="constraint")
    ax.plot(candidate_taus, progress, color="tab:orange", lw=1.0, label="progress")
    ax.plot(candidate_taus, consensus, color="tab:purple", lw=1.0, label="consensus")
    if np.any(np.isfinite(np.asarray(r_consensus, dtype=float))):
        ax.plot(candidate_taus, r_consensus, color="tab:brown", lw=1.0, label="r_consensus")

    pred_tau = int(learner.stage_ends_[demo_idx][0])
    ax.axvline(pred_tau, color="black", linestyle="--", lw=1.0, label="pred boundary")
    if learner.true_taus[demo_idx] is not None:
        ax.axvline(int(learner.true_taus[demo_idx]), color="green", linestyle=":", lw=1.0, label="true boundary")
    if getattr(learner, "segmentation_history", None) and len(learner.segmentation_history) >= 2:
        prev_tau = int(learner.segmentation_history[-2][demo_idx][0])
        ax.axvline(prev_tau, color="dimgray", linestyle="-.", lw=1.0, label="prev pred boundary")

    ax.set_title("Demo0 local-tau cost profile", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("boundary index", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("cost", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def plot_scdp_results_4panel(learner, it):
    if plt is None:
        return
    fig = plt.figure(figsize=PAPER_FIGSIZE)

    ax1 = fig.add_subplot(2, 2, 1)
    _draw_trajectories(ax1, learner, it)

    ax2 = fig.add_subplot(2, 2, 2)
    _draw_learning_curves(ax2, learner)

    ax3 = fig.add_subplot(2, 2, 3)
    _draw_feature_bands(ax3, learner)

    ax4 = fig.add_subplot(2, 2, 4)
    _draw_cost_profile(ax4, learner, demo_idx=0)

    save_figure(fig, learner_plot_dir(learner) / f"plot4panel_iter_{int(it):04d}.png", dpi=220)
