from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from .io import learner_plot_dir, save_figure
from utils.models import GaussianModel

PAPER_FIGSIZE = (8.4, 6.0)
PAPER_TITLE_SIZE = 9
PAPER_LABEL_SIZE = 8
PAPER_TICK_SIZE = 7
PAPER_LEGEND_SIZE = 6.5
STAGE_COLORS = ["#D55E00", "#0072B2", "#CC79A7", "#009E73"]


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


def _score_threshold(learner, feat_idx):
    if hasattr(learner, "_is_equality_feature") and learner._is_equality_feature(int(feat_idx)):
        return float(getattr(learner, "equality_dispersion_ratio_threshold", 0.1))
    return float(getattr(learner, "inequality_score_activation_threshold", -0.5))


def _matrix_text_color(value, vmax):
    if not np.isfinite(value):
        return "black"
    threshold = 0.55 * float(max(vmax, 1e-6))
    return "white" if abs(float(value)) >= threshold else "black"


def _feature_stage_is_active_for_display(learner, local_stage_params, stage_idx, feat_idx):
    if getattr(learner, "feature_activation_mode", "fixed_mask") == "score":
        try:
            score = float(local_stage_params[stage_idx].feature_scores[feat_idx])
        except Exception:
            return False
        if not np.isfinite(score):
            return False
        return (_score_threshold(learner, feat_idx) - score) > 0.0
    try:
        return int(learner.r[stage_idx, feat_idx]) == 1
    except Exception:
        return False


def _xy_point(point):
    arr = np.asarray(point, dtype=float).reshape(-1)
    return arr[:2]


def _true_cutpoints_for_demo(learner, demo_idx):
    true_cutpoints = getattr(learner, "true_cutpoints", None)
    if true_cutpoints is not None and demo_idx < len(true_cutpoints):
        cuts = true_cutpoints[demo_idx]
        if cuts is None:
            return []
        return [int(x) for x in np.asarray(cuts, dtype=int).reshape(-1).tolist()]
    if getattr(learner, "true_taus", None) is not None and learner.true_taus[demo_idx] is not None:
        return [int(learner.true_taus[demo_idx])]
    return []


def _stage_colors(num_states):
    num_states = int(num_states)
    return [STAGE_COLORS[i % len(STAGE_COLORS)] for i in range(max(num_states, 1))]


def _shortest_coverage_width(values, coverage: float = 0.7) -> float:
    xs = np.sort(np.asarray(values, dtype=float).reshape(-1))
    n = xs.size
    if n == 0:
        return np.nan
    if n == 1:
        return 0.0
    coverage = float(np.clip(coverage, 1e-6, 1.0))
    window = max(int(np.ceil(coverage * n)), 1)
    if window >= n:
        return float(xs[-1] - xs[0])
    widths = xs[window - 1 :] - xs[: n - window + 1]
    return float(np.min(widths))


def _mean_abs_centered_dispersion(values) -> float:
    xs = np.asarray(values, dtype=float).reshape(-1)
    if xs.size == 0:
        return np.nan
    center = float(np.median(xs))
    return float(np.mean(np.abs(xs - center)))


def _finite_1d(values):
    xs = np.asarray(values, dtype=float).reshape(-1)
    return xs[np.isfinite(xs)]


def _safe_hist(ax, values, *, max_bins, min_bins, color, alpha, label):
    xs = _finite_1d(values)
    if xs.size == 0:
        return
    if xs.size == 1:
        ax.axvline(float(xs[0]), color=color, alpha=max(alpha, 0.45), lw=2.0, label=label)
        return
    data_min = float(np.min(xs))
    data_max = float(np.max(xs))
    span = float(data_max - data_min)
    if span < 1e-8:
        pad = 1e-3
        bins = 3
        hist_range = (data_min - pad, data_max + pad)
    else:
        bins = int(min(max(xs.size // 2, min_bins), max_bins))
        bins = max(bins, 2)
        hist_range = (data_min, data_max)
    ax.hist(
        xs,
        bins=bins,
        range=hist_range,
        density=False,
        alpha=alpha,
        color=color,
        label=label,
    )


def _draw_trajectories(ax, learner, it, demo_idx=0):
    X = np.asarray(learner.demos[demo_idx], dtype=float)
    stage_ends = learner.stage_ends_[demo_idx]
    starts, ends = _segment_bounds(stage_ends)
    colors = _stage_colors(learner.num_states)
    for k, (s, e) in enumerate(zip(starts, ends)):
        ax.scatter(X[s : e + 1, 0], X[s : e + 1, 1], color=colors[k], s=5, alpha=0.45)
    for j, tt in enumerate(_true_cutpoints_for_demo(learner, demo_idx)):
        ax.scatter(X[tt, 0], X[tt, 1], color="black", marker="x", s=20, linewidths=0.9, label="true boundary" if j == 0 else "")

    if hasattr(learner.env, "obs_radius") and (hasattr(learner.env, "obs_center") or hasattr(learner.env, "obs_center_xy")):
        center = getattr(learner.env, "obs_center", None)
        if center is None:
            center = getattr(learner.env, "obs_center_xy")
        cx, cy = np.asarray(center, dtype=float).reshape(-1)[:2]
        r = learner.env.obs_radius
        ax.add_patch(plt.Circle((cx, cy), r, color="gray", fill=False, linestyle="-", label="obstacle"))
    if hasattr(learner.env, "subgoal"):
        sg = _xy_point(learner.env.subgoal)
        ax.scatter(sg[0], sg[1], color="black", marker="X", s=28, label="true stage end")
    if hasattr(learner.env, "goal"):
        gg = _xy_point(learner.env.goal)
        ax.scatter(gg[0], gg[1], color="black", marker="X", s=28, label="true stage end")
    for k, sg in enumerate(getattr(learner, "stage_subgoals", []) or []):
        pt = _xy_point(sg)
        ax.scatter(
            pt[0], pt[1], color=colors[k], marker="D", s=30,
            edgecolors="black", linewidths=0.9, label=f"shared stage {k + 1} end"
        )

    ax.set_title(f"Iter {int(it)}: demo {demo_idx} trajectory & goals", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("x", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("y", fontsize=PAPER_LABEL_SIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_trajectories_overview(ax, learner, it):
    colors = _stage_colors(learner.num_states)
    for i, X in enumerate(learner.demos):
        X = np.asarray(X, dtype=float)
        stage_ends = learner.stage_ends_[i]
        starts, ends = _segment_bounds(stage_ends)
        for k, (s, e) in enumerate(zip(starts, ends)):
            ax.scatter(X[s : e + 1, 0], X[s : e + 1, 1], color=colors[k], s=4, alpha=0.35)
        for j, tt in enumerate(_true_cutpoints_for_demo(learner, i)):
            ax.scatter(X[tt, 0], X[tt, 1], color="black", marker="x", s=18, linewidths=0.9, label="true boundary" if i == 0 and j == 0 else "")

    if hasattr(learner.env, "obs_radius") and (hasattr(learner.env, "obs_center") or hasattr(learner.env, "obs_center_xy")):
        center = getattr(learner.env, "obs_center", None)
        if center is None:
            center = getattr(learner.env, "obs_center_xy")
        cx, cy = np.asarray(center, dtype=float).reshape(-1)[:2]
        r = learner.env.obs_radius
        ax.add_patch(plt.Circle((cx, cy), r, color="gray", fill=False, linestyle="-", label="obstacle"))
    if hasattr(learner.env, "subgoal"):
        sg = _xy_point(learner.env.subgoal)
        ax.scatter(sg[0], sg[1], color="black", marker="X", s=28, label="true stage end")
    if hasattr(learner.env, "goal"):
        gg = _xy_point(learner.env.goal)
        ax.scatter(gg[0], gg[1], color="black", marker="X", s=28, label="true stage end")
    for k, sg in enumerate(getattr(learner, "stage_subgoals", []) or []):
        pt = _xy_point(sg)
        ax.scatter(
            pt[0], pt[1], color=colors[k], marker="D", s=30,
            edgecolors="black", linewidths=0.9, label=f"shared stage {k + 1} end"
        )

    ax.set_title(f"Iter {int(it)}: trajectories & SCDP goals", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("x", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("y", fontsize=PAPER_LABEL_SIZE)
    ax.set_aspect("equal", adjustable="box")
    ax.set_ylim(-0.5, 0.5)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_learning_curves(ax, learner):
    iters = np.arange(len(learner.loss_total))
    weighted_constraint = np.asarray(learner.loss_constraint, dtype=float)
    weighted_progress = learner.lambda_progress * np.asarray(learner.loss_progress, dtype=float)
    weighted_subgoal_consensus = (
        np.asarray(learner.subgoal_consensus_lambda_hist[: len(learner.loss_subgoal_consensus)], dtype=float)
        * np.asarray(learner.loss_subgoal_consensus, dtype=float)
    )
    weighted_param_consensus = (
        np.asarray(learner.param_consensus_lambda_hist[: len(learner.loss_param_consensus)], dtype=float)
        * np.asarray(learner.loss_param_consensus, dtype=float)
    )
    weighted_feature_score_consensus = (
        np.asarray(learner.feature_score_consensus_lambda_hist[: len(learner.loss_feature_score_consensus)], dtype=float)
        * np.asarray(learner.loss_feature_score_consensus, dtype=float)
    )
    ax.plot(iters, learner.loss_total, color="black", lw=1.3, label="total")
    ax.plot(iters, weighted_constraint, color="tab:red", lw=1.0, label="constraint")
    ax.plot(iters, weighted_progress, color="tab:orange", lw=1.0, label="progress")
    ax.plot(iters, weighted_subgoal_consensus, color="tab:purple", lw=1.0, label="subgoal_consensus")
    ax.plot(iters, weighted_param_consensus, color="tab:cyan", lw=1.0, label="param_consensus")
    ax.plot(iters, weighted_feature_score_consensus, color="tab:brown", lw=1.0, label="feature_score_consensus")
    ax.set_title("SCDP learning curves", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("iteration", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("objective", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_constraint_cost_matrix(ax, learner, demo_idx=0):
    if not getattr(learner, "current_stage_params_per_demo", None):
        ax.axis("off")
        return
    raw_score_matrix = np.asarray(
        [stage_params.feature_scores for stage_params in learner.current_stage_params_per_demo[demo_idx]],
        dtype=float,
    ).T
    if raw_score_matrix.size == 0:
        ax.axis("off")
        return
    threshold_vec = np.asarray([_score_threshold(learner, i) for i in range(raw_score_matrix.shape[0])], dtype=float)
    score_margin_matrix = threshold_vec[:, None] - raw_score_matrix
    vmax = float(np.nanmax(np.abs(score_margin_matrix))) if score_margin_matrix.size > 0 else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    im = ax.imshow(score_margin_matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    stage_labels = [f"s{i + 1}" for i in range(score_margin_matrix.shape[1])]
    feature_labels = [_feature_name(learner, i) for i in range(score_margin_matrix.shape[0])]
    ax.set_title(f"Demo {demo_idx} constraint margin", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xticks(range(score_margin_matrix.shape[1]))
    ax.set_xticklabels(stage_labels)
    ax.set_yticks(range(score_margin_matrix.shape[0]))
    ax.set_yticklabels(feature_labels)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    for i in range(score_margin_matrix.shape[0]):
        for j in range(score_margin_matrix.shape[1]):
            value = float(score_margin_matrix[i, j])
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                color=_matrix_text_color(value, vmax),
                fontsize=8,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_feature_bands(ax, learner, demo_idx=0):
    if not getattr(learner, "current_stage_params_per_demo", None):
        ax.axis("off")
        return
    X0 = learner.demos[demo_idx]
    T0 = len(X0)
    t_axis = np.arange(T0)
    Fz = learner.standardized_features[demo_idx]
    raw_values = (
        Fz * learner.feat_std[learner.selected_feature_columns][None, :]
        + learner.feat_mean[learner.selected_feature_columns][None, :]
    )
    max_features = min(learner.num_features, 6)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max_features))
    stage_ends = learner.stage_ends_[demo_idx]
    starts, ends = _segment_bounds(stage_ends)
    local_stage_params = learner.current_stage_params_per_demo[demo_idx]
    for m in range(max_features):
        label = _feature_name(learner, m)
        ax.plot(t_axis, raw_values[:, m], lw=1.0, color=colors[m], label=label)
        for k in range(learner.num_states):
            if not _feature_stage_is_active_for_display(learner, local_stage_params, k, m):
                continue
            summary = local_stage_params[k].model_summaries[m]
            fid = learner.selected_feature_columns[m]
            s = starts[k]
            e = ends[k]
            stage_vals_raw = np.asarray(raw_values[s : e + 1, m], dtype=float)
            if learner._is_equality_feature(m):
                stage_median_raw = float(np.median(stage_vals_raw))
                stage_std_raw = max(float(np.std(stage_vals_raw)), 0.05)
                low_raw = stage_median_raw - stage_std_raw
                high_raw = stage_median_raw + stage_std_raw
            else:
                low = summary.get("L")
                high = summary.get("U")
                if low is None or high is None:
                    continue
                low_raw = low * learner.feat_std[fid] + learner.feat_mean[fid]
                high_raw = high * learner.feat_std[fid] + learner.feat_mean[fid]
            ax.fill_between(
                t_axis[s : e + 1],
                low_raw,
                high_raw,
                color=colors[m],
                alpha=0.18 if k == 0 else 0.14,
                zorder=1,
            )
            ax.plot(t_axis[s : e + 1], np.full(e - s + 1, low_raw), color=colors[m], lw=0.8, alpha=0.9, zorder=2)
            ax.plot(t_axis[s : e + 1], np.full(e - s + 1, high_raw), color=colors[m], lw=0.8, alpha=0.9, zorder=2)
    for j, cp in enumerate(learner.stage_ends_[demo_idx][:-1]):
        ax.axvline(int(cp), color="black", linestyle="--", lw=1.0, label="pred boundary" if j == 0 else "")
    for j, cp in enumerate(_true_cutpoints_for_demo(learner, demo_idx)):
        ax.axvline(int(cp), color="green", linestyle=":", lw=1.0, label="true boundary" if j == 0 else "")
    ax.set_title(f"Demo {demo_idx} local feature constraints", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("time", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("raw feature value", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_cost_profile(ax, learner, demo_idx=0):
    if learner.num_states != 2:
        if not getattr(learner, "current_stage_params_per_demo", None):
            ax.axis("off")
            return
        stage_labels = [f"s{k + 1}" for k in range(learner.num_states)]
        stage_ends = learner.stage_ends_[demo_idx]
        starts, ends = _segment_bounds(stage_ends)
        seg_lengths = np.asarray([e - s + 1 for s, e in zip(starts, ends)], dtype=float)
        stage_constraint = []
        stage_progress = []
        for stage_idx, (s, e) in enumerate(zip(starts, ends)):
            _, c_cost, p_cost = learner._fit_segment_stage(demo_idx, stage_idx, s, e)
            stage_constraint.append(float(c_cost))
            stage_progress.append(learner.lambda_progress * float(p_cost))
        x = np.arange(learner.num_states)
        ax.bar(x - 0.22, seg_lengths, width=0.22, color="tab:gray", label="segment length")
        ax.bar(x, stage_constraint, width=0.22, color="tab:red", label="constraint")
        ax.bar(x + 0.22, stage_progress, width=0.22, color="tab:orange", label="progress")
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels)
        ax.set_title(f"Demo {demo_idx} per-stage summary", fontsize=PAPER_TITLE_SIZE, pad=4)
        ax.set_xlabel("stage", fontsize=PAPER_LABEL_SIZE)
        ax.set_ylabel("value", fontsize=PAPER_LABEL_SIZE)
        ax.tick_params(labelsize=PAPER_TICK_SIZE)
        _legend(ax)
        return
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
    subgoal_consensus = []
    param_consensus = []
    feature_score_consensus = []
    for tau in candidate_taus:
        lam_subgoal_consensus = float(getattr(learner, "current_subgoal_consensus_lambda", 0.0))
        lam_param_consensus = float(getattr(learner, "current_param_consensus_lambda", 0.0))
        lam_feature_score_consensus = float(getattr(learner, "current_feature_score_consensus_lambda", 0.0))
        info = learner._candidate_cost(
            demo_idx=demo_idx,
            stage_ends=[int(tau), int(T - 1)],
            lam_subgoal_consensus=lam_subgoal_consensus,
            lam_param_consensus=lam_param_consensus,
            lam_feature_score_consensus=lam_feature_score_consensus,
            shared_stage_subgoals=learner.shared_stage_subgoals,
            shared_param_vectors=learner.shared_param_vectors,
            shared_r_mean=getattr(learner, "shared_r_mean", None),
            shared_feature_score_mean=getattr(learner, "shared_feature_score_mean", None),
        )
        if info is None:
            total.append(np.nan)
            constraint.append(np.nan)
            progress.append(np.nan)
            subgoal_consensus.append(np.nan)
            param_consensus.append(np.nan)
            feature_score_consensus.append(np.nan)
            continue
        constraint.append(float(info["constraint"]))
        progress.append(learner.lambda_progress * float(info["progress"]))
        subgoal_consensus.append(lam_subgoal_consensus * float(info["subgoal_consensus"]))
        param_consensus.append(lam_param_consensus * float(info["param_consensus"]))
        feature_score_consensus.append(lam_feature_score_consensus * float(info.get("feature_score_consensus", 0.0)))
        total.append(float(info["total"]))

    ax.plot(candidate_taus, total, color="black", lw=1.4, label="total")
    ax.plot(candidate_taus, constraint, color="tab:red", lw=1.0, label="constraint")
    ax.plot(candidate_taus, progress, color="tab:orange", lw=1.0, label="progress")
    if np.any(np.isfinite(np.asarray(subgoal_consensus, dtype=float))):
        ax.plot(candidate_taus, subgoal_consensus, color="tab:purple", lw=1.0, label="subgoal_consensus")
    if np.any(np.isfinite(np.asarray(param_consensus, dtype=float))):
        ax.plot(candidate_taus, param_consensus, color="tab:cyan", lw=1.0, label="param_consensus")
    if np.any(np.isfinite(np.asarray(feature_score_consensus, dtype=float))):
        ax.plot(candidate_taus, feature_score_consensus, color="tab:brown", lw=1.0, label="feature_score_consensus")

    pred_tau = int(learner.stage_ends_[demo_idx][0])
    ax.axvline(pred_tau, color="black", linestyle="--", lw=1.0, label="pred boundary")
    for j, cp in enumerate(_true_cutpoints_for_demo(learner, demo_idx)):
        ax.axvline(int(cp), color="green", linestyle=":", lw=1.0, label="true boundary" if j == 0 else "")
    if getattr(learner, "segmentation_history", None) and len(learner.segmentation_history) >= 2:
        prev_tau = int(learner.segmentation_history[-2][demo_idx][0])
        ax.axvline(prev_tau, color="dimgray", linestyle="-.", lw=1.0, label="prev pred boundary")

    ax.set_title(f"Demo {demo_idx} local-tau cost profile", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("boundary index", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("cost", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _current_consensus_lambdas(learner):
    return (
        float(getattr(learner, "current_subgoal_consensus_lambda", 0.0)),
        float(getattr(learner, "current_param_consensus_lambda", 0.0)),
        float(getattr(learner, "current_feature_score_consensus_lambda", 0.0)),
    )


def _scan_cutpoint_range(learner, T, fixed_cp, vary_index):
    duration_min = np.asarray(learner.duration_min, dtype=int)
    duration_max = np.asarray(learner.duration_max, dtype=int)
    if vary_index == 1:
        cp1 = int(fixed_cp)
        low = max(cp1 + int(duration_min[1]), int(T - 1 - duration_max[2]))
        high = min(cp1 + int(duration_max[1]), int(T - 1 - duration_min[2]))
    else:
        cp2 = int(fixed_cp)
        low = max(int(duration_min[0] - 1), cp2 - int(duration_max[1]), 0)
        high = min(int(duration_max[0] - 1), cp2 - int(duration_min[1]), cp2 - 1)
    if high < low:
        return np.asarray([], dtype=int)
    return np.arange(low, high + 1, dtype=int)


def _gt_shared_true_cut_cost_breakdown(learner, demo_idx):
    all_true_cutpoints = [_true_cutpoints_for_demo(learner, i) for i in range(len(learner.demos))]
    if any(len(cuts) != learner.num_states - 1 for cuts in all_true_cutpoints):
        return None
    lam_subgoal_consensus, lam_param_consensus, lam_feature_score_consensus = _current_consensus_lambdas(learner)

    zero_shared_subgoals = [np.zeros_like(np.asarray(learner.stage_subgoals[k], dtype=float)) for k in range(learner.num_states)]
    zero_shared_params = [[None for _ in range(learner.num_features)] for _ in range(learner.num_states)]
    true_infos = []
    for cur_demo_idx, true_cutpoints in enumerate(all_true_cutpoints):
        T_cur = len(learner.demos[cur_demo_idx])
        info = learner._candidate_cost(
            demo_idx=cur_demo_idx,
            stage_ends=[int(x) for x in true_cutpoints] + [int(T_cur - 1)],
            lam_subgoal_consensus=0.0,
            lam_param_consensus=0.0,
            lam_feature_score_consensus=0.0,
            shared_stage_subgoals=zero_shared_subgoals,
            shared_param_vectors=zero_shared_params,
            shared_r_mean=None,
            shared_feature_score_mean=None,
        )
        if info is None:
            return None
        true_infos.append(info)

    gt_shared_stage_subgoals, gt_shared_param_vectors = learner._shared_from_selected(true_infos)
    gt_shared_feature_score_mean = None
    gt_shared_r_mean = None
    if getattr(learner, "use_score_mode", False):
        gt_shared_feature_score_mean = np.mean(
            np.stack([
                np.stack([stage_params.feature_scores for stage_params in info["stage_params"]], axis=0)
                for info in true_infos
            ], axis=0),
            axis=0,
        )
    else:
        gt_shared_r_mean = np.mean(
            np.stack([
                np.stack([stage_params.active_mask for stage_params in info["stage_params"]], axis=0)
                for info in true_infos
            ], axis=0),
            axis=0,
        )

    true_cutpoints = all_true_cutpoints[demo_idx]
    T = len(learner.demos[demo_idx])
    info = learner._candidate_cost(
        demo_idx=demo_idx,
        stage_ends=[int(x) for x in true_cutpoints] + [int(T - 1)],
        lam_subgoal_consensus=lam_subgoal_consensus,
        lam_param_consensus=lam_param_consensus,
        lam_feature_score_consensus=lam_feature_score_consensus,
        shared_stage_subgoals=gt_shared_stage_subgoals,
        shared_param_vectors=gt_shared_param_vectors,
        shared_r_mean=gt_shared_r_mean,
        shared_feature_score_mean=gt_shared_feature_score_mean,
    )
    if info is None:
        return None
    return {
        "cutpoints": [int(x) for x in true_cutpoints],
        "total": float(info["total"]),
        "constraint": float(info["constraint"]),
        "progress": learner.lambda_progress * float(info["progress"]),
        "subgoal_consensus": lam_subgoal_consensus * float(info["subgoal_consensus"]),
        "param_consensus": lam_param_consensus * float(info["param_consensus"]),
        "feature_score_consensus": lam_feature_score_consensus * float(info.get("feature_score_consensus", 0.0)),
    }
    

def _annotate_true_cut_costs(ax, learner, demo_idx, vary_index, show_components):
    breakdown = _gt_shared_true_cut_cost_breakdown(learner, demo_idx)
    if breakdown is None:
        return
    true_cutpoints = breakdown["cutpoints"]
    if vary_index >= len(true_cutpoints):
        return
    x_true = int(true_cutpoints[vary_index])
    series = [
        ("total", "black"),
        ("constraint", "tab:red"),
        ("progress", "tab:orange"),
        ("subgoal_consensus", "tab:purple"),
        ("param_consensus", "tab:cyan"),
        ("feature_score_consensus", "tab:brown"),
    ]
    plotted_lines = [("total", "black")] if not show_components else series
    for key, color in plotted_lines:
        y_val = float(breakdown[key])
        if np.isfinite(y_val):
            ax.scatter(
                [x_true], [y_val],
                color=color,
                marker="X",
                s=30,
                edgecolors="black",
                linewidths=0.5,
                zorder=4,
                label="_nolegend_",
            )

    info_lines = [f"GT-shared true cuts total={breakdown['total']:.2f}"]
    if show_components:
        info_lines.extend([
            f"c={breakdown['constraint']:.2f}",
            f"p={breakdown['progress']:.2f}",
            f"sg={breakdown['subgoal_consensus']:.2f}",
            f"pm={breakdown['param_consensus']:.2f}",
            f"fs={breakdown['feature_score_consensus']:.2f}",
        ])
    ax.text(
        0.98,
        0.98,
        "\n".join(info_lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="0.6"),
    )


def _local_stage_cost_breakdown(learner, demo_idx, stage_ends):
    starts, ends = _segment_bounds(stage_ends)
    feature_constraint = np.zeros((learner.num_features, learner.num_states), dtype=float)
    progress = np.zeros(learner.num_states, dtype=float)
    for stage_idx, (s, e) in enumerate(zip(starts, ends)):
        stage_params, _, progress_cost = learner._fit_segment_stage(demo_idx, stage_idx, s, e)
        contrib = np.asarray(getattr(stage_params, "feature_constraint_costs", np.zeros(learner.num_features)), dtype=float)
        feature_constraint[:, stage_idx] = contrib
        progress[stage_idx] = learner.lambda_progress * float(progress_cost)
    return feature_constraint, progress


def _draw_learned_vs_true_local_cost_delta(fig, axes, learner, demo_idx=0):
    true_cutpoints = _true_cutpoints_for_demo(learner, demo_idx)
    if len(true_cutpoints) != learner.num_states - 1:
        for ax in axes:
            ax.axis("off")
        return

    learned_stage_ends = [int(x) for x in learner.stage_ends_[demo_idx]]
    true_stage_ends = [int(x) for x in true_cutpoints] + [int(len(learner.demos[demo_idx]) - 1)]
    learned_feature_constraint, learned_progress = _local_stage_cost_breakdown(learner, demo_idx, learned_stage_ends)
    true_feature_constraint, true_progress = _local_stage_cost_breakdown(learner, demo_idx, true_stage_ends)

    delta_constraint = learned_feature_constraint - true_feature_constraint
    delta_progress = learned_progress - true_progress

    ax1, ax2 = axes
    vmax = float(np.nanmax(np.abs(delta_constraint))) if delta_constraint.size > 0 else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    im = ax1.imshow(delta_constraint, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax1.set_title(f"Demo {demo_idx} learned-true constraint cost", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax1.set_xticks(range(learner.num_states))
    ax1.set_xticklabels([f"s{k + 1}" for k in range(learner.num_states)])
    ax1.set_yticks(range(learner.num_features))
    ax1.set_yticklabels([_feature_name(learner, i) for i in range(learner.num_features)])
    ax1.tick_params(labelsize=PAPER_TICK_SIZE)
    for i in range(delta_constraint.shape[0]):
        for j in range(delta_constraint.shape[1]):
            value = float(delta_constraint[i, j])
            ax1.text(j, i, f"{value:.2f}", ha="center", va="center", color=_matrix_text_color(value, vmax), fontsize=8)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    x = np.arange(learner.num_states)
    bar_colors = [_stage_colors(learner.num_states)[k] for k in range(learner.num_states)]
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.bar(x, delta_progress, color=bar_colors, width=0.6)
    for k, value in enumerate(delta_progress):
        ax2.text(k, float(value), f"{float(value):.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    ax2.set_title(f"Demo {demo_idx} learned-true progress cost", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"s{k + 1}" for k in range(learner.num_states)])
    ax2.set_xlabel("stage", fontsize=PAPER_LABEL_SIZE)
    ax2.set_ylabel("weighted progress delta", fontsize=PAPER_LABEL_SIZE)
    ax2.tick_params(labelsize=PAPER_TICK_SIZE)


def _stage_params_for_stage_ends(learner, demo_idx, stage_ends):
    starts, ends = _segment_bounds(stage_ends)
    out = []
    for stage_idx, (s, e) in enumerate(zip(starts, ends)):
        stage_params, _, _ = learner._fit_segment_stage(demo_idx, stage_idx, s, e)
        out.append(stage_params)
    return out, starts, ends


def _plot_cutpoint_feature_distribution_compare(learner, it, demo_idx=0, vary_index=1):
    if plt is None or learner.num_states != 3:
        return
    true_cutpoints = _true_cutpoints_for_demo(learner, demo_idx)
    if len(true_cutpoints) != 2:
        return

    learned_stage_ends = [int(x) for x in learner.stage_ends_[demo_idx]]
    learned_cp1, learned_cp2 = int(learned_stage_ends[0]), int(learned_stage_ends[1])
    true_cp1, true_cp2 = int(true_cutpoints[0]), int(true_cutpoints[1])
    T = len(learner.demos[demo_idx])

    if vary_index == 1:
        compare_stage_ends = [
            ("learned cp2", [learned_cp1, learned_cp2, T - 1]),
            ("true cp2", [learned_cp1, true_cp2, T - 1]),
        ]
        stage_indices = [1, 2]
        filename = f"plot_cp2_compare_demo_{int(demo_idx):02d}_iter_{int(it):04d}.png"
        title = f"Demo {demo_idx} | fixed cp1={learned_cp1} | learned vs true cp2"
    else:
        compare_stage_ends = [
            ("learned cp1", [learned_cp1, learned_cp2, T - 1]),
            ("true cp1", [true_cp1, learned_cp2, T - 1]),
        ]
        stage_indices = [0, 1]
        filename = f"plot_cp1_compare_demo_{int(demo_idx):02d}_iter_{int(it):04d}.png"
        title = f"Demo {demo_idx} | fixed cp2={learned_cp2} | learned vs true cp1"

    valid_rows = []
    for label, stage_ends in compare_stage_ends:
        stage_ends = [int(x) for x in stage_ends]
        if not (0 <= stage_ends[0] < stage_ends[1] < stage_ends[2] == T - 1):
            continue
        row_stage_params, starts, ends = _stage_params_for_stage_ends(learner, demo_idx, stage_ends)
        valid_rows.append((label, stage_ends, row_stage_params, starts, ends))
    if not valid_rows:
        return

    F = np.asarray(learner.standardized_features[demo_idx], dtype=float)
    feature_names = [_feature_name(learner, feat_idx) for feat_idx in range(learner.num_features)]
    n_rows = len(valid_rows) * len(stage_indices)
    fig, axes = plt.subplots(
        n_rows,
        learner.num_features,
        figsize=(4.7 * learner.num_features, 3.6 * n_rows),
        squeeze=False,
    )

    row_idx = 0
    for scenario_label, stage_ends, stage_params_list, starts, ends in valid_rows:
        for stage_idx in stage_indices:
            s = int(starts[stage_idx])
            e = int(ends[stage_idx])
            vals_by_feat = np.asarray(F[s : e + 1], dtype=float)
            stage_params = stage_params_list[stage_idx]
            for feat_idx, kind in enumerate(learner.feature_model_types):
                ax = axes[row_idx][feat_idx]
                vals = np.asarray(vals_by_feat[:, feat_idx], dtype=float)
                full_vals = np.asarray(F[:, feat_idx], dtype=float)
                summary = stage_params.model_summaries[feat_idx]
                fitted_model = learner._vector_to_model(kind, learner._summary_to_vector(kind, summary))
                baseline_model = GaussianModel(
                    mu=float(np.mean(vals)),
                    sigma=float(max(np.std(vals), 1e-6)),
                )

                lo = float(min(np.min(vals), np.min(full_vals), getattr(fitted_model, "L", np.min(vals)), getattr(baseline_model, "L", np.min(vals))))
                hi = float(max(np.max(vals), np.max(full_vals), getattr(fitted_model, "U", np.max(vals)), getattr(baseline_model, "U", np.max(vals))))
                pad = max(0.15 * (hi - lo + 1e-6), 0.2)
                xs = np.linspace(lo - pad, hi + pad, 300)

                _safe_hist(
                    ax,
                    full_vals,
                    max_bins=24,
                    min_bins=10,
                    color="tab:gray",
                    alpha=0.18,
                    label="full demo",
                )
                _safe_hist(
                    ax,
                    vals,
                    max_bins=16,
                    min_bins=6,
                    color="tab:blue",
                    alpha=0.35,
                    label="segment",
                )
                ax.plot(xs, np.exp(fitted_model.logpdf(xs)), color="tab:red", lw=2.0, label="fitted")
                ax.plot(xs, np.exp(baseline_model.logpdf(xs)), color="tab:green", lw=2.0, linestyle="--", label="gaussian baseline")
                ax.axvline(float(np.mean(vals)), color="tab:blue", lw=1.0, linestyle=":", alpha=0.8)
                ax.axvline(float(np.mean(full_vals)), color="tab:gray", lw=1.0, linestyle="-.", alpha=0.9)

                fitted_neglog = -np.asarray(fitted_model.logpdf(vals), dtype=float)
                baseline_neglog = -np.asarray(baseline_model.logpdf(vals), dtype=float)
                fitted_step = float(np.mean(fitted_neglog))
                baseline_step = float(np.mean(baseline_neglog))
                raw_score = float(stage_params.feature_scores[feat_idx])
                weighted_cost = float(np.asarray(stage_params.feature_constraint_costs, dtype=float)[feat_idx])
                info_lines = [
                    f"steps = {len(vals)}",
                    f"fitted avg NLL = {fitted_step:.3f}",
                    f"baseline avg NLL = {baseline_step:.3f}",
                    f"avg NLL gain = {baseline_step - fitted_step:.3f}",
                    f"raw score = {raw_score:.3f}",
                    f"weighted cost = {weighted_cost:.3f}",
                ]
                if learner._is_equality_feature(feat_idx):
                    stage_dispersion = float(_mean_abs_centered_dispersion(vals))
                    uncertainty_bonus = float(
                        getattr(learner, "equality_dispersion_uncertainty_c", 0.1) / np.sqrt(max(len(vals), 1))
                    )
                    info_lines.append(f"local dispersion = {stage_dispersion:.3f}")
                    info_lines.append(f"uncertainty bonus = {uncertainty_bonus:.3f}")
                    info_lines.append(f"adjusted score = {stage_dispersion + uncertainty_bonus:.3f}")
                    info_lines.append(f"threshold = {float(getattr(learner, 'equality_dispersion_ratio_threshold', 0.1)):.3f}")

                ax.set_title(f"{scenario_label} | stage {stage_idx + 1} | {feature_names[feat_idx]}", fontsize=PAPER_TITLE_SIZE, pad=4)
                ax.set_xlabel("standardized feature value", fontsize=PAPER_LABEL_SIZE)
                ax.set_ylabel("count / density", fontsize=PAPER_LABEL_SIZE)
                ax.tick_params(labelsize=PAPER_TICK_SIZE)
                ax.text(
                    0.02,
                    0.98,
                    "\n".join(info_lines),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
                )
                _legend(ax)
            row_idx += 1

    fig.suptitle(title, fontsize=12)
    save_figure(fig, learner_plot_dir(learner) / filename, dpi=190)


def _draw_two_cut_scan(ax, learner, demo_idx=0, vary_index=1, show_components=True):
    if learner.num_states != 3:
        ax.axis("off")
        return

    stage_ends = learner.stage_ends_[demo_idx]
    cp1 = int(stage_ends[0])
    cp2 = int(stage_ends[1])
    T = len(learner.demos[demo_idx])
    candidate_values = _scan_cutpoint_range(learner, T, cp1 if vary_index == 1 else cp2, vary_index)
    if candidate_values.size == 0:
        ax.text(0.5, 0.5, "No feasible cutpoints under duration bounds.", ha="center", va="center")
        ax.axis("off")
        return

    lam_subgoal_consensus, lam_param_consensus, lam_feature_score_consensus = _current_consensus_lambdas(learner)
    total = []
    constraint = []
    progress = []
    subgoal_consensus = []
    param_consensus = []
    feature_score_consensus = []
    feature_constraint_by_feat = [[] for _ in range(learner.num_features)]

    for value in candidate_values:
        cand_cp1 = int(value) if vary_index == 0 else cp1
        cand_cp2 = cp2 if vary_index == 0 else int(value)
        if cand_cp1 >= cand_cp2:
            total.append(np.nan)
            constraint.append(np.nan)
            progress.append(np.nan)
            subgoal_consensus.append(np.nan)
            param_consensus.append(np.nan)
            feature_score_consensus.append(np.nan)
            for feat_values in feature_constraint_by_feat:
                feat_values.append(np.nan)
            continue

        info = learner._candidate_cost(
            demo_idx=demo_idx,
            stage_ends=[cand_cp1, cand_cp2, int(T - 1)],
            lam_subgoal_consensus=lam_subgoal_consensus,
            lam_param_consensus=lam_param_consensus,
            lam_feature_score_consensus=lam_feature_score_consensus,
            shared_stage_subgoals=learner.shared_stage_subgoals,
            shared_param_vectors=learner.shared_param_vectors,
            shared_r_mean=getattr(learner, "shared_r_mean", None),
            shared_feature_score_mean=getattr(learner, "shared_feature_score_mean", None),
        )
        if info is None:
            total.append(np.nan)
            constraint.append(np.nan)
            progress.append(np.nan)
            subgoal_consensus.append(np.nan)
            param_consensus.append(np.nan)
            feature_score_consensus.append(np.nan)
            for feat_values in feature_constraint_by_feat:
                feat_values.append(np.nan)
            continue

        total.append(float(info["total"]))
        constraint.append(float(info["constraint"]))
        progress.append(learner.lambda_progress * float(info["progress"]))
        subgoal_consensus.append(lam_subgoal_consensus * float(info["subgoal_consensus"]))
        param_consensus.append(lam_param_consensus * float(info["param_consensus"]))
        feature_score_consensus.append(lam_feature_score_consensus * float(info.get("feature_score_consensus", 0.0)))
        for feat_idx in range(learner.num_features):
            feat_cost = 0.0
            for stage_params in info["stage_params"]:
                feat_contrib = np.asarray(
                    getattr(stage_params, "feature_constraint_costs", np.zeros(learner.num_features)),
                    dtype=float,
                )
                if feat_idx < feat_contrib.size:
                    feat_cost += float(feat_contrib[feat_idx])
            feature_constraint_by_feat[feat_idx].append(feat_cost)

    ax.plot(candidate_values, total, color="black", lw=1.4, label="total")
    if show_components:
        ax.plot(candidate_values, constraint, color="tab:red", lw=1.0, label="constraint")
        feat_colors = plt.cm.Set1(np.linspace(0.0, 1.0, max(learner.num_features, 3)))
        for feat_idx in range(min(learner.num_features, 3)):
            feat_vals = np.asarray(feature_constraint_by_feat[feat_idx], dtype=float)
            if np.any(np.isfinite(feat_vals)):
                ax.plot(
                    candidate_values,
                    feat_vals,
                    lw=0.9,
                    linestyle="-.",
                    color=feat_colors[feat_idx],
                    label=f"{_feature_name(learner, feat_idx)} constraint",
                )
        ax.plot(candidate_values, progress, color="tab:orange", lw=1.0, label="progress")
        if np.any(np.isfinite(np.asarray(subgoal_consensus, dtype=float))):
            ax.plot(candidate_values, subgoal_consensus, color="tab:purple", lw=1.0, label="subgoal_consensus")
        if np.any(np.isfinite(np.asarray(param_consensus, dtype=float))):
            ax.plot(candidate_values, param_consensus, color="tab:cyan", lw=1.0, label="param_consensus")
        if np.any(np.isfinite(np.asarray(feature_score_consensus, dtype=float))):
            ax.plot(candidate_values, feature_score_consensus, color="tab:brown", lw=1.0, label="feature_score_consensus")

    current_value = cp2 if vary_index == 1 else cp1
    ax.axvline(current_value, color="black", linestyle="--", lw=1.0, label="pred boundary")
    true_cutpoints = _true_cutpoints_for_demo(learner, demo_idx)
    if vary_index < len(true_cutpoints):
        ax.axvline(int(true_cutpoints[vary_index]), color="green", linestyle=":", lw=1.0, label="true boundary")
    _annotate_true_cut_costs(ax, learner, demo_idx, vary_index, show_components)

    fixed_label = f"cp1={cp1}" if vary_index == 1 else f"cp2={cp2}"
    varying_label = "cp2" if vary_index == 1 else "cp1"
    ax.set_title(f"Demo {demo_idx} scan {varying_label} | fixed {fixed_label}", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel(f"{varying_label} index", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("cost", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def plot_scdp_results_4panel(learner, it, demo_idx=0):
    if plt is None:
        return
    if learner.num_states == 3:
        fig = plt.figure(figsize=(9.2, 8.6))

        ax1 = fig.add_subplot(3, 2, 1)
        _draw_trajectories(ax1, learner, it, demo_idx=demo_idx)

        ax2 = fig.add_subplot(3, 2, 2)
        _draw_constraint_cost_matrix(ax2, learner, demo_idx=demo_idx)

        ax3 = fig.add_subplot(3, 2, 3)
        _draw_feature_bands(ax3, learner, demo_idx=demo_idx)

        ax4 = fig.add_subplot(3, 2, 4)
        _draw_cost_profile(ax4, learner, demo_idx=demo_idx)

        ax5 = fig.add_subplot(3, 2, 5)
        _draw_two_cut_scan(ax5, learner, demo_idx=demo_idx, vary_index=1, show_components=True)

        ax6 = fig.add_subplot(3, 2, 6)
        _draw_two_cut_scan(ax6, learner, demo_idx=demo_idx, vary_index=0, show_components=True)
    else:
        fig = plt.figure(figsize=PAPER_FIGSIZE)

        ax1 = fig.add_subplot(2, 2, 1)
        _draw_trajectories(ax1, learner, it, demo_idx=demo_idx)

        ax2 = fig.add_subplot(2, 2, 2)
        _draw_constraint_cost_matrix(ax2, learner, demo_idx=demo_idx)

        ax3 = fig.add_subplot(2, 2, 3)
        _draw_feature_bands(ax3, learner, demo_idx=demo_idx)

        ax4 = fig.add_subplot(2, 2, 4)
        _draw_cost_profile(ax4, learner, demo_idx=demo_idx)

    save_figure(fig, learner_plot_dir(learner) / f"plot4panel_demo_{int(demo_idx):02d}_iter_{int(it):04d}.png", dpi=220)

    fig_delta = plt.figure(figsize=(8.8, 3.8))
    ax_delta_1 = fig_delta.add_subplot(1, 2, 1)
    ax_delta_2 = fig_delta.add_subplot(1, 2, 2)
    _draw_learned_vs_true_local_cost_delta(fig_delta, (ax_delta_1, ax_delta_2), learner, demo_idx=demo_idx)
    save_figure(fig_delta, learner_plot_dir(learner) / f"plot_cost_delta_demo_{int(demo_idx):02d}_iter_{int(it):04d}.png", dpi=220)
    _plot_cutpoint_feature_distribution_compare(learner, it, demo_idx=demo_idx, vary_index=1)
    _plot_cutpoint_feature_distribution_compare(learner, it, demo_idx=demo_idx, vary_index=0)


def plot_scdp_results_4panel_overview(learner, it):
    if plt is None:
        return
    fig = plt.figure(figsize=PAPER_FIGSIZE)

    ax1 = fig.add_subplot(2, 2, 1)
    _draw_trajectories_overview(ax1, learner, it)

    ax2 = fig.add_subplot(2, 2, 2)
    _draw_learning_curves(ax2, learner)

    ax3 = fig.add_subplot(2, 2, 3)
    _draw_feature_bands(ax3, learner, demo_idx=0)

    ax4 = fig.add_subplot(2, 2, 4)
    _draw_cost_profile(ax4, learner, demo_idx=0)

    save_figure(fig, learner_plot_dir(learner) / f"plot4panel_iter_{int(it):04d}.png", dpi=220)
