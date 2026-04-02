# visualization/plot4panel.py
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    from matplotlib.patches import Ellipse
except ModuleNotFoundError:
    Ellipse = None
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa
except ModuleNotFoundError:
    Axes3D = None

from .io import learner_plot_dir, save_figure

PAPER_FIGSIZE = (7.2, 5.2)
PAPER_TITLE_SIZE = 9
PAPER_LABEL_SIZE = 8
PAPER_TICK_SIZE = 7
PAPER_LEGEND_SIZE = 6.5
STAGE_COLORS = ["#D55E00", "#0072B2", "#CC79A7", "#009E73"]


def _true_first_cutpoint(learner, demo_idx: int):
    true_cutpoints = getattr(learner, "true_cutpoints", None)
    if true_cutpoints is not None and demo_idx < len(true_cutpoints):
        cuts = true_cutpoints[demo_idx]
        if cuts is not None:
            arr = np.asarray(cuts, dtype=int).reshape(-1)
            if arr.size > 0:
                return int(arr[0])
    if getattr(learner, "true_taus", None) is not None and learner.true_taus[demo_idx] is not None:
        return int(learner.true_taus[demo_idx])
    return None


def _true_cutpoints(learner, demo_idx: int):
    true_cutpoints = getattr(learner, "true_cutpoints", None)
    if true_cutpoints is not None and demo_idx < len(true_cutpoints):
        cuts = true_cutpoints[demo_idx]
        if cuts is None:
            return []
        return [int(x) for x in np.asarray(cuts, dtype=int).reshape(-1).tolist()]
    first = _true_first_cutpoint(learner, demo_idx)
    return [] if first is None else [int(first)]


def _draw_true_cutpoint_markers(ax, X, cutpoints, *, colors, label, is_3d=False, size=28, zorder=12):
    for j, tt in enumerate(cutpoints):
        color = colors[j % len(colors)] if colors else "black"
        coords = np.asarray(X[int(tt)], dtype=float)
        if is_3d:
            ax.scatter(
                coords[0], coords[1], coords[2],
                color=color, marker="x", s=size, linewidths=1.4,
                label=label if j == 0 else "", depthshade=False, zorder=zorder,
            )
        else:
            ax.scatter(
                coords[0], coords[1],
                color=color, marker="x", s=size, linewidths=1.4,
                label=label if j == 0 else "", zorder=zorder,
            )


def _stage_colors(num_stages: int):
    return [STAGE_COLORS[i % len(STAGE_COLORS)] for i in range(max(int(num_stages), 1))]


def _stage_ends_from_gamma(gamma):
    z = np.argmax(np.asarray(gamma, dtype=float), axis=1).astype(int)
    return np.where(np.diff(z) != 0)[0].astype(int).tolist() + [int(len(z) - 1)]


def _coerce_stage_ends(boundary_like, gamma, T: int):
    if boundary_like is None:
        return _stage_ends_from_gamma(gamma)
    arr = np.asarray(boundary_like)
    if arr.ndim == 0:
        tau = int(arr)
        return [tau, int(T - 1)]
    flat = np.asarray(arr, dtype=int).reshape(-1).tolist()
    if not flat:
        return [int(T - 1)]
    if flat[-1] != T - 1:
        flat = flat + [int(T - 1)]
    return [int(x) for x in flat]


def _segment_bounds(stage_ends):
    starts = []
    ends = []
    prev = -1
    for end in stage_ends:
        starts.append(prev + 1)
        ends.append(int(end))
        prev = int(end)
    return starts, ends


def _shared_stage_subgoals(learner):
    subgoals = getattr(learner, "stage_subgoals", None)
    if subgoals is not None:
        return [np.asarray(g, dtype=float) for g in subgoals]
    out = []
    if getattr(learner, "g1", None) is not None:
        out.append(np.asarray(learner.g1, dtype=float))
    if getattr(learner, "g2", None) is not None:
        out.append(np.asarray(learner.g2, dtype=float))
    return out


def _stage_subgoals_hist(learner, stage_idx: int):
    hist = getattr(learner, "stage_subgoals_hist", None)
    if hist:
        pts = []
        for item in hist:
            if stage_idx < len(item):
                pts.append(np.asarray(item[stage_idx], dtype=float))
        if pts:
            return np.stack(pts, axis=0)
    if stage_idx == 0 and getattr(learner, "g1_hist", None):
        return np.stack(learner.g1_hist, axis=0)
    if stage_idx == 1 and getattr(learner, "g2_hist", None):
        return np.stack(learner.g2_hist, axis=0)
    return None


def _feature_column_idx(learner, local_idx: int) -> int:
    if hasattr(learner, "feature_specs") and local_idx < len(learner.feature_specs):
        spec = learner.feature_specs[local_idx]
        if "column_idx" in spec:
            return int(spec["column_idx"])
        if "raw_id" in spec:
            return int(spec["raw_id"])
    return int(learner.selected_raw_feature_ids[local_idx])


def _feature_name(learner, local_idx: int) -> str:
    if hasattr(learner, "feature_specs") and local_idx < len(learner.feature_specs):
        spec = learner.feature_specs[local_idx]
        name = spec.get("name")
        if name is not None:
            return str(name)
    return f"f{int(local_idx)}"


def _feature_kind(learner, local_idx: int) -> str:
    feature_model_types = getattr(learner, "feature_model_types", None)
    if feature_model_types is None or local_idx >= len(feature_model_types):
        return "gaussian"
    return str(feature_model_types[local_idx]).lower()


def _kind_is_equality(kind: str) -> bool:
    return str(kind).lower() in {"gauss", "gaussian", "student_t", "studentt", "t", "gauss_zero", "zero_gaussian"}


def _summary_center_z(summary: dict, kind: str):
    kind_l = str(kind).lower()
    if _kind_is_equality(kind_l):
        if "mu" in summary:
            return float(summary["mu"])
        if kind_l in {"gauss_zero", "zero_gaussian"}:
            return 0.0
    if kind_l in {
        "margin_exp_lower",
        "marginexp",
        "margin_exp",
        "margin_exp_lower_left_hn",
        "marginexp_left_hn",
        "margin_exp_left_hn",
        "margin_exp_upper",
        "marginexp_upper",
        "margin_exp_upper",
        "margin_exp_upper_right_hn",
        "marginexp_upper_right_hn",
        "margin_exp_upper_right_hn",
    } and "b" in summary:
        return float(summary["b"])
    if "mu" in summary:
        return float(summary["mu"])
    if "b" in summary:
        return float(summary["b"])
    return np.nan


def _summary_interval(summary: dict):
    low = summary.get("L")
    high = summary.get("U")
    if low is None or high is None:
        return None, None
    return float(low), float(high)


def _z_to_display(learner, fid: int, value: float, standardized: bool):
    if not np.isfinite(value):
        return np.nan
    if standardized:
        return float(value)
    return float(value) * float(learner.feat_std[fid]) + float(learner.feat_mean[fid])


def _summary_center_display(learner, fid: int, summary: dict, kind: str, standardized: bool):
    center_z = _summary_center_z(summary, kind)
    return _z_to_display(learner, fid, center_z, standardized)


def _summary_interval_display(learner, fid: int, summary: dict, standardized: bool):
    low_z, high_z = _summary_interval(summary)
    if low_z is None or high_z is None:
        return np.nan, np.nan
    return (
        _z_to_display(learner, fid, low_z, standardized),
        _z_to_display(learner, fid, high_z, standardized),
    )


def _local_demo_model_summary(learner, demo_idx: int, stage_idx: int, feat_idx: int, kind: str):
    demo_param_vectors = getattr(learner, "demo_param_vectors_", None)
    if demo_param_vectors is not None and demo_idx < len(demo_param_vectors):
        try:
            vec = demo_param_vectors[demo_idx][stage_idx][feat_idx]
        except Exception:
            vec = None
        if vec is not None and hasattr(learner, "_vector_to_model"):
            model = learner._vector_to_model(kind, vec)
            return dict(model.get_summary())
    try:
        model = learner.feature_models[stage_idx][feat_idx]
    except Exception:
        return None
    return dict(model.get_summary())


def _local_demo_learned_value_raw(learner, demo_idx: int, stage_idx: int, feat_idx: int, kind: str):
    summary = _local_demo_model_summary(learner, demo_idx, stage_idx, feat_idx, kind)
    if not isinstance(summary, dict):
        return np.nan
    fid = _feature_column_idx(learner, feat_idx)
    return _summary_center_display(learner, fid, summary, kind, standardized=False)


def _raw_feature_matrix_for_demo(learner, demo_idx: int) -> np.ndarray:
    Fz = _standardized_feature_matrix_for_demo(learner, demo_idx)
    M = Fz.shape[1]
    feature_column_indices = np.asarray([_feature_column_idx(learner, m) for m in range(M)], dtype=int)
    feat_std = np.asarray(learner.feat_std, dtype=float)[feature_column_indices]
    feat_mean = np.asarray(learner.feat_mean, dtype=float)[feature_column_indices]
    return Fz * feat_std[None, :] + feat_mean[None, :]


def _standardized_feature_matrix_for_demo(learner, demo_idx: int) -> np.ndarray:
    X = learner.demos[demo_idx]
    if hasattr(learner, "_features_for_demo_matrix"):
        return np.asarray(learner._features_for_demo_matrix(X), dtype=float)
    standardized_features = getattr(learner, "standardized_features", None)
    if standardized_features is not None and demo_idx < len(standardized_features):
        return np.asarray(standardized_features[demo_idx], dtype=float)
    raise AttributeError(f"{learner.__class__.__name__} does not provide standardized feature access.")


def _plot_feature_evolution_panel(ax, learner, taus, gammas, demo_idx: int = 0, standardized: bool = False):
    X = learner.demos[demo_idx]
    gamma = gammas[demo_idx]
    T = len(X)
    t_axis = np.arange(T)
    M = int(getattr(learner, "num_features", 0))
    feature_column_indices = np.asarray([_feature_column_idx(learner, m) for m in range(M)], dtype=int)

    feat_vals = (
        _standardized_feature_matrix_for_demo(learner, demo_idx)
        if standardized
        else _raw_feature_matrix_for_demo(learner, demo_idx)
    )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if len(color_cycle) < M:
        from itertools import cycle, islice
        color_cycle = list(islice(cycle(color_cycle), M))
    is_joint_feature_hmm = str(getattr(learner, "feature_emission_mode", "")) == "joint_gmm"

    for m in range(M):
        feature_label = (
            learner.feature_specs[m]["name"]
            if hasattr(learner, "feature_specs") and m < len(learner.feature_specs)
            else f"f{m}"
        )
        ax.plot(
            t_axis,
            feat_vals[:, m],
            "-",
            color=color_cycle[m],
            label=f"{feature_label}(t)",
        )

    stage_ends = _coerce_stage_ends(taus[demo_idx], gamma, T)
    starts, ends = _segment_bounds(stage_ends)
    for j, cp in enumerate(stage_ends[:-1]):
        ax.axvline(cp, color="black", linestyle="--", label="learned cutpoint" if j == 0 else "")
    true_cutpoints = _true_cutpoints(learner, demo_idx)
    for j, cp in enumerate(true_cutpoints):
        ax.axvline(cp, color="gray", linestyle=":", label="true cutpoint" if j == 0 else "")

    for m in range(M):
        color_m = color_cycle[m]
        feature_name = (
            learner.feature_specs[m]["name"]
            if hasattr(learner, "feature_specs") and m < len(learner.feature_specs)
            else f"f{m}"
        )
        fid = feature_column_indices[m]
        for stage_idx, (s, e) in enumerate(zip(starts, ends)):
            t_seg = t_axis[s : e + 1]
            if len(t_seg) == 0:
                continue
            if is_joint_feature_hmm or learner.r[stage_idx, m] != 1:
                continue
            model = learner.feature_models[stage_idx][m]
            info = model.get_summary()
            model_type = info.get("type", "base")
            vals_seg = np.asarray(feat_vals[s : e + 1, m], dtype=float)
            if vals_seg.size == 0:
                continue
            low, up = _summary_interval_display(learner, fid, info, standardized)
            center = _summary_center_display(learner, fid, info, model_type, standardized)
            if np.isfinite(low) and np.isfinite(up):
                ax.fill_between(
                    t_seg,
                    low,
                    up,
                    color=color_m,
                    alpha=max(0.05, 0.12 - 0.02 * stage_idx) if not _kind_is_equality(model_type) else max(0.14, 0.22 - 0.02 * stage_idx),
                    linewidth=0,
                    zorder=2,
                )
                ax.plot(t_seg, np.full_like(t_seg, low, dtype=float), "-", color=color_m, alpha=0.55, linewidth=0.9, zorder=3)
                ax.plot(t_seg, np.full_like(t_seg, up, dtype=float), "-", color=color_m, alpha=0.55, linewidth=0.9, zorder=3)
            if np.isfinite(center):
                ax.plot(t_seg, np.full_like(t_seg, center, dtype=float), "--", color=color_m, alpha=0.75, linewidth=1.0, zorder=3)
            if model_type in ("margin_exp_lower", "margin_exp_lower_left_hn", "margin_exp_upper", "margin_exp_upper_right_hn"):
                ref_value = _reference_constraint_value(learner.env, feature_name, stage=stage_idx)
                if ref_value is not None:
                    ref_plot = float(ref_value)
                    if standardized:
                        ref_plot = (ref_plot - float(learner.feat_mean[fid])) / max(float(learner.feat_std[fid]), 1e-12)
                    ax.axhline(ref_plot, color=color_m, linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("t", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("standardized feature values" if standardized else "feature values", fontsize=PAPER_LABEL_SIZE)
    ax.set_title("Feature evolution (standardized)" if standardized else "Feature evolution (raw)", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)

    lines, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(lines, labels):
        if l is None:
            continue
        l = str(l).strip()
        if l == "" or l.startswith("_"):
            continue
        if l not in by_label:
            by_label[l] = h
    ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)


def _learned_constraint_value_raw(values_raw, kind: str):
    vals = np.asarray(values_raw, dtype=float).reshape(-1)
    if vals.size == 0:
        return np.nan
    kind_l = str(kind).lower()
    if kind_l in {"margin_exp_lower", "marginexp", "margin_exp", "margin_exp_lower_left_hn", "marginexp_left_hn", "margin_exp_left_hn"}:
        return float(np.quantile(vals, 0.02))
    if kind_l in {"margin_exp_upper", "marginexp_upper", "margin_exp_upper", "margin_exp_upper_right_hn", "marginexp_upper_right_hn", "margin_exp_upper_right_hn"}:
        return float(np.quantile(vals, 0.98))
    return float(np.median(vals))


def _active_mask_for_demo_stage_feature(learner, demo_idx: int, stage_idx: int, feat_idx: int) -> bool:
    demo_r = getattr(learner, "demo_r_matrices_", None)
    if demo_r is not None and demo_idx < len(demo_r):
        arr = np.asarray(demo_r[demo_idx], dtype=float)
        if arr.ndim == 2 and stage_idx < arr.shape[0] and feat_idx < arr.shape[1]:
            return bool(arr[stage_idx, feat_idx] > 0.5)
    r = getattr(learner, "r", None)
    if r is not None:
        arr = np.asarray(r, dtype=float)
        if arr.ndim == 2 and stage_idx < arr.shape[0] and feat_idx < arr.shape[1]:
            return bool(arr[stage_idx, feat_idx] > 0.5)
    return False


def _feature_has_reference_constraint(learner, feat_idx: int) -> bool:
    feature_name = _feature_name(learner, feat_idx)
    num_stages = int(getattr(learner, "num_stages", 0))
    for stage_idx in range(num_stages):
        if _reference_constraint_value(learner.env, feature_name, stage=stage_idx) is not None:
            return True
    return False


def _plot_constraint_parameter_panels(fig, gs_cell, learner, taus, gammas, stage_colors):
    M = int(getattr(learner, "num_features", 0))
    if M <= 0:
        return
    feature_indices = [
        feat_idx
        for feat_idx in range(M)
        if _feature_has_reference_constraint(learner, feat_idx)
        or any(
            _active_mask_for_demo_stage_feature(learner, demo_idx, stage_idx, feat_idx)
            for demo_idx in range(len(learner.demos))
            for stage_idx in range(int(getattr(learner, "num_stages", 0)))
        )
    ]
    if not feature_indices:
        return

    n_panels = len(feature_indices)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / max(n_cols, 1)))
    sub_gs = gs_cell.subgridspec(n_rows, n_cols, wspace=0.35, hspace=0.55)
    demo_x = np.arange(len(learner.demos))
    for panel_idx, feat_idx in enumerate(feature_indices):
        ax = fig.add_subplot(sub_gs[panel_idx // n_cols, panel_idx % n_cols])
        feature_name = _feature_name(learner, feat_idx)
        kind = _feature_kind(learner, feat_idx)
        any_series = False

        for stage_idx in range(int(getattr(learner, "num_stages", 0))):
            learned_vals = []
            active_any = False
            for demo_idx, (X, boundary_like, gamma) in enumerate(zip(learner.demos, taus, gammas)):
                T = len(X)
                stage_ends = _coerce_stage_ends(boundary_like, gamma, T)
                starts, ends = _segment_bounds(stage_ends)
                if stage_idx >= len(starts):
                    learned_vals.append(np.nan)
                    continue
                is_active = _active_mask_for_demo_stage_feature(learner, demo_idx, stage_idx, feat_idx)
                if not is_active:
                    learned_vals.append(np.nan)
                    continue
                active_any = True
                learned_vals.append(_local_demo_learned_value_raw(learner, demo_idx, stage_idx, feat_idx, kind))

            if active_any:
                ax.plot(
                    demo_x,
                    np.asarray(learned_vals, dtype=float),
                    marker="o",
                    linewidth=1.1,
                    markersize=3.0,
                    color=stage_colors[stage_idx % len(stage_colors)],
                    label=f"s{stage_idx + 1} learned",
                )
                any_series = True

            ref_value = _reference_constraint_value(learner.env, feature_name, stage=stage_idx)
            if ref_value is not None:
                ax.axhline(
                    float(ref_value),
                    color=stage_colors[stage_idx % len(stage_colors)],
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.85,
                    label=f"s{stage_idx + 1} true",
                )
                any_series = True

        if not any_series:
            ax.axis("off")
            continue

        ax.set_title(f"{feature_name}", fontsize=PAPER_TITLE_SIZE, pad=4)
        ax.set_xlabel("demo", fontsize=PAPER_LABEL_SIZE)
        ax.set_ylabel("learned value", fontsize=PAPER_LABEL_SIZE)
        ax.set_xticks(demo_x)
        ax.set_xticklabels([str(i) for i in demo_x], fontsize=PAPER_TICK_SIZE)
        ax.tick_params(labelsize=PAPER_TICK_SIZE)
        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            if l and l not in by_label:
                by_label[l] = h
        if by_label:
            ax.legend(
                by_label.values(),
                by_label.keys(),
                loc="best",
                fontsize=max(PAPER_LEGEND_SIZE - 1.0, 5.5),
                frameon=False,
                handlelength=1.2,
                borderpad=0.2,
            )

    for panel_idx in range(n_panels, n_rows * n_cols):
        ax = fig.add_subplot(sub_gs[panel_idx // n_cols, panel_idx % n_cols])
        ax.axis("off")


def _plot_cutpoint_evolution_panels(fig, subplotspec, learner, max_cols: int = 4):
    history = getattr(learner, "segmentation_history_", None)
    if not history:
        ax = fig.add_subplot(subplotspec)
        ax.axis("off")
        return
    history = [
        [list(map(int, ends)) for ends in snapshot]
        for snapshot in history
        if snapshot is not None
    ]
    if not history:
        ax = fig.add_subplot(subplotspec)
        ax.axis("off")
        return

    num_iters = len(history)
    num_demos = len(history[0])
    if num_demos == 0:
        ax = fig.add_subplot(subplotspec)
        ax.axis("off")
        return
    num_cutpoints = max(len(history[0][0]) - 1, 0)
    if num_cutpoints <= 0:
        ax = fig.add_subplot(subplotspec)
        ax.axis("off")
        return

    xs = np.arange(num_iters, dtype=int)
    colors = _stage_colors(num_cutpoints + 1)
    true_cutpoints = getattr(learner, "true_cutpoints", None)
    max_cols = max(1, min(int(max_cols), num_demos))
    ncols = max_cols
    nrows = int(np.ceil(num_demos / float(ncols)))
    subgs = subplotspec.subgridspec(nrows, ncols, wspace=0.25, hspace=0.35)

    y_max = 0.0
    for snapshot in history:
        for demo_idx in range(num_demos):
            for cp_idx in range(num_cutpoints):
                y_max = max(y_max, float(snapshot[demo_idx][cp_idx]))
    if true_cutpoints is not None:
        for demo_idx in range(min(num_demos, len(true_cutpoints))):
            cuts = true_cutpoints[demo_idx]
            if cuts is None:
                continue
            arr = np.asarray(cuts, dtype=int).reshape(-1)
            if arr.size > 0:
                y_max = max(y_max, float(np.max(arr)))
    y_max = max(y_max, 1.0)

    first_ax = None
    for demo_idx in range(nrows * ncols):
        row = demo_idx // ncols
        col = demo_idx % ncols
        ax = fig.add_subplot(subgs[row, col])
        if demo_idx >= num_demos:
            ax.axis("off")
            continue
        if first_ax is None:
            first_ax = ax
        for cp_idx in range(num_cutpoints):
            ys = np.asarray([snapshot[demo_idx][cp_idx] for snapshot in history], dtype=float)
            ax.plot(xs, ys, color=colors[cp_idx], linewidth=1.3, label=f"cp{cp_idx + 1}" if demo_idx == 0 else "")
            if true_cutpoints is not None and demo_idx < len(true_cutpoints):
                cuts = true_cutpoints[demo_idx]
                if cuts is not None:
                    arr = np.asarray(cuts, dtype=int).reshape(-1)
                    if cp_idx < arr.size:
                        ax.axhline(
                            float(arr[cp_idx]),
                            color=colors[cp_idx],
                            linestyle=":",
                            linewidth=0.9,
                            alpha=0.9,
                            label=f"cp{cp_idx + 1} true" if demo_idx == 0 else "",
                        )
        ax.set_title(f"demo {demo_idx}", fontsize=PAPER_TITLE_SIZE, pad=3)
        ax.set_ylim(-0.5, y_max + 0.5)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=PAPER_TICK_SIZE)
        if row == nrows - 1:
            ax.set_xlabel("iteration", fontsize=PAPER_LABEL_SIZE)
        if col == 0:
            ax.set_ylabel("index", fontsize=PAPER_LABEL_SIZE)

    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            if l and l not in by_label:
                by_label[l] = h
        if by_label:
            first_ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=PAPER_LEGEND_SIZE, frameon=False)


def _has_cutpoint_evolution_history(learner) -> bool:
    history = getattr(learner, "segmentation_history_", None)
    if not history:
        return False
    snapshots = [
        [list(map(int, ends)) for ends in snapshot]
        for snapshot in history
        if snapshot is not None
    ]
    if not snapshots or not snapshots[0]:
        return False
    return max(len(snapshots[0][0]) - 1, 0) > 0


def _reference_constraint_value(env, feature_name: str, stage: int):
    specs = getattr(env, "constraint_specs", None)
    true_constraints = getattr(env, "true_constraints", {}) or {}
    if not isinstance(specs, list):
        return None
    for spec in specs:
        if spec.get("feature_name") != feature_name or int(spec.get("stage", -1)) != int(stage):
            continue
        oracle_key = spec.get("oracle_key")
        if oracle_key is None:
            return None
        return true_constraints.get(oracle_key)
    return None


def _summary_feature_names(learner, metrics=None):
    if isinstance(metrics, dict):
        names = metrics.get("ConstraintFeatureNames")
        if isinstance(names, list) and names:
            return [str(x) for x in names]
    return [_feature_name(learner, i) for i in range(int(getattr(learner, "num_features", 0)))]


def _summary_stage_labels(learner):
    return [f"s{i + 1}" for i in range(int(getattr(learner, "num_stages", 0)))]


def _draw_summary_heatmap(ax, matrix, title, *, feature_names, stage_labels, cmap="viridis", fmt=".2f", vmin=None, vmax=None):
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        ax.axis("off")
        return

    finite = arr[np.isfinite(arr)]
    if vmin is None:
        vmin = float(np.min(finite)) if finite.size > 0 else 0.0
    if vmax is None:
        vmax = float(np.max(finite)) if finite.size > 0 else 1.0
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1e-6

    masked = np.ma.masked_invalid(arr)
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("#d9d9d9")
    im = ax.imshow(masked, aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xticks(range(arr.shape[1]))
    ax.set_xticklabels(stage_labels)
    ax.set_yticks(range(arr.shape[0]))
    ax.set_yticklabels(feature_names)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    scale = max(abs(vmin), abs(vmax), 1e-6)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            text = "nan" if not np.isfinite(value) else format(float(value), fmt)
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if abs(float(value) if np.isfinite(value) else 0.0) >= 0.55 * scale else "black",
                fontsize=6.8,
            )
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.ax.tick_params(labelsize=PAPER_TICK_SIZE - 0.5, width=0.6, length=2.5)
    cbar.outline.set_linewidth(0.6)


def _draw_eval_metric_text(ax, metrics):
    ax.axis("off")
    if not isinstance(metrics, dict):
        ax.text(0.02, 0.98, "No evaluation metrics.", ha="left", va="top", fontsize=8, family="monospace", transform=ax.transAxes)
        return

    scalar_metrics = {}
    for key, value in metrics.items():
        if np.isscalar(value):
            value_f = float(value)
            if np.isfinite(value_f):
                scalar_metrics[str(key)] = value_f
    preferred_keys = [
        "MeanAbsCutpointError",
        "MeanStageSubgoalError",
        "MeanConstraintError",
        "ConstraintActivationAccuracy",
    ]
    ordered_keys = [key for key in preferred_keys if key in scalar_metrics]
    ordered_keys += [key for key in sorted(scalar_metrics.keys()) if key not in ordered_keys]

    ax.set_title("evaluation metrics", fontsize=PAPER_TITLE_SIZE, pad=4)
    if not ordered_keys:
        ax.text(0.02, 0.98, "No scalar metrics.", ha="left", va="top", fontsize=8, transform=ax.transAxes)
        return
    y = 0.92
    for idx, key in enumerate(ordered_keys):
        ax.text(0.03, y, key, ha="left", va="top", fontsize=7.5, color="#444444", transform=ax.transAxes)
        ax.text(
            0.97,
            y,
            f"{scalar_metrics[key]:.4f}",
            ha="right",
            va="top",
            fontsize=9.5 if idx < 4 else 8.0,
            color="black",
            transform=ax.transAxes,
        )
        y -= 0.17 if idx < 4 else 0.13
        if y < 0.06:
            break
    ax.plot([0.03, 0.97], [0.975, 0.975], color="#333333", lw=0.8, transform=ax.transAxes, clip_on=False)


def _draw_cylinder_wire(ax, center_xy, radius, z0=0.0, height=0.5,
                        color="gray", alpha=0.6, n_theta=80, n_z=10):
    """English documentation omitted during cleanup."""
    cx, cy = center_xy
    theta = np.linspace(0, 2*np.pi, n_theta)
    z = np.linspace(z0, z0 + height, n_z)
    th, zz = np.meshgrid(theta, z)
    xx = cx + radius * np.cos(th)
    yy = cy + radius * np.sin(th)
    ax.plot_wireframe(xx, yy, zz, color=color, alpha=alpha, linewidth=0.6)

def _set_axes_equal_3d_from_xyz(ax, xyz):
    """
    xyz: (N,3) array of points you want to fit.
    Forces equal data scale AND equal box aspect.
    """
    xyz = np.asarray(xyz)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)

    spans = maxs - mins
    max_span = spans.max()
    centers = (maxs + mins) / 2.0

    # equal data limits
    ax.set_xlim(centers[0] - max_span/2, centers[0] + max_span/2)
    ax.set_ylim(centers[1] - max_span/2, centers[1] + max_span/2)
    ax.set_zlim(centers[2] - max_span/2, centers[2] + max_span/2)

    # equal box aspect (if supported)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


def _env_has_xy_obstacle(env) -> bool:
    return hasattr(env, "obs_center") and hasattr(env, "obs_radius")


def _env_has_3d_obstacle(env) -> bool:
    return hasattr(env, "obs_center_xy") and hasattr(env, "obs_radius")


def _draw_env_xy_obstacles(ax, env):
    if hasattr(env, "stage1_aux_obstacle_centers") and hasattr(env, "stage1_aux_obstacle_radii"):
        cx, cy = np.asarray(env.obs_center, dtype=float).reshape(-1)[:2]
        ax.add_patch(plt.Circle((cx, cy), float(env.obs_radius), color='gray', fill=False, linestyle='-', label='obstacle'))
        for center, radius in zip(np.asarray(env.stage1_aux_obstacle_centers, dtype=float), np.asarray(env.stage1_aux_obstacle_radii, dtype=float)):
            aux_x, aux_y = np.asarray(center, dtype=float).reshape(-1)[:2]
            ax.add_patch(
                plt.Circle(
                    (float(aux_x), float(aux_y)),
                    float(radius),
                    color='gray',
                    fill=False,
                    linestyle=(0, (3, 2)),
                    alpha=0.95,
                )
            )
        return
    if _env_has_xy_obstacle(env):
        cx, cy = env.obs_center
        r = env.obs_radius
        ax.add_patch(plt.Circle((cx, cy), r, color='gray', fill=False, linestyle='-', label='obstacle'))


def _is_pickplace(env) -> bool:
    return getattr(env, "eval_tag", "") == "PickPlace"


def _is_press_slide_insert(env) -> bool:
    return getattr(env, "eval_tag", "") == "S4SlideInsert"


def _is_sphere_inspect(env) -> bool:
    return str(getattr(env, "eval_tag", "")).startswith("S5SphereInspect")


def _xy_point(point):
    arr = np.asarray(point, dtype=float).reshape(-1)
    return arr[:2]


def _true_stage_end_point(env, label_name: str, demo_idx: int | None = None):
    if bool(getattr(env, "hide_true_stage_end_markers", False)):
        return None
    demo_attr = "demo_subgoals" if str(label_name) == "subgoal" else "demo_goals"
    demo_points = getattr(env, demo_attr, None)
    if demo_idx is not None and demo_points is not None and 0 <= int(demo_idx) < len(demo_points):
        pt = demo_points[int(demo_idx)]
        if pt is not None:
            return np.asarray(pt, dtype=float).reshape(-1)
    pt = getattr(env, label_name, None)
    if pt is None:
        return None
    return np.asarray(pt, dtype=float).reshape(-1)


def _all_true_stage_end_points(env, label_name: str):
    if bool(getattr(env, "hide_true_stage_end_markers", False)):
        return []
    demo_attr = "demo_subgoals" if str(label_name) == "subgoal" else "demo_goals"
    demo_points = getattr(env, demo_attr, None)
    if demo_points is not None:
        out = []
        for pt in demo_points:
            if pt is None:
                continue
            out.append(np.asarray(pt, dtype=float).reshape(-1))
        if out:
            return out
    pt = getattr(env, label_name, None)
    if pt is None:
        return []
    return [np.asarray(pt, dtype=float).reshape(-1)]


def _trajectory_figsize(learner):
    if _is_press_slide_insert(learner.env):
        return (10.6, 6.6)
    return PAPER_FIGSIZE


def _trajectory_legend_kwargs(env):
    if _is_press_slide_insert(env):
        return {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0)}
    return {"loc": "best"}


def _configure_2d_trajectory_axes(ax, env, pts):
    pts = np.asarray(pts, dtype=float)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    dx = float(max(xmax - xmin, 1e-6))
    dy = float(max(ymax - ymin, 1e-6))
    if _is_press_slide_insert(env):
        xpad = 0.07 * dx
        ypad = max(0.02, 0.22 * dy)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(ymin - ypad, ymax + ypad)
        ax.set_ylabel("z")
        ax.set_aspect("auto")
        try:
            ax.set_box_aspect(0.78)
        except Exception:
            pass
        return
    pad = 0.05 * max(dx, dy)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("z" if _is_pickplace(env) else "y")
    ax.set_aspect("equal", adjustable="box")


def _draw_chmm_gmms(ax, learner, x_dim: int):
    if plt is None:
        return
    if bool(getattr(learner, "use_relative_stage_state", False)):
        return
    if not all(hasattr(learner, attr) for attr in ("gmm_weights", "gmm_means", "gmm_covs")):
        return
    use_3d = x_dim == 3 and not _is_press_slide_insert(learner.env)
    state_colors = ["#D55E00", "#0072B2", "#CC79A7", "#009E73"]
    x_mean = np.asarray(getattr(learner, "x_mean", np.zeros((1, x_dim))), dtype=float).reshape(-1)
    x_std = np.asarray(getattr(learner, "x_std", np.ones((1, x_dim))), dtype=float).reshape(-1)
    if x_mean.size < 2:
        x_mean = np.pad(x_mean, (0, 2 - x_mean.size), constant_values=0.0)
    if x_std.size < 2:
        x_std = np.pad(x_std, (0, 2 - x_std.size), constant_values=1.0)
    mean_xy = x_mean[:2]
    std_xy = np.maximum(x_std[:2], 1e-12)

    if use_3d:
        for state_idx, (weights, means) in enumerate(zip(learner.gmm_weights, learner.gmm_means)):
            means = np.asarray(means, dtype=float)
            if means.ndim != 2 or means.shape[1] < 3:
                continue
            for comp_idx, (w, mu) in enumerate(zip(weights, means)):
                mu_raw = np.asarray(mu[:3], dtype=float).copy()
                raw_std = np.ones_like(mu_raw)
                raw_mean = np.zeros_like(mu_raw)
                raw_std[: min(len(x_std), 3)] = x_std[: min(len(x_std), 3)]
                raw_mean[: min(len(x_mean), 3)] = x_mean[: min(len(x_mean), 3)]
                mu_raw = mu_raw * raw_std + raw_mean
                alpha = float(np.clip(w, 0.15, 0.95))
                ax.scatter(
                    mu_raw[0],
                    mu_raw[1],
                    mu_raw[2],
                    c=state_colors[state_idx % len(state_colors)],
                    marker="o",
                    s=20 + 40 * float(w),
                    alpha=alpha,
                    depthshade=False,
                    label=f"state {state_idx + 1} GMM mean" if comp_idx == 0 else "",
                    zorder=11,
                )
        return

    if Ellipse is None:
        return
    for state_idx, (weights, means, covs) in enumerate(zip(learner.gmm_weights, learner.gmm_means, learner.gmm_covs)):
        means = np.asarray(means, dtype=float)
        covs = np.asarray(covs, dtype=float)
        if means.ndim != 2 or means.shape[1] < 2 or covs.ndim != 3 or covs.shape[1] < 2 or covs.shape[2] < 2:
            continue
        for comp_idx, (w, mu, cov) in enumerate(zip(weights, means, covs)):
            mu_raw = np.asarray(mu[:2], dtype=float) * std_xy + mean_xy
            cov_xy = np.asarray(cov[:2, :2], dtype=float)
            cov_raw = np.diag(std_xy) @ cov_xy @ np.diag(std_xy)
            try:
                eigvals, eigvecs = np.linalg.eigh(cov_raw)
            except np.linalg.LinAlgError:
                continue
            eigvals = np.clip(eigvals, 1e-9, None)
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
            width = 4.0 * np.sqrt(eigvals[0])
            height = 4.0 * np.sqrt(eigvals[1])
            alpha = float(np.clip(w, 0.18, 0.75))
            ell = Ellipse(
                xy=(float(mu_raw[0]), float(mu_raw[1])),
                width=float(width),
                height=float(height),
                angle=angle,
                facecolor="none",
                edgecolor=state_colors[state_idx % len(state_colors)],
                linewidth=1.0,
                alpha=alpha,
                linestyle="-" if state_idx == 0 else "--",
                label=f"state {state_idx + 1} GMM" if comp_idx == 0 else "",
                zorder=6,
            )
            ax.add_patch(ell)
            ax.scatter(
                float(mu_raw[0]),
                float(mu_raw[1]),
                c=state_colors[state_idx % len(state_colors)],
                s=12 + 28 * float(w),
                alpha=alpha,
                zorder=7,
            )



def plot_demos_goals_snapshot(ax, learner, taus, gammas, title=None, show_legend=True):
    if plt is None:
        return ax

    if _is_sphere_inspect(learner.env) and int(getattr(learner, "num_stages", 0)) >= 5:
        demo_idx = 0
        X = np.asarray(learner.demos[demo_idx], dtype=float)
        T = len(X)
        boundary_like = taus[demo_idx] if demo_idx < len(taus) else None
        gamma = gammas[demo_idx]
        stage_ends = _coerce_stage_ends(boundary_like, gamma, T)
        pred_cutpoints = [int(x) for x in np.asarray(stage_ends[:-1], dtype=int).reshape(-1).tolist()]
        true_cutpoints = _true_cutpoints(learner, demo_idx)
        Fz = _standardized_feature_matrix_for_demo(learner, demo_idx)
        feature_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(Fz.shape[1], 1)))
        t = np.arange(T, dtype=float)
        if title is not None:
            ax.set_title(title, fontsize=PAPER_TITLE_SIZE, pad=4)
        for feat_idx in range(Fz.shape[1]):
            ax.plot(
                t,
                np.asarray(Fz[:, feat_idx], dtype=float),
                lw=1.15,
                alpha=0.95,
                color=feature_colors[feat_idx],
                label=_feature_name(learner, feat_idx),
            )
        for j, cp in enumerate(pred_cutpoints):
            ax.axvline(cp, color="black", linestyle="--", linewidth=0.95, alpha=0.85, label="pred cutpoint" if j == 0 else "")
        for j, cp in enumerate(true_cutpoints):
            ax.axvline(cp, color="#666666", linestyle=":", linewidth=0.95, alpha=0.9, label="true cutpoint" if j == 0 else "")
        ax.set_xlabel("t")
        ax.set_ylabel("z feature")
        ax.grid(alpha=0.24)
        ax.tick_params(labelsize=PAPER_TICK_SIZE)
        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            l = str(l).strip()
            if l and not l.startswith("_") and l not in by_label:
                by_label[l] = h
        if show_legend and by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=PAPER_LEGEND_SIZE, frameon=False, loc="upper right")
        return ax

    shared_subgoals = _shared_stage_subgoals(learner)
    colors = _stage_colors(getattr(learner, "num_stages", len(shared_subgoals) or 2))
    X_dim = learner.demos[0].shape[1]
    is_press = _is_press_slide_insert(learner.env)
    use_3d = X_dim == 3 and not is_press
    traj_marker_size = 3.0 if is_press else 4.0
    goal_marker_size = 22 if is_press else 28
    subgoal_marker_size = 20 if is_press else 24
    cutpoint_marker_size = 18 if is_press else 24
    if title is not None:
        ax.set_title(title, fontsize=PAPER_TITLE_SIZE, pad=4)

    if use_3d:
        for i, (X, boundary_like, gamma) in enumerate(zip(learner.demos, taus, gammas)):
            X = np.asarray(X)
            T = len(X)
            stage_ends = _coerce_stage_ends(boundary_like, gamma, T)
            starts, ends = _segment_bounds(stage_ends)
            for k, (s, e) in enumerate(zip(starts, ends)):
                ax.scatter(X[s : e + 1, 0], X[s : e + 1, 1], X[s : e + 1, 2], color=colors[k], s=3, alpha=0.35, depthshade=False)
    else:
        for i, (X, boundary_like, gamma) in enumerate(zip(learner.demos, taus, gammas)):
            X = np.asarray(X)
            T = len(X)
            stage_ends = _coerce_stage_ends(boundary_like, gamma, T)
            starts, ends = _segment_bounds(stage_ends)
            for k, (s, e) in enumerate(zip(starts, ends)):
                ax.scatter(X[s : e + 1, 0], X[s : e + 1, 1], color=colors[k], s=traj_marker_size, alpha=0.30 if is_press else 0.35)

    if not use_3d and _env_has_xy_obstacle(learner.env):
        _draw_env_xy_obstacles(ax, learner.env)

    _draw_chmm_gmms(ax, learner, X_dim)

    if use_3d:
        for pt_idx, sg in enumerate(_all_true_stage_end_points(learner.env, "subgoal")):
            ax.scatter(sg[0], sg[1], sg[2], color='black', marker='X', s=28, alpha=0.55, label='true stage end' if pt_idx == 0 else "")
        subgoal_count = len(_all_true_stage_end_points(learner.env, "subgoal"))
        for pt_idx, gg in enumerate(_all_true_stage_end_points(learner.env, "goal")):
            ax.scatter(gg[0], gg[1], gg[2], color='black', marker='X', s=28, alpha=0.55, label='true stage end' if (subgoal_count == 0 and pt_idx == 0) else "")
    elif not _is_pickplace(learner.env):
        for pt_idx, sg in enumerate(_all_true_stage_end_points(learner.env, "subgoal")):
            sg_xy = _xy_point(sg)
            ax.scatter(sg_xy[0], sg_xy[1], color='black', marker='X', s=goal_marker_size, alpha=0.55, label='true stage end' if pt_idx == 0 else "")
        subgoal_count = len(_all_true_stage_end_points(learner.env, "subgoal"))
        for pt_idx, gg in enumerate(_all_true_stage_end_points(learner.env, "goal")):
            gg_xy = _xy_point(gg)
            ax.scatter(gg_xy[0], gg_xy[1], color='black', marker='X', s=goal_marker_size, alpha=0.55, label='true stage end' if (subgoal_count == 0 and pt_idx == 0) else "")

    history_linestyles = ["-", "--", ":", "-."]
    for k, sg in enumerate(shared_subgoals):
        hist = _stage_subgoals_hist(learner, k)
        if hist is not None and len(hist) > 1:
            ls = history_linestyles[k % len(history_linestyles)]
            if use_3d:
                ax.plot(hist[:, 0], hist[:, 1], hist[:, 2], ls, lw=1.8, alpha=0.7, color='black', label=f'g{k + 1} history')
                ax.scatter(hist[:, 0], hist[:, 1], hist[:, 2], s=8, alpha=0.45, color='black')
            else:
                ax.plot(hist[:, 0], hist[:, 1], ls, lw=1.8, alpha=0.7, color='black', label=f'g{k + 1} history')
                ax.scatter(hist[:, 0], hist[:, 1], s=8, alpha=0.45, color='black')

    if use_3d:
        for i, X in enumerate(learner.demos):
            _draw_true_cutpoint_markers(
                ax,
                np.asarray(X),
                _true_cutpoints(learner, i),
                colors=colors,
                label='true cutpoint' if i == 0 else "",
                is_3d=True,
                size=28,
                zorder=13,
            )
    else:
        for i, X in enumerate(learner.demos):
            _draw_true_cutpoint_markers(
                ax,
                np.asarray(X),
                _true_cutpoints(learner, i),
                colors=colors,
                label='true cutpoint' if i == 0 else "",
                is_3d=False,
                size=cutpoint_marker_size,
                zorder=11,
            )

    if use_3d:
        for k, sg in enumerate(shared_subgoals):
            ax.scatter(sg[0], sg[1], sg[2], color=colors[k], marker='D', s=24, edgecolors='black', linewidths=0.8, label=f'est. stage {k + 1} end')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    else:
        for k, sg in enumerate(shared_subgoals):
            g_xy = _xy_point(sg)
            ax.scatter(g_xy[0], g_xy[1], color=colors[k], marker='D', s=subgoal_marker_size, edgecolors='black', linewidths=0.8, label=f'est. stage {k + 1} end')
        ax.set_xlabel("x")
        pts = []
        for X in learner.demos:
            pts.append(np.asarray(X)[:, :2])
        for pt in _all_true_stage_end_points(learner.env, "subgoal"):
            pts.append(_xy_point(pt)[None, :])
        for pt in _all_true_stage_end_points(learner.env, "goal"):
            pts.append(_xy_point(pt)[None, :])
        _configure_2d_trajectory_axes(ax, learner.env, np.concatenate(pts, axis=0))

    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l is None:
            continue
        l = str(l).strip()
        if l == "" or l.startswith("_"):
            continue
        if l not in by_label:
            by_label[l] = h
    if show_legend and by_label:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            fontsize=PAPER_LEGEND_SIZE,
            frameon=False,
            **_trajectory_legend_kwargs(learner.env),
        )
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    return ax


def plot_results_4panel(learner, taus, it, gammas, alphas, betas, xis_list, aux_list, save_name=None, metrics=None):
    if plt is None:
        return
    """
    English documentation omitted during cleanup.
    English documentation omitted during cleanup.

    English documentation omitted during cleanup.
      - demos, true_taus, env
      - g1, g2, g1_hist, g2_hist
      - loss_loglik, loss_feat
      - get_bounds_for_plot(k_sigma=2)
      - _features_for_demo(X)
      English documentation omitted during cleanup.
    """

    plot_context = getattr(learner, "plot_context", "joint")
    shared_subgoals = _shared_stage_subgoals(learner)
    stage_colors = _stage_colors(getattr(learner, "num_stages", len(shared_subgoals) or 2))
    has_learning_history = len(getattr(learner, "loss_loglik", []) or []) > 1
    show_learning_row = plot_context != "posthoc" or has_learning_history
    learning_title = {
        "joint": "Joint learning curves",
        "posthoc": "Segmentation learning curves",
        "fchmm": "Feature-Constrained HMM learning curves",
        "hmm": "HMM learning curves",
    }.get(plot_context, "Learning curves")
    decomposition_title = {
        "joint": "Joint posterior decomposition",
        "posthoc": "Posthoc constraint decomposition",
        "fchmm": "Feature-Constrained HMM posterior factors",
        "hmm": "HMM posterior factors",
    }.get(plot_context, "Posterior decomposition")

    # ==========================================================
    # (1) Trajectories + obstacle + goals + g history + cutpoints
    # ==========================================================
    extra_panel_indices = [
        feat_idx
        for feat_idx in range(int(getattr(learner, "num_features", 0)))
        if _feature_has_reference_constraint(learner, feat_idx)
        or any(
            _active_mask_for_demo_stage_feature(learner, demo_idx, stage_idx, feat_idx)
            for demo_idx in range(len(learner.demos))
            for stage_idx in range(int(getattr(learner, "num_stages", 0)))
        )
    ]
    extra_rows = 0 if not extra_panel_indices else int(np.ceil(len(extra_panel_indices) / min(4, len(extra_panel_indices))))
    show_cutpoint_row = _has_cutpoint_evolution_history(learner)
    show_summary_row = isinstance(metrics, dict) and (
        np.asarray(metrics.get("ConstraintErrorMatrix", []), dtype=float).ndim == 2
        or any(np.isscalar(v) and np.isfinite(float(v)) for v in metrics.values())
    )
    cut_demo_rows = int(np.ceil(max(len(learner.demos), 1) / 4.0))
    base_w, base_h = _trajectory_figsize(learner)
    summary_height = 1.5 if show_summary_row else 0.0
    cutpoint_height = 0.9 + 0.55 * max(cut_demo_rows - 1, 0) if show_cutpoint_row else 0.0
    learning_height = 1.0 if show_learning_row else 0.0
    fig = plt.figure(figsize=(base_w, base_h + 1.2 + learning_height + cutpoint_height + summary_height + 1.7 * extra_rows))
    total_rows = 2 + (1 if show_learning_row else 0) + (1 if show_cutpoint_row else 0) + (1 if show_summary_row else 0) + max(extra_rows, 0)
    height_ratios = [1.0]
    if show_learning_row:
        height_ratios.append(1.0)
    height_ratios.append(0.95)
    if show_cutpoint_row:
        height_ratios.append(0.9 + 0.55 * max(cut_demo_rows - 1, 0))
    if show_summary_row:
        height_ratios.append(0.95)
    if extra_rows > 0:
        height_ratios.extend([0.7] * extra_rows)
    gs = fig.add_gridspec(
        total_rows,
        2,
        height_ratios=height_ratios,
    )
    X_dim = learner.demos[0].shape[1]
    is_press = _is_press_slide_insert(learner.env)
    use_3d = X_dim == 3 and not is_press
    traj_marker_size = 1.8 if is_press else 2.5
    goal_marker_size = 24 if is_press else 28
    subgoal_marker_size = 20 if is_press else 24
    cutpoint_marker_size = 18 if is_press else 24
    if use_3d:
        ax = fig.add_subplot(gs[0, 0], projection='3d')
    else:
        ax = fig.add_subplot(gs[0, 0])

    ax.set_title("Demos & goals" if plot_context == "posthoc" else f"Iter {it}: demos & goals", fontsize=PAPER_TITLE_SIZE, pad=4)

    X_dim = learner.demos[0].shape[1]

    # ================== demos + cutpoints ==================
    if use_3d:
        for i, (X, boundary_like, gamma) in enumerate(zip(learner.demos, taus, gammas)):
            X = np.asarray(X)
            T = len(X)
            stage_ends = _coerce_stage_ends(boundary_like, gamma, T)
            starts, ends = _segment_bounds(stage_ends)
            for k, (s, e) in enumerate(zip(starts, ends)):
                ax.scatter(
                    X[s : e + 1, 0], X[s : e + 1, 1], X[s : e + 1, 2],
                    color=stage_colors[k], s=2, alpha=0.35, depthshade=False
                )

    else:
        for i, (X, boundary_like, gamma) in enumerate(zip(learner.demos, taus, gammas)):
            X = np.asarray(X)
            T = len(X)
            stage_ends = _coerce_stage_ends(boundary_like, gamma, T)
            starts, ends = _segment_bounds(stage_ends)
            for k, (s, e) in enumerate(zip(starts, ends)):
                ax.scatter(
                    X[s : e + 1, 0],
                    X[s : e + 1, 1],
                    color=stage_colors[k],
                    s=traj_marker_size,
                    alpha=0.28 if is_press else 0.35,
                )

    # ================== obstacle ==================
    if use_3d and _env_has_3d_obstacle(learner.env):
        # 3D env: obs_center_xy + obs_radius
        cx, cy = learner.env.obs_center_xy
        r = learner.env.obs_radius
        # _draw_cylinder_wire(
        #     ax,
        #     center_xy=(cx, cy),
        #     radius=r,
        #     z0=0.0,
        #     height=0.5,
        #     color='gray'
        # )
    elif not use_3d and _env_has_xy_obstacle(learner.env):
        _draw_env_xy_obstacles(ax, learner.env)

    _draw_chmm_gmms(ax, learner, X_dim)

    # ================== true goals ==================
    if use_3d:
        for pt_idx, sg in enumerate(_all_true_stage_end_points(learner.env, "subgoal")):
            ax.scatter(sg[0], sg[1], sg[2], color='black', marker='X', s=28, alpha=0.55, label='true stage end' if pt_idx == 0 else "")
        subgoal_count = len(_all_true_stage_end_points(learner.env, "subgoal"))
        for pt_idx, gg in enumerate(_all_true_stage_end_points(learner.env, "goal")):
            ax.scatter(gg[0], gg[1], gg[2], color='black', marker='X', s=28, alpha=0.55, label='true stage end' if (subgoal_count == 0 and pt_idx == 0) else "")
    else:
        if not _is_pickplace(learner.env):
            for pt_idx, sg in enumerate(_all_true_stage_end_points(learner.env, "subgoal")):
                sg_xy = _xy_point(sg)
                ax.scatter(sg_xy[0], sg_xy[1], color='black', marker='X', s=goal_marker_size, alpha=0.55, label='true stage end' if pt_idx == 0 else "")
            subgoal_count = len(_all_true_stage_end_points(learner.env, "subgoal"))
            for pt_idx, gg in enumerate(_all_true_stage_end_points(learner.env, "goal")):
                gg_xy = _xy_point(gg)
                ax.scatter(gg_xy[0], gg_xy[1], color='black', marker='X', s=goal_marker_size, alpha=0.55, label='true stage end' if (subgoal_count == 0 and pt_idx == 0) else "")

    history_linestyles = ["-", "--", ":", "-."]
    for k, sg in enumerate(shared_subgoals):
        hist = _stage_subgoals_hist(learner, k)
        if hist is not None and len(hist) > 1:
            ls = history_linestyles[k % len(history_linestyles)]
            if use_3d:
                ax.plot(hist[:, 0], hist[:, 1], hist[:, 2], ls, lw=1.8, alpha=0.7, color='black', label=f'g{k + 1} history')
                ax.scatter(hist[:, 0], hist[:, 1], hist[:, 2], s=8, alpha=0.45, color='black')
            else:
                ax.plot(hist[:, 0], hist[:, 1], ls, lw=1.8, alpha=0.7, color='black', label=f'g{k + 1} history')
                ax.scatter(hist[:, 0], hist[:, 1], s=8, alpha=0.45, color='black')

    if use_3d:
        for i, X in enumerate(learner.demos):
            _draw_true_cutpoint_markers(
                ax,
                np.asarray(X),
                _true_cutpoints(learner, i),
                colors=stage_colors,
                label='true cutpoint' if i == 0 else "",
                is_3d=True,
                size=28,
                zorder=13,
            )
    else:
        for i, X in enumerate(learner.demos):
            _draw_true_cutpoint_markers(
                ax,
                np.asarray(X),
                _true_cutpoints(learner, i),
                colors=stage_colors,
                label='true cutpoint' if i == 0 else "",
                is_3d=False,
                size=cutpoint_marker_size,
                zorder=11,
            )

    if use_3d:
        for k, sg in enumerate(shared_subgoals):
            ax.scatter(sg[0], sg[1], sg[2], color=stage_colors[k], marker='D', s=24, edgecolors='black', linewidths=0.8, label=f'est. stage {k + 1} end')
    else:
        for k, sg in enumerate(shared_subgoals):
            g_xy = _xy_point(sg)
            ax.scatter(g_xy[0], g_xy[1], color=stage_colors[k], marker='D', s=subgoal_marker_size, edgecolors='black', linewidths=0.8, label=f'est. stage {k + 1} end')

    # ================== axes labels + scaling ==================
    if use_3d:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        # ---- collect all points for 3D scaling ----
        xyz_all = []

        for X in learner.demos:
            X = np.asarray(X)
            if X.shape[1] == 2:
                X = np.pad(X, ((0, 0), (0, 1)), mode='constant')
            xyz_all.append(X)

        for pt in _all_true_stage_end_points(learner.env, "subgoal"):
            xyz_all.append(np.asarray(pt, dtype=float)[None, :])
        for pt in _all_true_stage_end_points(learner.env, "goal"):
            xyz_all.append(np.asarray(pt, dtype=float)[None, :])

        # learned goals + history
        for k in range(len(shared_subgoals)):
            hist = _stage_subgoals_hist(learner, k)
            if hist is not None:
                xyz_all.append(np.asarray(hist))
            xyz_all.append(np.asarray(shared_subgoals[k])[None, :])

        # English comment omitted during cleanup.
        if _env_has_3d_obstacle(learner.env):
            cx, cy = learner.env.obs_center_xy
            r = learner.env.obs_radius
            z0, z1 = 0.0, 0.5
            cyl_bbox = np.array([
                [cx - r, cy - r, z0],
                [cx + r, cy + r, z1]
            ])
            xyz_all.append(cyl_bbox)

        xyz_all = np.concatenate(xyz_all, axis=0)
        _set_axes_equal_3d_from_xyz(ax, xyz_all)

    else:
        ax.set_xlabel("x")
        ax.set_ylabel("z" if _is_pickplace(learner.env) or _is_press_slide_insert(learner.env) else "y")

        # English comment omitted during cleanup.
        pts = []
        for X in learner.demos:
            pts.append(np.asarray(X)[:, :2])
        if _is_pickplace(learner.env):
            if hasattr(learner.env, "pick_point"):
                pick_xy = _xy_point(learner.env.pick_point)
                ax.scatter(pick_xy[0], pick_xy[1], c="green", marker="*", s=40, label="pick")
                pts.append(pick_xy[None, :])
            if hasattr(learner.env, "place_point"):
                place_xy = _xy_point(learner.env.place_point)
                ax.scatter(place_xy[0], place_xy[1], c="green", marker="s", s=34, label="place")
                pts.append(place_xy[None, :])
            if hasattr(learner.env, "retreat_point"):
                retreat_xy = _xy_point(learner.env.retreat_point)
                ax.scatter(retreat_xy[0], retreat_xy[1], c="green", marker="P", s=34, label="retreat")
                pts.append(retreat_xy[None, :])
        else:
            for pt in _all_true_stage_end_points(learner.env, "subgoal"):
                pts.append(_xy_point(pt)[None, :])
            for pt in _all_true_stage_end_points(learner.env, "goal"):
                pts.append(_xy_point(pt)[None, :])

        if _env_has_xy_obstacle(learner.env):
            cx, cy = learner.env.obs_center
            r = learner.env.obs_radius
            pts.append(np.array([[cx - r, cy - r], [cx + r, cy + r]]))

        pts = np.concatenate(pts, axis=0)
        _configure_2d_trajectory_axes(ax, learner.env, pts)

    # English comment omitted during cleanup.
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l is None:
            continue
        l = str(l).strip()
        if l == "" or l.startswith("_"):
            continue
        if l not in by_label:
            by_label[l] = h

    ax.legend(
        by_label.values(),
        by_label.keys(),
        fontsize=PAPER_LEGEND_SIZE,
        frameon=False,
        handlelength=1.2,
        borderpad=0.2,
        **_trajectory_legend_kwargs(learner.env),
    )
    ax.tick_params(labelsize=PAPER_TICK_SIZE)

    next_row = 1
    if show_learning_row:
        ax1 = fig.add_subplot(gs[0, 1])
        iters = np.arange(len(learner.loss_loglik))

        loss_curve_label = getattr(learner, "loss_label", "Log-likelihood")
        ax1.plot(iters, learner.loss_loglik, '-o', color='black', label=loss_curve_label, markersize=2.5, linewidth=1.0)
        ax1.set_xlabel("Iteration", fontsize=PAPER_LABEL_SIZE)
        ax1.set_ylabel(loss_curve_label, fontsize=PAPER_LABEL_SIZE)
        ax1.set_title(learning_title, fontsize=PAPER_TITLE_SIZE, pad=4)
        ax1.tick_params(labelsize=PAPER_TICK_SIZE)

        ax2 = ax1.twinx()
        ax2.set_ylabel("Metrics", fontsize=PAPER_LABEL_SIZE)
        ax2.tick_params(labelsize=PAPER_TICK_SIZE)

        metrics_hist = getattr(learner, "metrics_hist", None)
        if isinstance(metrics_hist, dict) and len(metrics_hist) > 0:
            color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            from itertools import cycle
            color_iter = cycle(color_cycle)
            metric_values = []

            for name, seq in metrics_hist.items():
                if not seq:
                    continue
                Tm = min(len(seq), len(iters))
                plot_seq = []
                seq_values = []
                for v in seq[:Tm]:
                    if not np.isscalar(v):
                        plot_seq.append(np.nan)
                        continue
                    v_f = float(v)
                    if np.isfinite(v_f):
                        seq_values.append(v_f)
                        plot_seq.append(v_f)
                    else:
                        plot_seq.append(np.nan)
                valid_seq = seq_values
                if not valid_seq:
                    continue
                metric_values.extend(valid_seq)
                ax2.plot(
                    iters[:Tm],
                    np.asarray(plot_seq, dtype=float),
                    linestyle='--',
                    linewidth=1.0,
                    label=name,
                    color=next(color_iter),
                )

            if metric_values:
                ymax = max(metric_values)
                ax2.set_ylim(0.0, max(1.0, ymax * 1.15))

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)
        else:
            ax1.legend(loc='best', fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)
        posterior_row = 2
    else:
        posterior_row = 1

    # ==========================================================
    # (3) Feature evolution (demo 0): raw + standardized
    # ==========================================================
    feature_row = 1
    ax_feat_raw = fig.add_subplot(gs[feature_row, 0])
    _plot_feature_evolution_panel(ax_feat_raw, learner, taus, gammas, demo_idx=0, standardized=False)

    ax_feat_std = fig.add_subplot(gs[feature_row, 1])
    _plot_feature_evolution_panel(ax_feat_std, learner, taus, gammas, demo_idx=0, standardized=True)

    # ==========================================================
    # (4) Posterior decomposition / state evidence
    # ==========================================================
    ax = fig.add_subplot(gs[posterior_row, :])

    alpha0 = alphas[0]
    beta0 = betas[0]
    gamma0 = gammas[0]
    X0 = learner.demos[0]
    T0 = len(X0)
    t_axis = np.arange(T0)
    stage_ends0 = _coerce_stage_ends(taus[0], gamma0, T0)
    true_cutpoints0 = _true_cutpoints(learner, 0)
    eps = 1e-8

    # English comment omitted during cleanup.
    # English comment omitted during cleanup.
    # English comment omitted during cleanup.
    # English comment omitted during cleanup.
    F = learner._features_for_demo_matrix(X0)  # (T0, M)
    T0, M = F.shape
    K = learner.num_stages

    feat_scale = float(getattr(learner, "feat_weight", 1.0))
    if hasattr(learner, "_feature_loglik_matrix"):
        ll_feat_state = feat_scale * learner._feature_loglik_matrix(X0, demo_idx=0)
    else:
        rel_logpdf = np.zeros((T0, K, M))
        irrel_logpdf = np.zeros((T0, M))
        for m in range(M):
            irrel_logpdf[:, m] = learner._log_irrelevant(F[:, m])
        for k in range(K):
            for m in range(M):
                rel_logpdf[:, k, m] = learner.feature_models[k][m].logpdf(F[:, m])
        ll_feat_state = np.zeros((T0, K))
        for k in range(K):
            for m in range(M):
                if learner.r[k, m] == 1:
                    ll_feat_state[:, k] += rel_logpdf[:, k, m]
                else:
                    ll_feat_state[:, k] += irrel_logpdf[:, m]
        ll_feat_state *= feat_scale

    ll_emit_full = learner._emission_loglik(X0)
    ll_x_state = ll_emit_full - ll_feat_state
    if K == 2:
        current_g1 = shared_subgoals[0]
        current_g2 = shared_subgoals[1]
        ll_feat1 = ll_feat_state[:, 0]
        ll_feat2 = ll_feat_state[:, 1]
        d_feat = ll_feat2 - ll_feat1
        ll_x1 = ll_x_state[:, 0]
        ll_x2 = ll_x_state[:, 1]
        d_x = ll_x2 - ll_x1

        d_prog = None
        prog_weight = float(getattr(learner, "prog_weight", 0.0))
        if prog_weight > 0 and T0 > 1:
            ll_prog1 = np.zeros(T0)
            ll_prog2 = np.zeros(T0)
            progress_delta_scale = float(getattr(learner, "progress_delta_scale", 10.0))
            X_next = X0[1:]
            d1_curr = np.linalg.norm(X0[:-1] - np.asarray(current_g1)[None, :], axis=1)
            d1_next = np.linalg.norm(X_next - np.asarray(current_g1)[None, :], axis=1)
            d2_curr = np.linalg.norm(X0[:-1] - np.asarray(current_g2)[None, :], axis=1)
            d2_next = np.linalg.norm(X_next - np.asarray(current_g2)[None, :], axis=1)
            delta1 = d1_next - d1_curr
            delta2 = d2_next - d2_curr
            ll_prog1[:-1] = prog_weight * (
                np.log1p(np.exp(np.clip(progress_delta_scale * delta1, -60.0, 60.0))) / progress_delta_scale
            )
            ll_prog2[:-1] = prog_weight * (
                np.log1p(np.exp(np.clip(progress_delta_scale * delta2, -60.0, 60.0))) / progress_delta_scale
            )
            d_prog = ll_prog2 - ll_prog1
        d_trans = None
        if hasattr(learner, "_transition_logprob"):
            logA0 = learner._transition_logprob(X0, return_aux=False)
            p12 = np.zeros(T0)
            if logA0.shape[0] > 0:
                p12[:-1] = np.exp(logA0[:, 0, 1])
                p12[-1] = p12[-2]
            d_trans = np.log((p12 + eps) / (1.0 - p12 + eps))
        post_odds = np.log((gamma0[:, 1] + eps) / (gamma0[:, 0] + eps))
        ax.plot(t_axis, d_x, '-', lw=1.2, color='tab:cyan', label='x diff')
        ax.plot(t_axis, d_feat, '-', lw=1.1, color='tab:red', label='feat diff')
        if d_prog is not None:
            ax.plot(t_axis, d_prog, '-', lw=1.1, color='tab:blue', label='prog diff')
        if d_trans is not None and getattr(learner, "plot_context", "") not in {"fchmm", "hmm"}:
            ax.plot(t_axis, d_trans, '--', lw=1.2, color='tab:orange', label='trans diff')
        ax.plot(t_axis, post_odds, '-', lw=1.6, color='black', label='posterior log-odds')
        for j, cp in enumerate(stage_ends0[:-1]):
            ax.axvline(cp, color='black', linestyle='--', label='learned cutpoint' if j == 0 else "")
        for j, cp in enumerate(true_cutpoints0):
            ax.axvline(cp, color='gray', linestyle=':', label='true cutpoint' if j == 0 else "")
        ax.axhline(0, color='gray', lw=1, alpha=0.4)
        ax.set_ylabel("log-odds diff", fontsize=PAPER_LABEL_SIZE)
        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1, labels1, loc='best', fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)
    else:
        for stage_idx in range(K):
            ax.plot(t_axis, ll_x_state[:, stage_idx], '-', lw=1.0, color=stage_colors[stage_idx], alpha=0.65, label=f'x score s{stage_idx + 1}')
            ax.plot(t_axis, ll_feat_state[:, stage_idx], '--', lw=1.0, color=stage_colors[stage_idx], alpha=0.95, label=f'feat score s{stage_idx + 1}')
        for j, cp in enumerate(stage_ends0[:-1]):
            ax.axvline(cp, color='black', linestyle='--', label='learned cutpoint' if j == 0 else "")
        for j, cp in enumerate(true_cutpoints0):
            ax.axvline(cp, color='gray', linestyle=':', label='true cutpoint' if j == 0 else "")
        ax.set_ylabel("state score", fontsize=PAPER_LABEL_SIZE)
        lines1, labels1 = ax.get_legend_handles_labels()
        ax.legend(lines1, labels1, loc='best', fontsize=PAPER_LEGEND_SIZE, frameon=False, handlelength=1.2, borderpad=0.2)

    ax.set_title(decomposition_title, fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("t", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)

    next_row = posterior_row + 1
    if show_cutpoint_row:
        _plot_cutpoint_evolution_panels(fig, gs[next_row, :], learner, max_cols=4)
        next_row += 1

    if show_summary_row:
        summary_gs = gs[next_row, :].subgridspec(1, 2, wspace=0.32)
        ax_metrics = fig.add_subplot(summary_gs[0, 0])
        _draw_eval_metric_text(ax_metrics, metrics)

        ax_error = fig.add_subplot(summary_gs[0, 1])
        error_matrix = np.asarray(metrics.get("ConstraintErrorMatrix", []), dtype=float)
        _draw_summary_heatmap(
            ax_error,
            error_matrix.T if error_matrix.ndim == 2 else error_matrix,
            "normalized constraint error",
            feature_names=_summary_feature_names(learner, metrics),
            stage_labels=_summary_stage_labels(learner),
            cmap="YlOrRd",
            fmt=".3f",
        )
        next_row += 1

    if extra_rows > 0:
        _plot_constraint_parameter_panels(fig, gs[next_row:, :], learner, taus, gammas, stage_colors)

    fig.tight_layout(pad=0.5, w_pad=0.6, h_pad=0.8)
    if save_name is None:
        if plot_context == "posthoc":
            save_name = "training_summary_posthoc_final.png"
        else:
            save_name = f"plot4panel_iter_{int(it):04d}.png"
    save_figure(fig, learner_plot_dir(learner) / str(save_name), dpi=220)


    # # ==========================================================
    # English comment omitted during cleanup.
    # # ==========================================================
    # plt.figure(figsize=(10, 4))
    # xi0 = xis_list[0]
    # aux0 = aux_list[0]
    #
    # p12 = aux0["p12"][:-1]
    # p12 = np.clip(p12, learner.trans_eps, 1.0 - learner.trans_eps)
    #
    # xi01 = xi0[:, 0, 1]
    # xi00 = xi0[:, 0, 0]
    #
    # A_pos = xi01 / (p12 + 1e-12)
    # A_neg = xi00 / (1.0 - p12 + 1e-12)
    # A = A_pos - A_neg
    #
    # t_axis2 = np.arange(len(A))
    # plt.plot(t_axis2, A_pos, label="A_pos = xi01 / p_t")
    # plt.plot(t_axis2, A_neg, label="A_neg = xi00 / (1-p_t)")
    # plt.plot(t_axis2, A, lw=3, label="A = A_pos - A_neg", color="black")
    # plt.axhline(0, color="gray", lw=1)
    # plt.xlabel("t")
    # plt.ylabel("value")
    # plt.title("Transition gradient difference term")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    #
    # # effective force
    # sigma = getattr(learner, "d0_trans", getattr(learner, "trans_delta", 0.2))
    # d = aux0["dists"][:-1]
    # e = np.exp(-0.5 * (d ** 2) / (sigma ** 2 + 1e-12))
    # F = A * e * d / (sigma ** 2 + 1e-12)
    #
    # plt.figure(figsize=(10, 4))
    # plt.plot(F, lw=2, label="effective force magnitude (signed)")
    # plt.axhline(0, color="gray")
    # plt.legend()
    # plt.title("Signed effective transition force on g1")
    # plt.show()


def plot_feature_model_debug(learner, gammas, stages=(0, 1), max_bins=40):
    if plt is None:
        return
    """
    English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
    English documentation omitted during cleanup.
      - weighted mean log p_rel
      - weighted mean log p_bg
      - diff = rel - bg

    English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
      English documentation omitted during cleanup.
    """
    stages = tuple(stages)
    K = learner.num_stages
    M = learner.num_features

    assert max(stages) < K, f"stages={stages}    state    K={K}"

    # English comment omitted during cleanup.
    Fz_all = []      #      demo   Fz (T_i, M)
    gamma_all = []   #      demo   gamma (T_i, K)
    for X, gamma in zip(learner.demos, gammas):
        Fz = learner._features_for_demo_matrix(X)   # (T_i, M)
        Fz_all.append(Fz)
        gamma_all.append(gamma)

    # English comment omitted during cleanup.
    for k in stages:
        fig, axes = plt.subplots(
            nrows=int(np.ceil(M / 2)),
            ncols=2,
            figsize=(7.2, 2.4 * int(np.ceil(M / 2))),
        )
        axes = np.asarray(axes).reshape(-1)  #     1D list
        fig.suptitle(f"Stage {k}: feature distributions", fontsize=PAPER_TITLE_SIZE)

        for m in range(M):
            ax = axes[m]
            # English comment omitted during cleanup.
            z_list = []
            w_list = []
            for Fz, gamma in zip(Fz_all, gamma_all):
                z_list.append(Fz[:, m])
                w_list.append(gamma[:, k])
            z_all = np.concatenate(z_list, axis=0).astype(float)
            w_all = np.concatenate(w_list, axis=0).astype(float)

            # English comment omitted during cleanup.
            w_all = np.maximum(w_all, 0.0)
            if np.sum(w_all) <= 1e-8:
                ax.set_title(f"f{m} (no effective data)")
                ax.axis("off")
                continue

            # English comment omitted during cleanup.
            # English comment omitted during cleanup.
            z_min = np.percentile(z_all, 1.0)
            z_max = np.percentile(z_all, 99.0)
            if z_max <= z_min:
                z_min = z_all.min()
                z_max = z_all.max()
            pad = 0.1 * (z_max - z_min + 1e-6)
            z_min -= pad
            z_max += pad

            # English comment omitted during cleanup.
            bins = min(max_bins, int(len(z_all) / 10) + 5)
            ax.hist(
                z_all,
                bins=bins,
                range=(z_min, z_max),
                weights=w_all,
                density=False,
                alpha=0.4,
                edgecolor="none",
                label="data (weighted hist)",
            )

            # English comment omitted during cleanup.
            y_max = 1e-8
            for p in ax.patches:
                y_max = max(y_max, p.get_height())
            if y_max <= 0:
                y_max = 1.0

            # English comment omitted during cleanup.
            z_grid = np.linspace(z_min, z_max, 300)

            # English comment omitted during cleanup.
            logp_bg_grid = learner._log_irrelevant(z_grid)
            # English comment omitted during cleanup.
            model = learner.feature_models[k][m]
            logp_rel_grid = model.logpdf(z_grid)

            # English comment omitted during cleanup.
            #   curve(x) = exp(logp(x) - max_logp) * y_max
            max_bg = np.max(logp_bg_grid)
            max_rel = np.max(logp_rel_grid)

            curve_bg = np.exp(logp_bg_grid - max_bg) * y_max
            curve_rel = np.exp(logp_rel_grid - max_rel) * y_max

            ax.plot(
                z_grid,
                curve_bg,
                "--",
                linewidth=1.8,
                label="background (shape only)",
            )
            ax.plot(
                z_grid,
                curve_rel,
                "-",
                linewidth=1.8,
                label="feature model (shape only)",
            )

            # English comment omitted during cleanup.
            w_norm = w_all / (np.sum(w_all) + 1e-12)
            logp_bg_all = learner._log_irrelevant(z_all)
            logp_rel_all = model.logpdf(z_all)

            mean_ll_bg = float(np.sum(w_norm * logp_bg_all))
            mean_ll_rel = float(np.sum(w_norm * logp_rel_all))
            diff = mean_ll_rel - mean_ll_bg

            # English comment omitted during cleanup.
            feat_type = getattr(learner, "feature_model_types", None)
            if feat_type is not None:
                type_str = feat_type[m]
            else:
                type_str = type(model).__name__

            ax.set_title(
                f"Stage {k}, f{m} ({type_str})\n"
                f" log p_rel ={mean_ll_rel:.2f}, "
                f" log p_bg ={mean_ll_bg:.2f},  ={diff:.2f}",
                fontsize=7,
            )
            ax.set_xlabel("z", fontsize=PAPER_LABEL_SIZE)
            ax.set_ylabel("weighted counts", fontsize=PAPER_LABEL_SIZE)
            ax.tick_params(labelsize=PAPER_TICK_SIZE)

            ax.legend(fontsize=PAPER_LEGEND_SIZE, loc="best", frameon=False, handlelength=1.2, borderpad=0.2)

        # English comment omitted during cleanup.
        for j in range(M, len(axes)):
            axes[j].axis("off")

        plt.tight_layout(rect=[0, 0.02, 1, 0.95], pad=0.5, w_pad=0.5, h_pad=0.8)
        save_figure(fig, learner_plot_dir(learner) / f"feature_debug_stage_{int(k)}.png", dpi=220)
