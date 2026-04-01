from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except ModuleNotFoundError:
    Axes3D = None

from .io import learner_plot_dir, save_figure
from utils.models import GaussianModel

PAPER_FIGSIZE = (8.4, 6.0)
PAPER_TITLE_SIZE = 9
PAPER_LABEL_SIZE = 8
PAPER_TICK_SIZE = 7
PAPER_LEGEND_SIZE = 6.5
STAGE_COLORS = ["#D55E00", "#0072B2", "#CC79A7", "#009E73"]


def _legend(ax, *, outside=False):
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l is None:
            continue
        text = str(l).strip()
        if text and not text.startswith("_") and text not in by_label:
            by_label[text] = h
    if by_label:
        kwargs = {"loc": "best"}
        if outside:
            kwargs = {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0)}
        ax.legend(by_label.values(), by_label.keys(), fontsize=PAPER_LEGEND_SIZE, frameon=False, **kwargs)


def _style_paper_axis(ax, *, grid_axis=None, grid_alpha=0.16):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=PAPER_TICK_SIZE, width=0.7, length=3.0)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color="#cfcfcf", linewidth=0.6, alpha=grid_alpha)


def _add_slim_colorbar(im, ax):
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.ax.tick_params(labelsize=PAPER_TICK_SIZE - 0.5, width=0.6, length=2.5)
    cbar.outline.set_linewidth(0.6)
    return cbar


def _segment_bounds(stage_ends):
    starts = []
    ends = []
    prev = -1
    for end in stage_ends:
        starts.append(prev + 1)
        ends.append(int(end))
        prev = int(end)
    return starts, ends


def _core_bounds_for_display(learner, s, e):
    if hasattr(learner, "_segment_core_bounds"):
        core_s, core_e = learner._segment_core_bounds(int(s), int(e))
        return int(core_s), int(core_e)
    return int(s), int(e)


def _feature_name(learner, local_idx):
    schema = getattr(learner, "raw_feature_specs", None) or []
    selected_col = int(learner.selected_feature_columns[local_idx])
    for i, spec in enumerate(schema):
        if int(spec.get("column_idx", i)) == selected_col:
            return str(spec.get("name", f"f{local_idx}"))
    return f"f{local_idx}"


def _feature_plot_colors(num_features: int):
    n = max(int(num_features), 1)
    return plt.cm.tab10(np.linspace(0.0, 1.0, n))


def _feature_kind(learner, local_idx):
    feature_model_types = getattr(learner, "feature_model_types", None)
    if feature_model_types is None or local_idx >= len(feature_model_types):
        return "gaussian"
    return str(feature_model_types[local_idx]).lower()


def _kind_is_auto_display(kind: str) -> bool:
    return str(kind).lower() in {"auto", "auto_constraint", "auto_eq_ineq", "auto_constraint_type"}


def _kind_is_equality_display(kind: str) -> bool:
    return str(kind).lower() in {"gauss", "gaussian", "student_t", "studentt", "t", "zero_gauss", "zero_gaussian"}


def _summary_center_z(summary: dict, kind: str):
    kind_l = str(kind).lower()
    if _kind_is_equality_display(kind_l):
        if "mu" in summary:
            return float(summary["mu"])
        if kind_l in {"zero_gauss", "zero_gaussian"}:
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


def _z_to_display(learner, feat_idx: int, value: float, standardized: bool):
    if not np.isfinite(value):
        return np.nan
    fid = learner.selected_feature_columns[feat_idx]
    if standardized:
        return float(value)
    return float(value) * float(learner.feat_std[fid]) + float(learner.feat_mean[fid])


def _summary_center_display(learner, feat_idx: int, summary: dict, kind: str, standardized: bool):
    return _z_to_display(learner, feat_idx, _summary_center_z(summary, kind), standardized)


def _summary_interval_display(learner, feat_idx: int, summary: dict, standardized: bool):
    low_z, high_z = _summary_interval(summary)
    if low_z is None or high_z is None:
        return np.nan, np.nan
    return (
        _z_to_display(learner, feat_idx, low_z, standardized),
        _z_to_display(learner, feat_idx, high_z, standardized),
    )


def _stage_feature_kind_for_display(learner, local_stage_params, stage_idx, feat_idx):
    try:
        stage_params = local_stage_params[int(stage_idx)]
        selected_kinds = getattr(stage_params, "selected_feature_kinds", None)
        if selected_kinds is not None and int(feat_idx) < len(selected_kinds):
            kind = str(selected_kinds[int(feat_idx)]).lower()
            if kind:
                if kind == "unconstrained":
                    return "student_t"
                return kind
    except Exception:
        pass
    base_kind = _feature_kind(learner, feat_idx)
    if _kind_is_auto_display(base_kind):
        return "student_t"
    return base_kind


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


def _raw_feature_matrix_for_demo(learner, demo_idx: int) -> np.ndarray:
    Fz = np.asarray(learner.standardized_features[demo_idx], dtype=float)
    feat_std = np.asarray(learner.feat_std, dtype=float)[learner.selected_feature_columns]
    feat_mean = np.asarray(learner.feat_mean, dtype=float)[learner.selected_feature_columns]
    return Fz * feat_std[None, :] + feat_mean[None, :]


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


def _feature_has_constraint_reference(learner, feat_idx: int) -> bool:
    feature_name = _feature_name(learner, feat_idx)
    for stage_idx in range(int(getattr(learner, "num_stages", 0))):
        if _reference_constraint_value(learner.env, feature_name, stage_idx) is not None:
            return True
    return False


def _plot_constraint_parameter_panels(ax_specs, learner):
    if plt is None or not getattr(learner, "current_stage_params_per_demo", None):
        return
    feature_indices = list(range(int(getattr(learner, "num_features", 0))))
    if not feature_indices:
        return

    n_panels = len(feature_indices)
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / max(n_cols, 1)))
    sub_gs = ax_specs.subgridspec(n_rows, n_cols, wspace=0.35, hspace=0.55)
    demo_x = np.arange(len(learner.demos))
    colors = _stage_colors(learner.num_stages)

    for panel_idx, feat_idx in enumerate(feature_indices):
        ax = plt.figure(plt.gcf().number).add_subplot(sub_gs[panel_idx // n_cols, panel_idx % n_cols])
        feature_name = _feature_name(learner, feat_idx)
        any_series = False

        for stage_idx in range(int(getattr(learner, "num_stages", 0))):
            learned_vals = []
            active_any = False
            for demo_idx, (stage_ends, local_stage_params) in enumerate(
                zip(learner.stage_ends_, learner.current_stage_params_per_demo)
            ):
                starts, ends = _segment_bounds(stage_ends)
                if stage_idx >= len(starts):
                    learned_vals.append(np.nan)
                    continue
                if not _feature_stage_is_active_for_display(learner, local_stage_params, stage_idx, feat_idx):
                    learned_vals.append(np.nan)
                    continue
                active_any = True
                stage_kind = _stage_feature_kind_for_display(learner, local_stage_params, stage_idx, feat_idx)
                summary = local_stage_params[stage_idx].model_summaries[feat_idx]
                learned_vals.append(_summary_center_display(learner, feat_idx, summary, stage_kind, standardized=False))

            if active_any:
                ax.plot(
                    demo_x,
                    np.asarray(learned_vals, dtype=float),
                    marker="o",
                    linewidth=1.35,
                    markersize=3.4,
                    color=colors[stage_idx % len(colors)],
                    markerfacecolor=colors[stage_idx % len(colors)],
                    markeredgecolor="white",
                    markeredgewidth=0.35,
                    label=f"s{stage_idx + 1} learned",
                )
                any_series = True

            ref_value = _reference_constraint_value(learner.env, feature_name, stage=stage_idx)
            if ref_value is not None:
                ax.axhline(
                    float(ref_value),
                    color=colors[stage_idx % len(colors)],
                    linestyle=(0, (4, 2)),
                    linewidth=1.0,
                    alpha=0.9,
                    label=f"s{stage_idx + 1} true",
                )
                any_series = True

        if not any_series:
            ax.set_title(f"{feature_name}", fontsize=PAPER_TITLE_SIZE, pad=4)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            continue

        ax.set_title(f"{feature_name}", fontsize=PAPER_TITLE_SIZE, pad=4)
        ax.set_xlabel("demo", fontsize=PAPER_LABEL_SIZE)
        ax.set_ylabel("learned value", fontsize=PAPER_LABEL_SIZE)
        ax.set_xticks(demo_x)
        ax.set_xticklabels([str(i) for i in demo_x], fontsize=PAPER_TICK_SIZE)
        _style_paper_axis(ax, grid_axis="y", grid_alpha=0.18)
        _legend(ax)

    for panel_idx in range(n_panels, n_rows * n_cols):
        ax = plt.figure(plt.gcf().number).add_subplot(sub_gs[panel_idx // n_cols, panel_idx % n_cols])
        ax.axis("off")


def _constraint_type_bucket_for_display(learner, local_stage_params, stage_idx, feat_idx):
    if not _feature_stage_is_active_for_display(learner, local_stage_params, stage_idx, feat_idx):
        return "unconstrained"
    kind = _stage_feature_kind_for_display(learner, local_stage_params, stage_idx, feat_idx)
    if _kind_is_equality_display(kind):
        return "equality"
    if str(kind).lower() in {
        "margin_exp_lower", "marginexp", "margin_exp",
        "margin_exp_lower_left_hn", "marginexp_left_hn", "margin_exp_left_hn",
    }:
        return "lower"
    if str(kind).lower() in {
        "margin_exp_upper", "marginexp_upper", "margin_exp_upper",
        "margin_exp_upper_right_hn", "marginexp_upper_right_hn", "margin_exp_upper_right_hn",
    }:
        return "upper"
    return "unconstrained"


def plot_constraint_type_summary(learner, it, *, plot_dir=None):
    if plt is None or not getattr(learner, "current_stage_params_per_demo", None):
        return
    num_features = int(getattr(learner, "num_features", 0))
    num_stages = int(getattr(learner, "num_stages", 0))
    num_demos = len(getattr(learner, "current_stage_params_per_demo", []) or [])
    if num_features <= 0 or num_stages <= 0 or num_demos <= 0:
        return

    feature_indices = list(range(num_features))
    n_cols = min(4, max(1, num_features))
    n_rows = int(np.ceil(num_features / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.1 * n_cols, 3.2 * n_rows),
        squeeze=False,
    )

    bucket_order = ["unconstrained", "equality", "lower", "upper"]
    bucket_colors = {
        "unconstrained": "#b8b8b8",
        "equality": "#0072B2",
        "lower": "#D55E00",
        "upper": "#009E73",
    }
    x = np.arange(num_stages, dtype=float)

    for panel_idx, feat_idx in enumerate(feature_indices):
        ax = axes[panel_idx // n_cols][panel_idx % n_cols]
        counts = {bucket: np.zeros(num_stages, dtype=float) for bucket in bucket_order}
        for local_stage_params in learner.current_stage_params_per_demo:
            for stage_idx in range(num_stages):
                bucket = _constraint_type_bucket_for_display(learner, local_stage_params, stage_idx, feat_idx)
                counts[bucket][stage_idx] += 1.0
        denom = max(float(num_demos), 1.0)
        bottom = np.zeros(num_stages, dtype=float)
        for bucket in bucket_order:
            frac = counts[bucket] / denom
            ax.bar(
                x,
                frac,
                bottom=bottom,
                width=0.72,
                color=bucket_colors[bucket],
                edgecolor="white",
                linewidth=0.5,
                label=bucket,
            )
            bottom += frac
        ax.set_title(_feature_name(learner, feat_idx), fontsize=PAPER_TITLE_SIZE, pad=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{k + 1}" for k in range(num_stages)], fontsize=PAPER_TICK_SIZE)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("fraction", fontsize=PAPER_LABEL_SIZE)
        _style_paper_axis(ax, grid_axis="y", grid_alpha=0.16)
        if panel_idx == 0:
            _legend(ax, outside=True)

    for panel_idx in range(num_features, n_rows * n_cols):
        axes[panel_idx // n_cols][panel_idx % n_cols].axis("off")

    fig.suptitle(f"Constraint Type Summary | iter {int(it):04d}", fontsize=12)
    save_figure(
        fig,
        learner_plot_dir(learner, plot_dir=plot_dir) / f"constraint_type_summary_iter_{int(it):04d}.png",
        dpi=220,
    )


def _score_threshold(learner, feat_idx, stage_idx=None):
    try:
        score_threshold_matrix = np.asarray(getattr(learner, "score_threshold_matrix", None), dtype=float)
        if score_threshold_matrix.ndim == 2 and score_threshold_matrix.size > 0:
            row_idx = 0 if stage_idx is None else int(stage_idx)
            row_idx = max(0, min(row_idx, score_threshold_matrix.shape[0] - 1))
            col_idx = max(0, min(int(feat_idx), score_threshold_matrix.shape[1] - 1))
            return float(score_threshold_matrix[row_idx, col_idx])
    except Exception:
        pass
    if hasattr(learner, "_is_equality_feature") and learner._is_equality_feature(int(feat_idx)):
        if hasattr(learner, "_equality_score_threshold"):
            return float(learner._equality_score_threshold())
        return float(getattr(learner, "equality_dispersion_ratio_threshold", 0.1))
    return float(getattr(learner, "inequality_score_activation_threshold", -0.5))


def _matrix_text_color(value, vmax):
    if not np.isfinite(value):
        return "black"
    threshold = 0.55 * float(max(vmax, 1e-6))
    return "white" if abs(float(value)) >= threshold else "black"


def _feature_stage_is_active_for_display(learner, local_stage_params, stage_idx, feat_idx):
    if getattr(learner, "feature_activation_mode", "fixed_mask") in {"score", "joint_mask_search"}:
        try:
            score = float(local_stage_params[stage_idx].feature_scores[feat_idx])
        except Exception:
            return False
        if not np.isfinite(score):
            return False
        return (_score_threshold(learner, feat_idx, stage_idx=stage_idx) - score) > 0.0
    try:
        return int(learner.r[stage_idx, feat_idx]) == 1
    except Exception:
        return False


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


def _is_press_slide_insert(env) -> bool:
    return getattr(env, "eval_tag", "") == "S4SlideInsert"


def _is_sphere_inspect(env) -> bool:
    return str(getattr(env, "eval_tag", "")).startswith("S5SphereInspect")


def _trajectory_figsize(learner, *, three_row=False):
    if _is_press_slide_insert(learner.env):
        return (11.0, 9.2) if three_row else (10.6, 6.8)
    return (9.2, 8.6) if three_row else PAPER_FIGSIZE


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
        ax.set_ylabel("z", fontsize=PAPER_LABEL_SIZE)
        ax.set_aspect("auto")
        try:
            ax.set_box_aspect(0.78)
        except Exception:
            pass
        return
    pad = 0.05 * max(dx, dy)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_ylabel("y", fontsize=PAPER_LABEL_SIZE)
    ax.set_aspect("equal", adjustable="box")


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


def _stage_colors(num_stages):
    num_stages = int(num_stages)
    return [STAGE_COLORS[i % len(STAGE_COLORS)] for i in range(max(num_stages, 1))]


def _draw_true_cutpoint_markers(ax, X, cutpoints, colors, *, label, size, zorder):
    for j, tt in enumerate(cutpoints):
        color = colors[j % len(colors)] if colors else "black"
        ax.scatter(
            X[int(tt), 0],
            X[int(tt), 1],
            marker="x",
            s=size,
            color=color,
            linewidths=1.4,
            label=label if j == 0 else "",
            zorder=zorder,
        )


def _set_axes_equal_3d_from_xyz(ax, xyz):
    pts = np.asarray(xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.size == 0:
        return
    mins = np.min(pts, axis=0)
    maxs = np.max(pts, axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    max_span = float(np.max(spans))
    centers = 0.5 * (mins + maxs)
    half = 0.55 * max_span
    ax.set_xlim(centers[0] - half, centers[0] + half)
    ax.set_ylim(centers[1] - half, centers[1] + half)
    ax.set_zlim(centers[2] - half, centers[2] + half)
    try:
        ax.set_box_aspect([1.0, 1.0, 1.0])
    except Exception:
        pass


def _projection_axis_labels(dims):
    labels = ["x", "y", "z"]
    return labels[int(dims[0])], labels[int(dims[1])]


def _draw_stage_background(ax, starts, ends, colors, *, alpha=0.07):
    for stage_idx, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(
            float(s),
            float(e),
            color=colors[stage_idx % len(colors)],
            alpha=alpha,
            lw=0.0,
            zorder=0,
        )


def _draw_sphere_reference_wireframe(ax, env):
    if not _is_sphere_inspect(env):
        return
    center = np.asarray(getattr(env, "sphere_center", np.zeros(3)), dtype=float).reshape(-1)
    radius = float(getattr(env, "sphere_radius", 1.0))
    theta = np.linspace(0.0, 2.0 * np.pi, 26)
    phi = np.linspace(0.0, np.pi, 16)
    th, ph = np.meshgrid(theta, phi)
    xx = center[0] + radius * np.cos(th) * np.sin(ph)
    yy = center[1] + radius * np.sin(th) * np.sin(ph)
    zz = center[2] + radius * np.cos(ph)
    ax.plot_wireframe(xx, yy, zz, color="#8c8c8c", alpha=0.18, linewidth=0.45, rstride=1, cstride=1)


def _draw_sphere_projection_circle(ax, env, dims):
    if not _is_sphere_inspect(env):
        return
    center = np.asarray(getattr(env, "sphere_center", np.zeros(3)), dtype=float).reshape(-1)
    radius = float(getattr(env, "sphere_radius", 1.0))
    circle = plt.Circle(
        (float(center[int(dims[0])]), float(center[int(dims[1])])),
        radius,
        fill=False,
        color="#8c8c8c",
        linestyle=(0, (3, 2)),
        linewidth=0.9,
        alpha=0.8,
    )
    ax.add_patch(circle)


def _draw_planar_obstacles(ax, env):
    if hasattr(env, "stage1_aux_obstacle_centers") and hasattr(env, "stage1_aux_obstacle_radii"):
        cx, cy = np.asarray(env.obs_center, dtype=float).reshape(-1)[:2]
        ax.add_patch(plt.Circle((cx, cy), float(env.obs_radius), color="gray", fill=False, linestyle="-", label="obstacle"))
        for idx, (center, radius) in enumerate(zip(np.asarray(env.stage1_aux_obstacle_centers, dtype=float), np.asarray(env.stage1_aux_obstacle_radii, dtype=float))):
            aux_x, aux_y = np.asarray(center, dtype=float).reshape(-1)[:2]
            ax.add_patch(
                plt.Circle(
                    (float(aux_x), float(aux_y)),
                    float(radius),
                    color="gray",
                    fill=False,
                    linestyle=(0, (3, 2)),
                    alpha=0.95,
                    label="obstacle" if idx == 0 and not ax.patches else None,
                )
            )
        return
    if hasattr(env, "obs_radius") and (hasattr(env, "obs_center") or hasattr(env, "obs_center_xy")):
        center = getattr(env, "obs_center", None)
        if center is None:
            center = getattr(env, "obs_center_xy")
        cx, cy = np.asarray(center, dtype=float).reshape(-1)[:2]
        r = env.obs_radius
        ax.add_patch(plt.Circle((cx, cy), r, color="gray", fill=False, linestyle="-", label="obstacle"))


def _draw_sphere_trajectory_3d(ax, learner, it, demo_idx=0):
    X = np.asarray(learner.demos[demo_idx], dtype=float)
    stage_ends = learner.stage_ends_[demo_idx]
    starts, ends = _segment_bounds(stage_ends)
    colors = _stage_colors(learner.num_stages)

    _draw_sphere_reference_wireframe(ax, learner.env)
    for stage_idx, (s, e) in enumerate(zip(starts, ends)):
        color = colors[stage_idx % len(colors)]
        pts = X[s : e + 1]
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=color,
            lw=1.5,
            alpha=0.95,
            label=f"stage {stage_idx + 1}" if stage_idx == 0 else "",
        )
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=color,
            s=8,
            alpha=0.32,
            depthshade=False,
        )

    true_cutpoints = _true_cutpoints_for_demo(learner, demo_idx)
    for cp_idx, cp in enumerate(true_cutpoints):
        cp_pt = X[int(cp)]
        ax.scatter(
            cp_pt[0],
            cp_pt[1],
            cp_pt[2],
            color=colors[cp_idx % len(colors)],
            marker="x",
            s=34,
            linewidths=1.4,
            depthshade=False,
            label="true boundary" if cp_idx == 0 else "",
            zorder=10,
        )

    for label_name, marker in [("subgoal", "X"), ("goal", "X")]:
        pt = getattr(learner.env, label_name, None)
        if pt is None:
            continue
        pt = np.asarray(pt, dtype=float).reshape(-1)
        if pt.size < 3:
            continue
        ax.scatter(
            pt[0],
            pt[1],
            pt[2],
            color="black",
            marker=marker,
            s=40,
            depthshade=False,
            label="true stage end",
            zorder=12,
        )

    center = np.asarray(getattr(learner.env, "sphere_center", np.zeros(3)), dtype=float).reshape(-1)
    radius = float(getattr(learner.env, "sphere_radius", 1.0))
    corners = np.array(
        [
            center + np.array([sx, sy, sz], dtype=float) * radius
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )
    _set_axes_equal_3d_from_xyz(ax, np.vstack([X, corners]))
    ax.view_init(elev=24, azim=38)
    ax.set_title(f"Iter {int(it)}: demo {demo_idx} 3D trajectory", fontsize=PAPER_TITLE_SIZE, pad=6)
    ax.set_xlabel("x", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("y", fontsize=PAPER_LABEL_SIZE)
    ax.set_zlabel("z", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax, outside=False)


def _draw_sphere_projection(ax, learner, it, demo_idx=0, dims=(0, 1)):
    X = np.asarray(learner.demos[demo_idx], dtype=float)
    stage_ends = learner.stage_ends_[demo_idx]
    starts, ends = _segment_bounds(stage_ends)
    colors = _stage_colors(learner.num_stages)
    lx, ly = _projection_axis_labels(dims)

    _draw_sphere_projection_circle(ax, learner.env, dims)
    for stage_idx, (s, e) in enumerate(zip(starts, ends)):
        color = colors[stage_idx % len(colors)]
        pts = X[s : e + 1]
        ax.plot(
            pts[:, int(dims[0])],
            pts[:, int(dims[1])],
            color=color,
            lw=1.5,
            alpha=0.95,
        )
        ax.scatter(
            pts[:, int(dims[0])],
            pts[:, int(dims[1])],
            color=color,
            s=10,
            alpha=0.35,
        )

    true_cutpoints = _true_cutpoints_for_demo(learner, demo_idx)
    for cp_idx, cp in enumerate(true_cutpoints):
        cp_pt = X[int(cp)]
        ax.scatter(
            cp_pt[int(dims[0])],
            cp_pt[int(dims[1])],
            color=colors[cp_idx % len(colors)],
            marker="x",
            s=30,
            linewidths=1.2,
            label="true boundary" if cp_idx == 0 else "",
            zorder=10,
        )

    for label_name in ("subgoal", "goal"):
        pt = getattr(learner.env, label_name, None)
        if pt is None:
            continue
        pt = np.asarray(pt, dtype=float).reshape(-1)
        if pt.size <= max(int(dims[0]), int(dims[1])):
            continue
        ax.scatter(
            pt[int(dims[0])],
            pt[int(dims[1])],
            color="black",
            marker="X",
            s=32,
            label="true stage end" if label_name == "subgoal" else "",
            zorder=12,
        )

    pts_2d = X[:, [int(dims[0]), int(dims[1])]]
    center = np.asarray(getattr(learner.env, "sphere_center", np.zeros(3)), dtype=float).reshape(-1)
    radius = float(getattr(learner.env, "sphere_radius", 1.0))
    ref_pts = np.array(
        [
            [center[int(dims[0])] - radius, center[int(dims[1])] - radius],
            [center[int(dims[0])] + radius, center[int(dims[1])] + radius],
        ],
        dtype=float,
    )
    _configure_2d_trajectory_axes(ax, learner.env, np.vstack([pts_2d, ref_pts]))
    ax.set_title(f"{lx}{ly} projection", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel(lx, fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel(ly, fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_sphere_feature_overview(ax, learner, demo_idx=0):
    F_raw = _raw_feature_matrix_for_demo(learner, demo_idx)
    if F_raw.ndim != 2 or F_raw.size == 0:
        ax.axis("off")
        return

    priority = [
        "surface_distance",
        "tool_normal_alignment_error",
        "speed",
        "angular_speed",
        "noise_aux",
    ]
    feature_order = []
    for name in priority:
        for feat_idx in range(min(int(getattr(learner, "num_features", 0)), F_raw.shape[1])):
            if _feature_name(learner, feat_idx) == name and feat_idx not in feature_order:
                feature_order.append(feat_idx)
    for feat_idx in range(min(int(getattr(learner, "num_features", 0)), F_raw.shape[1])):
        if feat_idx not in feature_order:
            feature_order.append(feat_idx)
    feature_order = feature_order[:5]

    t_axis = np.arange(F_raw.shape[0], dtype=int)
    stage_ends = learner.stage_ends_[demo_idx]
    starts, ends = _segment_bounds(stage_ends)
    stage_colors = _stage_colors(learner.num_stages)
    feature_colors = _feature_plot_colors(len(feature_order))

    _draw_stage_background(ax, starts, ends, stage_colors, alpha=0.06)
    for local_idx, feat_idx in enumerate(feature_order):
        feature_name = _feature_name(learner, feat_idx)
        color = feature_colors[local_idx % len(feature_colors)]
        values = np.asarray(F_raw[:, feat_idx], dtype=float)
        ax.plot(t_axis, values, color=color, lw=1.25, label=feature_name)
        for stage_idx, (s, e) in enumerate(zip(starts, ends)):
            ref_value = _reference_constraint_value(learner.env, feature_name, stage=stage_idx)
            if ref_value is None:
                continue
            ax.hlines(
                float(ref_value),
                float(s),
                float(e),
                color=color,
                linewidth=0.95,
                linestyle=(0, (3, 2)),
                alpha=0.85,
            )

    for cp_idx, cp in enumerate(stage_ends[:-1]):
        ax.axvline(int(cp), color="black", linestyle="--", lw=1.0, label="pred boundary" if cp_idx == 0 else "")
    for cp_idx, cp in enumerate(_true_cutpoints_for_demo(learner, demo_idx)):
        ax.axvline(int(cp), color="green", linestyle=":", lw=1.0, label="true boundary" if cp_idx == 0 else "")

    ax.set_title(f"Demo {demo_idx} key feature traces", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("time", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("raw value", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax, outside=False)


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
    abs_dev = np.abs(xs - center)
    keep = max(int(np.ceil(0.8 * abs_dev.size)), 1)
    trimmed_abs_dev = np.partition(abs_dev, keep - 1)[:keep]
    return float(np.mean(trimmed_abs_dev))


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
        density=True,
        alpha=alpha,
        color=color,
        label=label,
    )


def _draw_trajectories(ax, learner, it, demo_idx=0):
    X = np.asarray(learner.demos[demo_idx], dtype=float)
    stage_ends = learner.stage_ends_[demo_idx]
    starts, ends = _segment_bounds(stage_ends)
    colors = _stage_colors(learner.num_stages)
    is_press = _is_press_slide_insert(learner.env)
    traj_marker_size = 3.4 if is_press else 5.0
    goal_marker_size = 24 if is_press else 28
    cutpoint_marker_size = 22 if is_press else 30
    for k, (s, e) in enumerate(zip(starts, ends)):
        ax.scatter(X[s : e + 1, 0], X[s : e + 1, 1], color=colors[k], s=traj_marker_size, alpha=0.32 if is_press else 0.45)

    _draw_planar_obstacles(ax, learner.env)
    sg = _true_stage_end_point(learner.env, "subgoal", demo_idx=demo_idx)
    if sg is not None:
        sg = _xy_point(sg)
        ax.scatter(sg[0], sg[1], color="black", marker="X", s=goal_marker_size, label="true stage end")
    gg = _true_stage_end_point(learner.env, "goal", demo_idx=demo_idx)
    if gg is not None:
        gg = _xy_point(gg)
        ax.scatter(gg[0], gg[1], color="black", marker="X", s=goal_marker_size, label="true stage end")
    _draw_true_cutpoint_markers(
        ax,
        X,
        _true_cutpoints_for_demo(learner, demo_idx),
        colors,
        label="true boundary",
        size=cutpoint_marker_size,
        zorder=12,
    )

    ax.set_title(f"Iter {int(it)}: demo {demo_idx} trajectory", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("x", fontsize=PAPER_LABEL_SIZE)
    pts = [X[:, :2]]
    if sg is not None:
        pts.append(_xy_point(sg)[None, :])
    if gg is not None:
        pts.append(_xy_point(gg)[None, :])
    _configure_2d_trajectory_axes(ax, learner.env, np.concatenate(pts, axis=0))
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax, outside=is_press)


def _draw_trajectories_overview(ax, learner, it):
    colors = _stage_colors(learner.num_stages)
    is_press = _is_press_slide_insert(learner.env)
    traj_marker_size = 2.4 if is_press else 4.0
    goal_marker_size = 24 if is_press else 28
    cutpoint_marker_size = 20 if is_press else 26
    for i, X in enumerate(learner.demos):
        X = np.asarray(X, dtype=float)
        stage_ends = learner.stage_ends_[i]
        starts, ends = _segment_bounds(stage_ends)
        for k, (s, e) in enumerate(zip(starts, ends)):
            ax.scatter(X[s : e + 1, 0], X[s : e + 1, 1], color=colors[k], s=traj_marker_size, alpha=0.28 if is_press else 0.35)

    _draw_planar_obstacles(ax, learner.env)
    for pt_idx, pt in enumerate(_all_true_stage_end_points(learner.env, "subgoal")):
        xy = _xy_point(pt)
        ax.scatter(xy[0], xy[1], color="black", marker="X", s=goal_marker_size, alpha=0.55, label="true stage end" if pt_idx == 0 else "")
    goal_offset = len(_all_true_stage_end_points(learner.env, "subgoal"))
    for pt_idx, pt in enumerate(_all_true_stage_end_points(learner.env, "goal")):
        xy = _xy_point(pt)
        ax.scatter(xy[0], xy[1], color="black", marker="X", s=goal_marker_size, alpha=0.55, label="true stage end" if (goal_offset == 0 and pt_idx == 0) else "")
    for i, X in enumerate(learner.demos):
        _draw_true_cutpoint_markers(
            ax,
            np.asarray(X, dtype=float),
            _true_cutpoints_for_demo(learner, i),
            colors,
            label="true boundary" if i == 0 else "",
            size=cutpoint_marker_size,
            zorder=12,
        )

    ax.set_title(f"Iter {int(it)}: trajectories & SWCL goals", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("x", fontsize=PAPER_LABEL_SIZE)
    pts = [np.asarray(X, dtype=float)[:, :2] for X in learner.demos]
    for pt in _all_true_stage_end_points(learner.env, "subgoal"):
        pts.append(_xy_point(pt)[None, :])
    for pt in _all_true_stage_end_points(learner.env, "goal"):
        pts.append(_xy_point(pt)[None, :])
    _configure_2d_trajectory_axes(ax, learner.env, np.concatenate(pts, axis=0))
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax, outside=is_press)


def _draw_learning_curves(ax, learner):
    iters = np.arange(len(learner.loss_total))
    weighted_constraint = np.asarray(learner.loss_constraint, dtype=float)
    weighted_short_segment_penalty = np.asarray(getattr(learner, "loss_short_segment_penalty", []), dtype=float)
    weighted_progress = learner.lambda_progress * np.asarray(learner.loss_progress, dtype=float)
    weighted_subgoal_consensus = (
        np.asarray(learner.subgoal_consensus_lambda_hist[: len(learner.loss_subgoal_consensus)], dtype=float)
        * np.asarray(learner.loss_subgoal_consensus, dtype=float)
    )
    weighted_param_consensus = (
        np.asarray(learner.param_consensus_lambda_hist[: len(learner.loss_param_consensus)], dtype=float)
        * np.asarray(learner.loss_param_consensus, dtype=float)
    )
    weighted_activation_consensus = (
        np.asarray(getattr(learner, "activation_consensus_lambda_hist", learner.feature_score_consensus_lambda_hist)[: len(getattr(learner, "loss_activation_consensus", learner.loss_feature_score_consensus))], dtype=float)
        * np.asarray(getattr(learner, "loss_activation_consensus", learner.loss_feature_score_consensus), dtype=float)
    )
    ax.plot(iters, learner.loss_total, color="black", lw=1.3, label="total")
    ax.plot(iters, weighted_constraint, color="tab:red", lw=1.0, label="constraint")
    if weighted_short_segment_penalty.size == iters.size:
        ax.plot(iters, weighted_short_segment_penalty, color="tab:olive", lw=1.0, label="short_segment_penalty")
    ax.plot(iters, weighted_progress, color="tab:orange", lw=1.0, label="progress")
    ax.plot(iters, weighted_subgoal_consensus, color="tab:purple", lw=1.0, label="subgoal_consensus")
    ax.plot(iters, weighted_param_consensus, color="tab:cyan", lw=1.0, label="param_consensus")
    ax.plot(iters, weighted_activation_consensus, color="tab:brown", lw=1.0, label="activation_consensus")
    ax.set_title("SWCL learning curves", fontsize=PAPER_TITLE_SIZE, pad=4)
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
    threshold_mat = np.zeros_like(raw_score_matrix, dtype=float)
    for feat_idx in range(raw_score_matrix.shape[0]):
        for stage_idx in range(raw_score_matrix.shape[1]):
            threshold_mat[feat_idx, stage_idx] = float(_score_threshold(learner, feat_idx, stage_idx=stage_idx))
    score_margin_matrix = threshold_mat - raw_score_matrix
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


def _draw_avg_constraint_cost_matrix(ax, learner, demo_idx=0):
    if not getattr(learner, "current_stage_params_per_demo", None):
        ax.axis("off")
        return
    stage_params_list = learner.current_stage_params_per_demo[demo_idx]
    if not stage_params_list:
        ax.axis("off")
        return
    matrix = np.zeros((learner.num_features, learner.num_stages), dtype=float)
    for stage_idx in range(learner.num_stages):
        stage_params = stage_params_list[stage_idx]
        scores = np.asarray(stage_params.feature_scores, dtype=float)
        stage_len = max(int(learner.stage_ends_[demo_idx][stage_idx] - (-1 if stage_idx == 0 else learner.stage_ends_[demo_idx][stage_idx - 1])), 1)
        for feat_idx in range(learner.num_features):
            thr = float(_score_threshold(learner, feat_idx, stage_idx=stage_idx))
            margin = thr - float(scores[feat_idx])
            matrix[feat_idx, stage_idx] = float(margin) / float(stage_len)
    vmax = float(np.nanmax(np.abs(matrix))) if matrix.size > 0 else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    stage_labels = [f"s{i + 1}" for i in range(matrix.shape[1])]
    feature_labels = [_feature_name(learner, i) for i in range(matrix.shape[0])]
    ax.set_title(f"Demo {demo_idx} margin / stage length", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(stage_labels)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(feature_labels)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                color=_matrix_text_color(value, vmax),
                fontsize=8,
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _draw_feature_bands(ax, learner, demo_idx=0, standardized=False):
    if not getattr(learner, "current_stage_params_per_demo", None):
        ax.axis("off")
        return
    X0 = learner.demos[demo_idx]
    T0 = len(X0)
    t_axis = np.arange(T0)
    Fz = learner.standardized_features[demo_idx]
    values = (
        Fz
        if standardized
        else (
            Fz * learner.feat_std[learner.selected_feature_columns][None, :]
            + learner.feat_mean[learner.selected_feature_columns][None, :]
        )
    )
    max_features = min(learner.num_features, 6)
    colors = _feature_plot_colors(max_features)
    stage_ends = learner.stage_ends_[demo_idx]
    starts, ends = _segment_bounds(stage_ends)
    local_stage_params = learner.current_stage_params_per_demo[demo_idx]
    for m in range(max_features):
        label = _feature_name(learner, m)
        ax.plot(t_axis, values[:, m], lw=1.0, color=colors[m], label=label)
        for k in range(learner.num_stages):
            if not _feature_stage_is_active_for_display(learner, local_stage_params, k, m):
                continue
            summary = local_stage_params[k].model_summaries[m]
            fid = learner.selected_feature_columns[m]
            s = starts[k]
            e = ends[k]
            core_s, core_e = _core_bounds_for_display(learner, s, e)
            stage_vals = np.asarray(values[core_s : core_e + 1, m], dtype=float)
            t_seg = t_axis[core_s : core_e + 1]
            if stage_vals.size == 0 or t_seg.size == 0:
                continue
            kind = _stage_feature_kind_for_display(learner, local_stage_params, k, m)
            low, high = _summary_interval_display(learner, m, summary, standardized)
            if not np.isfinite(low) or not np.isfinite(high):
                if learner._is_equality_feature(m):
                    stage_median = float(np.median(stage_vals))
                    stage_std = max(float(np.std(stage_vals)), 0.05 if not standardized else 0.02)
                    low = stage_median - stage_std
                    high = stage_median + stage_std
                else:
                    continue
            ax.fill_between(
                t_seg,
                low,
                high,
                color=colors[m],
                alpha=0.18 if k == 0 else 0.14,
                zorder=1,
            )
            ax.plot(t_seg, np.full(t_seg.size, low), color=colors[m], lw=0.8, alpha=0.9, zorder=2)
            ax.plot(t_seg, np.full(t_seg.size, high), color=colors[m], lw=0.8, alpha=0.9, zorder=2)
            center = _summary_center_display(learner, m, summary, kind, standardized)
            if np.isfinite(center):
                ax.plot(t_seg, np.full(t_seg.size, center), color=colors[m], lw=0.9, alpha=0.8, linestyle="--", zorder=2)
    for j, cp in enumerate(learner.stage_ends_[demo_idx][:-1]):
        ax.axvline(int(cp), color="black", linestyle="--", lw=1.0, label="pred boundary" if j == 0 else "")
    for j, cp in enumerate(_true_cutpoints_for_demo(learner, demo_idx)):
        ax.axvline(int(cp), color="green", linestyle=":", lw=1.0, label="true boundary" if j == 0 else "")
    ax.set_title(
        f"Demo {demo_idx} local feature constraints ({'standardized' if standardized else 'raw'})",
        fontsize=PAPER_TITLE_SIZE,
        pad=4,
    )
    ax.set_xlabel("time", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("standardized feature value" if standardized else "raw feature value", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_activation_rate_history(ax, learner):
    history = getattr(learner, "activation_rate_history", None) or []
    if not history:
        ax.axis("off")
        return
    hist = np.asarray(history, dtype=float)
    if hist.ndim != 3 or hist.shape[1] == 0 or hist.shape[2] == 0:
        ax.axis("off")
        return

    num_iters, num_stages, num_features = hist.shape
    row_labels = []
    rows = []
    for stage_idx in range(num_stages):
        for feat_idx in range(num_features):
            row_labels.append(f"s{stage_idx + 1}:{_feature_name(learner, feat_idx)}")
            rows.append(hist[:, stage_idx, feat_idx])
    matrix = np.asarray(rows, dtype=float)

    im = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title("activation rate history", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("iteration", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("stage:feature", fontsize=PAPER_LABEL_SIZE)
    ax.set_xticks(range(num_iters))
    ax.set_xticklabels([str(i + 1) for i in range(num_iters)])
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    _style_paper_axis(ax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = float(matrix[i, j])
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=_matrix_text_color(value, 1.0),
                fontsize=6.8,
            )
    _add_slim_colorbar(im, ax)


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
    _style_paper_axis(ax)
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
                color=_matrix_text_color(float(value) if np.isfinite(value) else 0.0, scale),
                fontsize=6.8,
            )
    _add_slim_colorbar(im, ax)


def _draw_final_activation_rate_matrix(ax, learner):
    summary = getattr(learner, "posthoc_activation_summary_", None) or {}
    matrix = np.asarray(summary.get("activation_rate_matrix", []), dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        ax.axis("off")
        return
    _draw_summary_heatmap(
        ax,
        matrix.T,
        "activation rate",
        feature_names=_summary_feature_names(learner),
        stage_labels=_summary_stage_labels(learner),
        cmap="Blues",
        fmt=".2f",
        vmin=0.0,
        vmax=1.0,
    )


def _draw_final_activation_proto_matrix(ax, learner):
    matrix = np.asarray(getattr(learner, "shared_activation_proto", []), dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        ax.axis("off")
        return
    _draw_summary_heatmap(
        ax,
        matrix.T,
        "activation proto",
        feature_names=_summary_feature_names(learner),
        stage_labels=_summary_stage_labels(learner),
        cmap="Greys",
        fmt=".0f" if np.all(np.isclose(matrix[np.isfinite(matrix)], np.rint(matrix[np.isfinite(matrix)]))) else ".2f",
        vmin=0.0,
        vmax=1.0,
    )


def _draw_eval_metric_text(ax, metrics):
    ax.axis("off")
    if not isinstance(metrics, dict):
        ax.text(0.02, 0.98, "No evaluation metrics.", ha="left", va="top", fontsize=8, family="monospace", transform=ax.transAxes)
        return
    preferred_keys = [
        "MeanAbsCutpointError",
        "MeanStageSubgoalError",
        "MeanConstraintError",
    ]
    scalar_metrics = {}
    for key, value in metrics.items():
        if np.isscalar(value):
            value_f = float(value)
            if np.isfinite(value_f):
                scalar_metrics[str(key)] = value_f
    ordered_keys = [key for key in preferred_keys if key in scalar_metrics]
    ordered_keys += [key for key in sorted(scalar_metrics.keys()) if key not in ordered_keys]
    ax.set_title("evaluation metrics", fontsize=PAPER_TITLE_SIZE, pad=4)
    if not ordered_keys:
        ax.text(0.02, 0.98, "No scalar metrics.", ha="left", va="top", fontsize=8, transform=ax.transAxes)
        return
    y = 0.92
    for idx, key in enumerate(ordered_keys):
        value = scalar_metrics[key]
        ax.text(
            0.03,
            y,
            key,
            ha="left",
            va="top",
            fontsize=7.5,
            color="#444444",
            transform=ax.transAxes,
        )
        ax.text(
            0.97,
            y,
            f"{value:.4f}",
            ha="right",
            va="top",
            fontsize=9.5 if idx < 3 else 8.0,
            color="black",
            transform=ax.transAxes,
        )
        y -= 0.18 if idx < 3 else 0.14
    ax.plot([0.03, 0.97], [0.975, 0.975], color="#333333", lw=0.8, transform=ax.transAxes, clip_on=False)


def _draw_constraint_error_matrix(ax, learner, metrics):
    if not isinstance(metrics, dict):
        ax.axis("off")
        return
    matrix = np.asarray(metrics.get("ConstraintErrorMatrix", []), dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        ax.axis("off")
        return
    _draw_summary_heatmap(
        ax,
        matrix.T,
        "normalized constraint error",
        feature_names=_summary_feature_names(learner, metrics),
        stage_labels=_summary_stage_labels(learner),
        cmap="YlOrRd",
        fmt=".3f",
    )


def _draw_cost_profile(ax, learner, demo_idx=0):
    if learner.num_stages != 2:
        if not getattr(learner, "current_stage_params_per_demo", None):
            ax.axis("off")
            return
        stage_labels = [f"s{k + 1}" for k in range(learner.num_stages)]
        stage_ends = learner.stage_ends_[demo_idx]
        starts, ends = _segment_bounds(stage_ends)
        seg_lengths = np.asarray([e - s + 1 for s, e in zip(starts, ends)], dtype=float)
        stage_constraint = []
        stage_short_segment_penalty = []
        stage_progress = []
        for stage_idx, (s, e) in enumerate(zip(starts, ends)):
            info = learner._segment_stage_cost_info(
                demo_idx=demo_idx,
                stage_idx=stage_idx,
                s=s,
                e=e,
                lam_subgoal_consensus=0.0,
                lam_param_consensus=0.0,
                lam_activation_consensus=0.0,
                shared_stage_subgoals=[np.zeros_like(np.asarray(learner.stage_subgoals[k], dtype=float)) for k in range(learner.num_stages)],
                shared_param_vectors=[[None for _ in range(learner.num_features)] for _ in range(learner.num_stages)],
                shared_r_mean=None,
                shared_feature_score_mean=None,
            )
            if info is None:
                stage_constraint.append(np.nan)
                stage_short_segment_penalty.append(np.nan)
                stage_progress.append(np.nan)
            else:
                stage_constraint.append(float(info["constraint"]))
                stage_short_segment_penalty.append(float(info.get("short_segment_penalty", 0.0)))
                stage_progress.append(learner.lambda_progress * float(info["progress"]))
        x = np.arange(learner.num_stages)
        ax.bar(x - 0.30, seg_lengths, width=0.18, color="tab:gray", label="segment length")
        ax.bar(x - 0.10, stage_constraint, width=0.18, color="tab:red", label="constraint")
        ax.bar(x + 0.10, stage_short_segment_penalty, width=0.18, color="tab:olive", label="short_segment_penalty")
        ax.bar(x + 0.30, stage_progress, width=0.18, color="tab:orange", label="progress")
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
    short_segment_penalty = []
    progress = []
    subgoal_consensus = []
    param_consensus = []
    activation_consensus = []
    for tau in candidate_taus:
        lam_subgoal_consensus = float(getattr(learner, "current_subgoal_consensus_lambda", 0.0))
        lam_param_consensus = float(getattr(learner, "current_param_consensus_lambda", 0.0))
        lam_activation_consensus = float(
            getattr(learner, "current_activation_consensus_lambda", getattr(learner, "current_feature_score_consensus_lambda", 0.0))
        )
        info = learner._candidate_cost(
            demo_idx=demo_idx,
            stage_ends=[int(tau), int(T - 1)],
            lam_subgoal_consensus=lam_subgoal_consensus,
            lam_param_consensus=lam_param_consensus,
            lam_activation_consensus=lam_activation_consensus,
            shared_stage_subgoals=learner.shared_stage_subgoals,
            shared_param_vectors=learner.shared_param_vectors,
            shared_r_mean=getattr(learner, "shared_r_mean", None),
            shared_feature_score_mean=getattr(learner, "shared_feature_score_mean", None),
        )
        if info is None:
            total.append(np.nan)
            constraint.append(np.nan)
            short_segment_penalty.append(np.nan)
            progress.append(np.nan)
            subgoal_consensus.append(np.nan)
            param_consensus.append(np.nan)
            activation_consensus.append(np.nan)
            continue
        constraint.append(float(info["constraint"]))
        short_segment_penalty.append(float(info.get("short_segment_penalty", 0.0)))
        progress.append(learner.lambda_progress * float(info["progress"]))
        subgoal_consensus.append(lam_subgoal_consensus * float(info["subgoal_consensus"]))
        param_consensus.append(lam_param_consensus * float(info["param_consensus"]))
        activation_consensus.append(lam_activation_consensus * float(info.get("activation_consensus", info.get("feature_score_consensus", 0.0))))
        total.append(float(info["total"]))

    ax.plot(candidate_taus, total, color="black", lw=1.4, label="total")
    ax.plot(candidate_taus, constraint, color="tab:red", lw=1.0, label="constraint")
    if np.any(np.isfinite(np.asarray(short_segment_penalty, dtype=float))):
        ax.plot(candidate_taus, short_segment_penalty, color="tab:olive", lw=1.0, label="short_segment_penalty")
    ax.plot(candidate_taus, progress, color="tab:orange", lw=1.0, label="progress")
    if np.any(np.isfinite(np.asarray(subgoal_consensus, dtype=float))):
        ax.plot(candidate_taus, subgoal_consensus, color="tab:purple", lw=1.0, label="subgoal_consensus")
    if np.any(np.isfinite(np.asarray(param_consensus, dtype=float))):
        ax.plot(candidate_taus, param_consensus, color="tab:cyan", lw=1.0, label="param_consensus")
    if np.any(np.isfinite(np.asarray(activation_consensus, dtype=float))):
        ax.plot(candidate_taus, activation_consensus, color="tab:brown", lw=1.0, label="activation_consensus")

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


def _draw_cutpoint_evolution(ax, learner):
    history = [list(item) for item in getattr(learner, "segmentation_history", [])]
    if not history:
        ax.axis("off")
        return
    current_stage_ends = [list(map(int, ends)) for ends in getattr(learner, "stage_ends_", [])]
    if current_stage_ends and (not history or history[-1] != current_stage_ends):
        history = history + [current_stage_ends]

    num_iters = len(history)
    num_demos = len(history[0]) if history else 0
    num_cutpoints = max(int(getattr(learner, "num_stages", 1)) - 1, 0)
    if num_demos == 0 or num_cutpoints == 0:
        ax.axis("off")
        return

    x = np.arange(num_iters, dtype=int)
    demo_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(num_demos, 3)))
    cut_styles = ["-", "--", "-.", ":"]
    true_cutpoints = [_true_cutpoints_for_demo(learner, demo_idx) for demo_idx in range(num_demos)]

    for demo_idx in range(num_demos):
        color = demo_colors[demo_idx % len(demo_colors)]
        for cp_idx in range(num_cutpoints):
            style = cut_styles[cp_idx % len(cut_styles)]
            values = np.asarray([int(history[it_idx][demo_idx][cp_idx]) for it_idx in range(num_iters)], dtype=float)
            ax.plot(
                x,
                values,
                color=color,
                linestyle=style,
                alpha=0.9,
                lw=1.2,
                marker="o",
                ms=2.2,
                label=f"cp{cp_idx + 1}" if demo_idx == 0 else "",
            )
            if cp_idx < len(true_cutpoints[demo_idx]):
                ax.axhline(
                    float(true_cutpoints[demo_idx][cp_idx]),
                    color=color,
                    linestyle=style,
                    alpha=0.22,
                    lw=0.9,
                    label="_nolegend_",
                )

    ax.set_title("Cutpoint evolution across training", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("iteration", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("cutpoint index", fontsize=PAPER_LABEL_SIZE)
    ax.set_xticks(x)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    ax.text(
        0.02,
        0.98,
        "color = demo\nlinestyle = cutpoint\nfaint horizontal = true cutpoint",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.82, edgecolor="0.7"),
    )
    _legend(ax)


def _current_consensus_lambdas(learner):
    return (
        float(getattr(learner, "current_subgoal_consensus_lambda", 0.0)),
        float(getattr(learner, "current_param_consensus_lambda", 0.0)),
        float(getattr(learner, "current_activation_consensus_lambda", getattr(learner, "current_feature_score_consensus_lambda", 0.0))),
    )


def _scan_cutpoint_range(learner, T, fixed_cutpoints, vary_index):
    fixed_cutpoints = [int(x) for x in fixed_cutpoints]
    num_cutpoints = int(learner.num_stages) - 1
    if vary_index < 0 or vary_index >= num_cutpoints:
        return np.asarray([], dtype=int)

    duration_min = np.asarray(learner.duration_min, dtype=int)
    duration_max = np.asarray(learner.duration_max, dtype=int)
    prev_end = -1 if vary_index == 0 else int(fixed_cutpoints[vary_index - 1])
    next_end = int(T - 1) if vary_index == num_cutpoints - 1 else int(fixed_cutpoints[vary_index + 1])

    low = max(
        int(prev_end + duration_min[vary_index]),
        int(next_end - duration_max[vary_index + 1]),
    )
    high = min(
        int(prev_end + duration_max[vary_index]),
        int(next_end - duration_min[vary_index + 1]),
    )
    if high < low:
        return np.asarray([], dtype=int)
    return np.arange(low, high + 1, dtype=int)


def _local_stage_cost_breakdown(learner, demo_idx, stage_ends):
    starts, ends = _segment_bounds(stage_ends)
    feature_constraint = np.zeros((learner.num_features, learner.num_stages), dtype=float)
    progress = np.zeros(learner.num_stages, dtype=float)
    for stage_idx, (s, e) in enumerate(zip(starts, ends)):
        stage_params, _, progress_cost = learner._fit_segment_stage(demo_idx, stage_idx, s, e)
        contrib = np.asarray(getattr(stage_params, "feature_constraint_costs", np.zeros(learner.num_features)), dtype=float)
        feature_constraint[:, stage_idx] = contrib
        progress[stage_idx] = learner.lambda_progress * float(progress_cost)
    return feature_constraint, progress


def _short_segment_penalty_for_stage(learner, stage_len):
    if not getattr(learner, "use_score_mode", False):
        return 0.0
    if not any(learner._is_equality_feature(feat_idx) for feat_idx in range(learner.num_features)):
        return 0.0
    return float(getattr(learner, "short_segment_penalty_c", 0.1) / np.sqrt(max(int(stage_len), 1)))


def _draw_learned_vs_true_local_cost_delta(fig, axes, learner, demo_idx=0):
    true_cutpoints = _true_cutpoints_for_demo(learner, demo_idx)
    if len(true_cutpoints) != learner.num_stages - 1:
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
    ax1.set_xticks(range(learner.num_stages))
    ax1.set_xticklabels([f"s{k + 1}" for k in range(learner.num_stages)])
    ax1.set_yticks(range(learner.num_features))
    ax1.set_yticklabels([_feature_name(learner, i) for i in range(learner.num_features)])
    ax1.tick_params(labelsize=PAPER_TICK_SIZE)
    for i in range(delta_constraint.shape[0]):
        for j in range(delta_constraint.shape[1]):
            value = float(delta_constraint[i, j])
            ax1.text(j, i, f"{value:.2f}", ha="center", va="center", color=_matrix_text_color(value, vmax), fontsize=8)
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    x = np.arange(learner.num_stages)
    bar_colors = [_stage_colors(learner.num_stages)[k] for k in range(learner.num_stages)]
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.bar(x, delta_progress, color=bar_colors, width=0.6)
    for k, value in enumerate(delta_progress):
        ax2.text(k, float(value), f"{float(value):.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    ax2.set_title(f"Demo {demo_idx} learned-true progress cost", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"s{k + 1}" for k in range(learner.num_stages)])
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
    if plt is None or learner.num_stages < 3:
        return
    true_cutpoints = _true_cutpoints_for_demo(learner, demo_idx)
    n_cutpoints = learner.num_stages - 1
    if len(true_cutpoints) != n_cutpoints or vary_index < 0 or vary_index >= n_cutpoints:
        return

    learned_stage_ends = [int(x) for x in learner.stage_ends_[demo_idx]]
    learned_cutpoints = [int(x) for x in learned_stage_ends[:-1]]
    true_cutpoints = [int(x) for x in true_cutpoints]
    T = len(learner.demos[demo_idx])

    learned_candidate = list(learned_cutpoints) + [T - 1]
    true_candidate = list(learned_cutpoints)
    true_candidate[vary_index] = int(true_cutpoints[vary_index])
    true_candidate = true_candidate + [T - 1]
    compare_stage_ends = [
        (f"learned cp{vary_index + 1}", learned_candidate),
        (f"true cp{vary_index + 1}", true_candidate),
    ]
    stage_indices = [vary_index, vary_index + 1]
    filename = f"plot_cp{vary_index + 1}_compare_demo_{int(demo_idx):02d}_iter_{int(it):04d}.png"
    fixed_parts = [
        f"cp{k + 1}={learned_cutpoints[k]}"
        for k in range(n_cutpoints)
        if k != vary_index
    ]
    title = (
        f"Demo {demo_idx} | fixed learned {', '.join(fixed_parts)} | "
        f"learned vs true cp{vary_index + 1}"
    )

    valid_rows = []
    for label, stage_ends in compare_stage_ends:
        stage_ends = [int(x) for x in stage_ends]
        cutpoints_only = stage_ends[:-1]
        if len(cutpoints_only) != n_cutpoints:
            continue
        if any(cp < 0 or cp >= T - 1 for cp in cutpoints_only):
            continue
        if any(cutpoints_only[k] >= cutpoints_only[k + 1] for k in range(len(cutpoints_only) - 1)):
            continue
        if int(stage_ends[-1]) != T - 1:
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
            core_s, core_e = _core_bounds_for_display(learner, s, e)
            vals_by_feat = np.asarray(F[core_s : core_e + 1], dtype=float)
            stage_params = stage_params_list[stage_idx]
            for feat_idx, _ in enumerate(learner.feature_model_types):
                ax = axes[row_idx][feat_idx]
                vals = np.asarray(vals_by_feat[:, feat_idx], dtype=float)
                full_vals = np.asarray(F[:, feat_idx], dtype=float)
                if vals.size == 0:
                    ax.axis("off")
                    continue
                kind = _stage_feature_kind_for_display(learner, stage_params_list, stage_idx, feat_idx)
                is_equality_feature = _kind_is_equality_display(kind)
                show_full_demo = is_equality_feature
                is_dispersion_equality = (
                    is_equality_feature
                    and getattr(learner, "equality_score_mode", "dispersion") == "dispersion"
                )
                fitted_model = None
                baseline_model = None
                if not is_dispersion_equality:
                    summary = stage_params.model_summaries[feat_idx]
                    fitted_model = learner._vector_to_model(kind, learner._summary_to_vector(kind, summary))
                    if is_equality_feature:
                        baseline_model = GaussianModel(
                            mu=float(np.mean(full_vals)),
                            sigma=float(max(np.std(full_vals), 1e-6)),
                        )
                    else:
                        baseline_model = learner._fit_student_t_baseline(vals)

                if show_full_demo:
                    lo_candidates = [np.min(vals), np.min(full_vals)]
                    hi_candidates = [np.max(vals), np.max(full_vals)]
                    if fitted_model is not None:
                        lo_candidates.append(getattr(fitted_model, "L", np.min(vals)))
                        hi_candidates.append(getattr(fitted_model, "U", np.max(vals)))
                    if baseline_model is not None:
                        lo_candidates.append(getattr(baseline_model, "L", np.min(vals)))
                        hi_candidates.append(getattr(baseline_model, "U", np.max(vals)))
                    lo = float(min(lo_candidates))
                    hi = float(max(hi_candidates))
                    pad = max(0.15 * (hi - lo + 1e-6), 0.2)
                else:
                    q_lo, q_hi = np.quantile(vals, [0.02, 0.98])
                    center = float(np.median(vals))
                    span = max(float(q_hi - q_lo), 0.03)
                    lo = min(float(np.min(vals)), center - 1.25 * span)
                    hi = max(float(np.max(vals)), center + 1.25 * span)
                    pad = max(0.08 * (hi - lo + 1e-6), 0.01)
                xs = np.linspace(lo - pad, hi + pad, 300)

                if show_full_demo:
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
                if fitted_model is not None:
                    ax.plot(xs, np.exp(fitted_model.logpdf(xs)), color="tab:red", lw=2.0, label="fitted")
                if baseline_model is not None:
                    ax.plot(
                        xs,
                        np.exp(baseline_model.logpdf(xs)),
                        color="tab:green",
                        lw=2.0,
                        linestyle="--",
                        label="baseline",
                    )
                ax.axvline(float(np.mean(vals)), color="tab:blue", lw=1.0, linestyle=":", alpha=0.8)
                if show_full_demo:
                    ax.axvline(float(np.mean(full_vals)), color="tab:gray", lw=1.0, linestyle="-.", alpha=0.9)

                raw_score = float(stage_params.feature_scores[feat_idx])
                threshold = float(_score_threshold(learner, feat_idx, stage_idx=stage_idx))
                score_margin = threshold - raw_score
                weighted_cost = float(np.asarray(stage_params.feature_constraint_costs, dtype=float)[feat_idx])
                info_lines = [
                    f"core steps = {len(vals)}",
                    f"segment = [{s},{e}]",
                    f"core = [{core_s},{core_e}]",
                    f"raw score = {raw_score:.3f}",
                    f"margin = {score_margin:.3f}",
                    f"weighted cost = {weighted_cost:.3f}",
                ]
                if fitted_model is not None and baseline_model is not None:
                    fitted_ll = np.asarray(fitted_model.logpdf(vals), dtype=float)
                    baseline_ll = np.asarray(baseline_model.logpdf(vals), dtype=float)
                    fitted_step = float(np.mean(fitted_ll))
                    baseline_step = float(np.mean(baseline_ll))
                    info_lines[3:3] = [
                        f"fitted avg LL = {fitted_step:.3f}",
                        f"baseline avg LL = {baseline_step:.3f}",
                        f"avg LL gain = {fitted_step - baseline_step:.3f}",
                    ]
                if is_equality_feature:
                    short_segment_penalty = _short_segment_penalty_for_stage(learner, len(vals))
                    if getattr(learner, "equality_score_mode", "dispersion") == "gaussian_ll_gain":
                        global_baseline_model = GaussianModel(
                            mu=float(np.mean(full_vals)),
                            sigma=float(max(np.std(full_vals), 1e-6)),
                        )
                        global_baseline_ll = np.asarray(global_baseline_model.logpdf(vals), dtype=float)
                        global_baseline_step = float(np.mean(global_baseline_ll))
                        info_lines.append(f"global gaussian avg LL = {global_baseline_step:.3f}")
                        info_lines.append(f"gaussian ll gain = {fitted_step - global_baseline_step:.3f}")
                        info_lines.append(f"short seg penalty = {short_segment_penalty:.3f}")
                    else:
                        stage_dispersion = float(_mean_abs_centered_dispersion(vals))
                        info_lines.append(f"local dispersion = {stage_dispersion:.3f}")
                        info_lines.append(f"short seg penalty = {short_segment_penalty:.3f}")
                    info_lines.append(f"threshold = {threshold:.3f}")
                else:
                    info_lines.append(f"threshold = {threshold:.3f}")

                ax.set_title(f"{scenario_label} | stage {stage_idx + 1} | {feature_names[feat_idx]}", fontsize=PAPER_TITLE_SIZE, pad=4)
                ax.set_xlabel("standardized feature value", fontsize=PAPER_LABEL_SIZE)
                ax.set_ylabel("density", fontsize=PAPER_LABEL_SIZE)
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


def _draw_single_cut_scan(ax, learner, demo_idx=0, vary_index=0, show_components=True):
    if learner.num_stages < 3:
        ax.axis("off")
        return

    stage_ends = learner.stage_ends_[demo_idx]
    learned_cutpoints = [int(x) for x in stage_ends[:-1]]
    num_cutpoints = len(learned_cutpoints)
    if vary_index < 0 or vary_index >= num_cutpoints:
        ax.axis("off")
        return
    T = len(learner.demos[demo_idx])
    candidate_values = _scan_cutpoint_range(learner, T, learned_cutpoints, vary_index)
    if candidate_values.size == 0:
        ax.text(0.5, 0.5, "No feasible cutpoints under duration bounds.", ha="center", va="center")
        ax.axis("off")
        return

    lam_subgoal_consensus, lam_param_consensus, lam_activation_consensus = _current_consensus_lambdas(learner)
    total = []
    constraint = []
    short_segment_penalty = []
    progress = []
    subgoal_consensus = []
    param_consensus = []
    activation_consensus = []
    feature_constraint_by_feat = [[] for _ in range(learner.num_features)]

    for value in candidate_values:
        candidate_cutpoints = list(learned_cutpoints)
        candidate_cutpoints[vary_index] = int(value)
        if any(candidate_cutpoints[k] >= candidate_cutpoints[k + 1] for k in range(len(candidate_cutpoints) - 1)):
            total.append(np.nan)
            constraint.append(np.nan)
            short_segment_penalty.append(np.nan)
            progress.append(np.nan)
            subgoal_consensus.append(np.nan)
            param_consensus.append(np.nan)
            activation_consensus.append(np.nan)
            for feat_values in feature_constraint_by_feat:
                feat_values.append(np.nan)
            continue

        info = learner._candidate_cost(
            demo_idx=demo_idx,
            stage_ends=[int(x) for x in candidate_cutpoints] + [int(T - 1)],
            lam_subgoal_consensus=lam_subgoal_consensus,
            lam_param_consensus=lam_param_consensus,
            lam_activation_consensus=lam_activation_consensus,
            shared_stage_subgoals=learner.shared_stage_subgoals,
            shared_param_vectors=learner.shared_param_vectors,
            shared_r_mean=getattr(learner, "shared_r_mean", None),
            shared_feature_score_mean=getattr(learner, "shared_feature_score_mean", None),
        )
        if info is None:
            total.append(np.nan)
            constraint.append(np.nan)
            short_segment_penalty.append(np.nan)
            progress.append(np.nan)
            subgoal_consensus.append(np.nan)
            param_consensus.append(np.nan)
            activation_consensus.append(np.nan)
            for feat_values in feature_constraint_by_feat:
                feat_values.append(np.nan)
            continue

        total.append(float(info["total"]))
        constraint.append(float(info["constraint"]))
        short_segment_penalty.append(float(info.get("short_segment_penalty", 0.0)))
        progress.append(learner.lambda_progress * float(info["progress"]))
        subgoal_consensus.append(lam_subgoal_consensus * float(info["subgoal_consensus"]))
        param_consensus.append(lam_param_consensus * float(info["param_consensus"]))
        activation_consensus.append(lam_activation_consensus * float(info.get("activation_consensus", info.get("feature_score_consensus", 0.0))))
        feat_costs = np.zeros(learner.num_features, dtype=float)
        for stage_params in info["stage_params"]:
            feat_contrib = np.asarray(
                getattr(stage_params, "feature_constraint_costs", np.zeros(learner.num_features)),
                dtype=float,
            )
            feat_costs[: min(len(feat_costs), len(feat_contrib))] += feat_contrib[: min(len(feat_costs), len(feat_contrib))]
        for feat_idx in range(learner.num_features):
            feature_constraint_by_feat[feat_idx].append(float(feat_costs[feat_idx]))

    ax.plot(candidate_values, total, color="black", lw=1.4, label="total")
    if show_components:
        ax.plot(candidate_values, constraint, color="crimson", lw=1.0, label="constraint")
        if np.any(np.isfinite(np.asarray(short_segment_penalty, dtype=float))):
            ax.plot(candidate_values, short_segment_penalty, color="tab:olive", lw=1.0, label="short_segment_penalty")
        feat_colors = _feature_plot_colors(learner.num_features)
        for feat_idx in range(learner.num_features):
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
        ax.plot(candidate_values, progress, color="tab:cyan", lw=1.0, label="progress")
        if np.any(np.isfinite(np.asarray(subgoal_consensus, dtype=float))):
            ax.plot(candidate_values, subgoal_consensus, color="tab:purple", lw=1.0, label="subgoal_consensus")
        if np.any(np.isfinite(np.asarray(param_consensus, dtype=float))):
            ax.plot(candidate_values, param_consensus, color="tab:pink", lw=1.0, label="param_consensus")
        if np.any(np.isfinite(np.asarray(activation_consensus, dtype=float))):
            ax.plot(candidate_values, activation_consensus, color="tab:brown", lw=1.0, label="activation_consensus")

    current_value = int(learned_cutpoints[vary_index])
    ax.axvline(current_value, color="black", linestyle="--", lw=1.0, label="pred boundary")
    true_cutpoints = _true_cutpoints_for_demo(learner, demo_idx)
    if vary_index < len(true_cutpoints):
        ax.axvline(int(true_cutpoints[vary_index]), color="green", linestyle=":", lw=1.0, label="true boundary")

    fixed_label = ", ".join(
        f"cp{k + 1}={learned_cutpoints[k]}"
        for k in range(num_cutpoints)
        if k != vary_index
    )
    varying_label = f"cp{vary_index + 1}"
    title = f"Demo {demo_idx} scan {varying_label}"
    if fixed_label:
        title += f" | fixed {fixed_label}"
    ax.set_title(title, fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel(f"{varying_label} index", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("cost", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_two_cut_scan(ax, learner, demo_idx=0, vary_index=1, show_components=True):
    _draw_single_cut_scan(ax, learner, demo_idx=demo_idx, vary_index=vary_index, show_components=show_components)


def plot_swcl_results_4panel(learner, it, demo_idx=0):
    if plt is None:
        return
    if _is_press_slide_insert(learner.env) and learner.num_stages == 4:
        fig = plt.figure(figsize=(11.4, 12.6))

        ax1 = fig.add_subplot(4, 2, 1)
        _draw_trajectories(ax1, learner, it, demo_idx=demo_idx)

        ax2 = fig.add_subplot(4, 2, 2)
        _draw_constraint_cost_matrix(ax2, learner, demo_idx=demo_idx)

        ax3 = fig.add_subplot(4, 2, 3)
        _draw_feature_bands(ax3, learner, demo_idx=demo_idx, standardized=False)

        ax4 = fig.add_subplot(4, 2, 4)
        _draw_feature_bands(ax4, learner, demo_idx=demo_idx, standardized=True)

        ax5 = fig.add_subplot(4, 2, 5)
        _draw_single_cut_scan(ax5, learner, demo_idx=demo_idx, vary_index=0, show_components=True)

        ax6 = fig.add_subplot(4, 2, 6)
        _draw_single_cut_scan(ax6, learner, demo_idx=demo_idx, vary_index=1, show_components=True)

        ax7 = fig.add_subplot(4, 2, 7)
        _draw_single_cut_scan(ax7, learner, demo_idx=demo_idx, vary_index=2, show_components=True)

        ax8 = fig.add_subplot(4, 2, 8)
        ax8.axis("off")
    elif _is_sphere_inspect(learner.env) and learner.num_stages in {4, 5}:
        if learner.num_stages == 4:
            fig = plt.figure(figsize=(12.6, 12.8))

            ax1 = fig.add_subplot(4, 2, 1, projection="3d")
            _draw_sphere_trajectory_3d(ax1, learner, it, demo_idx=demo_idx)

            ax2 = fig.add_subplot(4, 2, 2)
            _draw_constraint_cost_matrix(ax2, learner, demo_idx=demo_idx)

            ax3 = fig.add_subplot(4, 2, 3)
            _draw_sphere_feature_overview(ax3, learner, demo_idx=demo_idx)

            ax4 = fig.add_subplot(4, 2, 4)
            _draw_feature_bands(ax4, learner, demo_idx=demo_idx, standardized=False)

            ax5 = fig.add_subplot(4, 2, 5)
            _draw_feature_bands(ax5, learner, demo_idx=demo_idx, standardized=True)

            ax6 = fig.add_subplot(4, 2, 6)
            _draw_single_cut_scan(ax6, learner, demo_idx=demo_idx, vary_index=0, show_components=True)

            ax7 = fig.add_subplot(4, 2, 7)
            _draw_single_cut_scan(ax7, learner, demo_idx=demo_idx, vary_index=1, show_components=True)

            ax8 = fig.add_subplot(4, 2, 8)
            _draw_single_cut_scan(ax8, learner, demo_idx=demo_idx, vary_index=2, show_components=True)
        else:
            fig = plt.figure(figsize=(12.8, 15.6))

            ax1 = fig.add_subplot(5, 2, 1, projection="3d")
            _draw_sphere_trajectory_3d(ax1, learner, it, demo_idx=demo_idx)

            ax2 = fig.add_subplot(5, 2, 2)
            _draw_constraint_cost_matrix(ax2, learner, demo_idx=demo_idx)

            ax3 = fig.add_subplot(5, 2, 3)
            _draw_sphere_feature_overview(ax3, learner, demo_idx=demo_idx)

            ax4 = fig.add_subplot(5, 2, 4)
            _draw_feature_bands(ax4, learner, demo_idx=demo_idx, standardized=False)

            ax5 = fig.add_subplot(5, 2, 5)
            _draw_feature_bands(ax5, learner, demo_idx=demo_idx, standardized=True)

            ax6 = fig.add_subplot(5, 2, 6)
            _draw_single_cut_scan(ax6, learner, demo_idx=demo_idx, vary_index=0, show_components=True)

            ax7 = fig.add_subplot(5, 2, 7)
            _draw_single_cut_scan(ax7, learner, demo_idx=demo_idx, vary_index=1, show_components=True)

            ax8 = fig.add_subplot(5, 2, 8)
            _draw_single_cut_scan(ax8, learner, demo_idx=demo_idx, vary_index=2, show_components=True)

            ax9 = fig.add_subplot(5, 2, 9)
            _draw_single_cut_scan(ax9, learner, demo_idx=demo_idx, vary_index=3, show_components=True)

            ax10 = fig.add_subplot(5, 2, 10)
            ax10.axis("off")
    elif learner.num_stages == 3:
        fig = plt.figure(figsize=_trajectory_figsize(learner, three_row=True))

        ax1 = fig.add_subplot(3, 2, 1)
        _draw_trajectories(ax1, learner, it, demo_idx=demo_idx)

        ax2 = fig.add_subplot(3, 2, 2)
        _draw_constraint_cost_matrix(ax2, learner, demo_idx=demo_idx)

        ax3 = fig.add_subplot(3, 2, 3)
        _draw_feature_bands(ax3, learner, demo_idx=demo_idx, standardized=False)

        ax4 = fig.add_subplot(3, 2, 4)
        _draw_feature_bands(ax4, learner, demo_idx=demo_idx, standardized=True)

        ax5 = fig.add_subplot(3, 2, 5)
        _draw_two_cut_scan(ax5, learner, demo_idx=demo_idx, vary_index=0, show_components=True)

        ax6 = fig.add_subplot(3, 2, 6)
        _draw_two_cut_scan(ax6, learner, demo_idx=demo_idx, vary_index=1, show_components=True)
    else:
        fig = plt.figure(figsize=_trajectory_figsize(learner, three_row=False))

        ax1 = fig.add_subplot(2, 2, 1)
        _draw_trajectories(ax1, learner, it, demo_idx=demo_idx)

        ax2 = fig.add_subplot(2, 2, 2)
        _draw_constraint_cost_matrix(ax2, learner, demo_idx=demo_idx)

        ax3 = fig.add_subplot(2, 2, 3)
        _draw_feature_bands(ax3, learner, demo_idx=demo_idx, standardized=False)

        ax4 = fig.add_subplot(2, 2, 4)
        _draw_feature_bands(ax4, learner, demo_idx=demo_idx, standardized=True)

    save_figure(fig, learner_plot_dir(learner) / f"plot4panel_demo_{int(demo_idx):02d}_iter_{int(it):04d}.png", dpi=220)
    for vary_index in range(max(int(learner.num_stages) - 1, 0)):
        _plot_cutpoint_feature_distribution_compare(learner, it, demo_idx=demo_idx, vary_index=vary_index)


def plot_swcl_results_4panel_overview(learner, it, *, metrics=None, plot_dir=None):
    if plt is None:
        return
    feature_indices = []
    if getattr(learner, "current_stage_params_per_demo", None):
        feature_indices = list(range(int(getattr(learner, "num_features", 0))))
    param_rows = 0 if not feature_indices else int(np.ceil(len(feature_indices) / min(4, len(feature_indices))))
    has_error_row = bool(
        isinstance(metrics, dict)
        and np.asarray(metrics.get("ConstraintErrorMatrix", []), dtype=float).ndim == 2
        and np.asarray(metrics.get("ConstraintErrorMatrix", []), dtype=float).size > 0
    )
    base_w, base_h = _trajectory_figsize(learner, three_row=False)
    total_rows = 4 + param_rows
    fig = plt.figure(figsize=(base_w, base_h + 1.9 * param_rows + 3.8))
    gs = fig.add_gridspec(
        total_rows,
        2,
        height_ratios=[1.0] + ([0.85] * param_rows if param_rows > 0 else []) + [1.05, 1.0, 0.95],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_trajectories_overview(ax1, learner, it)

    ax2 = fig.add_subplot(gs[0, 1])
    _draw_learning_curves(ax2, learner)

    next_row = 1
    if param_rows > 0:
        _plot_constraint_parameter_panels(gs[next_row : next_row + param_rows, :], learner)
        next_row += param_rows

    ax3 = fig.add_subplot(gs[next_row, 0])
    _draw_activation_rate_history(ax3, learner)

    ax4 = fig.add_subplot(gs[next_row, 1])
    _draw_final_activation_rate_matrix(ax4, learner)

    next_row += 1

    ax5 = fig.add_subplot(gs[next_row, 0])
    _draw_final_activation_proto_matrix(ax5, learner)

    ax6 = fig.add_subplot(gs[next_row, 1])
    _draw_cutpoint_evolution(ax6, learner)

    next_row += 1

    ax7 = fig.add_subplot(gs[next_row, 0])
    _draw_eval_metric_text(ax7, metrics)

    ax8 = fig.add_subplot(gs[next_row, 1])
    _draw_constraint_error_matrix(ax8, learner, metrics) if has_error_row else ax8.axis("off")

    save_figure(fig, learner_plot_dir(learner, plot_dir=plot_dir) / f"training_summary_iter_{int(it):04d}.png", dpi=220)
    plot_constraint_type_summary(learner, it, plot_dir=plot_dir)
