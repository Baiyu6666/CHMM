from __future__ import annotations

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

from evaluation import evaluate_model_metrics
from methods.cores.swcl import _hard_gammas_from_stage_ends
from visualization.io import learner_plot_dir, save_figure
from visualization.learned_constraints_matrix import (
    _prepare_constraints_matrix,
    plot_learned_constraints_matrix_paper,
    plot_true_constraints_matrix_paper,
    plot_true_vs_learned_constraints_matrix_paper,
)
from visualization.top_view_scene import (
    draw_top_view_scene,
    top_view_scene_limit_points,
)


PAPER_TITLE_SIZE = 9
PAPER_LABEL_SIZE = 8
PAPER_TICK_SIZE = 7
PAPER_LEGEND_SIZE = 6.5
STAGE_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MODE_ORDER = ("inactive", "eq", "lb", "ub")
MODE_LABEL = {"inactive": "inactive", "eq": "eq", "lb": "lb", "ub": "ub"}
MODE_COLOR = {"inactive": "#6B7280", "eq": "#2563EB", "lb": "#059669", "ub": "#DC2626"}


def clear_map_plot_outputs(model) -> None:
    """Remove MAP plots from an earlier run before writing final-only outputs."""
    out_dir = learner_plot_dir(model)
    root_patterns = (
        "pooled_density.png",
        "pooled_density_iter_*.png",
        "density_demo_*.png",
        "map_mstep_summary.png",
        "map_mstep_summary_iter_*.png",
        "summary.png",
        "summary_iter_*.png",
        "DP_demo_*.png",
    )
    for pattern in root_patterns:
        for path in out_dir.glob(pattern):
            if path.is_file():
                path.unlink()
    paper_dir = out_dir / "paper_figures"
    if paper_dir.is_dir():
        for path in paper_dir.glob("paper_map_*.png"):
            if path.is_file():
                path.unlink()
        try:
            paper_dir.rmdir()
        except OSError:
            pass


def _stage_colors(n):
    return [STAGE_COLORS[i % len(STAGE_COLORS)] for i in range(max(int(n), 1))]


def _feature_name(learner, local_idx: int) -> str:
    if hasattr(learner, "feature_specs") and int(local_idx) < len(learner.feature_specs):
        name = learner.feature_specs[int(local_idx)].get("name")
        if name is not None:
            return str(name)
    schema = getattr(learner, "raw_feature_specs", None) or []
    columns = list(getattr(learner, "selected_feature_columns", []))
    if int(local_idx) < len(columns):
        col = int(columns[int(local_idx)])
        for spec_idx, spec in enumerate(schema):
            if int(spec.get("column_idx", spec_idx)) == col:
                return str(spec.get("name", f"f{local_idx}"))
    return f"f{int(local_idx)}"


def _segment_bounds(stage_ends):
    starts, ends = [], []
    start = 0
    for end in stage_ends:
        end_i = int(end)
        starts.append(start)
        ends.append(end_i)
        start = end_i + 1
    return starts, ends


def _true_cutpoints_for_demo(learner, demo_idx: int):
    true_cutpoints = getattr(learner, "true_cutpoints", None)
    if true_cutpoints is not None and int(demo_idx) < len(true_cutpoints):
        cuts = true_cutpoints[int(demo_idx)]
        if cuts is not None:
            return [int(x) for x in np.asarray(cuts, dtype=int).reshape(-1).tolist()]
    true_taus = getattr(learner, "true_taus", None)
    if true_taus is not None and int(demo_idx) < len(true_taus) and true_taus[int(demo_idx)] is not None:
        return [int(true_taus[int(demo_idx)])]
    return []


def _xy_point(point):
    return np.asarray(point, dtype=float).reshape(-1)[:2]


def _draw_obstacles(ax, env, demo_index=None, all_demos=False):
    draw_top_view_scene(
        ax,
        env,
        demo_index=demo_index,
        all_demos=all_demos,
    )
    if hasattr(env, "obs_center") and hasattr(env, "obs_radius"):
        center = _xy_point(getattr(env, "obs_center"))
        ax.add_patch(plt.Circle((center[0], center[1]), float(env.obs_radius), fill=False, color="gray", lw=1.0, label="obstacle"))
    if hasattr(env, "stage1_aux_obstacle_centers") and hasattr(env, "stage1_aux_obstacle_radii"):
        for center, radius in zip(np.asarray(env.stage1_aux_obstacle_centers, dtype=float), np.asarray(env.stage1_aux_obstacle_radii, dtype=float)):
            xy = _xy_point(center)
            ax.add_patch(plt.Circle((xy[0], xy[1]), float(radius), fill=False, color="gray", lw=0.9, linestyle=(0, (3, 2))))
    if hasattr(env, "get_true_reference_lines"):
        for idx, spec in enumerate(env.get_true_reference_lines()):
            point = _xy_point(spec["point"])
            direction = _xy_point(spec["direction"])
            ax.axline(
                point,
                point + direction,
                color=spec.get("color", "#475569"),
                linestyle=(0, (4, 3)),
                linewidth=1.0,
                alpha=0.8,
                label=str(spec.get("name", f"true line {idx + 1}")),
            )


def _nearby_obstacle_limit_points(
    env,
    data_points: np.ndarray,
    demo_index=None,
    all_demos=False,
) -> list[np.ndarray]:
    data_points = np.asarray(data_points, dtype=float)
    if data_points.ndim != 2 or data_points.shape[1] < 2 or data_points.size == 0:
        return []
    xy = data_points[:, :2]
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.size == 0:
        return []
    lo = np.min(xy, axis=0)
    hi = np.max(xy, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    guard = np.maximum(0.55 * span, 0.015)
    out = []
    scene_points = top_view_scene_limit_points(
        env,
        demo_index=demo_index,
        all_demos=all_demos,
    )
    if len(scene_points):
        out.append(scene_points)

    def maybe_add_circle(center, radius):
        center_xy = _xy_point(center)
        radius_f = max(float(radius), 0.0)
        circle_lo = center_xy - radius_f
        circle_hi = center_xy + radius_f
        if np.all(circle_hi >= lo - guard) and np.all(circle_lo <= hi + guard):
            out.append(np.asarray([circle_lo, circle_hi], dtype=float))

    if hasattr(env, "obs_center") and hasattr(env, "obs_radius"):
        maybe_add_circle(getattr(env, "obs_center"), getattr(env, "obs_radius"))
    if hasattr(env, "stage1_aux_obstacle_centers") and hasattr(env, "stage1_aux_obstacle_radii"):
        for center, radius in zip(np.asarray(env.stage1_aux_obstacle_centers, dtype=float), np.asarray(env.stage1_aux_obstacle_radii, dtype=float)):
            maybe_add_circle(center, radius)
    return out


def _set_compact_trajectory_limits(
    ax,
    env,
    data_points: np.ndarray,
    demo_index=None,
    all_demos=False,
):
    data_points = np.asarray(data_points, dtype=float)
    if data_points.ndim != 2 or data_points.shape[1] < 2 or data_points.size == 0:
        return
    limit_parts = [data_points[:, :2]]
    limit_parts.extend(
        _nearby_obstacle_limit_points(
            env,
            data_points,
            demo_index=demo_index,
            all_demos=all_demos,
        )
    )
    xy = np.vstack(limit_parts)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if xy.size == 0:
        return
    lo = np.min(xy, axis=0)
    hi = np.max(xy, axis=0)
    span = np.maximum(hi - lo, 1e-6)
    pad = max(0.10 * float(np.max(span)), 0.004)
    ax.set_xlim(float(lo[0] - pad), float(hi[0] + pad))
    ax.set_ylim(float(lo[1] - pad), float(hi[1] + pad))


def _legend(ax, *, outside=False, additional_axes=()):
    handles, labels = ax.get_legend_handles_labels()
    for additional_ax in additional_axes:
        additional_handles, additional_labels = additional_ax.get_legend_handles_labels()
        handles.extend(additional_handles)
        labels.extend(additional_labels)
    by_label = {}
    for handle, label in zip(handles, labels):
        text = "" if label is None else str(label).strip()
        if text and not text.startswith("_") and text not in by_label:
            by_label[text] = handle
    if by_label:
        kwargs = {"loc": "best"}
        if outside:
            kwargs = {"loc": "upper left", "bbox_to_anchor": (1.02, 1.0)}
        ax.legend(by_label.values(), by_label.keys(), fontsize=PAPER_LEGEND_SIZE, frameon=False, **kwargs)


def _mode_from_kind(learner, kind):
    if hasattr(learner, "_kind_to_mode"):
        try:
            return str(learner._kind_to_mode(kind))
        except Exception:
            pass
    kind_l = "" if kind is None else str(kind).lower()
    if kind_l in {"", "none", "inactive"}:
        return "inactive"
    if "lower" in kind_l or kind_l == "lb":
        return "lb"
    if "upper" in kind_l or kind_l == "ub":
        return "ub"
    return "eq"


def _shared_mode(learner, stage_idx: int, feat_idx: int) -> str:
    try:
        return _mode_from_kind(learner, learner.shared_param_kinds[int(stage_idx)][int(feat_idx)])
    except Exception:
        return "inactive"


def _shared_vector(learner, stage_idx: int, feat_idx: int):
    try:
        return learner.shared_param_vectors[int(stage_idx)][int(feat_idx)]
    except Exception:
        return None


def _map_semantics_for_mode(mode: str) -> str:
    mode_l = str(mode).lower()
    if mode_l == "eq":
        return "target_value"
    if mode_l == "lb":
        return "lower_bound"
    if mode_l == "ub":
        return "upper_bound"
    return ""


def _standardized_to_raw(learner, feat_idx: int, value: float) -> float:
    if not np.isfinite(float(value)):
        return np.nan
    columns = list(getattr(learner, "selected_feature_columns", []))
    raw_col = int(columns[int(feat_idx)]) if int(feat_idx) < len(columns) else int(feat_idx)
    mean = float(np.asarray(getattr(learner, "feat_mean", []), dtype=float)[raw_col])
    std = float(np.asarray(getattr(learner, "feat_std", []), dtype=float)[raw_col])
    return float(value) * std + mean


def _map_learned_constraint_payload(learner, base_metrics: dict | None = None) -> dict:
    payload = dict(base_metrics or {})
    num_stages = int(getattr(learner, "num_stages", 0))
    num_features = int(getattr(learner, "num_features", 0))
    values = np.full((num_stages, num_features), np.nan, dtype=float)
    semantics = np.full((num_stages, num_features), "", dtype=object)
    active = np.zeros((num_stages, num_features), dtype=int)

    for stage_idx in range(num_stages):
        for feat_idx in range(num_features):
            mode = _shared_mode(learner, stage_idx, feat_idx)
            semantics[stage_idx, feat_idx] = _map_semantics_for_mode(mode)
            if mode == "inactive":
                continue
            vec = _shared_vector(learner, stage_idx, feat_idx)
            if vec is None:
                continue
            arr = np.asarray(vec, dtype=float).reshape(-1)
            if arr.size == 0 or not np.isfinite(float(arr[0])):
                continue
            active[stage_idx, feat_idx] = 1
            # MAP stores the task-level equality target or inequality boundary
            # as eta in vector[0]. Other vector entries are nuisance parameters.
            values[stage_idx, feat_idx] = _standardized_to_raw(learner, feat_idx, float(arr[0]))

    payload["ConstraintLearnedValueMatrix"] = values.tolist()
    payload["ConstraintLearnedRawValueMatrix"] = values.tolist()
    payload["ConstraintLearnedSemanticsMatrix"] = semantics.tolist()
    payload["ConstraintPredictedActiveMask"] = active.tolist()
    payload["ConstraintFeatureNames"] = [_feature_name(learner, i) for i in range(num_features)]
    payload["ConstraintLearnedValueSource"] = "map_shared_eta"
    return payload


def _format_cost(value) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    return f"{value:.1f}" if abs(value) >= 100.0 else f"{value:.2f}"


def _map_mode_cost_rows(learner, demo_idx: int, stage_idx: int, feat_idx: int):
    if not hasattr(learner, "_local_mode_candidates_cached"):
        return None
    stage_ends = getattr(learner, "stage_ends_", None)
    if stage_ends is None or int(demo_idx) >= len(stage_ends):
        return None
    starts, ends = _segment_bounds(stage_ends[int(demo_idx)])
    if int(stage_idx) >= len(starts):
        return None
    if hasattr(learner, "_segment_core_bounds"):
        core_s, core_e = learner._segment_core_bounds(int(starts[int(stage_idx)]), int(ends[int(stage_idx)]))
    else:
        core_s, core_e = starts[int(stage_idx)], ends[int(stage_idx)]
    try:
        candidates = learner._local_mode_candidates_cached(int(demo_idx), int(core_s), int(core_e), int(feat_idx))
    except Exception:
        return None
    rows = []
    for mode in MODE_ORDER:
        fit = candidates.get(mode) if isinstance(candidates, dict) else None
        rows.append((mode, float(getattr(fit, "cost", np.nan)) if fit is not None else np.nan))
    finite = [(mode, cost) for mode, cost in rows if np.isfinite(cost)]
    best = min(finite, key=lambda item: (float(item[1]), MODE_ORDER.index(item[0])))[0] if finite else None
    return rows, best


def _draw_map_local_mode_cost_matrix(ax, learner, demo_idx=0, *, title=None, cell_fontsize=5.2):
    if not getattr(learner, "current_stage_params_per_demo", None):
        ax.axis("off")
        return
    num_stages = int(getattr(learner, "num_stages", 0))
    num_features = int(getattr(learner, "num_features", 0))
    ax.set_title(title or f"Demo {int(demo_idx)} local costs (color/underline = shared)", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlim(-0.5, num_stages - 0.5)
    ax.set_ylim(num_features - 0.5, -0.5)
    ax.set_xticks(range(num_stages))
    ax.set_xticklabels([f"stage {i + 1}" for i in range(num_stages)])
    ax.set_yticks(range(num_features))
    ax.set_yticklabels([_feature_name(learner, i) for i in range(num_features)])
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    for x in np.arange(-0.5, num_stages + 0.5, 1.0):
        ax.axvline(x, color="#c9c9c9", linewidth=0.7, zorder=0)
    for y in np.arange(-0.5, num_features + 0.5, 1.0):
        ax.axhline(y, color="#c9c9c9", linewidth=0.7, zorder=0)
    offsets = {"inactive": -0.27, "eq": -0.09, "lb": 0.09, "ub": 0.27}
    for feat_idx in range(num_features):
        for stage_idx in range(num_stages):
            result = _map_mode_cost_rows(learner, demo_idx, stage_idx, feat_idx)
            if result is None:
                continue
            rows, best_mode = result
            shared_mode = _shared_mode(learner, stage_idx, feat_idx)
            for mode, cost in rows:
                is_best = mode == best_mode
                text = f"{mode}: {_format_cost(cost)}"
                text_y = feat_idx + offsets[mode]
                color = MODE_COLOR[mode] if mode == shared_mode else ("#D62728" if is_best else "#333333")
                ax.text(
                    stage_idx,
                    text_y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=cell_fontsize,
                    color=color,
                    fontweight="bold" if mode in {best_mode, shared_mode} else "normal",
                )
                if mode == shared_mode:
                    half_width = min(0.42, max(0.15, 0.014 * len(text)))
                    ax.plot([stage_idx - half_width, stage_idx + half_width], [text_y + 0.055, text_y + 0.055], color=color, lw=0.9)


def _draw_map_shared_mode_cost_matrix(ax, learner, *, title="MAP shared pooled mode costs"):
    costs = getattr(learner, "map_shared_mode_costs_", None)
    num_stages = int(getattr(learner, "num_stages", 0))
    num_features = int(getattr(learner, "num_features", 0))
    if not costs or num_stages <= 0 or num_features <= 0:
        ax.axis("off")
        return
    ax.set_title(title, fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlim(-0.5, num_stages - 0.5)
    ax.set_ylim(num_features - 0.5, -0.5)
    ax.set_xticks(range(num_stages))
    ax.set_xticklabels([f"stage {i + 1}" for i in range(num_stages)])
    ax.set_yticks(range(num_features))
    ax.set_yticklabels([_feature_name(learner, i) for i in range(num_features)])
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    for x in np.arange(-0.5, num_stages + 0.5, 1.0):
        ax.axvline(x, color="#c9c9c9", linewidth=0.7, zorder=0)
    for y in np.arange(-0.5, num_features + 0.5, 1.0):
        ax.axhline(y, color="#c9c9c9", linewidth=0.7, zorder=0)
    offsets = {"inactive": -0.27, "eq": -0.09, "lb": 0.09, "ub": 0.27}
    for feat_idx in range(num_features):
        for stage_idx in range(num_stages):
            try:
                cell_costs = dict(costs[stage_idx][feat_idx])
            except Exception:
                cell_costs = {}
            finite = [(m, float(cell_costs.get(m, np.nan))) for m in MODE_ORDER if np.isfinite(float(cell_costs.get(m, np.nan)))]
            best_mode = min(finite, key=lambda item: (item[1], MODE_ORDER.index(item[0])))[0] if finite else None
            shared_mode = _shared_mode(learner, stage_idx, feat_idx)
            for mode in MODE_ORDER:
                value = float(cell_costs.get(mode, np.nan))
                is_best = mode == best_mode
                text = f"{mode}: {_format_cost(value)}"
                text_y = feat_idx + offsets[mode]
                color = MODE_COLOR[mode] if mode == shared_mode else ("#D62728" if is_best else "#333333")
                ax.text(
                    stage_idx,
                    text_y,
                    text,
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color=color,
                    fontweight="bold" if mode in {best_mode, shared_mode} else "normal",
                )
                if mode == shared_mode:
                    half_width = min(0.42, max(0.15, 0.014 * len(text)))
                    ax.plot([stage_idx - half_width, stage_idx + half_width], [text_y + 0.055, text_y + 0.055], color=color, lw=1.0)


def _draw_map_constraints_matrix(
    ax,
    learner,
    payload,
    *,
    value_key,
    semantics_key,
    active_key,
    title,
):
    try:
        matrix, text_matrix, stage_labels, display_labels = _prepare_constraints_matrix(
            payload,
            value_key=value_key,
            semantics_key=semantics_key,
            active_key=active_key,
            dataset_name=str(getattr(getattr(learner, "env", None), "eval_tag", "")),
        )
    except Exception:
        ax.axis("off")
        return

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#f3f3f3", "#54a24b", "#e45756", "#4c78a8"])
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=3, aspect="auto", interpolation="nearest")
    ax.set_title(title, fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xticks(np.arange(len(stage_labels)))
    ax.set_xticklabels([label.replace("stage ", "s") for label in stage_labels])
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_yticklabels(display_labels)
    ax.tick_params(axis="both", which="major", labelsize=PAPER_TICK_SIZE, length=3, width=0.8)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            code = int(matrix[row, col])
            ax.text(
                col,
                row,
                str(text_matrix[row, col]),
                ha="center",
                va="center",
                fontsize=7.0,
                color="white" if code else "black",
                linespacing=0.9,
            )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#333333")
    ax.set_xlim(-0.5, matrix.shape[1] - 0.5)
    ax.set_ylim(matrix.shape[0] - 0.5, -0.5)


def _draw_map_true_constraints_matrix(ax, learner, metrics=None, *, title="True constraints"):
    payload = _map_learned_constraint_payload(learner, metrics)
    _draw_map_constraints_matrix(
        ax,
        learner,
        payload,
        value_key="ConstraintTargetMatrix",
        semantics_key="ConstraintSemanticsMatrix",
        active_key="ConstraintTrueActiveMask",
        title=title,
    )


def _draw_map_learned_constraints_matrix(ax, learner, metrics=None, *, title="MAP learned constraints"):
    payload = _map_learned_constraint_payload(learner, metrics)
    _draw_map_constraints_matrix(
        ax,
        learner,
        payload,
        value_key="ConstraintLearnedValueMatrix",
        semantics_key="ConstraintLearnedSemanticsMatrix",
        active_key="ConstraintPredictedActiveMask",
        title=title,
    )


def _draw_trajectory(ax, learner, it, demo_idx=0, *, overview=False):
    colors = _stage_colors(getattr(learner, "num_stages", 1))
    demos = learner.demos if overview else [learner.demos[int(demo_idx)]]
    stage_ends_all = learner.stage_ends_ if overview else [learner.stage_ends_[int(demo_idx)]]
    limit_points = []
    for i, (X, stage_ends) in enumerate(zip(demos, stage_ends_all)):
        X = np.asarray(X, dtype=float)
        limit_points.append(X[:, :2])
        starts, ends = _segment_bounds(stage_ends)
        for k, (s, e) in enumerate(zip(starts, ends)):
            ax.scatter(X[s : e + 1, 0], X[s : e + 1, 1], color=colors[k], s=2.6, alpha=0.35)
        for cp_idx, cp in enumerate(_true_cutpoints_for_demo(learner, i if overview else demo_idx)):
            if 0 <= int(cp) < len(X):
                ax.scatter(X[int(cp), 0], X[int(cp), 1], color=colors[cp_idx % len(colors)], marker="x", s=24, lw=1.3, label="true boundary" if i == 0 and cp_idx == 0 else "")
    _draw_obstacles(
        ax,
        learner.env,
        demo_index=None if overview else demo_idx,
        all_demos=overview,
    )
    ax.set_title(f"Iter {int(it)}: MAP trajectories" if overview else f"Iter {int(it)}: demo {int(demo_idx)} trajectory", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("x", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("y", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    ax.set_aspect("equal", adjustable="box")
    if limit_points:
        _set_compact_trajectory_limits(
            ax,
            learner.env,
            np.vstack(limit_points),
            demo_index=None if overview else demo_idx,
            all_demos=overview,
        )
    _legend(ax)


def _draw_learning_curves(ax, learner):
    iters = np.arange(len(getattr(learner, "loss_total", []) or []))
    if iters.size == 0:
        ax.axis("off")
        return
    ax.plot(iters, learner.loss_total, color="black", lw=1.35, label="total")
    ax.plot(iters, learner.loss_constraint, color="tab:red", lw=1.0, label="constraint")
    progress = np.asarray(getattr(learner, "loss_progress", []), dtype=float)
    if progress.size == iters.size:
        ax.plot(iters, progress, color="tab:orange", lw=1.0, label="progress")
    ax.set_title("MAP objective", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("iteration", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("cost", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_feature_traces(ax, learner, demo_idx=0, *, standardized=False):
    Fz = np.asarray(learner.standardized_features[int(demo_idx)], dtype=float)
    if standardized:
        values = Fz
        ylabel = "standardized feature value"
    else:
        cols = np.asarray(getattr(learner, "selected_feature_columns", np.arange(Fz.shape[1])), dtype=int)
        values = Fz * np.asarray(learner.feat_std, dtype=float)[cols][None, :] + np.asarray(learner.feat_mean, dtype=float)[cols][None, :]
        ylabel = "raw feature value"
    t = np.arange(values.shape[0])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    starts, ends = _segment_bounds(learner.stage_ends_[int(demo_idx)])
    mode_styles = {
        "eq": {"linestyle": "--", "marker": "o", "label": "eq target"},
        "lb": {"linestyle": ":", "marker": "^", "label": "lower bound"},
        "ub": {"linestyle": "-.", "marker": "v", "label": "upper bound"},
    }
    mode_labels_seen = set()
    for feat_idx in range(values.shape[1]):
        color = colors[feat_idx % len(colors)]
        ax.plot(t, values[:, feat_idx], color=color, lw=1.0, label=_feature_name(learner, feat_idx))
        for stage_idx, (s, e) in enumerate(zip(starts, ends)):
            mode = _shared_mode(learner, stage_idx, feat_idx)
            if mode == "inactive":
                continue
            vec = _shared_vector(learner, stage_idx, feat_idx)
            if vec is None:
                continue
            center = float(np.asarray(vec, dtype=float).reshape(-1)[0])
            if not standardized:
                cols = np.asarray(getattr(learner, "selected_feature_columns", np.arange(values.shape[1])), dtype=int)
                center = center * float(learner.feat_std[int(cols[feat_idx])]) + float(learner.feat_mean[int(cols[feat_idx])])
            style = mode_styles.get(mode, mode_styles["eq"])
            label = style["label"] if style["label"] not in mode_labels_seen else ""
            mode_labels_seen.add(style["label"])
            ax.plot(
                t[s : e + 1],
                np.full(e - s + 1, center),
                linestyle=style["linestyle"],
                color=color,
                lw=1.15 if mode in {"lb", "ub"} else 0.95,
                alpha=0.88,
                label=label,
            )
            mid = int((int(s) + int(e)) // 2)
            ax.scatter(
                [mid],
                [center],
                marker=style["marker"],
                s=20,
                color=color,
                edgecolor="white",
                linewidth=0.45,
                zorder=5,
            )
    for j, cp in enumerate(learner.stage_ends_[int(demo_idx)][:-1]):
        ax.axvline(int(cp), color="black", linestyle="--", lw=1.0, label="pred boundary" if j == 0 else "")
    for j, cp in enumerate(_true_cutpoints_for_demo(learner, demo_idx)):
        ax.axvline(int(cp), color="green", linestyle=":", lw=1.0, label="true boundary" if j == 0 else "")
    ax.set_title(f"Demo {int(demo_idx)} shared MAP params ({'std' if standardized else 'raw'})", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("time", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_cutpoint_evolution(ax, learner):
    history = [list(item) for item in getattr(learner, "segmentation_history", [])]
    if not history:
        ax.axis("off")
        return
    num_cutpoints = max(int(getattr(learner, "num_stages", 1)) - 1, 0)
    if num_cutpoints <= 0:
        ax.axis("off")
        return
    x = np.arange(len(history))
    demo_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(history[0]), 3)))
    styles = ["-", "--", "-.", ":"]
    for demo_idx in range(len(history[0])):
        for cp_idx in range(num_cutpoints):
            vals = [int(snapshot[demo_idx][cp_idx]) for snapshot in history]
            ax.plot(x, vals, color=demo_colors[demo_idx], linestyle=styles[cp_idx % len(styles)], lw=1.2, marker="o", ms=2.2, label=f"cp{cp_idx + 1}" if demo_idx == 0 else "")
    ax.set_title("MAP cutpoint evolution", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("iteration", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("cutpoint index", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax)


def _draw_per_demo_cutpoint_comparison(ax, learner):
    stage_ends_all = [list(map(int, ends)) for ends in getattr(learner, "stage_ends_", [])]
    num_demos = min(len(getattr(learner, "demos", [])), len(stage_ends_all))
    num_cutpoints = max(int(getattr(learner, "num_stages", 1)) - 1, 0)
    if num_demos <= 0 or num_cutpoints <= 0:
        ax.axis("off")
        return

    colors = _stage_colors(num_cutpoints)
    absolute_errors = []
    comparable_demos = 0
    for demo_idx in range(num_demos):
        length = int(len(learner.demos[demo_idx]))
        denominator = float(max(length - 1, 1))
        predicted = np.asarray(stage_ends_all[demo_idx][:-1], dtype=float)
        true = np.asarray(_true_cutpoints_for_demo(learner, demo_idx), dtype=float)
        pair_count = min(num_cutpoints, predicted.size, true.size)
        y = float(demo_idx)

        ax.hlines(y, 0.0, 1.0, color="#D1D5DB", linewidth=0.55, zorder=0)
        if pair_count > 0:
            comparable_demos += 1
            absolute_errors.extend(np.abs(predicted[:pair_count] - true[:pair_count]).tolist())
        for cut_idx in range(pair_count):
            true_position = float(true[cut_idx] / denominator)
            predicted_position = float(predicted[cut_idx] / denominator)
            color = colors[cut_idx]
            ax.plot(
                [true_position, predicted_position],
                [y, y],
                color=color,
                linewidth=1.25,
                alpha=0.72,
                zorder=2,
            )
            ax.scatter(
                [true_position],
                [y],
                marker="o",
                s=25,
                facecolors="white",
                edgecolors=color,
                linewidths=1.15,
                zorder=3,
                label="true" if demo_idx == 0 and cut_idx == 0 else "",
            )
            ax.scatter(
                [predicted_position],
                [y],
                marker="x",
                s=27,
                color=color,
                linewidths=1.3,
                zorder=4,
                label="learned" if demo_idx == 0 and cut_idx == 0 else "",
            )

    for cut_idx, color in enumerate(colors):
        ax.scatter([], [], marker="s", s=18, color=color, label=f"cut {cut_idx + 1}")

    title = "Per-demo true vs learned cutpoints"
    if absolute_errors:
        title += f" | MAE={float(np.mean(absolute_errors)):.2f} samples"
    elif comparable_demos == 0:
        title += " | true cutpoints unavailable"
    ax.set_title(title, fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("normalized trajectory progress", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("demo", fontsize=PAPER_LABEL_SIZE)
    ax.set_xlim(-0.015, 1.015)
    ax.set_ylim(-0.65, float(num_demos) - 0.35)
    ax.set_xticks(np.linspace(0.0, 1.0, 5))
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks(np.arange(num_demos, dtype=float))
    ax.set_yticklabels([str(idx + 1) for idx in range(num_demos)])
    ax.invert_yaxis()
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.55, alpha=0.72)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    _legend(ax, outside=True)


def _scan_cutpoint_range(learner, T: int, fixed_cutpoints, vary_index: int) -> np.ndarray:
    fixed_cutpoints = [int(x) for x in fixed_cutpoints]
    num_cutpoints = int(getattr(learner, "num_stages", 1)) - 1
    vary_index = int(vary_index)
    if vary_index < 0 or vary_index >= num_cutpoints:
        return np.asarray([], dtype=int)

    duration_min = np.asarray(getattr(learner, "duration_min", np.ones(num_cutpoints + 1)), dtype=int)
    duration_max = np.asarray(getattr(learner, "duration_max", np.full(num_cutpoints + 1, int(T))), dtype=int)
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


def _map_stage_loss_breakdown_for_stage_ends(learner, demo_idx: int, stage_ends) -> dict | None:
    starts, ends = _segment_bounds(stage_ends)
    if len(starts) != int(getattr(learner, "num_stages", 0)):
        return None
    stage_total = []
    stage_constraint = []
    stage_progress = []
    for stage_idx, (s, e) in enumerate(zip(starts, ends)):
        try:
            info = learner._shared_interval_cost_info(int(demo_idx), int(stage_idx), int(s), int(e))
        except Exception:
            info = None
        if info is None:
            return None
        stage_total.append(float(info["weighted_total"]))
        stage_constraint.append(float(info["constraint"]))
        stage_progress.append(float(info["progress"]))
    return {
        "stage_total": np.asarray(stage_total, dtype=float),
        "stage_constraint": np.asarray(stage_constraint, dtype=float),
        "stage_progress": np.asarray(stage_progress, dtype=float),
        "total": float(np.sum(stage_total)),
        "constraint": float(np.sum(stage_constraint)),
        "progress": float(np.sum(stage_progress)),
    }


def _map_adjacent_feature_costs_for_stage_ends(learner, demo_idx: int, stage_ends, vary_index: int) -> np.ndarray | None:
    starts, ends = _segment_bounds(stage_ends)
    vary_index = int(vary_index)
    if len(starts) != int(getattr(learner, "num_stages", 0)) or vary_index + 1 >= len(starts):
        return None
    feature_costs = np.full(int(getattr(learner, "num_features", 0)), np.nan, dtype=float)
    for feat_idx in range(feature_costs.size):
        try:
            left = learner._shared_feature_interval_cost(
                int(demo_idx),
                vary_index,
                int(feat_idx),
                int(starts[vary_index]),
                int(ends[vary_index]),
            )
            right = learner._shared_feature_interval_cost(
                int(demo_idx),
                vary_index + 1,
                int(feat_idx),
                int(starts[vary_index + 1]),
                int(ends[vary_index + 1]),
            )
        except Exception:
            return None
        feature_costs[feat_idx] = float(left) + float(right)
    return feature_costs


def _finite_values(*arrays) -> np.ndarray:
    parts = []
    for values in arrays:
        arr = np.asarray(values, dtype=float).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.zeros(0, dtype=float)
    return np.concatenate(parts, axis=0)


def _values_at_cutpoints(candidate_values: np.ndarray, value_arrays: list[np.ndarray], cutpoints: list[int]) -> np.ndarray:
    candidate_values = np.asarray(candidate_values, dtype=int).reshape(-1)
    picked = []
    for cutpoint in cutpoints:
        matches = np.where(candidate_values == int(cutpoint))[0]
        if matches.size == 0:
            continue
        idx = int(matches[0])
        for values in value_arrays:
            arr = np.asarray(values, dtype=float).reshape(-1)
            if idx < arr.size and np.isfinite(arr[idx]):
                picked.append(float(arr[idx]))
    return np.asarray(picked, dtype=float)


def _values_near_cutpoints(candidate_values: np.ndarray, value_arrays: list[np.ndarray], cutpoints: list[int]) -> np.ndarray:
    candidate_values = np.asarray(candidate_values, dtype=int).reshape(-1)
    if candidate_values.size == 0 or not cutpoints:
        return np.zeros(0, dtype=float)
    window_radius = max(2, int(np.ceil(0.05 * candidate_values.size)))
    focus_mask = np.zeros(candidate_values.size, dtype=bool)
    for cutpoint in cutpoints:
        focus_mask |= np.abs(candidate_values - int(cutpoint)) <= window_radius
    picked = []
    for values in value_arrays:
        arr = np.asarray(values, dtype=float).reshape(-1)
        usable = min(arr.size, focus_mask.size)
        if usable == 0:
            continue
        local = arr[:usable][focus_mask[:usable]]
        local = local[np.isfinite(local)]
        if local.size:
            picked.append(local)
    if not picked:
        return np.zeros(0, dtype=float)
    return np.concatenate(picked, axis=0)


def _set_cut_scan_ylim(ax, candidate_values: np.ndarray, value_arrays: list[np.ndarray], focus_cutpoints: list[int]) -> None:
    finite = _finite_values(*value_arrays)
    if finite.size < 3:
        return
    candidate_set = set(np.asarray(candidate_values, dtype=int).reshape(-1).tolist())
    unique_focus_cutpoints = sorted(
        {int(cutpoint) for cutpoint in focus_cutpoints if int(cutpoint) in candidate_set}
    )
    focus_values = _values_at_cutpoints(candidate_values, value_arrays, unique_focus_cutpoints)
    local_values = _values_near_cutpoints(candidate_values, value_arrays, unique_focus_cutpoints)
    reference = local_values if local_values.size >= 3 else finite
    local_low, local_high = np.nanpercentile(reference, [10.0, 90.0])
    focus_low = float(np.nanmin(focus_values)) if focus_values.size else float(local_low)
    focus_high = float(np.nanmax(focus_values)) if focus_values.size else float(local_high)
    if len(unique_focus_cutpoints) == 1 and focus_values.size >= 2:
        local_q25, local_q75 = np.nanpercentile(reference, [25.0, 75.0])
        robust_span = max(
            float(focus_high - focus_low),
            float(local_q75 - local_q25),
            1e-9,
        )
        local_high = min(float(local_high), float(focus_high + 2.0 * robust_span))
    y_min = min(float(local_low), focus_low)
    y_upper = max(float(local_high), focus_high)
    if not np.isfinite(y_upper) or y_upper <= y_min:
        return
    span = max(float(y_upper - y_min), 1e-9)
    center = 0.5 * float(y_min + y_upper)
    expanded_half_span = span
    pad = 0.10 * (2.0 * span)
    ax.set_ylim(center - expanded_half_span - pad, center + expanded_half_span + pad)


def _draw_single_cut_scan(ax, learner, demo_idx=0, vary_index=0, *, show_components=True):
    num_stages = int(getattr(learner, "num_stages", 0))
    if num_stages < 2:
        ax.axis("off")
        return

    stage_ends = [int(x) for x in getattr(learner, "stage_ends_", [[]])[int(demo_idx)]]
    learned_cutpoints = [int(x) for x in stage_ends[:-1]]
    num_cutpoints = len(learned_cutpoints)
    vary_index = int(vary_index)
    if vary_index < 0 or vary_index >= num_cutpoints:
        ax.axis("off")
        return

    T = int(len(learner.demos[int(demo_idx)]))
    candidate_values = _scan_cutpoint_range(learner, T, learned_cutpoints, vary_index)
    if candidate_values.size == 0:
        ax.text(0.5, 0.5, "No feasible cutpoints.", ha="center", va="center", fontsize=8)
        ax.axis("off")
        return

    feature_cost_by_feat = [[] for _ in range(int(getattr(learner, "num_features", 0)))]
    total_feature_cost = []
    progress_cost = []
    for value in candidate_values:
        candidate_cutpoints = list(learned_cutpoints)
        candidate_cutpoints[vary_index] = int(value)
        if any(candidate_cutpoints[k] >= candidate_cutpoints[k + 1] for k in range(len(candidate_cutpoints) - 1)):
            feature_costs = None
            breakdown = None
        else:
            candidate_stage_ends = [int(x) for x in candidate_cutpoints] + [int(T - 1)]
            feature_costs = _map_adjacent_feature_costs_for_stage_ends(
                learner,
                int(demo_idx),
                candidate_stage_ends,
                vary_index,
            )
            breakdown = _map_stage_loss_breakdown_for_stage_ends(
                learner,
                int(demo_idx),
                candidate_stage_ends,
            )
        if feature_costs is None or breakdown is None:
            total_feature_cost.append(np.nan)
            progress_cost.append(np.nan)
            for values in feature_cost_by_feat:
                values.append(np.nan)
            continue
        feature_costs = np.asarray(feature_costs, dtype=float)
        total_feature_cost.append(float(np.nansum(feature_costs)))
        progress_cost.append(float(np.sum(breakdown["stage_progress"][vary_index : vary_index + 2])))
        for feat_idx, values in enumerate(feature_cost_by_feat):
            values.append(float(feature_costs[feat_idx]) if feat_idx < feature_costs.size else np.nan)

    constraint_arr = np.asarray(total_feature_cost, dtype=float)
    progress_arr = np.asarray(progress_cost, dtype=float)
    total_arr = constraint_arr + progress_arr
    current_value = int(learned_cutpoints[vary_index])
    ax.plot(candidate_values, total_arr, color="black", lw=1.45, label="total MAP")
    progress_ax = None
    if show_components:
        ax.plot(
            candidate_values,
            constraint_arr,
            color="#555555",
            linestyle="--",
            lw=1.1,
            alpha=0.9,
            label="sum features",
        )
        current_matches = np.flatnonzero(candidate_values == current_value)
        if current_matches.size and np.isfinite(progress_arr[int(current_matches[0])]):
            progress_reference = float(progress_arr[int(current_matches[0])])
        else:
            finite_progress = progress_arr[np.isfinite(progress_arr)]
            progress_reference = float(finite_progress[0]) if finite_progress.size else 0.0
        progress_delta = progress_arr - progress_reference
        progress_ax = ax.twinx()
        progress_ax.plot(
            candidate_values,
            progress_delta,
            color="#A21CAF",
            linestyle="-.",
            lw=1.1,
            alpha=0.9,
            label="progress delta (right)",
        )
        progress_ax.axhline(0.0, color="#A21CAF", linestyle=":", lw=0.65, alpha=0.35)
        progress_ax.set_ylabel("progress cost delta", color="#A21CAF", fontsize=PAPER_LABEL_SIZE)
        progress_ax.tick_params(axis="y", colors="#A21CAF", labelsize=PAPER_TICK_SIZE, pad=1.5)
    feature_colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(feature_cost_by_feat), 1)))
    for feat_idx, values in enumerate(feature_cost_by_feat):
        feat_values = np.asarray(values, dtype=float)
        if not np.any(np.isfinite(feat_values)):
            continue
        ax.plot(
            candidate_values,
            feat_values,
            color=feature_colors[feat_idx % len(feature_colors)],
            lw=1.08,
            alpha=0.92,
            label=_feature_name(learner, feat_idx),
        )

    ax.axvline(current_value, color="black", linestyle="--", lw=1.0, label="pred boundary")
    true_cutpoints = _true_cutpoints_for_demo(learner, demo_idx)
    focus_cutpoints = [current_value]
    if vary_index < len(true_cutpoints):
        true_value = int(true_cutpoints[vary_index])
        focus_cutpoints.append(true_value)
        ax.axvline(true_value, color="green", linestyle=":", lw=1.0, label="true boundary")

    finite_total = total_arr[np.isfinite(total_arr)]
    if finite_total.size:
        best_idx = int(np.nanargmin(total_arr))
        best_value = int(candidate_values[best_idx])
        focus_cutpoints.append(best_value)
        ax.axvline(best_value, color="#7C2D12", linestyle="-.", lw=0.9, alpha=0.75, label="scan min")

    fixed_label = ", ".join(
        f"cp{k + 1}={learned_cutpoints[k]}"
        for k in range(num_cutpoints)
        if k != vary_index
    )
    title = f"Demo {int(demo_idx)} scan cp{vary_index + 1}"
    if fixed_label:
        title += f" | fixed {fixed_label}"
    ax.set_title(title, fontsize=PAPER_TITLE_SIZE, pad=3)
    ax.set_xlabel(f"cp{vary_index + 1} index", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("adjacent-stage MAP cost", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE, pad=1.5)
    ax.grid(axis="y", color="#cfcfcf", linewidth=0.55, alpha=0.18)
    _set_cut_scan_ylim(
        ax,
        candidate_values,
        [total_arr, constraint_arr]
        + [np.asarray(values, dtype=float) for values in feature_cost_by_feat],
        focus_cutpoints,
    )
    _legend(ax, additional_axes=(() if progress_ax is None else (progress_ax,)))


def _draw_metrics(ax, metrics):
    ax.axis("off")
    if not isinstance(metrics, dict):
        ax.text(0.02, 0.98, "No metrics.", ha="left", va="top", fontsize=8, transform=ax.transAxes)
        return
    preferred = [
        "MeanAbsCutpointError",
        "SemanticConstraintF1",
        "MeanStageSubgoalError",
        "MeanParameterError",
        "MeanParameterErrorRaw",
    ]
    scalar = {}
    for key, value in metrics.items():
        if np.isscalar(value):
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(value_f):
                scalar[str(key)] = value_f
    keys = [k for k in preferred if k in scalar] + [k for k in sorted(scalar) if k not in preferred]
    ax.set_title("MAP evaluation metrics", fontsize=PAPER_TITLE_SIZE, pad=4)
    y = 0.92
    for idx, key in enumerate(keys[:7]):
        ax.text(0.03, y, key, ha="left", va="top", fontsize=7.4, color="#444444", transform=ax.transAxes)
        ax.text(0.97, y, f"{scalar[key]:.4f}", ha="right", va="top", fontsize=9.0 if idx < 4 else 8.0, transform=ax.transAxes)
        y -= 0.15


def _draw_parameter_error_matrix(ax, learner, metrics):
    metric_values = metrics or {}
    matrix = np.asarray(
        metric_values.get("ParameterErrorMatrix", metric_values.get("ConstraintErrorMatrix", [])),
        dtype=float,
    )
    if matrix.ndim != 2 or matrix.size == 0:
        ax.axis("off")
        return
    arr = matrix.T
    finite = arr[np.isfinite(arr)]
    vmax = float(np.nanmax(finite)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#d9d9d9")
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
    ax.set_title("parameter error", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xticks(range(arr.shape[1]))
    ax.set_xticklabels([f"s{i + 1}" for i in range(arr.shape[1])])
    ax.set_yticks(range(arr.shape[0]))
    names = (metrics or {}).get("ConstraintFeatureNames") or [_feature_name(learner, i) for i in range(arr.shape[0])]
    ax.set_yticklabels(names)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            ax.text(j, i, "nan" if not np.isfinite(value) else f"{float(value):.3f}", ha="center", va="center", fontsize=6.8)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.025)


def plot_map_mode_costs_paper(learner, demo_idx=0, save_path=None):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(max(4.6, 1.18 * learner.num_stages + 1.35), max(2.35, 0.82 * learner.num_features + 0.62)), constrained_layout=False)
    _draw_map_local_mode_cost_matrix(ax, learner, demo_idx=demo_idx, title=f"Demo {int(demo_idx)} MAP local mode costs", cell_fontsize=6.0)
    fig.tight_layout(pad=0.35)
    path = save_path or learner_plot_dir(learner) / f"paper_map_mode_costs_demo_{int(demo_idx):02d}.png"
    return save_figure(fig, path, dpi=300)


def plot_map_shared_modes_paper(learner, save_path=None):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(max(4.8, 1.25 * learner.num_stages + 1.4), max(2.35, 0.85 * learner.num_features + 0.62)), constrained_layout=False)
    _draw_map_shared_mode_cost_matrix(ax, learner)
    fig.tight_layout(pad=0.35)
    path = save_path or learner_plot_dir(learner) / "paper_map_shared_modes.png"
    return save_figure(fig, path, dpi=300)


def plot_map_true_cutpoint_trajectory_paper(learner, demo_idx=0, save_path=None):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(3.1, 2.25), constrained_layout=False)
    _draw_trajectory(ax, learner, 0, demo_idx=demo_idx, overview=False)
    fig.tight_layout(pad=0.35)
    path = save_path or learner_plot_dir(learner) / f"paper_map_trajectory_demo_{int(demo_idx):02d}.png"
    return save_figure(fig, path, dpi=300)


def plot_map_key_feature_traces_paper(learner, demo_idx=0, save_path=None):
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(5.0, 2.15), constrained_layout=False)
    _draw_feature_traces(ax, learner, demo_idx=demo_idx, standardized=True)
    fig.tight_layout(pad=0.35)
    path = save_path or learner_plot_dir(learner) / f"paper_map_key_feature_traces_demo_{int(demo_idx):02d}.png"
    return save_figure(fig, path, dpi=300)


def plot_map_demo_summary(learner, it, demo_idx=0):
    if plt is None:
        return None
    num_features = max(int(getattr(learner, "num_features", 1)), 1)
    num_stages = max(int(getattr(learner, "num_stages", 1)), 1)
    num_cutpoints = max(int(getattr(learner, "num_stages", 1)) - 1, 0)
    scan_rows = int(np.ceil(float(num_cutpoints) / 2.0)) if num_cutpoints > 0 else 0
    total_rows = 2 + scan_rows
    local_cost_row_height = max(2.5, 0.5 * float(num_features) + 0.6)
    feature_trace_row_height = 2.4
    scan_row_height = 2.0
    row_heights = [local_cost_row_height, feature_trace_row_height] + [scan_row_height] * scan_rows
    fig_height = float(sum(row_heights) + 0.7)
    fig_width = max(7.8, 1.15 * float(num_stages) + 3.2)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(total_rows, 2, height_ratios=row_heights)

    ax1 = fig.add_subplot(gs[0, 0])
    _draw_trajectory(ax1, learner, it, demo_idx=demo_idx, overview=False)
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_map_local_mode_cost_matrix(ax2, learner, demo_idx=demo_idx)
    ax3 = fig.add_subplot(gs[1, 0])
    _draw_feature_traces(ax3, learner, demo_idx=demo_idx, standardized=False)
    ax4 = fig.add_subplot(gs[1, 1])
    _draw_feature_traces(ax4, learner, demo_idx=demo_idx, standardized=True)
    for cut_idx in range(num_cutpoints):
        row = 2 + cut_idx // 2
        col = cut_idx % 2
        ax = fig.add_subplot(gs[row, col])
        _draw_single_cut_scan(ax, learner, demo_idx=demo_idx, vary_index=cut_idx, show_components=True)
    if num_cutpoints % 2 == 1 and scan_rows > 0:
        ax = fig.add_subplot(gs[total_rows - 1, 1])
        ax.axis("off")
    fig.tight_layout(pad=0.34, h_pad=0.48, w_pad=0.34)
    return save_figure(fig, learner_plot_dir(learner) / f"DP_demo_{int(demo_idx):02d}.png", dpi=220)


def plot_map_results_overview(learner, it, *, metrics=None, plot_dir=None, save_name=None):
    if plt is None:
        return None
    fig = plt.figure(figsize=(9.0, 13.2))
    gs = fig.add_gridspec(5, 2, height_ratios=[1.0, 1.18, 1.18, 1.02, 1.15])
    ax1 = fig.add_subplot(gs[0, 0])
    _draw_trajectory(ax1, learner, it, overview=True)
    ax2 = fig.add_subplot(gs[0, 1])
    _draw_learning_curves(ax2, learner)
    ax3 = fig.add_subplot(gs[1, :])
    _draw_map_true_constraints_matrix(ax3, learner, metrics)
    ax4 = fig.add_subplot(gs[2, :])
    _draw_map_learned_constraints_matrix(ax4, learner, metrics)
    ax5 = fig.add_subplot(gs[3, 0])
    _draw_cutpoint_evolution(ax5, learner)
    ax6 = fig.add_subplot(gs[3, 1])
    _draw_parameter_error_matrix(ax6, learner, metrics)
    ax7 = fig.add_subplot(gs[4, :])
    _draw_per_demo_cutpoint_comparison(ax7, learner)
    fig.tight_layout(pad=0.5, h_pad=0.62, w_pad=0.48)
    path = learner_plot_dir(learner, plot_dir=plot_dir) / (str(save_name) if save_name is not None else "summary.png")
    return save_figure(fig, path, dpi=220)


def plot_map_final_outputs(model, it: int) -> None:
    out_dir = learner_plot_dir(model)
    paper_dir = out_dir / "paper_figures"
    final_gammas = _hard_gammas_from_stage_ends([len(X) for X in model.demos], model.stage_ends_, model.num_stages)
    metrics = _map_learned_constraint_payload(model, evaluate_model_metrics(model, final_gammas, None))
    try:
        plot_learned_constraints_matrix_paper(metrics, save_path=paper_dir / "paper_map_learned_constraints.png", dataset_name=str(getattr(getattr(model, "env", None), "eval_tag", "")))
        plot_true_constraints_matrix_paper(metrics, save_path=paper_dir / "paper_map_true_constraint_active.png", dataset_name=str(getattr(getattr(model, "env", None), "eval_tag", "")))
        plot_true_vs_learned_constraints_matrix_paper(metrics, save_path=paper_dir / "paper_map_true_vs_learned_constraints.png", dataset_name=str(getattr(getattr(model, "env", None), "eval_tag", "")))
    except Exception as exc:
        if getattr(model, "verbose", False):
            print(f"[MAP] constraint matrix plot skipped: {exc}")
    plot_map_shared_modes_paper(model, save_path=paper_dir / "paper_map_shared_modes.png")
    for demo_idx in range(len(model.demos)):
        plot_map_true_cutpoint_trajectory_paper(model, demo_idx=demo_idx, save_path=paper_dir / f"paper_map_trajectory_demo_{int(demo_idx):02d}.png")
        plot_map_key_feature_traces_paper(model, demo_idx=demo_idx, save_path=paper_dir / f"paper_map_key_feature_traces_demo_{int(demo_idx):02d}.png")
        plot_map_mode_costs_paper(model, demo_idx=demo_idx, save_path=paper_dir / f"paper_map_mode_costs_demo_{int(demo_idx):02d}.png")
