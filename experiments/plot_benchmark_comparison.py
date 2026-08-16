from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.io import save_figure


METHOD_PRIORITY = [
    "swcl",
    "map",
    "map_pooled",
    "map_balanced_pooled",
    "map_balanced_vote",
    "gmmhmm",
    "fchmm",
    "arhsmm",
    "cluster",
    "changeforest",
]
METHOD_DISPLAY_LABELS = {
    "swcl": "SWCL (Ours)",
    "map": "MAP",
    "map_pooled": "MAP-Pooled",
    "map_balanced_pooled": "MAP-Balanced",
    "map_balanced_vote": "MAP-Balanced-Vote",
    "gmmhmm": "GMM-HMM",
    "fchmm": "FCHMM",
    "arhsmm": "ARHSMM",
    "cluster": "Cluster",
    "changeforest": "changeforest",
}
METHOD_COLORS = {
    "swcl": "#4C78A8",
    "map": "#B279A2",
    "map_pooled": "#9D755D",
    "map_balanced_pooled": "#E6AB02",
    "map_balanced_vote": "#7570B3",
    "gmmhmm": "#72B7B2",
    "fchmm": "#F58518",
    "arhsmm": "#54A24B",
    "cluster": "#E45756",
    "changeforest": "#FF9DA6",
}
DATASET_DISPLAY_LABELS = {
    "S3ObsAvoid": "S3ObsAvoid",
    "S4SlideInsert": "S4SlideInsert",
    "S5SphereInspect": "S5SphereInspect",
}
METRIC_SPECS = [
    ("MeanAbsCutpointError", "Cutpoint Error"),
    ("SemanticConstraintF1", "Constraint F1"),
    ("MeanParameterError", "Parameter Error"),
]
METRIC_ALIASES = {
    "MeanParameterError": ("MeanParameterError", "MeanConstraintError"),
    "MeanParameterErrorRaw": ("MeanParameterErrorRaw", "MeanConstraintErrorRaw"),
    "ParameterErrorMatrix": ("ParameterErrorMatrix", "ConstraintErrorMatrix"),
    "ParameterErrorMatrixRaw": ("ParameterErrorMatrixRaw", "ConstraintErrorMatrixRaw"),
}
MODEL_OBJECTIVE_SPECS = {
    "swcl": ("ModelObjectiveFinal", "min"),
    "map": ("ModelObjectiveFinal", "min"),
    "map_pooled": ("ModelObjectiveFinal", "min"),
    "map_balanced_pooled": ("ModelObjectiveFinal", "min"),
    "map_balanced_vote": ("ModelObjectiveFinal", "min"),
    "cluster": ("ModelObjectiveFinal", "min"),
    "gmmhmm": ("ModelObjectiveFinal", "max"),
    "changeforest": ("ModelObjectiveFinal", "max"),
    "arhsmm": ("ModelObjectiveFinal", "max"),
    "fchmm": ("ModelObjectiveFinal", "max"),
    "hmm": ("ModelObjectiveFinal", "max"),
}
POSTHOC_OBJECTIVE_SPECS = {
    "swcl": ("ModelObjectiveFinal", "min"),
    "map": ("ModelObjectiveFinal", "min"),
    "map_pooled": ("ModelObjectiveFinal", "min"),
    "map_balanced_pooled": ("ModelObjectiveFinal", "min"),
    "map_balanced_vote": ("ModelObjectiveFinal", "min"),
    "cluster": ("PosthocObjectiveFinal", "max"),
    "gmmhmm": ("PosthocObjectiveFinal", "max"),
    "changeforest": ("PosthocObjectiveFinal", "max"),
    "arhsmm": ("PosthocObjectiveFinal", "max"),
    "fchmm": ("ModelObjectiveFinal", "max"),
    "hmm": ("PosthocObjectiveFinal", "max"),
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _method_sort_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_PRIORITY.index(method), method)
    except ValueError:
        return (len(METHOD_PRIORITY), method)


def _ordered_methods(rows: list[dict]) -> list[str]:
    methods = sorted({str(row.get("method", "")) for row in rows if str(row.get("method", ""))}, key=_method_sort_key)
    return methods


def _ordered_datasets(rows: list[dict]) -> list[str]:
    return sorted({str(row.get("dataset", "")) for row in rows if str(row.get("dataset", ""))})


def _metric_payload(metrics: dict, metric_name: str, default):
    for key in METRIC_ALIASES.get(metric_name, (metric_name,)):
        if key in metrics:
            return metrics[key]
    return default


def _scalar_values(rows: list[dict], dataset: str, method: str, metric_name: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        if str(row.get("dataset", "")) != dataset or str(row.get("method", "")) != method:
            continue
        value = _metric_payload(row.get("metrics", {}), metric_name, np.nan)
        if np.isscalar(value):
            value_f = float(value)
            if np.isfinite(value_f):
                vals.append(value_f)
    return np.asarray(vals, dtype=float)


def _mean_std(samples: dict[str, dict[str, np.ndarray]], datasets: list[str], methods: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = np.full((len(methods), len(datasets)), np.nan, dtype=float)
    stds = np.full((len(methods), len(datasets)), np.nan, dtype=float)
    for mi, method in enumerate(methods):
        for di, dataset in enumerate(datasets):
            vals = np.asarray(samples[dataset][method], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            means[mi, di] = float(np.mean(vals))
            stds[mi, di] = float(np.std(vals, ddof=0))
    return means, stds


def _build_scalar_samples(rows: list[dict], datasets: list[str], methods: list[str], metric_name: str) -> dict[str, dict[str, np.ndarray]]:
    return {
        dataset: {
            method: _scalar_values(rows, dataset, method, metric_name)
            for method in methods
        }
        for dataset in datasets
    }


def _finite_objective_value(row: dict, key: str) -> float | None:
    value = row.get("objectives", {}).get(key, np.nan)
    if not np.isscalar(value):
        return None
    value_f = float(value)
    if not np.isfinite(value_f):
        return None
    return value_f


def _best_objective_row(
    candidates: list[dict],
    method: str,
    objective_specs: dict[str, tuple[str, str]],
) -> dict | None:
    objective_key, direction = objective_specs.get(method, ("ModelObjectiveFinal", "max"))
    finite_candidates: list[tuple[float, int, dict]] = []
    for row in candidates:
        objective_value = _finite_objective_value(row, objective_key)
        if objective_value is None:
            continue
        finite_candidates.append((objective_value, int(row.get("method_seed", 0)), row))
    if not finite_candidates:
        return None
    if direction == "min":
        return min(finite_candidates, key=lambda item: (item[0], item[1]))[2]
    return max(finite_candidates, key=lambda item: (item[0], -item[1]))[2]


def _select_best_seed_rows(
    rows: list[dict],
    objective_specs: dict[str, tuple[str, str]],
    *,
    keep_aggregated_methods: set[str] | None = None,
) -> list[dict]:
    keep_aggregated_methods = keep_aggregated_methods or set()
    selected: list[dict] = []
    for dataset in _ordered_datasets(rows):
        for method in _ordered_methods(rows):
            candidates = [
                row
                for row in rows
                if str(row.get("dataset", "")) == dataset and str(row.get("method", "")) == method
            ]
            if not candidates:
                continue
            if method in keep_aggregated_methods:
                selected.extend(candidates)
                continue
            best_row = _best_objective_row(candidates, method, objective_specs)
            if best_row is None:
                selected.extend(candidates)
                continue
            selected.append(best_row)
    return selected


def _best_objective_metric_values(
    rows: list[dict],
    datasets: list[str],
    methods: list[str],
    metric_name: str,
    objective_specs: dict[str, tuple[str, str]],
) -> np.ndarray:
    values = np.full((len(methods), len(datasets)), np.nan, dtype=float)
    for mi, method in enumerate(methods):
        for di, dataset in enumerate(datasets):
            candidates = [
                row
                for row in rows
                if str(row.get("dataset", "")) == dataset and str(row.get("method", "")) == method
            ]
            best_row = _best_objective_row(candidates, method, objective_specs)
            if best_row is None:
                continue
            value = _metric_payload(best_row.get("metrics", {}), metric_name, np.nan)
            if np.isscalar(value) and np.isfinite(float(value)):
                values[mi, di] = float(value)
    return values


def _grouped_metric_stats(
    rows: list[dict],
    datasets: list[str],
    methods: list[str],
    metric_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    samples = _build_scalar_samples(rows, datasets, methods, metric_name)
    means, stds = _mean_std(samples, datasets, methods)
    x = np.arange(len(datasets), dtype=float)
    width = 0.76 / max(len(methods), 1)
    return means, stds, x, width


def _draw_grouped_metric_bars(
    ax,
    *,
    datasets: list[str],
    methods: list[str],
    ylabel: str,
    means: np.ndarray,
    stds: np.ndarray,
    x: np.ndarray,
    width: float,
    show_xticklabels: bool = True,
    show_errorbars: bool = True,
) -> None:
    for mi, method in enumerate(methods):
        pos = x + (mi - (len(methods) - 1) / 2.0) * width
        color = METHOD_COLORS.get(method, "#999999")
        ax.bar(
            pos,
            means[mi],
            width=width * 0.86,
            color=color,
            alpha=0.82,
            edgecolor=color,
            linewidth=0.8,
            zorder=2,
            label=METHOD_DISPLAY_LABELS.get(method, method),
        )
        if show_errorbars:
            lower_err = np.minimum(stds[mi], np.maximum(means[mi], 0.0))
            upper_err = stds[mi]
            ax.errorbar(
                pos,
                means[mi],
                yerr=np.vstack([lower_err, upper_err]),
                fmt="none",
                ecolor="#1F2937",
                elinewidth=0.8,
                capsize=1.5,
                capthick=0.8,
                zorder=3,
            )
    for idx in range(len(datasets) - 1):
        ax.axvline(float(idx) + 0.5, color="#6B7280", linestyle="-", linewidth=0.8, alpha=0.28, zorder=0)

    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels([DATASET_DISPLAY_LABELS.get(ds, ds) for ds in datasets], rotation=0, ha="center")
        ax.tick_params(axis="x", labelsize=7, pad=0.0)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=7, pad=1.0)
    ax.set_ylabel(ylabel, fontsize=8, labelpad=2.0)
    ax.grid(axis="y", alpha=0.18, linewidth=0.7, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def _draw_best_objective_markers(
    ax,
    *,
    methods: list[str],
    x: np.ndarray,
    width: float,
    values: np.ndarray,
) -> None:
    for mi, _method in enumerate(methods):
        pos = x + (mi - (len(methods) - 1) / 2.0) * width
        finite = np.isfinite(values[mi])
        if not np.any(finite):
            continue
        ax.scatter(
            pos[finite],
            values[mi, finite],
            marker="D",
            s=9,
            facecolor="white",
            edgecolor="#111827",
            linewidth=0.65,
            zorder=5,
            clip_on=False,
        )


def _broken_axis_limits(means: np.ndarray, stds: np.ndarray) -> tuple[float, float, float] | None:
    finite = (means + stds)[np.isfinite(means + stds)]
    if finite.size < 2:
        return None
    finite = np.sort(finite.astype(float))
    max_val = float(finite[-1])
    second_max = float(finite[-2])
    if max_val <= 0.0:
        return None
    if max_val <= 2.5 * max(second_max, 1e-9):
        return None
    lower_top = max(second_max * 1.12, 1e-6)
    upper_bottom = max(lower_top * 1.18, max_val * 0.78)
    upper_top = max_val * 1.08
    if upper_bottom >= upper_top:
        return None
    return lower_top, upper_bottom, upper_top


def _plot_grouped_metric_bar(
    ax,
    *,
    datasets: list[str],
    methods: list[str],
    metric_name: str,
    ylabel: str,
    rows: list[dict],
) -> None:
    means, stds, x, width = _grouped_metric_stats(rows, datasets, methods, metric_name)
    _draw_grouped_metric_bars(
        ax,
        datasets=datasets,
        methods=methods,
        ylabel=ylabel,
        means=means,
        stds=stds,
        x=x,
        width=width,
        show_xticklabels=True,
    )


def _active_parameter_coords(rows: list[dict], dataset: str) -> tuple[list[str], list[tuple[int, int]]]:
    for row in rows:
        if str(row.get("dataset", "")) != dataset:
            continue
        metrics = row.get("metrics", {})
        matrix = np.asarray(_metric_payload(metrics, "ParameterErrorMatrix", []), dtype=float)
        true_active = np.asarray(metrics.get("ConstraintTrueActiveMask", []), dtype=bool)
        feature_names = list(metrics.get("ConstraintFeatureNames", []))
        if matrix.ndim != 2 or matrix.size == 0 or not feature_names:
            continue
        active_mask = true_active if true_active.shape == matrix.shape else np.isfinite(matrix)
        labels: list[str] = []
        coords: list[tuple[int, int]] = []
        for stage_idx in range(matrix.shape[0]):
            for feat_idx in range(matrix.shape[1]):
                if active_mask[stage_idx, feat_idx]:
                    labels.append(f"s{stage_idx + 1}:{feature_names[feat_idx]}")
                    coords.append((stage_idx, feat_idx))
        return labels, coords
    return [], []


def _parameter_matrix_for_dataset(rows: list[dict], dataset: str, methods: list[str]) -> tuple[np.ndarray, list[str]]:
    labels, coords = _active_parameter_coords(rows, dataset)
    matrix = np.full((len(methods), len(coords)), np.nan, dtype=float)
    for mi, method in enumerate(methods):
        for ci, (stage_idx, feat_idx) in enumerate(coords):
            vals: list[float] = []
            for row in rows:
                if str(row.get("dataset", "")) != dataset or str(row.get("method", "")) != method:
                    continue
                err = np.asarray(
                    _metric_payload(row.get("metrics", {}), "ParameterErrorMatrix", []),
                    dtype=float,
                )
                if err.ndim != 2:
                    continue
                if stage_idx >= err.shape[0] or feat_idx >= err.shape[1]:
                    continue
                value = err[stage_idx, feat_idx]
                if np.isfinite(value):
                    vals.append(float(value))
            if vals:
                matrix[mi, ci] = float(np.mean(vals))
    return matrix, labels


def plot_benchmark_comparison(summary: dict, save_path: Path) -> Path | None:
    if plt is None:
        return None

    rows = list(summary.get("results", []))
    methods = _ordered_methods(rows)
    datasets = _ordered_datasets(rows)
    if not rows or not methods or not datasets:
        return None

    n_metrics = len(METRIC_SPECS)
    fig_width = max(3.35, 0.72 * len(methods) + 1.3)
    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=(fig_width, 1.22 * n_metrics + 0.25),
        squeeze=False,
        constrained_layout=False,
    )
    axes_flat = axes.ravel()

    for ax, (metric_name, ylabel) in zip(axes_flat, METRIC_SPECS):
        _plot_grouped_metric_bar(
            ax,
            datasets=datasets,
            methods=methods,
            metric_name=metric_name,
            ylabel=ylabel,
            rows=rows,
        )
        if metric_name == "SemanticConstraintF1":
            ax.set_ylim(0.0, 1.05)

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS.get(method, "#999999"),
            edgecolor=METHOD_COLORS.get(method, "#999999"),
            alpha=0.82,
            label=METHOD_DISPLAY_LABELS.get(method, method),
        )
        for method in methods
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=max(len(methods), 1),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.028),
        fontsize=6.4,
        columnspacing=0.7,
        handletextpad=0.35,
        handlelength=1.0,
    )

    fig.tight_layout(rect=(0.015, 0.06, 0.995, 0.995), pad=0.05, h_pad=0.34)
    return save_figure(fig, save_path, dpi=300)


def plot_benchmark_comparison_with_best_objective(
    summary: dict,
    save_path: Path,
    *,
    objective_specs: dict[str, tuple[str, str]],
) -> Path | None:
    if plt is None:
        return None

    rows = list(summary.get("results", []))
    methods = _ordered_methods(rows)
    datasets = _ordered_datasets(rows)
    if not rows or not methods or not datasets:
        return None

    best_values_by_metric = {
        metric_name: _best_objective_metric_values(
            rows,
            datasets,
            methods,
            metric_name,
            objective_specs,
        )
        for metric_name, _ylabel in METRIC_SPECS
    }
    if not any(np.any(np.isfinite(values)) for values in best_values_by_metric.values()):
        return None

    n_metrics = len(METRIC_SPECS)
    fig_width = max(3.35, 0.72 * len(methods) + 1.3)
    fig, axes = plt.subplots(
        n_metrics,
        1,
        figsize=(fig_width, 1.22 * n_metrics + 0.25),
        squeeze=False,
        constrained_layout=False,
    )
    axes_flat = axes.ravel()

    for ax, (metric_name, ylabel) in zip(axes_flat, METRIC_SPECS):
        means, stds, x, width = _grouped_metric_stats(rows, datasets, methods, metric_name)
        _draw_grouped_metric_bars(
            ax,
            datasets=datasets,
            methods=methods,
            ylabel=ylabel,
            means=means,
            stds=stds,
            x=x,
            width=width,
            show_xticklabels=True,
        )
        _draw_best_objective_markers(
            ax,
            methods=methods,
            x=x,
            width=width,
            values=best_values_by_metric[metric_name],
        )
        if metric_name == "SemanticConstraintF1":
            ax.set_ylim(0.0, 1.05)

    method_handles = [
        Patch(
            facecolor=METHOD_COLORS.get(method, "#999999"),
            edgecolor=METHOD_COLORS.get(method, "#999999"),
            alpha=0.82,
            label=METHOD_DISPLAY_LABELS.get(method, method),
        )
        for method in methods
    ]
    fig.legend(
        handles=method_handles,
        frameon=False,
        ncol=max(len(methods), 1),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.028),
        fontsize=6.4,
        columnspacing=0.7,
        handletextpad=0.35,
        handlelength=1.0,
    )
    axes_flat[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markersize=3.2,
                markerfacecolor="white",
                markeredgecolor="#111827",
                markeredgewidth=0.65,
                label="Selected by model objective",
            )
        ],
        loc="upper right",
        frameon=False,
        fontsize=6.4,
        borderaxespad=0.35,
        handletextpad=0.35,
    )

    fig.tight_layout(rect=(0.015, 0.06, 0.995, 0.995), pad=0.05, h_pad=0.34)
    return save_figure(fig, save_path, dpi=300)


def _metric_mean_std(rows: list[dict], metric_name: str) -> tuple[float, float]:
    values: list[float] = []
    for row in rows:
        value = _metric_payload(row.get("metrics", {}), metric_name, np.nan)
        if np.isscalar(value) and np.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return float("nan"), float("nan")
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=0))


def _semantic_match_coverage(rows: list[dict]) -> float:
    matched = 0.0
    true_count = 0.0
    for row in rows:
        metrics = row.get("metrics", {})
        matched_value = metrics.get("SemanticConstraintMatchCount", 0.0)
        true_value = metrics.get("TrueConstraintCount", 0.0)
        if np.isscalar(matched_value) and np.isfinite(float(matched_value)):
            matched += float(matched_value)
        if np.isscalar(true_value) and np.isfinite(float(true_value)):
            true_count += float(true_value)
    if true_count <= 0.0:
        return 0.0
    return float(np.clip(matched / true_count, 0.0, 1.0))


def _coverage_marker_size(coverage: float) -> float:
    return 26.0 + 180.0 * float(np.clip(coverage, 0.0, 1.0))


def plot_joint_constraint_comparison(
    summary: dict,
    save_path: Path,
    *,
    objective_specs: dict[str, tuple[str, str]],
) -> Path | None:
    if plt is None:
        return None

    rows = list(summary.get("results", []))
    methods = _ordered_methods(rows)
    datasets = _ordered_datasets(rows)
    if not rows or not methods or not datasets:
        return None

    parameter_upper_candidates: list[float] = []
    for dataset in datasets:
        for method in methods:
            candidates = [
                row
                for row in rows
                if str(row.get("dataset", "")) == dataset and str(row.get("method", "")) == method
            ]
            parameter_mean, parameter_std = _metric_mean_std(candidates, "MeanParameterError")
            if np.isfinite(parameter_mean):
                parameter_upper_candidates.append(parameter_mean + max(parameter_std, 0.0))
            best_row = _best_objective_row(candidates, method, objective_specs)
            if best_row is not None:
                best_parameter = _metric_payload(best_row.get("metrics", {}), "MeanParameterError", np.nan)
                if np.isscalar(best_parameter) and np.isfinite(float(best_parameter)):
                    parameter_upper_candidates.append(float(best_parameter))

    finite_parameter_top = max(parameter_upper_candidates, default=0.05)
    parameter_data_top = max(finite_parameter_top * 1.14, 0.02)
    no_match_y = parameter_data_top * 1.10
    parameter_axis_top = parameter_data_top * 1.20

    fig_width = max(8.0, 1.25 * len(methods) + 2.2)
    fig = plt.figure(figsize=(fig_width, 5.35), constrained_layout=False)
    grid = fig.add_gridspec(
        2,
        len(datasets),
        height_ratios=[0.92, 1.28],
        hspace=0.42,
        wspace=0.24,
    )

    cutpoint_ax = fig.add_subplot(grid[0, :])
    cutpoint_means, cutpoint_stds, cutpoint_x, cutpoint_width = _grouped_metric_stats(
        rows,
        datasets,
        methods,
        "MeanAbsCutpointError",
    )
    _draw_grouped_metric_bars(
        cutpoint_ax,
        datasets=datasets,
        methods=methods,
        ylabel="Cutpoint Error",
        means=cutpoint_means,
        stds=cutpoint_stds,
        x=cutpoint_x,
        width=cutpoint_width,
        show_xticklabels=True,
    )
    cutpoint_ax.tick_params(axis="both", labelsize=9)
    cutpoint_ax.set_ylabel("Cutpoint Error", fontsize=10)
    best_cutpoints = _best_objective_metric_values(
        rows,
        datasets,
        methods,
        "MeanAbsCutpointError",
        objective_specs,
    )
    _draw_best_objective_markers(
        cutpoint_ax,
        methods=methods,
        x=cutpoint_x,
        width=cutpoint_width,
        values=best_cutpoints,
    )
    cutpoint_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="D",
                linestyle="none",
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor="#111827",
                markeredgewidth=0.75,
                label="Selected by model objective",
            )
        ],
        loc="upper right",
        frameon=False,
        fontsize=8,
        borderaxespad=0.35,
        handletextpad=0.4,
    )

    scatter_axes = []
    for dataset_idx, dataset in enumerate(datasets):
        scatter_ax = fig.add_subplot(grid[1, dataset_idx], sharey=scatter_axes[0] if scatter_axes else None)
        scatter_axes.append(scatter_ax)
        scatter_ax.axhspan(
            parameter_data_top,
            parameter_axis_top,
            color="#E5E7EB",
            alpha=0.65,
            zorder=0,
        )
        scatter_ax.text(
            0.98,
            no_match_y,
            "no parameter match",
            ha="right",
            va="center",
            fontsize=6.5,
            color="#4B5563",
        )

        for method in methods:
            candidates = [
                row
                for row in rows
                if str(row.get("dataset", "")) == dataset and str(row.get("method", "")) == method
            ]
            if not candidates:
                continue
            f1_mean, f1_std = _metric_mean_std(candidates, "SemanticConstraintF1")
            parameter_mean, parameter_std = _metric_mean_std(candidates, "MeanParameterError")
            if not np.isfinite(f1_mean):
                continue
            coverage = _semantic_match_coverage(candidates)
            color = METHOD_COLORS.get(method, "#999999")

            if np.isfinite(parameter_mean):
                scatter_ax.errorbar(
                    f1_mean,
                    parameter_mean,
                    xerr=[[min(max(f1_std, 0.0), f1_mean)], [min(max(f1_std, 0.0), 1.0 - f1_mean)]],
                    yerr=[[min(max(parameter_std, 0.0), parameter_mean)], [max(parameter_std, 0.0)]],
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=2.0,
                    alpha=0.72,
                    zorder=2,
                )
                scatter_ax.scatter(
                    [f1_mean],
                    [parameter_mean],
                    s=_coverage_marker_size(coverage),
                    marker="o",
                    color=color,
                    edgecolor="white",
                    linewidth=0.9,
                    alpha=0.88,
                    zorder=3,
                )
            else:
                scatter_ax.scatter(
                    [f1_mean],
                    [no_match_y],
                    s=42,
                    marker="x",
                    color=color,
                    linewidth=1.4,
                    zorder=4,
                )

            best_row = _best_objective_row(candidates, method, objective_specs)
            if best_row is None:
                continue
            best_f1 = _metric_payload(best_row.get("metrics", {}), "SemanticConstraintF1", np.nan)
            best_parameter = _metric_payload(best_row.get("metrics", {}), "MeanParameterError", np.nan)
            if not np.isscalar(best_f1) or not np.isfinite(float(best_f1)):
                continue
            best_y = float(best_parameter) if np.isscalar(best_parameter) and np.isfinite(float(best_parameter)) else no_match_y
            scatter_ax.scatter(
                [float(best_f1)],
                [best_y],
                s=34,
                marker="D",
                facecolor="white",
                edgecolor=color,
                linewidth=1.1,
                zorder=5,
                clip_on=False,
            )

        scatter_ax.set_xlim(-0.03, 1.03)
        scatter_ax.set_ylim(0.0, parameter_axis_top)
        scatter_ax.set_xlabel("Constraint F1", fontsize=9)
        scatter_ax.set_title(DATASET_DISPLAY_LABELS.get(dataset, dataset), fontsize=9.5, pad=4.0)
        scatter_ax.tick_params(axis="both", labelsize=8)
        scatter_ax.grid(alpha=0.18, linewidth=0.7, zorder=0)
        scatter_ax.spines["top"].set_visible(False)
        scatter_ax.spines["right"].set_visible(False)
        if dataset_idx == 0:
            scatter_ax.set_ylabel("Parameter Error", fontsize=9)
        else:
            scatter_ax.tick_params(axis="y", labelleft=False)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=5.5,
            markerfacecolor=METHOD_COLORS.get(method, "#999999"),
            markeredgecolor="white",
            label=METHOD_DISPLAY_LABELS.get(method, method),
        )
        for method in methods
    ]
    fig.legend(
        handles=method_handles,
        frameon=False,
        ncol=max(len(methods), 1),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        fontsize=8,
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.text(
        0.99,
        0.052,
        "Circle area = exact semantic-match coverage; × = no matched parameter",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.075, right=0.99, top=0.98, bottom=0.135)
    return save_figure(fig, save_path, dpi=300)


def plot_parameter_error_matrix_overview(summary: dict, save_path: Path) -> Path | None:
    if plt is None:
        return None

    rows = list(summary.get("results", []))
    methods = _ordered_methods(rows)
    datasets = _ordered_datasets(rows)
    if not rows or not methods or not datasets:
        return None

    ncols = len(datasets)
    fig_width = max(5.35, 4.35 * ncols)
    fig_height = 3.25
    fig, axes = plt.subplots(1, ncols, figsize=(fig_width, fig_height), squeeze=False, constrained_layout=False)
    axes_flat = axes.ravel()
    vmax = 0.0
    matrices: list[np.ndarray] = []
    labels_per_dataset: list[list[str]] = []
    for dataset in datasets:
        matrix, labels = _parameter_matrix_for_dataset(rows, dataset, methods)
        matrices.append(matrix)
        labels_per_dataset.append(labels)
        finite = matrix[np.isfinite(matrix)]
        if finite.size > 0:
            vmax = max(vmax, float(np.max(finite)))
    vmax = max(vmax, 1e-6)

    im = None
    for ax, dataset, matrix, labels in zip(axes_flat, datasets, matrices, labels_per_dataset):
        if matrix.size == 0 or matrix.shape[1] == 0:
            ax.axis("off")
            continue
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=vmax)
        ax.set_title(DATASET_DISPLAY_LABELS.get(dataset, dataset), fontsize=10, pad=6)
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([METHOD_DISPLAY_LABELS.get(method, method) for method in methods], fontsize=8)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                value = matrix[row_idx, col_idx]
                if not np.isfinite(value):
                    continue
                text_color = "white" if value >= 0.55 * vmax else "black"
                ax.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=text_color)

    left_margin = 0.95 / fig_width
    right_margin = 1.0 - (0.85 / fig_width)
    bottom_margin = 1.05 / fig_height
    top_margin = 1.0 - (0.34 / fig_height)
    fig.subplots_adjust(
        left=left_margin,
        right=right_margin,
        top=top_margin,
        bottom=bottom_margin,
        wspace=0.35,
    )
    if im is not None:
        colorbar_ax = fig.add_axes(
            [
                1.0 - (0.58 / fig_width),
                bottom_margin,
                0.11 / fig_width,
                top_margin - bottom_margin,
            ]
        )
        cbar = fig.colorbar(im, cax=colorbar_ax)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label("Normalized parameter error", fontsize=9)
    return save_figure(fig, save_path, dpi=300)


def plot_best_seed_comparison(
    summary: dict,
    save_path: Path,
    *,
    objective_specs: dict[str, tuple[str, str]],
    swcl_keep_aggregated: bool = True,
    allow_broken_axis: bool = False,
) -> Path | None:
    if plt is None:
        return None

    rows = list(summary.get("results", []))
    if not rows:
        return None
    objective_available = False
    for row in rows:
        method = str(row.get("method", ""))
        if method == "swcl":
            continue
        objective_key = objective_specs.get(method, ("ModelObjectiveFinal", "max"))[0]
        if _finite_objective_value(row, objective_key) is not None:
            objective_available = True
            break
    if not objective_available:
        return None
    selected_rows = _select_best_seed_rows(
        rows,
        objective_specs=objective_specs,
        keep_aggregated_methods={"swcl"} if swcl_keep_aggregated else set(),
    )
    methods = _ordered_methods(selected_rows)
    datasets = _ordered_datasets(selected_rows)
    if not selected_rows or not methods or not datasets:
        return None

    n_metrics = len(METRIC_SPECS)
    fig_width = max(3.35, 0.72 * len(methods) + 1.3)
    fig = plt.figure(figsize=(fig_width, 1.32 * n_metrics + 0.3), constrained_layout=False)
    outer = fig.add_gridspec(n_metrics, 1, hspace=0.18)

    for metric_idx, (metric_name, ylabel) in enumerate(METRIC_SPECS):
        means, stds, x, width = _grouped_metric_stats(selected_rows, datasets, methods, metric_name)
        limits = _broken_axis_limits(means, stds) if allow_broken_axis else None
        if limits is None:
            ax = fig.add_subplot(outer[metric_idx])
            _draw_grouped_metric_bars(
                ax,
                datasets=datasets,
                methods=methods,
                ylabel=ylabel,
                means=means,
                stds=stds,
                x=x,
                width=width,
                show_xticklabels=True,
                show_errorbars=False,
            )
            if metric_name == "SemanticConstraintF1":
                ax.set_ylim(0.0, 1.05)
            continue

        lower_top, upper_bottom, upper_top = limits
        inner = outer[metric_idx].subgridspec(2, 1, height_ratios=[0.66, 2.60], hspace=0.07)
        ax_top = fig.add_subplot(inner[0])
        ax_bottom = fig.add_subplot(inner[1], sharex=ax_top)

        _draw_grouped_metric_bars(
            ax_top,
            datasets=datasets,
            methods=methods,
            ylabel="",
            means=means,
            stds=stds,
            x=x,
            width=width,
            show_xticklabels=False,
            show_errorbars=False,
        )
        _draw_grouped_metric_bars(
            ax_bottom,
            datasets=datasets,
            methods=methods,
            ylabel=ylabel,
            means=means,
            stds=stds,
            x=x,
            width=width,
            show_xticklabels=True,
            show_errorbars=False,
        )
        ax_bottom.set_ylim(0.0, lower_top)
        ax_top.set_ylim(upper_bottom, upper_top)
        ax_top.spines["bottom"].set_visible(False)
        ax_bottom.spines["top"].set_visible(False)
        ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
        ax_top.tick_params(axis="y", labelsize=6.5, pad=1.0)
        ax_bottom.tick_params(axis="y", labelsize=7, pad=1.0)

        d = 0.012
        kwargs = dict(transform=ax_top.transAxes, color="#374151", clip_on=False, linewidth=0.8)
        ax_top.plot((-d, +d), (-d, +d), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
        kwargs = dict(transform=ax_bottom.transAxes, color="#374151", clip_on=False, linewidth=0.8)
        ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
        ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    legend_handles = [
        Patch(
            facecolor=METHOD_COLORS.get(method, "#999999"),
            edgecolor=METHOD_COLORS.get(method, "#999999"),
            alpha=0.82,
            label=METHOD_DISPLAY_LABELS.get(method, method),
        )
        for method in methods
    ]
    fig.legend(
        handles=legend_handles,
        frameon=False,
        ncol=max(len(methods), 1),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.020),
        fontsize=6.4,
        columnspacing=0.7,
        handletextpad=0.35,
        handlelength=1.0,
    )
    fig.subplots_adjust(left=0.145, right=0.99, top=0.965, bottom=0.115, hspace=0.18)
    return save_figure(fig, save_path, dpi=300)


def main():
    parser = argparse.ArgumentParser(description="Paper-style benchmark comparison plots.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to benchmark_results.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for the grouped bar figure. Defaults to <input stem>_comparison.png",
    )
    parser.add_argument(
        "--matrix-output",
        type=str,
        default=None,
        help="Optional output path for the parameter error matrix figure. Defaults to <input stem>_parameter_matrix.png",
    )
    parser.add_argument(
        "--mean-std-best-model-output",
        type=str,
        default=None,
        help="Optional output path for mean/std bars with best-model-objective markers. Defaults to <input stem>_comparison_with_best_model_objective.png",
    )
    parser.add_argument(
        "--joint-output",
        type=str,
        default=None,
        help="Optional output path for the cutpoint plus joint F1/parameter figure. Defaults to <input stem>_joint_constraint.png",
    )
    parser.add_argument(
        "--best-model-output",
        type=str,
        default=None,
        help="Optional output path for the best-model-objective seed comparison figure. Defaults to <input stem>_best_model_objective.png",
    )
    parser.add_argument(
        "--best-posthoc-output",
        type=str,
        default=None,
        help="Optional output path for the best-posthoc-objective seed comparison figure. Defaults to <input stem>_best_posthoc_objective.png",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path
    summary = _load_json(input_path)

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}_comparison.png")
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    saved = plot_benchmark_comparison(summary, output_path)
    if saved is not None:
        print(f"[Saved] {saved}")

    mean_std_best_model_output = (
        Path(args.mean_std_best_model_output)
        if args.mean_std_best_model_output
        else input_path.with_name(f"{input_path.stem}_comparison_with_best_model_objective.png")
    )
    if not mean_std_best_model_output.is_absolute():
        mean_std_best_model_output = PROJECT_ROOT / mean_std_best_model_output
    saved_mean_std_best_model = plot_benchmark_comparison_with_best_objective(
        summary,
        mean_std_best_model_output,
        objective_specs=MODEL_OBJECTIVE_SPECS,
    )
    if saved_mean_std_best_model is not None:
        print(f"[Saved] {saved_mean_std_best_model}")

    joint_output = Path(args.joint_output) if args.joint_output else input_path.with_name(
        f"{input_path.stem}_joint_constraint.png"
    )
    if not joint_output.is_absolute():
        joint_output = PROJECT_ROOT / joint_output
    saved_joint = plot_joint_constraint_comparison(
        summary,
        joint_output,
        objective_specs=MODEL_OBJECTIVE_SPECS,
    )
    if saved_joint is not None:
        print(f"[Saved] {saved_joint}")

    matrix_output = Path(args.matrix_output) if args.matrix_output else input_path.with_name(
        f"{input_path.stem}_parameter_matrix.png"
    )
    if not matrix_output.is_absolute():
        matrix_output = PROJECT_ROOT / matrix_output
    saved_matrix = plot_parameter_error_matrix_overview(summary, matrix_output)
    if saved_matrix is not None:
        print(f"[Saved] {saved_matrix}")

    best_model_output = Path(args.best_model_output) if args.best_model_output else input_path.with_name(
        f"{input_path.stem}_best_model_objective.png"
    )
    if not best_model_output.is_absolute():
        best_model_output = PROJECT_ROOT / best_model_output
    saved_best_model = plot_best_seed_comparison(
        summary,
        best_model_output,
        objective_specs=MODEL_OBJECTIVE_SPECS,
        allow_broken_axis=True,
    )
    if saved_best_model is not None:
        print(f"[Saved] {saved_best_model}")

    best_posthoc_output = Path(args.best_posthoc_output) if args.best_posthoc_output else input_path.with_name(
        f"{input_path.stem}_best_posthoc_objective.png"
    )
    if not best_posthoc_output.is_absolute():
        best_posthoc_output = PROJECT_ROOT / best_posthoc_output
    saved_best_posthoc = plot_best_seed_comparison(
        summary,
        best_posthoc_output,
        objective_specs=POSTHOC_OBJECTIVE_SPECS,
        allow_broken_axis=True,
    )
    if saved_best_posthoc is not None:
        print(f"[Saved] {saved_best_posthoc}")


if __name__ == "__main__":
    main()
