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
    from matplotlib.patches import Patch
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visualization.io import save_figure


METHOD_PRIORITY = ["swcl", "fchmm", "arhsmm", "cluster"]
METHOD_DISPLAY_LABELS = {
    "swcl": "SWCL (Ours)",
    "fchmm": "FCHMM",
    "arhsmm": "ARHSMM",
    "cluster": "Cluster",
}
METHOD_COLORS = {
    "swcl": "#4C78A8",
    "fchmm": "#F58518",
    "arhsmm": "#54A24B",
    "cluster": "#E45756",
}
DATASET_DISPLAY_LABELS = {
    "S3ObsAvoid": "S3ObsAvoid",
    "S4SlideInsert": "S4SlideInsert",
    "S5SphereInspect": "S5SphereInspect",
}
METRIC_SPECS = [
    ("MeanAbsCutpointError", "Cutpoint Error"),
    ("MeanConstraintError", "Constraint Error"),
]
MODEL_OBJECTIVE_SPECS = {
    "swcl": ("ModelObjectiveFinal", "min"),
    "cluster": ("ModelObjectiveFinal", "min"),
    "arhsmm": ("ModelObjectiveFinal", "max"),
    "fchmm": ("ModelObjectiveFinal", "max"),
    "hmm": ("ModelObjectiveFinal", "max"),
}
POSTHOC_OBJECTIVE_SPECS = {
    "swcl": ("ModelObjectiveFinal", "min"),
    "cluster": ("PosthocObjectiveFinal", "max"),
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


def _scalar_values(rows: list[dict], dataset: str, method: str, metric_name: str) -> np.ndarray:
    vals: list[float] = []
    for row in rows:
        if str(row.get("dataset", "")) != dataset or str(row.get("method", "")) != method:
            continue
        value = row.get("metrics", {}).get(metric_name, np.nan)
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
            objective_key, direction = objective_specs.get(method, ("ModelObjectiveFinal", "max"))
            finite_candidates: list[tuple[float, int, dict]] = []
            for row in candidates:
                objective_value = _finite_objective_value(row, objective_key)
                if objective_value is None:
                    continue
                finite_candidates.append((objective_value, int(row.get("method_seed", 0)), row))
            if not finite_candidates:
                selected.extend(candidates)
                continue
            reverse = direction == "max"
            finite_candidates.sort(key=lambda item: (item[0], -item[1]), reverse=reverse)
            selected.append(finite_candidates[0][2])
    return selected


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


def _active_constraint_coords(rows: list[dict], dataset: str) -> tuple[list[str], list[tuple[int, int]]]:
    for row in rows:
        if str(row.get("dataset", "")) != dataset:
            continue
        metrics = row.get("metrics", {})
        matrix = np.asarray(metrics.get("ConstraintErrorMatrix", []), dtype=float)
        feature_names = list(metrics.get("ConstraintFeatureNames", []))
        if matrix.ndim != 2 or matrix.size == 0 or not feature_names:
            continue
        labels: list[str] = []
        coords: list[tuple[int, int]] = []
        for stage_idx in range(matrix.shape[0]):
            for feat_idx in range(matrix.shape[1]):
                if np.isfinite(matrix[stage_idx, feat_idx]):
                    labels.append(f"s{stage_idx + 1}:{feature_names[feat_idx]}")
                    coords.append((stage_idx, feat_idx))
        return labels, coords
    return [], []


def _constraint_matrix_for_dataset(rows: list[dict], dataset: str, methods: list[str]) -> tuple[np.ndarray, list[str]]:
    labels, coords = _active_constraint_coords(rows, dataset)
    matrix = np.full((len(methods), len(coords)), np.nan, dtype=float)
    for mi, method in enumerate(methods):
        for ci, (stage_idx, feat_idx) in enumerate(coords):
            vals: list[float] = []
            for row in rows:
                if str(row.get("dataset", "")) != dataset or str(row.get("method", "")) != method:
                    continue
                err = np.asarray(row.get("metrics", {}).get("ConstraintErrorMatrix", []), dtype=float)
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

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 2.62), squeeze=False, constrained_layout=False)
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


def plot_constraint_error_matrix_overview(summary: dict, save_path: Path) -> Path | None:
    if plt is None:
        return None

    rows = list(summary.get("results", []))
    methods = _ordered_methods(rows)
    datasets = _ordered_datasets(rows)
    if not rows or not methods or not datasets:
        return None

    ncols = len(datasets)
    fig, axes = plt.subplots(1, ncols, figsize=(4.3 * ncols, 2.88), squeeze=False, constrained_layout=False)
    axes_flat = axes.ravel()
    vmax = 0.0
    matrices: list[np.ndarray] = []
    labels_per_dataset: list[list[str]] = []
    for dataset in datasets:
        matrix, labels = _constraint_matrix_for_dataset(rows, dataset, methods)
        matrices.append(matrix)
        labels_per_dataset.append(labels)
        finite = matrix[np.isfinite(matrix)]
        if finite.size > 0:
            vmax = max(vmax, float(np.max(finite)))
    vmax = max(vmax, 1e-6)

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

    cbar = fig.colorbar(im, ax=axes_flat.tolist(), fraction=0.018, pad=0.015)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Normalized constraint error", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.95, top=0.90, bottom=0.28, wspace=0.35)
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

    fig = plt.figure(figsize=(3.35, 2.95), constrained_layout=False)
    outer = fig.add_gridspec(2, 1, hspace=0.18)

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
        help="Optional output path for the constraint error matrix figure. Defaults to <input stem>_constraint_matrix.png",
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

    matrix_output = Path(args.matrix_output) if args.matrix_output else input_path.with_name(
        f"{input_path.stem}_constraint_matrix.png"
    )
    if not matrix_output.is_absolute():
        matrix_output = PROJECT_ROOT / matrix_output
    saved_matrix = plot_constraint_error_matrix_overview(summary, matrix_output)
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
