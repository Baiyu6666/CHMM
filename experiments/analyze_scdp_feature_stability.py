from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.registry import load_env
from experiments.config_loader import deep_merge, load_experiment_config
from methods.wrappers.joint_scdp import JointSCDPMethod
from utils.models import GaussianModel


def _feature_names(learner) -> list[str]:
    names: list[str] = []
    for local_idx in range(learner.num_features):
        selected_col = int(learner.selected_feature_columns[local_idx])
        name = None
        for i, spec in enumerate(learner.raw_feature_specs):
            if int(spec.get("column_idx", i)) == selected_col:
                name = str(spec.get("name", f"f{local_idx}"))
                break
        names.append(name or f"f{local_idx}")
    return names


def _fit_scdp(env_config: str, method_config: str, max_iter: int | None = None):
    cfg = load_experiment_config(env_config, method_config)
    dataset_cfg = dict(cfg["dataset"])
    method_cfg = dict(cfg["method"])
    dataset_name = dataset_cfg.pop("name")
    method_name = method_cfg.pop("name")
    if method_name != "scdp":
        raise ValueError(f"This script only supports scdp, got '{method_name}'.")
    dataset_method_overrides = dict(dataset_cfg.pop("method_overrides", {}))
    method_cfg = deep_merge(method_cfg, dataset_method_overrides.get(method_name, {}))
    method_cfg["plot_every"] = None
    method_cfg["verbose"] = False
    if max_iter is not None:
        method_cfg["max_iter"] = int(max_iter)

    dataset = load_env(dataset_name, **dataset_cfg)
    result = JointSCDPMethod(kwargs=method_cfg).fit(dataset)
    return dataset_name, dataset, result


def _mean_abs_centered_dispersion(values) -> float:
    xs = np.asarray(values, dtype=float).reshape(-1)
    if xs.size == 0:
        return np.nan
    center = float(np.median(xs))
    return float(np.mean(np.abs(xs - center)))


def _plot_demo_feature_costs(
    dataset_name: str,
    learner,
    demo_idx: int,
    output_path: Path,
    equality_dispersion_ratio_threshold: float | None = None,
    stage_ends: list[int] | tuple[int, ...] | None = None,
    stage_params=None,
) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is not installed.")
    if demo_idx < 0 or demo_idx >= len(learner.demos):
        raise IndexError(f"demo_idx {demo_idx} is out of range for {len(learner.demos)} demos.")

    feature_names = _feature_names(learner)
    stage_ends = learner.stage_ends_[demo_idx] if stage_ends is None else stage_ends
    starts = [0, int(stage_ends[0]) + 1]
    ends = [int(stage_ends[0]), int(stage_ends[1])]
    F = learner.standardized_features[demo_idx]

    fig, axes = plt.subplots(
        learner.num_states,
        learner.num_features,
        figsize=(4.6 * learner.num_features, 3.8 * learner.num_states),
        squeeze=False,
    )

    for stage_idx, (s, e) in enumerate(zip(starts, ends)):
        local_params = (
            learner.current_stage_params_per_demo[demo_idx][stage_idx]
            if stage_params is None
            else stage_params[stage_idx]
        )
        for feat_idx, kind in enumerate(learner.feature_model_types):
            ax = axes[stage_idx][feat_idx]
            vals = np.asarray(F[s : e + 1, feat_idx], dtype=float)
            full_vals = np.asarray(F[:, feat_idx], dtype=float)
            fitted_model = learner._vector_to_model(
                kind,
                learner._summary_to_vector(kind, local_params.model_summaries[feat_idx]),
            )
            kind_l = str(kind).lower()
            if kind_l in {"student_t", "studentt", "t", "gauss", "gaussian", "zero_gauss", "zero_gaussian"}:
                baseline_model = GaussianModel(
                    mu=float(np.mean(full_vals)),
                    sigma=float(max(np.std(full_vals), 1e-6)),
                )
            else:
                baseline_model = GaussianModel(
                    mu=float(np.mean(vals)),
                    sigma=float(max(np.std(vals), 1e-6)),
                )

            lo = float(
                min(
                    np.min(vals),
                    np.min(full_vals),
                    getattr(fitted_model, "L", np.min(vals)),
                    getattr(baseline_model, "L", np.min(vals)),
                )
            )
            hi = float(
                max(
                    np.max(vals),
                    np.max(full_vals),
                    getattr(fitted_model, "U", np.max(vals)),
                    getattr(baseline_model, "U", np.max(vals)),
                )
            )
            pad = max(0.15 * (hi - lo + 1e-6), 0.2)
            xs = np.linspace(lo - pad, hi + pad, 300)

            ax.hist(
                full_vals,
                bins=min(max(len(full_vals) // 2, 10), 24),
                density=True,
                alpha=0.18,
                color="tab:gray",
                label="full demo",
            )
            ax.hist(vals, bins=min(max(len(vals) // 2, 6), 16), density=True, alpha=0.35, color="tab:blue", label="demo segment")
            ax.plot(xs, np.exp(fitted_model.logpdf(xs)), color="tab:red", lw=2.0, label="fitted model")
            ax.plot(xs, np.exp(baseline_model.logpdf(xs)), color="tab:green", lw=2.0, linestyle="--", label="demo baseline")
            ax.axvline(float(np.mean(vals)), color="tab:blue", lw=1.0, linestyle=":", alpha=0.8)
            ax.axvline(float(np.mean(full_vals)), color="tab:gray", lw=1.0, linestyle="-.", alpha=0.9)

            fitted_neglog = -np.asarray(fitted_model.logpdf(vals), dtype=float)
            baseline_neglog = -np.asarray(baseline_model.logpdf(vals), dtype=float)
            fitted_step = float(np.mean(fitted_neglog))
            baseline_step = float(np.mean(baseline_neglog))
            stage_dispersion = float(_mean_abs_centered_dispersion(vals))
            info_lines = [
                f"steps = {len(vals)}",
                f"fitted avg NLL = {fitted_step:.3f}",
                f"baseline avg NLL = {baseline_step:.3f}",
                f"avg NLL gain = {baseline_step - fitted_step:.3f}",
            ]
            if kind_l in {"student_t", "studentt", "t"}:
                stage_sigma = float(getattr(fitted_model, "sigma", np.nan))
                info_lines.append(f"student-t sigma = {stage_sigma:.3f}")
                if equality_dispersion_ratio_threshold is not None:
                    uncertainty_bonus = 0.1 / np.sqrt(max(len(vals), 1))
                    info_lines.append(f"local dispersion = {stage_dispersion:.3f}")
                    info_lines.append(f"uncertainty bonus = {uncertainty_bonus:.3f}")
                    info_lines.append(f"adjusted score = {stage_dispersion + uncertainty_bonus:.3f}")
                    info_lines.append(f"ratio threshold = {float(equality_dispersion_ratio_threshold):.3f}")
            elif kind_l in {"gauss", "gaussian"}:
                stage_sigma = float(getattr(fitted_model, "sigma", np.nan))
                info_lines.append(f"gaussian sigma = {stage_sigma:.3f}")
                if equality_dispersion_ratio_threshold is not None:
                    uncertainty_bonus = 0.1 / np.sqrt(max(len(vals), 1))
                    info_lines.append(f"local dispersion = {stage_dispersion:.3f}")
                    info_lines.append(f"uncertainty bonus = {uncertainty_bonus:.3f}")
                    info_lines.append(f"adjusted score = {stage_dispersion + uncertainty_bonus:.3f}")
                    info_lines.append(f"ratio threshold = {float(equality_dispersion_ratio_threshold):.3f}")

            ax.set_title(f"Stage {stage_idx + 1} | {feature_names[feat_idx]}")
            ax.set_xlabel("standardized feature value")
            ax.set_ylabel("density")
            ax.text(
                0.02,
                0.98,
                "\n".join(info_lines),
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
            )
            ax.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle(f"SCDP feature costs | {dataset_name} | demo {demo_idx}", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _shortest_coverage_width(values, coverage: float = 0.8) -> float:
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


def plot_all_demo_feature_costs(dataset_name: str, learner, output_dir: str | Path | None = None) -> list[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is not installed.")
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "analysis" / "scdp_feature_costs" / dataset_name
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for demo_idx in range(len(learner.demos)):
        output_path = output_dir / f"demo_{demo_idx:02d}.png"
        outputs.append(_plot_demo_feature_costs(dataset_name, learner, demo_idx, output_path))
    return outputs


def _analyze_trained_scdp(dataset_name: str, dataset, result) -> dict[str, Any]:
    learner = result["model"]

    feature_names = _feature_names(learner)
    num_stages = learner.num_states
    num_features = learner.num_features

    sum_costs = [[[] for _ in range(num_features)] for _ in range(num_stages)]
    mean_costs = [[[] for _ in range(num_features)] for _ in range(num_stages)]
    stage_lengths = [[] for _ in range(num_stages)]
    per_demo_feature_values = [[] for _ in range(num_features)]
    per_demo_stage_feature_values = [[[] for _ in range(num_features)] for _ in range(num_stages)]
    param_names_per_feature = []
    param_values = [[None for _ in range(num_features)] for _ in range(num_stages)]
    for feat_idx, kind in enumerate(learner.feature_model_types):
        k = str(kind).lower()
        if k in {"gauss", "gaussian", "student_t", "studentt", "t"}:
            names = ["mu", "sigma"]
        elif k in {"zero_gauss", "zero_gaussian"}:
            names = ["sigma"]
        elif k in {"margin_exp_lower", "marginexp", "margin_exp"}:
            names = ["b", "lam"]
        else:
            names = ["param0"]
        param_names_per_feature.append(names)
        for stage_idx in range(num_stages):
            param_values[stage_idx][feat_idx] = {name: [] for name in names}

    for demo_idx, _ in enumerate(learner.demos):
        stage_ends = learner.stage_ends_[demo_idx]
        starts = [0, int(stage_ends[0]) + 1]
        ends = [int(stage_ends[0]), int(stage_ends[1])]
        F = learner.standardized_features[demo_idx]
        for feat_idx in range(num_features):
            per_demo_feature_values[feat_idx].append(np.asarray(F[:, feat_idx], dtype=float))
        for stage_idx, (s, e) in enumerate(zip(starts, ends)):
            stage_lengths[stage_idx].append(int(e - s + 1))
            local_params = learner.current_stage_params_per_demo[demo_idx][stage_idx]
            for feat_idx, kind in enumerate(learner.feature_model_types):
                summary = local_params.model_summaries[feat_idx]
                model = learner._vector_to_model(
                    kind,
                    learner._summary_to_vector(kind, summary),
                )
                vals = F[s : e + 1, feat_idx]
                per_demo_stage_feature_values[stage_idx][feat_idx].append(np.asarray(vals, dtype=float))
                neglog = -np.asarray(model.logpdf(vals), dtype=float)
                sum_costs[stage_idx][feat_idx].append(float(np.sum(neglog)))
                mean_costs[stage_idx][feat_idx].append(float(np.mean(neglog)))
                for name in param_names_per_feature[feat_idx]:
                    param_values[stage_idx][feat_idx][name].append(float(summary[name]))

    sum_mean = np.zeros((num_features, num_stages), dtype=float)
    sum_std = np.zeros((num_features, num_stages), dtype=float)
    mean_mean = np.zeros((num_features, num_stages), dtype=float)
    mean_std = np.zeros((num_features, num_stages), dtype=float)
    baseline_sum_mean = np.zeros((num_features, num_stages), dtype=float)
    baseline_sum_std = np.zeros((num_features, num_stages), dtype=float)
    baseline_step_mean_mean = np.zeros((num_features, num_stages), dtype=float)
    baseline_step_mean_std = np.zeros((num_features, num_stages), dtype=float)
    baseline_beats_fitted_sum_count = np.zeros((num_features, num_stages), dtype=int)
    baseline_beats_fitted_step_count = np.zeros((num_features, num_stages), dtype=int)
    baseline_beats_fitted_sum_frac = np.zeros((num_features, num_stages), dtype=float)
    baseline_beats_fitted_step_frac = np.zeros((num_features, num_stages), dtype=float)
    param_mean_by_stage_feature = [[{} for _ in range(num_features)] for _ in range(num_stages)]
    param_std_by_stage_feature = [[{} for _ in range(num_features)] for _ in range(num_stages)]
    for stage_idx in range(num_stages):
        for feat_idx in range(num_features):
            arr_sum = np.asarray(sum_costs[stage_idx][feat_idx], dtype=float)
            arr_mean = np.asarray(mean_costs[stage_idx][feat_idx], dtype=float)
            sum_mean[feat_idx, stage_idx] = float(arr_sum.mean())
            sum_std[feat_idx, stage_idx] = float(arr_sum.std())
            mean_mean[feat_idx, stage_idx] = float(arr_mean.mean())
            mean_std[feat_idx, stage_idx] = float(arr_mean.std())

            baseline_demo_sum_costs = []
            baseline_demo_step_costs = []
            for demo_full_vals, demo_vals, fitted_sum, fitted_step in zip(
                per_demo_feature_values[feat_idx],
                per_demo_stage_feature_values[stage_idx][feat_idx],
                arr_sum,
                arr_mean,
            ):
                baseline_model = GaussianModel(
                    mu=float(np.mean(demo_vals)),
                    sigma=float(max(np.std(demo_vals), 1e-6)),
                )
                baseline_neglog = -np.asarray(baseline_model.logpdf(demo_vals), dtype=float)
                baseline_sum = float(np.sum(baseline_neglog))
                baseline_step = float(np.mean(baseline_neglog))
                baseline_demo_sum_costs.append(baseline_sum)
                baseline_demo_step_costs.append(baseline_step)
                if baseline_sum > float(fitted_sum):
                    baseline_beats_fitted_sum_count[feat_idx, stage_idx] += 1
                if baseline_step > float(fitted_step):
                    baseline_beats_fitted_step_count[feat_idx, stage_idx] += 1

            baseline_arr_sum = np.asarray(baseline_demo_sum_costs, dtype=float)
            baseline_arr_step = np.asarray(baseline_demo_step_costs, dtype=float)
            baseline_sum_mean[feat_idx, stage_idx] = float(baseline_arr_sum.mean())
            baseline_sum_std[feat_idx, stage_idx] = float(baseline_arr_sum.std())
            baseline_step_mean_mean[feat_idx, stage_idx] = float(baseline_arr_step.mean())
            baseline_step_mean_std[feat_idx, stage_idx] = float(baseline_arr_step.std())
            n_demo = max(len(baseline_demo_sum_costs), 1)
            baseline_beats_fitted_sum_frac[feat_idx, stage_idx] = (
                float(baseline_beats_fitted_sum_count[feat_idx, stage_idx]) / float(n_demo)
            )
            baseline_beats_fitted_step_frac[feat_idx, stage_idx] = (
                float(baseline_beats_fitted_step_count[feat_idx, stage_idx]) / float(n_demo)
            )
            for name in param_names_per_feature[feat_idx]:
                arr = np.asarray(param_values[stage_idx][feat_idx][name], dtype=float)
                param_mean_by_stage_feature[stage_idx][feat_idx][name] = float(arr.mean())
                param_std_by_stage_feature[stage_idx][feat_idx][name] = float(arr.std())

    sigma_ratio_mean = np.full((num_features, num_stages), np.nan, dtype=float)
    sigma_ratio_std = np.full((num_features, num_stages), np.nan, dtype=float)
    w80_ratio_mean = np.full((num_features, num_stages), np.nan, dtype=float)
    w80_ratio_std = np.full((num_features, num_stages), np.nan, dtype=float)
    for stage_idx in range(num_stages):
        for feat_idx in range(num_features):
            k = str(learner.feature_model_types[feat_idx]).lower()
            if k not in {"gauss", "gaussian", "student_t", "studentt", "t"}:
                continue
            ratios = []
            w80_ratios = []
            for demo_idx in range(len(learner.demos)):
                demo_full_vals = np.asarray(per_demo_feature_values[feat_idx][demo_idx], dtype=float)
                demo_stage_vals = np.asarray(per_demo_stage_feature_values[stage_idx][feat_idx][demo_idx], dtype=float)
                demo_global_sigma = float(max(np.std(demo_full_vals), 1e-6))
                stage_sigma = float(param_values[stage_idx][feat_idx]["sigma"][demo_idx])
                ratios.append(stage_sigma / demo_global_sigma)
                global_w80 = max(_shortest_coverage_width(demo_full_vals, coverage=0.8), 1e-6)
                stage_w80 = _shortest_coverage_width(demo_stage_vals, coverage=0.8)
                w80_ratios.append(stage_w80 / global_w80)
            arr_ratio = np.asarray(ratios, dtype=float)
            sigma_ratio_mean[feat_idx, stage_idx] = float(arr_ratio.mean())
            sigma_ratio_std[feat_idx, stage_idx] = float(arr_ratio.std())
            arr_w80_ratio = np.asarray(w80_ratios, dtype=float)
            w80_ratio_mean[feat_idx, stage_idx] = float(arr_w80_ratio.mean())
            w80_ratio_std[feat_idx, stage_idx] = float(arr_w80_ratio.std())

    advantage_score = np.maximum(
        np.asarray(baseline_step_mean_mean, dtype=float) - np.asarray(mean_mean, dtype=float),
        0.0,
    )
    consistency_penalty = np.zeros((num_features, num_stages), dtype=float)
    for feat_idx in range(num_features):
        preferred_name = None
        for name in param_names_per_feature[feat_idx]:
            if name in {"b", "mu"}:
                preferred_name = name
                break
        if preferred_name is None:
            preferred_name = param_names_per_feature[feat_idx][0]
        raw_penalty = np.array(
            [
                float(param_std_by_stage_feature[stage_idx][feat_idx][preferred_name])
                for stage_idx in range(num_stages)
            ],
            dtype=float,
        )
        scale = float(np.mean(raw_penalty))
        if scale > 1e-12:
            consistency_penalty[feat_idx, :] = raw_penalty / scale
        else:
            consistency_penalty[feat_idx, :] = raw_penalty
    total_score = advantage_score - consistency_penalty
    eps = 1e-8
    advantage_relative = advantage_score / np.maximum(
        np.sum(advantage_score, axis=1, keepdims=True),
        eps,
    )
    consistency_relative = 1.0 - (
        consistency_penalty
        / np.maximum(np.sum(consistency_penalty, axis=1, keepdims=True), eps)
    )
    s_score = 0.5 * advantage_relative + 0.5 * consistency_relative
    mean_avg_nll_gain = np.maximum(
        np.asarray(baseline_step_mean_mean, dtype=float) - np.asarray(mean_mean, dtype=float),
        0.0,
    )

    return {
        "dataset": dataset_name,
        "n_demos": len(dataset.demos),
        "feature_names": feature_names,
        "final_taus": list(result["taus_hat"]),
        "stage_length_mean": [float(np.mean(v)) for v in stage_lengths],
        "stage_length_std": [float(np.std(v)) for v in stage_lengths],
        "sum_mean": sum_mean.tolist(),
        "sum_std": sum_std.tolist(),
        "step_mean_mean": mean_mean.tolist(),
        "step_mean_std": mean_std.tolist(),
        "baseline_sum_mean": baseline_sum_mean.tolist(),
        "baseline_sum_std": baseline_sum_std.tolist(),
        "baseline_step_mean_mean": baseline_step_mean_mean.tolist(),
        "baseline_step_mean_std": baseline_step_mean_std.tolist(),
        "baseline_beats_fitted_sum_count": baseline_beats_fitted_sum_count.tolist(),
        "baseline_beats_fitted_step_count": baseline_beats_fitted_step_count.tolist(),
        "baseline_beats_fitted_sum_frac": baseline_beats_fitted_sum_frac.tolist(),
        "baseline_beats_fitted_step_frac": baseline_beats_fitted_step_frac.tolist(),
        "param_names_per_feature": param_names_per_feature,
        "param_mean_by_stage_feature": param_mean_by_stage_feature,
        "param_std_by_stage_feature": param_std_by_stage_feature,
        "advantage_score": advantage_score.tolist(),
        "consistency_penalty": consistency_penalty.tolist(),
        "total_score": total_score.tolist(),
        "advantage_relative": advantage_relative.tolist(),
        "consistency_relative": consistency_relative.tolist(),
        "s_score": s_score.tolist(),
        "fitted_avg_nll_mean": mean_mean.tolist(),
        "fitted_avg_nll_std": mean_std.tolist(),
        "baseline_avg_nll_mean": baseline_step_mean_mean.tolist(),
        "baseline_avg_nll_std": baseline_step_mean_std.tolist(),
        "mean_avg_nll_gain": mean_avg_nll_gain.tolist(),
        "baseline_worse_frac": baseline_beats_fitted_step_frac.tolist(),
        "sigma_ratio_mean": sigma_ratio_mean.tolist(),
        "sigma_ratio_std": sigma_ratio_std.tolist(),
        "w80_ratio_mean": w80_ratio_mean.tolist(),
        "w80_ratio_std": w80_ratio_std.tolist(),
    }


def analyze_feature_stability(env_config: str, method_config: str, max_iter: int | None = None) -> dict[str, Any]:
    dataset_name, dataset, result = _fit_scdp(env_config, method_config, max_iter=max_iter)
    return _analyze_trained_scdp(dataset_name, dataset, result)


def _print_feature_stage_table(title: str, values, feature_names: list[str], stage_count: int, decimals: int = 4) -> None:
    arr = np.asarray(values)
    stage_labels = [f"stage{i + 1}" for i in range(stage_count)]
    row_label = "feature"
    row_width = max(len(row_label), max(len(name) for name in feature_names))

    if np.issubdtype(arr.dtype, np.integer):
        formatted = [[str(int(arr[i, j])) for j in range(arr.shape[1])] for i in range(arr.shape[0])]
    else:
        formatted = [[f"{float(arr[i, j]):.{decimals}f}" for j in range(arr.shape[1])] for i in range(arr.shape[0])]

    col_widths = []
    for j, label in enumerate(stage_labels):
        width = len(label)
        for i in range(len(feature_names)):
            width = max(width, len(formatted[i][j]))
        col_widths.append(width)

    print(title)
    header = row_label.ljust(row_width) + " | " + " | ".join(
        label.rjust(col_widths[j]) for j, label in enumerate(stage_labels)
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for i, feat_name in enumerate(feature_names):
        row = feat_name.ljust(row_width) + " | " + " | ".join(
            formatted[i][j].rjust(col_widths[j]) for j in range(len(stage_labels))
        )
        print(row)
    print()


def _print_param_table(title: str, param_values, feature_names: list[str], stage_count: int) -> None:
    stage_labels = [f"stage{i + 1}" for i in range(stage_count)]
    row_label = "feature"
    row_width = max(len(row_label), max(len(name) for name in feature_names))

    rendered = []
    col_widths = [len(label) for label in stage_labels]
    for feat_idx, feat_name in enumerate(feature_names):
        row = []
        for stage_idx in range(stage_count):
            cell = json.dumps(param_values[stage_idx][feat_idx], ensure_ascii=False, sort_keys=True)
            row.append(cell)
            col_widths[stage_idx] = max(col_widths[stage_idx], len(cell))
        rendered.append((feat_name, row))

    print(title)
    header = row_label.ljust(row_width) + " | " + " | ".join(
        label.rjust(col_widths[j]) for j, label in enumerate(stage_labels)
    )
    sep = "-" * len(header)
    print(header)
    print(sep)
    for feat_name, row in rendered:
        line = feat_name.ljust(row_width) + " | " + " | ".join(
            row[j].rjust(col_widths[j]) for j in range(stage_count)
        )
        print(line)
    print()


def _shortest_coverage_width(values: np.ndarray, coverage: float = 0.8) -> float:
    vals = np.sort(np.asarray(values, dtype=float).reshape(-1))
    n = len(vals)
    if n <= 1:
        return 0.0
    m = int(np.ceil(float(coverage) * n))
    m = min(max(m, 1), n)
    best = float("inf")
    for start in range(0, n - m + 1):
        width = float(vals[start + m - 1] - vals[start])
        if width < best:
            best = width
    return 0.0 if not np.isfinite(best) else best


def main():
    parser = argparse.ArgumentParser(description="Analyze per-stage, per-feature SCDP cost stability across demos.")
    parser.add_argument("--env-config", default="configs/envs/2DObsAvoid.json", type=str)
    parser.add_argument("--method-config", default="configs/methods/scdp.json", type=str)
    parser.add_argument("--max-iter", type=int, default=12)
    parser.add_argument("--plots-dir", type=str, default=None, help="Directory to save per-demo feature cost plots.")
    parser.add_argument("--no-plots", action="store_true", help="Skip per-demo feature cost plots.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a readable summary.")
    args = parser.parse_args()

    dataset_name, dataset, result = _fit_scdp(
        env_config=args.env_config,
        method_config=args.method_config,
        max_iter=args.max_iter,
    )
    out = _analyze_trained_scdp(dataset_name, dataset, result)

    plot_outputs: list[str] = []
    if not args.no_plots:
        if plt is None:
            out["plots_skipped"] = "matplotlib is not installed"
        else:
            plot_outputs = [str(path) for path in plot_all_demo_feature_costs(dataset_name, result["model"], args.plots_dir)]
            out["plot_outputs"] = plot_outputs

    if args.json:
        print(json.dumps(out, indent=2, ensure_ascii=False))
        return

    print("dataset", out["dataset"])
    print("n_demos", out["n_demos"])
    print("feature_names", out["feature_names"])
    print("final_taus", out["final_taus"])
    print("stage_length_mean", np.round(np.asarray(out["stage_length_mean"], dtype=float), 4))
    print("stage_length_std", np.round(np.asarray(out["stage_length_std"], dtype=float), 4))
    if plot_outputs:
        print("plot_dir", str(Path(plot_outputs[0]).parent))
    elif out.get("plots_skipped"):
        print("plots", out["plots_skipped"])
    print()
    print("How to read this summary")
    print("- Each per-demo plot shows one demo's `fitted avg NLL`, `baseline avg NLL`, and `avg NLL gain`.")
    print("- Here, the first four tables are the cross-demo mean/std versions of those same per-demo quantities.")
    print("- `MEAN_AVG_NLL_GAIN` is the direct aggregate counterpart of the plot's `avg NLL gain`.")
    print("- `BASELINE_WORSE_FRAC` is the fraction of demos where baseline avg NLL is worse than fitted avg NLL; use it as reference only.")
    print("- `LOCAL_DISPERSION = mean(abs(x - median(stage)))` is the equality-style concentration score; smaller means the stage is tighter.")
    print("- `W80_RATIO = shortest 80%-coverage width(stage) / shortest 80%-coverage width(global demo)` is a more direct concentration score; smaller means more concentrated.")
    print("- `PARAM_STD_BY_STAGE_FEATURE` measures cross-demo parameter consistency; smaller key-param std means more consistent.")
    print("- `ADVANTAGE_SCORE` is now exactly `MEAN_AVG_NLL_GAIN`.")
    print("- `CONSISTENCY_PENALTY` is lower when the key parameter (`b` or `mu`) is more consistent across demos.")
    print("- `S_SCORE` is the feature-internal relative score to compare stage1 vs stage2 for the same feature.")
    print()
    stage_count = len(out["stage_length_mean"])
    feature_names = list(out["feature_names"])
    _print_feature_stage_table("FITTED_AVG_NLL_MEAN", out["fitted_avg_nll_mean"], feature_names, stage_count)
    _print_feature_stage_table("FITTED_AVG_NLL_STD", out["fitted_avg_nll_std"], feature_names, stage_count)
    _print_feature_stage_table("BASELINE_AVG_NLL_MEAN", out["baseline_avg_nll_mean"], feature_names, stage_count)
    _print_feature_stage_table("BASELINE_AVG_NLL_STD", out["baseline_avg_nll_std"], feature_names, stage_count)
    _print_feature_stage_table("MEAN_AVG_NLL_GAIN", out["mean_avg_nll_gain"], feature_names, stage_count)
    _print_feature_stage_table("BASELINE_WORSE_FRAC", out["baseline_worse_frac"], feature_names, stage_count)
    _print_feature_stage_table("SIGMA_RATIO_MEAN", out["sigma_ratio_mean"], feature_names, stage_count)
    _print_feature_stage_table("SIGMA_RATIO_STD", out["sigma_ratio_std"], feature_names, stage_count)
    _print_feature_stage_table("W80_RATIO_MEAN", out["w80_ratio_mean"], feature_names, stage_count)
    _print_feature_stage_table("W80_RATIO_STD", out["w80_ratio_std"], feature_names, stage_count)
    _print_param_table("PARAM_STD_BY_STAGE_FEATURE", out["param_std_by_stage_feature"], feature_names, stage_count)
    _print_feature_stage_table("ADVANTAGE_SCORE", out["advantage_score"], feature_names, stage_count)
    _print_feature_stage_table("CONSISTENCY_PENALTY", out["consistency_penalty"], feature_names, stage_count)
    _print_feature_stage_table("TOTAL_SCORE", out["total_score"], feature_names, stage_count)
    _print_feature_stage_table("ADVANTAGE_RELATIVE", out["advantage_relative"], feature_names, stage_count)
    _print_feature_stage_table("CONSISTENCY_RELATIVE", out["consistency_relative"], feature_names, stage_count)
    _print_feature_stage_table("S_SCORE", out["s_score"], feature_names, stage_count)


if __name__ == "__main__":
    main()
