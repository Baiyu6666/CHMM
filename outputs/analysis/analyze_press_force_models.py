from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.press_slide_insert_2d import load_2d_press_slide_insert
from methods.cores.scdp import SegmentConsensusDPModel
from utils.models import MarginExpLowerEmission, StudentTModel

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for this script. Install matplotlib, then rerun "
        "`python outputs/analysis/analyze_press_force_models.py`."
    ) from exc


EPS = 1e-12
COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]
MODEL_NAMES = ["HalfStudentT", "StudentT", "Exponential", "MarginExp(current)"]


@dataclass
class FitResult:
    model: str
    params: dict
    b_hat: float
    avg_loglik: float
    total_loglik: float
    baseline_avg_loglik: float
    training_score: float


def _jsonify_value(value):
    if isinstance(value, (np.floating, float, np.integer, int)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonify_value(v) for v in value.tolist()]
    return value


def _load_current_bundle():
    env_cfg_path = ROOT / "configs" / "envs" / "2DPressSlideInsert.json"
    env_cfg = json.loads(env_cfg_path.read_text())
    return load_2d_press_slide_insert(
        n_demos=int(env_cfg.get("n_demos", 5)),
        seed=int(env_cfg.get("seed", 123)),
        env_kwargs=env_cfg.get("env_kwargs"),
    )


def _analysis_demo_idx():
    return 0


def _load_scdp_kwargs():
    method_cfg_path = ROOT / "configs" / "methods" / "scdp_auto.json"
    env_cfg_path = ROOT / "configs" / "envs" / "2DPressSlideInsert.json"
    method_cfg = json.loads(method_cfg_path.read_text())
    env_cfg = json.loads(env_cfg_path.read_text())
    kwargs = dict(method_cfg)
    kwargs.update(dict(env_cfg.get("method_overrides", {}).get("scdp", {})))
    kwargs["plot_every"] = None
    kwargs["plot_dir"] = str(OUTPUT_DIR)
    return kwargs


def _build_scdp_reference_learner(bundle):
    kwargs = _load_scdp_kwargs()
    return SegmentConsensusDPModel(
        demos=bundle.demos,
        env=bundle.env,
        true_taus=bundle.true_taus,
        true_cutpoints=getattr(bundle, "true_cutpoints", None),
        n_states=int(kwargs.get("n_states", 4)),
        seed=int(kwargs.get("seed", 0)),
        selected_raw_feature_ids=kwargs.get("selected_raw_feature_ids"),
        feature_model_types=kwargs.get("feature_model_types"),
        fixed_feature_mask=kwargs.get("fixed_feature_mask"),
        lambda_eq_constraint=float(kwargs.get("lambda_eq_constraint", kwargs.get("lambda_constraint", 1.0))),
        lambda_ineq_constraint=float(kwargs.get("lambda_ineq_constraint", kwargs.get("lambda_constraint", 1.0))),
        lambda_progress=float(kwargs.get("lambda_progress", 1.0)),
        lambda_subgoal_consensus=float(kwargs.get("lambda_subgoal_consensus", kwargs.get("lambda_consensus", 1.0))),
        lambda_param_consensus=float(kwargs.get("lambda_param_consensus", 1.0)),
        lambda_activation_consensus=float(
            kwargs.get(
                "lambda_activation_consensus",
                kwargs.get("lambda_feature_score_consensus", kwargs.get("lambda_r_consensus", 1.0)),
            )
        ),
        consensus_schedule=str(kwargs.get("consensus_schedule", "linear")),
        progress_delta_scale=float(kwargs.get("progress_delta_scale", 20.0)),
        duration_min=kwargs.get("duration_min"),
        duration_max=kwargs.get("duration_max"),
        feature_activation_mode=str(kwargs.get("feature_activation_mode", "score")),
        equality_score_mode=str(kwargs.get("equality_score_mode", "dispersion")),
        equality_dispersion_ratio_threshold=float(kwargs.get("equality_dispersion_ratio_threshold", 0.1)),
        constraint_core_trim=int(kwargs.get("constraint_core_trim", 0)),
        short_segment_penalty_c=float(kwargs.get("short_segment_penalty_c", kwargs.get("equality_score_uncertainty_c", 0.1))),
        equality_gaussian_score_activation_threshold=float(kwargs.get("equality_gaussian_score_activation_threshold", -0.5)),
        inequality_score_activation_threshold=float(kwargs.get("inequality_score_activation_threshold", -0.5)),
        fixed_true_cutpoint_prefix=int(kwargs.get("fixed_true_cutpoint_prefix", 0)),
        fixed_true_cutpoint_indices=kwargs.get("fixed_true_cutpoint_indices"),
        plot_every=None,
        plot_dir=str(OUTPUT_DIR),
    )


def _describe_stage_variant(stage):
    if stage["stage_name"] == "synthetic_gaussian":
        return "synthetic Gaussian (standardized)"
    if not stage["constrained"]:
        return "unconstrained (standardized)"
    notes = stage.get("notes") or {}
    parts = [f"true b(z)={stage['b_ref']:.3f}"]
    if "slack_diffuse" in notes:
        parts.append("diffuse slack")
    if "below_b_noise" in notes:
        parts.append("10% below-b noise")
    return " + ".join(parts)


def _stage_force_data(bundle, learner, add_below_b_noise, diffuse_slack, demo_idx):
    env = bundle.env
    force_local_idx = int(learner.feature_name_to_local_idx["force"])
    force_column_idx = int(learner.feature_specs[force_local_idx]["column_idx"])
    force_mean = float(learner.feat_mean[force_column_idx])
    force_std = float(learner.feat_std[force_column_idx])
    # stage 0 is intentionally unconstrained; use b_ref=0 only for comparison.
    stage_specs = [
        ("stage0_free", 0, 0.0, False),
        ("stage1_contact", 1, float(env.f_contact_min), True),
        ("stage2_slide", 2, float(env.f_slide_min), True),
        ("stage3_insert", 3, float(env.f_insert_min), True),
    ]
    out = []
    rng = np.random.default_rng(20260325)
    demo = bundle.demos[int(demo_idx)]
    labels = np.asarray(bundle.true_labels[int(demo_idx)], dtype=int)
    F = np.asarray(env.compute_all_features_matrix(demo), dtype=float)
    for stage_name, stage_idx, b_ref, constrained in stage_specs:
        raw_values = np.asarray(F[labels == stage_idx, force_column_idx], dtype=float)
        notes = {}
        if constrained and diffuse_slack:
            slack = np.maximum(raw_values - float(b_ref), 0.0)
            slack_scale = max(float(np.std(slack)), 0.02 * max(abs(float(b_ref)), 1.0), 0.008)
            widened_slack = (
                0.35 * slack
                + 0.65 * np.abs(rng.normal(loc=1.25 * slack_scale, scale=1.15 * slack_scale, size=len(slack)))
            )
            raw_values = float(b_ref) + widened_slack
            notes["slack_diffuse"] = {
                "kind": "positive_slack_widening",
                "mix_original": 0.35,
                "mix_added": 0.65,
                "gaussian_mean_before_abs": float(1.25 * slack_scale),
                "gaussian_std": float(1.15 * slack_scale),
            }
        if constrained and add_below_b_noise:
            noise_count = max(1, int(round(0.10 * len(raw_values))))
            noise_scale = max(0.02 * max(abs(float(b_ref)), 1.0), 0.008)
            noise_vals = []
            while len(noise_vals) < noise_count:
                batch_size = max(16, 3 * (noise_count - len(noise_vals)))
                raw_noise = rng.normal(loc=float(b_ref), scale=noise_scale, size=batch_size)
                below_b = raw_noise[raw_noise < float(b_ref)]
                if below_b.size:
                    noise_vals.extend(below_b.tolist())
            noise_vals = np.asarray(noise_vals[:noise_count], dtype=float)
            raw_values = np.concatenate([raw_values, noise_vals], axis=0)
            notes["below_b_noise"] = {
                "kind": "gaussian",
                "count": int(noise_count),
                "mean_before_filter": float(b_ref),
                "std": float(noise_scale),
                "selection": "sample around true b, keep only x < b",
            }
        values = (raw_values - force_mean) / force_std
        out.append(
            {
                "stage_name": stage_name,
                "stage_idx": int(stage_idx),
                "b_ref": float((float(b_ref) - force_mean) / force_std),
                "constrained": bool(constrained),
                "values": values,
                "force_stats": {
                    "raw_mean": force_mean,
                    "raw_std": force_std,
                    "force_column_idx": force_column_idx,
                    "force_local_idx": force_local_idx,
                    "demo_idx": int(demo_idx),
                },
                "notes": notes or None,
            }
        )
    return out


def _synthetic_gaussian_force_data(learner, n_samples=320, seed=20260325):
    rng = np.random.default_rng(seed)
    force_local_idx = int(learner.feature_name_to_local_idx["force"])
    force_column_idx = int(learner.feature_specs[force_local_idx]["column_idx"])
    force_mean = float(learner.feat_mean[force_column_idx])
    force_std = float(learner.feat_std[force_column_idx])
    raw_values = rng.normal(loc=0.78, scale=0.06, size=int(n_samples))
    values = (raw_values - force_mean) / force_std
    return {
        "stage_name": "synthetic_gaussian",
        "stage_idx": -1,
        "b_ref": float("nan"),
        "constrained": False,
        "values": np.asarray(values, dtype=float),
        "notes": {
            "distribution": "Gaussian then standardized with learner force stats",
            "raw_mean": 0.78,
            "raw_std": 0.06,
            "standardized_by_force_mean": force_mean,
            "standardized_by_force_std": force_std,
            "seed": int(seed),
        },
    }


def _safe_log(x):
    return np.log(np.maximum(np.asarray(x, dtype=float), EPS))


def _student_t_logpdf(x, sigma, nu):
    x = np.asarray(x, dtype=float)
    sigma = max(float(sigma), 1e-6)
    nu = max(float(nu), 1e-6)
    z2 = (x / sigma) ** 2
    log_norm = (
        math.lgamma((nu + 1.0) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * (math.log(nu) + math.log(math.pi))
        - math.log(sigma)
    )
    return log_norm - 0.5 * (nu + 1.0) * np.log1p(z2 / nu)


def _estimate_lower_quantile(values, q=0.02):
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        return np.nan
    order = np.argsort(x)
    vals_sorted = x[order]
    idx = int(np.searchsorted(np.linspace(1.0 / x.size, 1.0, x.size), float(q), side="left"))
    return float(vals_sorted[min(idx, len(vals_sorted) - 1)])


def _fit_student_t_training(learner, values):
    model = StudentTModel(mu=0.0, sigma=1.0)
    model.m_step_update([np.asarray(values, dtype=float)])
    model._update_interval()
    x = np.asarray(values, dtype=float)
    logpdf = np.asarray(model.logpdf(x), dtype=float)
    return model.get_summary(), logpdf


def _fit_margin_exp_training(learner, values):
    model = MarginExpLowerEmission(b_init=0.0, lam_init=1.0)
    model.m_step_update([np.asarray(values, dtype=float)])
    model._update_interval()
    x = np.asarray(values, dtype=float)
    logpdf = np.asarray(model.logpdf(x), dtype=float)
    return model.get_summary(), logpdf


def _fit_margin_exp_search(values, tail_nu=3.0):
    x = np.asarray(values, dtype=float)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    span = max(float(x_max - x_min), 1e-4)
    b_grid = np.linspace(x_min - 0.08 * span - 0.01, x_min, 100)
    best_total = -np.inf
    best_summary = None
    best_logpdf = None
    for b_hat in b_grid:
        slack = np.maximum(x - float(b_hat), 0.0)
        alpha = max(float(tail_nu), 1.0 + 1e-6)
        lam = max(float(np.mean(slack)) * (alpha - 1.0), 1e-6)
        logpdf = np.full_like(x, -1e9, dtype=float)
        mask = x >= float(b_hat)
        if np.any(mask):
            logpdf[mask] = math.log(alpha) - math.log(lam) - (alpha + 1.0) * np.log1p((x[mask] - float(b_hat)) / lam)
        total = float(np.sum(logpdf))
        if total > best_total:
            best_total = total
            best_logpdf = logpdf
            best_summary = {
                "type": "margin_exp_lower_search",
                "b": float(b_hat),
                "lam": float(lam),
                "tail_nu": float(alpha),
            }
    return best_summary, np.asarray(best_logpdf, dtype=float)


def _fit_half_student_t_training(learner, values):
    x = np.asarray(values, dtype=float)
    b_hat = _estimate_lower_quantile(x, q=0.02)
    slack = np.maximum(x - b_hat, 0.0)
    center = float(np.median(slack))
    sigma = max(float(1.4826 * np.median(np.abs(slack - center))), 1e-6)
    logpdf = np.full_like(x, -1e9, dtype=float)
    mask = x >= b_hat
    if np.any(mask):
        logpdf[mask] = math.log(2.0) + _student_t_logpdf(x[mask] - b_hat, sigma=sigma, nu=3.0)
    return {"b": float(b_hat), "sigma": float(sigma), "nu": 3.0}, logpdf


def _fit_exponential_training(learner, values):
    x = np.asarray(values, dtype=float)
    b_hat = _estimate_lower_quantile(x, q=0.02)
    slack = np.maximum(x - b_hat, 0.0)
    scale = max(float(np.mean(slack)), 1e-6)
    logpdf = np.full_like(x, -1e9, dtype=float)
    mask = x >= b_hat
    if np.any(mask):
        logpdf[mask] = -math.log(scale) - (x[mask] - b_hat) / scale
    return {"b": float(b_hat), "scale": float(scale)}, logpdf


def _fit_model_training_style(learner, model_name, values):
    if model_name == "StudentT":
        return _fit_student_t_training(learner, values)
    if model_name == "HalfStudentT":
        return _fit_half_student_t_training(learner, values)
    if model_name == "Exponential":
        return _fit_exponential_training(learner, values)
    if model_name == "MarginExp(current)":
        return _fit_margin_exp_search(values, tail_nu=3.0)
    raise ValueError(model_name)


def _raw_pdf(model_name, params, x):
    xx = np.asarray(x, dtype=float)
    if model_name == "StudentT":
        mu = float(params["mu"])
        sigma = max(float(params["sigma"]), 1e-6)
        nu = max(float(params.get("nu", 3.0)), 1e-6)
        return np.exp(_student_t_logpdf(xx - mu, sigma=sigma, nu=nu))
    if model_name == "HalfStudentT":
        b = float(params["b"])
        sigma = max(float(params["sigma"]), 1e-6)
        nu = max(float(params.get("nu", 3.0)), 1e-6)
        out = np.zeros_like(xx, dtype=float)
        mask = xx >= b
        if np.any(mask):
            out[mask] = np.exp(math.log(2.0) + _student_t_logpdf(xx[mask] - b, sigma=sigma, nu=nu))
        return out
    if model_name == "Exponential":
        b = float(params["b"])
        scale = max(float(params["scale"]), 1e-6)
        out = np.zeros_like(xx, dtype=float)
        mask = xx >= b
        if np.any(mask):
            out[mask] = np.exp(-((xx[mask] - b) / scale)) / scale
        return out
    if model_name == "MarginExp(current)":
        b = float(params["b"])
        lam = max(float(params["lam"]), 1e-6)
        alpha = max(float(params.get("tail_nu", 3.0)), 1e-6)
        out = np.zeros_like(xx, dtype=float)
        mask = xx >= b
        if np.any(mask):
            out[mask] = alpha / lam * (1.0 + (xx[mask] - b) / lam) ** (-(alpha + 1.0))
        return out
    raise ValueError(model_name)


def _loglik_with_violation_training_style(learner, values, model_name, left_mass=0.02):
    x = np.asarray(values, dtype=float)
    if model_name == "StudentT":
        return _fit_student_t_training(learner, x)
    params, _ = _fit_model_training_style(learner, model_name, x)
    b_hat = float(params["b"])
    left_mask = x < b_hat
    right_mask = ~left_mask
    scale_ref = max(float(params.get("scale", params.get("sigma", params.get("lam", 0.02)))), 1e-6)
    alpha_left = 1.0 / scale_ref
    logpdf = np.zeros_like(x, dtype=float)
    if np.any(left_mask):
        logpdf[left_mask] = math.log(left_mass) + math.log(alpha_left) + alpha_left * (x[left_mask] - b_hat)
    if np.any(right_mask):
        right_pdf = np.maximum(_raw_pdf(model_name, params, x[right_mask]), EPS)
        logpdf[right_mask] = math.log(1.0 - left_mass) + np.log(right_pdf)
    return params, logpdf


def _fit_model_result(learner, model_name, values, with_violation):
    _, baseline_logpdf = _fit_student_t_training(learner, values)
    if with_violation:
        params, logpdf = _loglik_with_violation_training_style(learner, values, model_name)
    else:
        params, logpdf = _fit_model_training_style(learner, model_name, values)
    avg_loglik = float(np.mean(logpdf))
    baseline_avg_loglik = float(np.mean(baseline_logpdf))
    if model_name == "StudentT":
        location = float(params["mu"])
    else:
        location = float(params["b"])
    return FitResult(
        model=model_name,
        params=params,
        b_hat=location,
        avg_loglik=avg_loglik,
        total_loglik=float(np.sum(logpdf)),
        baseline_avg_loglik=baseline_avg_loglik,
        training_score=float((-avg_loglik) - (-baseline_avg_loglik)),
    )


def _fit_all_models(learner, values):
    no_violation = []
    with_violation = []
    for model_name in MODEL_NAMES:
        no_violation.append(_fit_model_result(learner, model_name, values, with_violation=False))
        with_violation.append(_fit_model_result(learner, model_name, values, with_violation=True))
    return no_violation, with_violation


def _plot_density_grid(stage_results, output_name):
    fig, axes = plt.subplots(2, len(stage_results), figsize=(4.5 * len(stage_results), 8.0))
    if len(stage_results) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    row_titles = ["Training-style one-sided fit", "Violation-aware left tail"]
    for col_idx, stage in enumerate(stage_results):
        vals = np.asarray(stage["values"], dtype=float)
        fit_sets = [stage["fits_no_violation"], stage["fits_with_violation"]]
        q_lo, q_hi = np.quantile(vals, [0.05, 0.95])
        span = max(float(q_hi - q_lo), 0.04)
        b_candidates = [float(fit.b_hat) for fit_set in fit_sets for fit in fit_set]
        if np.isfinite(stage["b_ref"]):
            b_candidates.append(float(stage["b_ref"]))
        lo_ref = float(q_lo)
        hi_ref = float(q_hi)
        if b_candidates:
            lo_ref = min(lo_ref, min(b_candidates))
        left_pad = max(0.10 * span, 0.015)
        right_pad = max(0.12 * span, 0.02)
        x_min = lo_ref - left_pad
        x_max = hi_ref + right_pad
        x_grid = np.linspace(x_min, x_max, 500)
        hist_bins = min(max(len(vals) // 8, 10), 28)
        hist_density, _ = np.histogram(vals, bins=hist_bins, density=True)
        y_max = float(np.max(hist_density)) if hist_density.size else 0.0
        curve_sets = []
        for row_idx, fit_set in enumerate(fit_sets):
            row_curves = []
            for fit in fit_set:
                if row_idx == 0 or fit.model == "StudentT":
                    y = _raw_pdf(fit.model, fit.params, x_grid)
                else:
                    y = np.zeros_like(x_grid)
                    left_mask = x_grid < fit.b_hat
                    right_mask = ~left_mask
                    scale_ref = max(float(fit.params.get("scale", fit.params.get("sigma", fit.params.get("lam", 0.02)))), 1e-6)
                    alpha_left = 1.0 / scale_ref
                    if np.any(left_mask):
                        y[left_mask] = 0.02 * alpha_left * np.exp(alpha_left * (x_grid[left_mask] - fit.b_hat))
                    if np.any(right_mask):
                        y[right_mask] = 0.98 * _raw_pdf(fit.model, fit.params, x_grid[right_mask])
                row_curves.append(np.asarray(y, dtype=float))
                y_max = max(y_max, float(np.max(y)))
            curve_sets.append(row_curves)
        y_max = max(y_max, 1e-3)
        for row_idx, fit_set in enumerate(fit_sets):
            ax = axes[row_idx, col_idx]
            ax.hist(vals, bins=hist_bins, density=True, alpha=0.35, color="#AFAFAF", label="data")
            for color, fit, y in zip(COLORS, fit_set, curve_sets[row_idx]):
                ax.plot(x_grid, y, color=color, lw=1.35, label=fit.model)
            if np.isfinite(stage["b_ref"]):
                ax.axvline(float(stage["b_ref"]), color="black", linestyle="--", lw=1.1, label="ref b")
            title_suffix = _describe_stage_variant(stage)
            ax.set_title(f"{stage['stage_name']}\n{row_titles[row_idx]} | {title_suffix}", fontsize=10)
            ax.set_xlabel("standardized force")
            ax.set_xlim(float(x_grid[0]), float(x_grid[-1]))
            ax.set_ylim(0.0, 1.08 * y_max)
            if col_idx == 0:
                ax.set_ylabel("density")
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path = OUTPUT_DIR / output_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _plot_summary_grid(stage_results, output_name):
    fig, axes = plt.subplots(3, len(stage_results), figsize=(4.4 * len(stage_results), 12.0))
    if len(stage_results) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    x = np.arange(len(MODEL_NAMES))
    width = 0.34
    for col_idx, stage in enumerate(stage_results):
        no_ll = [float(fit.avg_loglik) for fit in stage["fits_no_violation"]]
        with_ll = [float(fit.avg_loglik) for fit in stage["fits_with_violation"]]
        no_score = [float(fit.training_score) for fit in stage["fits_no_violation"]]
        with_score = [float(fit.training_score) for fit in stage["fits_with_violation"]]
        no_b = [float(fit.b_hat) for fit in stage["fits_no_violation"]]
        with_b = [float(fit.b_hat) for fit in stage["fits_with_violation"]]

        ax_ll = axes[0, col_idx]
        bars_no = ax_ll.bar(x - width / 2.0, no_ll, width=width, color="#4C78A8", label="no violation")
        bars_with = ax_ll.bar(x + width / 2.0, with_ll, width=width, color="#F58518", label="with violation")
        ax_ll.set_xticks(x)
        ax_ll.set_xticklabels(MODEL_NAMES, rotation=20, ha="right", fontsize=8)
        ax_ll.set_title(f"{stage['stage_name']}\navg log-lik", fontsize=10)
        for bar, val in zip(bars_no, no_ll):
            ax_ll.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars_with, with_ll):
            ax_ll.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        if col_idx == 0:
            ax_ll.set_ylabel("avg_loglik")
            ax_ll.legend(frameon=False, fontsize=8)

        ax_score = axes[1, col_idx]
        bars_no = ax_score.bar(x - width / 2.0, no_score, width=width, color="#4C78A8", label="no violation")
        bars_with = ax_score.bar(x + width / 2.0, with_score, width=width, color="#F58518", label="with violation")
        ax_score.set_xticks(x)
        ax_score.set_xticklabels(MODEL_NAMES, rotation=20, ha="right", fontsize=8)
        ax_score.set_title(f"{stage['stage_name']}\ntraining score (-ll_gain)", fontsize=10)
        for bar, val in zip(bars_no, no_score):
            ax_score.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars_with, with_score):
            ax_score.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        if col_idx == 0:
            ax_score.set_ylabel("fitted_avg_nll - baseline_avg_nll")

        ax_b = axes[2, col_idx]
        bars_no = ax_b.bar(x - width / 2.0, no_b, width=width, color="#4C78A8", label="no violation")
        bars_with = ax_b.bar(x + width / 2.0, with_b, width=width, color="#F58518", label="with violation")
        ax_b.set_xticks(x)
        ax_b.set_xticklabels(MODEL_NAMES, rotation=20, ha="right", fontsize=8)
        ax_b.set_title(f"{stage['stage_name']}\nlocation (b / mu) in z-space", fontsize=10)
        if np.isfinite(stage["b_ref"]):
            ax_b.axhline(float(stage["b_ref"]), color="black", linestyle="--", lw=1.0)
        for bar, val in zip(bars_no, no_b):
            ax_b.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        for bar, val in zip(bars_with, with_b):
            ax_b.text(bar.get_x() + bar.get_width() / 2.0, val, f"{val:.3f}", ha="center", va="bottom", fontsize=7)
        if col_idx == 0:
            ax_b.set_ylabel("location")
    fig.tight_layout()
    out_path = OUTPUT_DIR / output_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _run_analysis_pass(bundle, learner, add_below_b_noise, diffuse_slack, tag, demo_idx):
    stage_results = _stage_force_data(
        bundle,
        learner,
        add_below_b_noise=add_below_b_noise,
        diffuse_slack=diffuse_slack,
        demo_idx=demo_idx,
    )
    stage_results.append(_synthetic_gaussian_force_data(learner))
    json_payload = {"stages": []}
    for stage in stage_results:
        fits_no, fits_with = _fit_all_models(learner, stage["values"])
        stage["fits_no_violation"] = fits_no
        stage["fits_with_violation"] = fits_with
        json_payload["stages"].append(
            {
                "stage_name": stage["stage_name"],
                "stage_idx": int(stage["stage_idx"]),
                "analysis_tag": tag,
                "demo_idx": int(demo_idx),
                "analysis_space": "standardized force feature used by current SCDP preprocessing",
                "b_ref": float(stage["b_ref"]),
                "constrained": bool(stage["constrained"]),
                "notes": stage.get("notes"),
                "force_stats": stage.get("force_stats"),
                "summary": {
                    "n": int(len(stage["values"])),
                    "min": float(np.min(stage["values"])),
                    "median": float(np.median(stage["values"])),
                    "mean": float(np.mean(stage["values"])),
                    "std": float(np.std(stage["values"])),
                    "max": float(np.max(stage["values"])),
                },
                "fits_no_violation": [
                    {
                        "model": fit.model,
                        "params": {k: _jsonify_value(v) for k, v in fit.params.items()},
                        "b_hat": float(fit.b_hat),
                        "avg_loglik": float(fit.avg_loglik),
                        "baseline_avg_loglik": float(fit.baseline_avg_loglik),
                        "training_score": float(fit.training_score),
                        "total_loglik": float(fit.total_loglik),
                    }
                    for fit in fits_no
                ],
                "fits_with_violation": [
                    {
                        "model": fit.model,
                        "params": {k: _jsonify_value(v) for k, v in fit.params.items()},
                        "b_hat": float(fit.b_hat),
                        "avg_loglik": float(fit.avg_loglik),
                        "baseline_avg_loglik": float(fit.baseline_avg_loglik),
                        "training_score": float(fit.training_score),
                        "total_loglik": float(fit.total_loglik),
                    }
                    for fit in fits_with
                ],
            }
        )

    fits_png = _plot_density_grid(stage_results, output_name=f"press_force_model_fits_{tag}.png")
    summary_png = _plot_summary_grid(stage_results, output_name=f"press_force_model_summary_{tag}.png")
    json_path = OUTPUT_DIR / f"press_force_model_report_{tag}.json"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    return fits_png, summary_png, json_path


def main():
    bundle = _load_current_bundle()
    learner = _build_scdp_reference_learner(bundle)
    demo_idx = _analysis_demo_idx()
    outputs = []
    outputs.extend(_run_analysis_pass(bundle, learner, add_below_b_noise=False, diffuse_slack=False, tag="clean", demo_idx=demo_idx))
    outputs.extend(_run_analysis_pass(bundle, learner, add_below_b_noise=True, diffuse_slack=False, tag="noisy", demo_idx=demo_idx))
    outputs.extend(_run_analysis_pass(bundle, learner, add_below_b_noise=False, diffuse_slack=True, tag="diffuse", demo_idx=demo_idx))
    for path in outputs:
        print(str(path))


if __name__ == "__main__":
    main()
