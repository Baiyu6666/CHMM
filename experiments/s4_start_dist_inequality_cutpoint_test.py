from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import t as student_t_dist

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.registry import load_env

PAPER_TITLE_SIZE = 9
PAPER_LABEL_SIZE = 8
PAPER_TICK_SIZE = 7
PAPER_LEGEND_SIZE = 6.5
STAGE_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
DEFAULT_SEED = 1342
DEFAULT_ENV_KWARGS = {
    "rollout_backend": "analytic",
    "observation_backend": "analytic",
    "dt": 0.4,
    "seg_lengths": [35, 14, 67, 21],
    "seg_length_jitter": [9, 3, 11, 7],
    "normal_load_min": 4.0,
    "normal_load_scale": 10.0,
}
DEFAULT_FEATURE = "insert_err,start_dist,normal_force"
DEFAULT_METHODS = "seg_q05"
DEFAULT_BEST_METHOD = "seg_q05"
DEFAULT_DIRECTION = "lower"
DEFAULT_MIN_LEN = 5
DEFAULT_TRIM_FRACTION = 0.
DEFAULT_TRIM_MIN_N = 7
DEFAULT_NU = 7.0
DEFAULT_SCORE_SPACE = "standardized"
DEFAULT_GAIN_THRESHOLD = 0.2
DEFAULT_SOFT_BOUNDARY_SCALE = 0.1
DEFAULT_BASELINE_FIT = "mle"
DEFAULT_BOUNDARY_INDEX = 1
DEFAULT_HALF_T_SCALE_QUANTILE = 0.9
# DEFAULT_SELECTION_OBJECTIVE = "total-margin"
DEFAULT_SELECTION_OBJECTIVE = "active-loglik"
DEFAULT_OBSERVATION_NOISE_SCALE = 0.003
S4_FEATURE_UNITS = {
    "surf_dist": "m",
    "center_dist": "m",
    "force": "N",
    "normal_force": "N",
    "orient_err": "rad",
    "speed": "m/s",
    "angular_speed": "rad/s",
    "noise": "",
    "start_dist": "m",
    "insert_err": "m",
}


def _style_paper_axis(ax, *, grid_axis: str | None = None, grid_alpha: float = 0.16) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(labelsize=PAPER_TICK_SIZE, width=0.7, length=3.0)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color="#cfcfcf", linewidth=0.6, alpha=grid_alpha)


def _legend(ax, *, ncol: int = 1, loc: str = "best") -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = {}
    for handle, label in zip(handles, labels):
        text = str(label).strip()
        if text and not text.startswith("_") and text not in by_label:
            by_label[text] = handle
    if by_label:
        ax.legend(
            by_label.values(),
            by_label.keys(),
            loc=loc,
            fontsize=PAPER_LEGEND_SIZE,
            frameon=False,
            handlelength=1.35,
            borderpad=0.2,
            ncol=ncol,
        )


def _feature_axis_label(feature_name: str, *, raw: bool = True) -> str:
    base_name, scale = _parse_feature_scale(str(feature_name))
    unit = S4_FEATURE_UNITS.get(str(base_name), "")
    prefix = "raw " if raw else ""
    if scale != 1.0:
        label = f"{base_name} (score x {scale:g})"
    else:
        label = str(feature_name)
    return f"{prefix}{label} ({unit})" if unit else f"{prefix}{label}"


@dataclass(frozen=True)
class ScoreMethod:
    name: str
    boundary_quantile: float
    baseline: str


@dataclass
class FitResult:
    score: float
    active_nll: float
    baseline_nll: float
    nll_gain: float
    b: float
    q05: float
    q50: float
    q95: float
    slack_scale: float
    slack_scale_raw: float
    slack_nu: float
    slack_q50: float
    slack_q75: float
    slack_q95: float
    baseline_mu: float
    baseline_scale: float
    baseline_nu: float
    baseline_fit: str
    n: int
    n_used: int
    direction: str = "lower"


@dataclass
class StudentTMLEFit:
    mu: float
    scale: float
    nu: float
    nll: float
    converged: bool


def _student_t_pdf(x: np.ndarray, *, mu: float, scale: float, nu: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    scale = max(float(scale), 1e-12)
    nu = max(float(nu), 1e-12)
    z = (x - float(mu)) / scale
    log_norm = (
        math.lgamma(0.5 * (nu + 1.0))
        - math.lgamma(0.5 * nu)
        - 0.5 * math.log(nu * math.pi)
        - math.log(scale)
    )
    return np.exp(log_norm - 0.5 * (nu + 1.0) * np.log1p((z * z) / nu))


def _student_t_profile_params(xs: np.ndarray, *, nu: float) -> tuple[float, float, float]:
    xs = np.asarray(xs, dtype=float).reshape(-1)
    if xs.size == 0:
        return 0.0, 1.0, float(nu)
    nu = max(float(nu), 1e-6)
    q10, q50, q90 = np.quantile(xs, [0.10, 0.50, 0.90])
    unit_width = float(student_t_dist.ppf(0.90, df=nu) - student_t_dist.ppf(0.10, df=nu))
    scale = float(q90 - q10) / max(unit_width, 1e-12)
    return float(q50), max(float(scale), 1e-12), float(nu)


def _student_t_profile_nll(xs: np.ndarray, *, mu: float, scale: float, nu: float) -> float:
    xs = np.asarray(xs, dtype=float).reshape(-1)
    if xs.size == 0:
        return 0.0
    pdf = np.maximum(_student_t_pdf(xs, mu=mu, scale=scale, nu=nu), 1e-300)
    return float(-np.mean(np.log(pdf)))


def _student_t_mle_params(xs: np.ndarray, *, nu: float) -> StudentTMLEFit:
    xs = np.asarray(xs, dtype=float).reshape(-1)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return StudentTMLEFit(mu=0.0, scale=1.0, nu=float(nu), nll=0.0, converged=False)

    nu = max(float(nu), 1e-6)
    init_mu, init_scale, _ = _student_t_profile_params(xs, nu=nu)
    init_nll = _student_t_profile_nll(xs, mu=init_mu, scale=init_scale, nu=nu)
    if xs.size < 3:
        return StudentTMLEFit(mu=init_mu, scale=init_scale, nu=nu, nll=init_nll, converged=False)

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    x_span = max(x_max - x_min, float(np.std(xs)), init_scale, 1e-12)
    mu_lo = x_min - 5.0 * x_span
    mu_hi = x_max + 5.0 * x_span
    scale_lo = max(1e-12, x_span * 1e-8)
    scale_hi = max(scale_lo * 10.0, x_span * 20.0)

    starts: list[tuple[float, float]] = [
        (init_mu, init_scale),
        (float(np.mean(xs)), max(float(np.std(xs)), scale_lo)),
        (float(np.quantile(xs, 0.25)), init_scale),
        (float(np.quantile(xs, 0.75)), init_scale),
    ]

    best_mu = float(init_mu)
    best_scale = float(init_scale)
    best_nll = float(init_nll)
    best_converged = False

    def objective(theta: np.ndarray) -> float:
        mu = float(theta[0])
        scale = float(np.exp(theta[1]))
        return _student_t_profile_nll(xs, mu=mu, scale=scale, nu=nu)

    bounds = [(mu_lo, mu_hi), (math.log(scale_lo), math.log(scale_hi))]
    for start_mu, start_scale in starts:
        theta0 = np.asarray([float(start_mu), math.log(max(float(start_scale), scale_lo))], dtype=float)
        try:
            result = minimize(objective, theta0, method="L-BFGS-B", bounds=bounds)
        except Exception:
            continue
        if not np.isfinite(result.fun):
            continue
        if float(result.fun) < best_nll:
            best_mu = float(result.x[0])
            best_scale = float(np.exp(result.x[1]))
            best_nll = float(result.fun)
            best_converged = bool(result.success)

    return StudentTMLEFit(mu=best_mu, scale=best_scale, nu=nu, nll=best_nll, converged=best_converged)


def _half_t_profile_params(slack: np.ndarray, *, nu: float, scale_quantile: float) -> tuple[float, float]:
    slack = np.maximum(np.asarray(slack, dtype=float).reshape(-1), 0.0)
    if slack.size == 0:
        return 1.0, float(nu)
    nu = max(float(nu), 1e-6)
    q = float(np.clip(scale_quantile, 0.5, 0.99))
    slack_q = float(np.quantile(slack, q))
    unit_q = float(student_t_dist.ppf(0.5 + 0.5 * q, df=nu))
    scale = slack_q / max(unit_q, 1e-12)
    return max(float(scale), 1e-12), float(nu)


def _half_t_profile_nll(slack: np.ndarray, *, scale: float, nu: float) -> float:
    slack = np.maximum(np.asarray(slack, dtype=float).reshape(-1), 0.0)
    if slack.size == 0:
        return 0.0
    pdf = 2.0 * _student_t_pdf(slack, mu=0.0, scale=scale, nu=nu)
    return float(-np.mean(np.log(np.maximum(pdf, 1e-300))))


def _softplus_stable(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _log_sigmoid_stable(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return -_softplus_stable(-z)


def _soft_half_t_profile_nll_on_x(
    xs: np.ndarray,
    *,
    b: float,
    scale: float,
    nu: float,
    softness: float,
    direction: str,
) -> float:
    xs = np.asarray(xs, dtype=float).reshape(-1)
    if xs.size == 0:
        return 0.0
    sign = 1.0 if str(direction).lower() == "lower" else -1.0
    softness = max(float(softness), 0.0)
    signed_slack = sign * (xs - float(b))
    if softness <= 1e-12:
        pdf = np.zeros_like(xs, dtype=float)
        ok = signed_slack >= 0.0
        if np.any(ok):
            pdf[ok] = 2.0 * _student_t_pdf(signed_slack[ok], mu=0.0, scale=scale, nu=nu)
        return float(-np.mean(np.log(np.maximum(pdf, 1e-300))))

    # Anchored soft boundary: exact half-t on the legal side, exponential leakage
    # on the illegal side. The mode stays exactly at b.
    half_t_at_zero = float(2.0 * _student_t_pdf(np.asarray([0.0]), mu=0.0, scale=scale, nu=nu)[0])
    log_partition = math.log1p(max(half_t_at_zero * softness, 0.0))
    log_pdf_x = np.empty_like(xs, dtype=float)
    ok = signed_slack >= 0.0
    if np.any(ok):
        legal_pdf = 2.0 * _student_t_pdf(signed_slack[ok], mu=0.0, scale=scale, nu=nu)
        log_pdf_x[ok] = np.log(np.maximum(legal_pdf, 1e-300)) - log_partition
    if np.any(~ok):
        log_pdf_x[~ok] = math.log(max(half_t_at_zero, 1e-300)) + signed_slack[~ok] / softness - log_partition
    return float(-np.mean(log_pdf_x))


def _trim_for_inequality(values: np.ndarray, *, trim_fraction: float, trim_min_n: int) -> np.ndarray:
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    frac = float(np.clip(trim_fraction, 0.0, 0.45))
    if vals.size < int(trim_min_n) or frac <= 0.0:
        return vals
    trim_n = int(np.floor(float(vals.size) * frac))
    if trim_n <= 0 or vals.size - trim_n < 3:
        return vals
    mu, scale, nu = _student_t_profile_params(vals, nu=3.0)
    pdf = np.maximum(_student_t_pdf(vals, mu=mu, scale=scale, nu=nu), 1e-300)
    nll = -np.log(pdf)
    keep_n = int(vals.size - trim_n)
    keep = np.argsort(nll, kind="mergesort")[:keep_n]
    kept = vals[keep]
    return kept if kept.size >= 3 else vals


def parse_method(text: str) -> ScoreMethod:
    raw = str(text).strip().lower()
    aliases = {
        "current": "seg_q05",
        "fast_fit_minus_baseline": "seg_q05",
        "q05": "seg_q05",
        "q10": "seg_q10",
    }
    raw = aliases.get(raw, raw)
    parts = raw.split("_q")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid method '{text}'. Use e.g. seg_q05 or seg_q10. Optional diagnostics: global_q05, global_q10, none_q05."
        )
    baseline = parts[0]
    if baseline not in {"seg", "global", "none"}:
        raise ValueError("Method baseline must be one of seg/global/none.")
    try:
        q = float("0." + parts[1])
    except ValueError as exc:
        raise ValueError(f"Invalid quantile suffix in method '{text}'.") from exc
    if q < 0.0 or q > 0.45:
        raise ValueError("Lower-bound quantile must be in [0, 0.45].")
    return ScoreMethod(name=raw, boundary_quantile=float(q), baseline=baseline)


def fit_half_t_score(
    values: np.ndarray,
    *,
    method: ScoreMethod,
    global_baseline_values: np.ndarray | None,
    trim_fraction: float,
    trim_min_n: int,
    nu: float,
    soft_boundary_scale: float = DEFAULT_SOFT_BOUNDARY_SCALE,
    direction: str = DEFAULT_DIRECTION,
    baseline_fit: str = DEFAULT_BASELINE_FIT,
    half_t_scale_quantile: float = DEFAULT_HALF_T_SCALE_QUANTILE,
    observation_noise_scale: float = DEFAULT_OBSERVATION_NOISE_SCALE,
) -> FitResult:
    direction_l = str(direction).lower()
    baseline_fit_l = str(baseline_fit).lower()
    obs_noise = max(float(observation_noise_scale), 0.0)
    if baseline_fit_l not in {"robust", "mle"}:
        raise ValueError(f"Invalid baseline_fit={baseline_fit!r}; expected robust or mle.")
    if direction_l == "auto":
        lower_fit = fit_half_t_score(
            values,
            method=method,
            global_baseline_values=global_baseline_values,
            trim_fraction=trim_fraction,
            trim_min_n=trim_min_n,
            nu=nu,
            soft_boundary_scale=soft_boundary_scale,
            direction="lower",
            baseline_fit=baseline_fit_l,
            half_t_scale_quantile=half_t_scale_quantile,
            observation_noise_scale=obs_noise,
        )
        upper_fit = fit_half_t_score(
            values,
            method=method,
            global_baseline_values=global_baseline_values,
            trim_fraction=trim_fraction,
            trim_min_n=trim_min_n,
            nu=nu,
            soft_boundary_scale=soft_boundary_scale,
            direction="upper",
            baseline_fit=baseline_fit_l,
            half_t_scale_quantile=half_t_scale_quantile,
            observation_noise_scale=obs_noise,
        )
        lower_gain = float(lower_fit.nll_gain) if np.isfinite(lower_fit.nll_gain) else float("-inf")
        upper_gain = float(upper_fit.nll_gain) if np.isfinite(upper_fit.nll_gain) else float("-inf")
        return upper_fit if upper_gain > lower_gain else lower_fit
    if direction_l not in {"lower", "upper"}:
        raise ValueError(f"Invalid direction={direction!r}; expected lower, upper, or auto.")

    raw_vals = np.asarray(values, dtype=float).reshape(-1)
    raw_vals = raw_vals[np.isfinite(raw_vals)]
    vals = _trim_for_inequality(raw_vals, trim_fraction=trim_fraction, trim_min_n=trim_min_n)
    if vals.size == 0:
        vals = raw_vals
    if vals.size == 0:
        return FitResult(
            score=float("nan"),
            active_nll=float("nan"),
            baseline_nll=float("nan"),
            nll_gain=float("nan"),
            b=float("nan"),
            q05=float("nan"),
            q50=float("nan"),
            q95=float("nan"),
            slack_scale=float("nan"),
            slack_scale_raw=float("nan"),
            slack_nu=float("nan"),
            slack_q50=float("nan"),
            slack_q75=float("nan"),
            slack_q95=float("nan"),
            baseline_mu=float("nan"),
            baseline_scale=float("nan"),
            baseline_nu=float(nu),
            baseline_fit=baseline_fit_l if method.baseline != "none" else "none",
            n=0,
            n_used=0,
            direction=direction_l,
        )

    q05, q50, q95 = np.quantile(vals, [0.05, 0.5, 0.95])
    if method.baseline == "global" and global_baseline_values is not None:
        baseline_source = np.asarray(global_baseline_values, dtype=float).reshape(-1)
        baseline_source = baseline_source[np.isfinite(baseline_source)]
    else:
        baseline_source = vals
    baseline_mu, baseline_scale, baseline_nu = _student_t_profile_params(baseline_source, nu=nu)
    baseline_mle_nll: float | None = None
    if method.baseline != "none" and baseline_fit_l == "mle":
        baseline_mle = _student_t_mle_params(baseline_source, nu=nu)
        baseline_mu = baseline_mle.mu
        baseline_scale = baseline_mle.scale
        baseline_nu = baseline_mle.nu
        if obs_noise <= 0.0 and baseline_source.size == vals.size and np.array_equal(baseline_source, vals):
            baseline_mle_nll = baseline_mle.nll
    baseline_scale = max(float(baseline_scale), obs_noise, 1e-12)

    boundary_q = method.boundary_quantile if direction_l == "lower" else 1.0 - method.boundary_quantile
    b = float(np.quantile(vals, boundary_q))
    sign = 1.0 if direction_l == "lower" else -1.0
    slack = np.maximum(sign * (vals - b), 0.0)
    slack_scale_raw, slack_nu = _half_t_profile_params(slack, nu=nu, scale_quantile=half_t_scale_quantile)
    slack_scale = max(float(slack_scale_raw), obs_noise, 1e-12)
    active_nll = _soft_half_t_profile_nll_on_x(
        vals,
        b=b,
        scale=slack_scale,
        nu=slack_nu,
        softness=max(float(soft_boundary_scale), 0.0) * slack_scale,
        direction=direction_l,
    )

    if method.baseline == "none":
        baseline_mu = float("nan")
        baseline_scale = float("nan")
        baseline_nu = float(nu)
        baseline_nll = float("nan")
        nll_gain = float("nan")
        score = float(math.exp(max(min(active_nll, 50.0), -50.0)))
    else:
        baseline_nll = (
            float(baseline_mle_nll)
            if baseline_mle_nll is not None
            else _student_t_profile_nll(vals, mu=baseline_mu, scale=baseline_scale, nu=baseline_nu)
        )
        score = float(active_nll - baseline_nll)
        nll_gain = float(baseline_nll - active_nll)

    sq50, sq75, sq95 = np.quantile(slack, [0.5, 0.75, 0.95])
    return FitResult(
        score=float(score),
        active_nll=float(active_nll),
        baseline_nll=float(baseline_nll),
        nll_gain=float(nll_gain),
        b=float(b),
        q05=float(q05),
        q50=float(q50),
        q95=float(q95),
        slack_scale=float(slack_scale),
        slack_scale_raw=float(slack_scale_raw),
        slack_nu=float(slack_nu),
        slack_q50=float(sq50),
        slack_q75=float(sq75),
        slack_q95=float(sq95),
        baseline_mu=float(baseline_mu),
        baseline_scale=float(baseline_scale),
        baseline_nu=float(baseline_nu),
        baseline_fit=baseline_fit_l if method.baseline != "none" else "none",
        n=int(raw_vals.size),
        n_used=int(vals.size),
        direction=direction_l,
    )


fit_lower_half_t_score = fit_half_t_score


def fit_to_dict(result: FitResult) -> dict[str, float | int]:
    return {
        "direction": result.direction,
        "score": result.score,
        "active_nll": result.active_nll,
        "baseline_nll": result.baseline_nll,
        "nll_gain": result.nll_gain,
        "b": result.b,
        "q05": result.q05,
        "q50": result.q50,
        "q95": result.q95,
        "slack_scale": result.slack_scale,
        "slack_scale_raw": result.slack_scale_raw,
        "slack_nu": result.slack_nu,
        "slack_q50": result.slack_q50,
        "slack_q75": result.slack_q75,
        "slack_q95": result.slack_q95,
        "baseline_mu": result.baseline_mu,
        "baseline_scale": result.baseline_scale,
        "baseline_nu": result.baseline_nu,
        "baseline_fit": result.baseline_fit,
        "n": result.n,
        "n_used": result.n_used,
    }


def _parse_feature_scale(feature_name: str) -> tuple[str, float]:
    text = str(feature_name).strip()
    match = re.fullmatch(r"(.+?)[_*xX]+scale[_-]?([0-9.+-eE]+)", text)
    if match:
        scale = float(match.group(2))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Feature scale must be positive and finite; got {scale}.")
        return match.group(1), scale
    match = re.fullmatch(r"(.+?)[_*xX]+([0-9.+-eE]+)", text)
    if match:
        scale = float(match.group(2))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Feature scale must be positive and finite; got {scale}.")
        return match.group(1), scale
    match = re.fullmatch(r"([0-9.+-eE]+)[xX_*]+(.+)", text)
    if match:
        scale = float(match.group(1))
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Feature scale must be positive and finite; got {scale}.")
        return match.group(2), scale
    return text, 1.0


def _scaled_feature_name(base_name: str, scale: float) -> str:
    scale_f = float(scale)
    if not np.isfinite(scale_f) or scale_f <= 0.0:
        raise ValueError(f"Feature scale must be positive and finite; got {scale}.")
    return f"{str(base_name)}_x{scale_f:g}"


def _find_feature_column(schema: Iterable[dict], name: str) -> int:
    aliases = {
        "normal_force": "force",
        "force": "normal_force",
    }
    requested = str(name)
    candidates = [requested]
    alias = aliases.get(requested)
    if alias is not None and alias not in candidates:
        candidates.append(alias)
    for i, spec in enumerate(schema):
        if str(spec.get("name", "")) in candidates:
            return int(spec.get("column_idx", i))
    raise KeyError(f"Feature '{name}' not found in schema.")


def load_s4_stage_pair_feature(args: argparse.Namespace, *, feature_name: str | None = None) -> tuple[list[np.ndarray], list[np.ndarray], dict]:
    env_kwargs = dict(DEFAULT_ENV_KWARGS)
    demo_index = int(args.demo_index)
    n_demos = int(args.n_demos) if args.n_demos is not None else demo_index + 1
    boundary_index = int(args.boundary_index)
    if boundary_index < 1:
        raise ValueError(f"--boundary-index is 1-based and must be >= 1; got {boundary_index}.")
    left_label = boundary_index - 1
    right_label = boundary_index
    if n_demos <= demo_index:
        raise ValueError(f"--n-demos must be greater than --demo-index; got n_demos={n_demos}, demo_index={demo_index}.")
    bundle = load_env(
        "S4SlideInsert",
        n_demos=n_demos,
        seed=int(args.seed),
        **env_kwargs,
    )
    schema = bundle.feature_schema or bundle.env.get_feature_schema()
    requested_feature_name = str(feature_name if feature_name is not None else args.feature)
    base_feature_name, feature_scale = _parse_feature_scale(requested_feature_name)
    feature_col = _find_feature_column(schema, base_feature_name)

    global_feature_values = []
    for demo in bundle.demos:
        F_full = np.asarray(bundle.env.compute_all_features_matrix(np.asarray(demo, dtype=float)), dtype=float)
        col = np.asarray(F_full[:, feature_col], dtype=float)
        global_feature_values.append(col[np.isfinite(col)])
    global_feature_values_arr = np.concatenate(global_feature_values, axis=0)
    feature_global_mean = float(np.mean(global_feature_values_arr))
    feature_global_std = float(np.std(global_feature_values_arr) + 1e-8)

    values_by_demo: list[np.ndarray] = []
    labels_by_demo: list[np.ndarray] = []
    demo_indices = [demo_index]
    for demo_idx in demo_indices:
        demo = np.asarray(bundle.demos[demo_idx], dtype=float)
        labels = np.asarray(bundle.true_labels[demo_idx], dtype=int)
        keep = np.logical_or(labels == left_label, labels == right_label)
        if not np.any(keep):
            available = sorted(int(x) + 1 for x in np.unique(labels))
            raise ValueError(
                f"Demo {demo_idx} has no samples for boundary {boundary_index} "
                f"(true stages {boundary_index}+{boundary_index + 1}); available true stages are {available}."
            )
        F = np.asarray(bundle.env.compute_all_features_matrix(demo), dtype=float)
        values_by_demo.append(np.asarray(F[keep, feature_col], dtype=float))
        pair_labels = np.where(np.asarray(labels[keep], dtype=int) == left_label, 0, 1)
        labels_by_demo.append(np.asarray(pair_labels, dtype=int))

    meta = {
        "env": "S4SlideInsert",
        "seed": int(args.seed),
        "env_kwargs": env_kwargs,
        "n_demos_loaded": n_demos,
        "n_demos_used": len(values_by_demo),
        "demo_indices": demo_indices,
        "feature": requested_feature_name,
        "base_feature": base_feature_name,
        "feature_scale": float(feature_scale),
        "raw_feature_scale": 1.0,
        "score_feature_scale": float(feature_scale),
        "feature_column": int(feature_col),
        "feature_global_mean": float(feature_global_mean),
        "feature_global_std": float(feature_global_std),
        "feature_standardization_scope": "loaded_demos_full_trajectories",
        "feature_standardization_n": int(global_feature_values_arr.size),
        "boundary_index": int(boundary_index),
        "true_stage_pair": [int(boundary_index), int(boundary_index + 1)],
        "source_label_pair": [int(left_label), int(right_label)],
    }
    return values_by_demo, labels_by_demo, meta


def standardize_values(values_by_demo: list[np.ndarray], *, mean: float, std: float) -> tuple[list[np.ndarray], float, float]:
    mean = float(mean)
    std = float(std)
    return [(v - mean) / std for v in values_by_demo], mean, std


def candidate_cut_counts(values_by_demo: list[np.ndarray], *, min_len: int) -> np.ndarray:
    if len(values_by_demo) != 1:
        raise ValueError("This quick script is single-demo only; expected exactly one demo.")
    length = int(len(values_by_demo[0]))
    min_len_i = int(min_len)
    if length < 2 * min_len_i:
        raise ValueError(f"Selected stage pair needs at least {2 * min_len_i} samples; got {length}.")
    return np.arange(min_len_i, length - min_len_i + 1, dtype=int)


def split_by_cut_count(values_by_demo: list[np.ndarray], cut_n: int, *, min_len: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    if len(values_by_demo) != 1:
        raise ValueError("This quick script is single-demo only; expected exactly one demo.")
    vals = np.asarray(values_by_demo[0], dtype=float)
    length = int(len(vals))
    cut_n_i = int(cut_n)
    min_len_i = int(min_len)
    if cut_n_i < min_len_i or cut_n_i > length - min_len_i:
        raise ValueError(f"Invalid cut_n={cut_n_i}; expected [{min_len_i}, {length - min_len_i}].")
    return vals[:cut_n_i], vals[cut_n_i:], [cut_n_i]


def selection_metric_description(selection_objective: str) -> str:
    objective = str(selection_objective).lower()
    if objective == "total-margin":
        return "max_split_total_margin=sum(stage_len*max(stage_gain-gain_threshold,0))"
    if objective == "selected-loglik":
        return "max_split_selected_loglik=-sum(stage_len*(active_nll if gain>threshold else baseline_nll))"
    if objective == "active-loglik":
        return "max_split_active_loglik=sum(stage_len*(-active_nll) if gain>threshold else 0)"
    raise ValueError(f"Unknown selection objective: {selection_objective}")


def true_cut_fractions(labels_by_demo: list[np.ndarray]) -> tuple[list[float], list[int]]:
    fracs = []
    counts = []
    for labels in labels_by_demo:
        n0 = int(np.count_nonzero(np.asarray(labels, dtype=int) == 0))
        counts.append(n0)
        fracs.append(float(n0) / float(len(labels)))
    return fracs, counts


def scan_methods(
    values_by_demo_score: list[np.ndarray],
    labels_by_demo: list[np.ndarray],
    methods: list[ScoreMethod],
    *,
    min_len: int,
    trim_fraction: float,
    trim_min_n: int,
    nu: float,
    gain_threshold: float,
    soft_boundary_scale: float,
    direction: str,
    baseline_fit: str,
    half_t_scale_quantile: float,
    selection_objective: str,
    observation_noise_scale: float,
) -> tuple[list[dict], dict[str, dict]]:
    selection_objective_l = str(selection_objective).lower()
    if selection_objective_l not in {"total-margin", "selected-loglik", "active-loglik"}:
        raise ValueError("selection_objective must be one of: total-margin, selected-loglik, active-loglik.")
    cut_candidates = candidate_cut_counts(values_by_demo_score, min_len=min_len)
    all_values = np.concatenate(values_by_demo_score, axis=0)
    true_fracs, true_counts = true_cut_fractions(labels_by_demo)
    rows: list[dict] = []
    summary: dict[str, dict] = {}

    for method in methods:
        merged = fit_lower_half_t_score(
            all_values,
            method=method,
            global_baseline_values=all_values,
            trim_fraction=trim_fraction,
            trim_min_n=trim_min_n,
            nu=nu,
            soft_boundary_scale=soft_boundary_scale,
            direction=direction,
            baseline_fit=baseline_fit,
            half_t_scale_quantile=half_t_scale_quantile,
            observation_noise_scale=observation_noise_scale,
        )
        best_row = None
        best_value = -float("inf")
        total_len = int(len(values_by_demo_score[0]))
        for cut_n in cut_candidates:
            stage1, stage2, cut_counts = split_by_cut_count(values_by_demo_score, int(cut_n), min_len=min_len)
            frac = float(cut_n) / float(total_len)
            fit1 = fit_lower_half_t_score(
                stage1,
                method=method,
                global_baseline_values=all_values,
                trim_fraction=trim_fraction,
                trim_min_n=trim_min_n,
                nu=nu,
                soft_boundary_scale=soft_boundary_scale,
                direction=direction,
                baseline_fit=baseline_fit,
                half_t_scale_quantile=half_t_scale_quantile,
                observation_noise_scale=observation_noise_scale,
            )
            fit2 = fit_lower_half_t_score(
                stage2,
                method=method,
                global_baseline_values=all_values,
                trim_fraction=trim_fraction,
                trim_min_n=trim_min_n,
                nu=nu,
                soft_boundary_scale=soft_boundary_scale,
                direction=direction,
                baseline_fit=baseline_fit,
                half_t_scale_quantile=half_t_scale_quantile,
                observation_noise_scale=observation_noise_scale,
            )
            split_sum = float(fit1.score * fit1.n_used + fit2.score * fit2.n_used)
            split_weighted_mean = split_sum / float(max(fit1.n_used + fit2.n_used, 1))
            stage1_gain = float(fit1.nll_gain) if np.isfinite(fit1.nll_gain) else float(-fit1.score)
            stage2_gain = float(fit2.nll_gain) if np.isfinite(fit2.nll_gain) else float(-fit2.score)
            merged_gain = float(merged.nll_gain) if np.isfinite(merged.nll_gain) else float(-merged.score)
            split_gain_sum = float(stage1_gain * fit1.n_used + stage2_gain * fit2.n_used)
            split_gain_weighted_mean = split_gain_sum / float(max(fit1.n_used + fit2.n_used, 1))
            stage1_margin = float(max(stage1_gain - gain_threshold, 0.0))
            stage2_margin = float(max(stage2_gain - gain_threshold, 0.0))
            stage1_total_margin = float(stage1_margin * fit1.n_used)
            stage2_total_margin = float(stage2_margin * fit2.n_used)
            split_total_margin = float(stage1_total_margin + stage2_total_margin)
            split_activation_margin = split_total_margin / float(max(fit1.n_used + fit2.n_used, 1))
            merged_margin = float(max(merged_gain - gain_threshold, 0.0))
            merged_total_margin = float(merged_margin * merged.n_used)
            stage1_selected_active = bool(stage1_gain > gain_threshold)
            stage2_selected_active = bool(stage2_gain > gain_threshold)
            merged_selected_active = bool(merged_gain > gain_threshold)
            stage1_selected_nll = float(fit1.active_nll if stage1_selected_active else fit1.baseline_nll)
            stage2_selected_nll = float(fit2.active_nll if stage2_selected_active else fit2.baseline_nll)
            merged_selected_nll = float(merged.active_nll if merged_selected_active else merged.baseline_nll)
            stage1_selected_loglik = float(-stage1_selected_nll * fit1.n_used)
            stage2_selected_loglik = float(-stage2_selected_nll * fit2.n_used)
            split_selected_nll_sum = float(stage1_selected_nll * fit1.n_used + stage2_selected_nll * fit2.n_used)
            split_selected_nll_weighted_mean = split_selected_nll_sum / float(max(fit1.n_used + fit2.n_used, 1))
            split_selected_loglik = float(-split_selected_nll_sum)
            merged_selected_loglik = float(-merged_selected_nll * merged.n_used)
            stage1_active_loglik = float(-fit1.active_nll * fit1.n_used) if stage1_selected_active else 0.0
            stage2_active_loglik = float(-fit2.active_nll * fit2.n_used) if stage2_selected_active else 0.0
            split_active_loglik = float(stage1_active_loglik + stage2_active_loglik)
            merged_active_loglik = float(-merged.active_nll * merged.n_used) if merged_selected_active else 0.0
            if selection_objective_l == "total-margin":
                selection_value = split_total_margin
            elif selection_objective_l == "selected-loglik":
                selection_value = split_selected_loglik
            else:
                selection_value = split_active_loglik
            row = {
                "method": method.name,
                "half_t_scale_quantile": float(half_t_scale_quantile),
                "selection_objective": str(selection_objective_l),
                "observation_noise_scale": float(observation_noise_scale),
                "cut_fraction": float(frac),
                "cut_n": int(cut_counts[0]),
                "mean_cut_n": float(np.mean(cut_counts)),
                "cut_counts": ";".join(str(x) for x in cut_counts),
                "true_cut_fraction": float(true_fracs[0]),
                "true_cut_n": int(true_counts[0]),
                "true_cut_fraction_mean": float(np.mean(true_fracs)),
                "true_cut_fraction_std": float(np.std(true_fracs)),
                "true_cut_counts": ";".join(str(x) for x in true_counts),
                "stage1_score": float(fit1.score),
                "stage2_score": float(fit2.score),
                "stage1_direction": str(fit1.direction),
                "stage2_direction": str(fit2.direction),
                "merged_direction": str(merged.direction),
                "stage1_gain": float(stage1_gain),
                "stage2_gain": float(stage2_gain),
                "split_gain_sum": float(split_gain_sum),
                "split_gain_weighted_mean": float(split_gain_weighted_mean),
                "merged_gain": float(merged_gain),
                "split_score_sum": float(split_sum),
                "split_score_weighted_mean": float(split_weighted_mean),
                "merged_score": float(merged.score),
                "stage1_activation_margin": float(stage1_margin),
                "stage2_activation_margin": float(stage2_margin),
                "stage1_total_margin": float(stage1_total_margin),
                "stage2_total_margin": float(stage2_total_margin),
                "split_total_margin": float(split_total_margin),
                "split_activation_margin_sum": float(split_total_margin),
                "split_activation_margin": float(split_activation_margin),
                "merged_activation_margin": float(merged_margin),
                "merged_total_margin": float(merged_total_margin),
                "stage1_selected_active": int(stage1_selected_active),
                "stage2_selected_active": int(stage2_selected_active),
                "merged_selected_active": int(merged_selected_active),
                "stage1_selected_nll": float(stage1_selected_nll),
                "stage2_selected_nll": float(stage2_selected_nll),
                "merged_selected_nll": float(merged_selected_nll),
                "stage1_selected_loglik": float(stage1_selected_loglik),
                "stage2_selected_loglik": float(stage2_selected_loglik),
                "split_selected_nll_sum": float(split_selected_nll_sum),
                "split_selected_nll_weighted_mean": float(split_selected_nll_weighted_mean),
                "split_selected_loglik": float(split_selected_loglik),
                "merged_selected_loglik": float(merged_selected_loglik),
                "stage1_active_loglik": float(stage1_active_loglik),
                "stage2_active_loglik": float(stage2_active_loglik),
                "split_active_loglik": float(split_active_loglik),
                "merged_active_loglik": float(merged_active_loglik),
                "selection_value": float(selection_value),
                "stage1_active_nll": float(fit1.active_nll),
                "stage2_active_nll": float(fit2.active_nll),
                "stage1_baseline_nll": float(fit1.baseline_nll),
                "stage2_baseline_nll": float(fit2.baseline_nll),
                "stage1_b": float(fit1.b),
                "stage2_b": float(fit2.b),
                "stage1_slack_scale": float(fit1.slack_scale),
                "stage2_slack_scale": float(fit2.slack_scale),
                "stage1_slack_scale_raw": float(fit1.slack_scale_raw),
                "stage2_slack_scale_raw": float(fit2.slack_scale_raw),
                "stage1_baseline_scale": float(fit1.baseline_scale),
                "stage2_baseline_scale": float(fit2.baseline_scale),
                "stage1_baseline_fit": str(fit1.baseline_fit),
                "stage2_baseline_fit": str(fit2.baseline_fit),
                "stage1_n": int(fit1.n),
                "stage2_n": int(fit2.n),
                "stage1_n_used": int(fit1.n_used),
                "stage2_n_used": int(fit2.n_used),
            }
            rows.append(row)
            if selection_value > best_value:
                best_value = selection_value
                best_row = row
        summary[method.name] = {
            "best": dict(best_row or {}),
            "selection_metric": selection_metric_description(selection_objective_l),
            "selection_objective": str(selection_objective_l),
            "gain_threshold": float(gain_threshold),
            "soft_boundary_scale": float(soft_boundary_scale),
            "direction": str(direction),
            "baseline_fit": str(baseline_fit),
            "half_t_scale_quantile": float(half_t_scale_quantile),
            "observation_noise_scale": float(observation_noise_scale),
            "merged": fit_to_dict(merged),
        }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _soft_half_t_pdf_on_x(grid: np.ndarray, fit: FitResult, *, soft_boundary_scale: float) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    softness = max(float(soft_boundary_scale), 0.0) * max(float(fit.slack_scale), 1e-12)
    sign = 1.0 if str(fit.direction).lower() == "lower" else -1.0
    signed = sign * (grid - float(fit.b))
    if softness <= 1e-12:
        out = np.full_like(grid, np.nan, dtype=float)
        mask = signed >= 0.0
        out[mask] = 2.0 * _student_t_pdf(
            signed[mask],
            mu=0.0,
            scale=fit.slack_scale,
            nu=fit.slack_nu,
        )
        return out
    half_t_at_zero = float(
        2.0 * _student_t_pdf(np.asarray([0.0]), mu=0.0, scale=fit.slack_scale, nu=fit.slack_nu)[0]
    )
    partition = 1.0 + max(half_t_at_zero * softness, 0.0)
    out = np.empty_like(grid, dtype=float)
    mask = signed >= 0.0
    if np.any(mask):
        out[mask] = 2.0 * _student_t_pdf(
            signed[mask],
            mu=0.0,
            scale=fit.slack_scale,
            nu=fit.slack_nu,
        )
    if np.any(~mask):
        out[~mask] = half_t_at_zero * np.exp(np.maximum(signed[~mask] / softness, -745.0))
    return out / partition


def plot_combined_report(
    path: Path,
    *,
    rows: list[dict],
    methods: list[ScoreMethod],
    values_by_demo_raw: list[np.ndarray],
    labels_by_demo: list[np.ndarray],
    feature_name: str,
    stage_pair_label: str,
    best_method: ScoreMethod,
    best_cut_n: int,
    min_len: int,
    trim_fraction: float,
    trim_min_n: int,
    nu: float,
    score_space: str,
    gain_threshold: float,
    soft_boundary_scale: float,
    direction: str,
    baseline_fit: str,
    half_t_scale_quantile: float,
    selection_objective: str,
    observation_noise_scale: float,
    values_by_demo_plot: list[np.ndarray] | None = None,
    plot_space_label: str = "raw",
    plot_axis_label: str | None = None,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)

    values_by_demo_plot = values_by_demo_raw if values_by_demo_plot is None else values_by_demo_plot
    plot_axis_label = plot_axis_label or _feature_axis_label(feature_name, raw=(plot_space_label == "raw"))
    true_fracs, true_counts = true_cut_fractions(labels_by_demo)
    true_cut_n = int(true_counts[0])
    all_plot = np.concatenate(values_by_demo_plot, axis=0)

    def _plot_fit_at_cut(cut_n: int) -> tuple[np.ndarray, np.ndarray, FitResult, FitResult, list[int]]:
        left_values, right_values, counts = split_by_cut_count(values_by_demo_plot, int(cut_n), min_len=min_len)
        left_fit = fit_lower_half_t_score(
            left_values,
            method=best_method,
            global_baseline_values=all_plot,
            trim_fraction=trim_fraction,
            trim_min_n=trim_min_n,
            nu=nu,
            soft_boundary_scale=soft_boundary_scale,
            direction=direction,
            baseline_fit=baseline_fit,
            half_t_scale_quantile=half_t_scale_quantile,
            observation_noise_scale=observation_noise_scale,
        )
        right_fit = fit_lower_half_t_score(
            right_values,
            method=best_method,
            global_baseline_values=all_plot,
            trim_fraction=trim_fraction,
            trim_min_n=trim_min_n,
            nu=nu,
            soft_boundary_scale=soft_boundary_scale,
            direction=direction,
            baseline_fit=baseline_fit,
            half_t_scale_quantile=half_t_scale_quantile,
            observation_noise_scale=observation_noise_scale,
        )
        return left_values, right_values, left_fit, right_fit, counts

    def _scan_row_at_cut(cut_n: int) -> dict | None:
        for row in rows:
            if str(row.get("method")) == str(best_method.name) and int(row.get("cut_n", -1)) == int(cut_n):
                return row
        return None

    best_stage1_plot, best_stage2_plot, best_fit1, best_fit2, best_cut_counts = _plot_fit_at_cut(best_cut_n)
    true_stage1_plot, true_stage2_plot, true_fit1, true_fit2, true_cut_counts = _plot_fit_at_cut(true_cut_n)
    best_scan_row = _scan_row_at_cut(best_cut_n)
    true_scan_row = _scan_row_at_cut(true_cut_n)
    best_fraction = float(best_cut_n) / float(len(values_by_demo_plot[0]))
    true_fraction = float(true_cut_n) / float(len(values_by_demo_plot[0]))
    fit_merged = fit_lower_half_t_score(
        all_plot,
        method=best_method,
        global_baseline_values=all_plot,
        trim_fraction=trim_fraction,
        trim_min_n=trim_min_n,
        nu=nu,
        soft_boundary_scale=soft_boundary_scale,
        direction=direction,
        baseline_fit=baseline_fit,
        half_t_scale_quantile=half_t_scale_quantile,
        observation_noise_scale=observation_noise_scale,
    )

    n_score_rows = max(len(methods), 1)
    fig_h = 2.05 * n_score_rows + 8.6
    fig = plt.figure(figsize=(11.2, fig_h), constrained_layout=False)
    grid = fig.add_gridspec(
        n_score_rows + 4,
        1,
        height_ratios=[1.0] * n_score_rows + [0.86, 1.48, 1.48, 0.78],
        hspace=0.36,
    )
    fig.suptitle(
        f"S4 {feature_name} inequality-only single-demo cutpoint scan ({stage_pair_label}, {score_space} score space; plot={plot_space_label})",
        fontsize=10,
        y=0.985,
    )

    for row_idx, method in enumerate(methods):
        ax = fig.add_subplot(grid[row_idx, 0])
        mr = [r for r in rows if r["method"] == method.name]
        if not mr:
            ax.axis("off")
            continue
        x = np.asarray([r["cut_n"] for r in mr], dtype=float)
        selection_objective_l = str(selection_objective).lower()
        if selection_objective_l == "selected-loglik":
            y1 = np.asarray([r["stage1_selected_loglik"] for r in mr], dtype=float)
            y2 = np.asarray([r["stage2_selected_loglik"] for r in mr], dtype=float)
            yw = np.asarray([r["split_selected_loglik"] for r in mr], dtype=float)
            merged = float(mr[0]["merged_selected_loglik"])
            y_label = "selected log-likelihood"
            title_value_name = "loglik"
        elif selection_objective_l == "active-loglik":
            y1 = np.asarray([r["stage1_active_loglik"] for r in mr], dtype=float)
            y2 = np.asarray([r["stage2_active_loglik"] for r in mr], dtype=float)
            yw = np.asarray([r["split_active_loglik"] for r in mr], dtype=float)
            merged = float(mr[0]["merged_active_loglik"])
            y_label = "active log-likelihood"
            title_value_name = "active_loglik"
        else:
            y1 = np.asarray([r["stage1_total_margin"] for r in mr], dtype=float)
            y2 = np.asarray([r["stage2_total_margin"] for r in mr], dtype=float)
            yw = np.asarray([r["split_total_margin"] for r in mr], dtype=float)
            merged = float(mr[0]["merged_total_margin"])
            y_label = "total margin"
            title_value_name = "margin"
        ysel = np.asarray([r.get("selection_value", r["split_total_margin"]) for r in mr], dtype=float)
        true_cut_n = int(mr[0]["true_cut_n"])
        best_idx = int(np.nanargmax(ysel))
        ax.plot(x, y1, color=STAGE_COLORS[0], linewidth=1.05, label="stage 1")
        ax.plot(x, y2, color=STAGE_COLORS[1], linewidth=1.05, label="stage 2")
        ax.plot(x, yw, color="#222222", linewidth=1.25, label="weighted")
        ax.axhline(merged, color="#7a7a7a", linewidth=0.85, linestyle=(0, (4, 2)), label="merged")
        if selection_objective_l != "selected-loglik":
            zero_label = "zero" if selection_objective_l == "active-loglik" else "active boundary"
            ax.axhline(0.0, color="#333333", linewidth=0.75, linestyle="-.", alpha=0.65, label=zero_label)
        ax.axvline(true_cut_n, color="#666666", linewidth=0.9, linestyle=":", label="true cut")
        ax.axvline(float(x[best_idx]), color="#222222", linewidth=0.9, linestyle=(0, (5, 2)), label="best cut")
        ax.scatter([x[best_idx]], [yw[best_idx]], color="#222222", s=18, zorder=5)
        ax.set_ylabel(y_label, fontsize=PAPER_LABEL_SIZE)
        ax.set_title(
            f"{method.name}: best cut={int(mr[best_idx]['cut_n'])}, true cut={true_cut_n}, {title_value_name}={yw[best_idx]:.4g}, selection={selection_objective}",
            fontsize=PAPER_TITLE_SIZE,
            loc="left",
            pad=3,
        )
        _style_paper_axis(ax, grid_axis="y", grid_alpha=0.18)
        _legend(ax, ncol=3, loc="best")
        if row_idx == n_score_rows - 1:
            ax.set_xlabel(f"candidate cut index in {stage_pair_label}", fontsize=PAPER_LABEL_SIZE)

    ax = fig.add_subplot(grid[n_score_rows, 0])
    trace = np.asarray(values_by_demo_plot[0], dtype=float)
    trace_t = np.arange(len(trace), dtype=int)
    true_cut_n = int(true_counts[0])
    ax.axvspan(-0.5, true_cut_n - 0.5, color=STAGE_COLORS[0], alpha=0.075, linewidth=0)
    ax.axvspan(true_cut_n - 0.5, len(trace) - 0.5, color=STAGE_COLORS[1], alpha=0.075, linewidth=0)
    ax.plot(trace_t, trace, color="#222222", linewidth=1.05, label=feature_name)
    ax.scatter(trace_t, trace, color="#222222", s=9, linewidth=0, zorder=3)
    ax.axvline(true_cut_n, color="#666666", linewidth=0.9, linestyle=":", label="true cut")
    ax.axvline(int(best_cut_n), color="#222222", linewidth=0.9, linestyle=(0, (5, 2)), label="best cut")
    ax.set_xlim(-0.5, len(trace) - 0.5)
    ax.set_title(f"{plot_space_label} feature trace in {stage_pair_label}", fontsize=PAPER_TITLE_SIZE, loc="left", pad=3)
    ax.set_xlabel(f"sample index in {stage_pair_label}", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel(plot_axis_label, fontsize=PAPER_LABEL_SIZE)
    _style_paper_axis(ax, grid_axis="y", grid_alpha=0.16)
    _legend(ax, ncol=3, loc="best")

    def _plot_distribution_pair(
        grid_slot,
        *,
        title_prefix: str,
        cut_n: int,
        scan_row: dict | None,
        left_values: np.ndarray,
        right_values: np.ndarray,
        left_fit: FitResult,
        right_fit: FitResult,
    ) -> None:
        bins = max(8, int(np.sqrt(max(len(left_values), len(right_values)))))
        dist_grid = grid_slot.subgridspec(
            1,
            3,
            width_ratios=[1.0, 1.0, 0.92],
            wspace=0.30,
        )
        dist_specs = [
            ("stage 1", left_values, left_fit, STAGE_COLORS[0]),
            ("stage 2", right_values, right_fit, STAGE_COLORS[1]),
        ]
        robust_fits: list[tuple[float, float, float, float]] = []
        mle_fits: list[StudentTMLEFit] = []
        for dist_idx, (stage_label, stage_values, fit, color) in enumerate(dist_specs):
            ax = fig.add_subplot(dist_grid[0, dist_idx])
            robust_mu, robust_scale, robust_nu = _student_t_profile_params(stage_values, nu=fit.baseline_nu)
            robust_scale = max(float(robust_scale), float(observation_noise_scale), 1e-12)
            robust_nll = _student_t_profile_nll(stage_values, mu=robust_mu, scale=robust_scale, nu=robust_nu)
            mle_fit = _student_t_mle_params(stage_values, nu=fit.baseline_nu)
            mle_scale = max(float(mle_fit.scale), float(observation_noise_scale), 1e-12)
            mle_nll = _student_t_profile_nll(stage_values, mu=mle_fit.mu, scale=mle_scale, nu=mle_fit.nu)
            mle_fit = StudentTMLEFit(
                mu=mle_fit.mu,
                scale=mle_scale,
                nu=mle_fit.nu,
                nll=mle_nll,
                converged=mle_fit.converged,
            )
            robust_fits.append((robust_mu, robust_scale, robust_nu, robust_nll))
            mle_fits.append(mle_fit)
            stage_lo = float(np.min(stage_values))
            stage_hi = float(np.max(stage_values))
            stage_span = max(stage_hi - stage_lo, 1e-9)
            stage_pad = max(1e-6, 0.10 * stage_span)
            x_min = stage_lo - stage_pad
            x_max = stage_hi + stage_pad
            grid_x = np.linspace(x_min, x_max, 700)
            ax.hist(
                stage_values,
                bins=bins,
                range=(x_min, x_max),
                density=True,
                color=color,
                alpha=0.30,
                edgecolor="white",
                linewidth=0.35,
                label="samples",
            )
            ax_pdf = ax.twinx()
            ax_pdf.plot(
                grid_x,
                _soft_half_t_pdf_on_x(grid_x, fit, soft_boundary_scale=soft_boundary_scale),
                color=color,
                linewidth=1.15,
                label="anchored half-t",
            )
            ax_pdf.plot(
                grid_x,
                _student_t_pdf(grid_x, mu=robust_mu, scale=robust_scale, nu=robust_nu),
                color=color,
                linewidth=0.9,
                linestyle="--",
                label="baseline robust",
            )
            ax_pdf.plot(
                grid_x,
                _student_t_pdf(grid_x, mu=mle_fit.mu, scale=mle_fit.scale, nu=mle_fit.nu),
                color="#222222",
                linewidth=0.9,
                linestyle=(0, (1.3, 1.3)),
                label="baseline MLE",
            )
            ax_pdf.axvline(fit.b, color=color, linewidth=0.75, alpha=0.78, label="b")
            scan_score = np.nan
            if scan_row is not None:
                scan_key = "stage1_score" if dist_idx == 0 else "stage2_score"
                scan_score = float(scan_row[scan_key])
            ax.set_xlim(x_min, x_max)
            ax_pdf.set_xlim(x_min, x_max)
            ax.set_title(
                f"{title_prefix} cut={int(cut_n)} | {stage_label}: {plot_space_label} {feature_name}",
                fontsize=PAPER_TITLE_SIZE,
                loc="left",
                pad=3,
            )
            ax.set_xlabel(plot_axis_label, fontsize=PAPER_LABEL_SIZE)
            if dist_idx == 0:
                ax.set_ylabel("sample density", fontsize=PAPER_LABEL_SIZE)
            ax_pdf.set_ylabel("model density", fontsize=PAPER_LABEL_SIZE)
            _style_paper_axis(ax, grid_axis="y", grid_alpha=0.16)
            ax_pdf.spines["top"].set_visible(False)
            ax_pdf.spines["left"].set_visible(False)
            ax_pdf.spines["right"].set_linewidth(0.8)
            ax_pdf.tick_params(labelsize=PAPER_TICK_SIZE, width=0.7, length=3.0)
            handles, labels = ax.get_legend_handles_labels()
            handles_pdf, labels_pdf = ax_pdf.get_legend_handles_labels()
            by_label = {}
            for handle, label in zip([*handles, *handles_pdf], [*labels, *labels_pdf]):
                text = str(label).strip()
                if text and not text.startswith("_") and text not in by_label:
                    by_label[text] = handle
            if by_label:
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc="upper right",
                    fontsize=PAPER_LEGEND_SIZE,
                    frameon=False,
                    handlelength=1.35,
                    borderpad=0.2,
                    ncol=1,
                )

        def _scan_score(dist_idx: int) -> float:
            if scan_row is None:
                return float("nan")
            scan_key = "stage1_score" if dist_idx == 0 else "stage2_score"
            return float(scan_row[scan_key])

        def _param_block(
            stage_label: str,
            fit: FitResult,
            scan_score: float,
            robust_fit: tuple[float, float, float, float],
            mle_fit: StudentTMLEFit,
        ) -> list[str]:
            scan_gain = float(-scan_score)
            activation_margin = float(max(scan_gain - gain_threshold, 0.0))
            total_margin = float(activation_margin * fit.n_used)
            robust_mu, robust_scale, robust_nu, robust_nll = robust_fit
            robust_gain_ref = float(robust_nll - fit.active_nll)
            mle_gain_ref = float(mle_fit.nll - fit.active_nll)
            return [
                f"{stage_label} dir={fit.direction}, score={scan_score:.5g}, score_base={fit.baseline_fit}",
                f"mean_gain={scan_gain:.5g}, mean_margin=max(gain-thr,0)={activation_margin:.5g}",
                f"total_margin=mean_margin*n_used={total_margin:.5g}",
                f"gain thr={gain_threshold:.4g}, anchored soft={soft_boundary_scale:.4g}*scale",
                f"obs noise scale={observation_noise_scale:.4g}",
                f"anchored half-t b={fit.b:.5g}, s={fit.slack_scale:.5g}, nu={fit.slack_nu:.3g}",
                f"scale raw/q{int(round(100.0 * half_t_scale_quantile))}={fit.slack_scale_raw:.4g}",
                f"robust base mu={robust_mu:.5g}, s={robust_scale:.5g}, nu={robust_nu:.3g}",
                f"MLE base mu={mle_fit.mu:.5g}, s={mle_fit.scale:.5g}, nu={mle_fit.nu:.3g}",
                f"logL ht={-fit.active_nll:.5g}, robust={-robust_nll:.5g}, MLE={-mle_fit.nll:.5g}",
                f"gain ref robust={robust_gain_ref:.5g}, MLE={mle_gain_ref:.5g}",
                f"slack q50/q75/q95={fit.slack_q50:.4g}/{fit.slack_q75:.4g}/{fit.slack_q95:.4g}",
                f"n={fit.n_used}/{fit.n}",
            ]

        param_ax = fig.add_subplot(dist_grid[0, 2])
        param_ax.axis("off")
        param_text = "\n".join(
            [
                f"{title_prefix} cut={int(cut_n)} params",
                *_param_block("stage1", left_fit, _scan_score(0), robust_fits[0], mle_fits[0]),
                "",
                *_param_block("stage2", right_fit, _scan_score(1), robust_fits[1], mle_fits[1]),
            ]
        )
        param_ax.text(
            0.02,
            0.98,
            param_text,
            transform=param_ax.transAxes,
            va="top",
            ha="left",
            fontsize=6.0,
            linespacing=0.90,
            color="#222222",
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#fbfbfb",
                "edgecolor": "#cfcfcf",
                "linewidth": 0.55,
                "alpha": 1.0,
            },
        )

    _plot_distribution_pair(
        grid[n_score_rows + 1, 0],
        title_prefix="Best",
        cut_n=best_cut_n,
        scan_row=best_scan_row,
        left_values=best_stage1_plot,
        right_values=best_stage2_plot,
        left_fit=best_fit1,
        right_fit=best_fit2,
    )
    _plot_distribution_pair(
        grid[n_score_rows + 2, 0],
        title_prefix="True",
        cut_n=true_cut_n,
        scan_row=true_scan_row,
        left_values=true_stage1_plot,
        right_values=true_stage2_plot,
        left_fit=true_fit1,
        right_fit=true_fit2,
    )

    ax = fig.add_subplot(grid[n_score_rows + 3, 0])
    ax.axis("off")
    def _scan_score_line(label: str, row: dict | None) -> str:
        if row is None:
            return f"{label:<10} scan metrics: unavailable"
        return (
            f"{label:<10} scan metrics ({score_space}): "
            f"stage1={float(row['stage1_score']):.6g}, stage2={float(row['stage2_score']):.6g}, "
            f"dir={row.get('stage1_direction', '?')}/{row.get('stage2_direction', '?')}, "
            f"selection_value={float(row.get('selection_value', row['split_total_margin'])):.6g}, "
            f"weighted_gain={float(row['split_gain_weighted_mean']):.6g}, "
            f"mean_margin={float(row['split_activation_margin']):.6g}, "
            f"total_margin={float(row['split_total_margin']):.6g}, "
            f"active_loglik={float(row.get('split_active_loglik', 0.0)):.6g}, "
            f"merged_total={float(row['merged_total_margin']):.6g}"
        )

    lines = [
        f"method={best_method.name} | direction={direction} | baseline_fit={baseline_fit} | selection={selection_objective} | best cut={int(best_cut_counts[0])} ({best_fraction:.4f}) | true cut={int(true_cut_counts[0])} ({true_fraction:.4f})",
        _scan_score_line("best", best_scan_row),
        _scan_score_line("true", true_scan_row),
        (
            f"gain_threshold={gain_threshold:.6g}, soft_boundary_scale={soft_boundary_scale:.6g} | merged dir={fit_merged.direction}, score={fit_merged.score:.6g}, "
            f"margin={max(fit_merged.nll_gain - gain_threshold, 0.0):.6g}, gain={fit_merged.nll_gain:.6g} | "
            f"half-t b={fit_merged.b:.6g}, scale={fit_merged.slack_scale:.6g}, nu={fit_merged.slack_nu:.3g} | "
            f"baseline mu={fit_merged.baseline_mu:.6g}, scale={fit_merged.baseline_scale:.6g}, nu={fit_merged.baseline_nu:.3g}"
        ),
    ]
    ax.plot([0.01, 0.99], [0.98, 0.98], color="#333333", lw=0.8, transform=ax.transAxes, clip_on=False)
    ax.text(0.01, 0.88, "\n".join(lines), va="top", ha="left", fontsize=6.2, color="#444444", family="monospace")

    fig.subplots_adjust(top=0.94, bottom=0.055, left=0.09, right=0.99, hspace=0.34)
    fig.savefig(path, dpi=220)
    plt.close(fig)

    return {
        "method": best_method.name,
        "best_cut_fraction": float(best_fraction),
        "cut_counts": [int(x) for x in best_cut_counts],
        "true_cut_fractions": [float(x) for x in true_fracs],
        "true_cut_counts": [int(x) for x in true_counts],
        "gain_threshold": float(gain_threshold),
        "soft_boundary_scale": float(soft_boundary_scale),
        "direction": str(direction),
        "baseline_fit": str(baseline_fit),
        "half_t_scale_quantile": float(half_t_scale_quantile),
        "selection_objective": str(selection_objective),
        "plot_space": str(plot_space_label),
        "stage1_plot": fit_to_dict(best_fit1),
        "stage2_plot": fit_to_dict(best_fit2),
        "true_stage1_plot": fit_to_dict(true_fit1),
        "true_stage2_plot": fit_to_dict(true_fit2),
        "merged_plot": fit_to_dict(fit_merged),
    }


def parse_feature_names(args: argparse.Namespace) -> list[str]:
    raw_items = []
    if getattr(args, "features", None):
        raw_items.append(str(args.features))
    elif getattr(args, "feature", None):
        raw_items.append(str(args.feature))
    names: list[str] = []
    for raw in raw_items:
        for part in raw.split(","):
            name = part.strip()
            if name:
                names.append(name)
    if getattr(args, "normal_force_scale", None) is not None:
        names.append(_scaled_feature_name("normal_force", float(args.normal_force_scale)))
    if getattr(args, "normal_force_scales", None):
        for part in str(args.normal_force_scales).split(","):
            text = part.strip()
            if text:
                names.append(_scaled_feature_name("normal_force", float(text)))
    if not names:
        raise ValueError("At least one feature is required.")
    return names


def _safe_stem(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text))


def run_one_feature(
    args: argparse.Namespace,
    *,
    feature_name: str,
    methods: list[ScoreMethod],
    best_method: ScoreMethod,
    gain_threshold: float,
    soft_boundary_scale: float,
    direction: str,
    baseline_fit: str,
    half_t_scale_quantile: float,
    selection_objective: str,
    observation_noise_scale: float,
) -> dict:
    values_raw, labels, meta = load_s4_stage_pair_feature(args, feature_name=feature_name)
    values_std, score_mean, score_std = standardize_values(
        values_raw,
        mean=float(meta["feature_global_mean"]),
        std=float(meta["feature_global_std"]),
    )
    score_feature_scale = float(meta.get("score_feature_scale", meta.get("feature_scale", 1.0)))
    values_score_base = values_std if args.score_space == "standardized" else values_raw
    values_score = [np.asarray(v, dtype=float) * score_feature_scale for v in values_score_base]
    observation_noise_scale_raw = (
        observation_noise_scale * score_std / max(score_feature_scale, 1e-12)
        if args.score_space == "standardized"
        else observation_noise_scale / max(score_feature_scale, 1e-12)
    )

    rows, summary = scan_methods(
        values_score,
        labels,
        methods,
        min_len=int(args.min_len),
        trim_fraction=float(args.trim_fraction),
        trim_min_n=int(args.trim_min_n),
        nu=float(args.nu),
        gain_threshold=gain_threshold,
        soft_boundary_scale=soft_boundary_scale,
        direction=direction,
        baseline_fit=baseline_fit,
        half_t_scale_quantile=half_t_scale_quantile,
        selection_objective=selection_objective,
        observation_noise_scale=observation_noise_scale,
    )
    boundary_index = int(meta["boundary_index"])
    stage_pair_label = f"true stages {boundary_index}+{boundary_index + 1}"
    feature_stem = f"{_safe_stem(str(meta['feature']))}_b{boundary_index}"
    csv_path = args.outdir / f"s4_{feature_stem}_inequality_cutpoint_scores.csv"
    write_csv(csv_path, rows)

    best = summary[best_method.name]["best"]
    best_fraction = float(best["cut_fraction"])
    best_cut_n = int(best["cut_n"])
    combined_plot_path = args.outdir / f"s4_{feature_stem}_inequality_cutpoint_combined.png"
    raw_fit_summary = plot_combined_report(
        combined_plot_path,
        rows=rows,
        methods=methods,
        values_by_demo_raw=values_raw,
        labels_by_demo=labels,
        feature_name=str(meta["feature"]),
        stage_pair_label=stage_pair_label,
        best_method=best_method,
        best_cut_n=best_cut_n,
        min_len=int(args.min_len),
        trim_fraction=float(args.trim_fraction),
        trim_min_n=int(args.trim_min_n),
        nu=float(args.nu),
        score_space=str(args.score_space),
        gain_threshold=gain_threshold,
        soft_boundary_scale=soft_boundary_scale,
        direction=direction,
        baseline_fit=baseline_fit,
        half_t_scale_quantile=half_t_scale_quantile,
        selection_objective=selection_objective,
        observation_noise_scale=observation_noise_scale_raw,
    )

    output = {
        "meta": meta,
        "score_space": args.score_space,
        "gain_threshold": gain_threshold,
        "soft_boundary_scale": soft_boundary_scale,
        "direction": direction,
        "baseline_fit": baseline_fit,
        "half_t_scale_quantile": half_t_scale_quantile,
        "selection_objective": selection_objective,
        "observation_noise_scale": observation_noise_scale,
        "observation_noise_scale_raw": observation_noise_scale_raw,
        "score_feature_scale": score_feature_scale,
        "selection_metric": summary[best_method.name].get("selection_metric"),
        "score_standardization": {
            "mean": score_mean,
            "std": score_std,
            "post_scale": score_feature_scale,
            "scope": meta.get("feature_standardization_scope"),
            "n": meta.get("feature_standardization_n"),
            "observation_noise_scale": observation_noise_scale,
            "observation_noise_scale_raw": observation_noise_scale_raw,
        },
        "methods": [m.__dict__ for m in methods],
        "best_method": best_method.__dict__,
        "scan_summary": summary,
        "best_raw_fit_summary": raw_fit_summary,
        "outputs": {
            "csv": str(csv_path),
            "combined_plot": str(combined_plot_path),
        },
    }
    json_path = args.outdir / f"s4_{feature_stem}_inequality_cutpoint_summary.json"
    with json_path.open("w") as f:
        json.dump(output, f, indent=2, allow_nan=True)

    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    print(f"wrote {combined_plot_path}")
    print(
        "best "
        f"feature={meta['feature']} "
        f"boundary={boundary_index} "
        f"selection={selection_objective} "
        f"method={best_method.name} cut_fraction={best_fraction:.4f} "
        f"cut_n={int(best['cut_n'])} "
        f"selection_value={float(best.get('selection_value', best['split_total_margin'])):.6g} "
        f"score={float(best['split_score_weighted_mean']):.6g} "
        f"mean_margin={float(best['split_activation_margin']):.6g} "
        f"total_margin={float(best['split_total_margin']):.6g} "
        f"true_cut_n={int(best['true_cut_n'])} "
        f"true_fraction={float(best['true_cut_fraction']):.4f}"
    )
    return {
        "feature": str(meta["feature"]),
        "base_feature": str(meta.get("base_feature", meta["feature"])),
        "feature_scale": float(meta.get("feature_scale", 1.0)),
        "score_feature_scale": float(meta.get("score_feature_scale", meta.get("feature_scale", 1.0))),
        "raw_feature_scale": float(meta.get("raw_feature_scale", 1.0)),
        "boundary_index": int(boundary_index),
        "true_stage_pair": [int(boundary_index), int(boundary_index + 1)],
        "summary_json": str(json_path),
        "scores_csv": str(csv_path),
        "combined_plot": str(combined_plot_path),
        "best": {
            "method": best_method.name,
            "cut_fraction": best_fraction,
            "cut_n": int(best["cut_n"]),
            "score": float(best["split_score_weighted_mean"]),
            "mean_margin": float(best["split_activation_margin"]),
            "total_margin": float(best["split_total_margin"]),
            "selection_value": float(best.get("selection_value", best["split_total_margin"])),
            "true_cut_n": int(best["true_cut_n"]),
            "true_cut_fraction": float(best["true_cut_fraction"]),
            "baseline_fit": baseline_fit,
            "half_t_scale_quantile": half_t_scale_quantile,
            "selection_objective": selection_objective,
            "observation_noise_scale": observation_noise_scale,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick S4 single-feature two-stage inequality cutpoint scan."
    )
    parser.add_argument("--outdir", type=Path, default=PROJECT_ROOT / "outputs/analysis/s4_start_dist_inequality_cutpoint")
    parser.add_argument("--n-demos", type=int, default=None, help="How many demos to generate before selecting --demo-index. Defaults to demo_index + 1.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--demo-index", type=int, default=0, help="Single demo index to test. Defaults to 0.")
    parser.add_argument(
        "--boundary-index",
        type=int,
        default=DEFAULT_BOUNDARY_INDEX,
        help="1-based true boundary to scan. 1 uses true stages 1+2; 3 uses true stages 3+4.",
    )
    parser.add_argument("--feature", type=str, default=DEFAULT_FEATURE, help="S4 feature name(s) to test. Use comma-separated names for multiple features.")
    parser.add_argument("--features", type=str, default=None, help="Alias for --feature; comma-separated feature names run in order.")
    parser.add_argument(
        "--normal-force-scale",
        type=float,
        default=None,
        help="Append one extra feature whose score-space values are normal_force standardized first, then multiplied by this positive scale.",
    )
    parser.add_argument(
        "--normal-force-scales",
        type=str,
        default=None,
        help="Append multiple post-standardization normal_force score scales, e.g. '0.1,10,100'.",
    )
    parser.add_argument("--methods", type=str, default=DEFAULT_METHODS, help="Default seg_q05 matches current trunc_t_lower_z score: segment q05 boundary and segment Student-t baseline.")
    parser.add_argument("--best-method", type=str, default=DEFAULT_BEST_METHOD)
    parser.add_argument("--direction", choices=["lower", "upper", "auto"], default=DEFAULT_DIRECTION)
    parser.add_argument(
        "--baseline-fit",
        choices=["robust", "mle"],
        default=DEFAULT_BASELINE_FIT,
        help="Baseline Student-t fit used for scan/segmentation scores. MLE keeps nu fixed and optimizes mu/scale.",
    )
    parser.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN)
    parser.add_argument("--trim-fraction", type=float, default=DEFAULT_TRIM_FRACTION)
    parser.add_argument("--trim-min-n", type=int, default=DEFAULT_TRIM_MIN_N)
    parser.add_argument("--nu", type=float, default=DEFAULT_NU)
    parser.add_argument("--score-space", choices=["standardized", "raw"], default=DEFAULT_SCORE_SPACE)
    parser.add_argument(
        "--gain-threshold",
        type=float,
        default=DEFAULT_GAIN_THRESHOLD,
        help="Script-local gain threshold for margin=max(gain-thr,0).",
    )
    parser.add_argument(
        "--soft-boundary-scale",
        type=float,
        default=DEFAULT_SOFT_BOUNDARY_SCALE,
        help="Softplus boundary width as a fraction of fitted half-t scale.",
    )
    parser.add_argument(
        "--half-t-scale-quantile",
        type=float,
        default=DEFAULT_HALF_T_SCALE_QUANTILE,
        help="Slack quantile used to fit half-t scale. Higher values reduce collapse when many samples lie exactly on the boundary.",
    )
    parser.add_argument(
        "--selection-objective",
        choices=["total-margin", "selected-loglik", "active-loglik"],
        default=DEFAULT_SELECTION_OBJECTIVE,
        help=(
            "Cut selection objective. total-margin uses total activated margin; "
            "selected-loglik uses active/baseline selected absolute log-likelihood; "
            "active-loglik uses active log-likelihood only when gain exceeds threshold and gives inactive stages 0."
        ),
    )
    parser.add_argument(
        "--observation-noise-scale",
        type=float,
        default=DEFAULT_OBSERVATION_NOISE_SCALE,
        help="Minimum model scale in score space. Applied to both baseline and half-t scales; use e.g. 0.03 with standardized score space.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    gain_threshold = float(args.gain_threshold)
    soft_boundary_scale = float(args.soft_boundary_scale)
    direction = str(args.direction)
    baseline_fit = str(args.baseline_fit)
    half_t_scale_quantile = float(np.clip(args.half_t_scale_quantile, 0.5, 0.99))
    selection_objective = str(args.selection_objective)
    observation_noise_scale = max(float(args.observation_noise_scale), 0.0)
    methods = [parse_method(x) for x in args.methods.split(",") if x.strip()]
    best_method = parse_method(args.best_method)
    if best_method.name not in {m.name for m in methods}:
        methods.insert(0, best_method)

    feature_names = parse_feature_names(args)
    aggregate = {
        "features": [],
        "boundary_index": int(args.boundary_index),
        "true_stage_pair": [int(args.boundary_index), int(args.boundary_index) + 1],
        "score_space": args.score_space,
        "gain_threshold": gain_threshold,
        "soft_boundary_scale": soft_boundary_scale,
        "direction": direction,
        "baseline_fit": baseline_fit,
        "half_t_scale_quantile": half_t_scale_quantile,
        "selection_objective": selection_objective,
        "observation_noise_scale": observation_noise_scale,
        "selection_metric": selection_metric_description(selection_objective),
    }
    for feature_name in feature_names:
        aggregate["features"].append(
            run_one_feature(
                args,
                feature_name=feature_name,
                methods=methods,
                best_method=best_method,
                gain_threshold=gain_threshold,
                soft_boundary_scale=soft_boundary_scale,
                direction=direction,
                baseline_fit=baseline_fit,
                half_t_scale_quantile=half_t_scale_quantile,
                selection_objective=selection_objective,
                observation_noise_scale=observation_noise_scale,
            )
        )
    if len(feature_names) > 1:
        aggregate_path = args.outdir / f"s4_b{int(args.boundary_index)}_multi_feature_inequality_cutpoint_summary.json"
        with aggregate_path.open("w") as f:
            json.dump(aggregate, f, indent=2, allow_nan=True)
        print(f"wrote {aggregate_path}")


if __name__ == "__main__":
    main()
