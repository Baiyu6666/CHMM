from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.registry import load_env


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _feature_index(schema: list[dict], name: str) -> int:
    for i, spec in enumerate(schema):
        if str(spec.get("name", "")) == str(name):
            return int(spec.get("column_idx", i))
    raise ValueError(f"Feature '{name}' not found in schema.")


def _soft_half_t_pdf(
    x: np.ndarray,
    *,
    b: float,
    tau: float,
    nu: float,
    rho: float,
    direction: str,
) -> np.ndarray:
    sign = 1.0 if str(direction).lower() == "lower" else -1.0
    y = sign * (np.asarray(x, dtype=float) - float(b))
    tau = max(float(tau), 1e-12)
    rho = max(float(rho), 1e-12)
    alpha = rho * tau
    half_t_at_zero = 2.0 * float(student_t.pdf(0.0, df=nu)) / tau
    z_norm = 1.0 + half_t_at_zero * alpha
    out = np.empty_like(y, dtype=float)
    ok = y >= 0.0
    out[ok] = (2.0 / tau) * student_t.pdf(y[ok] / tau, df=nu) / z_norm
    out[~ok] = half_t_at_zero * np.exp(y[~ok] / alpha) / z_norm
    return out


def _soft_half_t_fit(xs: np.ndarray, *, nu: float, rho: float, noise_floor: float, direction: str) -> dict:
    xs = np.asarray(xs, dtype=float).reshape(-1)
    if str(direction).lower() == "lower":
        b = float(np.quantile(xs, 0.05))
        slack = np.maximum(xs - b, 0.0)
    elif str(direction).lower() == "upper":
        b = float(np.quantile(xs, 0.95))
        slack = np.maximum(b - xs, 0.0)
    else:
        raise ValueError(f"Unsupported direction '{direction}'.")
    q90 = float(np.quantile(slack, 0.90))
    unit_q90 = float(student_t.ppf(0.95, df=float(nu)))
    tau0 = q90 / max(unit_q90, 1e-12)
    tau = max(float(tau0), float(noise_floor), 1e-6)
    pdf_values = _soft_half_t_pdf(xs, b=b, tau=tau, nu=nu, rho=rho, direction=direction)
    nll = float(-np.mean(np.log(np.maximum(pdf_values, 1e-300))))
    return {
        "direction": str(direction).lower(),
        "b": b,
        "slack_q90": q90,
        "student_t_ppf_0_95": unit_q90,
        "tau0": tau0,
        "tau": tau,
        "nll": nll,
    }


def _student_t_baseline_fit(xs: np.ndarray, *, nu: float, noise_floor: float) -> dict:
    xs = np.asarray(xs, dtype=float).reshape(-1)
    q10, q50, q90 = np.quantile(xs, [0.10, 0.50, 0.90])
    unit_width = float(student_t.ppf(0.90, df=nu) - student_t.ppf(0.10, df=nu))
    init_scale = max(float(q90 - q10) / max(unit_width, 1e-12), float(noise_floor), 1e-6)
    starts = [(float(q50), init_scale), (float(np.mean(xs)), max(float(np.std(xs)), float(noise_floor), 1e-6))]
    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    x_span = max(float(x_max - x_min), float(np.std(xs)), init_scale, 1e-6)
    mu_lo = x_min - 5.0 * x_span
    mu_hi = x_max + 5.0 * x_span
    scale_lo = max(float(noise_floor), 1e-6)
    scale_hi = max(scale_lo * 10.0, x_span * 20.0)

    best = None
    for start_mu, start_scale in starts:
        mu = float(np.clip(start_mu, mu_lo, mu_hi))
        scale = float(np.clip(start_scale, scale_lo, scale_hi))
        for _ in range(50):
            prev_mu, prev_scale = mu, scale
            z = (xs - mu) / max(scale, scale_lo)
            weights = (float(nu) + 1.0) / (float(nu) + z * z)
            weight_sum = float(np.sum(weights))
            if weight_sum <= 1e-12:
                break
            mu = float(np.sum(weights * xs) / weight_sum)
            scale = math.sqrt(max(float(np.mean(weights * (xs - mu) ** 2)), scale_lo * scale_lo))
            mu = float(np.clip(mu, mu_lo, mu_hi))
            scale = float(np.clip(scale, scale_lo, scale_hi))
            if max(abs(mu - prev_mu) / max(x_span, 1e-12), abs(scale - prev_scale) / max(prev_scale, 1e-12)) < 1e-6:
                break
        pdf_values = student_t.pdf((xs - mu) / scale, df=float(nu)) / scale
        nll = float(-np.mean(np.log(np.maximum(pdf_values, 1e-300))))
        if best is None or nll < best["nll"]:
            best = {"mu": float(mu), "scale": float(scale), "nll": nll}
    return best


def _stage1_obs_dist_values(bundle, feature_name: str) -> np.ndarray:
    schema = list(bundle.feature_schema or bundle.env.get_feature_schema())
    feat_idx = _feature_index(schema, feature_name)
    raw_features = [np.asarray(bundle.env.compute_all_features_matrix(X), dtype=float) for X in bundle.demos]
    full_stack = np.concatenate(raw_features, axis=0)
    feat_mean = np.mean(full_stack, axis=0)
    feat_std = np.std(full_stack, axis=0) + 1e-8

    values = []
    for F_raw, cuts in zip(raw_features, bundle.true_cutpoints):
        cut_arr = np.asarray(cuts, dtype=int).reshape(-1)
        if cut_arr.size == 0:
            continue
        end = int(cut_arr[0])
        F_z = (F_raw[:, feat_idx] - feat_mean[feat_idx]) / feat_std[feat_idx]
        values.append(F_z[: end + 1])
    if not values:
        raise ValueError("No stage-1 values found.")
    return np.concatenate(values, axis=0).astype(float)


def plot_soft_half_t_example(
    *,
    env_config: Path,
    output_dir: Path,
    feature_name: str,
    nu: float,
    rho: float,
    noise_floor: float,
    formats: list[str],
) -> list[Path]:
    cfg = _load_json(env_config)
    env_name = str(cfg.pop("name"))
    cfg.pop("method_overrides", None)
    bundle = load_env(env_name, **cfg)
    xs = _stage1_obs_dist_values(bundle, feature_name)
    xs = xs[np.isfinite(xs)]
    if xs.size < 3:
        raise ValueError("Need at least three finite samples.")

    lower_fit = _soft_half_t_fit(xs, nu=nu, rho=rho, noise_floor=noise_floor, direction="lower")
    upper_fit = _soft_half_t_fit(xs, nu=nu, rho=rho, noise_floor=noise_floor, direction="upper")
    baseline_fit = _student_t_baseline_fit(xs, nu=nu, noise_floor=noise_floor)

    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    pad = max(0.12 * (x_max - x_min), 0.12)
    grid = np.linspace(x_min - pad, x_max + pad, 500)
    lower_pdf = _soft_half_t_pdf(
        grid,
        b=float(lower_fit["b"]),
        tau=float(lower_fit["tau"]),
        nu=nu,
        rho=rho,
        direction="lower",
    )
    upper_pdf = _soft_half_t_pdf(
        grid,
        b=float(upper_fit["b"]),
        tau=float(upper_fit["tau"]),
        nu=nu,
        rho=rho,
        direction="upper",
    )
    baseline_pdf = student_t.pdf((grid - float(baseline_fit["mu"])) / float(baseline_fit["scale"]), df=float(nu)) / float(
        baseline_fit["scale"]
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "axes.titlesize": 7.4,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(3.35, 1.34), constrained_layout=False)
    bins = min(18, max(9, int(math.sqrt(xs.size))))
    ax.hist(
        xs,
        bins=bins,
        density=True,
        color="#D9E5F2",
        edgecolor="#4B6F8F",
        linewidth=0.45,
        alpha=0.95,
        label="Data",
    )
    ax.plot(grid, lower_pdf, color="#C0392B", linewidth=1.55, label="Lower-bound fit")
    ax.plot(grid, upper_pdf, color="#1F7A5B", linewidth=1.25, linestyle="-.", label="Upper-bound fit")
    ax.plot(grid, baseline_pdf, color="#2F3640", linewidth=1.15, linestyle=":", label="Baseline")
    ax.axvline(float(lower_fit["b"]), color="#303030", linewidth=0.75, linestyle="--", alpha=0.80, label=r"Boundary $b$")
    nll_text = (
        f"NLL lower={float(lower_fit['nll']):.2f} | "
        f"upper={float(upper_fit['nll']):.2f} | "
        f"base={float(baseline_fit['nll']):.2f}"
    )
    ax.text(
        0.985,
        0.52,
        nll_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.1,
        color="#303030",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.78),
    )
    ax.set_xlabel("Feature value (standardized)", labelpad=1.5)
    ax.set_ylabel("Density", labelpad=1.5)
    ax.legend(loc="upper right", frameon=False, handlelength=1.2, borderpad=0.1, labelspacing=0.18, ncol=2, columnspacing=0.55)
    ax.grid(axis="y", color="#D0D5DD", linewidth=0.45, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(width=0.55, length=2.2, pad=1.5)
    ax.set_ylim(bottom=0.0)
    fig.subplots_adjust(left=0.13, right=0.985, bottom=0.27, top=0.985)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    stem = "soft_half_t_example_distribution"
    for fmt in formats:
        out = output_dir / f"{stem}.{fmt}"
        fig.savefig(out, dpi=450 if fmt.lower() == "png" else None)
        saved.append(out)
    plt.close(fig)

    meta = {
        "feature_name": feature_name,
        "n_samples": int(xs.size),
        "standardized": True,
        "nu": float(nu),
        "rho": float(rho),
        "noise_floor": float(noise_floor),
        "lower_soft_half_t": lower_fit,
        "upper_soft_half_t": upper_fit,
        "student_t_baseline": baseline_fit,
    }
    (output_dir / f"{stem}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a compact soft half-t distribution example for paper figures.")
    parser.add_argument("--env-config", type=Path, default=PROJECT_ROOT / "configs/envs/S3ObsAvoid.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/paper_figures/soft_half_t")
    parser.add_argument("--feature-name", default="obs_dist")
    parser.add_argument("--nu", type=float, default=3.0)
    parser.add_argument("--rho", type=float, default=0.1)
    parser.add_argument("--noise-floor", type=float, default=0.003)
    parser.add_argument("--formats", default="png,pdf")
    args = parser.parse_args()

    formats = [item.strip().lower() for item in str(args.formats).split(",") if item.strip()]
    saved = plot_soft_half_t_example(
        env_config=args.env_config,
        output_dir=args.output_dir,
        feature_name=str(args.feature_name),
        nu=float(args.nu),
        rho=float(args.rho),
        noise_floor=float(args.noise_floor),
        formats=formats,
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
