from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .unified_experiment import run_experiment
from visualization.io import save_figure


USE_3D = True
DATASET_NAME = "3DObsAvoid" if USE_3D else "2DObsAvoid"
N_DEMOS = 12
DEMO_SEED = 123
INIT_MODE = "uniform_taus"
N_RUNS = 10
LAMBDA_LIST = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
MAX_ITER = 30
SAVE_PATH = Path("outputs/experiments/exp_feature_select_sensitivity_results.json")


def _ordered_metric_names(metrics_dict):
    preferred = ["loglik", "MAE_tau", "NMAE_tau", "e_g1", "e_g2", "AbsErr_d", "AbsErr_centerline", "AbsErr_v"]
    names = [name for name in preferred if name in metrics_dict]
    names.extend(sorted(name for name in metrics_dict if name not in names))
    return names


def _extract_joint_result(result):
    learner = result["joint_result"]["model"]
    metrics = dict(result["joint_result"]["metrics"])
    metrics["loglik"] = float(learner.loss_loglik[-1]) if getattr(learner, "loss_loglik", None) else np.nan
    return learner, metrics


def run_single(seed, r_lambda):
    result = run_experiment(
        dataset_name=DATASET_NAME,
        method_name="segcons",
        dataset_kwargs={"n_demos": N_DEMOS, "seed": DEMO_SEED},
        method_kwargs={
            "max_iter": MAX_ITER,
            "verbose": False,
            "plot_every": None,
            "seed": seed,
            "tau_init_mode": INIT_MODE,
            "auto_feature_select": True,
            "r_sparse_lambda": r_lambda,
        },
    )
    learner, metrics = _extract_joint_result(result)
    feature_names = [spec["name"] for spec in getattr(learner, "feature_specs", [])]
    active_features = {
        "stage1": [feature_names[m] for m in range(len(feature_names)) if learner.r[0, m] == 1],
        "stage2": [feature_names[m] for m in range(len(feature_names)) if learner.r[1, m] == 1],
    }
    out = {key: float(value) if np.isscalar(value) and np.isfinite(value) else value for key, value in metrics.items()}
    out["feature_names"] = feature_names
    out["r"] = np.asarray(learner.r, dtype=int).tolist()
    out["active_features"] = active_features
    return out


def plot_metric_boxplots(metrics_by_lambda, metric_names):
    if plt is None or not metric_names:
        return
    n_metrics = len(metric_names)
    ncols = min(3, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    labels = [str(l) for l in LAMBDA_LIST]
    for ax, metric in zip(axes, metric_names):
        vals = [metrics_by_lambda[lam][metric] for lam in LAMBDA_LIST]
        ax.boxplot(vals, labels=labels, showmeans=True, meanline=True)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3)
    for ax in axes[n_metrics:]:
        ax.axis("off")
    save_figure(fig, SAVE_PATH.with_name(SAVE_PATH.stem + "_boxplots.png"))


def run_experiment_sensitivity():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = {lam: {"feature_names": [], "r": [], "active_features": []} for lam in LAMBDA_LIST}
    metric_names = None

    for lam_idx, lam in enumerate(LAMBDA_LIST):
        print(f"[lambda={lam}]")
        for run in range(N_RUNS):
            seed = int(lam_idx * 10_000 + run)
            res = run_single(seed, lam)
            if metric_names is None:
                metric_names = _ordered_metric_names({k: v for k, v in res.items() if k not in {"feature_names", "r", "active_features"}})
                for lam0 in LAMBDA_LIST:
                    for name in metric_names:
                        metrics[lam0].setdefault(name, [])
            for key, val in res.items():
                metrics[lam][key].append(val)
            print(f"  run={run:02d} NMAE={float(res.get('NMAE_tau', np.nan)):.3f} active={res['active_features']}")

    SAVE_PATH.write_text(json.dumps({str(k): v for k, v in metrics.items()}, indent=2))
    print(f"[Saved] {SAVE_PATH}")
    plot_metric_boxplots(metrics, metric_names or [])
    return metrics


if __name__ == "__main__":
    run_experiment_sensitivity()
