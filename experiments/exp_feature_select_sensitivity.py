from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from envs import load_env
from evaluation import eval_goalhmm_auto
from methods import SegCons
from visualization.io import save_figure


USE_3D = True
DATASET_NAME = "3DObsAvoid" if USE_3D else "2DObsAvoid"
N_DEMOS = 12
DEMO_SEED = 123
INIT_MODE = "heuristic"
N_RUNS = 10
LAMBDA_LIST = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
MAX_ITER = 30
SAVE_PATH = Path("outputs/experiments/exp_feature_select_sensitivity_results.json")


def run_single(dataset, seed, r_lambda):
    np.random.seed(seed)
    learner = SegCons(
        demos=dataset.demos,
        env=dataset.env,
        true_taus=dataset.true_taus,
        g1_init=INIT_MODE,
        auto_feature_select=True,
        r_sparse_lambda=r_lambda,
        plot_every=None,
    )
    gammas = learner.fit(max_iter=MAX_ITER, verbose=False)
    dummy_xis = [np.zeros((len(X) - 1, 2, 2), dtype=float) for X in dataset.demos]
    metrics = eval_goalhmm_auto(learner, gammas, dummy_xis)
    return {
        "loglik": float(learner.loss_loglik[-1]) if learner.loss_loglik else np.nan,
        "NMAE_tau": float(metrics.get("NMAE_tau", np.nan)),
        "e_g1": float(metrics.get("e_g1", np.nan)),
        "e_g2": float(metrics.get("e_g2", np.nan)),
        "RelErr_d": float(metrics.get("RelErr_d", np.nan)),
        "RelErr_v": float(metrics.get("RelErr_v", np.nan)),
        "r": np.asarray(learner.r, dtype=int).tolist(),
    }


def plot_metric_boxplots(metrics_by_lambda):
    if plt is None:
        return
    metric_names = ["loglik", "NMAE_tau", "e_g1", "e_g2", "RelErr_d", "RelErr_v"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes = axes.ravel()
    labels = [str(l) for l in LAMBDA_LIST]
    for ax, metric in zip(axes, metric_names):
        vals = [metrics_by_lambda[lam][metric] for lam in LAMBDA_LIST]
        ax.boxplot(vals, labels=labels, showmeans=True, meanline=True)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3)
    save_figure(fig, SAVE_PATH.with_name(SAVE_PATH.stem + "_boxplots.png"))


def run_experiment():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_env(DATASET_NAME, n_demos=N_DEMOS, seed=DEMO_SEED)
    metrics = {
        lam: {
            "loglik": [],
            "NMAE_tau": [],
            "e_g1": [],
            "e_g2": [],
            "RelErr_d": [],
            "RelErr_v": [],
            "r": [],
        }
        for lam in LAMBDA_LIST
    }

    for lam_idx, lam in enumerate(LAMBDA_LIST):
        print(f"[lambda={lam}]")
        for run in range(N_RUNS):
            seed = int(lam_idx * 10_000 + run)
            res = run_single(dataset, seed, lam)
            for key, val in res.items():
                metrics[lam][key].append(val)
            print(f"  run={run:02d} NMAE={res['NMAE_tau']:.3f} r={res['r']}")

    SAVE_PATH.write_text(json.dumps({str(k): v for k, v in metrics.items()}, indent=2))
    print(f"[Saved] {SAVE_PATH}")
    plot_metric_boxplots(metrics)
    return metrics


if __name__ == "__main__":
    run_experiment()
