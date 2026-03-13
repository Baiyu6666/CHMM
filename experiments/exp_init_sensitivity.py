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
from methods.gmm_hmm_core import GMMHMM
from visualization.io import save_figure


USE_3D = False
DATASET_NAME = "3DObsAvoid" if USE_3D else "2DObsAvoid"
N_DEMOS = 12
DEMO_SEED = 123
N_RUNS_PER_SCHEME = 10
MAX_ITER = 30
INIT_SCHEMES = ["true_tau", "random", "heuristic"]
MODELS = ["goal", "gmm"]
SAVE_PATH = Path("exp_init_sensitivity_results.json")
SAVE_PATH = Path("outputs/experiments/exp_init_sensitivity_results.json")


def sample_taus_random_shared(demos, rng):
    taus = []
    lam = rng.rand()
    for X in demos:
        T = len(X)
        t = int(round(lam * (T - 1)))
        taus.append(int(np.clip(t, 1, T - 2)))
    return np.asarray(taus, dtype=int)


def sample_taus_heuristic_softmax(demos, rng, n_cand=200, temperature=0.3):
    all_pts = np.concatenate(demos, axis=0)
    n_cand = min(n_cand, len(all_pts))
    cand_pts = all_pts[rng.choice(len(all_pts), size=n_cand, replace=False)]
    scores = []
    taus_for_cand = []
    for c in cand_pts:
        taus = []
        dvals = []
        for X in demos:
            d = np.linalg.norm(X - c[None, :], axis=1)
            t = int(np.clip(np.argmin(d), 1, len(X) - 2))
            taus.append(t)
            dvals.append(float(d[t]))
        scores.append(float(np.mean(dvals)))
        taus_for_cand.append(np.asarray(taus, dtype=int))
    logits = -np.asarray(scores) / max(temperature, 1e-6)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum() + 1e-12
    return taus_for_cand[rng.choice(len(taus_for_cand), p=probs)]


def _extract_metrics(metrics_dict):
    return {
        "MAE_tau": float(metrics_dict.get("MAE_tau", np.nan)),
        "NMAE_tau": float(metrics_dict.get("NMAE_tau", np.nan)),
        "e_g1": float(metrics_dict.get("e_g1", np.nan)),
        "e_g2": float(metrics_dict.get("e_g2", np.nan)),
        "RelErr_d": float(metrics_dict.get("RelErr_d", np.nan)),
        "RelErr_v": float(metrics_dict.get("RelErr_v", np.nan)),
    }


def run_single_goal(dataset, taus_init, seed):
    np.random.seed(seed)
    learner = SegCons(
        demos=dataset.demos,
        env=dataset.env,
        true_taus=dataset.true_taus,
        tau_init=taus_init,
        g1_init="from_tau",
        g2_init="from_tau",
        plot_every=None,
    )
    gammas = learner.fit(max_iter=MAX_ITER, verbose=False)
    dummy_xis = [np.zeros((len(X) - 1, 2, 2), dtype=float) for X in dataset.demos]
    metrics = _extract_metrics(eval_goalhmm_auto(learner, gammas, dummy_xis))
    metrics["loglik"] = float(learner.loss_loglik[-1]) if learner.loss_loglik else np.nan
    return metrics


def run_single_gmm(dataset, taus_init, seed):
    np.random.seed(seed)
    learner = GMMHMM(
        demos=dataset.demos,
        env=dataset.env,
        true_taus=dataset.true_taus,
        tau_init=taus_init,
        plot_every=None,
    )
    learner.fit(max_iter=MAX_ITER, verbose=False)
    return {
        "loglik": float(learner.loss_loglik[-1]) if learner.loss_loglik else np.nan,
        "MAE_tau": float(learner.metric_tau_mae[-1]) if learner.metric_tau_mae else np.nan,
        "NMAE_tau": float(learner.metric_tau_nmae[-1]) if learner.metric_tau_nmae else np.nan,
        "e_g1": float(learner.metric_g1_err[-1]) if learner.metric_g1_err else np.nan,
        "e_g2": float(learner.metric_g2_err[-1]) if learner.metric_g2_err else np.nan,
        "RelErr_d": float(learner.metric_d_relerr[-1]) if learner.metric_d_relerr else np.nan,
        "RelErr_v": float(learner.metric_v_relerr[-1]) if learner.metric_v_relerr else np.nan,
    }


def plot_boxplots(results):
    if plt is None:
        return
    metric_names = ["loglik", "NMAE_tau", "e_g1", "e_g2", "RelErr_d", "RelErr_v"]
    labels = [f"{scheme}-{model}" for scheme in INIT_SCHEMES for model in MODELS]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes = axes.ravel()
    for ax, metric in zip(axes, metric_names):
        vals = [results[scheme][model][metric] for scheme in INIT_SCHEMES for model in MODELS]
        ax.boxplot(vals, labels=labels, showmeans=True, meanline=True)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3)
    save_figure(fig, SAVE_PATH.with_name(SAVE_PATH.stem + "_boxplots.png"))


def run_experiment():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_env(DATASET_NAME, n_demos=N_DEMOS, seed=DEMO_SEED)
    rng_master = np.random.RandomState(999)
    results = {
        scheme: {
            model: {
                "loglik": [],
                "MAE_tau": [],
                "NMAE_tau": [],
                "e_g1": [],
                "e_g2": [],
                "RelErr_d": [],
                "RelErr_v": [],
            }
            for model in MODELS
        }
        for scheme in INIT_SCHEMES
    }

    for scheme in INIT_SCHEMES:
        for run in range(N_RUNS_PER_SCHEME):
            seed = int(rng_master.randint(0, 1_000_000))
            if scheme == "true_tau":
                taus_init = np.asarray(dataset.true_taus, dtype=int)
            elif scheme == "random":
                taus_init = sample_taus_random_shared(dataset.demos, rng_master)
            else:
                taus_init = sample_taus_heuristic_softmax(dataset.demos, rng_master)

            goal_metrics = run_single_goal(dataset, taus_init, seed)
            gmm_metrics = run_single_gmm(dataset, taus_init, seed)
            for k, v in goal_metrics.items():
                results[scheme]["goal"][k].append(v)
            for k, v in gmm_metrics.items():
                results[scheme]["gmm"][k].append(v)
            print(f"[{scheme}][run={run}] goal={goal_metrics['NMAE_tau']:.3f} gmm={gmm_metrics['NMAE_tau']:.3f}")

    SAVE_PATH.write_text(json.dumps(results, indent=2))
    print(f"[Saved] {SAVE_PATH}")
    plot_boxplots(results)
    return results


if __name__ == "__main__":
    run_experiment()
