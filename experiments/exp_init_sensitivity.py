from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from envs import load_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from .unified_experiment import run_experiment
from methods.common.tau_init import resolve_tau_init_for_demos
from visualization.io import save_figure


USE_3D = False
DATASET_NAME = "3DObsAvoid" if USE_3D else "2DObsAvoid"
N_DEMOS = 12
DEMO_SEED = 123
N_RUNS_PER_SCHEME = 10
MAX_ITER = 30
INIT_SCHEMES = ["uniform_taus", "random_taus", "changepoint_warmstart"]
MODELS = ["segcons", "cghmm"]
SAVE_PATH = Path("outputs/experiments/exp_init_sensitivity_results.json")

def sample_taus_from_mode(demos, dataset, mode, seed):
    return resolve_tau_init_for_demos(
        demos,
        tau_init=None,
        tau_init_mode=mode,
        env=dataset.env,
        seed=seed,
        use_velocity=False,
        vel_weight=1.0,
        standardize=False,
        use_env_features=True,
        selected_raw_feature_ids=None,
    )


def sample_taus_changepoint_warmstart():
    result = run_experiment(
        dataset_name=DATASET_NAME,
        method_name="changepoint",
        dataset_kwargs={"n_demos": N_DEMOS, "seed": DEMO_SEED},
        method_kwargs={
            "segmenter": {
                "plot_every": None,
                "use_env_features": True,
                "seed": 0,
            },
            "constraints": {
                "refine_steps": 0,
                "verbose": False,
            },
        },
    )
    return np.asarray(result["segmentation"].taus, dtype=int)


def _ordered_metric_names(metrics_dict):
    preferred = ["loglik", "MAE_tau", "NMAE_tau", "e_g1", "e_g2", "AbsErr_d", "AbsErr_centerline", "AbsErr_v"]
    names = [name for name in preferred if name in metrics_dict]
    names.extend(sorted(name for name in metrics_dict if name not in names))
    return names


def _extract_metrics(method_name, result):
    if method_name == "segcons":
        learner = result["joint_result"]["model"]
        metrics = dict(result["joint_result"]["metrics"])
    else:
        learner = result["constraints"]["model"]
        metrics = dict(result["constraints"]["metrics"])
    metrics["loglik"] = float(learner.loss_loglik[-1]) if getattr(learner, "loss_loglik", None) else np.nan
    return metrics


def run_single_segcons(taus_init, seed):
    result = run_experiment(
        dataset_name=DATASET_NAME,
        method_name="segcons",
        dataset_kwargs={"n_demos": N_DEMOS, "seed": DEMO_SEED},
        method_kwargs={
            "max_iter": MAX_ITER,
            "verbose": False,
            "plot_every": None,
            "tau_init": taus_init,
            "seed": seed,
        },
    )
    return _extract_metrics("segcons", result)


def run_single_cghmm(taus_init, seed):
    result = run_experiment(
        dataset_name=DATASET_NAME,
        method_name="cghmm",
        dataset_kwargs={"n_demos": N_DEMOS, "seed": DEMO_SEED},
        method_kwargs={
            "segmenter": {
                "tau_init": taus_init,
                "plot_every": None,
                "max_iter": MAX_ITER,
                "verbose": False,
                "seed": seed,
            },
            "constraints": {
                "refine_steps": 5,
                "verbose": False,
            },
        },
    )
    return _extract_metrics("cghmm", result)


def plot_boxplots(results, metric_names):
    if plt is None or not metric_names:
        return
    labels = [f"{scheme}-{model}" for scheme in INIT_SCHEMES for model in MODELS]
    n_metrics = len(metric_names)
    ncols = min(3, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, metric in zip(axes, metric_names):
        vals = [results[scheme][model][metric] for scheme in INIT_SCHEMES for model in MODELS]
        ax.boxplot(vals, labels=labels, showmeans=True, meanline=True)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3)
    for ax in axes[n_metrics:]:
        ax.axis("off")
    save_figure(fig, SAVE_PATH.with_name(SAVE_PATH.stem + "_boxplots.png"))


def run_experiment_sensitivity():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_env(DATASET_NAME, n_demos=N_DEMOS, seed=DEMO_SEED)
    rng_master = np.random.RandomState(999)
    results = {scheme: {model: {} for model in MODELS} for scheme in INIT_SCHEMES}
    metric_names = None

    for scheme in INIT_SCHEMES:
        for run in range(N_RUNS_PER_SCHEME):
            seed = int(rng_master.randint(0, 1_000_000))
            if scheme == "changepoint_warmstart":
                taus_init = sample_taus_changepoint_warmstart()
            else:
                taus_init = sample_taus_from_mode(dataset.demos, dataset, scheme, seed)

            segcons_metrics = run_single_segcons(taus_init, seed)
            cghmm_metrics = run_single_cghmm(taus_init, seed)
            if metric_names is None:
                union_metrics = dict(segcons_metrics)
                union_metrics.update(cghmm_metrics)
                metric_names = _ordered_metric_names(union_metrics)
                for scheme_name in INIT_SCHEMES:
                    for model_name in MODELS:
                        for metric_name in metric_names:
                            results[scheme_name][model_name].setdefault(metric_name, [])

            for key in metric_names:
                results[scheme]["segcons"][key].append(float(segcons_metrics.get(key, np.nan)))
                results[scheme]["cghmm"][key].append(float(cghmm_metrics.get(key, np.nan)))

            print(
                f"[{scheme}][run={run}] "
                f"segcons={float(segcons_metrics.get('NMAE_tau', np.nan)):.3f} "
                f"cghmm={float(cghmm_metrics.get('NMAE_tau', np.nan)):.3f}"
            )

    SAVE_PATH.write_text(json.dumps(results, indent=2))
    print(f"[Saved] {SAVE_PATH}")
    plot_boxplots(results, metric_names or [])
    return results


if __name__ == "__main__":
    run_experiment_sensitivity()
