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

from envs import load_env

try:
    from .config_loader import deep_merge, load_experiment_config
    from .unified_experiment import run_experiment
except ImportError:
    from experiments.config_loader import deep_merge, load_experiment_config
    from experiments.unified_experiment import run_experiment
from methods.common.tau_init import clip_tau_for_sequence, extract_taus_hat, resolve_tau_init_for_demos
from visualization.io import save_figure
from visualization.plot4panel import plot_demos_goals_snapshot


DATASET_NAME = "2DObsAvoid"
RANDOM_TAU_RUNS = 6
MODELS = ["cghmm"]
SAVE_PATH = Path("outputs/experiments/exp_init_sensitivity_results.json")
SAVE_PATH = PROJECT_ROOT / SAVE_PATH


def _config_paths(dataset_name, method_name):
    return (
        PROJECT_ROOT / "configs" / "envs" / f"{dataset_name}.json",
        PROJECT_ROOT / "configs" / "methods" / f"{method_name}.json",
    )


def _load_joint_config(dataset_name, method_name):
    env_path, method_path = _config_paths(dataset_name, method_name)
    cfg = load_experiment_config(env_path, method_path)
    dataset_cfg = dict(cfg["dataset"])
    method_cfg = dict(cfg["method"])
    dataset_cfg.pop("name", None)
    method_cfg.pop("name", None)
    dataset_method_overrides = dict(dataset_cfg.pop("method_overrides", {}))
    method_cfg = deep_merge(method_cfg, dataset_method_overrides.get(method_name, {}))
    return dataset_cfg, method_cfg


def _effective_dataset_kwargs(dataset_name, method_name):
    dataset_cfg, _ = _load_joint_config(dataset_name, method_name)
    return dataset_cfg


def _effective_joint_method_kwargs(method_name, method_seed, tau_init, extra_method_kwargs=None):
    _, method_cfg = _load_joint_config(DATASET_NAME, method_name)
    method_cfg = dict(method_cfg)
    method_cfg["tau_init"] = np.asarray(tau_init, dtype=int)
    method_cfg["seed"] = int(method_seed)
    method_cfg["verbose"] = False
    method_cfg["plot_every"] = None
    if extra_method_kwargs:
        method_cfg = deep_merge(method_cfg, extra_method_kwargs)
    if method_name == "segcons" and method_cfg.get("fixed_feature_mask") is not None:
        method_cfg["auto_feature_select"] = False
    return method_cfg

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
    dataset_kwargs = _effective_dataset_kwargs(DATASET_NAME, "changepoint")
    result = run_experiment(
        dataset_name=DATASET_NAME,
        method_name="changepoint",
        dataset_kwargs=dataset_kwargs,
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


def sample_taus_random_ratio(demos, ratio):
    taus = []
    for X in demos:
        T = len(X)
        tau = int(round(float(ratio) * (T - 1)))
        taus.append(clip_tau_for_sequence(X, tau))
    return np.asarray(taus, dtype=int)


def runs_for_init_scheme(scheme):
    if scheme == "random_taus":
        ratios = np.linspace(0.1, 0.9, RANDOM_TAU_RUNS)
        return [{"seed": 0, "ratio": float(r), "label": f"ratio={r:.2f}"} for r in ratios]
    return [{"seed": 0, "ratio": None, "label": scheme}]


def build_comparison_specs(dataset):
    scheme = "random_taus"
    specs = []
    for run_spec in runs_for_init_scheme(scheme):
        taus_init = sample_taus_random_ratio(dataset.demos, run_spec["ratio"])
        specs.append(
            {
                "group": scheme,
                "label": run_spec["label"],
                "tau_init": taus_init,
                "method_seed": int(run_spec["seed"]),
                "method_overrides": {},
            }
        )
    return [scheme], specs


def _ordered_metric_names(metrics_dict):
    preferred = ["loglik", "MAE_tau", "NMAE_tau", "e_g1", "e_g2", "AbsErr_d", "AbsErr_centerline", "AbsErr_v"]
    names = [name for name in preferred if name in metrics_dict]
    names.extend(sorted(name for name in metrics_dict if name not in names))
    return names


def _extract_metrics(method_name, result):
    if method_name in {"segcons", "ccp"}:
        learner = result["joint_result"]["model"]
        metrics = dict(result["joint_result"]["metrics"])
        gammas = result["joint_result"]["gammas"]
    elif method_name == "cghmm":
        learner = result["segmentation"].model
        metrics = dict(result["constraints"]["metrics"])
        gammas = result["segmentation"].extras.get("gammas")
        if gammas is None:
            raise ValueError("cghmm segmentation result is missing gammas in extras.")
    else:
        learner = result["constraints"]["model"]
        metrics = dict(result["constraints"]["metrics"])
        gammas = result["constraints"]["gammas"]
    metrics["loglik"] = float(learner.loss_loglik[-1]) if getattr(learner, "loss_loglik", None) else np.nan
    return metrics, learner, gammas


def run_single_segcons(taus_init, seed, extra_method_kwargs=None):
    dataset_kwargs = _effective_dataset_kwargs(DATASET_NAME, "segcons")
    method_kwargs = _effective_joint_method_kwargs(
        "segcons",
        seed,
        taus_init,
        extra_method_kwargs=extra_method_kwargs,
    )
    result = run_experiment(
        dataset_name=DATASET_NAME,
        method_name="segcons",
        dataset_kwargs=dataset_kwargs,
        method_kwargs=method_kwargs,
    )
    return _extract_metrics("segcons", result)


def run_single_ccp(taus_init, seed, extra_method_kwargs=None):
    dataset_kwargs = _effective_dataset_kwargs(DATASET_NAME, "ccp")
    method_kwargs = _effective_joint_method_kwargs(
        "ccp",
        seed,
        taus_init,
        extra_method_kwargs=extra_method_kwargs,
    )
    result = run_experiment(
        dataset_name=DATASET_NAME,
        method_name="ccp",
        dataset_kwargs=dataset_kwargs,
        method_kwargs=method_kwargs,
    )
    return _extract_metrics("ccp", result)


def run_single_cghmm(taus_init, seed, extra_method_kwargs=None):
    dataset_kwargs = _effective_dataset_kwargs(DATASET_NAME, "cghmm")
    _, base_method_cfg = _load_joint_config(DATASET_NAME, "cghmm")
    method_kwargs = dict(base_method_cfg)
    segmenter_cfg = dict(method_kwargs.get("segmenter", {}))
    constraints_cfg = dict(method_kwargs.get("constraints", {}))
    segmenter_cfg["tau_init"] = np.asarray(taus_init, dtype=int)
    segmenter_cfg["seed"] = int(seed)
    segmenter_cfg["verbose"] = False
    segmenter_cfg["plot_every"] = None
    constraints_cfg["verbose"] = False
    method_kwargs["segmenter"] = segmenter_cfg
    method_kwargs["constraints"] = constraints_cfg
    if extra_method_kwargs:
        method_kwargs = deep_merge(method_kwargs, extra_method_kwargs)
    result = run_experiment(
        dataset_name=DATASET_NAME,
        method_name="cghmm",
        dataset_kwargs=dataset_kwargs,
        method_kwargs=method_kwargs,
    )
    return _extract_metrics("cghmm", result)


RUNNERS = {
    "segcons": run_single_segcons,
    "ccp": run_single_ccp,
    "cghmm": run_single_cghmm,
}


def _goal_record(learner, gammas, seed):
    return {
        "seed": int(seed),
        "learner": learner,
        "gammas": gammas,
    }


def _final_loss_value(learner):
    values = getattr(learner, "loss_loglik", None)
    if values:
        return float(values[-1])
    values = getattr(learner, "loss_total", None)
    if values:
        return float(values[-1])
    return float("nan")


def plot_boxplots(results, metric_names, group_names):
    if plt is None or not metric_names:
        return
    labels = [f"{group}-{model}" for group in group_names for model in MODELS]
    n_metrics = len(metric_names)
    ncols = min(3, n_metrics)
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, metric in zip(axes, metric_names):
        vals = [results[group][model][metric] for group in group_names for model in MODELS]
        ax.boxplot(vals, tick_labels=labels, showmeans=True, meanline=True)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(alpha=0.3)
    for ax in axes[n_metrics:]:
        ax.axis("off")
    save_figure(fig, SAVE_PATH.with_name(SAVE_PATH.stem + "_boxplots.png"))


def plot_goal_grids(goal_records, dataset, group_names):
    if plt is None:
        return
    runs_per_row = 4
    for model in MODELS:
        row_blocks_per_scheme = {
            scheme: int(np.ceil(max(len(goal_records[scheme][model]), 1) / runs_per_row))
            for scheme in group_names
        }
        row_offsets = {}
        offset = 0
        for scheme in group_names:
            row_offsets[scheme] = offset
            offset += row_blocks_per_scheme[scheme]
        total_rows = max(offset, 1)
        fig, axes = plt.subplots(
            total_rows,
            runs_per_row,
            figsize=(2.7 * runs_per_row, 2.1 * total_rows),
            squeeze=False,
            constrained_layout=False,
        )
        for scheme in group_names:
            n_runs = len(goal_records[scheme][model])
            for run_idx in range(n_runs):
                row = row_offsets[scheme] + (run_idx // runs_per_row)
                col = run_idx % runs_per_row
                ax = axes[row, col]
                record = goal_records[scheme][model][run_idx]
                learner = record["learner"]
                gammas = record["gammas"]
                taus = extract_taus_hat(gammas)
                base_label = record.get("label", f"seed={record['seed']}")
                if base_label.startswith(f"{scheme}\n"):
                    base_label = base_label[len(scheme) + 1 :]
                elif base_label == scheme:
                    base_label = ""
                loss_value = _final_loss_value(learner)
                title_lines = []
                if col == 0:
                    title_lines.append(str(scheme))
                if base_label:
                    title_lines.append(base_label)
                title_lines.append(f"loss={loss_value:.1f}")
                title = "\n".join(title_lines)
                show_legend = row == 0 and col == 0
                plot_demos_goals_snapshot(ax, learner, taus, gammas, title=title, show_legend=show_legend)
                ax.title.set_fontsize(7)
                if col != 0:
                    ax.set_ylabel("")
                if row != total_rows - 1:
                    ax.set_xlabel("")
            row_block_count = row_blocks_per_scheme[scheme]
            used_in_last_row = n_runs % runs_per_row
            if used_in_last_row != 0:
                row = row_offsets[scheme] + (row_block_count - 1)
                for col in range(used_in_last_row, runs_per_row):
                    axes[row, col].axis("off")
            for blank_row in range(row_offsets[scheme] + int(np.ceil(max(n_runs, 1) / runs_per_row)), row_offsets[scheme] + row_block_count):
                for col in range(runs_per_row):
                    axes[blank_row, col].axis("off")
        fig.subplots_adjust(left=0.04, right=0.995, top=0.98, bottom=0.05, wspace=0.16, hspace=0.18)
        save_figure(fig, SAVE_PATH.with_name(f"{SAVE_PATH.stem}_{model}_goals.png"))


def run_experiment_sensitivity():
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset_kwargs = _effective_dataset_kwargs(DATASET_NAME, MODELS[0])
    dataset = load_env(DATASET_NAME, **dataset_kwargs)
    group_names, comparison_specs = build_comparison_specs(dataset)
    results = {scheme: {model: {} for model in MODELS} for scheme in group_names}
    goal_records = {scheme: {model: [] for model in MODELS} for scheme in group_names}
    metric_names = None

    for run, spec in enumerate(comparison_specs):
            group = spec["group"]
            seed = int(spec["method_seed"])
            taus_init = np.asarray(spec["tau_init"], dtype=int)
            run_outputs = {}
            for model_name in MODELS:
                if model_name not in RUNNERS:
                    raise ValueError(f"Unsupported model '{model_name}' in MODELS.")
                run_outputs[model_name] = RUNNERS[model_name](
                    taus_init,
                    seed,
                    spec.get("method_overrides", {}),
                )
            if metric_names is None:
                union_metrics = {}
                for metrics, _, _ in run_outputs.values():
                    union_metrics.update(metrics)
                metric_names = _ordered_metric_names(union_metrics)
                for scheme_name in group_names:
                    for model_name in MODELS:
                        for metric_name in metric_names:
                            results[scheme_name][model_name].setdefault(metric_name, [])

            for model_name, (metrics, learner, gammas) in run_outputs.items():
                for key in metric_names:
                    results[group][model_name][key].append(float(metrics.get(key, np.nan)))
                record = _goal_record(learner, gammas, seed)
                record["label"] = spec["label"]
                goal_records[group][model_name].append(record)

            summary_bits = [
                f"{model_name}={float(run_outputs[model_name][0].get('NMAE_tau', np.nan)):.3f}"
                for model_name in MODELS
            ]
            print(f"[{group}][run={run}] " + " ".join(summary_bits))

    SAVE_PATH.write_text(json.dumps(results, indent=2))
    print(f"[Saved] {SAVE_PATH}")
    plot_boxplots(results, metric_names or [], group_names)
    plot_goal_grids(goal_records, dataset, group_names)
    return results


if __name__ == "__main__":
    run_experiment_sensitivity()
