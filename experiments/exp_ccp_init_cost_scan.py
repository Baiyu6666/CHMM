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
except ImportError:
    from experiments.config_loader import deep_merge, load_experiment_config

from methods.cores.ccp import ConstraintCompletionProgressModel
from visualization.io import save_figure


DATASET_NAME = "2DObsAvoid"
METHOD_NAME = "ccp"
SAVE_PATH = PROJECT_ROOT / "outputs" / "experiments" / "exp_ccp_init_cost_scan_results.json"
RATIOS = np.linspace(0.1, 0.9, 9)


def _load_joint_config(dataset_name: str, method_name: str):
    cfg = load_experiment_config(
        PROJECT_ROOT / "configs" / "envs" / f"{dataset_name}.json",
        PROJECT_ROOT / "configs" / "methods" / f"{method_name}.json",
    )
    dataset_cfg = dict(cfg["dataset"])
    method_cfg = dict(cfg["method"])
    dataset_cfg.pop("name", None)
    method_cfg.pop("name", None)
    dataset_method_overrides = dict(dataset_cfg.pop("method_overrides", {}))
    method_cfg = deep_merge(method_cfg, dataset_method_overrides.get(method_name, {}))
    return dataset_cfg, method_cfg


def _clip_tau(length: int, tau: int) -> int:
    return int(np.clip(int(tau), 1, int(length) - 2))


def _taus_from_ratio(demos, ratio: float) -> np.ndarray:
    taus = []
    for X in demos:
        tau = int(round(float(ratio) * (len(X) - 1)))
        taus.append(_clip_tau(len(X), tau))
    return np.asarray(taus, dtype=int)


def _build_model(dataset, method_cfg: dict, tau_init: np.ndarray) -> ConstraintCompletionProgressModel:
    return ConstraintCompletionProgressModel(
        demos=dataset.demos,
        env=dataset.env,
        true_taus=dataset.true_taus,
        n_states=method_cfg.get("n_states", 2),
        tau_init=tau_init,
        tau_init_mode=method_cfg.get("tau_init_mode", "uniform_taus"),
        seed=method_cfg.get("seed", 0),
        selected_raw_feature_ids=method_cfg.get("selected_raw_feature_ids"),
        feature_model_types=method_cfg.get("feature_model_types"),
        auto_feature_select=method_cfg.get("auto_feature_select", False),
        fixed_feature_mask=method_cfg.get("fixed_feature_mask"),
        r_sparse_lambda=method_cfg.get("r_sparse_lambda", 0.0),
        lambda_constraint=method_cfg.get("lambda_constraint", 1.0),
        lambda_end=method_cfg.get("lambda_end", 0.0),
        lambda_progress=method_cfg.get("lambda_progress", 1.0),
        progress_term_type=method_cfg.get("progress_term_type", "delta"),
        progress_delta_scale=method_cfg.get("progress_delta_scale", 20.0),
        duration_min=method_cfg.get("duration_min"),
        duration_max=method_cfg.get("duration_max"),
        duration_slack=method_cfg.get("duration_slack", 0),
        progress_rank_hidden_dim=method_cfg.get("progress_rank_hidden_dim", 32),
        progress_rank_steps=method_cfg.get("progress_rank_steps", 100),
        progress_rank_lr=method_cfg.get("progress_rank_lr", 1e-2),
        progress_rank_weight_decay=method_cfg.get("progress_rank_weight_decay", 1e-4),
        hard_negative_radius=method_cfg.get("hard_negative_radius", 2),
        random_negative_per_demo=method_cfg.get("random_negative_per_demo", 4),
        end_precision_floor=method_cfg.get("end_precision_floor", 1e-3),
        plot_every=None,
        plot_dir="outputs/plots",
    )


def _evaluate_init(dataset, method_cfg: dict, tau_init: np.ndarray, label: str) -> dict:
    learner = _build_model(dataset, method_cfg, tau_init)
    stage_ends = learner._initialize_stage_ends()
    learner.stage_ends_ = [list(item) for item in stage_ends]
    learner._update_feature_models(stage_ends)
    learner._update_completion_region(stage_ends)

    true_goal = np.asarray(dataset.env.goal, dtype=float)
    learner.end_mu[1] = true_goal.copy()
    learner.g2 = true_goal.copy()

    totals = learner._compute_total_objective(stage_ends)
    weighted = {
        "constraint": float(learner.lambda_constraint * totals["constraint"]),
        "end": float(learner.lambda_end * totals["end"]),
        "progress": float(learner.lambda_progress * totals["progress"]),
    }
    weighted["total"] = weighted["constraint"] + weighted["end"] + weighted["progress"]

    return {
        "label": label,
        "taus": [int(x) for x in tau_init],
        "mean_tau": float(np.mean(tau_init)),
        "g1": np.asarray(learner.end_mu[0], dtype=float).tolist(),
        "raw_costs": {k: float(v) for k, v in totals.items()},
        "weighted_costs": weighted,
    }


def _plot_results(rows: list[dict]) -> None:
    if plt is None:
        return
    ratio_rows = [row for row in rows if row["label"].startswith("ratio=")]
    if not ratio_rows:
        return

    xs = [float(row["label"].split("=")[1]) for row in ratio_rows]
    raw_total = [row["raw_costs"]["total"] for row in ratio_rows]
    raw_constraint = [row["raw_costs"]["constraint"] for row in ratio_rows]
    raw_end = [row["raw_costs"]["end"] for row in ratio_rows]
    raw_progress = [row["raw_costs"]["progress"] for row in ratio_rows]
    weighted_total = [row["weighted_costs"]["total"] for row in ratio_rows]
    weighted_constraint = [row["weighted_costs"]["constraint"] for row in ratio_rows]
    weighted_end = [row["weighted_costs"]["end"] for row in ratio_rows]
    weighted_progress = [row["weighted_costs"]["progress"] for row in ratio_rows]

    fig, axes = plt.subplots(2, 1, figsize=(7.6, 8.2), sharex=True)
    raw_ax, weighted_ax = axes

    raw_ax.plot(xs, raw_total, color="black", lw=1.6, marker="o", label="total")
    raw_ax.plot(xs, raw_constraint, color="tab:red", lw=1.1, marker="o", label="constraint")
    raw_ax.plot(xs, raw_end, color="tab:blue", lw=1.1, marker="o", label="completion")
    raw_ax.plot(xs, raw_progress, color="tab:orange", lw=1.1, marker="o", label="progress")

    weighted_ax.plot(xs, weighted_total, color="black", lw=1.6, marker="o", label="total")
    weighted_ax.plot(xs, weighted_constraint, color="tab:red", lw=1.1, marker="o", label="constraint")
    weighted_ax.plot(xs, weighted_end, color="tab:blue", lw=1.1, marker="o", label="completion")
    weighted_ax.plot(xs, weighted_progress, color="tab:orange", lw=1.1, marker="o", label="progress")

    gt_row = next((row for row in rows if row["label"] == "ground_truth_taus"), None)
    if gt_row is not None:
        raw_ax.axhline(gt_row["raw_costs"]["total"], color="black", linestyle=":", lw=1.0, label="gt total")
        raw_ax.axhline(gt_row["raw_costs"]["constraint"], color="tab:red", linestyle=":", lw=1.0, label="gt constraint")
        raw_ax.axhline(gt_row["raw_costs"]["end"], color="tab:blue", linestyle=":", lw=1.0, label="gt completion")
        raw_ax.axhline(gt_row["raw_costs"]["progress"], color="tab:orange", linestyle=":", lw=1.0, label="gt progress")

        weighted_ax.axhline(gt_row["weighted_costs"]["total"], color="black", linestyle=":", lw=1.0, label="gt total")
        weighted_ax.axhline(gt_row["weighted_costs"]["constraint"], color="tab:red", linestyle=":", lw=1.0, label="gt constraint")
        weighted_ax.axhline(gt_row["weighted_costs"]["end"], color="tab:blue", linestyle=":", lw=1.0, label="gt completion")
        weighted_ax.axhline(gt_row["weighted_costs"]["progress"], color="tab:orange", linestyle=":", lw=1.0, label="gt progress")

    raw_ax.set_title("CCP initial raw costs vs tau-init ratio")
    raw_ax.set_ylabel("raw cost")
    raw_ax.grid(alpha=0.3)

    weighted_ax.set_title("CCP initial weighted costs vs tau-init ratio")
    weighted_ax.set_xlabel("init ratio")
    weighted_ax.set_ylabel("weighted cost")
    weighted_ax.grid(alpha=0.3)

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            if l and l not in by_label:
                by_label[l] = h
        ax.legend(by_label.values(), by_label.keys(), frameon=False, loc="best")

    fig.tight_layout()
    save_figure(fig, SAVE_PATH.with_name(SAVE_PATH.stem + "_curves.png"))


def main():
    dataset_cfg, method_cfg = _load_joint_config(DATASET_NAME, METHOD_NAME)
    dataset = load_env(DATASET_NAME, **dataset_cfg)

    rows = []
    for ratio in RATIOS:
        tau_init = _taus_from_ratio(dataset.demos, float(ratio))
        rows.append(_evaluate_init(dataset, method_cfg, tau_init, f"ratio={ratio:.2f}"))

    true_taus = np.asarray(dataset.true_taus, dtype=int)
    rows.append(_evaluate_init(dataset, method_cfg, true_taus, "ground_truth_taus"))

    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": DATASET_NAME,
        "method": METHOD_NAME,
        "results": rows,
    }
    SAVE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _plot_results(rows)

    for row in rows:
        weighted = row["weighted_costs"]
        print(
            f"{row['label']}: total={weighted['total']:.6f} "
            f"constraint={weighted['constraint']:.6f} "
            f"end={weighted['end']:.6f} "
            f"progress={weighted['progress']:.6f} "
            f"mean_tau={row['mean_tau']:.3f}"
        )


if __name__ == "__main__":
    main()
