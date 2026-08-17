from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from envs.base import TaskBundle
from evaluation import evaluate_model_metrics
from visualization.io import plot_root, save_figure
from visualization.map_plots import _map_learned_constraint_payload, plot_map_results_overview

from ..cores.map import StageWiseMAPConstraintLearningModel

try:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def _is_stage_sweep(value: Any) -> bool:
    return isinstance(value, (list, tuple, range))


def _fit_single_map(kwargs: Dict[str, Any], dataset: TaskBundle) -> Dict[str, Any]:
    unsupported_weights = sorted(
        key
        for key in (
            "lambda_constraint",
            "lambda_eq_constraint",
            "lambda_ineq_constraint",
            "map_inactive_weight",
            "map_active_mode_penalty",
            "lambda_progress",
            "progress_delta_scale",
        )
        if key in kwargs
    )
    if unsupported_weights:
        raise ValueError(
            "MAP uses normalized likelihood costs; remove unsupported weighting parameters: "
            + ", ".join(unsupported_weights)
        )
    learner = StageWiseMAPConstraintLearningModel(
        demos=dataset.demos,
        env=dataset.env,
        precomputed_features=dataset.features,
        true_taus=dataset.true_taus,
        true_cutpoints=getattr(dataset, "true_cutpoints", None),
        n_stages=kwargs.get("n_stages", 2),
        seed=kwargs.get("seed", 0),
        selected_raw_feature_ids=kwargs.get("selected_raw_feature_ids"),
        fixed_feature_mask=kwargs.get("fixed_feature_mask"),
        force_inactive_feature_ids=kwargs.get("force_inactive_feature_ids"),
        duration_min=kwargs.get("duration_min"),
        duration_max=kwargs.get("duration_max"),
        constraint_core_trim=kwargs.get("constraint_core_trim", 0),
        short_segment_penalty_c=0.0,
        truncated_z_soft_boundary_scale=kwargs.get("truncated_z_soft_boundary_scale", 0.1),
        truncated_z_observation_noise_scale=kwargs.get("truncated_z_observation_noise_scale", 0.003),
        truncated_z_half_t_scale_quantile=kwargs.get("truncated_z_half_t_scale_quantile", 0.9),
        fixed_true_cutpoint_prefix=kwargs.get("fixed_true_cutpoint_prefix", 0),
        fixed_true_cutpoint_indices=kwargs.get("fixed_true_cutpoint_indices"),
        plot_every=kwargs.get("plot_every"),
        plot_dir=kwargs.get("plot_dir", "outputs/plots"),
        disable_plots=kwargs.get("disable_plots", False),
        verbose=kwargs.get("verbose", True),
        map_eq_sigma=kwargs.get("map_eq_sigma", 0.05),
        map_c_bg=kwargs.get("map_c_bg", 2.0),
        map_c_ineq=kwargs.get("map_c_ineq", 0.0),
        map_eq_distribution=kwargs.get("map_eq_distribution", "gaussian"),
        map_inactive_distribution=kwargs.get("map_inactive_distribution", "gaussian"),
        map_nu_eq=kwargs.get("map_nu_eq", 3.0),
        map_nu_inactive=kwargs.get("map_nu_inactive", 3.0),
        map_nu_ineq=kwargs.get("map_nu_ineq", 3.0),
        map_boundary_quantile=kwargs.get("map_boundary_quantile", 0.05),
        map_activation_prior=kwargs.get("map_activation_prior"),
        map_active_mode_prior=kwargs.get("map_active_mode_prior"),
        map_mode_aggregation=kwargs.get("map_mode_aggregation", "shared_vote"),
        map_vote_prior_scope=kwargs.get("map_vote_prior_scope", "shared"),
        map_refit_winning_voters=kwargs.get("map_refit_winning_voters", False),
        map_convergence_tol=kwargs.get("map_convergence_tol", 1e-6),
        map_demo_num_workers=kwargs.get("map_demo_num_workers"),
        map_mstep_boundary_trim=kwargs.get("map_mstep_boundary_trim", 0),
        map_progress_kappa=kwargs.get("map_progress_kappa"),
        map_progress_kappa_max=kwargs.get("map_progress_kappa_max", 100.0),
    )
    gammas = learner.fit(
        max_iter=kwargs.get("max_iter", 30),
        verbose=kwargs.get("verbose", True),
    )
    metrics = _map_learned_constraint_payload(learner, evaluate_model_metrics(learner, gammas, None))
    cutpoints_hat: List[List[int]] = [[int(x) for x in ends[:-1]] for ends in learner.stage_ends_]
    taus_hat: List[int] = [cuts[0] for cuts in cutpoints_hat] if learner.num_stages == 2 else []
    total_cost = float(learner.loss_total[-1]) if getattr(learner, "loss_total", None) else float("inf")
    constraint_cost = float(learner.loss_constraint[-1]) if getattr(learner, "loss_constraint", None) else 0.0
    progress_cost = float(learner.loss_progress[-1]) if getattr(learner, "loss_progress", None) else 0.0
    n_stages = int(learner.num_stages)
    final_plot_iter = max(len(getattr(learner, "loss_total", []) or []) - 1, 0)
    return {
        "model": learner,
        "gammas": gammas,
        "taus_hat": taus_hat,
        "cutpoints_hat": cutpoints_hat,
        "stage_ends_hat": [list(map(int, ends)) for ends in learner.stage_ends_],
        "metrics": metrics,
        "demo_r_matrices": [r.tolist() for r in learner.demo_r_matrices_],
        "demo_feature_score_matrices": [m.tolist() for m in getattr(learner, "demo_feature_score_matrices_", [])],
        "posthoc_activation_summary": getattr(learner, "posthoc_activation_summary_", None),
        "total_cost": total_cost,
        "constraint_cost": constraint_cost,
        "short_segment_penalty": 0.0,
        "progress_cost": progress_cost,
        "param_consensus_cost": 0.0,
        "activation_consensus_cost": 0.0,
        "stage_averaged_cost": float(constraint_cost + progress_cost),
        "n_stages": n_stages,
        "final_plot_iter": int(final_plot_iter),
        "map_mode_aggregation": str(learner.map_mode_aggregation),
        "map_vote_prior_scope": str(learner.map_vote_prior_scope),
        "map_refit_winning_voters": bool(learner.map_refit_winning_voters),
        "map_progress_kappa": (
            None if learner.map_progress_kappa is None else float(learner.map_progress_kappa)
        ),
        "map_progress_kappas": np.asarray(learner.map_progress_kappas_, dtype=float).tolist(),
        "map_progress_kappa_max": float(learner.map_progress_kappa_max),
        "map_progress_kappa_history": [
            np.asarray(values, dtype=float).tolist()
            for values in getattr(learner, "map_progress_kappa_history_", [])
        ],
        "map_shared_mode_votes": getattr(learner, "map_shared_mode_votes_", None),
    }


def _plot_stage_sweep_cost(sweep_results: List[Dict[str, Any]], *, plot_dir: str | None = None) -> None:
    if plt is None or not sweep_results:
        return
    ks = [int(item["n_stages"]) for item in sweep_results]
    stage_avg_costs = [float(item["stage_averaged_cost"]) for item in sweep_results]
    penalized_costs = [float(item["penalized_stage_cost"]) for item in sweep_results]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=False)
    ax.plot(ks, stage_avg_costs, marker="^", linewidth=1.8, markersize=5.5, label="stage-averaged cost")
    ax.plot(ks, penalized_costs, marker="s", linewidth=1.6, markersize=5.0, label="stage-averaged + penalty")
    ax.set_xlabel("num stages", fontsize=10)
    ax.set_ylabel("cost", fontsize=10)
    ax.set_title("MAP stage-count sweep", fontsize=11, pad=6)
    ax.set_xticks(ks)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(pad=0.6)
    save_figure(fig, plot_root(plot_dir) / "training_summary_num_stages_sweep.png", dpi=220)


@dataclass
class JointMAPMethod:
    kwargs: Dict[str, Any]

    def fit(self, dataset: TaskBundle) -> Dict[str, Any]:
        if dataset.env is None:
            raise ValueError("map requires a dataset env.")
        if _is_stage_sweep(self.kwargs.get("n_stages")):
            stage_candidates = [int(x) for x in self.kwargs.get("n_stages", [])]
            if not stage_candidates:
                raise ValueError("map stage sweep requires a non-empty n_stages list.")
            if any(k < 2 for k in stage_candidates):
                raise ValueError("map stage sweep candidates must all be at least 2.")
            stage_count_penalty = float(self.kwargs.get("stage_count_penalty", 1.0))
            sweep_results: List[Dict[str, Any]] = []
            for n_stages in stage_candidates:
                run_kwargs = dict(self.kwargs)
                run_kwargs["n_stages"] = int(n_stages)
                run_kwargs["plot_every"] = None
                single_result = _fit_single_map(run_kwargs, dataset)
                single_result["stage_count_penalty"] = stage_count_penalty
                single_result["penalized_stage_cost"] = float(
                    single_result["stage_averaged_cost"] + stage_count_penalty * float(single_result["n_stages"])
                )
                sweep_results.append(single_result)
                if not bool(self.kwargs.get("disable_plots", False)):
                    plot_map_results_overview(
                        single_result["model"],
                        int(single_result.get("final_plot_iter", 0)),
                        metrics=single_result["metrics"],
                        plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
                        save_name=f"training_summary_K{int(n_stages):02d}.png",
                    )
            if not bool(self.kwargs.get("disable_plots", False)):
                _plot_stage_sweep_cost(sweep_results, plot_dir=self.kwargs.get("plot_dir", "outputs/plots"))
            best_result = min(sweep_results, key=lambda item: (float(item["total_cost"]), int(item["n_stages"])))
            best_result = dict(best_result)
            best_result["stage_count_sweep"] = [
                {
                    "n_stages": int(item["n_stages"]),
                    "total_cost": float(item["total_cost"]),
                    "stage_averaged_cost": float(item["stage_averaged_cost"]),
                    "stage_count_penalty": float(item["stage_count_penalty"]),
                    "penalized_stage_cost": float(item["penalized_stage_cost"]),
                    "constraint_cost": float(item["constraint_cost"]),
                    "progress_cost": float(item["progress_cost"]),
                    "short_segment_penalty": 0.0,
                    "param_consensus_cost": 0.0,
                    "activation_consensus_cost": 0.0,
                    "metrics": dict(item["metrics"]),
                }
                for item in sweep_results
            ]
            best_result["selected_n_stages"] = int(best_result["n_stages"])
            return best_result

        single_result = _fit_single_map(self.kwargs, dataset)
        if not bool(self.kwargs.get("disable_plots", False)):
            plot_map_results_overview(
                single_result["model"],
                int(single_result.get("final_plot_iter", 0)),
                metrics=single_result["metrics"],
                plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
            )
        return single_result
