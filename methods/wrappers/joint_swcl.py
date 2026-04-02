from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from envs.base import TaskBundle
from evaluation import evaluate_model_metrics
from visualization.swcl_4panel import plot_swcl_results_4panel_overview
from visualization.io import plot_root, save_figure

from ..cores.swcl import StageWiseConstraintLearningModel

try:
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def _is_stage_sweep(value: Any) -> bool:
    return isinstance(value, (list, tuple, range))


def _fit_single_swcl(kwargs: Dict[str, Any], dataset: TaskBundle) -> Dict[str, Any]:
    learner = StageWiseConstraintLearningModel(
        demos=dataset.demos,
        env=dataset.env,
        true_taus=dataset.true_taus,
        true_cutpoints=getattr(dataset, "true_cutpoints", None),
        n_stages=kwargs.get("n_stages", 2),
        seed=kwargs.get("seed", 0),
        selected_raw_feature_ids=kwargs.get("selected_raw_feature_ids"),
        feature_model_types=kwargs.get("feature_model_types"),
        fixed_feature_mask=kwargs.get("fixed_feature_mask"),
        lambda_eq_constraint=kwargs.get("lambda_eq_constraint", kwargs.get("lambda_constraint", 1.0)),
        lambda_ineq_constraint=kwargs.get("lambda_ineq_constraint", kwargs.get("lambda_constraint", 1.0)),
        lambda_progress=kwargs.get("lambda_progress", 1.0),
        lambda_subgoal_consensus=kwargs.get(
            "lambda_subgoal_consensus",
            kwargs.get("lambda_consensus", 1.0),
        ),
        lambda_param_consensus=kwargs.get("lambda_param_consensus", 1.0),
        lambda_activation_consensus=kwargs.get(
            "lambda_activation_consensus",
            kwargs.get(
                "lambda_feature_score_consensus",
                kwargs.get("lambda_r_consensus", 1.0),
            ),
        ),
        consensus_schedule=kwargs.get("consensus_schedule", "linear"),
        progress_delta_scale=kwargs.get("progress_delta_scale", 20.0),
        duration_min=kwargs.get("duration_min"),
        duration_max=kwargs.get("duration_max"),
        feature_activation_mode=kwargs.get("feature_activation_mode", "fixed_mask"),
        equality_score_mode=kwargs.get("equality_score_mode", "dispersion"),
        equality_dispersion_ratio_threshold=kwargs.get("equality_dispersion_ratio_threshold", 0.1),
        constraint_core_trim=kwargs.get("constraint_core_trim", 0),
        short_segment_penalty_c=kwargs.get(
            "short_segment_penalty_c",
            kwargs.get("equality_score_uncertainty_c", 0.1),
        ),
        inequality_score_activation_threshold=kwargs.get("inequality_score_activation_threshold", -0.5),
        activation_proto_temperature=kwargs.get("activation_proto_temperature", 0.1),
        joint_mask_search_max_masks=kwargs.get("joint_mask_search_max_masks", 4096),
        fixed_true_cutpoint_prefix=kwargs.get("fixed_true_cutpoint_prefix", 0),
        fixed_true_cutpoint_indices=kwargs.get("fixed_true_cutpoint_indices"),
        plot_every=kwargs.get("plot_every"),
        plot_dir=kwargs.get("plot_dir", "outputs/plots"),
        verbose=kwargs.get("verbose", True),
    )
    gammas = learner.fit(
        max_iter=kwargs.get("max_iter", 30),
        verbose=kwargs.get("verbose", True),
    )
    metrics = evaluate_model_metrics(learner, gammas, None)
    cutpoints_hat: List[List[int]] = [[int(x) for x in ends[:-1]] for ends in learner.stage_ends_]
    taus_hat: List[int] = [cuts[0] for cuts in cutpoints_hat] if learner.num_stages == 2 else []
    total_cost = float(learner.loss_total[-1]) if getattr(learner, "loss_total", None) else float("inf")
    constraint_cost = float(learner.loss_constraint[-1]) if getattr(learner, "loss_constraint", None) else 0.0
    short_segment_penalty = (
        float(learner.loss_short_segment_penalty[-1]) if getattr(learner, "loss_short_segment_penalty", None) else 0.0
    )
    progress_cost = (
        float(getattr(learner, "lambda_progress", 1.0))
        * float(learner.loss_progress[-1])
        if getattr(learner, "loss_progress", None)
        else 0.0
    )
    subgoal_consensus_cost = (
        float(getattr(learner, "current_subgoal_consensus_lambda", 0.0))
        * float(learner.loss_subgoal_consensus[-1])
        if getattr(learner, "loss_subgoal_consensus", None)
        else 0.0
    )
    param_consensus_cost = (
        float(getattr(learner, "current_param_consensus_lambda", 0.0))
        * float(learner.loss_param_consensus[-1])
        if getattr(learner, "loss_param_consensus", None)
        else 0.0
    )
    activation_consensus_cost = (
        float(getattr(learner, "current_activation_consensus_lambda", 0.0))
        * float(learner.loss_activation_consensus[-1])
        if getattr(learner, "loss_activation_consensus", None)
        else 0.0
    )
    n_stages = int(learner.num_stages)
    stage_averaged_cost = (
        constraint_cost
        + progress_cost
        + (
            short_segment_penalty
            + subgoal_consensus_cost
            + param_consensus_cost
            + activation_consensus_cost
        ) / max(n_stages, 1)
    )
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
        "short_segment_penalty": short_segment_penalty,
        "progress_cost": progress_cost,
        "subgoal_consensus_cost": subgoal_consensus_cost,
        "param_consensus_cost": param_consensus_cost,
        "activation_consensus_cost": activation_consensus_cost,
        "stage_averaged_cost": float(stage_averaged_cost),
        "n_stages": n_stages,
    }


def _plot_stage_sweep_cost(sweep_results: List[Dict[str, Any]], *, plot_dir: str | None = None) -> None:
    if plt is None or not sweep_results:
        return
    ks = [int(item["n_stages"]) for item in sweep_results]
    stage_avg_costs = [float(item["stage_averaged_cost"]) for item in sweep_results]
    penalized_costs = [float(item["penalized_stage_cost"]) for item in sweep_results]
    fig, ax = plt.subplots(figsize=(5.2, 3.6), constrained_layout=False)
    ax.plot(ks, stage_avg_costs, marker="^", color="#54A24B", linewidth=1.8, markersize=5.5, label="stage-averaged cost")
    ax.plot(ks, penalized_costs, marker="s", color="#E45756", linewidth=1.6, markersize=5.0, label="stage-averaged + penalty")
    ax.set_xlabel("num stages", fontsize=10)
    ax.set_ylabel("cost", fontsize=10)
    ax.set_title("SWCL stage-count sweep", fontsize=11, pad=6)
    ax.set_xticks(ks)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout(pad=0.6)
    save_figure(fig, plot_root(plot_dir) / "training_summary_num_stages_sweep.png", dpi=220)


@dataclass
class JointSWCLMethod:
    kwargs: Dict[str, Any]

    def fit(self, dataset: TaskBundle) -> Dict[str, Any]:
        if dataset.env is None:
            raise ValueError("swcl requires a dataset env.")
        if _is_stage_sweep(self.kwargs.get("n_stages")):
            stage_candidates = [int(x) for x in self.kwargs.get("n_stages", [])]
            if not stage_candidates:
                raise ValueError("swcl stage sweep requires a non-empty n_stages list.")
            if any(k < 2 for k in stage_candidates):
                raise ValueError("swcl stage sweep candidates must all be at least 2.")
            stage_count_penalty = float(self.kwargs.get("stage_count_penalty", 1.0))

            sweep_results: List[Dict[str, Any]] = []
            for n_stages in stage_candidates:
                run_kwargs = dict(self.kwargs)
                run_kwargs["n_stages"] = int(n_stages)
                run_kwargs["plot_every"] = None
                single_result = _fit_single_swcl(run_kwargs, dataset)
                single_result["stage_count_penalty"] = stage_count_penalty
                single_result["penalized_stage_cost"] = float(
                    single_result["stage_averaged_cost"] + stage_count_penalty * float(single_result["n_stages"])
                )
                sweep_results.append(single_result)
                if not bool(self.kwargs.get("disable_plots", False)):
                    plot_swcl_results_4panel_overview(
                        single_result["model"],
                        int(run_kwargs.get("max_iter", 30)),
                        metrics=single_result["metrics"],
                        plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
                        save_name=f"training_summary_K{int(n_stages):02d}.png",
                        include_constraint_type_summary=False,
                    )

            if not bool(self.kwargs.get("disable_plots", False)):
                _plot_stage_sweep_cost(sweep_results, plot_dir=self.kwargs.get("plot_dir", "outputs/plots"))

            best_result = min(
                sweep_results,
                key=lambda item: (float(item["total_cost"]), int(item["n_stages"])),
            )
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
                    "short_segment_penalty": float(item["short_segment_penalty"]),
                    "subgoal_consensus_cost": float(item["subgoal_consensus_cost"]),
                    "param_consensus_cost": float(item["param_consensus_cost"]),
                    "activation_consensus_cost": float(item["activation_consensus_cost"]),
                    "metrics": dict(item["metrics"]),
                }
                for item in sweep_results
            ]
            best_result["selected_n_stages"] = int(best_result["n_stages"])
            return best_result

        single_result = _fit_single_swcl(self.kwargs, dataset)
        if not bool(self.kwargs.get("disable_plots", False)):
            plot_swcl_results_4panel_overview(
                single_result["model"],
                int(self.kwargs.get("max_iter", 30)),
                metrics=single_result["metrics"],
                plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
            )
        return single_result
