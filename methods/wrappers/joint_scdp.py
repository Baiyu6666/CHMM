from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from envs.base import TaskBundle
from evaluation import eval_goalhmm_auto
from visualization.scdp_4panel import plot_scdp_results_4panel_overview

from ..cores.scdp import SegmentConsensusDPModel


@dataclass
class JointSCDPMethod:
    kwargs: Dict[str, Any]

    def fit(self, dataset: TaskBundle) -> Dict[str, Any]:
        if dataset.env is None:
            raise ValueError("scdp requires a dataset env.")

        learner = SegmentConsensusDPModel(
            demos=dataset.demos,
            env=dataset.env,
            true_taus=dataset.true_taus,
            true_cutpoints=getattr(dataset, "true_cutpoints", None),
            n_states=self.kwargs.get("n_states", 2),
            seed=self.kwargs.get("seed", 0),
            selected_raw_feature_ids=self.kwargs.get("selected_raw_feature_ids"),
            feature_model_types=self.kwargs.get("feature_model_types"),
            fixed_feature_mask=self.kwargs.get("fixed_feature_mask"),
            lambda_eq_constraint=self.kwargs.get("lambda_eq_constraint", self.kwargs.get("lambda_constraint", 1.0)),
            lambda_ineq_constraint=self.kwargs.get("lambda_ineq_constraint", self.kwargs.get("lambda_constraint", 1.0)),
            lambda_progress=self.kwargs.get("lambda_progress", 1.0),
            lambda_subgoal_consensus=self.kwargs.get(
                "lambda_subgoal_consensus",
                self.kwargs.get("lambda_consensus", 1.0),
            ),
            lambda_param_consensus=self.kwargs.get("lambda_param_consensus", 1.0),
            lambda_activation_consensus=self.kwargs.get(
                "lambda_activation_consensus",
                self.kwargs.get(
                    "lambda_feature_score_consensus",
                    self.kwargs.get("lambda_r_consensus", 1.0),
                ),
            ),
            consensus_schedule=self.kwargs.get("consensus_schedule", "linear"),
            progress_delta_scale=self.kwargs.get("progress_delta_scale", 20.0),
            duration_min=self.kwargs.get("duration_min"),
            duration_max=self.kwargs.get("duration_max"),
            feature_activation_mode=self.kwargs.get("feature_activation_mode", "fixed_mask"),
            equality_score_mode=self.kwargs.get("equality_score_mode", "dispersion"),
            equality_dispersion_ratio_threshold=self.kwargs.get("equality_dispersion_ratio_threshold", 0.1),
            constraint_core_trim=self.kwargs.get("constraint_core_trim", 0),
            short_segment_penalty_c=self.kwargs.get(
                "short_segment_penalty_c",
                self.kwargs.get("equality_score_uncertainty_c", 0.1),
            ),
            inequality_score_activation_threshold=self.kwargs.get("inequality_score_activation_threshold", -0.5),
            activation_proto_temperature=self.kwargs.get("activation_proto_temperature", 0.1),
            joint_mask_search_max_masks=self.kwargs.get("joint_mask_search_max_masks", 4096),
            fixed_true_cutpoint_prefix=self.kwargs.get("fixed_true_cutpoint_prefix", 0),
            fixed_true_cutpoint_indices=self.kwargs.get("fixed_true_cutpoint_indices"),
            plot_every=self.kwargs.get("plot_every"),
            plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
            verbose=self.kwargs.get("verbose", True),
        )
        gammas = learner.fit(
            max_iter=self.kwargs.get("max_iter", 30),
            verbose=self.kwargs.get("verbose", True),
        )
        metrics = eval_goalhmm_auto(learner, gammas, None)
        plot_scdp_results_4panel_overview(
            learner,
            int(self.kwargs.get("max_iter", 30)),
            metrics=metrics,
            plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
        )
        cutpoints_hat: List[List[int]] = [[int(x) for x in ends[:-1]] for ends in learner.stage_ends_]
        taus_hat: List[int] = [cuts[0] for cuts in cutpoints_hat] if learner.num_states == 2 else []
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
        }
