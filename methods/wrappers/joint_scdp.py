from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from envs.base import TaskBundle

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
            n_states=self.kwargs.get("n_states", 2),
            tau_init=self.kwargs.get("tau_init"),
            tau_init_mode=self.kwargs.get("tau_init_mode", "uniform_taus"),
            seed=self.kwargs.get("seed", 0),
            selected_raw_feature_ids=self.kwargs.get("selected_raw_feature_ids"),
            feature_model_types=self.kwargs.get("feature_model_types"),
            fixed_feature_mask=self.kwargs.get("fixed_feature_mask"),
            lambda_constraint=self.kwargs.get("lambda_constraint", 1.0),
            lambda_progress=self.kwargs.get("lambda_progress", 1.0),
            lambda_consensus=self.kwargs.get("lambda_consensus", 1.0),
            lambda_r_consensus=self.kwargs.get("lambda_r_consensus", 1.0),
            consensus_schedule=self.kwargs.get("consensus_schedule", "linear"),
            progress_delta_scale=self.kwargs.get("progress_delta_scale", 20.0),
            duration_min=self.kwargs.get("duration_min"),
            duration_max=self.kwargs.get("duration_max"),
            sigma_floor=self.kwargs.get("sigma_floor", 0.1),
            lam_floor=self.kwargs.get("lam_floor", 0.1),
            auto_feature_activation=self.kwargs.get(
                "auto_feature_activation",
                self.kwargs.get("demo_local_baseline_gate", False),
            ),
            demo_local_baseline_gate=self.kwargs.get("demo_local_baseline_gate", False),
            equality_w70_ratio_threshold=self.kwargs.get("equality_w70_ratio_threshold", 0.2),
            plot_every=self.kwargs.get("plot_every"),
            plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
        )
        gammas = learner.fit(
            max_iter=self.kwargs.get("max_iter", 30),
            verbose=self.kwargs.get("verbose", True),
        )
        taus_hat: List[int] = [int(ends[0]) for ends in learner.stage_ends_]
        return {
            "model": learner,
            "gammas": gammas,
            "taus_hat": taus_hat,
            "metrics": {k: v[-1] for k, v in learner.metrics_hist.items()} if learner.metrics_hist else {},
            "demo_r_matrices": [r.tolist() for r in learner.demo_r_matrices_],
        }
