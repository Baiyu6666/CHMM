from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from envs.base import TaskBundle

from ..cores.ccp import ConstraintCompletionProgressModel


@dataclass
class JointCCPMethod:
    kwargs: Dict[str, Any]

    def fit(self, dataset: TaskBundle) -> Dict[str, Any]:
        if dataset.env is None:
            raise ValueError("ccp requires a dataset env.")

        learner = ConstraintCompletionProgressModel(
            demos=dataset.demos,
            env=dataset.env,
            true_taus=dataset.true_taus,
            n_states=self.kwargs.get("n_states", 2),
            tau_init=self.kwargs.get("tau_init"),
            tau_init_mode=self.kwargs.get("tau_init_mode", "uniform_taus"),
            seed=self.kwargs.get("seed", 0),
            selected_raw_feature_ids=self.kwargs.get("selected_raw_feature_ids"),
            feature_model_types=self.kwargs.get("feature_model_types"),
            auto_feature_select=self.kwargs.get("auto_feature_select", False),
            fixed_feature_mask=self.kwargs.get("fixed_feature_mask"),
            r_sparse_lambda=self.kwargs.get("r_sparse_lambda", 0.0),
            lambda_constraint=self.kwargs.get("lambda_constraint", 1.0),
            lambda_end=self.kwargs.get("lambda_end", 0.5),
            lambda_progress=self.kwargs.get("lambda_progress", self.kwargs.get("lambda_gain", 1.0)),
            progress_term_type=self.kwargs.get("progress_term_type", "ranking"),
            progress_delta_scale=self.kwargs.get("progress_delta_scale", self.kwargs.get("delta_progress_scale", 20.0)),
            duration_min=self.kwargs.get("duration_min"),
            duration_max=self.kwargs.get("duration_max"),
            duration_slack=self.kwargs.get("duration_slack", 0),
            progress_rank_hidden_dim=self.kwargs.get("progress_rank_hidden_dim", self.kwargs.get("gain_hidden_dim", 32)),
            progress_rank_steps=self.kwargs.get("progress_rank_steps", self.kwargs.get("gain_steps", 100)),
            progress_rank_lr=self.kwargs.get("progress_rank_lr", self.kwargs.get("gain_lr", 1e-2)),
            progress_rank_weight_decay=self.kwargs.get("progress_rank_weight_decay", self.kwargs.get("gain_weight_decay", 1e-4)),
            hard_negative_radius=self.kwargs.get("hard_negative_radius", 2),
            random_negative_per_demo=self.kwargs.get("random_negative_per_demo", 4),
            end_precision_floor=self.kwargs.get("end_precision_floor", 1e-3),
            plot_every=self.kwargs.get("plot_every"),
            plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
        )
        gammas = learner.fit(
            max_iter=self.kwargs.get("max_iter", 30),
            verbose=self.kwargs.get("verbose", True),
        )
        taus_hat: List[int] | None
        if learner.num_states == 2:
            taus_hat = [int(ends[0]) for ends in learner.stage_ends_]
        else:
            taus_hat = None
        return {
            "model": learner,
            "gammas": gammas,
            "taus_hat": taus_hat,
            "metrics": {k: v[-1] for k, v in learner.metrics_hist.items()} if learner.metrics_hist else {},
        }
