from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from envs.base import TaskBundle
from evaluation import eval_goalhmm_auto
from ..common.tau_init import extract_taus_hat
from ..cores.segcons import SegmentConstraintModel


@dataclass
class JointSegConsMethod:
    kwargs: Dict[str, Any]

    def fit(self, dataset: TaskBundle) -> Dict[str, Any]:
        if dataset.env is None:
            raise ValueError("segcons requires a dataset env.")
        max_iter = self.kwargs.get("max_iter", 30)
        plot_every = self.kwargs.get("plot_every")
        if plot_every is None:
            plot_every = max_iter

        learner = SegmentConstraintModel(
            demos=dataset.demos,
            env=dataset.env,
            true_taus=dataset.true_taus,
            g2_init=self.kwargs.get("g2_init", None),
            tau_init=self.kwargs.get("tau_init"),
            tau_init_mode=self.kwargs.get("tau_init_mode", "uniform_taus"),
            seed=self.kwargs.get("seed", 0),
            selected_raw_feature_ids=self.kwargs.get("selected_raw_feature_ids"),
            feature_model_types=self.kwargs.get("feature_model_types"),
            auto_feature_select=self.kwargs.get("auto_feature_select", True),
            fixed_feature_mask=self.kwargs.get("fixed_feature_mask"),
            r_sparse_lambda=self.kwargs.get("r_sparse_lambda", 0.3),
            feat_weight=self.kwargs.get("feat_weight", 1.0),
            prog_weight=self.kwargs.get("prog_weight", 1.0),
            trans_weight=self.kwargs.get("trans_weight", 1.0),
            posterior_temp=self.kwargs.get("posterior_temp", 1.0),
            prog_kappa1=self.kwargs.get("prog_kappa1", 8.0),
            prog_kappa2=self.kwargs.get("prog_kappa2", 6.0),
            fixed_sigma_irrelevant=self.kwargs.get("fixed_sigma_irrelevant", 1.0),
            trans_eps=self.kwargs.get("trans_eps", 1e-6),
            delta_init=self.kwargs.get("delta_init", 0.15),
            trans_b_init=self.kwargs.get("trans_b_init", -1.0),
            learn_transition=self.kwargs.get("learn_transition", False),
            lr_delta=self.kwargs.get("lr_delta", 5e-4),
            lr_b=self.kwargs.get("lr_b", 5e-4),
            g_steps=self.kwargs.get("g_steps", 5),
            g_lr=self.kwargs.get("g_lr", 5e-4),
            g_grad_clip=self.kwargs.get("g_grad_clip"),
            g1_vmf_weight=self.kwargs.get("g1_vmf_weight", 1.0),
            g1_trans_weight=self.kwargs.get("g1_trans_weight", 1.0),
            plot_every=plot_every,
            plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
            eval_fn=self.kwargs.get("eval_fn", eval_goalhmm_auto),
        )

        gammas = learner.fit(
            max_iter=max_iter,
            verbose=self.kwargs.get("verbose", True),
        )
        taus_hat: List[int] = extract_taus_hat(gammas)
        return {
            "model": learner,
            "gammas": gammas,
            "taus_hat": taus_hat,
            "metrics": {k: v[-1] for k, v in learner.metrics_hist.items()} if learner.metrics_hist else {},
        }
