from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from envs.base import TaskBundle
from evaluation import eval_goalhmm_auto
from .base import SegmentationResult
from .segcons import SegCons
from visualization.plot4panel import plot_results_4panel


def _hard_gammas_from_taus(demos: List[np.ndarray], taus: List[int]) -> List[np.ndarray]:
    gammas = []
    for X, tau in zip(demos, taus):
        T = len(X)
        t = int(np.clip(tau, 1, T - 2))
        gamma = np.zeros((T, 2), dtype=float)
        gamma[: t + 1, 0] = 1.0
        gamma[t + 1 :, 1] = 1.0
        gammas.append(gamma)
    return gammas


@dataclass
class PostHocConstraintLearner:
    kwargs: Dict[str, Any]

    def fit(self, dataset: TaskBundle, segmentation: SegmentationResult) -> Dict[str, Any]:
        if dataset.env is None:
            raise ValueError("Posthoc constraint learner requires a dataset env.")
        taus = segmentation.taus
        if taus is None:
            taus = []
            for X, cuts in zip(dataset.demos, segmentation.cutpoints):
                if len(cuts) > 0:
                    taus.append(int(cuts[0]))
                else:
                    taus.append(max(1, len(X) // 2))

        learner = SegCons(
            demos=dataset.demos,
            env=dataset.env,
            true_taus=dataset.true_taus,
            tau_init=taus,
            g1_init="from_tau",
            g2_init="heuristic",
            auto_feature_select=self.kwargs.get("auto_feature_select", True),
            fixed_feature_mask=self.kwargs.get("fixed_feature_mask"),
            r_sparse_lambda=self.kwargs.get("r_sparse_lambda", 0.3),
            feature_ids=self.kwargs.get("feature_ids"),
            feature_types=self.kwargs.get("feature_types"),
            learned_features=self.kwargs.get("learned_features"),
            feat_weight=self.kwargs.get("feat_weight", 1.0),
            prog_weight=self.kwargs.get("prog_weight", 1.0),
            trans_weight=0.0,
            plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
            plot_every=None,
            eval_fn=None,
        )

        gammas = _hard_gammas_from_taus(dataset.demos, taus)
        dummy_xis = [np.zeros((len(X) - 1, 2, 2), dtype=float) for X in dataset.demos]
        dummy_aux = [None for _ in dataset.demos]

        refine_steps = int(self.kwargs.get("refine_steps", 5))
        for _ in range(refine_steps):
            learner._mstep_update_learned_features(gammas)
            learner._mstep_update_features(gammas)
            learner._mstep_update_feature_mask(gammas)
            learner._mstep_update_goals(gammas, dummy_xis, dummy_aux)

        dummy_alphas = [np.zeros_like(gamma) for gamma in gammas]
        dummy_betas = [np.zeros_like(gamma) for gamma in gammas]
        plot_results_4panel(
            learner,
            taus,
            refine_steps,
            gammas,
            dummy_alphas,
            dummy_betas,
            dummy_xis,
            dummy_aux,
        )

        metrics = eval_goalhmm_auto(learner, gammas, dummy_xis)
        return {
            "model": learner,
            "gammas": gammas,
            "metrics": metrics,
        }
