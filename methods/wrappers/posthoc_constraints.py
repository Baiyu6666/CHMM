from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from envs.base import TaskBundle
from evaluation import eval_goalhmm_auto
from ..base import SegmentationResult, format_training_log
from ..cores.segcons import SegmentConstraintModel
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


def _resolve_single_cut_taus(dataset: TaskBundle, segmentation: SegmentationResult) -> List[int]:
    if segmentation.taus is not None:
        return [int(t) for t in segmentation.taus]

    resolved: List[int] = []
    invalid_examples: List[str] = []
    for idx, (X, cuts) in enumerate(zip(dataset.demos, segmentation.cutpoints)):
        if len(cuts) != 1:
            invalid_examples.append(f"demo {idx}: {len(cuts)} cutpoints")
            continue
        resolved.append(int(cuts[0]))

    if invalid_examples:
        raise ValueError(
            "Posthoc constraint learning requires a single-cut segmentation for every demo. "
            f"Got incompatible cutpoints from '{segmentation.method_name}': {', '.join(invalid_examples)}"
        )
    return resolved


def _compute_fixed_tau_objective(learner, gammas, xis_list, aux_list):
    total_ll = 0.0
    total_feat_ll = 0.0
    total_prog_ll = 0.0
    total_trans_ll = 0.0

    for X, gamma, xi, aux in zip(learner.demos, gammas, xis_list, aux_list):
        ll_emit, ll_feat_k, ll_prog = learner._emission_loglik(X, return_parts=True)
        logA = learner._transition_logprob(X, return_aux=False)
        total_ll += float(np.sum(gamma * ll_emit))
        total_feat_ll += float(np.sum(gamma * ll_feat_k))
        total_prog_ll += float(np.sum(gamma * ll_prog))
        if xi is not None and len(xi) > 0:
            finite_mask = np.isfinite(logA)
            total_trans_ll += float(np.sum(xi[finite_mask] * logA[finite_mask]))

    return total_ll, total_feat_ll, total_prog_ll, total_trans_ll

@dataclass
class PostHocConstraintLearner:
    kwargs: Dict[str, Any]

    def fit(self, dataset: TaskBundle, segmentation: SegmentationResult) -> Dict[str, Any]:
        if dataset.env is None:
            raise ValueError("Posthoc constraint learner requires a dataset env.")
        resolved_kwargs = dict(self.kwargs)
        taus = _resolve_single_cut_taus(dataset, segmentation)

        learner = SegmentConstraintModel(
            demos=dataset.demos,
            env=dataset.env,
            true_taus=dataset.true_taus,
            tau_init=taus,
            g2_init=None,
            auto_feature_select=resolved_kwargs.get("auto_feature_select", False),
            fixed_feature_mask=resolved_kwargs.get("fixed_feature_mask"),
            r_sparse_lambda=resolved_kwargs.get("r_sparse_lambda", 0.3),
            selected_raw_feature_ids=resolved_kwargs.get("selected_raw_feature_ids"),
            feature_model_types=resolved_kwargs.get("feature_model_types"),
            feat_weight=resolved_kwargs.get("feat_weight", 1.0),
            prog_weight=resolved_kwargs.get("prog_weight", 1.0),
            trans_weight=0.0,
            plot_dir=resolved_kwargs.get("plot_dir", "outputs/plots"),
            plot_every=None,
            eval_fn=None,
        )
        learner.plot_context = "posthoc"

        gammas = _hard_gammas_from_taus(dataset.demos, taus)
        dummy_xis = [np.zeros((len(X) - 1, 2, 2), dtype=float) for X in dataset.demos]
        dummy_aux = [None for _ in dataset.demos]
        for gamma, xi in zip(gammas, dummy_xis):
            xi[:, 0, 0] = gamma[:-1, 0]
            xi[:, 1, 1] = gamma[:-1, 1]

        learner.loss_loglik = []
        learner.loss_feat = []
        learner.loss_prog = []
        learner.loss_trans = []
        learner.metrics_hist = {}
        learner.loss_label = "Objective"

        refine_steps = int(resolved_kwargs.get("refine_steps", 5))
        verbose = bool(resolved_kwargs.get("verbose", True))
        for it in range(refine_steps):
            learner._mstep_update_features(gammas)
            learner._mstep_update_feature_mask(gammas)
            learner._mstep_update_goals(gammas, dummy_xis, dummy_aux)
            total_ll, total_feat_ll, total_prog_ll, total_trans_ll = _compute_fixed_tau_objective(
                learner, gammas, dummy_xis, dummy_aux
            )
            learner.loss_loglik.append(total_ll)
            learner.loss_feat.append(total_feat_ll)
            learner.loss_prog.append(total_prog_ll)
            learner.loss_trans.append(total_trans_ll)
            metrics = eval_goalhmm_auto(learner, gammas, dummy_xis)
            for name, value in metrics.items():
                learner.metrics_hist.setdefault(name, []).append(value)
            should_log = ((it + 1) % 10 == 0) or (it == refine_steps - 1)
            if verbose and should_log:
                print(
                    format_training_log(
                        "POSTHOC",
                        it,
                        losses={
                            "loss": total_ll,
                            "feat": total_feat_ll,
                            "prog": total_prog_ll,
                            "trans": total_trans_ll,
                        },
                        metrics=metrics,
                        extras={"taus": taus, "r": learner.r.tolist()},
                    )
                )

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
