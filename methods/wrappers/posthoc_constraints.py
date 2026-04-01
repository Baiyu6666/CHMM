from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from envs.base import TaskBundle
from evaluation import evaluate_model_metrics
from ..base import SegmentationResult, format_training_log
from ..cores.posthoc_constraint_model import FixedTauConstraintModel
from visualization.plot4panel import plot_results_4panel


def _hard_gammas_from_labels(labels: List[np.ndarray], num_stages: int) -> List[np.ndarray]:
    gammas = []
    for z in labels:
        z = np.asarray(z, dtype=int).reshape(-1)
        gamma = np.zeros((len(z), int(num_stages)), dtype=float)
        gamma[np.arange(len(z)), z] = 1.0
        gammas.append(gamma)
    return gammas


def _stage_ends_from_labels(labels: List[np.ndarray]) -> List[List[int]]:
    stage_ends = []
    for z in labels:
        z = np.asarray(z, dtype=int).reshape(-1)
        cuts = np.where(np.diff(z) != 0)[0].astype(int)
        stage_ends.append([int(x) for x in cuts.tolist()] + [int(len(z) - 1)])
    return stage_ends


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
        labels = [np.asarray(z, dtype=int) for z in segmentation.labels]
        if getattr(segmentation.model, "stage_ends_", None) is not None:
            stage_ends = [
                [int(x) for x in np.asarray(ends, dtype=int).reshape(-1).tolist()]
                for ends in segmentation.model.stage_ends_
            ]
            num_stages = int(getattr(segmentation.model, "num_stages", len(stage_ends[0]) if stage_ends else 2))
        else:
            num_stages = int(max(int(np.max(z)) for z in labels) + 1) if labels else 2
            stage_ends = _stage_ends_from_labels(labels)

        learner = FixedTauConstraintModel(
            demos=dataset.demos,
            env=dataset.env,
            true_taus=dataset.true_taus,
            true_cutpoints=getattr(dataset, "true_cutpoints", None),
            num_stages=num_stages,
            stage_ends_init=stage_ends,
            g2_init=None,
            fixed_feature_mask=resolved_kwargs.get("fixed_feature_mask"),
            selected_raw_feature_ids=resolved_kwargs.get("selected_raw_feature_ids"),
            feature_model_types=resolved_kwargs.get("feature_model_types"),
            feat_weight=resolved_kwargs.get("feat_weight", 1.0),
            prog_weight=resolved_kwargs.get("prog_weight", 1.0),
            trans_weight=0.0,
            constraint_core_trim=resolved_kwargs.get("constraint_core_trim", 0),
            plot_dir=resolved_kwargs.get("plot_dir", "outputs/plots"),
            plot_every=None,
            eval_fn=None,
        )
        learner.plot_context = "posthoc"

        gammas = _hard_gammas_from_labels(labels, num_stages=num_stages)
        dummy_xis = [np.zeros((len(X) - 1, num_stages, num_stages), dtype=float) for X in dataset.demos]
        dummy_aux = [None for _ in dataset.demos]
        for gamma, xi, z in zip(gammas, dummy_xis, labels):
            for t in range(max(len(z) - 1, 0)):
                xi[t, int(z[t]), int(z[t + 1])] = 1.0

        learner.loss_loglik = []
        learner.loss_feat = []
        learner.loss_prog = []
        learner.loss_trans = []
        learner.metrics_hist = {}
        learner.loss_label = "Objective"

        upstream_history = []
        if segmentation.method_name == "cluster":
            history = getattr(segmentation.model, "objective_history_", None)
            if history is not None:
                upstream_history = [float(x) for x in history if np.isscalar(x) and np.isfinite(float(x))]
                if upstream_history:
                    learner.loss_loglik = list(upstream_history)
                    learner.loss_label = "Segmentation objective"
        elif segmentation.method_name == "arhsmm":
            history = (segmentation.extras.get("segmentation_history") or {}).get("loglik")
            if history is not None:
                upstream_history = [float(x) for x in history if np.isscalar(x) and np.isfinite(float(x))]
                if upstream_history:
                    learner.loss_loglik = list(upstream_history)
                    learner.loss_label = "Segmentation log-likelihood"

        verbose = bool(resolved_kwargs.get("verbose", True))
        learner._mstep_update_features(gammas)
        learner._mstep_update_goals(gammas, dummy_xis, dummy_aux)
        total_ll, total_feat_ll, total_prog_ll, total_trans_ll = _compute_fixed_tau_objective(
            learner, gammas, dummy_xis, dummy_aux
        )
        if not upstream_history:
            learner.loss_loglik.append(total_ll)
        learner.loss_feat.append(total_feat_ll)
        learner.loss_prog.append(total_prog_ll)
        learner.loss_trans.append(total_trans_ll)
        metrics = evaluate_model_metrics(learner, gammas, dummy_xis)
        for name, value in metrics.items():
            if np.isscalar(value):
                value_f = float(value)
                if np.isfinite(value_f):
                    learner.metrics_hist.setdefault(name, []).append(value_f)
        if verbose:
            print(
                format_training_log(
                    "POSTHOC",
                    0,
                    losses={
                        "loss": total_ll,
                        "feat": total_feat_ll,
                        "prog": total_prog_ll,
                        "trans": total_trans_ll,
                    },
                    metrics=metrics,
                    extras={"stage_ends": learner.stage_ends_, "r": learner.r.tolist()},
                )
            )

        boundary_like = [
            [int(x) for x in ends[:-1]] if num_stages > 2 else int(ends[0])
            for ends in learner.stage_ends_
        ]
        dummy_alphas = [np.zeros_like(gamma) for gamma in gammas]
        dummy_betas = [np.zeros_like(gamma) for gamma in gammas]
        plot_results_4panel(
            learner,
            boundary_like,
            1,
            gammas,
            dummy_alphas,
            dummy_betas,
            dummy_xis,
            dummy_aux,
            save_name="training_summary_posthoc_final.png",
            metrics=metrics,
        )
        return {
            "model": learner,
            "gammas": gammas,
            "metrics": metrics,
        }
