from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from envs.base import TaskBundle
from evaluation import evaluate_model_metrics
from ..base import SegmentationResult, format_training_log
from ..cores.map import StageWiseMAPConstraintLearningModel
from ..cores.posthoc_constraint_model import FixedTauConstraintModel
from visualization.map_plots import plot_map_results_overview
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

    for X, gamma, xi, aux in zip(learner.demos, gammas, xis_list, aux_list):
        ll_emit, ll_feat_k = learner._emission_loglik(X, return_parts=True)
        total_ll += float(np.sum(gamma * ll_emit))
        total_feat_ll += float(np.sum(gamma * ll_feat_k))

    return total_ll, total_feat_ll


def _normalize_posthoc_training_mode(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    aliases = {
        "fixed": "swcl",
        "fixed_tau": "swcl",
        "legacy": "swcl",
        "swcl": "swcl",
        "map_pooled": "pooled",
        "pooled": "pooled",
        "pooled_nll": "pooled",
        "demo_vote": "voting",
        "map_vote": "voting",
        "shared_vote": "voting",
        "vote": "voting",
        "voting": "voting",
    }
    if text not in aliases:
        raise ValueError("posthoc_training_mode must be one of: swcl, pooled, voting.")
    return aliases[text]

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

        training_mode = _normalize_posthoc_training_mode(
            resolved_kwargs.get("posthoc_training_mode", "swcl")
        )
        if training_mode == "swcl":
            learner = FixedTauConstraintModel(
                demos=dataset.demos,
                env=dataset.env,
                precomputed_features=dataset.features,
                true_taus=dataset.true_taus,
                true_cutpoints=getattr(dataset, "true_cutpoints", None),
                num_stages=num_stages,
                stage_ends_init=stage_ends,
                g2_init=None,
                fixed_feature_mask=resolved_kwargs.get("fixed_feature_mask"),
                selected_raw_feature_ids=resolved_kwargs.get("selected_raw_feature_ids"),
                feature_model_types=resolved_kwargs.get("feature_model_types"),
                constraint_core_trim=resolved_kwargs.get("constraint_core_trim", 0),
                plot_dir=resolved_kwargs.get("plot_dir", "outputs/plots"),
                plot_every=None,
                eval_fn=None,
            )
            gammas = _hard_gammas_from_labels(labels, num_stages=num_stages)
        else:
            learner = StageWiseMAPConstraintLearningModel(
                demos=dataset.demos,
                env=dataset.env,
                precomputed_features=dataset.features,
                true_taus=dataset.true_taus,
                true_cutpoints=getattr(dataset, "true_cutpoints", None),
                n_stages=num_stages,
                seed=resolved_kwargs.get("seed", 0),
                selected_raw_feature_ids=resolved_kwargs.get("selected_raw_feature_ids"),
                force_inactive_feature_ids=resolved_kwargs.get("force_inactive_feature_ids"),
                map_progress_kappa=0.0,
                duration_min=1,
                constraint_core_trim=resolved_kwargs.get("constraint_core_trim", 0),
                short_segment_penalty_c=0.0,
                truncated_z_soft_boundary_scale=resolved_kwargs.get("truncated_z_soft_boundary_scale", 0.1),
                truncated_z_observation_noise_scale=resolved_kwargs.get("truncated_z_observation_noise_scale", 0.003),
                truncated_z_half_t_scale_quantile=resolved_kwargs.get("truncated_z_half_t_scale_quantile", 0.9),
                plot_every=None,
                plot_dir=resolved_kwargs.get("plot_dir", "outputs/plots"),
                disable_plots=True,
                eval_fn=None,
                verbose=False,
                map_eq_sigma=resolved_kwargs.get("map_eq_sigma", 0.05),
                map_c_bg=resolved_kwargs.get("map_c_bg", 2.0),
                map_c_ineq=resolved_kwargs.get("map_c_ineq", 0.0),
                map_eq_distribution=resolved_kwargs.get("map_eq_distribution", "gaussian"),
                map_inactive_distribution=resolved_kwargs.get("map_inactive_distribution", "gaussian"),
                map_nu_eq=resolved_kwargs.get("map_nu_eq", 3.0),
                map_nu_inactive=resolved_kwargs.get("map_nu_inactive", 3.0),
                map_nu_ineq=resolved_kwargs.get("map_nu_ineq", 3.0),
                map_boundary_quantile=resolved_kwargs.get("map_boundary_quantile", 0.05),
                map_activation_prior=resolved_kwargs.get("map_activation_prior"),
                map_active_mode_prior=resolved_kwargs.get("map_active_mode_prior"),
                map_mode_aggregation="pooled" if training_mode == "pooled" else "shared_vote",
                map_vote_prior_scope=resolved_kwargs.get("map_vote_prior_scope", "shared"),
                map_refit_winning_voters=resolved_kwargs.get("map_refit_winning_voters", False),
                map_convergence_tol=0.0,
                map_demo_num_workers=1,
                map_mstep_boundary_trim=resolved_kwargs.get("map_mstep_boundary_trim", 0),
            )
            gammas = learner.fit_fixed_segments(stage_ends, verbose=False)

        learner.posthoc_training_mode = training_mode
        learner.plot_context = f"posthoc_{training_mode}"
        dummy_xis = [np.zeros((len(X) - 1, num_stages, num_stages), dtype=float) for X in dataset.demos]
        dummy_aux = [None for _ in dataset.demos]
        for gamma, xi in zip(gammas, dummy_xis):
            z = np.argmax(gamma, axis=1).astype(int)
            for t in range(max(len(z) - 1, 0)):
                xi[t, int(z[t]), int(z[t + 1])] = 1.0

        learner.loss_loglik = []
        learner.loss_feat = []
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
        elif segmentation.method_name == "changeforest":
            history = getattr(segmentation.model, "objective_history_", None)
            if history is not None:
                upstream_history = [float(x) for x in history if np.isscalar(x) and np.isfinite(float(x))]
                if upstream_history:
                    learner.loss_loglik = list(upstream_history)
                    learner.loss_label = "Changeforest split gain"
        elif segmentation.method_name == "arhsmm":
            history = (segmentation.extras.get("segmentation_history") or {}).get("loglik")
            if history is not None:
                upstream_history = [float(x) for x in history if np.isscalar(x) and np.isfinite(float(x))]
                if upstream_history:
                    learner.loss_loglik = list(upstream_history)
                    learner.loss_label = "Segmentation log-likelihood"

        verbose = bool(resolved_kwargs.get("verbose", True))
        if training_mode == "swcl":
            learner._mstep_update_features(gammas)
            learner._mstep_update_goals(gammas, dummy_xis, dummy_aux)
            total_ll, total_feat_ll = _compute_fixed_tau_objective(
                learner, gammas, dummy_xis, dummy_aux
            )
        else:
            total_ll = -float(learner.loss_total[-1])
            total_feat_ll = -float(learner.loss_constraint[-1])
        learner.posthoc_total_objective_ = float(total_ll)
        learner.posthoc_feature_objective_ = float(total_feat_ll)
        if not upstream_history:
            learner.loss_loglik.append(total_ll)
        learner.loss_feat.append(total_feat_ll)
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
                    },
                    metrics=metrics,
                    extras={
                        "mode": training_mode,
                        "stage_ends": learner.stage_ends_,
                        "r": learner.r.tolist(),
                    },
                )
            )

        boundary_like = [
            [int(x) for x in ends[:-1]] if num_stages > 2 else int(ends[0])
            for ends in learner.stage_ends_
        ]
        if not bool(resolved_kwargs.get("disable_plots", False)):
            if training_mode == "swcl":
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
            else:
                plot_map_results_overview(
                    learner,
                    0,
                    metrics=metrics,
                    save_name="training_summary_posthoc_final.png",
                )
        return {
            "model": learner,
            "gammas": gammas,
            "metrics": metrics,
            "posthoc_training_mode": training_mode,
        }
