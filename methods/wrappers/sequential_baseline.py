from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from envs.base import TaskBundle

from ..backends.changepoint import segment_changepoint
from ..backends.hmm import segment_with_hmm
from ..base import SegmentationResult, labels_to_cutpoints, labels_to_taus
from ..common.tau_init import extract_taus_hat
from ..cores.cghmm_core import CGHMM

@dataclass
class SequentialBaselineSegmenter:
    method: str
    kwargs: Dict[str, Any]

    def run(self, dataset: TaskBundle) -> SegmentationResult:
        taus = None
        if self.method == "cghmm":
            if dataset.env is None:
                raise ValueError("CG-HMM segmenter requires a dataset env with feature API.")
            resolved_kwargs = dict(self.kwargs)
            max_iter = self.kwargs.get("max_iter", 30)
            plot_every = self.kwargs.get("plot_every")
            if plot_every is None:
                plot_every = max_iter

            model = CGHMM(
                demos=dataset.demos,
                env=dataset.env,
                true_taus=dataset.true_taus,
                tau_init=resolved_kwargs.get("tau_init"),
                tau_init_mode=resolved_kwargs.get("tau_init_mode", "uniform_taus"),
                seed=resolved_kwargs.get("seed", 0),
                gmm_K=resolved_kwargs.get("gmm_K", 3),
                gmm_reg=resolved_kwargs.get("gmm_reg", 1e-6),
                use_xy_vel=resolved_kwargs.get("use_xy_vel", False),
                standardize_x=resolved_kwargs.get("standardize_x", True),
                standardize_feat=resolved_kwargs.get("standardize_feat", True),
                selected_raw_feature_ids=resolved_kwargs.get("selected_raw_feature_ids"),
                feature_model_types=resolved_kwargs.get("feature_model_types"),
                fixed_feature_mask=resolved_kwargs.get("fixed_feature_mask"),
                plot_every=plot_every,
                plot_dir=resolved_kwargs.get("plot_dir", "outputs/plots"),
            )
            posts = model.fit(
                max_iter=max_iter,
                verbose=resolved_kwargs.get("verbose", True),
            )
            labels = [np.argmax(gamma, axis=1).astype(int) for gamma in posts]
            taus = extract_taus_hat(posts)
            seg_hist = {}
            extras = {"gammas": posts}
        elif self.method == "arhmm":
            resolved_kwargs = dict(self.kwargs)
            hmm_method = "ar"
            labels, model, seg_hist = segment_with_hmm(
                dataset.demos,
                env=dataset.env,
                true_taus=dataset.true_taus,
                method=hmm_method,
                n_states=resolved_kwargs.get("n_states", 2),
                sticky=resolved_kwargs.get("sticky", 10.0),
                n_iter=resolved_kwargs.get("max_iter", 30),
                verbose=resolved_kwargs.get("verbose", True),
                seed=resolved_kwargs.get("seed", 0),
                use_velocity=resolved_kwargs.get("use_velocity", False),
                vel_weight=resolved_kwargs.get("vel_weight", 1.0),
                standardize=resolved_kwargs.get("standardize", True),
                use_env_features=resolved_kwargs.get("use_env_features", True),
                selected_raw_feature_ids=resolved_kwargs.get("selected_raw_feature_ids"),
                tau_init=resolved_kwargs.get("tau_init"),
                tau_init_mode=resolved_kwargs.get("tau_init_mode", "uniform_taus"),
                left_right=resolved_kwargs.get("left_right", True),
            )
            extras = {"segmentation_history": seg_hist}
        elif self.method == "changepoint":
            resolved_kwargs = dict(self.kwargs)
            labels = segment_changepoint(
                dataset.demos,
                env=dataset.env,
                K=resolved_kwargs.get("n_states", 2),
                cost_type=resolved_kwargs.get("cost_type", "gaussian"),
                use_velocity=resolved_kwargs.get("use_velocity", True),
                vel_weight=resolved_kwargs.get("vel_weight", 1.0),
                standardize=resolved_kwargs.get("standardize", True),
                use_env_features=resolved_kwargs.get("use_env_features", True),
                selected_raw_feature_ids=resolved_kwargs.get("selected_raw_feature_ids"),
                min_len=resolved_kwargs.get("min_len", 1),
            )
            model = None
            seg_hist = {}
            extras = {"segmentation_history": seg_hist}
        else:
            raise ValueError(f"Unsupported segmenter '{self.method}'.")

        labels = [np.asarray(z, dtype=int) for z in labels]
        return SegmentationResult(
            method_name=self.method,
            labels=labels,
            cutpoints=labels_to_cutpoints(labels),
            taus=taus if taus is not None else labels_to_taus(labels),
            model=model,
            extras=extras,
        )
