from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from envs.base import TaskBundle
from .gmm_hmm_core import GMMHMM

from .base import SegmentationResult, labels_to_cutpoints


@dataclass
class GMMHMMSegmenter:
    kwargs: Dict[str, Any]

    def run(self, dataset: TaskBundle) -> SegmentationResult:
        if dataset.env is None:
            raise ValueError("GMM-HMM segmenter requires a dataset env with feature API.")
        max_iter = self.kwargs.get("max_iter", 30)
        plot_every = self.kwargs.get("plot_every")
        if plot_every is None:
            plot_every = max_iter

        model = GMMHMM(
            demos=dataset.demos,
            env=dataset.env,
            true_taus=dataset.true_taus,
            gmm_K=self.kwargs.get("gmm_K", 3),
            gmm_reg=self.kwargs.get("gmm_reg", 1e-6),
            use_xy_vel=self.kwargs.get("use_xy_vel", False),
            standardize_feat=self.kwargs.get("standardize_feat", True),
            plot_every=plot_every,
            plot_dir=self.kwargs.get("plot_dir", "outputs/plots"),
        )
        posts = model.fit(
            max_iter=max_iter,
            verbose=self.kwargs.get("verbose", True),
        )
        labels = [np.argmax(gamma, axis=1).astype(int) for gamma in posts]
        taus = []
        for gamma in posts:
            idx = np.where(gamma[:, 1] > 0.5)[0]
            tau_hat = int(idx[0]) if len(idx) > 0 else int(np.argmax(gamma[:, 1]))
            taus.append(tau_hat)

        return SegmentationResult(
            method_name="gmmhmm",
            labels=labels,
            cutpoints=labels_to_cutpoints(labels),
            taus=taus,
            model=model,
            extras={"gammas": posts},
        )
