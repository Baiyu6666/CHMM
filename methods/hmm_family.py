from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from envs.base import TaskBundle

from .base import SegmentationResult, labels_to_cutpoints, labels_to_taus
from .changepoint_backend import segment_changepoint
from .hmm_backend import segment_with_hmm


@dataclass
class HMMFamilySegmenter:
    method: str
    kwargs: Dict[str, Any]

    def run(self, dataset: TaskBundle) -> SegmentationResult:
        if self.method in {"hdphmm", "arhmm"}:
            hmm_method = "hdp" if self.method == "hdphmm" else "ar"
            labels, model, _, _, _, _, seg_hist = segment_with_hmm(
                dataset.demos,
                method=hmm_method,
                n_states=self.kwargs.get("n_states", 2),
                sticky=self.kwargs.get("sticky", 10.0),
                n_iter=self.kwargs.get("max_iter", 30),
                verbose=self.kwargs.get("verbose", True),
                seed=self.kwargs.get("seed", 0),
                use_velocity=self.kwargs.get("use_velocity", False),
                vel_weight=self.kwargs.get("vel_weight", 1.0),
                standardize=self.kwargs.get("standardize", False),
                left_right=self.kwargs.get("left_right", True),
            )
        elif self.method == "changepoint":
            labels, _, _, _, _ = segment_changepoint(
                dataset.demos,
                K=self.kwargs.get("n_states", 2),
                cost_type=self.kwargs.get("cost_type", "gaussian"),
                use_velocity=self.kwargs.get("use_velocity", True),
                vel_weight=self.kwargs.get("vel_weight", 1.0),
                standardize=self.kwargs.get("standardize", True),
                min_len=self.kwargs.get("min_len", 1),
            )
            model = None
            seg_hist = {}
        else:
            raise ValueError(f"Unsupported segmenter '{self.method}'.")

        labels = [np.asarray(z, dtype=int) for z in labels]
        return SegmentationResult(
            method_name=self.method,
            labels=labels,
            cutpoints=labels_to_cutpoints(labels),
            taus=labels_to_taus(labels),
            model=model,
            extras={"segmentation_history": seg_hist},
        )
