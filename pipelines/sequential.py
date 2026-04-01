from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from envs.base import TaskBundle
from methods import PostHocConstraintLearner, build_sequential_method
from evaluation import evaluate_model_metrics


@dataclass
class SequentialPipeline:
    segmenter_name: str
    segmenter_kwargs: Dict[str, Any]
    constraint_kwargs: Dict[str, Any]

    def run(self, dataset: TaskBundle) -> Dict[str, Any]:
        segmenter = build_sequential_method(self.segmenter_name, **self.segmenter_kwargs)
        segmentation = segmenter.run(dataset)
        if self.segmenter_name == "fchmm":
            model = segmentation.model
            gammas = segmentation.extras.get("gammas", None) if segmentation.extras else None
            if model is None or gammas is None:
                raise ValueError("fchmm sequential pipeline expects the segmenter to return a trained model and gammas.")
            metrics = evaluate_model_metrics(model, gammas, None)
            constraints = {
                "model": model,
                "gammas": gammas,
                "metrics": metrics,
            }
            return {
                "pipeline": "sequential",
                "dataset": dataset,
                "segmentation": segmentation,
                "constraints": constraints,
            }
        constraints = PostHocConstraintLearner(self.constraint_kwargs).fit(dataset, segmentation)
        return {
            "pipeline": "sequential",
            "dataset": dataset,
            "segmentation": segmentation,
            "constraints": constraints,
        }
