from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from envs.base import TaskBundle
from methods import PostHocConstraintLearner, build_sequential_method


@dataclass
class SequentialPipeline:
    segmenter_name: str
    segmenter_kwargs: Dict[str, Any]
    constraint_kwargs: Dict[str, Any]

    def run(self, dataset: TaskBundle) -> Dict[str, Any]:
        segmenter = build_sequential_method(self.segmenter_name, **self.segmenter_kwargs)
        segmentation = segmenter.run(dataset)
        constraints = PostHocConstraintLearner(self.constraint_kwargs).fit(dataset, segmentation)
        return {
            "pipeline": "sequential",
            "dataset": dataset,
            "segmentation": segmentation,
            "constraints": constraints,
        }
