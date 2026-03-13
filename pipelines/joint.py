from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from envs.base import TaskBundle
from methods import JointSegConsMethod


@dataclass
class JointPipeline:
    kwargs: Dict[str, Any]

    def run(self, dataset: TaskBundle) -> Dict[str, Any]:
        method = JointSegConsMethod(kwargs=self.kwargs)
        fitted = method.fit(dataset)
        return {
            "pipeline": "joint",
            "dataset": dataset,
            "joint_result": fitted,
        }
