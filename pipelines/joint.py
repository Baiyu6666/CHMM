from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from envs.base import TaskBundle
from methods import build_joint_method


@dataclass
class JointPipeline:
    method_name: str
    kwargs: Dict[str, Any]

    def run(self, dataset: TaskBundle) -> Dict[str, Any]:
        method = build_joint_method(self.method_name, **self.kwargs)
        fitted = method.fit(dataset)
        return {
            "pipeline": "joint",
            "dataset": dataset,
            "joint_result": fitted,
        }
