from __future__ import annotations

from typing import Any

from .wrappers.joint_ccp import JointCCPMethod
from .wrappers.joint_segcons import JointSegConsMethod
from .wrappers.sequential_baseline import SequentialBaselineSegmenter

JOINT_METHODS = frozenset({"ccp", "segcons"})
SEQUENTIAL_METHODS = frozenset({"cghmm", "arhmm", "changepoint"})
ALL_METHODS = tuple(sorted(JOINT_METHODS | SEQUENTIAL_METHODS))


def method_pipeline_kind(method_name: str) -> str:
    if method_name in JOINT_METHODS:
        return "joint"
    if method_name in SEQUENTIAL_METHODS:
        return "sequential"
    raise ValueError(
        f"Unknown method '{method_name}'. "
        f"Available: {', '.join(ALL_METHODS)}"
    )


def build_sequential_method(method_name: str, **kwargs: Any):
    if method_name in SEQUENTIAL_METHODS:
        return SequentialBaselineSegmenter(method=method_name, kwargs=kwargs)
    raise ValueError(
        f"Unknown sequential method '{method_name}'. "
        f"Available: {', '.join(sorted(SEQUENTIAL_METHODS))}"
    )


def build_joint_method(method_name: str, **kwargs: Any):
    if method_name == "segcons":
        return JointSegConsMethod(kwargs=kwargs)
    if method_name == "ccp":
        return JointCCPMethod(kwargs=kwargs)
    raise ValueError(
        f"Unknown joint method '{method_name}'. "
        f"Available: {', '.join(sorted(JOINT_METHODS))}"
    )
