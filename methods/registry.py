from __future__ import annotations

from typing import Any

from .gmm_hmm import GMMHMMSegmenter
from .hmm_family import HMMFamilySegmenter


def build_sequential_method(method_name: str, **kwargs: Any):
    if method_name == "gmmhmm":
        return GMMHMMSegmenter(kwargs=kwargs)
    if method_name in {"hdphmm", "arhmm", "changepoint"}:
        return HMMFamilySegmenter(method=method_name, kwargs=kwargs)
    raise ValueError(
        f"Unknown sequential method '{method_name}'. "
        "Available: gmmhmm, hdphmm, arhmm, changepoint"
    )
