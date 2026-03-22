from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


@dataclass
class SegmentationResult:
    method_name: str
    labels: List[np.ndarray]
    cutpoints: List[np.ndarray]
    taus: Optional[List[int]] = None
    model: Any | None = None
    extras: Dict[str, Any] = field(default_factory=dict)


def labels_to_cutpoints(labels: List[np.ndarray]) -> List[np.ndarray]:
    return [np.where(np.diff(z.astype(int)) != 0)[0] for z in labels]


def labels_to_taus(labels: List[np.ndarray]) -> Optional[List[int]]:
    taus: List[int] = []
    for z in labels:
        cuts = np.where(np.diff(z.astype(int)) != 0)[0]
        if len(cuts) != 1:
            return None
        taus.append(int(cuts[0]))
    return taus


def compute_cutpoint_metrics(
    cutpoints_hat: List[np.ndarray] | List[List[int]] | None,
    true_cutpoints: List[np.ndarray] | List[List[int]] | None,
    demos: List[np.ndarray],
) -> Dict[str, float]:
    if cutpoints_hat is None or true_cutpoints is None:
        return {}

    mae_list: List[float] = []
    exact_match: List[float] = []
    for pred, true, X in zip(cutpoints_hat, true_cutpoints, demos):
        if true is None:
            continue
        pred_arr = np.asarray(pred, dtype=int).reshape(-1)
        true_arr = np.asarray(true, dtype=int).reshape(-1)
        if pred_arr.size != true_arr.size:
            exact_match.append(0.0)
            continue
        errs = np.abs(pred_arr - true_arr)
        if errs.size > 0:
            mae_list.append(float(np.mean(errs)))
        else:
            mae_list.append(0.0)
        exact_match.append(float(np.all(errs == 0)))

    metrics: Dict[str, float] = {}
    if mae_list:
        metrics["MeanAbsCutpointError"] = float(np.mean(mae_list))
    if exact_match:
        metrics["CutpointExactMatchRate"] = float(np.mean(exact_match))
    return metrics


def format_training_log(
    method_name: str,
    iteration: int,
    *,
    losses: Mapping[str, float] | None = None,
    metrics: Mapping[str, float] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> str:
    parts = [f"[{method_name}] iter {int(iteration) + 1:03d}"]

    if losses:
        for name, value in losses.items():
            if value is None:
                continue
            parts.append(f"{name}={float(value):.3f}")

    if metrics:
        for name in sorted(metrics.keys()):
            value = metrics[name]
            if value is None:
                continue
            if np.isscalar(value):
                value_f = float(value)
                if np.isfinite(value_f):
                    parts.append(f"{name}={value_f:.3f}")

    if extras:
        for name, value in extras.items():
            parts.append(f"{name}={value}")

    return " | ".join(parts)
