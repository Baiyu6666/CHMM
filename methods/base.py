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


def compute_tau_metrics(
    taus_hat: List[int] | np.ndarray | None,
    true_taus: List[int] | np.ndarray | None,
    demos: List[np.ndarray],
) -> Dict[str, float]:
    if taus_hat is None or true_taus is None:
        return {}

    mae_list: List[float] = []
    nmae_list: List[float] = []
    for tau_hat, tau_true, X in zip(taus_hat, true_taus, demos):
        if tau_true is None:
            continue
        err = abs(int(tau_hat) - int(tau_true))
        mae_list.append(float(err))
        nmae_list.append(float(err / max(len(X), 1)))

    metrics: Dict[str, float] = {}
    if mae_list:
        metrics["MAE_tau"] = float(np.mean(mae_list))
        metrics["NMAE_tau"] = float(np.mean(nmae_list))
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
            if value is None or not np.isfinite(value):
                continue
            parts.append(f"{name}={float(value):.3f}")

    if extras:
        for name, value in extras.items():
            parts.append(f"{name}={value}")

    return " | ".join(parts)
