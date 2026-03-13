from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

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
