from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class TaskBundle:
    name: str
    demos: List[np.ndarray]
    env: Optional[Any] = None
    true_taus: Optional[List[int]] = None
    true_labels: Optional[List[np.ndarray]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
