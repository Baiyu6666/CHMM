from __future__ import annotations

from typing import Any, Callable, Dict

from .base import TaskBundle
from .line_2d import load_line_2d
from .obs_avoid_2d import load_2d_obs_avoid
from .obs_avoid_3d import load_3d_obs_avoid
from .pick_place import load_pick_place
from .sine_corridor_3d import load_3d_sine_corridor


ENV_REGISTRY: Dict[str, Callable[..., TaskBundle]] = {
    "2DObsAvoid": load_2d_obs_avoid,
    "3DObsAvoid": load_3d_obs_avoid,
    "3DSineCorridor": load_3d_sine_corridor,
    "2Dline": load_line_2d,
    "PickPlace": load_pick_place,
}


def load_env(name: str, **kwargs: Any) -> TaskBundle:
    try:
        loader = ENV_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown environment '{name}'. Available: {sorted(ENV_REGISTRY)}") from exc
    return loader(**kwargs)
