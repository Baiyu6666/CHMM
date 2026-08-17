from .s5.config import (
    S5_METRIC_SCALE,
    apply_default_s5_loader_config as _resolve_active_s5_loader_config,
)
from .s5.constants import S5_DEMO_CACHE_VERSION, S5_FEATURE_EXTRACTOR_VERSION
from .s5.dataset import load_S5SphereInspect
from .s5.execution import check_s5_reference_waypoints_ik, simulate_s5_demo_from_reference
from .s5.task import S5SphereInspectEnv

_S5_METRIC_SCALE = S5_METRIC_SCALE
_S5_DEMO_CACHE_VERSION = S5_DEMO_CACHE_VERSION
_S5_FEATURE_EXTRACTOR_VERSION = S5_FEATURE_EXTRACTOR_VERSION


def _apply_default_s5_loader_config(env_cfg):
    return _resolve_active_s5_loader_config(env_cfg)


apply_default_s5_loader_config = _apply_default_s5_loader_config

__all__ = [
    "S5SphereInspectEnv",
    "_apply_default_s5_loader_config",
    "apply_default_s5_loader_config",
    "check_s5_reference_waypoints_ik",
    "load_S5SphereInspect",
    "simulate_s5_demo_from_reference",
]
