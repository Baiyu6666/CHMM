from __future__ import annotations

from importlib import import_module

from .base import TaskBundle
from .registry import ENV_REGISTRY, load_env


_LAZY_EXPORTS = {
    "BarCleanEnv": (".BarClean", "BarCleanEnv"),
    "load_BarClean": (".BarClean", "load_BarClean"),
    "BarInspectEnv": (".BarInspect", "BarInspectEnv"),
    "BarInspectScene": (".BarInspect", "BarInspectScene"),
    "load_BarInspect": (".BarInspect", "load_BarInspect"),
    "S3ObsAvoidEnv": (".S3ObsAvoid", "S3ObsAvoidEnv"),
    "load_S3ObsAvoid": (".S3ObsAvoid", "load_S3ObsAvoid"),
    "S3ObsAvoidRealEnv": (".S3ObsAvoidReal", "S3ObsAvoidRealEnv"),
    "load_S3ObsAvoidReal": (".S3ObsAvoidReal", "load_S3ObsAvoidReal"),
    "S4SlideInsertEnv": (".S4SlideInsert", "S4SlideInsertEnv"),
    "load_S4SlideInsert": (".S4SlideInsert", "load_S4SlideInsert"),
    "S5SphereInspectEnv": (".S5SphereInspect", "S5SphereInspectEnv"),
    "load_S5SphereInspect": (".S5SphereInspect", "load_S5SphereInspect"),
}


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, package=__name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(name)


__all__ = [
    "BarCleanEnv",
    "BarInspectEnv",
    "BarInspectScene",
    "ENV_REGISTRY",
    "S3ObsAvoidEnv",
    "S3ObsAvoidRealEnv",
    "S4SlideInsertEnv",
    "S5SphereInspectEnv",
    "TaskBundle",
    "load_BarClean",
    "load_BarInspect",
    "load_S3ObsAvoid",
    "load_S3ObsAvoidReal",
    "load_S4SlideInsert",
    "load_S5SphereInspect",
    "load_env",
]
