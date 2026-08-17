from .config import (
    S5DatasetConfig,
    S5ExecutionConfig,
    S5GeometryConfig,
    S5NoiseConfig,
    S5OrientationConfig,
    S5PathConfig,
    S5SyntheticPreset,
    S5TimingConfig,
    S5_SYNTHETIC_V20,
    S5_SYNTHETIC_V21,
    S5_SYNTHETIC_V22,
    S5_SYNTHETIC_V23,
    active_s5_env_kwargs,
    apply_default_s5_loader_config,
)
from .time_parameterization import (
    FixedStepTimeParameterizer,
    TimeParameterizedPath,
    concatenate_stage_timestamps,
    gaussian_slowdown_weights,
)

__all__ = [
    "S5DatasetConfig",
    "S5ExecutionConfig",
    "S5GeometryConfig",
    "S5NoiseConfig",
    "S5OrientationConfig",
    "S5PathConfig",
    "S5SphereInspectEnv",
    "S5SyntheticPreset",
    "S5TimingConfig",
    "S5_SYNTHETIC_V20",
    "S5_SYNTHETIC_V21",
    "S5_SYNTHETIC_V22",
    "S5_SYNTHETIC_V23",
    "FixedStepTimeParameterizer",
    "TimeParameterizedPath",
    "active_s5_env_kwargs",
    "apply_default_s5_loader_config",
    "concatenate_stage_timestamps",
    "gaussian_slowdown_weights",
    "load_S5SphereInspect",
]


def __getattr__(name):
    if name == "S5SphereInspectEnv":
        from .task import S5SphereInspectEnv

        return S5SphereInspectEnv
    if name == "load_S5SphereInspect":
        from .dataset import load_S5SphereInspect

        return load_S5SphereInspect
    raise AttributeError(name)
