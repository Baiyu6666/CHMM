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
    "2DLine": load_line_2d,
    "PickPlace": load_pick_place,
}


def _validate_task_bundle(bundle: TaskBundle) -> TaskBundle:
    if bundle.env is None or not bundle.demos:
        return bundle

    schema = bundle.feature_schema
    if schema is None and hasattr(bundle.env, "get_feature_schema"):
        schema = bundle.env.get_feature_schema()

    if schema is not None:
        ids = [int(spec.get("id", i)) for i, spec in enumerate(schema)]
        column_indices = [int(spec.get("column_idx", i)) for i, spec in enumerate(schema)]
        names = [str(spec.get("name", f"f{i}")) for i, spec in enumerate(schema)]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate feature ids in dataset '{bundle.name}': {ids}")
        if len(column_indices) != len(set(column_indices)):
            raise ValueError(
                f"Duplicate feature column indices in dataset '{bundle.name}': {column_indices}"
            )
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate feature names in dataset '{bundle.name}': {names}")

        f0 = bundle.env.compute_all_features_matrix(bundle.demos[0])
        num_cols = int(f0.shape[1])
        if len(schema) != num_cols:
            raise ValueError(
                f"Dataset '{bundle.name}' feature_schema length {len(schema)} "
                f"does not match feature matrix columns {num_cols}."
            )
        if sorted(column_indices) != list(range(num_cols)):
            raise ValueError(
                f"Dataset '{bundle.name}' feature_schema column_idx values must form 0..{num_cols - 1}, "
                f"got {column_indices}."
            )

    if bundle.true_labels is not None:
        for i, (X, z) in enumerate(zip(bundle.demos, bundle.true_labels)):
            if len(X) != len(z):
                raise ValueError(
                    f"Dataset '{bundle.name}' demo/label length mismatch at demo {i}: "
                    f"len(demo)={len(X)}, len(labels)={len(z)}"
                )

    if bundle.true_constraints is None:
        true_constraints = getattr(bundle.env, "true_constraints", None)
        if true_constraints is not None:
            bundle.true_constraints = dict(true_constraints)

    if bundle.constraint_specs is None:
        constraint_specs = getattr(bundle.env, "constraint_specs", None)
        if constraint_specs is not None:
            bundle.constraint_specs = list(constraint_specs)

    return bundle


def load_env(name: str, **kwargs: Any) -> TaskBundle:
    try:
        loader = ENV_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown environment '{name}'. Available: {sorted(ENV_REGISTRY)}") from exc
    return _validate_task_bundle(loader(**kwargs))
