from __future__ import annotations

from importlib import import_module
from typing import Any, Callable, Dict

import numpy as np

from .base import TaskBundle


class _LazyEnvLoader:
    def __init__(self, module_name: str, function_name: str):
        self.module_name = str(module_name)
        self.function_name = str(function_name)
        self._loader = None

    def __call__(self, **kwargs: Any) -> TaskBundle:
        if self._loader is None:
            module = import_module(self.module_name, package=__package__)
            self._loader = getattr(module, self.function_name)
        return self._loader(**kwargs)

ENV_REGISTRY: Dict[str, Callable[..., TaskBundle]] = {
    "S3ObsAvoid": _LazyEnvLoader(".S3ObsAvoid", "load_S3ObsAvoid"),
    "S3ObsAvoidReal": _LazyEnvLoader(".S3ObsAvoidReal", "load_S3ObsAvoidReal"),
    "S5SphereInspect": _LazyEnvLoader(".S5SphereInspect", "load_S5SphereInspect"),
    "S4SlideInsert": _LazyEnvLoader(".S4SlideInsert", "load_S4SlideInsert"),
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

        if bundle.features is not None:
            if len(bundle.features) != len(bundle.demos):
                raise ValueError(
                    f"Dataset '{bundle.name}' contains {len(bundle.features)} feature matrices "
                    f"for {len(bundle.demos)} demos."
                )
            validated_features = []
            frozen_feature_dim = None
            for demo_idx, (demo, features) in enumerate(zip(bundle.demos, bundle.features)):
                matrix = np.asarray(features, dtype=float)
                if matrix.ndim != 2 or len(matrix) != len(demo):
                    raise ValueError(
                        f"Dataset '{bundle.name}' feature matrix {demo_idx} must be 2D and "
                        f"aligned with its {len(demo)}-sample demo."
                    )
                if not np.all(np.isfinite(matrix)):
                    raise ValueError(
                        f"Dataset '{bundle.name}' feature matrix {demo_idx} contains non-finite values."
                    )
                if frozen_feature_dim is None:
                    frozen_feature_dim = int(matrix.shape[1])
                elif int(matrix.shape[1]) != frozen_feature_dim:
                    raise ValueError(
                        f"Dataset '{bundle.name}' feature matrix {demo_idx} has {matrix.shape[1]} columns; "
                        f"expected {frozen_feature_dim}."
                    )
                validated_features.append(matrix)
            bundle.features = validated_features
            f0 = validated_features[0]
        else:
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

    if bundle.true_cutpoints is None:
        if bundle.true_labels is not None:
            bundle.true_cutpoints = [
                np.where(np.diff(np.asarray(z, dtype=int)) != 0)[0].astype(int)
                for z in bundle.true_labels
            ]
        elif bundle.true_taus is not None:
            bundle.true_cutpoints = [
                None if tau is None else np.asarray([int(tau)], dtype=int)
                for tau in bundle.true_taus
            ]

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
