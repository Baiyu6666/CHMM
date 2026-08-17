from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def resolve_feature_matrices(
    demos: Sequence[np.ndarray],
    env,
    precomputed_features: Sequence[np.ndarray] | None = None,
) -> list[np.ndarray]:
    if precomputed_features is None:
        return [np.asarray(env.compute_all_features_matrix(demo), dtype=float) for demo in demos]

    if len(precomputed_features) != len(demos):
        raise ValueError(
            "precomputed_features must contain one matrix per demonstration: "
            f"got {len(precomputed_features)} for {len(demos)} demos."
        )

    resolved = []
    feature_dim = None
    for demo_idx, (demo, features) in enumerate(zip(demos, precomputed_features)):
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"precomputed feature matrix {demo_idx} must be two-dimensional.")
        if len(matrix) != len(demo):
            raise ValueError(
                f"precomputed feature matrix {demo_idx} has {len(matrix)} rows for a {len(demo)}-sample demo."
            )
        if feature_dim is None:
            feature_dim = int(matrix.shape[1])
        elif int(matrix.shape[1]) != feature_dim:
            raise ValueError("all precomputed feature matrices must have the same number of columns.")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"precomputed feature matrix {demo_idx} contains non-finite values.")
        resolved.append(matrix.copy())
    return resolved
