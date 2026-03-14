from __future__ import annotations

import numpy as np


def clip_tau_for_sequence(x, tau):
    return int(np.clip(int(tau), 1, len(x) - 2))


def extract_tau_hat(gamma, xi=None):
    gamma = np.asarray(gamma, dtype=float)
    if xi is not None:
        xi = np.asarray(xi, dtype=float)
        if xi.ndim == 3 and xi.shape[0] > 0:
            xi01 = xi[:, 0, 1]
            if np.all(np.isfinite(xi01)) and float(np.sum(xi01)) > 1e-12:
                return int(np.argmax(xi01))

    idx = np.where(gamma[:, 1] > 0.5)[0]
    if len(idx) > 0:
        return int(idx[0])
    return int(np.argmax(gamma[:, 1]))


def extract_taus_hat(gammas, xis_list=None):
    if xis_list is None:
        xis_list = [None] * len(gammas)
    return [extract_tau_hat(gamma, xi) for gamma, xi in zip(gammas, xis_list)]


def _uniform_taus(X_list):
    taus = []
    for x in X_list:
        t = int(round(0.5 * (len(x) - 1)))
        taus.append(clip_tau_for_sequence(x, t))
    return np.asarray(taus, dtype=int)


def _random_taus(X_list, rng):
    lam = float(np.clip(rng.rand(), 0.1, 0.9))
    taus = []
    for x in X_list:
        t = int(round(lam * (len(x) - 1)))
        taus.append(clip_tau_for_sequence(x, t))
    return np.asarray(taus, dtype=int)


def _changepoint_warmstart_taus(
    X_list,
    env=None,
    use_velocity=False,
    vel_weight=1.0,
    standardize=False,
    use_env_features=True,
    selected_raw_feature_ids=None,
):
    from ..backends.changepoint import segment_fixed_K_CP

    taus = []
    for x in X_list:
        cps, _ = segment_fixed_K_CP(
            x,
            K=2,
            use_velocity=use_velocity,
            vel_weight=vel_weight,
            standardize=standardize,
            env=env,
            use_env_features=use_env_features,
            selected_raw_feature_ids=selected_raw_feature_ids,
        )
        if len(cps) < 1:
            t = len(x) // 2
        else:
            t = int(cps[0])
        taus.append(clip_tau_for_sequence(x, t))
    return np.asarray(taus, dtype=int)


def resolve_tau_init_for_demos(
    X_list,
    tau_init=None,
    tau_init_mode="uniform_taus",
    env=None,
    seed: int = 0,
    use_velocity=False,
    vel_weight=1.0,
    standardize=False,
    use_env_features=True,
    selected_raw_feature_ids=None,
):
    if tau_init is not None:
        tau_init = np.asarray(tau_init, dtype=int)
        if len(tau_init) != len(X_list):
            raise ValueError("tau_init must match the number of demos.")
        return np.asarray(
            [clip_tau_for_sequence(x, t) for x, t in zip(X_list, tau_init)],
            dtype=int,
        )

    mode = str(tau_init_mode).lower()
    rng = np.random.RandomState(int(seed))
    if mode == "uniform_taus":
        return _uniform_taus(X_list)
    if mode == "random_taus":
        return _random_taus(X_list, rng)
    if mode == "changepoint_warmstart":
        return _changepoint_warmstart_taus(
            X_list,
            env=env,
            use_velocity=use_velocity,
            vel_weight=vel_weight,
            standardize=standardize,
            use_env_features=use_env_features,
            selected_raw_feature_ids=selected_raw_feature_ids,
        )
    raise ValueError(
        "tau_init_mode must be one of {'uniform_taus', 'random_taus', 'changepoint_warmstart'}."
    )
