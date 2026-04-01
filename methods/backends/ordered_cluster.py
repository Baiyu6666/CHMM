from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

from .changepoint import _resolve_selected_feature_columns


@dataclass
class OrderedClusterModel:
    num_stages: int
    centers_: np.ndarray
    stage_ends_: List[List[int]]
    feature_mean_: np.ndarray
    feature_std_: np.ndarray
    block_slices_: List[List[int]]
    block_weights_: np.ndarray
    objective_history_: List[float] = field(default_factory=list)
    segmentation_history_: List[List[List[int]]] = field(default_factory=list)


def _build_point_features(
    X_list: Sequence[np.ndarray],
    *,
    env=None,
    use_state: bool = True,
    use_velocity: bool = False,
    velocity_weight: float = 1.0,
    use_env_features: bool = True,
    selected_raw_feature_ids=None,
    state_distance_weight: float = 1.0,
    velocity_distance_weight: float = 1.0,
    feature_distance_weight: float = 1.0,
):
    feats = []
    feature_cols = None
    block_slices = []
    block_weights = []
    if use_env_features:
        if env is None:
            raise ValueError("env must be provided when use_env_features=True.")
        feature_cols = _resolve_selected_feature_columns(env, selected_raw_feature_ids)

    for X in X_list:
        X = np.asarray(X, dtype=float)
        parts = []
        local_slices = []
        local_weights = []
        start = 0
        if use_state:
            parts.append(X)
            local_slices.append([start, start + X.shape[1]])
            local_weights.append(float(state_distance_weight))
            start += X.shape[1]
        if use_velocity:
            vel = np.zeros_like(X)
            if len(X) > 1:
                vel[1:] = X[1:] - X[:-1]
                vel[0] = vel[1]
            parts.append(float(velocity_weight) * vel)
            local_slices.append([start, start + vel.shape[1]])
            local_weights.append(float(velocity_distance_weight))
            start += vel.shape[1]
        if use_env_features:
            F = np.asarray(env.compute_all_features_matrix(X), dtype=float)
            chosen = F[:, feature_cols]
            parts.append(chosen)
            for feat_idx in range(chosen.shape[1]):
                local_slices.append([start + feat_idx, start + feat_idx + 1])
                local_weights.append(float(feature_distance_weight))
            start += chosen.shape[1]
        if not parts:
            raise ValueError("At least one of use_state/use_velocity/use_env_features must be enabled.")
        feats.append(np.concatenate(parts, axis=1))
        if not block_slices:
            block_slices = local_slices
            block_weights = local_weights
        elif local_slices != block_slices or not np.allclose(local_weights, block_weights):
            raise ValueError("Inconsistent block layout across demos.")
    return feats, block_slices, np.asarray(block_weights, dtype=float)


def _standardize_features(F_list: Sequence[np.ndarray]):
    stack = np.concatenate([np.asarray(F, dtype=float) for F in F_list], axis=0)
    mean = np.mean(stack, axis=0)
    std = np.std(stack, axis=0) + 1e-8
    out = [(np.asarray(F, dtype=float) - mean[None, :]) / std[None, :] for F in F_list]
    return out, mean, std


def _uniform_stage_ends(T: int, num_stages: int, min_len: int) -> List[int]:
    ends = np.linspace(0, T, num_stages + 1, dtype=int)[1:] - 1
    ends[-1] = T - 1
    for k in range(num_stages - 1):
        min_end = (k + 1) * min_len - 1
        max_end = ends[k + 1] - min_len
        ends[k] = int(np.clip(ends[k], min_end, max_end))
    return [int(x) for x in ends.tolist()]


def _random_stage_ends(T: int, num_stages: int, min_len: int, proportions: np.ndarray) -> List[int]:
    extra = T - num_stages * min_len
    if extra < 0:
        raise ValueError(
            f"Sequence length {T} is too short for {num_stages} stages with minimum segment length {min_len}."
        )
    desired = float(extra) * np.asarray(proportions, dtype=float)
    extra_parts = np.floor(desired).astype(int)
    remainder = int(extra - np.sum(extra_parts))
    if remainder > 0:
        frac_order = np.argsort(desired - extra_parts)[::-1]
        extra_parts[frac_order[:remainder]] += 1
    lengths = extra_parts + int(min_len)
    ends = np.cumsum(lengths) - 1
    ends[-1] = T - 1
    return [int(x) for x in ends.tolist()]


def _init_stage_ends(X_list: Sequence[np.ndarray], num_stages: int, min_len: int, rng: np.random.RandomState, mode: str):
    mode_l = str(mode).lower()
    out = []
    shared_proportions = None
    if num_stages > 2 and mode_l in {"random_taus", "random_stage_ends", "random"}:
        shared_proportions = rng.dirichlet(np.full(num_stages, 0.5, dtype=float))
    for X in X_list:
        T = len(X)
        if T < num_stages * min_len:
            raise ValueError(
                f"Sequence length {T} is too short for {num_stages} stages with minimum segment length {min_len}."
            )
        if mode_l in {"random_taus", "random_stage_ends", "random"}:
            if num_stages == 2:
                lam = float(np.clip(rng.rand(), 0.1, 0.9))
                tau = int(round(lam * (T - 1)))
                tau = int(np.clip(tau, min_len - 1, T - min_len - 1))
                out.append([tau, T - 1])
            else:
                out.append(_random_stage_ends(T, num_stages, min_len, shared_proportions))
        else:
            out.append(_uniform_stage_ends(T, num_stages, min_len))
    return out


def _labels_from_stage_ends(T: int, stage_ends: Sequence[int]) -> np.ndarray:
    labels = np.zeros(int(T), dtype=int)
    start = 0
    for stage_idx, end in enumerate(np.asarray(stage_ends, dtype=int).reshape(-1)):
        end_i = int(end)
        labels[start : end_i + 1] = int(stage_idx)
        start = end_i + 1
    return labels


def _update_centers(F_list: Sequence[np.ndarray], stage_ends_list: Sequence[Sequence[int]], num_stages: int, prev_centers=None):
    D = int(np.asarray(F_list[0], dtype=float).shape[1])
    sums = np.zeros((num_stages, D), dtype=float)
    counts = np.zeros(num_stages, dtype=int)
    for F, stage_ends in zip(F_list, stage_ends_list):
        labels = _labels_from_stage_ends(len(F), stage_ends)
        for stage_idx in range(num_stages):
            mask = labels == stage_idx
            if np.any(mask):
                sums[stage_idx] += np.sum(F[mask], axis=0)
                counts[stage_idx] += int(np.sum(mask))
    if prev_centers is None:
        prev_centers = np.zeros((num_stages, D), dtype=float)
    centers = np.asarray(prev_centers, dtype=float).copy()
    global_mean = np.mean(np.concatenate(F_list, axis=0), axis=0)
    for stage_idx in range(num_stages):
        if counts[stage_idx] > 0:
            centers[stage_idx] = sums[stage_idx] / float(counts[stage_idx])
        elif stage_idx > 0:
            centers[stage_idx] = centers[stage_idx - 1]
        else:
            centers[stage_idx] = global_mean
    return centers


def _block_weighted_point_cost(F: np.ndarray, centers: np.ndarray, block_slices: Sequence[Sequence[int]], block_weights: np.ndarray):
    F = np.asarray(F, dtype=float)
    centers = np.asarray(centers, dtype=float)
    K = int(centers.shape[0])
    point_cost = np.zeros((F.shape[0], K), dtype=float)
    for block_idx, (start, end) in enumerate(block_slices):
        start_i = int(start)
        end_i = int(end)
        weight = float(block_weights[block_idx])
        if weight == 0.0 or end_i <= start_i:
            continue
        delta = F[:, None, start_i:end_i] - centers[None, :, start_i:end_i]
        point_cost += weight * np.sum(delta * delta, axis=2)
    return point_cost


def _segment_with_fixed_centers(F: np.ndarray, centers: np.ndarray, block_slices: Sequence[Sequence[int]], block_weights: np.ndarray, min_len: int):
    F = np.asarray(F, dtype=float)
    centers = np.asarray(centers, dtype=float)
    T = int(F.shape[0])
    K = int(centers.shape[0])
    if T < K * int(min_len):
        raise ValueError(f"Sequence length {T} is too short for {K} stages with minimum segment length {min_len}.")

    point_cost = _block_weighted_point_cost(F, centers, block_slices, block_weights)
    prefix = np.zeros((K, T + 1), dtype=float)
    prefix[:, 1:] = np.cumsum(point_cost.T, axis=1)

    def segment_cost(stage_idx: int, start: int, end: int) -> float:
        return float(prefix[stage_idx, end + 1] - prefix[stage_idx, start])

    dp = np.full((K, T), np.inf, dtype=float)
    prev = np.full((K, T), -1, dtype=int)

    for end in range(min_len - 1, T - min_len * (K - 1)):
        dp[0, end] = segment_cost(0, 0, end)

    for stage_idx in range(1, K):
        end_low = (stage_idx + 1) * min_len - 1
        end_high = T - min_len * (K - stage_idx - 1) - 1
        for end in range(end_low, end_high + 1):
            prev_low = stage_idx * min_len - 1
            prev_high = end - min_len
            best = np.inf
            arg = -1
            for prev_end in range(prev_low, prev_high + 1):
                cand = dp[stage_idx - 1, prev_end] + segment_cost(stage_idx, prev_end + 1, end)
                if cand < best:
                    best = float(cand)
                    arg = int(prev_end)
            dp[stage_idx, end] = best
            prev[stage_idx, end] = arg

    stage_ends = [T - 1]
    cur_end = T - 1
    for stage_idx in range(K - 1, 0, -1):
        prev_end = int(prev[stage_idx, cur_end])
        if prev_end < 0:
            raise RuntimeError("Failed to backtrack ordered cluster segmentation.")
        stage_ends.append(prev_end)
        cur_end = prev_end
    stage_ends.reverse()
    objective = float(dp[K - 1, T - 1])
    return [int(x) for x in stage_ends], objective


def segment_ordered_cluster(
    X_list,
    *,
    env=None,
    n_stages: int = 2,
    selected_raw_feature_ids=None,
    use_state: bool = True,
    use_velocity: bool = False,
    velocity_weight: float = 1.0,
    use_env_features: bool = True,
    state_distance_weight: float = 1.0,
    velocity_distance_weight: float = 1.0,
    feature_distance_weight: float = 1.0,
    standardize: bool = True,
    min_len: int = 3,
    max_iter: int = 20,
    n_init: int = 8,
    init_mode: str = "random_stage_ends",
    seed: int = 0,
    verbose: bool = True,
):
    X_list = [np.asarray(X, dtype=float) for X in X_list]
    raw_features, block_slices, block_weights = _build_point_features(
        X_list,
        env=env,
        use_state=use_state,
        use_velocity=use_velocity,
        velocity_weight=velocity_weight,
        use_env_features=use_env_features,
        selected_raw_feature_ids=selected_raw_feature_ids,
        state_distance_weight=state_distance_weight,
        velocity_distance_weight=velocity_distance_weight,
        feature_distance_weight=feature_distance_weight,
    )
    if standardize:
        F_list, mean, std = _standardize_features(raw_features)
    else:
        F_list = [np.asarray(F, dtype=float) for F in raw_features]
        mean = np.zeros(F_list[0].shape[1], dtype=float)
        std = np.ones(F_list[0].shape[1], dtype=float)

    rng_master = np.random.RandomState(int(seed))
    best = None
    num_stages = int(n_stages)
    min_len = int(min_len)
    max_iter = int(max_iter)
    n_init = max(int(n_init), 1)

    for init_idx in range(n_init):
        rng = np.random.RandomState(int(rng_master.randint(0, 2**31 - 1)))
        stage_ends = _init_stage_ends(X_list, num_stages, min_len, rng, init_mode)
        centers = None
        objective_history = []
        segmentation_history = [[list(ends) for ends in stage_ends]]

        for iteration in range(max_iter):
            centers = _update_centers(F_list, stage_ends, num_stages, prev_centers=centers)
            new_stage_ends = []
            total_objective = 0.0
            for F in F_list:
                ends_i, obj_i = _segment_with_fixed_centers(
                    F,
                    centers,
                    block_slices,
                    block_weights,
                    min_len=min_len,
                )
                new_stage_ends.append(ends_i)
                total_objective += float(obj_i)
            objective_history.append(float(total_objective))
            segmentation_history.append([[int(x) for x in ends] for ends in new_stage_ends])
            if verbose:
                print(f"[ordered_cluster] init {init_idx + 1:02d} iter {iteration + 1:02d} | objective={total_objective:.3f}")
            if new_stage_ends == stage_ends:
                stage_ends = new_stage_ends
                break
            stage_ends = new_stage_ends

        centers = _update_centers(F_list, stage_ends, num_stages, prev_centers=centers)
        labels = [_labels_from_stage_ends(len(F), ends) for F, ends in zip(F_list, stage_ends)]
        final_objective = float(objective_history[-1]) if objective_history else float("inf")
        model = OrderedClusterModel(
            num_stages=num_stages,
            centers_=np.asarray(centers, dtype=float),
            stage_ends_=[[int(x) for x in ends] for ends in stage_ends],
            feature_mean_=np.asarray(mean, dtype=float),
            feature_std_=np.asarray(std, dtype=float),
            block_slices_=[[int(v) for v in pair] for pair in block_slices],
            block_weights_=np.asarray(block_weights, dtype=float),
            objective_history_=[float(x) for x in objective_history],
            segmentation_history_=[[[int(v) for v in ends] for ends in snapshot] for snapshot in segmentation_history],
        )
        if best is None or final_objective < best["objective"]:
            best = {
                "labels": labels,
                "model": model,
                "objective": final_objective,
            }

    return best["labels"], best["model"]
