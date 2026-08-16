from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .ordered_cluster import _build_point_features, _standardize_features


@dataclass
class ChangeForestModel:
    num_stages: int
    stage_ends_: List[List[int]]
    feature_mean_: np.ndarray
    feature_std_: np.ndarray
    split_history_: List[List[Dict[str, Any]]] = field(default_factory=list)
    total_gain_: float = 0.0
    objective_history_: List[float] = field(default_factory=list)


@dataclass(frozen=True)
class _IntervalProfile:
    start: int
    stop: int
    gain_curve: np.ndarray
    best_split: int
    max_gain: float
    p_value: float | None
    is_significant: bool | None


@dataclass(frozen=True)
class _SplitCandidate:
    start: int
    stop: int
    split: int
    gain: float
    profile: _IntervalProfile


def _load_changeforest():
    try:
        from changeforest import Control, changeforest
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The changeforest baseline requires the optional 'changeforest' package. "
            "Install project dependencies with 'pip install -r requirements.txt'."
        ) from exc
    return Control, changeforest


def _control_minimum_fraction(segment_length: int, min_len: int) -> float:
    fraction = min(float(min_len) / float(segment_length), 0.1)
    return float(np.clip(fraction, np.nextafter(0.0, 1.0), np.nextafter(0.5, 0.0)))


def _extract_gain_curve(result, segment_length: int) -> np.ndarray:
    optimizer_result = getattr(result, "optimizer_result", None)
    gain_results = getattr(optimizer_result, "gain_results", None)
    if gain_results:
        gain = getattr(gain_results[-1], "gain", None)
        if gain is not None:
            curve = np.asarray(gain, dtype=float).reshape(-1)
            if curve.size == int(segment_length):
                return curve

    curve = np.full(int(segment_length), -np.inf, dtype=float)
    best_split = getattr(result, "best_split", None)
    max_gain = getattr(result, "max_gain", None)
    if best_split is not None and max_gain is not None:
        split_idx = int(best_split)
        if 0 <= split_idx < curve.size:
            curve[split_idx] = float(max_gain)
    return curve


def _fit_interval_profile(
    features: np.ndarray,
    *,
    start: int,
    stop: int,
    min_len: int,
    method: str,
    seed: int,
    random_forest_n_estimators: int,
    random_forest_max_depth: int | None,
    random_forest_max_features: str | int | None,
    random_forest_n_jobs: int,
    model_selection_n_permutations: int,
) -> _IntervalProfile:
    Control, changeforest = _load_changeforest()
    segment = np.asarray(features[start:stop], dtype=float)
    segment_length = int(len(segment))
    control = Control(
        minimal_relative_segment_length=_control_minimum_fraction(segment_length, min_len),
        model_selection_n_permutations=int(model_selection_n_permutations),
        seed=int(seed),
        random_forest_n_estimators=int(random_forest_n_estimators),
        random_forest_max_depth=random_forest_max_depth,
        random_forest_max_features=random_forest_max_features,
        random_forest_n_jobs=int(random_forest_n_jobs),
    )
    result = changeforest(segment, str(method), "bs", control)
    best_split = getattr(result, "best_split", None)
    max_gain = getattr(result, "max_gain", None)
    if best_split is None or max_gain is None:
        raise RuntimeError(
            f"changeforest did not return a split candidate for interval [{start}, {stop})."
        )
    if int(model_selection_n_permutations) == 0:
        p_value = None
        is_significant = None
    else:
        p_value = getattr(result, "p_value", None)
        is_significant = getattr(result, "is_significant", None)
    return _IntervalProfile(
        start=int(start),
        stop=int(stop),
        gain_curve=_extract_gain_curve(result, segment_length),
        best_split=int(best_split),
        max_gain=float(max_gain),
        p_value=None if p_value is None else float(p_value),
        is_significant=None if is_significant is None else bool(is_significant),
    )


def _remaining_segment_capacity(segments: Sequence[Tuple[int, int]], min_len: int) -> int:
    return int(sum((int(stop) - int(start)) // int(min_len) for start, stop in segments))


def _best_feasible_candidate(
    profile: _IntervalProfile,
    *,
    segments: Sequence[Tuple[int, int]],
    target_num_stages: int,
    min_len: int,
) -> _SplitCandidate | None:
    segment_length = int(profile.stop - profile.start)
    other_capacity = _remaining_segment_capacity(
        [bounds for bounds in segments if bounds != (profile.start, profile.stop)],
        min_len,
    )
    candidates = []
    for local_split in range(int(min_len), segment_length - int(min_len) + 1):
        split_capacity = local_split // int(min_len) + (segment_length - local_split) // int(min_len)
        if other_capacity + split_capacity < int(target_num_stages):
            continue
        gain = float(profile.gain_curve[local_split])
        if np.isfinite(gain):
            candidates.append((gain, local_split))
    if not candidates:
        return None
    gain, local_split = max(candidates, key=lambda item: (item[0], -item[1]))
    return _SplitCandidate(
        start=int(profile.start),
        stop=int(profile.stop),
        split=int(profile.start + local_split),
        gain=float(gain),
        profile=profile,
    )


def _segment_one_demo(
    features: np.ndarray,
    *,
    demo_index: int,
    n_stages: int,
    min_len: int,
    method: str,
    seed: int,
    random_forest_n_estimators: int,
    random_forest_max_depth: int | None,
    random_forest_max_features: str | int | None,
    random_forest_n_jobs: int,
    model_selection_n_permutations: int,
):
    T = int(len(features))
    if T < int(n_stages) * int(min_len):
        raise ValueError(
            f"Sequence length {T} is too short for {n_stages} stages with minimum length {min_len}."
        )
    if int(n_stages) == 1:
        return np.zeros(T, dtype=int), [T - 1], []

    segments: List[Tuple[int, int]] = [(0, T)]
    profile_cache: Dict[Tuple[int, int], _IntervalProfile] = {}
    split_history: List[Dict[str, Any]] = []

    while len(segments) < int(n_stages):
        candidates: List[_SplitCandidate] = []
        for start, stop in segments:
            if int(stop - start) < 2 * int(min_len):
                continue
            bounds = (int(start), int(stop))
            if bounds not in profile_cache:
                interval_seed = (
                    int(seed)
                    + (int(demo_index) + 1) * 1_000_003
                    + (int(start) + 1) * 10_007
                    + int(stop) * 101
                ) % (2**32)
                profile_cache[bounds] = _fit_interval_profile(
                    features,
                    start=start,
                    stop=stop,
                    min_len=min_len,
                    method=method,
                    seed=interval_seed,
                    random_forest_n_estimators=random_forest_n_estimators,
                    random_forest_max_depth=random_forest_max_depth,
                    random_forest_max_features=random_forest_max_features,
                    random_forest_n_jobs=random_forest_n_jobs,
                    model_selection_n_permutations=model_selection_n_permutations,
                )
            candidate = _best_feasible_candidate(
                profile_cache[bounds],
                segments=segments,
                target_num_stages=n_stages,
                min_len=min_len,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            raise RuntimeError(
                f"Unable to construct {n_stages} fixed-K changeforest stages for demo {demo_index}."
            )
        chosen = max(candidates, key=lambda item: (item.gain, -item.start, -item.split))
        segments.remove((chosen.start, chosen.stop))
        segments.extend([(chosen.start, chosen.split), (chosen.split, chosen.stop)])
        segments.sort()
        split_history.append(
            {
                "start": int(chosen.start),
                "stop": int(chosen.stop),
                "split": int(chosen.split),
                "gain": float(chosen.gain),
                "unrestricted_best_split": int(chosen.start + chosen.profile.best_split),
                "unrestricted_max_gain": float(chosen.profile.max_gain),
                "p_value": chosen.profile.p_value,
                "is_significant": chosen.profile.is_significant,
            }
        )

    labels = np.zeros(T, dtype=int)
    for stage_idx, (start, stop) in enumerate(segments):
        labels[start:stop] = int(stage_idx)
    stage_ends = [int(stop - 1) for _, stop in segments]
    return labels, stage_ends, split_history


def segment_with_changeforest(
    X_list,
    *,
    env=None,
    n_stages: int = 2,
    selected_raw_feature_ids=None,
    use_state: bool = False,
    use_velocity: bool = True,
    velocity_weight: float = 1.0,
    use_env_features: bool = False,
    standardize: bool = True,
    min_len: int = 3,
    method: str = "random_forest",
    seed: int = 0,
    random_forest_n_estimators: int = 50,
    random_forest_max_depth: int | None = 1,
    random_forest_max_features: str | int | None = "sqrt",
    random_forest_n_jobs: int = 1,
    model_selection_n_permutations: int = 0,
    verbose: bool = True,
):
    X_list = [np.asarray(X, dtype=float) for X in X_list]
    n_stages = int(n_stages)
    min_len = int(min_len)
    if n_stages < 1:
        raise ValueError("n_stages must be at least 1.")
    if min_len < 1:
        raise ValueError("min_len must be at least 1.")
    if str(method) not in {"random_forest", "knn", "change_in_mean"}:
        raise ValueError("method must be one of: random_forest, knn, change_in_mean.")
    if int(model_selection_n_permutations) < 0:
        raise ValueError("model_selection_n_permutations must be non-negative.")

    raw_features, _, _ = _build_point_features(
        X_list,
        env=env,
        use_state=use_state,
        use_velocity=use_velocity,
        velocity_weight=velocity_weight,
        use_env_features=use_env_features,
        selected_raw_feature_ids=selected_raw_feature_ids,
    )
    if standardize:
        features, feature_mean, feature_std = _standardize_features(raw_features)
    else:
        features = [np.asarray(values, dtype=float) for values in raw_features]
        feature_dim = int(features[0].shape[1])
        feature_mean = np.zeros(feature_dim, dtype=float)
        feature_std = np.ones(feature_dim, dtype=float)

    labels = []
    stage_ends = []
    histories = []
    for demo_index, values in enumerate(features):
        demo_labels, demo_stage_ends, demo_history = _segment_one_demo(
            values,
            demo_index=demo_index,
            n_stages=n_stages,
            min_len=min_len,
            method=method,
            seed=seed,
            random_forest_n_estimators=random_forest_n_estimators,
            random_forest_max_depth=random_forest_max_depth,
            random_forest_max_features=random_forest_max_features,
            random_forest_n_jobs=random_forest_n_jobs,
            model_selection_n_permutations=model_selection_n_permutations,
        )
        labels.append(demo_labels)
        stage_ends.append(demo_stage_ends)
        histories.append(demo_history)
        if verbose:
            cutpoints = demo_stage_ends[:-1]
            total_gain = float(sum(item["gain"] for item in demo_history))
            print(
                f"[changeforest] demo {demo_index:02d} | "
                f"cutpoints={cutpoints} | total_gain={total_gain:.3f}"
            )

    total_gain = float(sum(item["gain"] for history in histories for item in history))
    model = ChangeForestModel(
        num_stages=n_stages,
        stage_ends_=stage_ends,
        feature_mean_=np.asarray(feature_mean, dtype=float),
        feature_std_=np.asarray(feature_std, dtype=float),
        split_history_=histories,
        total_gain_=total_gain,
        objective_history_=[total_gain],
    )
    return labels, model
