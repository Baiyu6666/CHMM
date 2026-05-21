from __future__ import annotations

from typing import Sequence
from pathlib import Path
import shutil
import subprocess

import numpy as np


def parse_int_list(text: str | None) -> list[int]:
    if text is None or not str(text).strip():
        return []
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def plan_seed_list(seed: int, plan_seeds: str | None, n_plans: int | None) -> list[int]:
    explicit = parse_int_list(plan_seeds)
    if explicit:
        return explicit
    count = int(1 if n_plans is None else max(1, int(n_plans)))
    return [int(seed) + i for i in range(count)]


def concat_mp4_files(inputs: Sequence[str | Path], output_path: str | Path) -> Path:
    paths = [Path(p) for p in inputs if p is not None and Path(p).exists()]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not paths:
        return out
    if len(paths) == 1:
        shutil.copyfile(paths[0], out)
        return out
    list_path = out.with_suffix(out.suffix + ".concat.txt")
    list_path.write_text(
        "".join(f"file '{str(p.resolve()).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for p in paths),
        encoding="utf-8",
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(list_path),
                "-c",
                "copy",
                str(out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    finally:
        try:
            list_path.unlink()
        except OSError:
            pass
    return out


def stage_labels_from_cutpoints(length: int, cutpoints: Sequence[int] | np.ndarray | None) -> np.ndarray:
    T = int(max(0, length))
    labels = np.zeros(T, dtype=int)
    if T <= 0:
        return labels
    cuts = [] if cutpoints is None else np.asarray(cutpoints, dtype=int).reshape(-1).tolist()
    cuts = sorted(int(c) for c in cuts if 0 <= int(c) < T - 1)
    start = 0
    for stage_idx, end in enumerate(cuts + [T - 1]):
        labels[start:int(end) + 1] = int(stage_idx)
        start = int(end) + 1
    return labels


def _constraint_kind(spec: dict) -> str:
    text = str(spec.get("semantics", "")).strip().lower()
    if text in {"target", "target_value", "equality", "eq", "equal"}:
        return "target"
    if text in {"upper", "upper_bound", "max", "maximum", "<=", "leq"}:
        return "upper"
    if text in {"lower", "lower_bound", "min", "minimum", ">=", "geq"}:
        return "lower"
    return text


def apply_inequality_constraint_clearance(
    constraint_values: dict,
    constraint_specs: Sequence[dict],
    *,
    upper_scale: float = 0.96,
    lower_scale: float = 1.04,
) -> dict:
    out = dict(constraint_values or {})
    for spec in list(constraint_specs or []):
        feature_name = str(spec.get("feature_name", ""))
        stage = int(spec.get("stage", -1))
        if not feature_name or stage < 0:
            continue
        key = f"s{stage + 1}:{feature_name}"
        if key not in out:
            continue
        value = float(out[key])
        if not np.isfinite(value):
            continue
        kind = _constraint_kind(spec)
        if kind == "upper":
            out[key] = float(value * float(upper_scale))
        elif kind == "lower":
            out[key] = float(value * float(lower_scale))
    return out


def _feature_name_to_idx(feature_schema: Sequence[dict], n_features: int) -> tuple[list[str], dict[str, int]]:
    names = [f"feature_{i}" for i in range(int(n_features))]
    mapping = {name: i for i, name in enumerate(names)}
    for idx, spec in enumerate(list(feature_schema or [])):
        col = int(spec.get("column_idx", spec.get("id", idx)))
        if 0 <= col < int(n_features):
            name = str(spec.get("name", f"feature_{col}"))
            names[col] = name
            mapping[name] = col
    return names, mapping


def constraint_violation_stats(
    *,
    features_list: Sequence[np.ndarray],
    cutpoints_list: Sequence[Sequence[int] | np.ndarray],
    feature_schema: Sequence[dict],
    constraint_specs: Sequence[dict],
    true_constraints: dict,
    equality_tolerance: float = 1e-4,
) -> dict:
    if not features_list:
        return {
            "average_violation_rate": None,
            "total_timesteps": 0,
            "violating_timesteps": 0,
            "feature_stage_violation_rate_matrix": [],
            "feature_names": [],
            "stage_names": [],
        }

    n_features = max(int(np.asarray(F).shape[1]) for F in features_list if np.asarray(F).ndim == 2)
    feature_names, name_to_idx = _feature_name_to_idx(feature_schema, n_features)
    n_stages = 0
    for F, cutpoints in zip(features_list, cutpoints_list):
        labels = stage_labels_from_cutpoints(int(np.asarray(F).shape[0]), cutpoints)
        if labels.size:
            n_stages = max(n_stages, int(np.max(labels)) + 1)
    for spec in list(constraint_specs or []):
        n_stages = max(n_stages, int(spec.get("stage", -1)) + 1)
    if n_stages <= 0:
        n_stages = 1

    violation_counts = np.zeros((n_stages, n_features), dtype=float)
    denom_counts = np.zeros((n_stages, n_features), dtype=float)
    any_violation_total = 0
    total_timesteps = 0
    tol = float(max(float(equality_tolerance), 0.0))

    for F_raw, cutpoints in zip(features_list, cutpoints_list):
        F = np.asarray(F_raw, dtype=float)
        if F.ndim != 2 or F.shape[0] == 0:
            continue
        labels = stage_labels_from_cutpoints(int(F.shape[0]), cutpoints)
        any_violation = np.zeros(int(F.shape[0]), dtype=bool)
        total_timesteps += int(F.shape[0])
        for spec in list(constraint_specs or []):
            feature_name = str(spec.get("feature_name", ""))
            if feature_name not in name_to_idx:
                continue
            feat_idx = int(name_to_idx[feature_name])
            if feat_idx >= int(F.shape[1]):
                continue
            stage_idx = int(spec.get("stage", -1))
            if stage_idx < 0 or stage_idx >= n_stages:
                continue
            oracle_key = str(spec.get("oracle_key", ""))
            if oracle_key not in true_constraints:
                continue
            bound = float(true_constraints[oracle_key])
            if not np.isfinite(bound):
                continue
            mask = labels == stage_idx
            if not np.any(mask):
                continue
            vals = F[:, feat_idx]
            finite = mask & np.isfinite(vals)
            if not np.any(finite):
                continue
            kind = _constraint_kind(spec)
            if kind == "target":
                violated = finite & (np.abs(vals - bound) > tol)
            elif kind == "upper":
                violated = finite & (vals > bound + tol)
            elif kind == "lower":
                violated = finite & (vals < bound - tol)
            else:
                continue
            denom_counts[stage_idx, feat_idx] += float(np.count_nonzero(finite))
            violation_counts[stage_idx, feat_idx] += float(np.count_nonzero(violated))
            any_violation |= violated
        any_violation_total += int(np.count_nonzero(any_violation))

    matrix = np.full((n_stages, n_features), np.nan, dtype=float)
    active = denom_counts > 0.0
    matrix[active] = violation_counts[active] / denom_counts[active]
    return {
        "average_violation_rate": None if total_timesteps <= 0 else float(any_violation_total / float(total_timesteps)),
        "total_timesteps": int(total_timesteps),
        "violating_timesteps": int(any_violation_total),
        "feature_stage_violation_rate_matrix": matrix.tolist(),
        "feature_names": feature_names,
        "stage_names": [f"stage{i + 1}" for i in range(n_stages)],
    }
