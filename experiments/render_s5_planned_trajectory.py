from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.S5SphereInspect import S5SphereInspectEnv, _apply_default_s5_loader_config
from experiments.render_metrics import (
    apply_inequality_constraint_clearance,
    concat_mp4_files,
    constraint_violation_stats,
    parse_int_list,
    plan_seed_list,
    print_render_violation_rates,
)


def _load_json(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return json.loads(p.read_text(encoding="utf-8"))


def _load_constraint_payload(path: str | Path) -> tuple[dict, Path]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    payload = json.loads(p.read_text(encoding="utf-8"))
    if "all_metrics" in payload and isinstance(payload.get("all_metrics"), dict):
        payload = dict(payload["all_metrics"])
    if p.name == "constraints.json":
        metrics_path = p.with_name("metrics.json")
        if metrics_path.exists():
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics_view = dict(metrics_payload.get("all_metrics", metrics_payload))
            for key in (
                "ConstraintPredictedActiveMask",
                "ConstraintLearnedSemanticsMatrix",
                "ConstraintLearnedValueMatrix",
                "ConstraintLearnedRawValueMatrix",
                "ConstraintTargetMatrix",
                "ConstraintFeatureNames",
            ):
                if key not in payload and key in metrics_view:
                    payload[key] = metrics_view[key]
    return payload, p


def _select_constraint_payload(payload: dict, *, method: str | None, dataset: str | None, method_seed: int | None) -> dict:
    if "results" not in payload:
        return dict(payload)
    rows = list(payload.get("results", []))
    candidates = []
    for row in rows:
        if method is not None and str(row.get("method", "")) != str(method):
            continue
        if dataset is not None and str(row.get("dataset", "")) != str(dataset):
            continue
        if method_seed is not None and int(row.get("method_seed", -1)) != int(method_seed):
            continue
        candidates.append(row)
    if not candidates:
        raise ValueError(
            "No matching benchmark result row. Use --benchmark-method, --benchmark-dataset, "
            "and --benchmark-method-seed to disambiguate benchmark_results.json."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Found {len(candidates)} matching benchmark rows. Add --benchmark-method, "
            "--benchmark-dataset, or --benchmark-method-seed."
        )
    out = dict(candidates[0].get("metrics", {}))
    out["benchmark_row"] = {
        "method": candidates[0].get("method"),
        "dataset": candidates[0].get("dataset"),
        "method_seed": candidates[0].get("method_seed"),
    }
    return out


def _finite_matrix_value(matrix, stage_idx: int, feature_idx: int):
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or stage_idx >= arr.shape[0] or feature_idx >= arr.shape[1]:
        return None
    value = float(arr[stage_idx, feature_idx])
    return value if np.isfinite(value) else None


def _matrix_bool_value(matrix, stage_idx: int, feature_idx: int):
    if matrix is None:
        return None
    arr = np.asarray(matrix)
    if arr.ndim != 2 or stage_idx >= arr.shape[0] or feature_idx >= arr.shape[1]:
        return None
    try:
        return bool(int(round(float(arr[stage_idx, feature_idx]))))
    except (TypeError, ValueError):
        return None


def _active_mask_from_matrix(matrix) -> np.ndarray | None:
    if matrix is None:
        return None
    arr = np.asarray(matrix)
    if arr.ndim != 2:
        return None
    try:
        return np.isfinite(arr.astype(float))
    except (TypeError, ValueError):
        return None


def _active_mask_from_payload(payload: dict, learned_matrix) -> np.ndarray | None:
    predicted = payload.get("ConstraintPredictedActiveMask")
    if predicted is not None:
        arr = np.asarray(predicted)
        if arr.ndim == 2:
            return np.rint(arr.astype(float)).astype(bool)
    return _active_mask_from_matrix(learned_matrix)


def _learned_constraint_specs_from_payload(payload: dict, feature_names: list[str]) -> list[dict]:
    semantics = payload.get("ConstraintLearnedSemanticsMatrix")
    learned = payload.get("ConstraintLearnedValueMatrix")
    active = payload.get("ConstraintPredictedActiveMask")
    if semantics is None:
        return []
    sem_arr = np.asarray(semantics, dtype=object)
    if sem_arr.ndim != 2:
        return []
    value_arr = None
    if learned is not None:
        try:
            value_arr = np.asarray(learned, dtype=float)
        except (TypeError, ValueError):
            value_arr = None
    active_arr = None
    if active is not None:
        try:
            active_arr = np.rint(np.asarray(active, dtype=float)).astype(bool)
        except (TypeError, ValueError):
            active_arr = None

    specs = []
    for stage_idx in range(sem_arr.shape[0]):
        for feature_idx in range(sem_arr.shape[1]):
            if feature_idx >= len(feature_names):
                continue
            if active_arr is not None and active_arr.shape == sem_arr.shape and not bool(active_arr[stage_idx, feature_idx]):
                continue
            if value_arr is not None and value_arr.shape == sem_arr.shape and not np.isfinite(float(value_arr[stage_idx, feature_idx])):
                continue
            semantic = str(sem_arr[stage_idx, feature_idx]).strip()
            if not semantic:
                continue
            specs.append(
                {
                    "feature_name": str(feature_names[feature_idx]),
                    "stage": int(stage_idx),
                    "semantics": semantic,
                }
            )
    return specs


def _constraint_specs_from_payload(
    payload: dict,
    env: S5SphereInspectEnv,
    *,
    constraint_source: str,
    feature_names: list[str],
) -> list[dict]:
    return list(payload.get("constraint_specs") or env.get_constraint_specs())


def _true_active_mask_from_specs(payload: dict, specs: list[dict], feature_names: list[str], shape: tuple[int, int]) -> np.ndarray:
    true_mask = payload.get("ConstraintTrueActiveMask")
    if true_mask is not None:
        arr = np.asarray(true_mask)
        if arr.ndim == 2:
            return np.rint(arr.astype(float)).astype(bool)
    out = np.zeros(shape, dtype=bool)
    for spec in specs:
        name = str(spec.get("feature_name", ""))
        if name not in feature_names:
            continue
        stage_idx = int(spec.get("stage", -1))
        feat_idx = int(feature_names.index(name))
        if 0 <= stage_idx < out.shape[0] and 0 <= feat_idx < out.shape[1]:
            out[stage_idx, feat_idx] = True
    return out


def _format_mask_entries(mask: np.ndarray, feature_names: list[str], limit: int = 20) -> str:
    rows = []
    for stage_idx, feat_idx in np.argwhere(mask)[: int(limit)]:
        name = feature_names[int(feat_idx)] if int(feat_idx) < len(feature_names) else f"feature_{int(feat_idx)}"
        rows.append(f"s{int(stage_idx) + 1}:{name}")
    suffix = "" if int(np.sum(mask)) <= int(limit) else f"; ... {int(np.sum(mask)) - int(limit)} more"
    return "; ".join(rows) + suffix


def _check_learned_mask_covers_gt(payload: dict, learned_matrix, feature_names: list[str], specs: list[dict]) -> None:
    learned_arr = _active_mask_from_payload(payload, learned_matrix)
    if learned_arr is None:
        raise ValueError("Cannot infer learned active mask from ConstraintPredictedActiveMask or ConstraintLearnedValueMatrix.")
    true_arr = _true_active_mask_from_specs(payload, specs, feature_names, learned_arr.shape)
    if true_arr.shape != learned_arr.shape:
        raise ValueError(
            f"GT active mask shape {true_arr.shape} does not match learned active mask shape {learned_arr.shape}."
        )
    missing = true_arr & ~learned_arr
    if np.any(missing):
        raise ValueError(
            "Learned active mask is missing GT constraints required for rendering: "
            + _format_mask_entries(missing, feature_names)
        )
    extra = learned_arr & ~true_arr
    if np.any(extra):
        warnings.warn(
            "Learned active mask contains extra constraints not in the GT active mask; "
            "they will be ignored for rendering: " + _format_mask_entries(extra, feature_names),
            RuntimeWarning,
            stacklevel=2,
        )


def _constraint_values_from_payload(payload: dict, env: S5SphereInspectEnv, *, constraint_source: str = "learned") -> dict:
    feature_names = list(payload.get("ConstraintFeatureNames", []))
    if not feature_names:
        feature_names = ["surf_dist", "normal_err", "speed", "ang_speed", "start_dist", "goal_dist"]
    source = str(constraint_source or "learned").strip().lower()
    if source not in {"learned", "target"}:
        raise ValueError("constraint_source must be 'learned' or 'target'.")
    learned = payload.get("ConstraintLearnedValueMatrix") if source == "learned" else payload.get("ConstraintTargetMatrix")
    if learned is None:
        raise ValueError(
            "Constraint JSON does not contain the required constraint value matrix for "
            f"source={source!r}. Rerun benchmark with updated metrics, or use --constraint-source target."
        )
    target = payload.get("ConstraintTargetMatrix")
    predicted_active = payload.get("ConstraintPredictedActiveMask") if source == "learned" else None
    specs = _constraint_specs_from_payload(payload, env, constraint_source=source, feature_names=feature_names)
    if source == "learned":
        _check_learned_mask_covers_gt(payload, learned, feature_names, specs)

    out = {}
    for spec in specs:
        name = str(spec.get("feature_name", ""))
        if name not in feature_names:
            continue
        stage_idx = int(spec.get("stage", 0))
        feature_idx = int(feature_names.index(name))
        value = _finite_matrix_value(learned, stage_idx, feature_idx)
        if value is None and source == "target":
            value = _finite_matrix_value(target, stage_idx, feature_idx)
        if value is None:
            if source == "learned":
                raise ValueError(f"Missing learned value for GT constraint s{stage_idx + 1}:{name}.")
            continue
        out[f"s{stage_idx + 1}:{name}"] = float(value)
    required = ["s2:surf_dist", "s2:normal_err", "s2:speed", "s4:surf_dist", "s4:speed"]
    missing = [key for key in required if key not in out]
    if missing:
        raise ValueError(f"Missing required S5 learned constraints: {missing}. Parsed values={out}")
    return out


def _parse_stage_lengths(text: str | None) -> dict | None:
    if text is None or not str(text).strip():
        return None
    out = {}
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid stage length item {item!r}; expected stage2:34.")
        key, value = item.split(":", 1)
        out[str(key).strip()] = int(value)
    return out


def _parse_vec3(text: str | None) -> tuple[float, float, float]:
    if text is None or not str(text).strip():
        return (0.10, 0.0, 0.04)
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if len(vals) != 3:
        raise ValueError(f"Expected a comma-separated 3-vector, got {text!r}.")
    return float(vals[0]), float(vals[1]), float(vals[2])


def _parse_optional_vec3(text: str | None) -> tuple[float, float, float] | None:
    if text is None or not str(text).strip():
        return None
    return _parse_vec3(text)


def _parse_frame_indices(text: str | None) -> list[int]:
    return parse_int_list(text)


def _constraint_value(constraint_values: dict, key: str, default: float) -> float:
    value = dict(constraint_values or {}).get(key)
    if value is None:
        return float(default)
    value = float(value)
    return value if np.isfinite(value) else float(default)


def _base_stage_lengths(env: S5SphereInspectEnv) -> list[int]:
    lengths = [int(x) for x in env.seg_lengths]
    while len(lengths) < 5:
        lengths.append(lengths[-1] if lengths else 18)
    lengths = lengths[:5]
    return lengths


def _enforce_stage_length_minima(lengths: list[int]) -> list[int]:
    out = [int(max(int(x), 3)) for x in lengths[:5]]
    while len(out) < 5:
        out.append(3)
    out[1] = max(int(out[1]), 8)
    out[2] = max(int(out[2]), 5)
    out[3] = max(int(out[3]), 8)
    out[4] = max(int(out[4]), 5)
    return out


def _stage_lengths(env: S5SphereInspectEnv, overrides: dict | None) -> list[int]:
    lengths = _base_stage_lengths(env)
    for key, value in dict(overrides or {}).items():
        text = str(key).strip().lower()
        if text.startswith("stage"):
            idx = int(text.replace("stage", "")) - 1
        elif text.startswith("s"):
            idx = int(text.replace("s", "")) - 1
        else:
            idx = int(text)
        if 0 <= idx < len(lengths):
            lengths[idx] = max(int(value), 3)
    return _enforce_stage_length_minima(lengths)


def _lengths_from_stage_ends(stage_ends, total_length: int | None = None) -> list[int] | None:
    ends = np.asarray(stage_ends, dtype=float).reshape(-1)
    finite = ends[np.isfinite(ends)]
    if finite.size < 4:
        return None
    ends_i = [int(round(v)) for v in finite[:5].tolist()]
    if len(ends_i) == 4:
        if total_length is None:
            return None
        ends_i.append(int(total_length) - 1)
    starts = [0] + [int(v) + 1 for v in ends_i[:-1]]
    lengths = [int(b) - int(a) + 1 for a, b in zip(starts, ends_i)]
    if len(lengths) != 5 or any(int(v) <= 0 for v in lengths):
        return None
    return lengths


def _learned_stage_length_prior(segmentation_path: str | Path | None) -> tuple[np.ndarray | None, int | None, str]:
    if segmentation_path is None:
        return None, None, "none"
    p = Path(segmentation_path)
    if not p.exists():
        return None, None, "missing"
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None, None, "unreadable"

    demo_lengths = [int(v) for v in payload.get("demo_lengths", []) if int(v) > 1]
    rows = []
    source = "predicted_stage_ends"
    for ends in payload.get("predicted_stage_ends", []) or []:
        lengths = _lengths_from_stage_ends(ends)
        if lengths is not None:
            rows.append(lengths)
    if not rows:
        source = "predicted_cutpoints"
        for i, cuts in enumerate(payload.get("predicted_cutpoints", []) or []):
            total = demo_lengths[i] if i < len(demo_lengths) else None
            lengths = _lengths_from_stage_ends(cuts, total_length=total)
            if lengths is not None:
                rows.append(lengths)
    if not rows:
        source = "true_cutpoints"
        for i, cuts in enumerate(payload.get("true_cutpoints", []) or []):
            total = demo_lengths[i] if i < len(demo_lengths) else None
            lengths = _lengths_from_stage_ends(cuts, total_length=total)
            if lengths is not None:
                rows.append(lengths)
    if not rows:
        return None, None, "empty"

    arr = np.asarray(rows, dtype=float)
    ratios = arr / np.maximum(np.sum(arr, axis=1, keepdims=True), 1.0)
    ratio = np.mean(ratios, axis=0)
    ratio = ratio / max(float(np.sum(ratio)), 1e-12)
    mean_total = int(round(float(np.mean(np.sum(arr, axis=1)))))
    return ratio.astype(float), max(mean_total, 5), source


def _allocate_lengths_from_ratio(ratio: np.ndarray, total_length: int) -> list[int]:
    ratio = np.asarray(ratio, dtype=float).reshape(-1)
    if ratio.size != 5 or not np.all(np.isfinite(ratio)) or float(np.sum(ratio)) <= 0.0:
        ratio = np.ones(5, dtype=float) / 5.0
    ratio = np.clip(ratio, 1e-6, None)
    ratio = ratio / float(np.sum(ratio))
    total = int(max(int(total_length), 5))
    raw = ratio * float(total)
    lengths = np.floor(raw).astype(int)
    lengths = np.maximum(lengths, 1)
    while int(np.sum(lengths)) < total:
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for idx in order:
            if int(np.sum(lengths)) >= total:
                break
            lengths[int(idx)] += 1
    while int(np.sum(lengths)) > total:
        order = np.argsort(-(lengths - 1))
        for idx in order:
            if int(np.sum(lengths)) <= total:
                break
            if lengths[int(idx)] > 1:
                lengths[int(idx)] -= 1
    return _enforce_stage_length_minima([int(v) for v in lengths.tolist()])


def _stage_reference_distances(start: np.ndarray, goal: np.ndarray, lengths: list[int]) -> np.ndarray:
    labels = np.repeat(np.arange(5), lengths)
    T = int(labels.size)
    if T <= 1:
        return np.zeros(5, dtype=float)
    ref = np.linspace(np.asarray(start, dtype=float), np.asarray(goal, dtype=float), T)
    dists = np.zeros(5, dtype=float)
    for stage_idx in range(5):
        idx = np.where(labels == stage_idx)[0]
        if idx.size >= 2:
            dists[stage_idx] = float(np.sum(np.linalg.norm(np.diff(ref[idx], axis=0), axis=1)))
    return dists


def _sample_polyline_count(points: np.ndarray, count: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    n = int(max(int(count), 2))
    if len(pts) <= 1:
        return np.repeat(pts[:1], n, axis=0)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(np.sum(seg))
    if total <= 1e-12:
        return np.repeat(pts[:1], n, axis=0)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    targets = np.linspace(0.0, total, n)
    out = []
    j = 0
    for target in targets:
        while j + 1 < len(s) and s[j + 1] < target:
            j += 1
        if j + 1 >= len(s):
            out.append(pts[-1])
            continue
        frac = (target - s[j]) / max(s[j + 1] - s[j], 1e-12)
        out.append((1.0 - frac) * pts[j] + frac * pts[j + 1])
    return np.asarray(out, dtype=float)


def _resample_stages_for_speed_cap(
    traj: np.ndarray,
    axis: np.ndarray,
    labels: np.ndarray,
    constraint_values: dict,
    *,
    env: S5SphereInspectEnv,
    speed_safety: float,
    global_speed_max: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    xyz_blocks = []
    axis_blocks = []
    lengths = []
    for stage_idx in range(5):
        idx = np.where(np.asarray(labels, dtype=int) == int(stage_idx))[0]
        if idx.size == 0:
            continue
        pts = np.asarray(traj[idx], dtype=float)
        axs = np.asarray(axis[idx], dtype=float)
        speed_candidates = []
        key = f"s{stage_idx + 1}:speed"
        if key in dict(constraint_values or {}):
            speed_candidates.append(_constraint_value(constraint_values, key, np.nan))
        if global_speed_max is not None and np.isfinite(float(global_speed_max)) and float(global_speed_max) > 0.0:
            speed_candidates.append(float(global_speed_max))
        speed_candidates = [float(v) for v in speed_candidates if np.isfinite(float(v)) and float(v) > 0.0]
        target_count = int(len(pts))
        if speed_candidates and len(pts) >= 2:
            max_step = max(min(speed_candidates) * float(np.clip(speed_safety, 0.10, 1.0)) * float(env.dt), 1e-8)
            path_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
            target_count = max(target_count, int(np.ceil(path_len / max_step)) + 1)
        xyz_blocks.append(_sample_polyline_count(pts, target_count))
        axis_blocks.append(_unit_rows(_sample_polyline_count(axs, target_count)))
        lengths.append(int(target_count))
    if not xyz_blocks:
        return np.asarray(traj, dtype=float), np.asarray(axis, dtype=float), np.asarray([], dtype=int), []
    traj_out = []
    axis_out = []
    labels_out = []
    for stage_idx, (xyz_block, axis_block) in enumerate(zip(xyz_blocks, axis_blocks)):
        traj_out.append(xyz_block)
        axis_out.append(axis_block)
        labels_out.extend([stage_idx] * len(xyz_block))
    out_traj = np.vstack(traj_out)
    out_axis = _unit_rows(np.vstack(axis_out))
    out_labels = np.asarray(labels_out, dtype=int)
    out_lengths = [int(np.sum(out_labels == i)) for i in range(5)]
    return out_traj, out_axis, out_labels, out_lengths


def _auto_stage_lengths(
    env: S5SphereInspectEnv,
    constraint_values: dict,
    *,
    overrides: dict | None,
    stage_length_source: str,
    segmentation_path: str | Path | None,
    start: np.ndarray,
    goal: np.ndarray,
    speed_safety: float,
    global_speed_max: float | None,
    stage_length_scale: float,
    stage4_length_multiplier: float,
) -> tuple[list[int], dict]:
    def _apply_stage4_multiplier(lengths: list[int], info: dict) -> tuple[list[int], dict]:
        multiplier = float(stage4_length_multiplier)
        if not np.isfinite(multiplier) or multiplier <= 0.0:
            multiplier = 1.0
        out = [int(v) for v in lengths]
        if len(out) >= 4 and abs(multiplier - 1.0) > 1e-12:
            out[3] = max(1, int(round(float(out[3]) * multiplier)))
        info = dict(info)
        info["stage4_length_multiplier"] = float(multiplier)
        info["lengths_after_stage4_multiplier"] = [int(v) for v in out]
        info["auto_total_length_after_stage4_multiplier"] = int(np.sum(out))
        return out, info

    if overrides:
        lengths = _stage_lengths(env, overrides)
        return lengths, {"source": "override", "stage4_length_multiplier": 1.0, "lengths": [int(v) for v in lengths]}

    source = str(stage_length_source or "learned-ratio").strip().lower()
    if source == "fixed":
        lengths = _stage_lengths(env, None)
        return _apply_stage4_multiplier(lengths, {"source": "fixed", "lengths": [int(v) for v in lengths]})

    ratio, mean_total, ratio_source = _learned_stage_length_prior(segmentation_path)
    if ratio is None or mean_total is None:
        base = np.asarray(_stage_lengths(env, None), dtype=float)
        ratio = base / float(np.sum(base))
        mean_total = int(np.sum(base))
    scale = float(stage_length_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    total = int(max(int(round(float(mean_total) * scale)), 5))
    lengths = _allocate_lengths_from_ratio(ratio, total)
    return _apply_stage4_multiplier(lengths, {
        "source": "learned-ratio",
        "ratio_source": ratio_source,
        "segmentation_path": None if segmentation_path is None else str(Path(segmentation_path)),
        "ratios": [float(v) for v in np.asarray(ratio, dtype=float).tolist()],
        "mean_total_length": int(mean_total),
        "stage_length_scale": float(scale),
        "speed_feasibility_correction": "post_projection_resample_only",
        "auto_total_length": int(np.sum(lengths)),
        "lengths": [int(v) for v in lengths],
    })


def _unit_rows(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def _project_to_sphere_offset(env: S5SphereInspectEnv, xyz: np.ndarray, offset: float) -> np.ndarray:
    rel = np.asarray(xyz, dtype=float) - env.sphere_center[None, :]
    normals = _unit_rows(rel)
    radius = max(float(env.sphere_radius) + float(offset), 1e-8)
    return env.sphere_center[None, :] + radius * normals


def _limit_edge_speeds(
    xyz: np.ndarray,
    edge_max_steps: np.ndarray,
    *,
    fixed_start: np.ndarray,
    fixed_goal: np.ndarray,
) -> None:
    edge_max_steps = np.asarray(edge_max_steps, dtype=float).reshape(-1)
    if xyz.shape[0] <= 1:
        return
    xyz[0] = np.asarray(fixed_start, dtype=float)
    for t in range(1, xyz.shape[0]):
        max_step = float(edge_max_steps[t])
        if not np.isfinite(max_step) or max_step <= 0.0:
            continue
        step = xyz[t] - xyz[t - 1]
        dist = float(np.linalg.norm(step))
        if dist > max_step:
            xyz[t] = xyz[t - 1] + step * (max_step / max(dist, 1e-12))
    xyz[-1] = np.asarray(fixed_goal, dtype=float)
    for t in range(xyz.shape[0] - 2, -1, -1):
        max_step = float(edge_max_steps[t + 1])
        if not np.isfinite(max_step) or max_step <= 0.0:
            continue
        step = xyz[t] - xyz[t + 1]
        dist = float(np.linalg.norm(step))
        if dist > max_step:
            xyz[t] = xyz[t + 1] + step * (max_step / max(dist, 1e-12))
    xyz[0] = np.asarray(fixed_start, dtype=float)
    xyz[-1] = np.asarray(fixed_goal, dtype=float)


def _limit_edge_speeds_preserve(
    xyz: np.ndarray,
    edge_max_steps: np.ndarray,
    preserve: np.ndarray,
    *,
    fixed_start: np.ndarray,
    fixed_goal: np.ndarray,
    n_rounds: int = 12,
) -> None:
    edge_max_steps = np.asarray(edge_max_steps, dtype=float).reshape(-1)
    preserve = np.asarray(preserve, dtype=bool).reshape(-1)
    if xyz.shape[0] <= 1:
        return
    preserve = np.resize(preserve, xyz.shape[0])
    preserve[0] = True
    preserve[-1] = True
    xyz[0] = np.asarray(fixed_start, dtype=float)
    xyz[-1] = np.asarray(fixed_goal, dtype=float)
    for _ in range(max(int(n_rounds), 1)):
        for t in range(1, xyz.shape[0]):
            max_step = float(edge_max_steps[t])
            if not np.isfinite(max_step) or max_step <= 0.0:
                continue
            step = xyz[t] - xyz[t - 1]
            dist = float(np.linalg.norm(step))
            if dist <= max_step:
                continue
            if preserve[t] and not preserve[t - 1]:
                xyz[t - 1] = xyz[t] - step * (max_step / max(dist, 1e-12))
            elif preserve[t - 1] and not preserve[t]:
                xyz[t] = xyz[t - 1] + step * (max_step / max(dist, 1e-12))
            elif not preserve[t] and not preserve[t - 1]:
                xyz[t] = xyz[t - 1] + step * (max_step / max(dist, 1e-12))
        xyz[0] = np.asarray(fixed_start, dtype=float)
        xyz[-1] = np.asarray(fixed_goal, dtype=float)
        for t in range(xyz.shape[0] - 2, -1, -1):
            max_step = float(edge_max_steps[t + 1])
            if not np.isfinite(max_step) or max_step <= 0.0:
                continue
            step = xyz[t] - xyz[t + 1]
            dist = float(np.linalg.norm(step))
            if dist <= max_step:
                continue
            if preserve[t] and not preserve[t + 1]:
                xyz[t + 1] = xyz[t] - step * (max_step / max(dist, 1e-12))
            elif preserve[t + 1] and not preserve[t]:
                xyz[t] = xyz[t + 1] + step * (max_step / max(dist, 1e-12))
            elif not preserve[t] and not preserve[t + 1]:
                xyz[t] = xyz[t + 1] + step * (max_step / max(dist, 1e-12))
        xyz[0] = np.asarray(fixed_start, dtype=float)
        xyz[-1] = np.asarray(fixed_goal, dtype=float)


def _apply_normal_error_bound(axis: np.ndarray, normals: np.ndarray, max_error: float) -> np.ndarray:
    out = _unit_rows(axis)
    n = _unit_rows(normals)
    bound = float(max(float(max_error), 0.0))
    dots = np.clip(np.sum(out * n, axis=1), -1.0, 1.0)
    angles = np.arccos(dots)
    mask = angles > bound
    if not np.any(mask):
        return out
    tangent = out[mask] - dots[mask, None] * n[mask]
    bad = np.linalg.norm(tangent, axis=1) <= 1e-10
    if np.any(bad):
        ref = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=float), (int(np.sum(bad)), 1))
        near = np.abs(np.sum(ref * n[mask][bad], axis=1)) > 0.9
        ref[near] = np.array([0.0, 1.0, 0.0], dtype=float)
        tangent[bad] = np.cross(n[mask][bad], ref)
    tangent = _unit_rows(tangent)
    out[mask] = np.cos(bound) * n[mask] + np.sin(bound) * tangent
    return _unit_rows(out)


def _limit_axis_angular_speed(axis: np.ndarray, edge_max_angles: np.ndarray, *, fixed_start: np.ndarray, fixed_goal: np.ndarray) -> None:
    edge_max_angles = np.asarray(edge_max_angles, dtype=float).reshape(-1)
    if axis.shape[0] <= 1:
        return
    axis[:] = _unit_rows(axis)
    axis[0] = np.asarray(fixed_start, dtype=float) / max(float(np.linalg.norm(fixed_start)), 1e-12)
    for t in range(1, axis.shape[0]):
        max_angle = float(edge_max_angles[t])
        if not np.isfinite(max_angle) or max_angle <= 0.0:
            continue
        prev = axis[t - 1] / max(float(np.linalg.norm(axis[t - 1])), 1e-12)
        cur = axis[t] / max(float(np.linalg.norm(axis[t])), 1e-12)
        dot = float(np.clip(np.dot(prev, cur), -1.0, 1.0))
        angle = float(np.arccos(dot))
        if angle > max_angle:
            tangent = cur - dot * prev
            if float(np.linalg.norm(tangent)) <= 1e-10:
                continue
            tangent = tangent / float(np.linalg.norm(tangent))
            axis[t] = np.cos(max_angle) * prev + np.sin(max_angle) * tangent
    axis[-1] = np.asarray(fixed_goal, dtype=float) / max(float(np.linalg.norm(fixed_goal)), 1e-12)
    for t in range(axis.shape[0] - 2, -1, -1):
        max_angle = float(edge_max_angles[t + 1])
        if not np.isfinite(max_angle) or max_angle <= 0.0:
            continue
        nxt = axis[t + 1] / max(float(np.linalg.norm(axis[t + 1])), 1e-12)
        cur = axis[t] / max(float(np.linalg.norm(axis[t])), 1e-12)
        dot = float(np.clip(np.dot(nxt, cur), -1.0, 1.0))
        angle = float(np.arccos(dot))
        if angle > max_angle:
            tangent = cur - dot * nxt
            if float(np.linalg.norm(tangent)) <= 1e-10:
                continue
            tangent = tangent / float(np.linalg.norm(tangent))
            axis[t] = np.cos(max_angle) * nxt + np.sin(max_angle) * tangent
    axis[:] = _unit_rows(axis)


def _resolve_optimizer_endpoints(env: S5SphereInspectEnv, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(seed))
    demo_traj, _demo_cutpoints = env.generate_demo(rng=rng)
    demo_traj = np.asarray(demo_traj, dtype=float)
    if demo_traj.ndim != 2 or demo_traj.shape[0] < 2 or demo_traj.shape[1] != 3:
        raise ValueError("S5 endpoint demo generation did not return a valid (T,3) trajectory.")
    return demo_traj[0].copy(), demo_traj[-1].copy()


def _horizontal_camera_direction(camera_yaw: float) -> np.ndarray:
    yaw = np.deg2rad(float(camera_yaw))
    direction = np.asarray([np.sin(yaw), -np.cos(yaw), 0.0], dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.asarray([1.0, 0.0, 0.0], dtype=float)
    return direction / norm


def _initial_axis_trace_from_endpoints(start: np.ndarray, goal: np.ndarray, T: int) -> np.ndarray:
    start = np.asarray(start, dtype=float).reshape(3)
    goal = np.asarray(goal, dtype=float).reshape(3)
    direction = goal - start
    if float(np.linalg.norm(direction)) <= 1e-10:
        direction = np.array([1.0, 0.0, 0.0], dtype=float)
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    return np.repeat(direction[None, :], int(max(int(T), 1)), axis=0)


def _plan_s5_stage_constraint_optimizer(
    env: S5SphereInspectEnv,
    scene: dict,
    constraint_values: dict,
    *,
    seed: int,
    stage_lengths: dict | None,
    stage_length_source: str,
    segmentation_path: str | Path | None,
    speed_safety: float,
    optimizer_iters: int,
    smooth_step: float,
    constraint_step: float,
    objective_step: float,
    global_speed_max: float | None,
    stage_length_scale: float,
    stage4_length_multiplier: float,
    start_xyz: tuple[float, float, float] | None = None,
    goal_xyz: tuple[float, float, float] | None = None,
) -> dict:
    """Direct S5 optimizer using endpoints, stage schedule, features, and constraints."""
    start, goal = _resolve_optimizer_endpoints(env, seed)
    endpoint_source = "seeded_demo_endpoints_only"
    if start_xyz is not None:
        start = np.asarray(start_xyz, dtype=float).reshape(3)
        endpoint_source = "explicit_start_seeded_goal"
    if goal_xyz is not None:
        goal = np.asarray(goal_xyz, dtype=float).reshape(3)
        endpoint_source = "explicit_goal" if start_xyz is None else "explicit_start_goal"
    lengths, length_info = _auto_stage_lengths(
        env,
        constraint_values,
        overrides=stage_lengths,
        stage_length_source=stage_length_source,
        segmentation_path=segmentation_path,
        start=start,
        goal=goal,
        speed_safety=speed_safety,
        global_speed_max=global_speed_max,
        stage_length_scale=stage_length_scale,
        stage4_length_multiplier=stage4_length_multiplier,
    )
    labels = np.repeat(np.arange(5), lengths)
    T = int(labels.size)
    cutpoints = np.where(np.diff(labels) != 0)[0].astype(int)

    reference = np.linspace(start, goal, T)
    traj = reference.copy()
    axis = _initial_axis_trace_from_endpoints(start, goal, T)

    edge_max_steps = np.full(T, np.inf, dtype=float)
    edge_max_angles = np.full(T, np.inf, dtype=float)
    preserve_position = np.zeros(T, dtype=bool)
    speed_safety = float(np.clip(speed_safety, 0.10, 1.0))
    if global_speed_max is not None and np.isfinite(float(global_speed_max)) and float(global_speed_max) > 0.0:
        edge_max_steps[:] = float(global_speed_max) * speed_safety * float(env.dt)
    for stage_idx in range(5):
        stage_no = stage_idx + 1
        if f"s{stage_no}:surf_dist" in dict(constraint_values or {}):
            preserve_position[labels == stage_idx] = True
        speed_key = f"s{stage_no}:speed"
        if speed_key in dict(constraint_values or {}):
            value = float(constraint_values[speed_key])
            if np.isfinite(value) and value > 0.0:
                edge_max_steps[labels == stage_idx] = min(
                    float(value) * speed_safety * float(env.dt),
                    float(np.nanmax(edge_max_steps[labels == stage_idx])),
                )
        ang_key = f"s{stage_no}:ang_speed"
        if ang_key in dict(constraint_values or {}):
            value = float(constraint_values[ang_key])
            if np.isfinite(value) and value > 0.0:
                edge_max_angles[labels == stage_idx] = min(
                    float(value) * speed_safety * float(env.dt),
                    float(np.nanmax(edge_max_angles[labels == stage_idx])),
                )

    alpha = float(np.clip(constraint_step, 0.0, 1.0))
    beta = float(np.clip(smooth_step, 0.0, 0.45))
    gamma = float(np.clip(objective_step, 0.0, 0.35))
    for _ in range(max(int(optimizer_iters), 0)):
        if T > 2 and beta > 0.0:
            traj[1:-1] += beta * (traj[:-2] + traj[2:] - 2.0 * traj[1:-1])
            axis[1:-1] += beta * (axis[:-2] + axis[2:] - 2.0 * axis[1:-1])
        if T > 2 and gamma > 0.0:
            traj[1:-1] += gamma * (reference[1:-1] - traj[1:-1])

        _limit_edge_speeds(traj, edge_max_steps, fixed_start=start, fixed_goal=goal)

        for stage_idx in range(5):
            mask = labels == stage_idx
            if not np.any(mask):
                continue
            stage_no = stage_idx + 1
            surf_key = f"s{stage_no}:surf_dist"
            if surf_key in dict(constraint_values or {}):
                surf = _constraint_value(constraint_values, surf_key, 0.0)
                target = _project_to_sphere_offset(env, traj[mask], surf)
                traj[mask] = (1.0 - alpha) * traj[mask] + alpha * target

            normals_stage = _unit_rows(traj[mask] - env.sphere_center[None, :])
            normal_key = f"s{stage_no}:normal_err"
            if normal_key in dict(constraint_values or {}):
                bound = _constraint_value(constraint_values, normal_key, float(env.tool_align_max_stage2))
                constrained_axis = _apply_normal_error_bound(axis[mask], normals_stage, bound)
                axis[mask] = (1.0 - alpha) * axis[mask] + alpha * constrained_axis

            start_key = f"s{stage_no}:start_dist"
            if start_key in dict(constraint_values or {}):
                target_dist = max(_constraint_value(constraint_values, start_key, 0.0), 0.0)
                rel = traj[mask] - start[None, :]
                dirs = _unit_rows(rel)
                target = start[None, :] + target_dist * dirs
                traj[mask] = (1.0 - alpha) * traj[mask] + alpha * target

            goal_key = f"s{stage_no}:goal_dist"
            if goal_key in dict(constraint_values or {}):
                target_dist = max(_constraint_value(constraint_values, goal_key, 0.0), 0.0)
                rel = traj[mask] - env.goal[None, :]
                dirs = _unit_rows(rel)
                target = env.goal[None, :] + target_dist * dirs
                traj[mask] = (1.0 - alpha) * traj[mask] + alpha * target

        _limit_edge_speeds_preserve(
            traj,
            edge_max_steps,
            preserve_position,
            fixed_start=start,
            fixed_goal=goal,
        )
        axis = _unit_rows(axis)
        _limit_axis_angular_speed(axis, edge_max_angles, fixed_start=axis[0], fixed_goal=axis[-1])
        traj[0] = start
        traj[-1] = goal

    normals = _unit_rows(traj - env.sphere_center[None, :])
    for stage_idx in range(5):
        surf_key = f"s{stage_idx + 1}:surf_dist"
        if surf_key in dict(constraint_values or {}):
            mask = labels == stage_idx
            traj[mask] = _project_to_sphere_offset(
                env,
                traj[mask],
                _constraint_value(constraint_values, surf_key, 0.0),
            )
        normal_key = f"s{stage_idx + 1}:normal_err"
        if normal_key in dict(constraint_values or {}):
            mask = labels == stage_idx
            axis[mask] = _apply_normal_error_bound(
                axis[mask],
                _unit_rows(traj[mask] - env.sphere_center[None, :]),
                _constraint_value(constraint_values, normal_key, float(env.tool_align_max_stage2)),
            )
    _limit_edge_speeds_preserve(
        traj,
        edge_max_steps,
        preserve_position,
        fixed_start=start,
        fixed_goal=goal,
    )
    traj, axis, labels, lengths = _resample_stages_for_speed_cap(
        traj,
        axis,
        labels,
        constraint_values,
        env=env,
        speed_safety=speed_safety,
        global_speed_max=global_speed_max,
    )
    traj[0] = start
    traj[-1] = goal
    cutpoints = np.where(np.diff(labels) != 0)[0].astype(int)
    preserve_position = np.zeros(len(traj), dtype=bool)
    for stage_idx in range(5):
        if f"s{stage_idx + 1}:surf_dist" in dict(constraint_values or {}):
            preserve_position[labels == stage_idx] = True
    edge_max_steps = np.full(len(traj), np.inf, dtype=float)
    if global_speed_max is not None and np.isfinite(float(global_speed_max)) and float(global_speed_max) > 0.0:
        edge_max_steps[:] = float(global_speed_max) * float(np.clip(speed_safety, 0.10, 1.0)) * float(env.dt)
    for stage_idx in range(5):
        key = f"s{stage_idx + 1}:speed"
        if key in dict(constraint_values or {}):
            value = float(constraint_values[key])
            if np.isfinite(value) and value > 0.0:
                edge_max_steps[labels == stage_idx] = np.minimum(
                    edge_max_steps[labels == stage_idx],
                    float(value) * float(np.clip(speed_safety, 0.10, 1.0)) * float(env.dt),
                )
        surf_key = f"s{stage_idx + 1}:surf_dist"
        if surf_key in dict(constraint_values or {}):
            mask = labels == stage_idx
            traj[mask] = _project_to_sphere_offset(env, traj[mask], _constraint_value(constraint_values, surf_key, 0.0))
        normal_key = f"s{stage_idx + 1}:normal_err"
        if normal_key in dict(constraint_values or {}):
            mask = labels == stage_idx
            axis[mask] = _apply_normal_error_bound(
                axis[mask],
                _unit_rows(traj[mask] - env.sphere_center[None, :]),
                _constraint_value(constraint_values, normal_key, float(env.tool_align_max_stage2)),
            )
    _limit_edge_speeds_preserve(
        traj,
        edge_max_steps,
        preserve_position,
        fixed_start=start,
        fixed_goal=goal,
    )
    axis = _unit_rows(axis)
    env.register_tool_axis_trace(traj, axis)
    return {
        "trajectory": np.asarray(traj, dtype=float),
        "tool_axis": np.asarray(axis, dtype=float),
        "true_cutpoints": cutpoints.astype(int),
        "rollout_backend": "direct_constraint_plan",
        "observation_backend": "analytic_raw",
        "planner": "s5_direct_stage_constraint_optimizer",
        "constraint_values": dict(constraint_values or {}),
        "global_speed_max": None if global_speed_max is None else float(global_speed_max),
        "stage_lengths": {f"stage{i + 1}": int(n) for i, n in enumerate(lengths)},
        "stage_length_info": dict(length_info),
        "scene": dict(scene or {}),
        "seed": int(seed),
        "start": start.tolist(),
        "goal": goal.tolist(),
        "endpoint_source": endpoint_source,
    }


def _feature_names(feature_schema: list[dict], dim: int) -> list[str]:
    names = [f"feature_{idx}" for idx in range(int(dim))]
    for idx, item in enumerate(feature_schema or []):
        col = int(item.get("column_idx", item.get("id", idx)))
        if 0 <= col < len(names):
            names[col] = str(item.get("name", names[col]))
    return names


def _stage_spans(cutpoints: list[int], length: int) -> list[tuple[int, int]]:
    cuts = [int(v) for v in cutpoints if 0 <= int(v) < int(length) - 1]
    ends = cuts + [int(length) - 1]
    starts = [0] + [int(v) + 1 for v in ends[:-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _plot_all_feature_profiles(
    *,
    features: np.ndarray,
    planned_features: np.ndarray,
    feature_schema: list[dict],
    cutpoints: list[int],
    constraint_payload: dict,
    constraint_values: dict,
    env: S5SphereInspectEnv,
    output_path: str | Path,
    title: str,
) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot S5 feature profiles.")

    F = np.asarray(features, dtype=float)
    F_plan = np.asarray(planned_features, dtype=float)
    if F.ndim != 2 or F_plan.ndim != 2:
        raise ValueError("features and planned_features must have shape (T, D).")
    dim = int(max(F.shape[1], F_plan.shape[1]))
    names = _feature_names(feature_schema, dim)
    spans = _stage_spans(cutpoints, max(len(F), len(F_plan)))
    specs = list(constraint_payload.get("constraint_specs") or env.get_constraint_specs())
    true_constraints = dict(constraint_payload.get("true_constraints") or env.true_constraints)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(7.0, 1.55 * dim)
    fig, axes = plt.subplots(dim, 1, figsize=(11.5, fig_height), sharex=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    t = np.arange(F.shape[0])
    t_plan = np.arange(F_plan.shape[0])

    true_label_used = False
    learned_label_used = False
    for feat_idx, ax in enumerate(axes):
        if feat_idx < F_plan.shape[1]:
            ax.plot(t_plan, F_plan[:, feat_idx], color="#D97706", linewidth=1.35, label="planned/reference")
        if feat_idx < F.shape[1]:
            ax.plot(t, F[:, feat_idx], color="#1D4ED8", linewidth=1.55, label="pybullet executed")
        for cp in cutpoints:
            if 0 <= int(cp) < max(len(F), len(F_plan)):
                ax.axvline(int(cp), color="#9CA3AF", linestyle="--", linewidth=0.9, alpha=0.75)

        feat_name = names[feat_idx]
        for spec in specs:
            if str(spec.get("feature_name", "")) != feat_name:
                continue
            stage_idx = int(spec.get("stage", -1))
            if stage_idx < 0 or stage_idx >= len(spans):
                continue
            x0, x1 = spans[stage_idx]
            oracle_key = str(spec.get("oracle_key", ""))
            if oracle_key in true_constraints:
                label = "true target/bound" if not true_label_used else None
                ax.hlines(
                    float(true_constraints[oracle_key]),
                    x0,
                    x1,
                    colors="#111827",
                    linestyles="--",
                    linewidth=1.25,
                    label=label,
                )
                true_label_used = True
            learned_key = f"s{stage_idx + 1}:{feat_name}"
            if learned_key in constraint_values:
                label = "planned constraint" if not learned_label_used else None
                ax.hlines(
                    float(constraint_values[learned_key]),
                    x0,
                    x1,
                    colors="#7C3AED",
                    linestyles=":",
                    linewidth=1.45,
                    label=label,
                )
                learned_label_used = True

        ax.set_ylabel(feat_name, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.20)

    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xlabel("t")
    fig.suptitle(str(title), fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def render_s5_planned_trajectory(
    *,
    constraints_json: str | Path,
    outdir: str | Path,
    seed: int,
    gui: int,
    fps: float,
    width: int,
    height: int,
    render_frame_stride: int,
    realtime: bool,
    gui_hold_seconds: float | None,
    camera_yaw: float,
    camera_pitch: float,
    camera_distance: float,
    camera_fov: float,
    camera_target_offset: tuple[float, float, float],
    hide_gripper: bool,
    draw_tool_bar: bool,
    tool_bar_length: float,
    tool_bar_radius: float,
    draw_stage_trace: bool,
    draw_executed_trace: bool,
    trace_stride: int,
    trace_width: float,
    draw_current_marker: bool,
    plot_features: bool,
    feature_overlay: bool,
    no_precheck: bool,
    no_filter: bool,
    constraint_source: str,
    plan_dt: float,
    speed_safety: float,
    stage_lengths: dict | None,
    benchmark_method: str | None,
    benchmark_dataset: str | None,
    benchmark_method_seed: int | None,
    execution_joint_noise_std: float,
    execution_joint_noise_smooth: float,
    execution_noise_seed: int | None,
    planner: str,
    optimizer_iters: int,
    optimizer_smooth_step: float,
    optimizer_constraint_step: float,
    optimizer_objective_step: float,
    strict_ik_filter: bool,
    stage_length_source: str,
    global_speed_max: float | None,
    stage_length_scale: float,
    stage4_length_multiplier: float,
    save_frame_indices: list[int],
    start_xyz: tuple[float, float, float] | None,
    goal_xyz: tuple[float, float, float] | None,
    goal_camera_offset: float,
    output_prefix: str = "s5_planned",
    video_path_override: str | Path | None = None,
) -> dict:
    out_dir = Path(outdir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = str(output_prefix or "s5_planned")

    cfg = _apply_default_s5_loader_config({})
    cfg["rollout_backend"] = "pybullet"
    cfg["observation_backend"] = "pybullet"
    cfg["eval_tag"] = "S5SphereInspectPlannedRender"
    if float(plan_dt) > 0.0:
        cfg["dt"] = float(plan_dt)
    if bool(no_precheck):
        cfg["pybullet_precheck_ik_waypoints"] = False
    if bool(no_filter):
        cfg["pybullet_filter_ik_valid"] = False
    env = S5SphereInspectEnv(**cfg)

    raw_payload, resolved_constraints_path = _load_constraint_payload(constraints_json)
    default_segmentation_path = (
        resolved_constraints_path.with_name("segmentation.json")
        if resolved_constraints_path.name == "constraints.json"
        else None
    )
    payload = _select_constraint_payload(
        raw_payload,
        method=benchmark_method,
        dataset=benchmark_dataset,
        method_seed=benchmark_method_seed,
    )
    raw_constraint_values = _constraint_values_from_payload(payload, env, constraint_source=str(constraint_source))
    feature_names = list(payload.get("ConstraintFeatureNames", [])) or [
        "surf_dist", "normal_err", "speed", "ang_speed", "start_dist", "goal_dist"
    ]
    constraint_specs_for_clearance = _constraint_specs_from_payload(
        payload,
        env,
        constraint_source=str(constraint_source),
        feature_names=feature_names,
    )
    constraint_values = apply_inequality_constraint_clearance(
        raw_constraint_values,
        constraint_specs_for_clearance,
        upper_scale=0.96,
        lower_scale=1.04,
    )

    scene = env.sample_scene()
    resolved_start_xyz = None if start_xyz is None else tuple(float(v) for v in start_xyz)
    resolved_goal_xyz = None if goal_xyz is None else tuple(float(v) for v in goal_xyz)
    goal_offset_vec = np.zeros(3, dtype=float)
    if resolved_goal_xyz is None and float(goal_camera_offset) != 0.0:
        default_start, default_goal = _resolve_optimizer_endpoints(env, seed)
        goal_offset_vec = _horizontal_camera_direction(camera_yaw) * float(goal_camera_offset)
        resolved_goal_xyz = tuple((np.asarray(default_goal, dtype=float) + goal_offset_vec).tolist())
    if str(planner).lower() == "optimizer":
        planned = _plan_s5_stage_constraint_optimizer(
            env,
            scene,
            constraint_values,
            seed=int(seed),
            stage_lengths=stage_lengths,
            stage_length_source=str(stage_length_source),
            segmentation_path=default_segmentation_path,
            speed_safety=float(speed_safety),
            optimizer_iters=int(optimizer_iters),
            smooth_step=float(optimizer_smooth_step),
            constraint_step=float(optimizer_constraint_step),
            objective_step=float(optimizer_objective_step),
            global_speed_max=global_speed_max,
            stage_length_scale=float(stage_length_scale),
            stage4_length_multiplier=float(stage4_length_multiplier),
            start_xyz=resolved_start_xyz,
            goal_xyz=resolved_goal_xyz,
        )
    else:
        if resolved_start_xyz is not None or resolved_goal_xyz is not None:
            raise ValueError("S5 endpoint overrides are currently supported only with --planner optimizer.")
        planned = env.plan_episode_from_constraints(
            scene,
            constraint_values,
            seed=int(seed),
            stage_lengths=stage_lengths,
            speed_safety=float(speed_safety),
        )
    print(f"[plan] points={len(planned['trajectory'])}, cutpoints={planned['true_cutpoints'].tolist()}")
    print(f"[plan] planner={planned.get('planner', planner)}")
    print(f"[plan] constraints={planned['constraint_values']}")
    print(f"[plan] inequality_clearance={{'upper_scale': 0.96, 'lower_scale': 1.04}}")
    print(f"[plan] endpoints start={planned.get('start')} goal={planned.get('goal')} source={planned.get('endpoint_source')}")

    strict_execution_checks = bool(strict_ik_filter) or str(planner).lower() != "optimizer"
    latent = env.execute_plan_pybullet(
        scene,
        planned,
        precheck=bool(strict_execution_checks) and not bool(no_precheck),
        filter_valid=bool(strict_execution_checks) and not bool(no_filter),
        execution_joint_noise_std=float(execution_joint_noise_std),
        execution_joint_noise_smooth=float(execution_joint_noise_smooth),
        execution_noise_seed=execution_noise_seed,
    )
    obs = env.compute_observation(latent, scene)

    planned_features = env.compute_all_features_matrix(
        np.asarray(planned["trajectory"], dtype=float),
        tool_axis=np.asarray(planned["tool_axis"], dtype=float),
        use_cached=False,
    )
    feature_plot_path = None
    cutpoints = [int(v) for v in np.asarray(planned["true_cutpoints"], dtype=int).reshape(-1).tolist()]
    if bool(plot_features):
        feature_plot_path = _plot_all_feature_profiles(
            features=np.asarray(obs["features"], dtype=float),
            planned_features=np.asarray(planned_features, dtype=float),
            feature_schema=list(obs["feature_schema"]),
            cutpoints=cutpoints,
            constraint_payload=payload,
            constraint_values=dict(planned["constraint_values"]),
            env=env,
            output_path=out_dir / f"{output_prefix}_features.png",
            title="S5 planned trajectory all-feature profiles",
        )

    np.savez_compressed(
        out_dir / f"{output_prefix}_rollout.npz",
        planned_trajectory=np.asarray(planned["trajectory"], dtype=float),
        planned_tool_axis=np.asarray(planned["tool_axis"], dtype=float),
        executed_trajectory=np.asarray(obs["trajectory"], dtype=float),
        executed_tool_axis=np.asarray(obs["tool_axis"], dtype=float),
        planned_features=np.asarray(planned_features, dtype=float),
        executed_features=np.asarray(obs["features"], dtype=float),
        cutpoints=np.asarray(planned["true_cutpoints"], dtype=int),
        joint_position_commands=np.asarray(obs.get("joint_position_commands", []), dtype=float),
        joint_position_commands_nominal=np.asarray(obs.get("joint_position_commands_nominal", []), dtype=float),
        execution_joint_noise=np.asarray(obs.get("execution_joint_noise", []), dtype=float),
        ik_position_error_world=np.asarray(obs.get("ik_position_error_world", []), dtype=float),
        ik_axis_error=np.asarray(obs.get("ik_axis_error", []), dtype=float),
    )

    output_path = None
    if int(gui) == 1:
        output_path = Path(video_path_override) if video_path_override is not None else out_dir / f"{output_prefix}_pybullet.mp4"
    effective_realtime = bool(realtime) or int(gui) == 2
    effective_hold_seconds = (-1.0 if int(gui) == 2 else 2.0) if gui_hold_seconds is None else float(gui_hold_seconds)
    render_summary = env.render_episode(
        scene,
        np.asarray(obs["trajectory"], dtype=float),
        output_path,
        backend="pybullet_video",
        cutpoints=cutpoints,
        tool_axis=np.asarray(obs["tool_axis"], dtype=float),
        joint_positions=obs.get("joint_positions"),
        title="S5 planned trajectory",
        gui=int(gui),
        fps=float(fps),
        width=int(width),
        height=int(height),
        render_frame_stride=int(render_frame_stride),
        realtime=bool(effective_realtime),
        gui_hold_seconds=float(effective_hold_seconds),
        camera_yaw=float(camera_yaw),
        camera_pitch=float(camera_pitch),
        camera_distance=float(camera_distance),
        camera_target=np.asarray(env.pybullet_world_center, dtype=float) + np.asarray(camera_target_offset, dtype=float),
        camera_fov=float(camera_fov),
        stage4_shell_offset=float(planned["constraint_values"].get("s4:surf_dist", env.get_true_constraints()["surface_near_target"])),
        hide_gripper=bool(hide_gripper),
        draw_tool_bar=bool(draw_tool_bar),
        tool_bar_length=float(tool_bar_length),
        tool_bar_radius=float(tool_bar_radius),
        draw_stage_trace=bool(draw_stage_trace),
        draw_executed_trace=bool(draw_executed_trace),
        trace_stride=int(trace_stride),
        trace_width=float(trace_width),
        draw_current_marker=bool(draw_current_marker),
        feature_overlay=bool(feature_overlay),
        feature_overlay_features=np.asarray(obs["features"], dtype=float),
        feature_overlay_names=_feature_names(list(obs["feature_schema"]), np.asarray(obs["features"], dtype=float).shape[1]),
        feature_overlay_specs=list(payload.get("constraint_specs") or env.get_constraint_specs()),
        feature_overlay_true_constraints=dict(payload.get("true_constraints") or env.true_constraints),
        feature_overlay_title=(
            "Executed trajectory feature profile (planned with learned constraints)"
            if str(constraint_source).lower() == "learned"
            else "Executed trajectory feature profile (planned with Ground truth constraints)"
        ),
        save_frame_indices=save_frame_indices,
        save_frame_dir=out_dir,
        save_frame_prefix=output_prefix,
    )

    violation_stats = constraint_violation_stats(
        features_list=[np.asarray(planned_features, dtype=float)],
        cutpoints_list=[cutpoints],
        feature_schema=env.get_feature_schema(),
        constraint_specs=env.get_constraint_specs(),
        true_constraints=env.get_true_constraints(),
        equality_tolerance=1e-4,
    )
    executed_violation_stats = constraint_violation_stats(
        features_list=[np.asarray(obs["features"], dtype=float)],
        cutpoints_list=[cutpoints],
        feature_schema=env.get_feature_schema(),
        constraint_specs=env.get_constraint_specs(),
        true_constraints=env.get_true_constraints(),
        equality_tolerance=1e-4,
    )

    summary = {
        "task": "s5_planned_trajectory_render",
        "constraints_json": str(Path(constraints_json)),
        "resolved_constraints_payload": str(resolved_constraints_path),
        "constraint_source": str(constraint_source),
        "plan_dt": float(env.dt),
        "seed": int(seed),
        "gui": int(gui),
        "raw_constraint_values": dict(raw_constraint_values),
        "constraint_values": dict(planned["constraint_values"]),
        "inequality_clearance": {"upper_scale": 0.96, "lower_scale": 1.04},
        "global_speed_max": None if planned.get("global_speed_max") is None else float(planned.get("global_speed_max")),
        "planner": str(planned.get("planner", planner)),
        "stage_lengths": dict(planned["stage_lengths"]),
        "stage_length_info": dict(planned.get("stage_length_info", {})),
        "start": list(planned.get("start", [])),
        "goal": list(planned.get("goal", [])),
        "endpoint_source": str(planned.get("endpoint_source", "")),
        "goal_camera_offset": float(goal_camera_offset),
        "goal_camera_offset_vector": goal_offset_vec.tolist(),
        "cutpoints": cutpoints,
        "trajectory_points": int(len(planned["trajectory"])),
        "feature_plot": None if feature_plot_path is None else str(Path(feature_plot_path).resolve()),
        "rollout_npz": str((out_dir / f"{output_prefix}_rollout.npz").resolve()),
        "video": render_summary,
        "saved_frames": list(render_summary.get("saved_frames", [])),
        "ik_filter": dict(latent.get("ik_filter", {})),
        "feature_overlay": bool(feature_overlay),
        "execution_joint_noise_std": float(execution_joint_noise_std),
        "execution_joint_noise_smooth": float(execution_joint_noise_smooth),
        "execution_noise_seed": None if execution_noise_seed is None else int(execution_noise_seed),
        "strict_ik_filter": bool(strict_execution_checks),
        "ik_position_error_mean": None if "ik_position_error_world" not in obs else float(np.mean(obs["ik_position_error_world"])),
        "ik_position_error_max": None if "ik_position_error_world" not in obs else float(np.max(obs["ik_position_error_world"])),
        "ik_axis_error_mean": None if "ik_axis_error" not in obs else float(np.mean(obs["ik_axis_error"])),
        "ik_axis_error_max": None if "ik_axis_error" not in obs else float(np.max(obs["ik_axis_error"])),
        "planned_constraint_violation": violation_stats,
        "constraint_violation": violation_stats,
        "executed_constraint_violation": executed_violation_stats,
    }
    summary_path = out_dir / f"{output_prefix}_render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {summary_path}")
    print(f"[saved] features={feature_plot_path}, video={render_summary.get('video_path')}")
    print_render_violation_rates(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan an S5 trajectory from learned constraints and render PyBullet execution.")
    parser.add_argument("--constraints-json", required=True, help="Path to constraints.json or benchmark_results.json.")
    parser.add_argument("--benchmark-method", default=None, help="Method row to select when --constraints-json is benchmark_results.json.")
    parser.add_argument("--benchmark-dataset", default=None, help="Dataset row to select when --constraints-json is benchmark_results.json.")
    parser.add_argument("--benchmark-method-seed", type=int, default=None, help="Method seed row to select from benchmark_results.json.")
    parser.add_argument("--outdir", default="outputs/s5_planned_render")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plan-seeds", default=None, help="Comma-separated seeds for rendering multiple planned trajectories.")
    parser.add_argument("--n-plans", type=int, default=1, help="Number of planned trajectories to render, starting from --seed, when --plan-seeds is not set.")
    parser.add_argument("--start-xyz", default=None, help="Optional S5-space start point override as x,y,z. Defaults to the current seeded start.")
    parser.add_argument("--goal-xyz", default=None, help="Optional S5-space goal point override as x,y,z.")
    parser.add_argument("--goal-camera-offset", type=float, default=0.04, help="If --goal-xyz is omitted, move the seeded goal this far horizontally toward the camera.")
    parser.add_argument("--gui", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=1360)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--render-frame-stride", type=int, default=1)
    parser.add_argument("--realtime", type=int, default=0)
    parser.add_argument("--gui-hold-seconds", type=float, default=None)
    parser.add_argument("--camera-yaw", type=float, default=90.0)
    parser.add_argument("--camera-pitch", type=float, default=-16.0)
    parser.add_argument("--camera-distance", type=float, default=1.35)
    parser.add_argument("--camera-fov", type=float, default=38.0)
    parser.add_argument(
        "--camera-target-offset",
        default="0.00,0.24,0.20",
        help="World-frame camera target offset from pybullet_world_center, e.g. '0.10,0,0.04'.",
    )
    parser.add_argument("--hide-gripper", type=int, default=1)
    parser.add_argument("--draw-tool-bar", type=int, default=1)
    parser.add_argument("--tool-bar-length", type=float, default=0.205)
    parser.add_argument("--tool-bar-radius", type=float, default=0.005)
    parser.add_argument("--draw-stage-trace", type=int, default=0)
    parser.add_argument("--draw-executed-trace", type=int, default=1)
    parser.add_argument("--trace-stride", type=int, default=1)
    parser.add_argument("--trace-width", type=float, default=3.0)
    parser.add_argument("--draw-current-marker", type=int, default=0)
    parser.add_argument("--plot-features", type=int, default=1)
    parser.add_argument("--feature-overlay", type=int, default=1)
    parser.add_argument("--save-frame-indices", default=None, help="Comma-separated source frame indices to save as PNGs in --outdir, e.g. 0,80,157.")
    parser.add_argument("--no-precheck", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--constraint-source", choices=["learned", "target"], default="learned")
    parser.add_argument("--fallback-target", type=int, default=None, help="Deprecated; use --constraint-source target.")
    parser.add_argument("--plan-dt", type=float, default=0.25, help="Planning sample interval used for S5 feature/speed constraints.")
    parser.add_argument("--speed-safety", type=float, default=1.0)
    parser.add_argument("--stage-lengths", default=None, help="Optional comma list, e.g. stage2:34,stage4:18.")
    parser.add_argument(
        "--stage-length-source",
        choices=["learned-ratio", "fixed"],
        default="learned-ratio",
        help="Stage lengths for optimizer plans. learned-ratio uses segmentation.json ratios and auto total length.",
    )
    parser.add_argument(
        "--stage-length-scale",
        type=float,
        default=1.0,
        help="Scale applied to the learned-ratio mean demonstration length.",
    )
    parser.add_argument(
        "--stage4-length-multiplier",
        type=float,
        default=3.0,
        help="Multiplier applied to the automatically selected stage 4 length for S5 planned render.",
    )
    parser.add_argument("--execution-joint-noise-std", type=float, default=0.0002)
    parser.add_argument("--execution-joint-noise-smooth", type=float, default=0.90)
    parser.add_argument("--execution-noise-seed", type=int, default=None)
    parser.add_argument("--planner", choices=["waypoint", "optimizer"], default="optimizer")
    parser.add_argument("--optimizer-iters", type=int, default=500)
    parser.add_argument("--optimizer-smooth-step", type=float, default=0.18)
    parser.add_argument("--optimizer-constraint-step", type=float, default=0.45)
    parser.add_argument("--optimizer-objective-step", type=float, default=0.08)
    parser.add_argument(
        "--global-speed-max",
        type=float,
        default=0.013,
        help="Global speed upper bound for optimizer plans. Use <=0 to disable.",
    )
    parser.add_argument(
        "--strict-ik-filter",
        type=int,
        default=0,
        help="1 to hard-fail optimizer plans on S5 PyBullet precheck/filter. Waypoint planner remains strict by default.",
    )
    args = parser.parse_args()
    constraint_source = str(args.constraint_source)
    if args.fallback_target is not None and int(args.fallback_target):
        constraint_source = "target"

    seeds = plan_seed_list(int(args.seed), args.plan_seeds, int(args.n_plans))
    out_dir = Path(args.outdir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    temp_videos = []
    multi = len(seeds) > 1
    for plan_idx, plan_seed in enumerate(seeds):
        prefix = "s5_planned" if not multi else f"s5_planned_seed_{int(plan_seed):03d}"
        video_override = None
        if multi and int(args.gui) == 1:
            video_override = out_dir / f"._tmp_{prefix}_pybullet.mp4"
        summary = render_s5_planned_trajectory(
            constraints_json=args.constraints_json,
            outdir=args.outdir,
            seed=int(plan_seed),
            gui=int(args.gui),
            fps=float(args.fps),
            width=int(args.width),
            height=int(args.height),
            render_frame_stride=int(args.render_frame_stride),
            realtime=bool(args.realtime),
            gui_hold_seconds=args.gui_hold_seconds,
            camera_yaw=float(args.camera_yaw),
            camera_pitch=float(args.camera_pitch),
            camera_distance=float(args.camera_distance),
            camera_fov=float(args.camera_fov),
            camera_target_offset=_parse_vec3(args.camera_target_offset),
            hide_gripper=bool(args.hide_gripper),
            draw_tool_bar=bool(args.draw_tool_bar),
            tool_bar_length=float(args.tool_bar_length),
            tool_bar_radius=float(args.tool_bar_radius),
            draw_stage_trace=bool(args.draw_stage_trace),
            draw_executed_trace=bool(args.draw_executed_trace),
            trace_stride=int(args.trace_stride),
            trace_width=float(args.trace_width),
            draw_current_marker=bool(args.draw_current_marker),
            plot_features=bool(args.plot_features),
            feature_overlay=bool(args.feature_overlay),
            no_precheck=bool(args.no_precheck),
            no_filter=bool(args.no_filter),
            constraint_source=constraint_source,
            plan_dt=float(args.plan_dt),
            speed_safety=float(args.speed_safety),
            stage_lengths=_parse_stage_lengths(args.stage_lengths),
            benchmark_method=args.benchmark_method,
            benchmark_dataset=args.benchmark_dataset,
            benchmark_method_seed=args.benchmark_method_seed,
            execution_joint_noise_std=float(args.execution_joint_noise_std),
            execution_joint_noise_smooth=float(args.execution_joint_noise_smooth),
            execution_noise_seed=(None if args.execution_noise_seed is None else int(args.execution_noise_seed) + int(plan_idx)),
            planner=str(args.planner),
            optimizer_iters=int(args.optimizer_iters),
            optimizer_smooth_step=float(args.optimizer_smooth_step),
            optimizer_constraint_step=float(args.optimizer_constraint_step),
            optimizer_objective_step=float(args.optimizer_objective_step),
            strict_ik_filter=bool(args.strict_ik_filter),
            stage_length_source=str(args.stage_length_source),
            global_speed_max=(None if float(args.global_speed_max) <= 0.0 else float(args.global_speed_max)),
            stage_length_scale=float(args.stage_length_scale),
            stage4_length_multiplier=float(args.stage4_length_multiplier),
            save_frame_indices=_parse_frame_indices(args.save_frame_indices),
            start_xyz=_parse_optional_vec3(args.start_xyz),
            goal_xyz=_parse_optional_vec3(args.goal_xyz),
            goal_camera_offset=float(args.goal_camera_offset),
            output_prefix=prefix,
            video_path_override=video_override,
        )
        summaries.append(summary)
        if video_override is not None:
            temp_videos.append(Path(video_override))

    if multi:
        final_video = None
        if int(args.gui) == 1 and temp_videos:
            final_video = concat_mp4_files(temp_videos, out_dir / "s5_planned_pybullet.mp4")
            for path in temp_videos:
                try:
                    path.unlink()
                except OSError:
                    pass
        features_list = []
        executed_features_list = []
        cutpoints_list = []
        for item in summaries:
            rollout_path = item.get("rollout_npz")
            if rollout_path and Path(rollout_path).exists():
                z = np.load(rollout_path)
                features_list.append(np.asarray(z["planned_features"], dtype=float))
                executed_features_list.append(np.asarray(z["executed_features"], dtype=float))
                cutpoints_list.append(np.asarray(z["cutpoints"], dtype=int))
        env_for_stats = S5SphereInspectEnv(**_apply_default_s5_loader_config({}))
        aggregate = {
            "task": "s5_planned_trajectory_render_multi",
            "seeds": [int(v) for v in seeds],
            "num_plans": int(len(seeds)),
            "video": None if final_video is None else str(Path(final_video).resolve()),
            "plans": summaries,
            "planned_constraint_violation": constraint_violation_stats(
                features_list=features_list,
                cutpoints_list=cutpoints_list,
                feature_schema=env_for_stats.get_feature_schema(),
                constraint_specs=env_for_stats.get_constraint_specs(),
                true_constraints=env_for_stats.get_true_constraints(),
                equality_tolerance=1e-4,
            ),
            "executed_constraint_violation": constraint_violation_stats(
                features_list=executed_features_list,
                cutpoints_list=cutpoints_list,
                feature_schema=env_for_stats.get_feature_schema(),
                constraint_specs=env_for_stats.get_constraint_specs(),
                true_constraints=env_for_stats.get_true_constraints(),
                equality_tolerance=1e-4,
            ),
        }
        aggregate["constraint_violation"] = aggregate["planned_constraint_violation"]
        (out_dir / "s5_planned_render_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[saved] {out_dir / 's5_planned_render_summary.json'}")
        print_render_violation_rates(aggregate)


if __name__ == "__main__":
    main()
