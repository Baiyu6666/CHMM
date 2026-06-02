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

from envs.S4SlideInsert import S4SlideInsertEnv
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
        raise ValueError("No matching benchmark result row.")
    if len(candidates) > 1:
        raise ValueError("Multiple benchmark rows matched; pass method/dataset/method-seed.")
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


def _canonical_feature_name(name: str) -> str:
    return str(name)


def _canonical_constraint_key(key: str) -> str:
    return str(key)


def _constraint_key_aliases(key: str) -> list[str]:
    return [str(key)]


def _canonical_constraint_spec(spec: dict) -> dict:
    out = dict(spec)
    if "feature_name" in out:
        out["feature_name"] = _canonical_feature_name(str(out.get("feature_name", "")))
    if "oracle_key" in out:
        out["oracle_key"] = _canonical_constraint_key(str(out.get("oracle_key", "")))
    return out


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
                    "feature_name": _canonical_feature_name(str(feature_names[feature_idx])),
                    "stage": int(stage_idx),
                    "semantics": semantic,
                }
            )
    return specs


def _constraint_specs_from_payload(
    payload: dict,
    env: S4SlideInsertEnv,
    *,
    constraint_source: str,
    feature_names: list[str],
) -> list[dict]:
    return [_canonical_constraint_spec(spec) for spec in list(payload.get("constraint_specs") or env.get_constraint_specs())]


def _true_active_mask_from_specs(payload: dict, specs: list[dict], feature_names: list[str], shape: tuple[int, int]) -> np.ndarray:
    true_mask = payload.get("ConstraintTrueActiveMask")
    if true_mask is not None:
        arr = np.asarray(true_mask)
        if arr.ndim == 2:
            return np.rint(arr.astype(float)).astype(bool)
    out = np.zeros(shape, dtype=bool)
    for spec in specs:
        name = _canonical_feature_name(str(spec.get("feature_name", "")))
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


def _validate_learned_active_matches_gt(payload: dict, learned_matrix, feature_names: list[str]) -> None:
    true_mask = payload.get("ConstraintTrueActiveMask")
    if true_mask is None:
        raise ValueError("Constraint JSON does not contain ConstraintTrueActiveMask; cannot validate learned active set.")
    true_arr = np.asarray(true_mask)
    if true_arr.ndim != 2:
        raise ValueError("ConstraintTrueActiveMask must be a 2D matrix.")
    true_arr = np.rint(true_arr.astype(float)).astype(bool)

    learned_arr = _active_mask_from_payload(payload, learned_matrix)
    if learned_arr is None:
        raise ValueError("Cannot infer learned active mask from ConstraintPredictedActiveMask or ConstraintLearnedValueMatrix.")
    if learned_arr.shape != true_arr.shape:
        raise ValueError(
            f"Learned active mask shape {learned_arr.shape} does not match GT active mask shape {true_arr.shape}."
        )

    mismatches = np.argwhere(learned_arr != true_arr)
    if mismatches.size == 0:
        return

    rows = []
    for stage_idx, feat_idx in mismatches[:20]:
        name = feature_names[int(feat_idx)] if int(feat_idx) < len(feature_names) else f"feature_{int(feat_idx)}"
        rows.append(
            f"s{int(stage_idx) + 1}:{name} learned={int(learned_arr[stage_idx, feat_idx])} "
            f"gt={int(true_arr[stage_idx, feat_idx])}"
        )
    suffix = "" if len(mismatches) <= 20 else f"; ... {len(mismatches) - 20} more"
    raise ValueError("Learned active feature-stage set does not match GT: " + "; ".join(rows) + suffix)


def _constraint_values_from_payload(
    payload: dict,
    env: S4SlideInsertEnv,
    *,
    constraint_source: str = "learned",
) -> dict:
    feature_names = [_canonical_feature_name(name) for name in list(payload.get("ConstraintFeatureNames", []))]
    if not feature_names:
        feature_names = ["surf_dist", "center_dist", "orient_err", "speed", "normal_force", "start_dist", "insert_err"]
    source = str(constraint_source or "learned").strip().lower()
    if source not in {"learned", "target"}:
        raise ValueError(f"Unsupported constraint source {constraint_source!r}.")
    if source == "target":
        out = {}
        for spec in env.get_constraint_specs():
            canon_spec = _canonical_constraint_spec(spec)
            oracle_key = str(canon_spec.get("oracle_key", ""))
            value_key = next((key for key in _constraint_key_aliases(oracle_key) if key in env.true_constraints), None)
            if value_key is None:
                continue
            stage_idx = int(canon_spec.get("stage", 0))
            name = str(canon_spec.get("feature_name", ""))
            out[f"s{stage_idx + 1}:{name}"] = float(env.true_constraints[value_key])
        return out
    else:
        learned = payload.get("ConstraintLearnedValueMatrix")
        if learned is None:
            raise ValueError("Constraint JSON does not contain ConstraintLearnedValueMatrix.")
    predicted_active = payload.get("ConstraintPredictedActiveMask") if source == "learned" else None
    specs = _constraint_specs_from_payload(payload, env, constraint_source=source, feature_names=feature_names)
    if source == "learned":
        _check_learned_mask_covers_gt(payload, learned, feature_names, specs)
    out = {}
    for spec in specs:
        name = _canonical_feature_name(str(spec.get("feature_name", "")))
        if name not in feature_names:
            continue
        stage_idx = int(spec.get("stage", 0))
        feature_idx = int(feature_names.index(name))
        value = _finite_matrix_value(learned, stage_idx, feature_idx)
        if value is None:
            if source == "learned":
                raise ValueError(f"Missing learned value for GT constraint s{stage_idx + 1}:{name}.")
            continue
        out[f"s{stage_idx + 1}:{name}"] = float(value)
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
            raise ValueError(f"Invalid stage length item {item!r}; expected stage3:67.")
        key, value = item.split(":", 1)
        out[str(key).strip()] = int(value)
    return out


def _parse_frame_indices(text: str | None) -> list[int]:
    return parse_int_list(text)


def _parse_rail_polyline(text: str | None):
    if text is None or not str(text).strip():
        return None
    pts = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        xy = [float(v.strip()) for v in item.split(",") if v.strip()]
        if len(xy) != 2:
            raise ValueError(f"Invalid rail polyline point {item!r}; expected x,y.")
        pts.append(xy)
    return pts or None


def _parse_optional_vec3(text: str | None):
    if text is None or not str(text).strip():
        return None
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if len(vals) != 3:
        raise ValueError(f"Invalid vec3 {text!r}; expected x,y,z.")
    return tuple(vals)


def _feature_names(feature_schema: list[dict], dim: int) -> list[str]:
    names = [f"feature_{idx}" for idx in range(int(dim))]
    for idx, item in enumerate(feature_schema or []):
        col = int(item.get("column_idx", item.get("id", idx)))
        if 0 <= col < len(names):
            names[col] = _canonical_feature_name(str(item.get("name", names[col])))
    return names


_S4_FEATURE_UNITS = {
    "surf_dist": "m",
    "center_dist": "m",
    "orient_err": "rad",
    "speed": "m/s",
    "angular_speed": "rad/s",
    "normal_force": "N",
    "start_dist": "m",
    "insert_err": "m",
}


def _feature_label_with_unit(name: str) -> str:
    name = str(name)
    unit = _S4_FEATURE_UNITS.get(name)
    return name if not unit else f"{name} [{unit}]"


def _stage_spans(cutpoints: list[int], length: int) -> list[tuple[int, int]]:
    cuts = [int(v) for v in cutpoints if 0 <= int(v) < int(length) - 1]
    ends = cuts + [int(length) - 1]
    starts = [0] + [int(v) + 1 for v in ends[:-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _plot_feature_profiles(
    *,
    env: S4SlideInsertEnv,
    planned_features: np.ndarray,
    executed_features: np.ndarray,
    cutpoints: list[int],
    constraint_payload: dict,
    constraint_values: dict,
    output_path: str | Path,
    use_env_true_constraints: bool = False,
) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot feature profiles.")
    Fp = np.asarray(planned_features, dtype=float)
    Fe = np.asarray(executed_features, dtype=float)
    dim = int(max(Fp.shape[1], Fe.shape[1]))
    names = _feature_names(env.get_feature_schema(), dim)
    spans = _stage_spans(cutpoints, max(len(Fp), len(Fe)))
    if bool(use_env_true_constraints):
        specs = list(env.get_constraint_specs())
        true_constraints = dict(env.true_constraints)
    else:
        specs = list(constraint_payload.get("constraint_specs") or env.get_constraint_specs())
        true_constraints = dict(constraint_payload.get("true_constraints") or env.true_constraints)
    specs = [_canonical_constraint_spec(spec) for spec in specs]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(dim, 1, figsize=(11.5, max(7.0, 1.45 * dim)), sharex=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    true_label_used = False
    learned_label_used = False
    for feat_idx, ax in enumerate(axes):
        if feat_idx < Fp.shape[1]:
            ax.plot(np.arange(len(Fp)), Fp[:, feat_idx], color="#D97706", linewidth=1.3, label="planned/reference")
        if feat_idx < Fe.shape[1]:
            ax.plot(np.arange(len(Fe)), Fe[:, feat_idx], color="#2563EB", linewidth=1.3, label="pybullet executed")
        for cp in cutpoints:
            ax.axvline(int(cp), color="#9CA3AF", linestyle="--", linewidth=0.8, alpha=0.75)
        feat_name = names[feat_idx]
        for spec in specs:
            if _canonical_feature_name(str(spec.get("feature_name", ""))) != feat_name:
                continue
            stage_idx = int(spec.get("stage", -1))
            if stage_idx < 0 or stage_idx >= len(spans):
                continue
            x0, x1 = spans[stage_idx]
            oracle_key = str(spec.get("oracle_key", ""))
            true_key = next((key for key in _constraint_key_aliases(oracle_key) if key in true_constraints), None)
            if true_key is not None:
                ax.hlines(
                    float(true_constraints[true_key]),
                    x0,
                    x1,
                    colors="#111827",
                    linestyles="--",
                    linewidth=1.1,
                    label="true target/bound" if not true_label_used else None,
                )
                true_label_used = True
            learned_key = f"s{stage_idx + 1}:{feat_name}"
            if learned_key in constraint_values:
                ax.hlines(
                    float(constraint_values[learned_key]),
                    x0,
                    x1,
                    colors="#7C3AED",
                    linestyles=":",
                    linewidth=1.35,
                    label="planned constraint" if not learned_label_used else None,
                )
                learned_label_used = True
        ax.set_ylabel(_feature_label_with_unit(feat_name), rotation=0, ha="right", va="center")
        ax.grid(alpha=0.18)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xlabel("t")
    fig.suptitle("S4 planned trajectory feature profiles", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _load_env_config() -> dict:
    cfg = _load_json(PROJECT_ROOT / "configs/envs/S4SlideInsert.json")
    cfg.pop("name", None)
    cfg.pop("n_demos", None)
    cfg.pop("seed", None)
    cfg.pop("method_overrides", None)
    return cfg


def _stage_lengths(env: S4SlideInsertEnv, overrides: dict | None) -> list[int]:
    lengths = [int(x) for x in env.seg_lengths]
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
    return lengths


def _auto_stage_lengths_for_rail(
    env: S4SlideInsertEnv,
    constraint_values: dict,
    overrides: dict | None,
) -> list[int]:
    lengths = _stage_lengths(env, overrides)
    if overrides:
        return lengths
    if not hasattr(env, "rail_total_length"):
        return lengths
    rail_total = max(float(env.rail_total_length()), 1e-8)
    dt = max(float(env.dt), 1e-8)

    def cv(key: str, default: float) -> float:
        value = dict(constraint_values or {}).get(key)
        if value is None:
            return float(default)
        value = float(value)
        return value if np.isfinite(value) and value > 1e-8 else float(default)

    v3 = cv("s3:speed", float(getattr(env, "v_insert_max", env.v3_target)))
    v4 = cv("s4:speed", float(getattr(env, "v_seat_max", env.v4_target)))
    base_l3 = max(int(lengths[2]), 3)
    base_l4 = max(int(lengths[3]), 3)
    base_d3 = max(v3 * dt * base_l3, 1e-8)
    base_d4 = max(v4 * dt * max(base_l4 - 1, 1), 1e-8)
    stage4_fraction = float(np.clip(base_d4 / (base_d3 + base_d4), 0.08, 0.35))
    d4 = rail_total * stage4_fraction
    d3 = max(rail_total - d4, v3 * dt)
    lengths[2] = max(int(np.ceil(d3 / max(v3 * dt, 1e-8))), 8)
    lengths[3] = max(int(np.ceil(d4 / max(v4 * dt, 1e-8))) + 1, 6)
    return lengths


def _scale_total_stage_lengths(lengths: list[int], scale: float) -> list[int]:
    vals = np.asarray([max(int(v), 1) for v in lengths], dtype=float)
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    target_total = max(int(round(float(np.sum(vals)) * scale)), len(vals))
    raw = vals / max(float(np.sum(vals)), 1e-12) * float(target_total)
    out = np.floor(raw).astype(int)
    out = np.maximum(out, 1)
    while int(np.sum(out)) < target_total:
        order = np.argsort(-(raw - np.floor(raw)))
        for idx in order:
            if int(np.sum(out)) >= target_total:
                break
            out[int(idx)] += 1
    while int(np.sum(out)) > target_total:
        order = np.argsort(-(out - 1))
        for idx in order:
            if int(np.sum(out)) <= target_total:
                break
            if out[int(idx)] > 1:
                out[int(idx)] -= 1
    return [int(v) for v in out.tolist()]


def _constraint_value(constraint_values: dict, key: str, default: float) -> float:
    values = dict(constraint_values or {})
    value = None
    for alias in _constraint_key_aliases(key):
        if alias in values:
            value = values[alias]
            break
    if value is None:
        return float(default)
    value = float(value)
    return value if np.isfinite(value) else float(default)


def _limit_stage_speed(xyz: np.ndarray, start: int, end: int, max_step: float) -> None:
    if max_step <= 0.0 or end <= start:
        return
    for t in range(start + 1, end + 1):
        step = xyz[t] - xyz[t - 1]
        dist = float(np.linalg.norm(step))
        if dist > max_step:
            xyz[t] = xyz[t - 1] + step * (max_step / max(dist, 1e-12))
    for t in range(end - 1, start - 1, -1):
        step = xyz[t] - xyz[t + 1]
        dist = float(np.linalg.norm(step))
        if dist > max_step:
            xyz[t] = xyz[t + 1] + step * (max_step / max(dist, 1e-12))


def _limit_edge_speeds(
    xyz: np.ndarray,
    edge_max_steps: np.ndarray,
    *,
    fixed_start: np.ndarray | None = None,
    fixed_goal: np.ndarray | None = None,
) -> None:
    edge_max_steps = np.asarray(edge_max_steps, dtype=float).reshape(-1)
    if xyz.shape[0] <= 1:
        return
    if fixed_start is not None:
        xyz[0] = np.asarray(fixed_start, dtype=float)
    for t in range(1, xyz.shape[0]):
        max_step = float(edge_max_steps[t])
        if max_step <= 0.0:
            continue
        step = xyz[t] - xyz[t - 1]
        dist = float(np.linalg.norm(step))
        if dist > max_step:
            xyz[t] = xyz[t - 1] + step * (max_step / max(dist, 1e-12))
    if fixed_goal is not None:
        xyz[-1] = np.asarray(fixed_goal, dtype=float)
    for t in range(xyz.shape[0] - 2, -1, -1):
        max_step = float(edge_max_steps[t + 1])
        if max_step <= 0.0:
            continue
        step = xyz[t] - xyz[t + 1]
        dist = float(np.linalg.norm(step))
        if dist > max_step:
            xyz[t] = xyz[t + 1] + step * (max_step / max(dist, 1e-12))
    if fixed_start is not None:
        xyz[0] = np.asarray(fixed_start, dtype=float)
    if fixed_goal is not None:
        xyz[-1] = np.asarray(fixed_goal, dtype=float)


def _interp_by_arclength(block: np.ndarray) -> np.ndarray:
    vals = np.asarray(block, dtype=float)
    if vals.ndim != 2 or vals.shape[0] < 3:
        return vals.copy()
    xyz = vals[:, :3]
    edge = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    total = float(np.sum(edge))
    if not np.isfinite(total) or total <= 1e-10:
        return vals.copy()
    s_old = np.concatenate([[0.0], np.cumsum(edge)])
    keep = np.concatenate([[True], np.diff(s_old) > 1e-12])
    if int(np.sum(keep)) < 2:
        return vals.copy()
    s_old = s_old[keep]
    source = vals[keep].copy()
    source[:, 3] = np.unwrap(source[:, 3])
    s_new = np.linspace(0.0, total, vals.shape[0])
    out = np.empty_like(vals)
    for d in range(vals.shape[1]):
        out[:, d] = np.interp(s_new, s_old, source[:, d])
    out[:, 3] = ((out[:, 3] + np.pi) % (2.0 * np.pi)) - np.pi
    out[0] = vals[0]
    out[-1] = vals[-1]
    return out


def _smooth_stage_speed_timing(
    env: S4SlideInsertEnv,
    traj: np.ndarray,
    labels: np.ndarray,
    constraint_values: dict,
    *,
    blend: float = 1.0,
) -> np.ndarray:
    out = np.asarray(traj, dtype=float).copy()
    labels = np.asarray(labels, dtype=int).reshape(-1)
    blend = float(np.clip(blend, 0.0, 1.0))
    if blend <= 0.0 or out.ndim != 2 or out.shape[0] != labels.size:
        return out
    use_rail = hasattr(env, "project_to_rail") and hasattr(env, "rail_pose_at_s")
    for stage_idx in np.unique(labels):
        idx = np.flatnonzero(labels == int(stage_idx))
        if idx.size < 3:
            continue
        block = out[idx].copy()
        candidate = None
        if use_rail and int(stage_idx) > 0:
            stage_no = int(stage_idx) + 1
            proj = env.project_to_rail(block[:, :2])
            s_old = np.asarray(proj.get("s", []), dtype=float).reshape(-1)
            if s_old.size == idx.size and np.all(np.isfinite(s_old)):
                s_new = np.linspace(float(s_old[0]), float(s_old[-1]), idx.size)
                surf = _constraint_value(
                    constraint_values,
                    f"s{stage_no}:surf_dist",
                    float(env.true_constraints.get("surface_target", 0.0)),
                )
                center = _constraint_value(
                    constraint_values,
                    f"s{stage_no}:center_dist",
                    float(env.clearance_target),
                )
                orient = _constraint_value(
                    constraint_values,
                    f"s{stage_no}:orient_err",
                    float(env.theta_stage2_end),
                )
                signed = np.asarray(proj.get("signed_dist", np.zeros(idx.size)), dtype=float).reshape(-1)
                sign = 1.0 if signed.size == 0 or float(np.nanmean(signed)) >= 0.0 else -1.0
                points, _tangents, normals, angles = env.rail_pose_at_s(s_new)
                candidate = np.zeros_like(block)
                candidate[:, :2] = (
                    np.asarray(points, dtype=float).reshape((-1, 2))
                    + np.asarray(normals, dtype=float).reshape((-1, 2)) * (sign * abs(float(center)))
                )
                candidate[:, 2] = env.surface_height(candidate[:, :2]) + float(surf)
                candidate[:, 3] = np.asarray(angles, dtype=float).reshape(-1) + float(orient)
        if candidate is None:
            candidate = _interp_by_arclength(block)
        candidate[0] = block[0]
        candidate[-1] = block[-1]
        out[idx] = (1.0 - blend) * block + blend * candidate
    out[0] = np.asarray(traj, dtype=float)[0]
    out[-1] = np.asarray(traj, dtype=float)[-1]
    return out


def _reproject_stage_orientation_targets(
    env: S4SlideInsertEnv,
    traj: np.ndarray,
    labels: np.ndarray,
    constraint_values: dict,
) -> np.ndarray:
    out = np.asarray(traj, dtype=float).copy()
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if out.ndim != 2 or out.shape[0] != labels.size or not (hasattr(env, "project_to_rail") and hasattr(env, "rail_pose_at_s")):
        return out
    proj = env.project_to_rail(out[:, :2])
    rail_angle = np.asarray(proj.get("angle", np.zeros(out.shape[0])), dtype=float).reshape(-1)
    if rail_angle.size != out.shape[0] or not np.all(np.isfinite(rail_angle)):
        return out
    theta_prev = out[:, 3].copy()
    for stage_idx in (1, 2, 3):
        mask = labels == int(stage_idx)
        if not np.any(mask):
            continue
        stage_no = int(stage_idx) + 1
        orient = _constraint_value(
            constraint_values,
            f"s{stage_no}:orient_err",
            float(env.theta_stage2_end if stage_idx < 3 else env.theta_stage4_end),
        )
        target = rail_angle[mask] + float(orient)
        current = theta_prev[mask]
        delta = np.arctan2(np.sin(target - current), np.cos(target - current))
        out[mask, 3] = current + delta
    out[:, 3] = ((out[:, 3] + np.pi) % (2.0 * np.pi)) - np.pi
    return out


def _plan_s4_stage_constraint_optimizer(
    env: S4SlideInsertEnv,
    scene: dict,
    constraint_values: dict,
    *,
    seed: int,
    stage_lengths: dict | None,
    speed_safety: float,
    global_speed_max: float | None,
    optimizer_iters: int,
    smooth_step: float,
    constraint_step: float,
    objective_step: float,
    speed_smooth: float,
) -> dict:
    """Simple direct trajectory optimizer for testing learned/GT constraints.

    This intentionally avoids intermediate task waypoints. It fixes only the
    initial pose, the final inserted pose, the stage schedule, and the per-stage
    constraints, then alternates smoothing with constraint projections.
    """
    lengths = _stage_lengths(env, stage_lengths)
    labels = np.repeat(np.arange(4), lengths)
    T = int(labels.size)
    cutpoints = np.where(np.diff(labels) != 0)[0].astype(int)

    surf2 = _constraint_value(constraint_values, "s2:surf_dist", float(env.true_constraints.get("surface_target", 0.0)))
    center2 = _constraint_value(constraint_values, "s2:center_dist", float(env.clearance_target))
    theta2 = _constraint_value(constraint_values, "s2:orient_err", float(env.theta_stage2_end))
    surf4 = _constraint_value(constraint_values, "s4:surf_dist", float(env.true_constraints.get("surface_target", 0.0)))
    center4 = _constraint_value(constraint_values, "s4:center_dist", float(env.clearance_target))
    theta4 = _constraint_value(constraint_values, "s4:orient_err", float(env.theta_stage4_end))
    rail_total = float(env.rail_total_length()) if hasattr(env, "rail_total_length") else abs(float(env.slot_x) - float(env.start[0]))
    end_point, _end_tangent, end_normal, end_angle = env.rail_pose_at_s(rail_total) if hasattr(env, "rail_pose_at_s") else (
        np.asarray([float(env.slot_x), float(env.clearance_target)], dtype=float),
        np.asarray([1.0, 0.0], dtype=float),
        np.asarray([0.0, 1.0], dtype=float),
        0.0,
    )
    end_point = np.asarray(end_point, dtype=float).reshape(2)
    end_normal = np.asarray(end_normal, dtype=float).reshape(2)
    end_angle = float(np.asarray(end_angle).reshape(-1)[0])
    start = np.asarray(
        [float(env.start[0]), 0.75 * float(env.slot_half_width), float(env.start[1]), float(env.theta_start)],
        dtype=float,
    )
    goal = np.asarray(
        [
            end_point[0] + float(center4) * end_normal[0],
            end_point[1] + float(center4) * end_normal[1],
            float(env.surface_height(np.asarray([[end_point[0] + float(center4) * end_normal[0], end_point[1] + float(center4) * end_normal[1]]], dtype=float))[0]) + float(surf4),
            end_angle + float(theta4),
        ],
        dtype=float,
    )
    if hasattr(env, "rail_pose_at_s"):
        l1, l2, l3, l4 = [int(v) for v in lengths]
        v2 = _constraint_value(constraint_values, "s2:speed", float(getattr(env, "v_align_max", env.v2_target)))
        v3 = _constraint_value(constraint_values, "s3:speed", float(getattr(env, "v_insert_max", env.v3_target)))
        v4 = _constraint_value(constraint_values, "s4:speed", float(getattr(env, "v_seat_max", env.v4_target)))
        s4 = rail_total
        s3 = max(0.0, s4 - max(v4, 1e-8) * float(env.dt) * max(l4 - 1, 1))
        s2 = max(0.0, s3 - max(v3, 1e-8) * float(env.dt) * max(l3, 1))
        s1 = max(0.0, s2 - max(v2, 1e-8) * float(env.dt) * max(l2, 1))
        reference = np.zeros((T, 4), dtype=float)
        p1, _t1, n1, a1 = env.rail_pose_at_s(s1)
        p1 = np.asarray(p1, dtype=float).reshape(2)
        n1 = np.asarray(n1, dtype=float).reshape(2)
        a1 = float(np.asarray(a1).reshape(-1)[0])
        w1_xy = np.asarray([p1[0] + float(center2) * n1[0], p1[1] + float(center2) * n1[1]], dtype=float)
        w1 = np.asarray([w1_xy[0], w1_xy[1], float(env.surface_height(w1_xy[None, :])[0]) + float(surf2), a1 + float(theta2)], dtype=float)
        reference[:l1] = np.linspace(start, w1, l1, endpoint=False)
        cursor = l1
        for n_steps, sa, sb, surf, center, theta, endpoint in (
            (l2, s1, s2, surf2, center2, theta2, False),
            (l3, s2, s3, _constraint_value(constraint_values, "s3:surf_dist", surf2), _constraint_value(constraint_values, "s3:center_dist", center2), _constraint_value(constraint_values, "s3:orient_err", theta2), False),
            (l4, s3, s4, surf4, center4, theta4, True),
        ):
            if n_steps <= 0:
                continue
            ss = np.linspace(float(sa), float(sb), int(n_steps), endpoint=bool(endpoint))
            pts, _tangents, normals, angles = env.rail_pose_at_s(ss)
            block = np.zeros((int(n_steps), 4), dtype=float)
            block[:, :2] = np.asarray(pts, dtype=float).reshape((-1, 2)) + np.asarray(normals, dtype=float).reshape((-1, 2)) * float(center)
            block[:, 2] = env.surface_height(block[:, :2]) + float(surf)
            block[:, 3] = np.asarray(angles, dtype=float).reshape(-1) + float(theta)
            reference[cursor:cursor + int(n_steps)] = block
            cursor += int(n_steps)
        reference[0] = start
        reference[-1] = goal
    else:
        reference = np.linspace(start, goal, T)
    traj = reference.copy()
    if global_speed_max is None or not np.isfinite(float(global_speed_max)) or float(global_speed_max) <= 0.0:
        edge_max_steps = np.full(T, np.inf, dtype=float)
    else:
        edge_max_steps = np.full(T, float(global_speed_max) * float(speed_safety) * float(env.dt), dtype=float)
    for stage_idx in (1, 2, 3):
        speed_key = f"s{stage_idx + 1}:speed"
        if speed_key not in dict(constraint_values or {}):
            continue
        value = float(constraint_values[speed_key])
        if np.isfinite(value):
            stage_max_step = max(float(value) * float(speed_safety) * float(env.dt), 1e-8)
            edge_max_steps[labels == stage_idx] = np.minimum(edge_max_steps[labels == stage_idx], stage_max_step)

    alpha = float(np.clip(constraint_step, 0.0, 1.0))
    beta = float(np.clip(smooth_step, 0.0, 0.45))
    gamma = float(np.clip(objective_step, 0.0, 0.35))
    for _ in range(max(int(optimizer_iters), 0)):
        if T > 2 and beta > 0.0:
            traj[1:-1] += beta * (traj[:-2] + traj[2:] - 2.0 * traj[1:-1])
        if T > 2 and gamma > 0.0:
            traj[1:-1, :2] += gamma * (reference[1:-1, :2] - traj[1:-1, :2])

        for stage_idx in (1, 2, 3):
            mask = labels == stage_idx
            stage_no = stage_idx + 1
            surf = _constraint_value(
                constraint_values,
                f"s{stage_no}:surf_dist",
                float(env.true_constraints.get("surface_target", 0.0)),
            )
            center = _constraint_value(
                constraint_values,
                f"s{stage_no}:center_dist",
                float(env.clearance_target),
            )
            orient = _constraint_value(
                constraint_values,
                f"s{stage_no}:orient_err",
                float(env.theta_stage2_end),
            )
            if hasattr(env, "project_to_rail") and hasattr(env, "rail_pose_at_s"):
                proj = env.project_to_rail(traj[mask, :2])
                signed = np.asarray(proj["signed_dist"], dtype=float)
                sign = 1.0 if float(np.mean(signed)) >= 0.0 else -1.0
                s_proj = np.asarray(proj["s"], dtype=float)
                points, _tangents, normals, angles = env.rail_pose_at_s(s_proj)
                target_xy = np.asarray(points, dtype=float).reshape((-1, 2)) + np.asarray(normals, dtype=float).reshape((-1, 2)) * (sign * abs(float(center)))
                target_theta = np.asarray(angles, dtype=float).reshape(-1) + float(orient)
                traj[mask, :2] = (1.0 - alpha) * traj[mask, :2] + alpha * target_xy
                traj[mask, 3] = (1.0 - alpha) * traj[mask, 3] + alpha * target_theta
            else:
                current_y = traj[mask, 1]
                sign = 1.0 if float(np.mean(current_y)) >= float(env.clearance_target) else -1.0
                target_y = float(env.clearance_target) if abs(center) < 1e-9 else float(env.clearance_target) + sign * abs(float(center))
                traj[mask, 1] = (1.0 - alpha) * traj[mask, 1] + alpha * target_y
                traj[mask, 3] = (1.0 - alpha) * traj[mask, 3] + alpha * (float(env.slot_theta) + float(orient))
            target_z = env.surface_height(traj[mask, :2]) + float(surf)
            traj[mask, 2] = (1.0 - alpha) * traj[mask, 2] + alpha * target_z

        xyz = traj[:, :3]
        _limit_edge_speeds(xyz, edge_max_steps, fixed_start=start[:3], fixed_goal=goal[:3])
        traj[:, :3] = xyz
        traj[0] = start
        traj[-1] = goal

    traj = _smooth_stage_speed_timing(env, traj, labels, constraint_values, blend=float(speed_smooth))
    traj = _reproject_stage_orientation_targets(env, traj, labels, constraint_values)

    normal_force = np.zeros(T, dtype=float)
    for stage_idx, key in [(1, "s2:normal_force"), (2, "s3:normal_force"), (3, "s4:normal_force")]:
        normal_force[labels == stage_idx] = max(_constraint_value(constraint_values, key, float(env.normal_load_min)), 0.0)

    return {
        "trajectory": traj,
        "planned_trajectory": traj,
        "true_cutpoints": cutpoints,
        "true_labels": labels,
        "normal_force_trace": normal_force,
        "normal_load_trace": normal_force,
        "constraint_values": dict(constraint_values or {}),
        "stage_lengths": {f"stage{i + 1}": int(n) for i, n in enumerate(lengths)},
        "global_speed_max": None if global_speed_max is None else float(global_speed_max),
        "planner": "s4_direct_stage_constraint_optimizer",
        "scene": dict(scene or {}),
        "seed": int(seed),
    }


def render_s4_planned_trajectory(
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
    camera_target,
    plot_features: bool,
    constraint_source: str,
    speed_safety: float,
    global_speed_max: float | None,
    stage_lengths: dict | None,
    stage_length_scale: float,
    benchmark_method: str | None,
    benchmark_dataset: str | None,
    benchmark_method_seed: int | None,
    visualize_normal_load: bool,
    feature_overlay: bool,
    execution_control: str,
    torque_kp: float,
    torque_kd: float,
    torque_limit: float,
    torque_substep_target_interp: bool,
    torque_arm_preload_scale: float,
    torque_preload_scale: float,
    torque_preload_max: float,
    torque_preload_indent_min: float,
    torque_preload_indent_max: float,
    torque_preload_indent_per_newton: float,
    torque_preload_adaptive_indent_gain: float,
    torque_preload_adaptive_indent_max: float,
    torque_force_feedback_gain: float,
    torque_force_relief_max: float,
    torque_force_command_max: float,
    torque_use_contact_proxy: bool,
    torque_contact_proxy_clearance: float,
    torque_apply_external_preload: bool,
    torque_contact_distance_tol: float,
    torque_contact_stiffness: float,
    torque_contact_damping: float,
    torque_slider_constraint_force: float,
    torque_slider_constraint_erp: float,
    torque_slider_constraint_force_min: float,
    torque_slider_constraint_force_threshold: float,
    torque_slider_constraint_force_per_newton: float,
    execution_joint_noise_std: float,
    execution_joint_noise_smooth: float,
    execution_noise_seed: int | None,
    execution_normal_load_noise_std: float,
    execution_normal_load_noise_smooth: float,
    execution_normal_load_noise_seed: int | None,
    save_frame_indices: list[int],
    planner: str,
    optimizer_iters: int,
    optimizer_smooth_step: float,
    optimizer_constraint_step: float,
    optimizer_objective_step: float,
    optimizer_speed_smooth: float,
    rail_shape: str,
    rail_bend_amp: float,
    rail_polyline,
    surface_tilt_x: float,
    surface_tilt_y: float,
    output_prefix: str = "s4_planned",
    video_path_override: str | Path | None = None,
) -> dict:
    out_dir = Path(outdir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = _load_env_config()
    env_cfg.update(
        {
            "rollout_backend": "pybullet",
            "observation_backend": "pybullet",
            "pybullet_camera_yaw": float(camera_yaw),
            "pybullet_camera_pitch": float(camera_pitch),
            "pybullet_camera_distance": float(camera_distance),
            "pybullet_camera_fov": float(camera_fov),
            "pybullet_render_width": int(width),
            "pybullet_render_height": int(height),
            "pybullet_torque_kp": float(torque_kp),
            "pybullet_torque_kd": float(torque_kd),
            "pybullet_torque_limit": float(torque_limit),
            "pybullet_torque_substep_target_interp": bool(torque_substep_target_interp),
            "pybullet_torque_arm_preload_scale": float(torque_arm_preload_scale),
            "pybullet_torque_preload_scale": float(torque_preload_scale),
            "pybullet_torque_preload_max": float(torque_preload_max),
            "pybullet_torque_preload_indent_min": float(torque_preload_indent_min),
            "pybullet_torque_preload_indent_max": float(torque_preload_indent_max),
            "pybullet_torque_preload_indent_per_newton": float(torque_preload_indent_per_newton),
            "pybullet_torque_preload_adaptive_indent_gain": float(torque_preload_adaptive_indent_gain),
            "pybullet_torque_preload_adaptive_indent_max": float(torque_preload_adaptive_indent_max),
            "pybullet_torque_force_feedback_gain": float(torque_force_feedback_gain),
            "pybullet_torque_force_relief_max": float(torque_force_relief_max),
            "pybullet_torque_force_command_max": float(torque_force_command_max),
            "pybullet_torque_use_contact_proxy": bool(torque_use_contact_proxy),
            "pybullet_torque_contact_proxy_clearance": float(torque_contact_proxy_clearance),
            "pybullet_torque_apply_external_preload": bool(torque_apply_external_preload),
            "pybullet_torque_contact_distance_tol": float(torque_contact_distance_tol),
            "pybullet_torque_contact_stiffness": float(torque_contact_stiffness),
            "pybullet_torque_contact_damping": float(torque_contact_damping),
            "pybullet_torque_slider_constraint_force": float(torque_slider_constraint_force),
            "pybullet_torque_slider_constraint_erp": float(torque_slider_constraint_erp),
            "pybullet_torque_slider_constraint_force_min": float(torque_slider_constraint_force_min),
            "pybullet_torque_slider_constraint_force_threshold": float(torque_slider_constraint_force_threshold),
            "pybullet_torque_slider_constraint_force_per_newton": float(torque_slider_constraint_force_per_newton),
            "rail_shape": str(rail_shape),
            "rail_bend_amp": float(rail_bend_amp),
            "rail_polyline": rail_polyline,
            "surface_tilt_x": float(surface_tilt_x),
            "surface_tilt_y": float(surface_tilt_y),
        }
    )
    if camera_target is not None:
        env_cfg["pybullet_camera_target"] = tuple(float(v) for v in np.asarray(camera_target, dtype=float).reshape(3))
    env = S4SlideInsertEnv(**env_cfg)

    raw_payload, resolved_constraints_path = _load_constraint_payload(constraints_json)
    payload = _select_constraint_payload(
        raw_payload,
        method=benchmark_method,
        dataset=benchmark_dataset,
        method_seed=benchmark_method_seed,
    )
    raw_constraint_values = _constraint_values_from_payload(
        payload,
        env,
        constraint_source=str(constraint_source),
    )
    feature_names = [_canonical_feature_name(name) for name in list(payload.get("ConstraintFeatureNames", []))] or [
        "surf_dist", "center_dist", "orient_err", "speed", "normal_force", "start_dist", "insert_err"
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
    base_stage_lengths = _auto_stage_lengths_for_rail(env, constraint_values, stage_lengths)
    scaled_stage_lengths = _scale_total_stage_lengths(base_stage_lengths, stage_length_scale)
    resolved_stage_lengths = {f"stage{i + 1}": int(n) for i, n in enumerate(scaled_stage_lengths)}
    scene = env.sample_scene(seed=int(seed))
    if str(planner).lower() == "optimizer":
        planned = _plan_s4_stage_constraint_optimizer(
            env,
            scene,
            constraint_values,
            seed=int(seed),
            stage_lengths=resolved_stage_lengths,
            speed_safety=float(speed_safety),
            global_speed_max=global_speed_max,
            optimizer_iters=int(optimizer_iters),
            smooth_step=float(optimizer_smooth_step),
            constraint_step=float(optimizer_constraint_step),
            objective_step=float(optimizer_objective_step),
            speed_smooth=float(optimizer_speed_smooth),
        )
    else:
        planned = env.plan_episode_from_constraints(
            scene,
            constraint_values,
            seed=int(seed),
            stage_lengths=resolved_stage_lengths,
            speed_safety=float(speed_safety),
        )
    print(f"[plan] points={len(planned['trajectory'])}, cutpoints={planned['true_cutpoints'].tolist()}")
    print(f"[plan] planner={planned.get('planner', planner)}")
    print(f"[plan] constraints={planned['constraint_values']}")
    print(f"[plan] inequality_clearance={{'upper_scale': 0.96, 'lower_scale': 1.04}}")

    output_prefix = str(output_prefix or "s4_planned")
    video_path = (Path(video_path_override) if video_path_override is not None else out_dir / f"{output_prefix}_pybullet.mp4") if int(gui) == 1 else None
    effective_realtime = bool(realtime) or int(gui) == 2
    effective_hold_seconds = (-1.0 if int(gui) == 2 else 0.0) if gui_hold_seconds is None else float(gui_hold_seconds)
    latent = env.execute_plan_pybullet(
        scene,
        planned,
        gui=int(gui),
        video_path=video_path,
        fps=float(fps),
        width=int(width),
        height=int(height),
        render_frame_stride=int(render_frame_stride),
        video_end_hold_seconds=2.0,
        realtime=bool(effective_realtime),
        gui_hold_seconds=float(effective_hold_seconds),
        visualize_normal_load=bool(visualize_normal_load),
        feature_overlay=bool(feature_overlay),
        feature_overlay_title=(
            "Executed trajectory feature profile (planned with learned constraints)"
            if str(constraint_source).lower() == "learned"
            else "Executed trajectory feature profile (planned with Ground truth constraints)"
        ),
        execution_control=str(execution_control),
        execution_joint_noise_std=float(execution_joint_noise_std),
        execution_joint_noise_smooth=float(execution_joint_noise_smooth),
        execution_noise_seed=execution_noise_seed,
        execution_normal_load_noise_std=float(execution_normal_load_noise_std),
        execution_normal_load_noise_smooth=float(execution_normal_load_noise_smooth),
        execution_normal_load_noise_seed=execution_normal_load_noise_seed,
        save_frame_indices=save_frame_indices,
        save_frame_dir=out_dir,
        save_frame_prefix=output_prefix,
    )
    obs = env.compute_observation(latent, scene)

    planned_traj = np.asarray(planned["trajectory"], dtype=float)
    executed_traj = np.asarray(obs["trajectory"], dtype=float)
    contact_slider_traj = np.asarray(obs.get("contact_slider_trajectory", []), dtype=float)
    executed_feature_traj = executed_traj
    planned_normal_force = np.asarray(planned.get("normal_force_trace", planned.get("normal_load_trace", [])), dtype=float)
    executed_normal_force = np.asarray(
        obs.get("normal_force_trace", obs.get("normal_load_trace", latent.get("normal_force_trace", latent.get("normal_load_trace", [])))),
        dtype=float,
    )
    if planned_normal_force.size == len(planned_traj):
        env.register_normal_load_trace(planned_traj, planned_normal_force)
    planned_features = np.asarray(env.compute_all_features_matrix(planned_traj), dtype=float)
    if executed_normal_force.size == len(executed_traj):
        env.register_normal_load_trace(executed_traj, executed_normal_force)
    executed_features = np.asarray(env.compute_all_features_matrix(executed_feature_traj), dtype=float)
    executed_surf_dist = np.asarray(obs.get("executed_surf_dist_trace", []), dtype=float).reshape(-1)
    if executed_surf_dist.size == len(executed_features) and executed_features.shape[1] > 0:
        executed_features[:, 0] = executed_surf_dist
    cutpoints = [int(v) for v in np.asarray(planned["true_cutpoints"], dtype=int).reshape(-1).tolist()]

    feature_plot_path = None
    if bool(plot_features):
        feature_plot_path = _plot_feature_profiles(
            env=env,
            planned_features=planned_features,
            executed_features=executed_features,
            cutpoints=cutpoints,
            constraint_payload=payload,
            constraint_values=dict(planned["constraint_values"]),
            output_path=out_dir / f"{output_prefix}_features.png",
            use_env_true_constraints=str(constraint_source).lower() == "target",
        )

    np.savez_compressed(
        out_dir / f"{output_prefix}_rollout.npz",
        planned_trajectory=planned_traj,
        executed_trajectory=executed_traj,
        contact_slider_trajectory=contact_slider_traj,
        executed_feature_trajectory=executed_feature_traj,
        planned_features=planned_features,
        executed_features=executed_features,
        cutpoints=np.asarray(cutpoints, dtype=int),
        normal_force_trace=executed_normal_force,
        planned_normal_force_trace=planned_normal_force,
        measured_normal_force_trace=np.asarray(obs.get("measured_normal_force_trace", obs.get("measured_normal_load_trace", [])), dtype=float),
        normal_load_trace=executed_normal_force,
        planned_normal_load_trace=planned_normal_force,
        measured_normal_load_trace=np.asarray(obs.get("measured_normal_load_trace", obs.get("measured_normal_force_trace", [])), dtype=float),
        preload_command_trace=np.asarray(obs.get("preload_command_trace", []), dtype=float),
        preload_force_command_trace=np.asarray(obs.get("preload_force_command_trace", []), dtype=float),
        preload_indent_trace=np.asarray(obs.get("preload_indent_trace", []), dtype=float),
        executed_surf_dist_trace=executed_surf_dist,
        execution_normal_force_noise=np.asarray(obs.get("execution_normal_force_noise", obs.get("execution_normal_load_noise", [])), dtype=float),
        execution_normal_load_noise=np.asarray(obs.get("execution_normal_load_noise", obs.get("execution_normal_force_noise", [])), dtype=float),
        joint_positions=np.asarray(obs.get("joint_positions", []), dtype=float),
        joint_position_commands=np.asarray(obs.get("joint_position_commands", []), dtype=float),
        joint_position_commands_nominal=np.asarray(obs.get("joint_position_commands_nominal", []), dtype=float),
        joint_position_commands_planned=np.asarray(obs.get("joint_position_commands_planned", []), dtype=float),
        joint_position_commands_planned_nominal=np.asarray(obs.get("joint_position_commands_planned_nominal", []), dtype=float),
        joint_torque_commands=np.asarray(obs.get("joint_torque_commands", []), dtype=float),
        execution_joint_noise=np.asarray(obs.get("execution_joint_noise", []), dtype=float),
        ik_position_error_world=np.asarray(obs.get("ik_position_error_world", []), dtype=float),
    )

    violation_stats = constraint_violation_stats(
        features_list=[planned_features],
        cutpoints_list=[cutpoints],
        feature_schema=env.get_feature_schema(),
        constraint_specs=env.get_constraint_specs(),
        true_constraints=env.get_true_constraints(),
        equality_tolerance=1e-3,
    )
    executed_violation_stats = constraint_violation_stats(
        features_list=[executed_features],
        cutpoints_list=[cutpoints],
        feature_schema=env.get_feature_schema(),
        constraint_specs=env.get_constraint_specs(),
        true_constraints=env.get_true_constraints(),
        equality_tolerance=1e-3,
    )

    summary = {
        "task": "s4_planned_trajectory_render",
        "constraints_json": str(Path(constraints_json)),
        "resolved_constraints_payload": str(resolved_constraints_path),
        "constraint_source": str(constraint_source),
        "planner": str(planned.get("planner", planner)),
        "seed": int(seed),
        "gui": int(gui),
        "raw_constraint_values": dict(raw_constraint_values),
        "constraint_values": dict(planned["constraint_values"]),
        "inequality_clearance": {"upper_scale": 0.96, "lower_scale": 1.04},
        "stage_lengths": dict(planned["stage_lengths"]),
        "global_speed_max": planned.get("global_speed_max"),
        "optimizer_speed_smooth": float(optimizer_speed_smooth),
        "rail_shape": str(getattr(env, "rail_shape", "straight")),
        "rail_bend_amp": float(getattr(env, "rail_bend_amp", 0.0)),
        "rail_polyline": env.get_rail_polyline(num=64).tolist() if hasattr(env, "get_rail_polyline") else None,
        "cutpoints": cutpoints,
        "trajectory_points": int(len(planned_traj)),
        "feature_plot": None if feature_plot_path is None else str(Path(feature_plot_path).resolve()),
        "rollout_npz": str((out_dir / f"{output_prefix}_rollout.npz").resolve()),
        "video": None if video_path is None else str(video_path.resolve()),
        "frames": int(latent.get("frames", 0)),
        "saved_frames": list(latent.get("saved_frames", [])),
        "feature_overlay": bool(feature_overlay),
        "execution_control": str(execution_control),
        "robot_backend": str(obs.get("robot_backend", latent.get("robot_backend", ""))),
        "torque_kp": float(torque_kp),
        "torque_kd": float(torque_kd),
        "torque_limit": float(torque_limit),
        "torque_substep_target_interp": bool(torque_substep_target_interp),
        "torque_arm_preload_scale": float(torque_arm_preload_scale),
        "torque_preload_scale": float(torque_preload_scale),
        "torque_preload_max": float(torque_preload_max),
        "torque_preload_indent_min": float(torque_preload_indent_min),
        "torque_preload_indent_max": float(torque_preload_indent_max),
        "torque_preload_indent_per_newton": float(torque_preload_indent_per_newton),
        "torque_preload_adaptive_indent_gain": float(torque_preload_adaptive_indent_gain),
        "torque_preload_adaptive_indent_max": float(torque_preload_adaptive_indent_max),
        "torque_force_feedback_gain": float(torque_force_feedback_gain),
        "torque_force_relief_max": float(torque_force_relief_max),
        "torque_force_command_max": float(torque_force_command_max),
        "torque_use_contact_proxy": bool(torque_use_contact_proxy),
        "torque_contact_proxy_clearance": float(torque_contact_proxy_clearance),
        "torque_apply_external_preload": bool(torque_apply_external_preload),
        "torque_contact_distance_tol": float(torque_contact_distance_tol),
        "torque_contact_stiffness": float(torque_contact_stiffness),
        "torque_contact_damping": float(torque_contact_damping),
        "torque_slider_constraint_force": float(torque_slider_constraint_force),
        "torque_slider_constraint_erp": float(torque_slider_constraint_erp),
        "torque_slider_constraint_force_min": float(torque_slider_constraint_force_min),
        "torque_slider_constraint_force_threshold": float(torque_slider_constraint_force_threshold),
        "torque_slider_constraint_force_per_newton": float(torque_slider_constraint_force_per_newton),
        "execution_joint_noise_std": float(execution_joint_noise_std),
        "execution_joint_noise_smooth": float(execution_joint_noise_smooth),
        "execution_noise_seed": None if execution_noise_seed is None else int(execution_noise_seed),
        "execution_normal_force_noise_std": float(execution_normal_load_noise_std),
        "execution_normal_force_noise_smooth": float(execution_normal_load_noise_smooth),
        "execution_normal_force_noise_seed": None if execution_normal_load_noise_seed is None else int(execution_normal_load_noise_seed),
        "execution_normal_load_noise_std": float(execution_normal_load_noise_std),
        "execution_normal_load_noise_smooth": float(execution_normal_load_noise_smooth),
        "execution_normal_load_noise_seed": None if execution_normal_load_noise_seed is None else int(execution_normal_load_noise_seed),
        "ik_position_error_mean": None if "ik_position_error_world" not in obs else float(np.mean(obs["ik_position_error_world"])),
        "ik_position_error_max": None if "ik_position_error_world" not in obs else float(np.max(obs["ik_position_error_world"])),
        "surface_tilt_x": float(getattr(env, "surface_tilt_x", 0.0)),
        "surface_tilt_y": float(getattr(env, "surface_tilt_y", 0.0)),
        "planned_constraint_violation": violation_stats,
        "constraint_violation": violation_stats,
        "executed_constraint_violation": executed_violation_stats,
    }
    summary_path = out_dir / f"{output_prefix}_render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[saved] {summary_path}")
    print(f"[saved] features={feature_plot_path}, video={video_path}")
    print_render_violation_rates(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan an S4 realistic trajectory from learned constraints and render PyBullet execution.")
    parser.add_argument("--constraints-json", required=True, help="Path to constraints.json or benchmark_results.json.")
    parser.add_argument("--benchmark-method", default=None)
    parser.add_argument("--benchmark-dataset", default=None)
    parser.add_argument("--benchmark-method-seed", type=int, default=None)
    parser.add_argument("--outdir", default="outputs/s4_planned_render")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--plan-seeds", default=None, help="Comma-separated seeds for rendering multiple planned trajectories.")
    parser.add_argument("--n-plans", type=int, default=1, help="Number of planned trajectories to render, starting from --seed, when --plan-seeds is not set.")
    parser.add_argument("--gui", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=1360)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--render-frame-stride", type=int, default=1)
    parser.add_argument("--realtime", type=int, default=0)
    parser.add_argument("--gui-hold-seconds", type=float, default=None)
    parser.add_argument("--camera-yaw", type=float, default=38.0)
    parser.add_argument("--camera-pitch", type=float, default=-33.0)
    parser.add_argument("--camera-distance", type=float, default=0.90)
    parser.add_argument("--camera-fov", type=float, default=42.0)
    parser.add_argument("--camera-target", default="0.72,0.14,0.54", help="World-frame camera target as x,y,z. Increase x/y to pan the S4 view right, moving the robot toward the left side of the video area.")
    parser.add_argument("--plot-features", type=int, default=1)
    parser.add_argument("--constraint-source", choices=["learned", "target"], default="learned")
    parser.add_argument("--speed-safety", type=float, default=1.0)
    parser.add_argument("--global-speed-max", type=float, default=0.016)
    parser.add_argument("--stage-lengths", default=None, help="Optional comma list, e.g. stage3:67,stage4:21.")
    parser.add_argument("--stage-length-scale", type=float, default=1.20, help="Scale the total planned trajectory length while preserving stage ratios.")
    parser.add_argument("--visualize-normal-force", "--visualize-normal-load", dest="visualize_normal_load", type=int, default=0)
    parser.add_argument("--feature-overlay", type=int, default=1)
    parser.add_argument("--execution-control", choices=["position", "torque_preload"], default="position", help="S4 PyBullet execution controller. torque_preload uses torque-level tracking with an attached slider and table contact normal-load measurement.")
    parser.add_argument("--torque-kp", type=float, default=450.0)
    parser.add_argument("--torque-kd", type=float, default=70.0)
    parser.add_argument("--torque-limit", type=float, default=500.0)
    parser.add_argument("--torque-substep-target-interp", type=int, default=1)
    parser.add_argument("--torque-arm-preload-scale", type=float, default=1.0)
    parser.add_argument("--torque-preload-scale", type=float, default=1.0)
    parser.add_argument("--torque-preload-max", type=float, default=30.0)
    parser.add_argument("--torque-preload-indent-min", type=float, default=0.0)
    parser.add_argument("--torque-preload-indent-max", type=float, default=0.014)
    parser.add_argument("--torque-preload-indent-per-newton", type=float, default=0.0009)
    parser.add_argument("--torque-preload-adaptive-indent-gain", type=float, default=0.0)
    parser.add_argument("--torque-preload-adaptive-indent-max", type=float, default=0.014)
    parser.add_argument("--torque-force-feedback-gain", type=float, default=28.0)
    parser.add_argument("--torque-force-relief-max", type=float, default=80.0)
    parser.add_argument("--torque-force-command-max", type=float, default=260.0)
    parser.add_argument("--torque-use-contact-proxy", type=int, default=1)
    parser.add_argument("--torque-contact-proxy-clearance", type=float, default=0.0)
    parser.add_argument("--torque-apply-external-preload", type=int, default=0)
    parser.add_argument("--torque-contact-distance-tol", type=float, default=1e-5)
    parser.add_argument("--torque-contact-stiffness", type=float, default=7000.0)
    parser.add_argument("--torque-contact-damping", type=float, default=30.0)
    parser.add_argument("--torque-slider-constraint-force", type=float, default=1000000.0)
    parser.add_argument("--torque-slider-constraint-erp", type=float, default=0.0)
    parser.add_argument("--torque-slider-constraint-force-min", type=float, default=2.5)
    parser.add_argument("--torque-slider-constraint-force-threshold", type=float, default=5.0)
    parser.add_argument("--torque-slider-constraint-force-per-newton", type=float, default=5.0)
    parser.add_argument("--save-frame-indices", default=None, help="Comma-separated source frame indices to save as PNGs in --outdir, e.g. 0,80,157.")
    parser.add_argument("--execution-joint-noise-std", type=float, default=0.0002)
    parser.add_argument("--execution-joint-noise-smooth", type=float, default=0.90)
    parser.add_argument("--execution-noise-seed", type=int, default=None)
    parser.add_argument("--execution-normal-force-noise-std", "--execution-normal-load-noise-std", dest="execution_normal_load_noise_std", type=float, default=0.0025)
    parser.add_argument("--execution-normal-force-noise-smooth", "--execution-normal-load-noise-smooth", dest="execution_normal_load_noise_smooth", type=float, default=0.85)
    parser.add_argument("--execution-normal-force-noise-seed", "--execution-normal-load-noise-seed", dest="execution_normal_load_noise_seed", type=int, default=None)
    parser.add_argument("--planner", choices=["waypoint", "optimizer"], default="optimizer")
    parser.add_argument("--optimizer-iters", type=int, default=500)
    parser.add_argument("--optimizer-smooth-step", type=float, default=0.18)
    parser.add_argument("--optimizer-constraint-step", type=float, default=0.45)
    parser.add_argument("--optimizer-objective-step", type=float, default=0.08)
    parser.add_argument("--optimizer-speed-smooth", type=float, default=1.0, help="Stage-wise arc-length retiming strength for smoother planned speed. Use 0 to disable.")
    parser.add_argument("--rail-shape", choices=["straight", "sine", "polyline"], default="straight")
    parser.add_argument("--rail-bend-amp", type=float, default=0.012)
    parser.add_argument("--rail-polyline", default=None, help="Optional transfer rail centerline as 'x,y;x,y;...'.")
    parser.add_argument("--surface-tilt-x", type=float, default=0.0, help="Surface height slope dz/dx in S4 coordinates.")
    parser.add_argument("--surface-tilt-y", type=float, default=0.0, help="Surface height slope dz/dy in S4 coordinates.")
    args = parser.parse_args()

    seeds = plan_seed_list(int(args.seed), args.plan_seeds, int(args.n_plans))
    out_dir = Path(args.outdir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    temp_videos = []
    multi = len(seeds) > 1
    for plan_idx, plan_seed in enumerate(seeds):
        prefix = "s4_planned" if not multi else f"s4_planned_seed_{int(plan_seed):03d}"
        video_override = None
        if multi and int(args.gui) == 1:
            video_override = out_dir / f"._tmp_{prefix}_pybullet.mp4"
        summary = render_s4_planned_trajectory(
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
            camera_target=_parse_optional_vec3(args.camera_target),
            plot_features=bool(args.plot_features),
            constraint_source=str(args.constraint_source),
            speed_safety=float(args.speed_safety),
            global_speed_max=None if float(args.global_speed_max) <= 0.0 else float(args.global_speed_max),
            stage_lengths=_parse_stage_lengths(args.stage_lengths),
            stage_length_scale=float(args.stage_length_scale),
            benchmark_method=args.benchmark_method,
            benchmark_dataset=args.benchmark_dataset,
            benchmark_method_seed=args.benchmark_method_seed,
            visualize_normal_load=bool(args.visualize_normal_load),
            feature_overlay=bool(args.feature_overlay),
            execution_control=str(args.execution_control),
            torque_kp=float(args.torque_kp),
            torque_kd=float(args.torque_kd),
            torque_limit=float(args.torque_limit),
            torque_substep_target_interp=bool(args.torque_substep_target_interp),
            torque_arm_preload_scale=float(args.torque_arm_preload_scale),
            torque_preload_scale=float(args.torque_preload_scale),
            torque_preload_max=float(args.torque_preload_max),
            torque_preload_indent_min=float(args.torque_preload_indent_min),
            torque_preload_indent_max=float(args.torque_preload_indent_max),
            torque_preload_indent_per_newton=float(args.torque_preload_indent_per_newton),
            torque_preload_adaptive_indent_gain=float(args.torque_preload_adaptive_indent_gain),
            torque_preload_adaptive_indent_max=float(args.torque_preload_adaptive_indent_max),
            torque_force_feedback_gain=float(args.torque_force_feedback_gain),
            torque_force_relief_max=float(args.torque_force_relief_max),
            torque_force_command_max=float(args.torque_force_command_max),
            torque_use_contact_proxy=bool(args.torque_use_contact_proxy),
            torque_contact_proxy_clearance=float(args.torque_contact_proxy_clearance),
            torque_apply_external_preload=bool(args.torque_apply_external_preload),
            torque_contact_distance_tol=float(args.torque_contact_distance_tol),
            torque_contact_stiffness=float(args.torque_contact_stiffness),
            torque_contact_damping=float(args.torque_contact_damping),
            torque_slider_constraint_force=float(args.torque_slider_constraint_force),
            torque_slider_constraint_erp=float(args.torque_slider_constraint_erp),
            torque_slider_constraint_force_min=float(args.torque_slider_constraint_force_min),
            torque_slider_constraint_force_threshold=float(args.torque_slider_constraint_force_threshold),
            torque_slider_constraint_force_per_newton=float(args.torque_slider_constraint_force_per_newton),
            execution_joint_noise_std=float(args.execution_joint_noise_std),
            execution_joint_noise_smooth=float(args.execution_joint_noise_smooth),
            execution_noise_seed=(None if args.execution_noise_seed is None else int(args.execution_noise_seed) + int(plan_idx)),
            execution_normal_load_noise_std=float(args.execution_normal_load_noise_std),
            execution_normal_load_noise_smooth=float(args.execution_normal_load_noise_smooth),
            execution_normal_load_noise_seed=(None if args.execution_normal_load_noise_seed is None else int(args.execution_normal_load_noise_seed) + int(plan_idx)),
            save_frame_indices=_parse_frame_indices(args.save_frame_indices),
            planner=str(args.planner),
            optimizer_iters=int(args.optimizer_iters),
            optimizer_smooth_step=float(args.optimizer_smooth_step),
            optimizer_constraint_step=float(args.optimizer_constraint_step),
            optimizer_objective_step=float(args.optimizer_objective_step),
            optimizer_speed_smooth=float(args.optimizer_speed_smooth),
            rail_shape=str(args.rail_shape),
            rail_bend_amp=float(args.rail_bend_amp),
            rail_polyline=_parse_rail_polyline(args.rail_polyline),
            surface_tilt_x=float(args.surface_tilt_x),
            surface_tilt_y=float(args.surface_tilt_y),
            output_prefix=prefix,
            video_path_override=video_override,
        )
        summaries.append(summary)
        if video_override is not None:
            temp_videos.append(Path(video_override))

    if multi:
        final_video = None
        if int(args.gui) == 1 and temp_videos:
            final_video = concat_mp4_files(temp_videos, out_dir / "s4_planned_pybullet.mp4")
            for path in temp_videos:
                try:
                    path.unlink()
                except OSError:
                    pass
        aggregate = {
            "task": "s4_planned_trajectory_render_multi",
            "seeds": [int(v) for v in seeds],
            "num_plans": int(len(seeds)),
            "video": None if final_video is None else str(Path(final_video).resolve()),
            "plans": summaries,
        }
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
        env_for_stats = S4SlideInsertEnv(**_load_env_config())
        aggregate["planned_constraint_violation"] = constraint_violation_stats(
            features_list=features_list,
            cutpoints_list=cutpoints_list,
            feature_schema=env_for_stats.get_feature_schema(),
            constraint_specs=env_for_stats.get_constraint_specs(),
            true_constraints=env_for_stats.get_true_constraints(),
            equality_tolerance=1e-3,
        )
        aggregate["constraint_violation"] = aggregate["planned_constraint_violation"]
        aggregate["executed_constraint_violation"] = constraint_violation_stats(
            features_list=executed_features_list,
            cutpoints_list=cutpoints_list,
            feature_schema=env_for_stats.get_feature_schema(),
            constraint_specs=env_for_stats.get_constraint_specs(),
            true_constraints=env_for_stats.get_true_constraints(),
            equality_tolerance=1e-3,
        )
        (out_dir / "s4_planned_render_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[saved] {out_dir / 's4_planned_render_summary.json'}")
        print_render_violation_rates(aggregate)


if __name__ == "__main__":
    main()
