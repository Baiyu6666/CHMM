from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.S4SlideInsertRealistic import S4SlideInsertRealisticEnv


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
    env: S4SlideInsertRealisticEnv,
    *,
    constraint_source: str = "learned",
) -> dict:
    feature_names = list(payload.get("ConstraintFeatureNames", []))
    if not feature_names:
        feature_names = ["surf_dist", "centerline_dist", "orient_err", "speed", "normal_load", "start_dist", "insertion_err"]
    source = str(constraint_source or "learned").strip().lower()
    if source not in {"learned", "target"}:
        raise ValueError(f"Unsupported constraint source {constraint_source!r}.")
    if source == "target":
        out = {}
        for spec in env.get_constraint_specs():
            oracle_key = str(spec.get("oracle_key", ""))
            if oracle_key not in env.true_constraints:
                continue
            stage_idx = int(spec.get("stage", 0))
            name = str(spec.get("feature_name", ""))
            out[f"s{stage_idx + 1}:{name}"] = float(env.true_constraints[oracle_key])
        return out
    else:
        learned = payload.get("ConstraintLearnedValueMatrix")
        if learned is None:
            raise ValueError("Constraint JSON does not contain ConstraintLearnedValueMatrix.")
        _validate_learned_active_matches_gt(payload, learned, feature_names)
    predicted_active = payload.get("ConstraintPredictedActiveMask") if source == "learned" else None
    specs = list(payload.get("constraint_specs") or env.get_constraint_specs())
    out = {}
    for spec in specs:
        name = str(spec.get("feature_name", ""))
        if name not in feature_names:
            continue
        stage_idx = int(spec.get("stage", 0))
        feature_idx = int(feature_names.index(name))
        active = _matrix_bool_value(predicted_active, stage_idx, feature_idx)
        if active is False:
            continue
        value = _finite_matrix_value(learned, stage_idx, feature_idx)
        if value is not None:
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


def _plot_feature_profiles(
    *,
    env: S4SlideInsertRealisticEnv,
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
            if str(spec.get("feature_name", "")) != feat_name:
                continue
            stage_idx = int(spec.get("stage", -1))
            if stage_idx < 0 or stage_idx >= len(spans):
                continue
            x0, x1 = spans[stage_idx]
            oracle_key = str(spec.get("oracle_key", ""))
            if oracle_key in true_constraints:
                ax.hlines(
                    float(true_constraints[oracle_key]),
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
        ax.set_ylabel(feat_name, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.18)
    axes[0].legend(loc="upper right", frameon=False, ncol=2)
    axes[-1].set_xlabel("t")
    fig.suptitle("S4 planned trajectory feature profiles", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _load_env_config() -> dict:
    cfg = _load_json(PROJECT_ROOT / "configs/envs/S4SlideInsertRealistic.json")
    cfg.pop("name", None)
    cfg.pop("n_demos", None)
    cfg.pop("seed", None)
    cfg.pop("method_overrides", None)
    return cfg


def _stage_lengths(env: S4SlideInsertRealisticEnv, overrides: dict | None) -> list[int]:
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
    env: S4SlideInsertRealisticEnv,
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


def _constraint_value(constraint_values: dict, key: str, default: float) -> float:
    value = dict(constraint_values or {}).get(key)
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


def _plan_s4_stage_constraint_optimizer(
    env: S4SlideInsertRealisticEnv,
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
    center2 = _constraint_value(constraint_values, "s2:centerline_dist", float(env.clearance_target))
    theta2 = _constraint_value(constraint_values, "s2:orient_err", float(env.theta_stage2_end))
    surf4 = _constraint_value(constraint_values, "s4:surf_dist", float(env.true_constraints.get("surface_target", 0.0)))
    center4 = _constraint_value(constraint_values, "s4:centerline_dist", float(env.clearance_target))
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
            (l3, s2, s3, _constraint_value(constraint_values, "s3:surf_dist", surf2), _constraint_value(constraint_values, "s3:centerline_dist", center2), _constraint_value(constraint_values, "s3:orient_err", theta2), False),
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
                f"s{stage_no}:centerline_dist",
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

    normal_load = np.zeros(T, dtype=float)
    for stage_idx, key in [(1, "s2:normal_load"), (2, "s3:normal_load"), (3, "s4:normal_load")]:
        normal_load[labels == stage_idx] = max(_constraint_value(constraint_values, key, float(env.normal_load_min)), 0.0)

    return {
        "trajectory": traj,
        "planned_trajectory": traj,
        "true_cutpoints": cutpoints,
        "true_labels": labels,
        "normal_load_trace": normal_load,
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
    plot_features: bool,
    constraint_source: str,
    speed_safety: float,
    global_speed_max: float | None,
    stage_lengths: dict | None,
    benchmark_method: str | None,
    benchmark_dataset: str | None,
    benchmark_method_seed: int | None,
    visualize_normal_load: bool,
    feature_overlay: bool,
    execution_joint_noise_std: float,
    execution_joint_noise_smooth: float,
    execution_noise_seed: int | None,
    execution_normal_load_noise_std: float,
    execution_normal_load_noise_smooth: float,
    execution_normal_load_noise_seed: int | None,
    planner: str,
    optimizer_iters: int,
    optimizer_smooth_step: float,
    optimizer_constraint_step: float,
    optimizer_objective_step: float,
    rail_shape: str,
    rail_bend_amp: float,
    rail_polyline,
    surface_tilt_x: float,
    surface_tilt_y: float,
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
            "rail_shape": str(rail_shape),
            "rail_bend_amp": float(rail_bend_amp),
            "rail_polyline": rail_polyline,
            "surface_tilt_x": float(surface_tilt_x),
            "surface_tilt_y": float(surface_tilt_y),
        }
    )
    env = S4SlideInsertRealisticEnv(**env_cfg)

    raw_payload, resolved_constraints_path = _load_constraint_payload(constraints_json)
    payload = _select_constraint_payload(
        raw_payload,
        method=benchmark_method,
        dataset=benchmark_dataset,
        method_seed=benchmark_method_seed,
    )
    constraint_values = _constraint_values_from_payload(
        payload,
        env,
        constraint_source=str(constraint_source),
    )
    resolved_stage_lengths = {f"stage{i + 1}": int(n) for i, n in enumerate(_auto_stage_lengths_for_rail(env, constraint_values, stage_lengths))}
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

    video_path = out_dir / "s4_planned_pybullet.mp4" if int(gui) == 1 else None
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
            else "Executed trajectory feature profile (planned with GT constraints)"
        ),
        execution_joint_noise_std=float(execution_joint_noise_std),
        execution_joint_noise_smooth=float(execution_joint_noise_smooth),
        execution_noise_seed=execution_noise_seed,
        execution_normal_load_noise_std=float(execution_normal_load_noise_std),
        execution_normal_load_noise_smooth=float(execution_normal_load_noise_smooth),
        execution_normal_load_noise_seed=execution_normal_load_noise_seed,
    )
    obs = env.compute_observation(latent, scene)

    planned_traj = np.asarray(planned["trajectory"], dtype=float)
    executed_traj = np.asarray(obs["trajectory"], dtype=float)
    planned_normal_load = np.asarray(planned.get("normal_load_trace", []), dtype=float)
    executed_normal_load = np.asarray(obs.get("normal_load_trace", latent.get("normal_load_trace", [])), dtype=float)
    if planned_normal_load.size == len(planned_traj):
        env.register_normal_load_trace(planned_traj, planned_normal_load)
    if executed_normal_load.size == len(executed_traj):
        env.register_normal_load_trace(executed_traj, executed_normal_load)
    planned_features = np.asarray(env.compute_all_features_matrix(planned_traj), dtype=float)
    executed_features = np.asarray(obs["features"], dtype=float)
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
            output_path=out_dir / "s4_planned_features.png",
            use_env_true_constraints=str(constraint_source).lower() == "target",
        )

    np.savez_compressed(
        out_dir / "s4_planned_rollout.npz",
        planned_trajectory=planned_traj,
        executed_trajectory=executed_traj,
        planned_features=planned_features,
        executed_features=executed_features,
        cutpoints=np.asarray(cutpoints, dtype=int),
        normal_load_trace=executed_normal_load,
        planned_normal_load_trace=planned_normal_load,
        execution_normal_load_noise=np.asarray(obs.get("execution_normal_load_noise", []), dtype=float),
        joint_positions=np.asarray(obs.get("joint_positions", []), dtype=float),
        joint_position_commands=np.asarray(obs.get("joint_position_commands", []), dtype=float),
        joint_position_commands_nominal=np.asarray(obs.get("joint_position_commands_nominal", []), dtype=float),
        execution_joint_noise=np.asarray(obs.get("execution_joint_noise", []), dtype=float),
        ik_position_error_world=np.asarray(obs.get("ik_position_error_world", []), dtype=float),
    )

    summary = {
        "task": "s4_planned_trajectory_render",
        "constraints_json": str(Path(constraints_json)),
        "resolved_constraints_payload": str(resolved_constraints_path),
        "constraint_source": str(constraint_source),
        "planner": str(planned.get("planner", planner)),
        "seed": int(seed),
        "gui": int(gui),
        "constraint_values": dict(planned["constraint_values"]),
        "stage_lengths": dict(planned["stage_lengths"]),
        "global_speed_max": planned.get("global_speed_max"),
        "rail_shape": str(getattr(env, "rail_shape", "straight")),
        "rail_bend_amp": float(getattr(env, "rail_bend_amp", 0.0)),
        "rail_polyline": env.get_rail_polyline(num=64).tolist() if hasattr(env, "get_rail_polyline") else None,
        "cutpoints": cutpoints,
        "trajectory_points": int(len(planned_traj)),
        "feature_plot": None if feature_plot_path is None else str(Path(feature_plot_path).resolve()),
        "rollout_npz": str((out_dir / "s4_planned_rollout.npz").resolve()),
        "video": None if video_path is None else str(video_path.resolve()),
        "frames": int(latent.get("frames", 0)),
        "feature_overlay": bool(feature_overlay),
        "execution_joint_noise_std": float(execution_joint_noise_std),
        "execution_joint_noise_smooth": float(execution_joint_noise_smooth),
        "execution_noise_seed": None if execution_noise_seed is None else int(execution_noise_seed),
        "execution_normal_load_noise_std": float(execution_normal_load_noise_std),
        "execution_normal_load_noise_smooth": float(execution_normal_load_noise_smooth),
        "execution_normal_load_noise_seed": None if execution_normal_load_noise_seed is None else int(execution_normal_load_noise_seed),
        "ik_position_error_mean": None if "ik_position_error_world" not in obs else float(np.mean(obs["ik_position_error_world"])),
        "ik_position_error_max": None if "ik_position_error_world" not in obs else float(np.max(obs["ik_position_error_world"])),
        "surface_tilt_x": float(getattr(env, "surface_tilt_x", 0.0)),
        "surface_tilt_y": float(getattr(env, "surface_tilt_y", 0.0)),
    }
    summary_path = out_dir / "s4_planned_render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[saved] {summary_path}")
    print(f"[saved] features={feature_plot_path}, video={video_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan an S4 realistic trajectory from learned constraints and render PyBullet execution.")
    parser.add_argument("--constraints-json", required=True, help="Path to constraints.json or benchmark_results.json.")
    parser.add_argument("--benchmark-method", default=None)
    parser.add_argument("--benchmark-dataset", default=None)
    parser.add_argument("--benchmark-method-seed", type=int, default=None)
    parser.add_argument("--outdir", default="outputs/s4_planned_render")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gui", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--render-frame-stride", type=int, default=1)
    parser.add_argument("--realtime", type=int, default=0)
    parser.add_argument("--gui-hold-seconds", type=float, default=None)
    parser.add_argument("--camera-yaw", type=float, default=128.0)
    parser.add_argument("--camera-pitch", type=float, default=-29.0)
    parser.add_argument("--camera-distance", type=float, default=0.84)
    parser.add_argument("--camera-fov", type=float, default=42.0)
    parser.add_argument("--plot-features", type=int, default=1)
    parser.add_argument("--constraint-source", choices=["learned", "target"], default="learned")
    parser.add_argument("--speed-safety", type=float, default=1.0)
    parser.add_argument("--global-speed-max", type=float, default=0.012)
    parser.add_argument("--stage-lengths", default=None, help="Optional comma list, e.g. stage3:67,stage4:21.")
    parser.add_argument("--visualize-normal-load", type=int, default=0)
    parser.add_argument("--feature-overlay", type=int, default=1)
    parser.add_argument("--execution-joint-noise-std", type=float, default=0.002)
    parser.add_argument("--execution-joint-noise-smooth", type=float, default=0.90)
    parser.add_argument("--execution-noise-seed", type=int, default=None)
    parser.add_argument("--execution-normal-load-noise-std", type=float, default=0.025)
    parser.add_argument("--execution-normal-load-noise-smooth", type=float, default=0.85)
    parser.add_argument("--execution-normal-load-noise-seed", type=int, default=None)
    parser.add_argument("--planner", choices=["waypoint", "optimizer"], default="waypoint")
    parser.add_argument("--optimizer-iters", type=int, default=500)
    parser.add_argument("--optimizer-smooth-step", type=float, default=0.18)
    parser.add_argument("--optimizer-constraint-step", type=float, default=0.45)
    parser.add_argument("--optimizer-objective-step", type=float, default=0.08)
    parser.add_argument("--rail-shape", choices=["straight", "sine", "polyline"], default="straight")
    parser.add_argument("--rail-bend-amp", type=float, default=0.012)
    parser.add_argument("--rail-polyline", default=None, help="Optional transfer rail centerline as 'x,y;x,y;...'.")
    parser.add_argument("--surface-tilt-x", type=float, default=0.0, help="Surface height slope dz/dx in S4 coordinates.")
    parser.add_argument("--surface-tilt-y", type=float, default=0.0, help="Surface height slope dz/dy in S4 coordinates.")
    args = parser.parse_args()

    render_s4_planned_trajectory(
        constraints_json=args.constraints_json,
        outdir=args.outdir,
        seed=int(args.seed),
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
        plot_features=bool(args.plot_features),
        constraint_source=str(args.constraint_source),
        speed_safety=float(args.speed_safety),
        global_speed_max=None if float(args.global_speed_max) <= 0.0 else float(args.global_speed_max),
        stage_lengths=_parse_stage_lengths(args.stage_lengths),
        benchmark_method=args.benchmark_method,
        benchmark_dataset=args.benchmark_dataset,
        benchmark_method_seed=args.benchmark_method_seed,
        visualize_normal_load=bool(args.visualize_normal_load),
        feature_overlay=bool(args.feature_overlay),
        execution_joint_noise_std=float(args.execution_joint_noise_std),
        execution_joint_noise_smooth=float(args.execution_joint_noise_smooth),
        execution_noise_seed=args.execution_noise_seed,
        execution_normal_load_noise_std=float(args.execution_normal_load_noise_std),
        execution_normal_load_noise_smooth=float(args.execution_normal_load_noise_smooth),
        execution_normal_load_noise_seed=args.execution_normal_load_noise_seed,
        planner=str(args.planner),
        optimizer_iters=int(args.optimizer_iters),
        optimizer_smooth_step=float(args.optimizer_smooth_step),
        optimizer_constraint_step=float(args.optimizer_constraint_step),
        optimizer_objective_step=float(args.optimizer_objective_step),
        rail_shape=str(args.rail_shape),
        rail_bend_amp=float(args.rail_bend_amp),
        rail_polyline=_parse_rail_polyline(args.rail_polyline),
        surface_tilt_x=float(args.surface_tilt_x),
        surface_tilt_y=float(args.surface_tilt_y),
    )


if __name__ == "__main__":
    main()
