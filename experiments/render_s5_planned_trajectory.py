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

from envs.S5SphereInspect import S5SphereInspectEnv, _apply_default_s5_loader_config


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
    if "ConstraintLearnedValueMatrix" not in payload and p.name == "constraints.json":
        metrics_path = p.with_name("metrics.json")
        if metrics_path.exists():
            metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "ConstraintLearnedValueMatrix" in metrics_payload:
                return metrics_payload, metrics_path
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


def _constraint_values_from_payload(payload: dict, env: S5SphereInspectEnv, *, fallback_target: bool = False) -> dict:
    feature_names = list(payload.get("ConstraintFeatureNames", []))
    if not feature_names:
        feature_names = ["surf_dist", "normal_err", "speed", "ang_speed", "start_dist", "goal_dist"]
    learned = payload.get("ConstraintLearnedValueMatrix")
    if learned is None:
        if not fallback_target:
            raise ValueError(
                "Constraint JSON does not contain ConstraintLearnedValueMatrix. "
                "This older constraints.json only stores target/error values, and the signed learned "
                "threshold cannot be reconstructed from that. Rerun the benchmark with the updated "
                "metrics code, or pass --fallback-target 1 "
                "to render a true-constraint reference instead of a learned-constraint plan."
            )
        learned = payload.get("ConstraintTargetMatrix")
    target = payload.get("ConstraintTargetMatrix")
    specs = list(payload.get("constraint_specs") or env.get_constraint_specs())

    out = {}
    for spec in specs:
        name = str(spec.get("feature_name", ""))
        if name not in feature_names:
            continue
        stage_idx = int(spec.get("stage", 0))
        feature_idx = int(feature_names.index(name))
        value = _finite_matrix_value(learned, stage_idx, feature_idx)
        if value is None and bool(fallback_target):
            value = _finite_matrix_value(target, stage_idx, feature_idx)
        if value is None:
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
    hide_gripper: bool,
    draw_tool_bar: bool,
    tool_bar_length: float,
    tool_bar_radius: float,
    plot_features: bool,
    no_precheck: bool,
    no_filter: bool,
    fallback_target: bool,
    speed_safety: float,
    stage_lengths: dict | None,
    benchmark_method: str | None,
    benchmark_dataset: str | None,
    benchmark_method_seed: int | None,
) -> dict:
    out_dir = Path(outdir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _apply_default_s5_loader_config({})
    cfg["rollout_backend"] = "pybullet"
    cfg["observation_backend"] = "pybullet"
    cfg["eval_tag"] = "S5SphereInspectPlannedRender"
    if bool(no_precheck):
        cfg["pybullet_precheck_ik_waypoints"] = False
    if bool(no_filter):
        cfg["pybullet_filter_ik_valid"] = False
    env = S5SphereInspectEnv(**cfg)

    raw_payload, resolved_constraints_path = _load_constraint_payload(constraints_json)
    payload = _select_constraint_payload(
        raw_payload,
        method=benchmark_method,
        dataset=benchmark_dataset,
        method_seed=benchmark_method_seed,
    )
    constraint_values = _constraint_values_from_payload(payload, env, fallback_target=bool(fallback_target))

    scene = env.sample_scene()
    planned = env.plan_episode_from_constraints(
        scene,
        constraint_values,
        seed=int(seed),
        stage_lengths=stage_lengths,
        speed_safety=float(speed_safety),
    )
    print(f"[plan] points={len(planned['trajectory'])}, cutpoints={planned['true_cutpoints'].tolist()}")
    print(f"[plan] constraints={planned['constraint_values']}")

    latent = env.execute_plan_pybullet(
        scene,
        planned,
        precheck=not bool(no_precheck),
        filter_valid=not bool(no_filter),
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
            output_path=out_dir / "s5_planned_features.png",
            title="S5 planned trajectory all-feature profiles",
        )

    np.savez_compressed(
        out_dir / "s5_planned_rollout.npz",
        planned_trajectory=np.asarray(planned["trajectory"], dtype=float),
        planned_tool_axis=np.asarray(planned["tool_axis"], dtype=float),
        executed_trajectory=np.asarray(obs["trajectory"], dtype=float),
        executed_tool_axis=np.asarray(obs["tool_axis"], dtype=float),
        planned_features=np.asarray(planned_features, dtype=float),
        executed_features=np.asarray(obs["features"], dtype=float),
        cutpoints=np.asarray(planned["true_cutpoints"], dtype=int),
    )

    output_path = None
    if int(gui) == 1:
        output_path = out_dir / "s5_planned_pybullet.mp4"
    effective_realtime = bool(realtime) or int(gui) == 2
    effective_hold_seconds = (-1.0 if int(gui) == 2 else 0.0) if gui_hold_seconds is None else float(gui_hold_seconds)
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
        camera_fov=float(camera_fov),
        hide_gripper=bool(hide_gripper),
        draw_tool_bar=bool(draw_tool_bar),
        tool_bar_length=float(tool_bar_length),
        tool_bar_radius=float(tool_bar_radius),
    )

    summary = {
        "task": "s5_planned_trajectory_render",
        "constraints_json": str(Path(constraints_json)),
        "resolved_constraints_payload": str(resolved_constraints_path),
        "fallback_target": bool(fallback_target),
        "seed": int(seed),
        "gui": int(gui),
        "constraint_values": dict(planned["constraint_values"]),
        "stage_lengths": dict(planned["stage_lengths"]),
        "cutpoints": cutpoints,
        "trajectory_points": int(len(planned["trajectory"])),
        "feature_plot": None if feature_plot_path is None else str(Path(feature_plot_path).resolve()),
        "rollout_npz": str((out_dir / "s5_planned_rollout.npz").resolve()),
        "video": render_summary,
        "ik_filter": dict(latent.get("ik_filter", {})),
    }
    summary_path = out_dir / "s5_planned_render_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {summary_path}")
    print(f"[saved] features={feature_plot_path}, video={render_summary.get('video_path')}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan an S5 trajectory from learned constraints and render PyBullet execution.")
    parser.add_argument("--constraints-json", required=True, help="Path to constraints.json or benchmark_results.json.")
    parser.add_argument("--benchmark-method", default=None, help="Method row to select when --constraints-json is benchmark_results.json.")
    parser.add_argument("--benchmark-dataset", default=None, help="Dataset row to select when --constraints-json is benchmark_results.json.")
    parser.add_argument("--benchmark-method-seed", type=int, default=None, help="Method seed row to select from benchmark_results.json.")
    parser.add_argument("--outdir", default="outputs/s5_planned_render")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gui", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--render-frame-stride", type=int, default=1)
    parser.add_argument("--realtime", type=int, default=0)
    parser.add_argument("--gui-hold-seconds", type=float, default=None)
    parser.add_argument("--camera-yaw", type=float, default=90.0)
    parser.add_argument("--camera-pitch", type=float, default=-34.0)
    parser.add_argument("--camera-distance", type=float, default=1.62)
    parser.add_argument("--camera-fov", type=float, default=38.0)
    parser.add_argument("--hide-gripper", type=int, default=1)
    parser.add_argument("--draw-tool-bar", type=int, default=0)
    parser.add_argument("--tool-bar-length", type=float, default=0.105)
    parser.add_argument("--tool-bar-radius", type=float, default=0.005)
    parser.add_argument("--plot-features", type=int, default=1)
    parser.add_argument("--no-precheck", action="store_true")
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--fallback-target", type=int, default=0, help="Use true target matrix if learned matrix is missing.")
    parser.add_argument("--speed-safety", type=float, default=1.0)
    parser.add_argument("--stage-lengths", default=None, help="Optional comma list, e.g. stage2:34,stage4:18.")
    args = parser.parse_args()

    render_s5_planned_trajectory(
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
        hide_gripper=bool(args.hide_gripper),
        draw_tool_bar=bool(args.draw_tool_bar),
        tool_bar_length=float(args.tool_bar_length),
        tool_bar_radius=float(args.tool_bar_radius),
        plot_features=bool(args.plot_features),
        no_precheck=bool(args.no_precheck),
        no_filter=bool(args.no_filter),
        fallback_target=bool(args.fallback_target),
        speed_safety=float(args.speed_safety),
        stage_lengths=_parse_stage_lengths(args.stage_lengths),
        benchmark_method=args.benchmark_method,
        benchmark_dataset=args.benchmark_dataset,
        benchmark_method_seed=args.benchmark_method_seed,
    )


if __name__ == "__main__":
    main()
