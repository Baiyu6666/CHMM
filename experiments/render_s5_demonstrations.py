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


def _parse_csv_ints(text: str | None) -> list[int] | None:
    if text is None or not str(text).strip():
        return None
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values, e.g. 0.18,0,0.04.")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _demo_indices(n_demos: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        return [int(np.clip(v, 0, max(0, int(n_demos) - 1))) for v in explicit]
    return list(range(int(n_demos)))


def _plot_feature_traces(
    *,
    features: np.ndarray,
    reference_features: np.ndarray | None = None,
    feature_schema: list[dict],
    cutpoints: list[int],
    output_path: str | Path,
    title: str,
) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib is required to plot S5 feature traces.")
    F = np.asarray(features, dtype=float)
    if F.ndim != 2 or F.shape[0] == 0:
        raise ValueError("features must have shape (T, D).")
    F_ref = None if reference_features is None else np.asarray(reference_features, dtype=float)
    if F_ref is not None and (F_ref.ndim != 2 or F_ref.shape[0] == 0):
        raise ValueError("reference_features must have shape (T, D).")
    schema = list(feature_schema or [])
    name_by_idx = {int(item.get("id", idx)): str(item.get("name", f"feature_{idx}")) for idx, item in enumerate(schema)}
    names = ["surf_dist", "normal_err", "speed", "ang_speed"]
    name_to_idx = {name: idx for idx, name in name_by_idx.items()}
    plot_items: list[tuple[int, str]] = []
    for name in names:
        if name in name_to_idx:
            plot_items.append((int(name_to_idx[name]), name))
    if not plot_items:
        plot_items = [(idx, name_by_idx.get(idx, f"feature_{idx}")) for idx in range(min(4, F.shape[1]))]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(F.shape[0])
    t_ref = None if F_ref is None else np.arange(F_ref.shape[0])
    fig, axes = plt.subplots(len(plot_items), 1, figsize=(10.2, 2.05 * len(plot_items)), sharex=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    for ax, (feat_idx, feat_name) in zip(axes, plot_items):
        if feat_idx >= F.shape[1]:
            continue
        if F_ref is not None and feat_idx < F_ref.shape[1]:
            ax.plot(t_ref, F_ref[:, feat_idx], color="#D97706", linewidth=1.65, label="planned/reference")
        ax.plot(t, F[:, feat_idx], color="#1D4ED8", linewidth=1.85, label="pybullet executed")
        for cp in cutpoints:
            if 0 <= int(cp) < len(t):
                ax.axvline(int(cp), color="#6B7280", linestyle="--", linewidth=1.0, alpha=0.72)
        ax.set_ylabel(feat_name)
        ax.grid(alpha=0.22)
    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("t")
    fig.suptitle(str(title), fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def _format_float(value, precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{int(precision)}g}"
    except (TypeError, ValueError):
        return "n/a"


def _feature_names(feature_schema: list[dict], dim: int) -> list[str]:
    names = [f"feature_{idx}" for idx in range(int(dim))]
    for idx, item in enumerate(feature_schema or []):
        col = int(item.get("column_idx", item.get("id", idx)))
        if 0 <= col < len(names):
            names[col] = str(item.get("name", names[col]))
    return names


def render_s5_demonstrations(
    *,
    n_demos: int,
    seed: int,
    demo_indices: list[int] | None,
    outdir: str | Path,
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
    rollout_backend: str,
    no_precheck: bool,
    no_filter: bool,
    no_ik_checks: bool,
) -> dict:
    out_dir = Path(outdir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _apply_default_s5_loader_config({})
    cfg["rollout_backend"] = str(rollout_backend)
    cfg["observation_backend"] = "pybullet" if str(rollout_backend).lower() == "pybullet" else "analytic_raw"
    cfg["eval_tag"] = "S5SphereInspectDemoRender"
    if bool(no_ik_checks) or bool(no_precheck):
        cfg["pybullet_precheck_ik_waypoints"] = False
    if bool(no_ik_checks) or bool(no_filter):
        cfg["pybullet_filter_ik_valid"] = False
    env = S5SphereInspectEnv(**cfg)

    selected = _demo_indices(int(n_demos), demo_indices)
    summaries: list[dict] = []
    pybullet_client = None
    pybullet_module = None
    if int(gui) == 2:
        try:
            import pybullet as pybullet_module
        except ModuleNotFoundError as exc:
            raise RuntimeError("pybullet is required for gui=2 playback.") from exc
        pybullet_client = int(pybullet_module.connect(pybullet_module.GUI))

    try:
        for local_idx, demo_idx in enumerate(selected, start=1):
            scene = env.sample_scene()
            scene["demo_index"] = int(demo_idx)
            demo_seed = env.demo_seed_for_index(int(seed), int(demo_idx))
            print(
                f"[demo {local_idx:02d}/{len(selected):02d}] "
                f"sampling idx={demo_idx}, base_seed={demo_seed}",
                flush=True,
            )

            def _print_attempt_progress(*, attempt: int, max_attempts: int, attempt_seed, report: dict):
                status = "accepted" if bool(report.get("valid", False)) else str(report.get("reason", "rejected"))
                prefix = (
                    f"  [attempt {attempt + 1:02d}/{int(max_attempts):02d}] "
                    f"seed={attempt_seed} {status} "
                    f"pos={_format_float(report.get('max_position_error'))} "
                    f"axis={_format_float(report.get('max_axis_error'))}"
                )
                if "waypoint_indices" in report:
                    print(
                        f"{prefix} "
                        f"c_axis={_format_float(report.get('constrained_max_axis_error'))} "
                        f"worst_t={report.get('worst_index', 'n/a')}",
                        flush=True,
                    )
                else:
                    print(
                        f"{prefix} "
                        f"c_axis={_format_float(report.get('constrained_max_axis_error'))} "
                        f"speed_ratio={_format_float(report.get('max_speed_ratio'))}",
                        flush=True,
                    )

            rollout_kwargs = {}
            if str(rollout_backend).lower() == "pybullet":
                rollout_kwargs["progress_callback"] = _print_attempt_progress
            latent = env.rollout_demo(scene, seed=demo_seed, **rollout_kwargs)
            obs = env.compute_observation(latent, scene)

            trajectory = np.asarray(obs["trajectory"], dtype=float)
            tool_axis = obs.get("tool_axis")
            joint_positions = obs.get("joint_positions")
            cutpoints = [int(v) for v in np.asarray(obs["true_cutpoints"], dtype=int).reshape(-1).tolist()]
            feature_plot_path = None
            reference_features = None
            feature_plot_series = ["pybullet executed"]
            if "reference_trajectory" in obs:
                reference_tool_axis = obs.get("reference_tool_axis")
                reference_features = env.compute_all_features_matrix(
                    np.asarray(obs["reference_trajectory"], dtype=float),
                    tool_axis=(None if reference_tool_axis is None else np.asarray(reference_tool_axis, dtype=float)),
                    use_cached=False,
                )
                feature_plot_series.insert(0, "planned/reference")
            if bool(plot_features):
                feature_plot_path = _plot_feature_traces(
                    features=np.asarray(obs["features"], dtype=float),
                    reference_features=None if reference_features is None else np.asarray(reference_features, dtype=float),
                    feature_schema=list(obs["feature_schema"]),
                    cutpoints=cutpoints,
                    output_path=out_dir / f"s5_demo_{int(demo_idx):02d}_features.png",
                    title=f"S5SphereInspect demo {int(demo_idx)} feature traces",
                )

            output_path = None
            if int(gui) == 1:
                output_path = out_dir / f"s5_demo_{int(demo_idx):02d}.mp4"
            effective_realtime = bool(realtime) or int(gui) == 2
            effective_hold_seconds = (
                (-1.0 if int(gui) == 2 else 2.0)
                if gui_hold_seconds is None
                else float(gui_hold_seconds)
            )
            feature_matrix = np.asarray(obs["features"], dtype=float)
            render_summary = env.render_episode(
                scene,
                trajectory,
                output_path,
                backend="pybullet_video",
                cutpoints=cutpoints,
                tool_axis=tool_axis,
                joint_positions=joint_positions,
                title=f"S5SphereInspect demo {int(demo_idx)}",
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
                camera_target=np.asarray(env.pybullet_world_center, dtype=float)
                + np.asarray(camera_target_offset, dtype=float),
                camera_fov=float(camera_fov),
                stage4_shell_offset=float(env.get_true_constraints()["surface_near_target"]),
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
                feature_overlay_features=feature_matrix,
                feature_overlay_names=_feature_names(list(obs["feature_schema"]), feature_matrix.shape[1]),
                feature_overlay_specs=list(env.get_constraint_specs()),
                feature_overlay_true_constraints=dict(env.true_constraints),
                feature_overlay_title="Executed demonstration feature profile",
                connect_client=pybullet_client is None,
            )
            summary = {
                "demo_local_index": int(local_idx),
                "demo_index": int(demo_idx),
                "seed": int(demo_seed),
                "rollout_backend": str(latent.get("rollout_backend", env.rollout_backend)),
                "observation_backend": str(obs["observation_spec"]["default_observation_backend"]),
                "trajectory_points": int(len(trajectory)),
                "true_cutpoints": cutpoints,
                "feature_plot": None if feature_plot_path is None else str(Path(feature_plot_path).resolve()),
                "feature_plot_series": feature_plot_series,
                "feature_overlay": bool(feature_overlay),
                "video": render_summary,
            }
            if "ik_position_error_world" in obs:
                ik_pos = np.asarray(obs["ik_position_error_world"], dtype=float)
                summary["ik_position_error_world"] = {
                    "mean": float(np.mean(ik_pos)),
                    "max": float(np.max(ik_pos)),
                }
            if "ik_axis_error" in obs:
                ik_axis = np.asarray(obs["ik_axis_error"], dtype=float)
                summary["ik_axis_error_rad"] = {
                    "mean": float(np.mean(ik_axis)),
                    "max": float(np.max(ik_axis)),
                }
            if "ik_filter" in obs:
                summary["ik_filter"] = dict(obs["ik_filter"])
            if "reference_seed" in obs:
                summary["reference_seed"] = obs["reference_seed"]
            summaries.append(summary)
            ik_filter = summary.get("ik_filter", {})
            attempt_text = ""
            if ik_filter:
                attempt_text = (
                    f", accepted_seed={summary.get('reference_seed')}, "
                    f"attempt={int(ik_filter.get('attempt', 0)) + 1}/{ik_filter.get('max_attempts')}"
                )
            print(
                f"[demo {local_idx:02d}/{len(selected):02d}] "
                f"idx={demo_idx}, points={len(trajectory)}, "
                f"features={feature_plot_path}, "
                f"video={render_summary.get('video_path')}, frames={render_summary.get('frames_written')}"
                f"{attempt_text}"
            )
    finally:
        if pybullet_client is not None and pybullet_module is not None:
            pybullet_module.disconnect(pybullet_client)

    out = {
        "task": "s5_demonstration_render",
        "gui": int(gui),
        "rollout_backend": str(rollout_backend),
        "pybullet_precheck_ik_waypoints": bool(env.pybullet_precheck_ik_waypoints),
        "pybullet_filter_ik_valid": bool(env.pybullet_filter_ik_valid),
        "seed": int(seed),
        "n_requested_demos": int(n_demos),
        "demo_indices": selected,
        "demos": summaries,
    }
    summary_path = out_dir / "s5_demonstration_render_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {summary_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Render S5SphereInspect demonstrations as PyBullet playback videos.")
    parser.add_argument("--n-demos", type=int, default=1, help="Number of demos to render.")
    parser.add_argument("--seed", type=int, default=7, help="Dataset/demo seed.")
    parser.add_argument("--demo-indices", type=str, default=None, help="Comma-separated demo indices. Defaults to 0..n_demos-1.")
    parser.add_argument("--outdir", default="outputs/s5_demo_render", help="Output directory.")
    parser.add_argument(
        "--gui",
        type=int,
        choices=[0, 1, 2],
        default=1,
        help="0: dry-run/no video; 1: DIRECT offscreen MP4; 2: GUI playback/no video.",
    )
    parser.add_argument("--rollout-backend", choices=["pybullet", "analytic"], default="pybullet")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--render-frame-stride", type=int, default=1)
    parser.add_argument("--realtime", type=int, default=0)
    parser.add_argument(
        "--gui-hold-seconds",
        type=float,
        default=None,
        help="Extra hold time after GUI playback. Defaults to waiting for SPACE for gui=2 and 2 seconds otherwise.",
    )
    parser.add_argument("--camera-yaw", type=float, default=90.0)
    parser.add_argument("--camera-pitch", type=float, default=-8.0)
    parser.add_argument("--camera-distance", type=float, default=1.45)
    parser.add_argument("--camera-fov", type=float, default=38.0)
    parser.add_argument(
        "--camera-target-offset",
        default="0.00,0.24,0.07",
        help="World-frame camera target offset from pybullet_world_center, e.g. '0,0.26,0.04'.",
    )
    parser.add_argument("--hide-gripper", type=int, default=1, help="1 to hide Robotiq gripper links and keep the URDF task tool visible.")
    parser.add_argument("--draw-tool-bar", type=int, default=0, help="1 to draw a detached debug bar instead of the URDF task tool.")
    parser.add_argument("--tool-bar-length", type=float, default=0.165)
    parser.add_argument("--tool-bar-radius", type=float, default=0.005)
    parser.add_argument("--draw-stage-trace", type=int, default=0)
    parser.add_argument("--draw-executed-trace", type=int, default=1)
    parser.add_argument("--trace-stride", type=int, default=1)
    parser.add_argument("--trace-width", type=float, default=3.0)
    parser.add_argument("--draw-current-marker", type=int, default=0)
    parser.add_argument("--plot-features", type=int, default=1, help="1 to save per-demo S5 feature trace PNGs.")
    parser.add_argument("--feature-overlay", type=int, default=1, help="1 to overlay feature traces on rendered videos.")
    parser.add_argument("--no-precheck", action="store_true", help="Disable waypoint IK precheck before full PyBullet rollout.")
    parser.add_argument("--no-filter", action="store_true", help="Disable final IK/rollout rejection filter.")
    parser.add_argument("--no-ik-checks", action="store_true", help="Disable both waypoint precheck and final IK/rollout filter.")
    args = parser.parse_args()

    render_s5_demonstrations(
        n_demos=int(args.n_demos),
        seed=int(args.seed),
        demo_indices=_parse_csv_ints(args.demo_indices),
        outdir=args.outdir,
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
        rollout_backend=str(args.rollout_backend),
        no_precheck=bool(args.no_precheck),
        no_filter=bool(args.no_filter),
        no_ik_checks=bool(args.no_ik_checks),
    )


if __name__ == "__main__":
    main()
