from __future__ import annotations

import argparse
import json
import shutil
import subprocess
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

from envs.S5SphereInspect import _apply_default_s5_loader_config, load_S5SphereInspect
from experiments.render_metrics import constraint_violation_stats, parse_int_list, print_render_violation_rates


_DISPLAY_UNITS = {
    "surf_dist": "mm",
    "normal_err": "deg",
    "speed": "mm/s",
    "ang_speed": "deg/s",
    "start_dist": "mm",
    "goal_dist": "mm",
}

_DISPLAY_SCALE = {
    "surf_dist": 1000.0,
    "normal_err": 180.0 / np.pi,
    "speed": 1000.0,
    "ang_speed": 180.0 / np.pi,
    "start_dist": 1000.0,
    "goal_dist": 1000.0,
}


def _parse_csv_ints(text: str | None) -> list[int] | None:
    if text is None or not str(text).strip():
        return None
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_frame_indices(text: str | None) -> list[int]:
    return parse_int_list(text)


def _parse_vec3(text: str) -> tuple[float, float, float]:
    parts = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Expected three comma-separated values, e.g. 0.18,0,0.04.")
    return float(parts[0]), float(parts[1]), float(parts[2])


def _concat_mp4_segments(segment_paths: list[Path], output_path: Path) -> Path | None:
    if not segment_paths:
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to concatenate multi-demo S5 videos.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_list_path = output_path.with_suffix(".concat.txt")

    def _quote_concat_path(path: Path) -> str:
        return "'" + str(path.resolve()).replace("'", "'\\''") + "'"

    concat_list_path.write_text(
        "".join(f"file {_quote_concat_path(path)}\n" for path in segment_paths),
        encoding="utf-8",
    )
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c",
        "copy",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    finally:
        try:
            concat_list_path.unlink()
        except OSError:
            pass
    return output_path


def _demo_indices(n_demos: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        indices = [int(v) for v in explicit]
        if any(v < 0 for v in indices):
            raise ValueError(f"demo indices must be non-negative: {indices}")
        return indices
    return list(range(int(n_demos)))


def _bundle_scene(bundle, demo_idx: int) -> dict:
    scene_specs = list(bundle.meta.get("scene_specs", [])) if getattr(bundle, "meta", None) else []
    if 0 <= int(demo_idx) < len(scene_specs):
        return dict(scene_specs[int(demo_idx)])
    scene = bundle.env.sample_scene()
    scene["demo_index"] = int(demo_idx)
    return scene


def _bundle_tool_axis(env, trajectory: np.ndarray):
    lookup = getattr(env, "_lookup_cached_tool_axis_trace", None)
    if callable(lookup):
        axis = lookup(np.asarray(trajectory, dtype=float))
        if axis is not None:
            return np.asarray(axis, dtype=float)
    return None


def _training_feature_names_from_config() -> list[str] | None:
    cfg_path = PROJECT_ROOT / "configs" / "envs" / "S5SphereInspect.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    selected = (
        cfg.get("method_overrides", {})
        .get("swcl", {})
        .get("selected_raw_feature_ids")
    )
    if not selected:
        return None
    return [str(name) for name in selected]


def _filter_features_for_display(
    features: np.ndarray,
    feature_schema: list[dict],
    selected_names: list[str] | None,
) -> tuple[np.ndarray, list[dict]]:
    F = np.asarray(features, dtype=float)
    schema = list(feature_schema or [])
    if not selected_names:
        return F, schema
    name_to_col = {}
    for idx, item in enumerate(schema):
        name = str(item.get("name", f"feature_{idx}"))
        col = int(item.get("column_idx", item.get("id", idx)))
        if 0 <= col < F.shape[1]:
            name_to_col[name] = col
    cols = [name_to_col[name] for name in selected_names if name in name_to_col]
    if not cols:
        return F, schema
    scales = np.asarray([float(_DISPLAY_SCALE.get(name, 1.0)) for name in selected_names if name in name_to_col], dtype=float)
    filtered_schema = []
    for new_idx, name in enumerate(selected_names):
        if name not in name_to_col:
            continue
        old = next((dict(item) for item in schema if str(item.get("name", "")) == name), {"name": name})
        old["id"] = int(new_idx)
        old["column_idx"] = int(new_idx)
        if name in _DISPLAY_UNITS:
            old["unit"] = _DISPLAY_UNITS[name]
        filtered_schema.append(old)
    return F[:, cols] * scales[None, :], filtered_schema


def _display_units_from_schema(feature_schema: list[dict]) -> dict[str, str]:
    out = {}
    for item in feature_schema or []:
        name = str(item.get("name", ""))
        unit = str(item.get("unit", ""))
        if name and unit:
            out[name] = unit
    return out


def _scale_true_constraints_for_display(
    true_constraints: dict,
    constraint_specs: list[dict],
) -> dict:
    scaled = dict(true_constraints or {})
    for spec in constraint_specs:
        name = str(spec.get("feature_name", ""))
        key = str(spec.get("oracle_key", ""))
        if not key or key not in scaled:
            continue
        try:
            scaled[key] = float(scaled[key]) * float(_DISPLAY_SCALE.get(name, 1.0))
        except (TypeError, ValueError):
            pass
    return scaled


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
    names = [name_by_idx[idx] for idx in sorted(name_by_idx)]
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
        unit = next((str(item.get("unit", "")) for item in schema if str(item.get("name", "")) == feat_name), "")
        ax.set_ylabel(feat_name if not unit else f"{feat_name} [{unit}]")
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
    playback_speed: float,
    playback_label: str | None,
    rollout_backend: str,
    no_precheck: bool,
    no_filter: bool,
    no_ik_checks: bool,
    save_frame_indices: list[int],
) -> dict:
    out_dir = Path(outdir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    playback_speed = float(max(float(playback_speed), 1e-6))

    cfg = _apply_default_s5_loader_config({})
    cfg["rollout_backend"] = str(rollout_backend)
    cfg["observation_backend"] = "pybullet" if str(rollout_backend).lower() == "pybullet" else "analytic_raw"
    cfg["eval_tag"] = "S5SphereInspect"
    cfg["cache_demos"] = True
    if bool(no_ik_checks) or bool(no_precheck):
        cfg["pybullet_precheck_ik_waypoints"] = False
    if bool(no_ik_checks) or bool(no_filter):
        cfg["pybullet_filter_ik_valid"] = False

    selected = _demo_indices(int(n_demos), demo_indices)
    bundle_n_demos = max(int(n_demos), max(selected) + 1 if selected else 0)
    bundle = load_S5SphereInspect(
        n_demos=int(bundle_n_demos),
        seed=int(seed),
        env_kwargs=cfg,
        demo_kwargs={},
    )
    env = bundle.env
    playback_real_time_multiplier = (
        float(getattr(env, "dt", 1.0))
        * float(fps)
        * float(playback_speed)
        * float(max(int(render_frame_stride), 1))
    )
    if playback_label is None:
        playback_label = f"{float(playback_real_time_multiplier):g}x real time"
    display_feature_names = _training_feature_names_from_config()
    cache_meta = None if not getattr(bundle, "meta", None) else bundle.meta.get("demo_cache")
    if cache_meta:
        status = "hit" if bool(cache_meta.get("hit", False)) else "miss"
        print(f"[S5 demo cache] {status}: {cache_meta.get('path')}", flush=True)

    summaries: list[dict] = []
    violation_features: list[np.ndarray] = []
    violation_cutpoints: list[list[int]] = []
    combine_video = int(gui) == 1 and len(selected) > 1
    segment_paths: list[Path] = []
    segment_dir = out_dir / "_s5_demo_segments"
    if combine_video:
        if segment_dir.exists():
            shutil.rmtree(segment_dir)
        segment_dir.mkdir(parents=True, exist_ok=True)
    pybullet_client = None
    pybullet_module = None
    if int(gui) == 2:
        try:
            import pybullet as pybullet_module
        except ModuleNotFoundError as exc:
            raise RuntimeError("pybullet is required for gui=2 playback.") from exc
        pybullet_client = int(pybullet_module.connect(pybullet_module.GUI))

    try:
        for selection_index, demo_idx in enumerate(selected):
            scene = _bundle_scene(bundle, int(demo_idx))
            trajectory = np.asarray(bundle.demos[int(demo_idx)], dtype=float)
            cutpoints = [int(v) for v in np.asarray(bundle.true_cutpoints[int(demo_idx)], dtype=int).reshape(-1).tolist()]
            tool_axis = _bundle_tool_axis(env, trajectory)
            latent = {
                "trajectory": trajectory,
                "true_cutpoints": np.asarray(cutpoints, dtype=int),
                "tool_axis": tool_axis,
                "rollout_backend": str(rollout_backend),
                "observation_backend": cfg["observation_backend"],
            }
            obs = env.compute_observation(latent, scene)
            if (
                str(rollout_backend).lower() == "pybullet"
                and obs.get("joint_positions") is None
                and obs.get("tool_axis") is not None
            ):
                replay_latent = env.execute_plan_pybullet(
                    scene,
                    {
                        "trajectory": np.asarray(obs["trajectory"], dtype=float),
                        "tool_axis": np.asarray(obs["tool_axis"], dtype=float),
                        "true_cutpoints": np.asarray(obs["true_cutpoints"], dtype=int),
                        "planner": "s5_demo_pybullet_replay",
                    },
                    precheck=False,
                    filter_valid=False,
                    execution_joint_noise_std=0.0,
                )
                obs = env.compute_observation(replay_latent, scene)
            print(
                f"[demo {int(demo_idx):02d}] loaded from training demo bundle "
                f"(item {selection_index + 1}/{len(selected)})",
                flush=True,
            )

            trajectory = np.asarray(obs["trajectory"], dtype=float)
            tool_axis = obs.get("tool_axis")
            joint_positions = obs.get("joint_positions")
            cutpoints = [int(v) for v in np.asarray(obs["true_cutpoints"], dtype=int).reshape(-1).tolist()]
            feature_matrix = np.asarray(obs["features"], dtype=float)
            display_feature_matrix, display_feature_schema = _filter_features_for_display(
                feature_matrix,
                list(obs["feature_schema"]),
                display_feature_names,
            )
            display_constraint_specs = [
                dict(spec)
                for spec in env.get_constraint_specs()
                if display_feature_names is None or str(spec.get("feature_name", "")) in set(display_feature_names)
            ]
            display_true_constraints = _scale_true_constraints_for_display(
                dict(env.true_constraints),
                display_constraint_specs,
            )
            feature_plot_path = None
            reference_features = None
            display_reference_features = None
            feature_plot_series = ["pybullet executed"]
            if "reference_trajectory" in obs:
                reference_tool_axis = obs.get("reference_tool_axis")
                reference_features = env.compute_all_features_matrix(
                    np.asarray(obs["reference_trajectory"], dtype=float),
                    tool_axis=(None if reference_tool_axis is None else np.asarray(reference_tool_axis, dtype=float)),
                    use_cached=False,
                )
                display_reference_features, _ = _filter_features_for_display(
                    reference_features,
                    list(obs["feature_schema"]),
                    display_feature_names,
                )
                feature_plot_series.insert(0, "planned/reference")
            if bool(plot_features):
                feature_plot_path = _plot_feature_traces(
                    features=display_feature_matrix,
                    reference_features=display_reference_features,
                    feature_schema=display_feature_schema,
                    cutpoints=cutpoints,
                    output_path=out_dir / f"s5_demo_{int(demo_idx):02d}_features.png",
                    title=f"S5SphereInspect demo {int(demo_idx)} feature traces",
                )

            output_path = None
            if int(gui) == 1:
                if combine_video:
                    output_path = segment_dir / f"s5_demo_{int(demo_idx):02d}.mp4"
                else:
                    output_path = out_dir / f"s5_demo_{int(demo_idx):02d}.mp4"
            effective_realtime = bool(realtime) or int(gui) == 2
            if combine_video:
                pause_seconds = 1.5 if gui_hold_seconds is None else float(gui_hold_seconds)
                effective_hold_seconds = (
                    0.0
                    if int(selection_index) == len(selected) - 1
                    else float(max(0.0, pause_seconds))
                )
            else:
                effective_hold_seconds = (
                    (-1.0 if int(gui) == 2 else 2.0)
                    if gui_hold_seconds is None
                    else float(gui_hold_seconds)
                )
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
                feature_overlay_features=display_feature_matrix,
                feature_overlay_names=_feature_names(display_feature_schema, display_feature_matrix.shape[1]),
                feature_overlay_units=_display_units_from_schema(display_feature_schema),
                feature_overlay_specs=display_constraint_specs,
                feature_overlay_true_constraints=display_true_constraints,
                feature_overlay_title="Executed demonstration feature profile",
                playback_speed=float(playback_speed),
                playback_label=playback_label,
                connect_client=pybullet_client is None,
                save_frame_indices=save_frame_indices,
                save_frame_dir=out_dir,
                save_frame_prefix=f"s5_demo_{int(demo_idx):02d}",
            )
            if combine_video and output_path is not None:
                segment_paths.append(Path(output_path))
            violation_stats = constraint_violation_stats(
                features_list=[feature_matrix],
                cutpoints_list=[cutpoints],
                feature_schema=env.get_feature_schema(),
                constraint_specs=env.get_constraint_specs(),
                true_constraints=env.get_true_constraints(),
                equality_tolerance=1e-4,
            )
            violation_features.append(feature_matrix)
            violation_cutpoints.append(cutpoints)
            summary = {
                "demo_index": int(demo_idx),
                "seed": int(seed),
                "rollout_backend": str(latent.get("rollout_backend", env.rollout_backend)),
                "observation_backend": str(obs["observation_spec"]["default_observation_backend"]),
                "trajectory_points": int(len(trajectory)),
                "true_cutpoints": cutpoints,
                "feature_plot": None if feature_plot_path is None else str(Path(feature_plot_path).resolve()),
                "feature_plot_series": feature_plot_series,
                "feature_overlay": bool(feature_overlay),
                "playback_speed": float(playback_speed),
                "playback_real_time_multiplier": float(playback_real_time_multiplier),
                "playback_label": playback_label,
                "video": render_summary,
                "saved_frames": list(render_summary.get("saved_frames", [])),
                "constraint_violation": violation_stats,
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
                f"[demo {int(demo_idx):02d}] item={selection_index + 1}/{len(selected)}, "
                f"points={len(trajectory)}, "
                f"features={feature_plot_path}, "
                f"video={render_summary.get('video_path')}, frames={render_summary.get('frames_written')}"
                f"{attempt_text}"
            )
    finally:
        if pybullet_client is not None and pybullet_module is not None:
            pybullet_module.disconnect(pybullet_client)

    combined_video_path = None
    if combine_video:
        combined_video_path = _concat_mp4_segments(segment_paths, out_dir / "s5_demonstrations.mp4")
        if combined_video_path is not None:
            print(f"[saved] {combined_video_path}", flush=True)

    out = {
        "task": "s5_demonstration_render",
        "gui": int(gui),
        "rollout_backend": str(rollout_backend),
        "pybullet_precheck_ik_waypoints": bool(env.pybullet_precheck_ik_waypoints),
        "pybullet_filter_ik_valid": bool(env.pybullet_filter_ik_valid),
        "seed": int(seed),
        "n_requested_demos": int(n_demos),
        "demo_indices": selected,
        "demo_cache": cache_meta,
        "combined_video": None if combined_video_path is None else str(Path(combined_video_path).resolve()),
        "combined_video_segments": [str(path.resolve()) for path in segment_paths],
        "inter_demo_pause_seconds": (
            None if not combine_video else float(1.5 if gui_hold_seconds is None else gui_hold_seconds)
        ),
        "playback_speed": float(playback_speed),
        "playback_real_time_multiplier": float(playback_real_time_multiplier),
        "playback_label": playback_label,
        "constraint_violation": constraint_violation_stats(
            features_list=violation_features,
            cutpoints_list=violation_cutpoints,
            feature_schema=env.get_feature_schema(),
            constraint_specs=env.get_constraint_specs(),
            true_constraints=env.get_true_constraints(),
            equality_tolerance=1e-4,
        ),
        "demos": summaries,
    }
    summary_path = out_dir / "s5_demonstration_render_summary.json"
    summary_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[saved] {summary_path}")
    print_render_violation_rates(out)
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
    parser.add_argument("--width", type=int, default=1360)
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
    parser.add_argument("--camera-pitch", type=float, default=-16.0)
    parser.add_argument("--camera-distance", type=float, default=1.35)
    parser.add_argument("--camera-fov", type=float, default=38.0)
    parser.add_argument(
        "--camera-target-offset",
        default="0.00,0.24,0.20",
        help="World-frame camera target offset from pybullet_world_center, e.g. '0,0.26,0.04'.",
    )
    parser.add_argument("--hide-gripper", type=int, default=1, help="1 to hide Robotiq gripper links.")
    parser.add_argument("--draw-tool-bar", type=int, default=1, help="1 to draw a detached debug bar instead of the URDF task tool.")
    parser.add_argument("--tool-bar-length", type=float, default=0.205)
    parser.add_argument("--tool-bar-radius", type=float, default=0.005)
    parser.add_argument("--draw-stage-trace", type=int, default=0)
    parser.add_argument("--draw-executed-trace", type=int, default=1)
    parser.add_argument("--trace-stride", type=int, default=1)
    parser.add_argument("--trace-width", type=float, default=3.0)
    parser.add_argument("--draw-current-marker", type=int, default=0)
    parser.add_argument("--plot-features", type=int, default=1, help="1 to save per-demo S5 feature trace PNGs.")
    parser.add_argument("--feature-overlay", type=int, default=1, help="1 to overlay feature traces on rendered videos.")
    parser.add_argument("--playback-speed", type=float, default=1.0, help="MP4 playback speed multiplier; the corner label reports real-time speed.")
    parser.add_argument("--playback-label", default=None, help="Optional lower-left video label. Defaults to '<multiplier>x real time'.")
    parser.add_argument("--save-frame-indices", default=None, help="Comma-separated source frame indices to save as PNGs in --outdir, e.g. 0,80,157.")
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
        playback_speed=float(args.playback_speed),
        playback_label=args.playback_label,
        rollout_backend=str(args.rollout_backend),
        no_precheck=bool(args.no_precheck),
        no_filter=bool(args.no_filter),
        no_ik_checks=bool(args.no_ik_checks),
        save_frame_indices=_parse_frame_indices(args.save_frame_indices),
    )


if __name__ == "__main__":
    main()
