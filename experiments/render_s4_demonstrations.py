from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.S4SlideInsert import S4SlideInsertEnv
from envs.S4SlideInsert import S4PyBulletPlaybackSession
from experiments.render_metrics import (
    concat_mp4_files,
    constraint_violation_stats,
    parse_int_list,
    print_render_violation_rates,
)

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None


def _plot_features(env, planned, executed, cutpoints, out_path: Path, planned_normal_load=None, executed_normal_load=None):
    if plt is None:
        return None
    names = [spec['name'] for spec in env.get_feature_schema()]
    if planned_normal_load is not None:
        load = np.asarray(planned_normal_load, dtype=float)
        if len(load) == len(planned):
            env.register_normal_load_trace(planned, load)
    planned_f = env.compute_all_features_matrix(planned)
    if executed_normal_load is not None:
        load = np.asarray(executed_normal_load, dtype=float)
        if len(load) == len(executed):
            env.register_normal_load_trace(executed, load)
    executed_f = env.compute_all_features_matrix(executed)
    n = len(names)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 2.35 * rows), squeeze=False)
    t = np.arange(len(executed))
    for j, name in enumerate(names):
        ax = axes[j // cols][j % cols]
        ax.plot(t, planned_f[:, j], color='#D97706', linewidth=1.4, label='planned')
        ax.plot(t, executed_f[:, j], color='#2563EB', linewidth=1.2, label='pybullet')
        for cp in np.asarray(cutpoints, dtype=int).reshape(-1):
            ax.axvline(int(cp), color='0.2', linestyle='--', linewidth=0.7, alpha=0.35)
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.18)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis('off')
    axes[0][0].legend(fontsize=8, loc='best')
    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def _parse_indices(text: str | None, n: int):
    if not text:
        return list(range(n))
    return [int(x.strip()) for x in text.split(',') if x.strip()]


def _parse_frame_indices(text: str | None) -> list[int]:
    return parse_int_list(text)


def _parse_optional_vec3(text: str | None):
    if text is None or not str(text).strip():
        return None
    vals = [float(v.strip()) for v in str(text).split(",") if v.strip()]
    if len(vals) != 3:
        raise argparse.ArgumentTypeError(f"Invalid vec3 {text!r}; expected x,y,z.")
    return tuple(vals)


def main():
    parser = argparse.ArgumentParser(description='Render S4SlideInsert PyBullet demonstrations.')
    parser.add_argument('--n-demos', '--n_demos', dest='n_demos', type=int, default=1)
    parser.add_argument('--seed', type=int, default=12342)
    parser.add_argument('--demo-indices', default=None)
    parser.add_argument('--outdir', default='outputs/s4_realistic_render')
    parser.add_argument('--fps', type=float, default=15.0)
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=900)
    parser.add_argument('--render-frame-stride', type=int, default=1)
    parser.add_argument('--video-end-hold-seconds', type=float, default=2.0)
    parser.add_argument(
        '--gui',
        type=int,
        choices=[0, 1, 2],
        default=1,
        help='0: dry-run/no video; 1: DIRECT offscreen MP4; 2: GUI playback/no video.',
    )
    parser.add_argument('--realtime', type=int, default=0)
    parser.add_argument(
        '--gui-hold-seconds',
        type=float,
        default=None,
        help='Extra hold time after GUI playback. Defaults to waiting for SPACE for gui=2 and 0 otherwise.',
    )
    parser.add_argument('--camera-yaw', type=float, default=38.0)
    parser.add_argument('--camera-pitch', type=float, default=-33.0)
    parser.add_argument('--camera-distance', type=float, default=0.90)
    parser.add_argument('--camera-fov', type=float, default=42.0)
    parser.add_argument(
        '--camera-target',
        default='0.72,0.14,0.54',
        help='World-frame camera target as x,y,z. Increase x/y to pan the S4 view right, moving the robot toward the left side of the video area.',
    )
    parser.add_argument('--plot-features', type=int, default=1)
    parser.add_argument('--visualize-normal-force', '--visualize-normal-load', dest='visualize_normal_load', type=int, default=0)
    parser.add_argument('--feature-overlay', type=int, default=1)
    parser.add_argument('--save-frame-indices', default=None, help='Comma-separated source frame indices to save as PNGs in --outdir, e.g. 0,80,157.')
    parser.add_argument('--execution-normal-force-noise-std', '--execution-normal-load-noise-std', dest='execution_normal_load_noise_std', type=float, default=0.025)
    parser.add_argument('--execution-normal-force-noise-smooth', '--execution-normal-load-noise-smooth', dest='execution_normal_load_noise_smooth', type=float, default=0.85)
    parser.add_argument('--execution-normal-force-noise-seed', '--execution-normal-load-noise-seed', dest='execution_normal_load_noise_seed', type=int, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    camera_target = _parse_optional_vec3(args.camera_target)
    env = S4SlideInsertEnv(
        rollout_backend='analytic',
        observation_backend='analytic',
        pybullet_camera_yaw=float(args.camera_yaw),
        pybullet_camera_pitch=float(args.camera_pitch),
        pybullet_camera_distance=float(args.camera_distance),
        pybullet_camera_fov=float(args.camera_fov),
        **({} if camera_target is None else {'pybullet_camera_target': camera_target}),
        pybullet_render_width=int(args.width),
        pybullet_render_height=int(args.height),
    )
    indices = _parse_indices(args.demo_indices, int(args.n_demos))
    save_frame_indices = _parse_frame_indices(args.save_frame_indices)
    summaries = []
    violation_features = []
    violation_cutpoints = []
    gui_mode = int(args.gui)
    write_video = gui_mode == 1
    combine_video = write_video and len(indices) > 1
    combined_video_target = outdir / 's4_demonstrations.mp4' if write_video else None
    segment_dir = outdir / '._s4_demo_segments'
    segment_paths = []
    if write_video:
        for old_segment in outdir.glob('demo_*.mp4'):
            try:
                old_segment.unlink()
            except OSError:
                pass
    if combine_video:
        if segment_dir.exists():
            shutil.rmtree(segment_dir)
        segment_dir.mkdir(parents=True, exist_ok=True)
    session = S4PyBulletPlaybackSession(env, force_gui=(gui_mode == 2)) if gui_mode in (1, 2) else None
    try:
        for local_idx, demo_idx in enumerate(indices):
            scene = env.sample_scene()
            scene['demo_index'] = int(demo_idx)
            seed = int(args.seed) + int(demo_idx)
            if not write_video:
                video_path = None
            elif combine_video:
                video_path = segment_dir / f'demo_{demo_idx:02d}.mp4'
            else:
                video_path = combined_video_target
            effective_realtime = bool(args.realtime) or gui_mode == 2
            if combine_video:
                pause_seconds = float(args.video_end_hold_seconds) if args.gui_hold_seconds is None else float(args.gui_hold_seconds)
                effective_hold_seconds = 0.0 if local_idx == len(indices) - 1 else float(max(0.0, pause_seconds))
            else:
                effective_hold_seconds = (-1.0 if gui_mode == 2 else 0.0) if args.gui_hold_seconds is None else float(args.gui_hold_seconds)
            print(f"[demo {demo_idx:02d}] sampling seed={seed}", flush=True)
            if session is None:
                latent = env.rollout_demo(
                    scene,
                    seed=seed,
                    backend='pybullet',
                    gui=gui_mode,
                    video_path=video_path,
                    fps=float(args.fps),
                    width=int(args.width),
                    height=int(args.height),
                    render_frame_stride=int(args.render_frame_stride),
                    video_end_hold_seconds=float(args.video_end_hold_seconds),
                    realtime=bool(effective_realtime),
                    gui_hold_seconds=float(effective_hold_seconds),
                    visualize_normal_load=bool(args.visualize_normal_load),
                    feature_overlay=bool(args.feature_overlay),
                    feature_overlay_title="Demonstration feature profile",
                    save_frame_indices=save_frame_indices,
                    save_frame_dir=outdir,
                    save_frame_prefix=f's4_demo_{int(demo_idx):02d}',
                    execution_normal_load_noise_std=float(args.execution_normal_load_noise_std),
                    execution_normal_load_noise_smooth=float(args.execution_normal_load_noise_smooth),
                    execution_normal_load_noise_seed=args.execution_normal_load_noise_seed if args.execution_normal_load_noise_seed is not None else seed,
                )
            else:
                latent = env.rollout_demo(scene, seed=seed, backend='analytic')
                planned_normal_load = np.asarray(latent.get('normal_force_trace', latent.get('normal_load_trace', [])), dtype=float)
                executed_normal_load, load_noise = env.apply_execution_normal_load_noise(
                    planned_normal_load,
                    noise_std=float(args.execution_normal_load_noise_std),
                    noise_smooth=float(args.execution_normal_load_noise_smooth),
                    seed=args.execution_normal_load_noise_seed if args.execution_normal_load_noise_seed is not None else seed,
                )
                playback = session.play(
                    np.asarray(latent['planned_trajectory'], dtype=float),
                    normal_load_trace=executed_normal_load,
                    true_cutpoints=latent.get('true_cutpoints'),
                    visualize_normal_load=bool(args.visualize_normal_load),
                    feature_overlay=bool(args.feature_overlay),
                    feature_overlay_title="Demonstration feature profile",
                    save_frame_indices=save_frame_indices,
                    save_frame_dir=outdir,
                    save_frame_prefix=f's4_demo_{int(demo_idx):02d}',
                    gui=gui_mode,
                    video_path=video_path,
                    fps=float(args.fps),
                    width=int(args.width),
                    height=int(args.height),
                    render_frame_stride=int(args.render_frame_stride),
                    video_end_hold_seconds=float(args.video_end_hold_seconds),
                    realtime=bool(effective_realtime),
                    gui_hold_seconds=float(effective_hold_seconds),
                )
                latent.update(playback)
                latent['planned_trajectory'] = np.asarray(playback['reference_trajectory'], dtype=float)
                latent['normal_force_trace'] = executed_normal_load
                latent['normal_load_trace'] = executed_normal_load
                latent['planned_normal_force_trace'] = planned_normal_load
                latent['planned_normal_load_trace'] = planned_normal_load
                latent['execution_normal_force_noise'] = load_noise
                latent['execution_normal_load_noise'] = load_noise
                latent['true_labels'] = latent.get('true_labels')
                latent['true_cutpoints'] = latent.get('true_cutpoints')
            obs = env.compute_observation(latent, scene)
            planned = np.asarray(latent.get('planned_trajectory', latent.get('reference_trajectory')), dtype=float)
            executed = np.asarray(obs['trajectory'], dtype=float)
            cutpoints = np.asarray(obs['true_cutpoints'], dtype=int)
            feature_path = None
            if bool(args.plot_features):
                feature_path = outdir / f'demo_{demo_idx:02d}_features.png'
                _plot_features(
                    env,
                    planned,
                    executed,
                    cutpoints,
                    feature_path,
                    planned_normal_load=latent.get('planned_normal_force_trace', latent.get('planned_normal_load_trace', latent.get('normal_force_trace', latent.get('normal_load_trace')))),
                    executed_normal_load=latent.get('normal_force_trace', latent.get('normal_load_trace')),
                )
            npz_path = outdir / f'demo_{demo_idx:02d}_rollout.npz'
            np.savez_compressed(
                npz_path,
                planned_trajectory=planned,
                executed_trajectory=executed,
                cutpoints=cutpoints,
                features=np.asarray(obs['features'], dtype=float),
                normal_force_trace=np.asarray(latent.get('normal_force_trace', latent.get('normal_load_trace', [])), dtype=float),
                planned_normal_force_trace=np.asarray(latent.get('planned_normal_force_trace', latent.get('planned_normal_load_trace', [])), dtype=float),
                execution_normal_force_noise=np.asarray(latent.get('execution_normal_force_noise', latent.get('execution_normal_load_noise', [])), dtype=float),
                normal_load_trace=np.asarray(latent.get('normal_load_trace', latent.get('normal_force_trace', [])), dtype=float),
                planned_normal_load_trace=np.asarray(latent.get('planned_normal_load_trace', latent.get('planned_normal_force_trace', [])), dtype=float),
                execution_normal_load_noise=np.asarray(latent.get('execution_normal_load_noise', latent.get('execution_normal_force_noise', [])), dtype=float),
                joint_positions=np.asarray(obs.get('joint_positions', []), dtype=float),
                joint_position_commands=np.asarray(obs.get('joint_position_commands', []), dtype=float),
                ik_position_error_world=np.asarray(obs.get('ik_position_error_world', []), dtype=float),
            )
            ik_err = np.asarray(obs.get('ik_position_error_world', []), dtype=float)
            violation_stats = constraint_violation_stats(
                features_list=[np.asarray(obs['features'], dtype=float)],
                cutpoints_list=[cutpoints],
                feature_schema=env.get_feature_schema(),
                constraint_specs=env.get_constraint_specs(),
                true_constraints=env.get_true_constraints(),
                equality_tolerance=1e-3,
            )
            violation_features.append(np.asarray(obs['features'], dtype=float))
            violation_cutpoints.append(cutpoints)
            if combine_video and video_path is not None:
                segment_paths.append(Path(video_path))
            summary = {
                'demo_index': int(demo_idx),
                'seed': int(seed),
                'points': int(len(executed)),
                'video': None if combine_video or video_path is None else str(video_path),
                'saved_frames': list(latent.get('saved_frames', [])),
                'features': None if feature_path is None else str(feature_path),
                'rollout': str(npz_path),
                'gui': int(args.gui),
                'fps': float(args.fps),
                'camera_target': None if camera_target is None else [float(v) for v in camera_target],
                'render_frame_stride': int(args.render_frame_stride),
                'video_end_hold_seconds': float(args.video_end_hold_seconds),
                'renderer_reused': bool(session is not None),
                'visualize_normal_force': bool(args.visualize_normal_load),
                'visualize_normal_load': bool(args.visualize_normal_load),
                'feature_overlay': bool(args.feature_overlay),
                'execution_normal_force_noise_std': float(args.execution_normal_load_noise_std),
                'execution_normal_force_noise_smooth': float(args.execution_normal_load_noise_smooth),
                'execution_normal_force_noise_seed': None if args.execution_normal_load_noise_seed is None else int(args.execution_normal_load_noise_seed),
                'execution_normal_load_noise_std': float(args.execution_normal_load_noise_std),
                'execution_normal_load_noise_smooth': float(args.execution_normal_load_noise_smooth),
                'execution_normal_load_noise_seed': None if args.execution_normal_load_noise_seed is None else int(args.execution_normal_load_noise_seed),
                'ik_position_error_mean': None if ik_err.size == 0 else float(np.mean(ik_err)),
                'ik_position_error_max': None if ik_err.size == 0 else float(np.max(ik_err)),
                'constraint_violation': violation_stats,
            }
            summaries.append(summary)
            print(f"[demo {demo_idx:02d}] points={len(executed)} video={video_path} features={feature_path}")
    finally:
        if session is not None:
            session.close()
    combined_video_path = None
    if combine_video:
        combined_video_path = concat_mp4_files(segment_paths, combined_video_target)
        try:
            shutil.rmtree(segment_dir)
        except OSError:
            pass
        print(f"[saved] {combined_video_path}")
    elif combined_video_target is not None and combined_video_target.exists():
        combined_video_path = combined_video_target
    summary_path = outdir / 's4_realistic_demonstration_render_summary.json'
    aggregate_violation = constraint_violation_stats(
        features_list=violation_features,
        cutpoints_list=violation_cutpoints,
        feature_schema=env.get_feature_schema(),
        constraint_specs=env.get_constraint_specs(),
        true_constraints=env.get_true_constraints(),
        equality_tolerance=1e-3,
    )
    aggregate = {
        'task': 's4_realistic_demonstration_render',
        'gui': int(args.gui),
        'combined_video': None if combined_video_path is None else str(Path(combined_video_path).resolve()),
        'combined_video_segments': [] if combine_video else [str(Path(path).resolve()) for path in segment_paths],
        'constraint_violation': aggregate_violation,
        'demos': summaries,
    }
    summary_path.write_text(
        json.dumps(aggregate, indent=2) + '\n',
        encoding='utf-8',
    )
    print(f'[saved] {summary_path}')
    print_render_violation_rates(aggregate)


if __name__ == '__main__':
    main()
