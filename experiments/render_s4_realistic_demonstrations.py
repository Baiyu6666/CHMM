from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.S4SlideInsertRealistic import S4SlideInsertRealisticEnv
from envs.s4_pybullet_backend import S4PyBulletPlaybackSession

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


def main():
    parser = argparse.ArgumentParser(description='Render S4SlideInsertRealistic PyBullet demonstrations.')
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
    parser.add_argument('--camera-yaw', type=float, default=128.0)
    parser.add_argument('--camera-pitch', type=float, default=-29.0)
    parser.add_argument('--camera-distance', type=float, default=0.84)
    parser.add_argument('--camera-fov', type=float, default=42.0)
    parser.add_argument('--plot-features', type=int, default=1)
    parser.add_argument('--visualize-normal-load', type=int, default=0)
    parser.add_argument('--feature-overlay', type=int, default=1)
    parser.add_argument('--execution-normal-load-noise-std', type=float, default=0.025)
    parser.add_argument('--execution-normal-load-noise-smooth', type=float, default=0.85)
    parser.add_argument('--execution-normal-load-noise-seed', type=int, default=None)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    env = S4SlideInsertRealisticEnv(
        rollout_backend='analytic',
        observation_backend='analytic',
        pybullet_camera_yaw=float(args.camera_yaw),
        pybullet_camera_pitch=float(args.camera_pitch),
        pybullet_camera_distance=float(args.camera_distance),
        pybullet_camera_fov=float(args.camera_fov),
        pybullet_render_width=int(args.width),
        pybullet_render_height=int(args.height),
    )
    indices = _parse_indices(args.demo_indices, int(args.n_demos))
    summaries = []
    gui_mode = int(args.gui)
    session = S4PyBulletPlaybackSession(env, force_gui=(gui_mode == 2)) if gui_mode in (1, 2) else None
    try:
        for demo_idx in indices:
            scene = env.sample_scene()
            scene['demo_index'] = int(demo_idx)
            seed = int(args.seed) + int(demo_idx)
            video_path = outdir / f'demo_{demo_idx:02d}.mp4' if gui_mode == 1 else None
            effective_realtime = bool(args.realtime) or gui_mode == 2
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
                    feature_overlay_title="Executed trajectory feature profile (demonstration)",
                    execution_normal_load_noise_std=float(args.execution_normal_load_noise_std),
                    execution_normal_load_noise_smooth=float(args.execution_normal_load_noise_smooth),
                    execution_normal_load_noise_seed=args.execution_normal_load_noise_seed if args.execution_normal_load_noise_seed is not None else seed,
                )
            else:
                latent = env.rollout_demo(scene, seed=seed, backend='analytic')
                planned_normal_load = np.asarray(latent.get('normal_load_trace', []), dtype=float)
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
                    feature_overlay_title="Executed trajectory feature profile (demonstration)",
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
                latent['normal_load_trace'] = executed_normal_load
                latent['planned_normal_load_trace'] = planned_normal_load
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
                    planned_normal_load=latent.get('planned_normal_load_trace', latent.get('normal_load_trace')),
                    executed_normal_load=latent.get('normal_load_trace'),
                )
            npz_path = outdir / f'demo_{demo_idx:02d}_rollout.npz'
            np.savez_compressed(
                npz_path,
                planned_trajectory=planned,
                executed_trajectory=executed,
                cutpoints=cutpoints,
                features=np.asarray(obs['features'], dtype=float),
                normal_load_trace=np.asarray(latent.get('normal_load_trace', []), dtype=float),
                planned_normal_load_trace=np.asarray(latent.get('planned_normal_load_trace', []), dtype=float),
                execution_normal_load_noise=np.asarray(latent.get('execution_normal_load_noise', []), dtype=float),
                joint_positions=np.asarray(obs.get('joint_positions', []), dtype=float),
                joint_position_commands=np.asarray(obs.get('joint_position_commands', []), dtype=float),
                ik_position_error_world=np.asarray(obs.get('ik_position_error_world', []), dtype=float),
            )
            ik_err = np.asarray(obs.get('ik_position_error_world', []), dtype=float)
            summary = {
                'demo_index': int(demo_idx),
                'seed': int(seed),
                'points': int(len(executed)),
                'video': None if video_path is None else str(video_path),
                'features': None if feature_path is None else str(feature_path),
                'rollout': str(npz_path),
                'gui': int(args.gui),
                'fps': float(args.fps),
                'render_frame_stride': int(args.render_frame_stride),
                'video_end_hold_seconds': float(args.video_end_hold_seconds),
                'renderer_reused': bool(session is not None),
                'visualize_normal_load': bool(args.visualize_normal_load),
                'feature_overlay': bool(args.feature_overlay),
                'execution_normal_load_noise_std': float(args.execution_normal_load_noise_std),
                'execution_normal_load_noise_smooth': float(args.execution_normal_load_noise_smooth),
                'execution_normal_load_noise_seed': None if args.execution_normal_load_noise_seed is None else int(args.execution_normal_load_noise_seed),
                'ik_position_error_mean': None if ik_err.size == 0 else float(np.mean(ik_err)),
                'ik_position_error_max': None if ik_err.size == 0 else float(np.max(ik_err)),
            }
            summaries.append(summary)
            print(f"[demo {demo_idx:02d}] points={len(executed)} video={video_path} features={feature_path}")
    finally:
        if session is not None:
            session.close()
    summary_path = outdir / 's4_realistic_demonstration_render_summary.json'
    summary_path.write_text(json.dumps({'task': 's4_realistic_demonstration_render', 'gui': int(args.gui), 'demos': summaries}, indent=2) + '\n', encoding='utf-8')
    print(f'[saved] {summary_path}')


if __name__ == '__main__':
    main()
