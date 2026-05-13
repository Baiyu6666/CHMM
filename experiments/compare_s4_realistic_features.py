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

from envs.S4SlideInsert import S4SlideInsertEnv
from envs.S4SlideInsertRealistic import S4SlideInsertRealisticEnv


COMMON_FEATURES = [
    ("surf_dist", 0, "surf_dist", 0, "metric"),
    ("force", 1, "normal_load", 5, "none"),
    ("orient_err", 2, "orient_err", 2, "none"),
    ("speed", 3, "speed", 3, "metric"),
    ("noise", 4, "noise", 6, "none"),
    ("start_dist", 5, "start_dist", 7, "metric"),
    ("insertion_err", 6, "insertion_err", 8, "metric"),
]

REALISTIC_EXTRA_FEATURES = [
    ("centerline_dist", 1),
    ("angular_speed", 4),
]

OLD_S4_TO_REALISTIC_UNIT = 0.16


def _parse_indices(text: str | None, n_demos: int) -> list[int]:
    if text is None or not str(text).strip():
        return list(range(int(n_demos)))
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def _cutpoints_from_labels(labels: np.ndarray) -> np.ndarray:
    return np.where(np.diff(np.asarray(labels, dtype=int)) != 0)[0].astype(int)


def _old_s4_observation(env: S4SlideInsertEnv, seed: int) -> dict:
    pos, theta, labels, force, speed = env.generate_demo(seed=int(seed))
    traj = np.c_[np.asarray(pos, dtype=float), np.asarray(theta, dtype=float), np.asarray(force, dtype=float)]
    env.register_force_trace(traj, force)
    env.register_speed_trace(traj, speed)
    return {
        "trajectory": traj,
        "features": env.compute_all_features_matrix(traj),
        "true_labels": np.asarray(labels, dtype=int),
        "true_cutpoints": _cutpoints_from_labels(labels),
    }


def _realistic_observation(env: S4SlideInsertRealisticEnv, seed: int, *, backend: str) -> dict:
    scene = env.sample_scene()
    latent = env.rollout_demo(
        scene,
        seed=int(seed),
        backend=str(backend),
        gui=0,
        visualize_normal_load=False,
        video_end_hold_seconds=0.0,
    )
    return env.compute_observation(latent, scene)


def _scaled_realistic_feature(values: np.ndarray, scale_kind: str, old_to_realistic_unit: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if scale_kind == "metric":
        return arr / max(float(old_to_realistic_unit), 1e-12)
    return arr


def _plot_demo(
    *,
    old_obs: dict,
    realistic_obs: dict,
    realistic_env: S4SlideInsertRealisticEnv,
    demo_idx: int,
    seed: int,
    backend: str,
    out_path: Path,
    include_extra: bool,
) -> dict:
    if plt is None:
        raise RuntimeError("matplotlib is required for compare_s4_realistic_features.py")

    old_features = np.asarray(old_obs["features"], dtype=float)
    real_features = np.asarray(realistic_obs["features"], dtype=float)
    cutpoints = np.asarray(old_obs["true_cutpoints"], dtype=int).reshape(-1)

    rows = len(COMMON_FEATURES) + (len(REALISTIC_EXTRA_FEATURES) if include_extra else 0)
    fig, axes = plt.subplots(rows, 1, figsize=(9.8, 1.85 * rows + 0.8), sharex=True)
    if rows == 1:
        axes = [axes]

    summary: dict[str, dict[str, float]] = {}
    axis_i = 0
    for old_name, old_idx, real_name, real_idx, scale_kind in COMMON_FEATURES:
        ax = axes[axis_i]
        old_y = old_features[:, old_idx]
        real_y = _scaled_realistic_feature(real_features[:, real_idx], scale_kind, OLD_S4_TO_REALISTIC_UNIT)
        corr = float(np.corrcoef(old_y, real_y)[0, 1]) if np.std(old_y) > 1e-12 and np.std(real_y) > 1e-12 else float("nan")
        mae = float(np.mean(np.abs(old_y - real_y)))
        summary[old_name] = {"corr": corr, "mae": mae}

        label = f"realistic {backend}"
        unit_note = "old units" if scale_kind == "metric" else "native"
        ax.plot(old_y, color="#1D4ED8", linewidth=1.7, label="old S4")
        ax.plot(real_y, color="#D97706", linewidth=1.35, label=label)
        ax.set_ylabel(f"{old_name}\n({unit_note})", fontsize=8)
        ax.text(0.995, 0.78, f"corr={corr:.3f}, MAE={mae:.4g}", transform=ax.transAxes, ha="right", fontsize=8)
        for cp in cutpoints:
            ax.axvline(int(cp), color="#111827", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.grid(alpha=0.20)
        axis_i += 1

    if include_extra:
        for name, idx in REALISTIC_EXTRA_FEATURES:
            ax = axes[axis_i]
            ax.plot(real_features[:, idx], color="#047857", linewidth=1.4, label=f"realistic {backend}")
            ax.set_ylabel(name, fontsize=8)
            for cp in cutpoints:
                ax.axvline(int(cp), color="#111827", linestyle="--", linewidth=0.7, alpha=0.35)
            ax.grid(alpha=0.20)
            axis_i += 1

    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    axes[-1].set_xlabel("t")
    fig.suptitle(f"S4 demo {int(demo_idx)} seed {int(seed)}: old S4 vs S4 realistic {backend}", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return summary


def compare_s4_realistic_features(
    *,
    n_demos: int,
    seed: int,
    demo_indices: list[int],
    outdir: str | Path,
    backend: str,
    include_extra: bool,
) -> list[Path]:
    old_env = S4SlideInsertEnv()
    realistic_env = S4SlideInsertRealisticEnv()
    out_dir = Path(outdir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    summaries = {}
    for demo_idx in demo_indices:
        if demo_idx < 0 or demo_idx >= int(n_demos):
            raise IndexError(f"demo index {demo_idx} is out of range for n_demos={n_demos}")
        demo_seed = int(seed) + int(demo_idx)
        old_obs = _old_s4_observation(old_env, demo_seed)
        realistic_obs = _realistic_observation(realistic_env, demo_seed, backend=backend)
        out_path = out_dir / f"demo_{int(demo_idx):02d}_{backend}.png"
        summaries[str(int(demo_idx))] = _plot_demo(
            old_obs=old_obs,
            realistic_obs=realistic_obs,
            realistic_env=realistic_env,
            demo_idx=int(demo_idx),
            seed=demo_seed,
            backend=backend,
            out_path=out_path,
            include_extra=bool(include_extra),
        )
        saved.append(out_path)

    summary_path = out_dir / f"summary_{backend}.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    saved.append(summary_path)
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare old S4 feature profiles against S4 realistic features.")
    parser.add_argument("--n-demos", "--n_demos", dest="n_demos", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--demo-indices", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="outputs/s4_realistic_feature_compare")
    parser.add_argument("--backend", choices=["pybullet", "analytic"], default="pybullet")
    parser.add_argument("--include-extra", type=int, default=1, help="Also plot features that only exist in realistic S4.")
    args = parser.parse_args()

    saved = compare_s4_realistic_features(
        n_demos=int(args.n_demos),
        seed=int(args.seed),
        demo_indices=_parse_indices(args.demo_indices, int(args.n_demos)),
        outdir=args.outdir,
        backend=str(args.backend),
        include_extra=bool(args.include_extra),
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
