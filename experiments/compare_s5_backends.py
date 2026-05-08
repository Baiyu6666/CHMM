from __future__ import annotations

import argparse
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
from visualization.io import save_figure


DEFAULT_FEATURE_NAMES = ("surf_dist", "normal_err", "speed", "ang_speed")


def _parse_csv_ints(text: str | None) -> list[int] | None:
    if text is None or not str(text).strip():
        return None
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _parse_csv_strings(text: str) -> list[str]:
    items = [item.strip() for item in str(text).split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected a comma-separated feature list.")
    return items


def _resolve_feature_indices(schema: list[dict], names: list[str]) -> tuple[list[int], list[str]]:
    name_to_idx = {str(spec.get("name", f"f{i}")): i for i, spec in enumerate(schema)}
    indices, resolved_names = [], []
    for name in names:
        if name not in name_to_idx:
            raise KeyError(f"Feature '{name}' not found in S5 feature schema: {sorted(name_to_idx)}")
        indices.append(int(name_to_idx[name]))
        resolved_names.append(str(name))
    return indices, resolved_names


def _build_env(*, rollout_backend: str, observation_backend: str, eval_tag: str) -> S5SphereInspectEnv:
    env_cfg = _apply_default_s5_loader_config({})
    env_cfg["rollout_backend"] = str(rollout_backend)
    env_cfg["observation_backend"] = str(observation_backend)
    env_cfg["eval_tag"] = str(eval_tag)
    return S5SphereInspectEnv(**env_cfg)


def _collect_observations(env: S5SphereInspectEnv, *, n_demos: int, seed: int) -> list[dict]:
    observations: list[dict] = []
    for demo_idx in range(int(n_demos)):
        scene = env.sample_scene()
        scene["demo_index"] = int(demo_idx)
        latent = env.rollout_demo(scene, seed=env.demo_seed_for_index(int(seed), int(demo_idx)))
        observation = env.compute_observation(latent, scene)
        observations.append(observation)
    return observations


def compare_s5_backends(
    *,
    n_demos: int,
    seed: int,
    demo_indices: list[int] | None,
    output_dir: str | Path | None,
    feature_names: list[str],
) -> list[Path]:
    if plt is None:
        raise RuntimeError("matplotlib is required for compare_s5_backends.py")

    analytic_raw_env = _build_env(
        rollout_backend="analytic",
        observation_backend="analytic_raw",
        eval_tag="S5SphereInspect",
    )
    pybullet_env = _build_env(
        rollout_backend="pybullet",
        observation_backend="pybullet",
        eval_tag="S5SphereInspectPyBullet",
    )

    raw_obs = _collect_observations(analytic_raw_env, n_demos=n_demos, seed=seed)
    pybullet_obs = _collect_observations(pybullet_env, n_demos=n_demos, seed=seed)

    schema = analytic_raw_env.get_feature_schema()
    feature_indices, feature_names = _resolve_feature_indices(schema, feature_names)

    if demo_indices is None:
        demo_indices = list(range(int(n_demos)))

    if output_dir is None:
        out_dir = PROJECT_ROOT / "outputs" / "analysis" / "s5_backend_compare"
    else:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for demo_idx in demo_indices:
        if demo_idx < 0 or demo_idx >= int(n_demos):
            raise IndexError(f"demo_idx={demo_idx} is out of range for n_demos={n_demos}")

        raw = raw_obs[int(demo_idx)]
        pyb = pybullet_obs[int(demo_idx)]
        cutpoints = [int(x) for x in np.asarray(raw["true_cutpoints"], dtype=int).reshape(-1).tolist()]

        fig, axes = plt.subplots(len(feature_indices), 1, figsize=(9.4, 2.15 * len(feature_indices) + 0.8), sharex=True)
        if len(feature_indices) == 1:
            axes = [axes]

        for ax, feat_idx, feat_name in zip(axes, feature_indices, feature_names):
            ax.plot(
                np.asarray(raw["features"], dtype=float)[:, feat_idx],
                label="analytic raw",
                color="#1D4ED8",
                linewidth=1.7,
            )
            ax.plot(
                np.asarray(pyb["features"], dtype=float)[:, feat_idx],
                label="pybullet",
                color="#047857",
                linewidth=1.5,
                alpha=0.92,
            )
            for cp in cutpoints:
                ax.axvline(int(cp), color="#111827", linestyle="--", linewidth=0.8, alpha=0.40)
            ax.set_ylabel(feat_name)
            ax.grid(alpha=0.2)

        axes[0].legend(loc="upper right", frameon=False, ncol=2)
        axes[-1].set_xlabel("t")
        fig.suptitle(f"S5 demo {int(demo_idx)}: analytic raw vs pybullet", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        out_path = out_dir / f"demo_{int(demo_idx):02d}.png"
        save_figure(fig, out_path, dpi=220)
        saved_paths.append(out_path)

    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare S5 analytic raw and pybullet feature traces.")
    parser.add_argument("--n-demos", type=int, default=5, help="How many demos to generate for comparison.")
    parser.add_argument("--seed", type=int, default=7, help="Dataset seed.")
    parser.add_argument("--demo-indices", type=str, default=None, help="Comma-separated demo indices. Defaults to all loaded demos.")
    parser.add_argument(
        "--features",
        type=_parse_csv_strings,
        default=list(DEFAULT_FEATURE_NAMES),
        help="Comma-separated feature names to compare. Defaults to surf_dist,normal_err,speed,ang_speed.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory.")
    args = parser.parse_args()

    saved = compare_s5_backends(
        n_demos=int(args.n_demos),
        seed=int(args.seed),
        demo_indices=_parse_csv_ints(args.demo_indices),
        output_dir=args.output_dir,
        feature_names=list(args.features),
    )
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
