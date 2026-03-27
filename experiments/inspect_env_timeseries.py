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

from envs import load_env
from experiments.config_loader import load_json
from visualization.io import save_figure


def _load_dataset_from_env_config(env_config_path: str | Path):
    cfg = dict(load_json(env_config_path))
    dataset_name = str(cfg.pop("name"))
    cfg.pop("method_overrides", None)
    return dataset_name, load_env(dataset_name, **cfg)


def _true_cutpoints(bundle, demo_idx: int):
    true_cutpoints = getattr(bundle, "true_cutpoints", None)
    if true_cutpoints is None or demo_idx >= len(true_cutpoints):
        return []
    cuts = true_cutpoints[demo_idx]
    if cuts is None:
        return []
    return [int(x) for x in np.asarray(cuts, dtype=int).reshape(-1).tolist()]


def _plot_timeseries(ax, series, name: str, cutpoints, color=None):
    t = np.arange(len(series))
    ax.plot(t, np.asarray(series, dtype=float), "-", lw=1.5, color=color)
    for j, cp in enumerate(cutpoints):
        ax.axvline(cp, color="black", linestyle="--", linewidth=0.9, label="true cutpoint" if j == 0 else "")
    ax.set_title(name, fontsize=9)
    ax.grid(alpha=0.25)


def main():
    parser = argparse.ArgumentParser(description="Inspect raw states and features of one demo without training.")
    parser.add_argument("--env-config", default="2DPressSlideInsert.json", type=str)
    parser.add_argument("--demo-idx", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if plt is None:
        raise RuntimeError("matplotlib is required for inspect_env_timeseries.py")

    dataset_name, bundle = _load_dataset_from_env_config(args.env_config)
    demo_idx = int(args.demo_idx)
    if demo_idx < 0 or demo_idx >= len(bundle.demos):
        raise IndexError(f"demo_idx={demo_idx} is out of range for dataset with {len(bundle.demos)} demos")

    X = np.asarray(bundle.demos[demo_idx], dtype=float)
    F = np.asarray(bundle.env.compute_all_features_matrix(X), dtype=float)
    cutpoints = _true_cutpoints(bundle, demo_idx)

    state_dim_to_plot = min(2, X.shape[1])
    state_names = []
    if state_dim_to_plot >= 1:
        state_names.append("x")
    if state_dim_to_plot >= 2:
        state_names.append("z" if dataset_name == "PickPlace" else "y")

    feature_schema = bundle.feature_schema or []
    feature_names = [str(spec.get("name", f"f{i}")) for i, spec in enumerate(feature_schema)]
    if len(feature_names) != F.shape[1]:
        feature_names = [f"f{i}" for i in range(F.shape[1])]

    total_rows = state_dim_to_plot + F.shape[1]
    fig, axes = plt.subplots(total_rows, 1, figsize=(11, max(7, 1.75 * total_rows)), sharex=True, squeeze=False)
    axes = axes[:, 0]

    fig.suptitle(f"{dataset_name} demo {demo_idx}: states and features", fontsize=11)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for i in range(state_dim_to_plot):
        color = color_cycle[i % len(color_cycle)] if color_cycle else None
        _plot_timeseries(axes[i], X[:, i], f"state: {state_names[i]}", cutpoints, color=color)
        axes[i].set_ylabel(state_names[i], fontsize=8)

    offset = state_dim_to_plot
    for i in range(F.shape[1]):
        color = color_cycle[(offset + i) % len(color_cycle)] if color_cycle else None
        _plot_timeseries(axes[offset + i], F[:, i], f"feature: {feature_names[i]}", cutpoints, color=color)
        axes[offset + i].set_ylabel(feature_names[i], fontsize=8)

    axes[-1].set_xlabel("t", fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, loc="upper right", fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    if args.output is None:
        output_path = PROJECT_ROOT / "outputs" / "analysis" / "env_timeseries" / dataset_name / f"demo_{demo_idx:02d}.png"
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
    save_figure(fig, output_path, dpi=180)
    print(output_path)


if __name__ == "__main__":
    main()
