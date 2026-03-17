from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from .ccp_4panel import PAPER_FIGSIZE, PAPER_LABEL_SIZE, PAPER_LEGEND_SIZE, PAPER_TICK_SIZE, PAPER_TITLE_SIZE
from .io import learner_plot_dir, save_figure


def _stage_step_cost_series(learner, demo_idx, stage_idx):
    X = learner.demos[demo_idx]
    T = len(X)
    if T <= 1:
        return np.zeros(0, dtype=float)
    return np.asarray(
        [learner._step_progress_cost(stage_idx, X[t], X[t + 1]) for t in range(T - 1)],
        dtype=float,
    )


def plot_ccp_progress_heatmaps(learner, it, demo_idx=0):
    if plt is None:
        return
    X = learner.demos[demo_idx]
    T = len(X)
    if T <= 1:
        return

    n_panels = min(int(learner.num_states), 2)
    fig, axes = plt.subplots(1, n_panels, figsize=PAPER_FIGSIZE)
    if n_panels == 1:
        axes = [axes]

    stage_ends = learner.stage_ends_[demo_idx]
    prev_end = -1
    for stage_idx in range(n_panels):
        ax = axes[stage_idx]
        costs = _stage_step_cost_series(learner, demo_idx, stage_idx)
        time_axis = np.arange(len(costs))
        ax.plot(time_axis, costs, color="tab:orange", lw=1.2, label="step progress cost")

        stage_end = int(stage_ends[stage_idx])
        stage_start = prev_end + 1
        assigned_end = max(stage_start, stage_end - 1)
        if assigned_end >= stage_start:
            ax.axvspan(stage_start, assigned_end, color="tab:blue", alpha=0.10, label="assigned stage span")
        ax.axvline(stage_end, color="black", linestyle="--", lw=1.0, label="stage boundary")

        center = np.asarray(learner.end_mu[stage_idx], dtype=float).reshape(-1)
        center_text = np.array2string(center[: min(2, len(center))], precision=2, suppress_small=True)
        ax.set_title(
            f"Stage {stage_idx + 1} timestep progress cost\ncenter={center_text}",
            fontsize=PAPER_TITLE_SIZE,
            pad=4,
        )
        ax.set_xlabel("timestep", fontsize=PAPER_LABEL_SIZE)
        ax.set_ylabel("cost", fontsize=PAPER_LABEL_SIZE)
        ax.tick_params(labelsize=PAPER_TICK_SIZE)
        prev_end = stage_end

        handles, labels = ax.get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels):
            if l is None:
                continue
            text = str(l).strip()
            if text and not text.startswith("_") and text not in by_label:
                by_label[text] = h
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=PAPER_LEGEND_SIZE, frameon=False, loc="best")

    fig.suptitle(f"CCP timestep progress costs at iter {int(it)} (demo {int(demo_idx)})", fontsize=PAPER_TITLE_SIZE)
    save_figure(fig, learner_plot_dir(learner) / f"progress_timestep_iter_{int(it):04d}.png", dpi=220)
