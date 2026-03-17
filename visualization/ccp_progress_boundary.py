from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

from .io import learner_plot_dir, save_figure

PAPER_FIGSIZE = (6.8, 4.2)
PAPER_TITLE_SIZE = 9
PAPER_LABEL_SIZE = 8
PAPER_TICK_SIZE = 7
PAPER_LEGEND_SIZE = 6.5


def plot_ccp_progress_boundary_profile(learner, it, demo_idx=0):
    if plt is None:
        return
    if learner.num_states != 2:
        return

    X = learner.demos[demo_idx]
    T = len(X)
    tau_min = max(1, int(learner.duration_min[0]) - 1)
    tau_max = min(T - 2, T - int(learner.duration_min[1]))
    if tau_max < tau_min:
        return

    taus = np.arange(tau_min, tau_max + 1)
    stage1_scores = np.array([learner._progress_score(demo_idx, 0, 0, int(tau)) for tau in taus], dtype=float)
    stage2_scores = np.array([learner._progress_score(demo_idx, 1, int(tau) + 1, T - 1) for tau in taus], dtype=float)
    total_scores = stage1_scores + stage2_scores

    fig, ax = plt.subplots(1, 1, figsize=PAPER_FIGSIZE)
    ax.plot(taus, stage1_scores, color="tab:blue", lw=1.2, label="stage1 score")
    ax.plot(taus, stage2_scores, color="tab:orange", lw=1.2, label="stage2 score")
    ax.plot(taus, total_scores, color="black", lw=1.4, label="sum score")

    learned_tau = int(learner.stage_ends_[demo_idx][0])
    ax.axvline(learned_tau, color="black", linestyle="--", lw=1.0, label="learned tau")
    if learner.true_taus[demo_idx] is not None:
        ax.axvline(int(learner.true_taus[demo_idx]), color="green", linestyle=":", lw=1.0, label="true tau")

    ax.set_title(f"Demo{int(demo_idx)} progress scores vs boundary at iter {int(it)}", fontsize=PAPER_TITLE_SIZE, pad=4)
    ax.set_xlabel("boundary tau", fontsize=PAPER_LABEL_SIZE)
    ax.set_ylabel("progress score", fontsize=PAPER_LABEL_SIZE)
    ax.tick_params(labelsize=PAPER_TICK_SIZE)

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

    save_figure(fig, learner_plot_dir(learner) / f"progress_boundary_iter_{int(it):04d}.png", dpi=220)
