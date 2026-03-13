# main_2d.py
# ------------------------------------------------------------
# Run on 2D environment:
#   - GoalHMM3D (actually works for 2D as well)
#   - (optionally) HMM-HMM baseline
#   - Visualize demos and learned segmentation in 2D
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from env.env2d import ObsAvoidEnv
from learner.goal_hmm import GoalHMM3D
from learner.gmm_hmm import GMMHMM


# ------------------------------------------------------------
# Simple 2D debug plot
# ------------------------------------------------------------
def debug_plot_2d_demos(env, demos, taus, title="2D demos with cutpoints"):
    """
    在 XY 平面画出:
      - 障碍物圆
      - 所有 demo 轨迹 (stage1 蓝, stage2 红)
      - cutpoint 黄色点
      - subgoal / goal
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # --- obstacle circle ---
    cx, cy = env.obs_center
    r = env.obs_radius
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(
        cx + r * np.cos(theta),
        cy + r * np.sin(theta),
        "k--",
        alpha=0.6,
        label="obstacle"
    )

    # --- demos ---
    for X, tau in zip(demos, taus):
        X = np.asarray(X)
        tau = int(tau)

        # stage1
        ax.plot(
            X[: tau + 1, 0],
            X[: tau + 1, 1],
            "b-",
            lw=1.5,
            alpha=0.8
        )
        # stage2
        ax.plot(
            X[tau:, 0],
            X[tau:, 1],
            "r-",
            lw=1.5,
            alpha=0.8
        )

        # cutpoint
        ax.scatter(
            X[tau, 0],
            X[tau, 1],
            c="yellow",
            edgecolors="k",
            s=30,
            zorder=5,
        )

    # --- start/subgoal/goal (如果 env 有的话) ---
    if hasattr(env, "start"):
        ax.scatter(env.start[0], env.start[1],
                   c="gray", marker="s", s=60, label="start")
    if hasattr(env, "subgoal"):
        ax.scatter(env.subgoal[0], env.subgoal[1],
                   c="orange", marker="D", s=60, label="subgoal")
    if hasattr(env, "goal"):
        ax.scatter(env.goal[0], env.goal[1],
                   c="green", marker="P", s=70, label="goal")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Main experiment on 2D env
# ------------------------------------------------------------
def run_experiment_2d(
    n_demos=10,
    seed=42,
    max_iter=40,
    run_baseline=True,
    init_mode='heuristic',
):
    np.random.seed(seed)

    # -------- Environment (2D) --------
    env = ObsAvoidEnv(
        start=(-1.5, 0.0),
        subgoal=(0.5, 0.0),
        goal=(0.0, 0.3),
        obs_center=(-0.5, 0.0),
        obs_radius=0.3,
        dt=1.0,
        noise_std=0.01,
    )

    # -------- Generate demos --------
    demos, true_taus = [], []
    print(f"[2D] Generating {n_demos} demos…")

    for _ in range(n_demos):
        X, tau = env.generate_demo(n1=20, direction=None)
        demos.append(X)
        true_taus.append(tau)
    env.estimate_oracle_constraints(demos, true_taus)
    print("[2D] Generated demos. Example shape:", demos[0].shape)
    print("[2D] True taus:", true_taus)

    # -------- Train GoalHMM3D (works for 2D as well) --------
    print("\n[2D] Training GoalHMM3D…")

    learner = GoalHMM3D(
        demos=demos,
        env=env,
        true_taus=true_taus,
        g1_init=init_mode,
        g2_init=None,         # 默认用终点均值
        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=1.0,
        g_steps=5,
        g_lr=5e-4,
        plot_every=max_iter,  # 只在最后一轮画 4-panel（如果你在 plot 里有调用）
        feature_ids=[0, 1],
        auto_feature_select=True,
        r_sparse_lambda=0.3,
        feature_types=["margin_exp_lower",  # 0: 距离 -> 下界不等式
                       "gauss",  # 1: 速度 -> 等式/窄带
                       "gauss",  # 2: 假障碍
                       "gauss"],
    )

    posts = learner.fit(max_iter=max_iter)

    # posterior taus based on learner
    taus_est = []
    for gamma in posts:
        idx = np.where(gamma[:, 1] > 0.5)[0]
        tau_hat = int(idx[0]) if len(idx) > 0 else int(np.argmax(gamma[:, 1]))
        taus_est.append(tau_hat)

    print("[2D] Estimated taus:", taus_est)
    print("[2D] Learned g1:", np.round(learner.g1, 3))
    print("[2D] Learned g2:", np.round(learner.g2, 3))

    # -------- Baseline (可选) --------
    if run_baseline:
        print("\n[2D] Training HMM-HMM baseline…")
        base = GMMHMM(
            demos=demos,
            env=env,
            true_taus=true_taus,
            plot_every=max_iter,
        )
        base.fit(max_iter=max_iter)

    # -------- Simple 2D visualization --------
    debug_plot_2d_demos(
        env,
        demos,
        taus_est,
        title="2D demos with GoalHMM3D segmentation",
    )


if __name__ == "__main__":
    run_experiment_2d(
        n_demos=10,
        seed=2224519,
        max_iter=45,
        run_baseline=True,
    )
