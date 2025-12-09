# main.py
# ------------------------------------------------------------
# Run:
#   - GoalHMM3D (soft learner)
#   - (optionally) HMMHMM baseline
#   - Then use optimal_control planner with learned constraints
#     to generate new 2-stage trajectories and visualize them
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from env.env2d import ObsAvoidEnv
from env.env3d import ObsAvoidEnv3D

from learner.goal_hmm import GoalHMM3D
from learner.gmm_hmm import GMMHMM

from render.pybullet_renderer import PyBulletRenderer3D
from planner.optimal_control import (
    plan_two_stage_trajectory,
    plan_oscillate_then_descend,
)


# ------------------------------------------------------------
# Helpers for 3D case
# ------------------------------------------------------------
def sample_random_start_3d(env, xy_jitter=0.5):
    """
    从 env 的 start 分布随机采一个 3D 起点。
    假设 env 有:
      - start_xy
      - start_z_range = (z_min, z_max)
    """
    x0 = np.zeros(3, dtype=float)
    x0[0] = env.start_xy[0] + np.random.uniform(-xy_jitter, xy_jitter)
    x0[1] = env.start_xy[1] + np.random.uniform(-xy_jitter, xy_jitter)
    x0[2] = np.random.uniform(env.start_z_range[0], env.start_z_range[1])
    return x0


def get_feature_constraints_from_learner(learner):
    """
    从 GoalHMM3D 中提取：
      - stage1 distance 的 2σ 下界（raw 距离） -> 用作 d_safe_min
      - stage2 speed 的  2σ 上界（raw 速度） -> 用作 v2_max

    返回:
      d_safe_min_raw, v2_max_raw
    """
    # ---- stage1: distance ----
    mu1 = float(learner.model_stage1.mu)
    sig1 = float(learner.model_stage1.sigma)
    # 这里先用 1σ，你自己可以改回 2σ
    z1_low = mu1 - 1.0 * sig1
    d_safe_min_raw = float(learner._inv1(z1_low))

    # ---- stage2: speed ----
    mu2 = float(learner.model_stage2.mu)
    sig2 = float(learner.model_stage2.sigma)
    z2_high = mu2 + 1.0 * sig2
    v2_max_raw = float(learner._inv2(z2_high))

    return d_safe_min_raw, v2_max_raw


def debug_plot_plans_3d(
    env,
    trajs,
    taus,
    title="Planned 2-stage trajectories",
    dt=0.02,
    v2_max=None,
):
    """
    用 matplotlib 画出 planner 得到的 3D 轨迹，
    前半段(阶段1)蓝色，后半段(阶段2)红色。
    另外再开一个图画速度随时间变化，并画出速度约束的最大值 v2_max。
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # ======================
    # Figure 1: 3D 轨迹
    # ======================
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # ---- 画 obstacle 的投影圈，方便检查约束 ----
    cx, cy = env.obs_center_xy
    r = env.obs_radius
    theta = np.linspace(0, 2 * np.pi, 100)

    # 在一个固定高度画圆圈（大致在 subgoal 高度）
    z_ring = env.subgoal_z if hasattr(env, "subgoal_z") else 0.3
    ax.plot(
        cx + r * np.cos(theta),
        cy + r * np.sin(theta),
        z_ring * np.ones_like(theta),
        "gray",
        alpha=0.4,
    )

    # ---- 画所有 planned 轨迹 ----
    for X, tau in zip(trajs, taus):
        X = np.asarray(X)
        tau = int(tau)

        ax.plot(
            X[: tau + 1, 0],
            X[: tau + 1, 1],
            X[: tau + 1, 2],
            "b-",
            lw=2,
            alpha=0.9,
        )
        ax.plot(
            X[tau:, 0],
            X[tau:, 1],
            X[tau:, 2],
            "r-",
            lw=2,
            alpha=0.9,
        )

    # ---- subgoal / goal 标记 ----
    sg = np.array([env.subgoal_xy[0], env.subgoal_xy[1], env.subgoal_z])
    gg = np.array([env.goal_xy[0], env.goal_xy[1], env.goal_z])
    ax.scatter(
        sg[0], sg[1], sg[2],
        c="orange", marker="D", s=80, label="subgoal"
    )
    ax.scatter(
        gg[0], gg[1], gg[2],
        c="green", marker="P", s=80, label="goal"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.view_init(elev=35, azim=-60)
    plt.tight_layout()

    # ======================
    # Figure 2: 速度随时间
    # ======================
    fig2, ax2 = plt.subplots(figsize=(7, 4))

    for idx, (X, tau) in enumerate(zip(trajs, taus)):
        X = np.asarray(X)
        v = np.diff(X, axis=0) / dt  # (T-1, 3)
        speed = np.linalg.norm(v, axis=1)  # (T-1,)
        t = np.arange(len(speed)) * dt

        # 多条轨迹叠加，用透明一点的线
        ax2.plot(t, speed, alpha=0.5, label=f"traj {idx}")

    # 画出速度约束最大值
    if v2_max is not None:
        t_max = 0.0
        if len(trajs) > 0:
            T0 = len(trajs[0])
            t_max = (T0 - 1) * dt
        ax2.hlines(
            v2_max,
            xmin=0.0,
            xmax=t_max,
            linestyles="dashed",
            label=f"v2_max = {v2_max:.3f}",
        )

    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("speed ||v||")
    ax2.set_title("Speed profiles of planned trajectories")
    ax2.legend(loc="best")
    ax2.grid(True)
    plt.tight_layout()

    plt.show()


# ------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------
def run_experiment(
    n_demos=10,
    seed=42,
    use_3d=True,
    max_iter=40,
    run_baseline=not True,
    render=True,
):
    np.random.seed(seed)

    # -------- Environment --------
    if use_3d:
        env = ObsAvoidEnv3D(
            start_xy=(-1.5, 0.0),
            subgoal_xy=(0.5, 0.0),
            goal_xy=(-0.2, 0.5),
            obs_center_xy=(-0.5, 0.0),
            obs_radius=0.3,
            start_z_range=(0.2, 0.7),
            subgoal_z=0.4,
            goal_z=0.05,
        )
    else:
        env = ObsAvoidEnv(
            start=(-1.5, 0.0),
            subgoal=(0.5, 0.0),
            goal=(0.0, 0.3),
            obs_center=(-0.5, 0.0),
            obs_radius=0.3,
        )

    # -------- Generate demos --------
    demos, true_taus = [], []
    print(f"Generating {n_demos} demos…")

    for _ in range(n_demos):
        if use_3d:
            X, tau = env.generate_demo_3d(n1=20, direction=None)
        else:
            X, tau = env.generate_demo(n1=20, direction=None)
        demos.append(X)
        true_taus.append(tau)

    print("Generated demos. Example shape:", demos[0].shape)

    # -------- Train GoalHMM3D --------
    print("\nTraining GoalHMM3D…")
    learner = GoalHMM3D(
        demos=demos,
        env=env,
        true_taus=true_taus,
        feat_weight=1.0,
        prog_weight=1.0,
        trans_weight=1.0,
        delta_init=0.15,
        learn_delta=True,
        vmf_lr=8e-4,
        g_step=0.2,
        vmf_steps=3,
        plot_every=max_iter,
        feature_ids=[0, 1, 2, 3],
        main_feat_stage1_raw=0,     # env feature 中表示 "distance-like" 的原始索引
        main_feat_stage2_raw=1,
        feature_types=["margin_exp_lower",  # 0: 距离 -> 下界不等式
                     "gauss",  # 1: 速度 -> 等式/窄带
                     "gauss",  # 2: 假障碍
                     "gauss"],
        auto_feature_select=True,
        r_sparse_lambda=0.3

    # g1_init="random"
    )
    posts = learner.fit(max_iter=max_iter)

    # posterior taus based on learner
    taus_est = []
    for gamma in posts:
        idx = np.where(gamma[:, 1] > 0.5)[0]
        tau_hat = int(idx[0]) if len(idx) > 0 else int(np.argmax(gamma[:, 1]))
        taus_est.append(tau_hat)

    # -------- Baseline (可选) --------
    if run_baseline and not use_3d:
        print("\nTraining HMM-HMM baseline…")
        base = GMMHMM(
            demos=demos,
            env=env,
            true_taus=true_taus,
            plot_every=max_iter,
        )
        base.fit(max_iter=max_iter)

    # ========================================================
    # 使用 learned constraint 做 two-stage optimal control 规划
    # ========================================================

    # if render and use_3d:
        # =====================================================================
        # 从 learner 中提取约束阈值：d_safe_min (stage1) 与 v2_max (stage2)
        # =====================================================================
        # d_safe_min_raw, v2_max_raw = get_feature_constraints_from_learner(learner)
        # print(
        #     f"\n[Constraints from learner] "
        #     f"d_safe_min = {d_safe_min_raw:.3f}, v2_max = {v2_max_raw:.3f}"
        # )
        # #
        # # =====================================================================
        # # Now plan new trajectories using learned constraints
        # # =====================================================================
        # print("\nPlanning new trajectories with learned constraints…")
        #
        # n_plan = 5
        # planned_trajs = []
        # planned_taus = []
        #
        # for i in range(n_plan):
        #     x_start = sample_random_start_3d(env)
        #     X_plan, tau_plan, X1, X2, x_sub, x_goal = plan_two_stage_trajectory(
        #         env,
        #         x_start=x_start,
        #         dt=0.2,
        #         d_safe_min=d_safe_min_raw,  # stage1 distance 约束：dist_xy >= d_safe_min
        #         v1_max=0.9,                 # stage1 max speed（可以继续当作超参数）
        #         v2_max=v2_max_raw,          # stage2 max speed：来自 stage2 速度高斯上界
        #         stage1_max_steps=1220,
        #         stage2_max_steps=3200,
        #         verbose=False,
        #     )
        #     planned_trajs.append(X_plan)
        #     planned_taus.append(tau_plan)
        #     print(f"[Main] planned traj {i}: T={len(X_plan)}, tau={tau_plan}")
        #
        # # --- Matplotlib debug visualization (3D + speed profile) ---
        # debug_plot_plans_3d(
        #     env,
        #     planned_trajs,
        #     planned_taus,
        #     title="Planned trajectories (two-stage controller)",
        #     dt=0.2,
        #     v2_max=v2_max_raw,
        # )
        #
        # # --- PyBullet render new planned trajectories ---
        # print("\nRendering planned trajectories…")
        # renderer = PyBulletRenderer3D(env)
        # renderer.setup_scene()
        # renderer.play_all(
        #     planned_trajs,
        #     planned_taus,
        #     g1=None,
        #     g2=None,
        #     v_target=0.2,
        #     min_dt=1 / 2000.0,
        #     max_dt=0.5,
        # )

        # ========================================================
        # 使用 learned constraint 做 transfer-planning
        # ========================================================
        # d_safe_min_raw, v2_max_raw = get_feature_constraints_from_learner(learner)
        # print(f"[Constraints] d_safe_min={d_safe_min_raw:.3f}, v2_max={v2_max_raw:.3f}")
        #
        # print("\nPlanning new trajectories WITH oscillation transfer…")
        # n_plan = 3
        # planned_trajs = []
        # planned_taus = []
        #
        # for i in range(n_plan):
        #     x_start = sample_random_start_3d(env, xy_jitter=0.5)
        #     x_start[0] += 1.2
        #     x_start[1] -= .5
        #
        #     X_plan, tau_plan, X1, X2, goal_above = plan_oscillate_then_descend(
        #         env,
        #         x_start=x_start,
        #         dt=0.2,
        #         d_safe_min=d_safe_min_raw,
        #         v1_max=0.9,
        #         v2_max=v2_max_raw,
        #         n_round_trips=2,
        #         stage1_max_steps_per_leg=80,
        #         stage2_max_steps=120,
        #         verbose=False,
        #     )
        #     planned_trajs.append(X_plan)
        #     planned_taus.append(tau_plan)
        #     print(f"[Main/Osc] traj {i}: T={len(X_plan)}, tau={tau_plan}")
        #
        # # debug plot（还能看到 stage1 很长，然后 stage2 一段垂直接近）
        # debug_plot_plans_3d(
        #     env,
        #     planned_trajs,
        #     planned_taus,
        #     title="Oscillate-then-descend trajectories",
        #     dt=0.2,
        #     v2_max=v2_max_raw,
        # )
        #
        # # Render（可以继续用 subsample）
        # renderer = PyBulletRenderer3D(env)
        # renderer.setup_scene()
        # renderer.play_all(
        #     planned_trajs,
        #     planned_taus,
        #     g1=None,
        #     g2=None,
        #     v_target=0.2,
        #     min_dt=1 / 60,
        #     max_dt=0.12,
        # )


if __name__ == "__main__":
    run_experiment(
        n_demos=10,
        seed=434421,
        use_3d=True,
        max_iter=45,
        run_baseline=True,
        render=True,
    )
