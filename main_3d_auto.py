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

def debug_plot_plans_3d(
    env,
    trajs,
    taus,
    title="Planned 2-stage trajectories",
    dt=0.02,
    v2_max=None,
):
    """
    画真正等比例 (1:1:1) 的 3D 轨迹：
    - 阶段1：蓝色
    - 阶段2：红色
    - 自动设定 xyz 等比例坐标范围
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    import numpy as np

    # ======================
    # Figure 1: 3D 轨迹
    # ======================
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # ---- 画 obstacle 的投影圈（用于检查避障）----
    if hasattr(env, "obs_center_xy") and hasattr(env, "obs_radius"):
        cx, cy = env.obs_center_xy
        r = env.obs_radius
        theta = np.linspace(0, 2 * np.pi, 200)

        # 在 subgoal 高度画一圈
        z_ring = env.subgoal_z if hasattr(env, "subgoal_z") else 0.3

        ax.plot(
            cx + r * np.cos(theta),
            cy + r * np.sin(theta),
            z_ring * np.ones_like(theta),
            color="gray",
            alpha=0.4,
            lw=1.5,
        )

    # ---- 收集所有点，用来设定等比例坐标范围 ----
    all_pts = []

    # ---- 画所有轨迹 ----
    for X, tau in zip(trajs, taus):
        X = np.asarray(X)
        tau = int(tau)

        all_pts.append(X)

        # stage 1
        ax.plot(
            X[: tau + 1, 0],
            X[: tau + 1, 1],
            X[: tau + 1, 2],
            color="blue",
            lw=2,
            alpha=0.9,
        )

        # stage 2
        ax.plot(
            X[tau:, 0],
            X[tau:, 1],
            X[tau:, 2],
            color="red",
            lw=2,
            alpha=0.9,
        )

    all_pts = np.concatenate(all_pts, axis=0)

    # ---- subgoal / goal ----
    sg = np.array([env.subgoal_xy[0], env.subgoal_xy[1], env.subgoal_z])
    gg = np.array([env.goal_xy[0], env.goal_xy[1], env.goal_z])

    ax.scatter(
        sg[0], sg[1], sg[2],
        c="orange", marker="D", s=80, label="subgoal", zorder=5
    )
    ax.scatter(
        gg[0], gg[1], gg[2],
        c="green", marker="P", s=90, label="goal", zorder=5
    )

    # 也把 subgoal / goal 加入坐标范围计算
    all_pts = np.vstack([all_pts, sg[None, :], gg[None, :]])

    # ======================
    # ✅ 关键：设置 1:1:1 等比例坐标
    # ======================
    x_min, y_min, z_min = all_pts.min(axis=0)
    x_max, y_max, z_max = all_pts.max(axis=0)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    z_mid = 0.5 * (z_min + z_max)

    max_range = max(
        x_max - x_min,
        y_max - y_min,
        z_max - z_min,
    )

    half = 0.5 * max_range

    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)

    # ✅ matplotlib >= 3.3 才支持
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass  # 老版本 matplotlib 忽略即可

    # ======================
    # 视觉优化
    # ======================
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.legend(loc="best")

    ax.view_init(elev=35, azim=-60)
    ax.grid(True)
    plt.show()


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
    render=not True,
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
    env.estimate_oracle_constraints(demos, true_taus)
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
        posterior_temp=10,

        g1_init='random',
        g_steps=10,
        g_lr=10e-4,
        plot_every=max_iter,
        feature_ids=[0, 1, 2, 3],
        feature_types=["margin_exp_lower",  # 0: 距离 -> 下界不等式
                     "gauss",  # 1: 速度 -> 等式/窄带
                     "gauss",  # 2: 假障碍
                     "gauss"],
        auto_feature_select=True,
        r_sparse_lambda=0.3
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

    if render and use_3d:
        assert (learner.feature_ids is None) or (0 in learner.feature_ids and 1 in learner.feature_ids)
        # =====================================================================
        # 从 learner 中提取约束阈值：d_safe_min (stage1) 与 v2_max (stage2)
        # =====================================================================
        d_safe_min_raw = learner.feature_models[0][0].L * learner.feat_std[0] + learner.feat_mean[0]
        v2_max_raw = learner.feature_models[1][1].U * learner.feat_std[1] + learner.feat_mean[1]
        print(
            f"\n[Constraints from learner] "
            f"d_safe_min = {d_safe_min_raw:.3f}, v2_max = {v2_max_raw:.3f}"
        )
        #
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
        print(f"[Constraints] d_safe_min={d_safe_min_raw:.3f}, v2_max={v2_max_raw:.3f}")

        print("\nPlanning new trajectories WITH oscillation transfer…")
        n_plan = 3
        planned_trajs = []
        planned_taus = []

        for i in range(n_plan):
            x_start = sample_random_start_3d(env, xy_jitter=0.5)
            x_start[0] += 1.2
            x_start[1] -= .5

            X_plan, tau_plan, X1, X2, goal_above = plan_oscillate_then_descend(
                env,
                x_start=x_start,
                dt=0.2,
                d_safe_min=d_safe_min_raw,
                v1_max=0.9,
                v2_max=v2_max_raw,
                n_round_trips=2,
                stage1_max_steps_per_leg=80,
                stage2_max_steps=120,
                verbose=False,
            )
            planned_trajs.append(X_plan)
            planned_taus.append(tau_plan)
            print(f"[Main/Osc] traj {i}: T={len(X_plan)}, tau={tau_plan}")

        # debug plot（还能看到 stage1 很长，然后 stage2 一段垂直接近）
        debug_plot_plans_3d(
            env,
            planned_trajs,
            planned_taus,
            title="Oscillate-then-descend trajectories",
            dt=0.2,
            v2_max=v2_max_raw,
        )

        # Render（可以继续用 subsample）
        renderer = PyBulletRenderer3D(env)
        renderer.setup_scene()
        renderer.play_all(
            planned_trajs,
            planned_taus,
            g1=None,
            g2=None,
            v_target=0.2,
            min_dt=1 / 60,
            max_dt=0.12,
        )


if __name__ == "__main__":
    run_experiment(
        n_demos=10,
        seed=43122,  # 43122在stage2   # 832, 圆柱左边
        use_3d=True,
        max_iter=60,
        run_baseline=True,
        render=True,
    )
