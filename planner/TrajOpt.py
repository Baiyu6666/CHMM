import numpy as np
import cvxpy as cp

from envs.obs_avoid_3d import ObsAvoidEnv3D
from visualization.pybullet_renderer import PyBulletRenderer3D


def scp_trajopt_stage(
    x_start,
    x_goal,
    T,
    dt,
    v_max,
    obs_center=None,
    obs_radius=None,
    obs_margin=0.05,
    safe_extra=None,
    n_scp_iters=10,
    w_goal=10.0,
    w_smooth=0.2,
):
    """
    单阶段 TrajOpt（SCP 版本，完全 QP，可用 OSQP 解决）：

    目标:
        min_X  Σ_t ||x_t - x_goal||^2  +  w_smooth Σ_t ||x_{t+1} - x_t||^2

    约束:
        1) 速度逐坐标限幅（近似 max-norm 速度约束）:
               -v_max*dt <= x_{t+1}^i - x_t^i <= v_max*dt

        2) （可选）障碍物约束，使用线性化 signed-distance:
               d_k + n_k^T (x_t_xy - xk_xy) >= R_safe

           其中 xk_xy 是当前迭代的轨迹点，n_k 是外法向。
    """
    x_start = np.asarray(x_start, float)
    x_goal = np.asarray(x_goal, float)

    # ------- 初始轨迹：直线 -------
    t_lin = np.linspace(0.0, 1.0, T + 1)
    Xk = (1.0 - t_lin)[:, None] * x_start[None, :] + t_lin[:, None] * x_goal[None, :]

    # 障碍物参数
    use_obs = obs_center is not None and obs_radius is not None
    if use_obs:
        obs_center = np.asarray(obs_center, float)
        geom_safe = obs_radius + obs_margin
        R_safe = max(geom_safe, float(safe_extra)) if safe_extra is not None else geom_safe

    for it in range(n_scp_iters):
        # ------- 构建 QP 变量和约束 -------
        X = cp.Variable((T + 1, 3))
        cons = []

        # 起点终点
        cons += [X[0] == x_start]
        cons += [X[T] == x_goal]

        # 1) 速度约束：逐坐标限幅，保证是线性约束
        for t in range(T):
            dv = X[t + 1] - X[t]           # (3,)
            for d in range(3):
                cons += [dv[d] <= v_max * dt]
                cons += [dv[d] >= -v_max * dt]

        # 2) 线性化的障碍约束（只对中间点）
        if use_obs:
            for t in range(1, T):
                xk_xy = Xk[t, :2]                   # 当前迭代的 xy
                diff = xk_xy - obs_center
                dk = np.linalg.norm(diff)

                if dk < 1e-6:
                    nk = np.array([1.0, 0.0])
                    dk = R_safe
                else:
                    nk = diff / dk                  # 外法向

                # 线性化 SDF 约束：
                # d(x) ≈ dk + n_k^T (x_xy - xk_xy) >= R_safe
                cons += [
                    dk + cp.sum(cp.multiply(nk, X[t, :2] - xk_xy)) >= R_safe
                ]

        # ------- 目标函数 -------
        cost = 0

        # 靠近目标
        for t in range(T + 1):
            cost += w_goal * cp.sum_squares(X[t] - x_goal)

        # 平滑正则
        for t in range(T):
            cost += w_smooth * cp.sum_squares(X[t + 1] - X[t])

        prob = cp.Problem(cp.Minimize(cost), cons)
        # 完全 QP：二次目标 + 线性约束，用 OSQP 即可
        prob.solve(solver=cp.OSQP, warm_start=True)

        if X.value is None:
            raise RuntimeError(f"SCP failed at iter {it}")

        Xk = X.value.copy()

    return Xk


def compute_T(start, goal, v_max, dt, max_steps, extra=8):
    start = np.asarray(start, float)
    goal = np.asarray(goal, float)
    D = np.linalg.norm(goal - start)
    T_min = int(np.ceil(D / (v_max * dt + 1e-9)))
    return min(max_steps, T_min + extra)


def plan_two_stage_trajopt(
    env: ObsAvoidEnv3D,
    x_start=None,
    dt=0.02,
    v1_max=0.9,
    v2_max=0.45,
    stage1_max_steps=80,
    stage2_max_steps=60,
    obs_margin=0.05,
    d_safe_min=None,
):
    # ---------- start / subgoal / goal ----------
    if x_start is None:
        x_start = np.array([
            env.start_xy[0],
            env.start_xy[1],
            np.random.uniform(*env.start_z_range)
        ], float)
    else:
        x_start = np.asarray(x_start, float)

    x_sub = np.array([env.subgoal_xy[0], env.subgoal_xy[1], env.subgoal_z], float)
    x_goal = np.array([env.goal_xy[0], env.goal_xy[1], env.goal_z], float)

    # ---------- horizon ----------
    T1 = compute_T(x_start, x_sub, v1_max, dt, stage1_max_steps)
    T2 = compute_T(x_sub, x_goal, v2_max, dt, stage2_max_steps)

    # ---------- Stage 1: 有障碍 ----------
    X1 = scp_trajopt_stage(
        x_start=x_start,
        x_goal=x_sub,
        T=T1,
        dt=dt,
        v_max=v1_max,
        obs_center=env.obs_center_xy,
        obs_radius=env.obs_radius,
        obs_margin=obs_margin,
        safe_extra=d_safe_min,
        n_scp_iters=12,
        w_goal=12.0,
        w_smooth=0.1,
    )

    # ---------- Stage 2: 无障碍 ----------
    X2 = scp_trajopt_stage(
        x_start=x_sub,
        x_goal=x_goal,
        T=T2,
        dt=dt,
        v_max=v2_max,
        obs_center=None,
        obs_radius=None,
        obs_margin=0.0,
        safe_extra=None,
        n_scp_iters=8,
        w_goal=12.0,
        w_smooth=0.1,
    )

    X_full = np.vstack([X1[:-1], X2])
    tau = X1.shape[0] - 1
    return X_full, tau, X1, X2, x_sub, x_goal


def main():
    # --- build env consistent with your original setup ---
    env = ObsAvoidEnv3D(
        start_xy=(-1.5, 0.0),
        subgoal_xy=(0.5, 0.0),
        goal_xy=(0.2, 0.5),
        obs_center_xy=(-0.5, 0.0),
        obs_radius=0.3,
        start_z_range=(0.2, 0.7),
        subgoal_z=0.4,
        goal_z=0.2,
    )

    print("\nRunning TrajOpt planner ...")

    X_full, tau, X1, X2, g1, g2 = plan_two_stage_trajopt(
        env,
        dt=0.02,
        v1_max=0.9,
        v2_max=0.45,
        stage1_max_steps=90,
        stage2_max_steps=60,
        obs_margin=0.05,
        d_safe_min=0.25,
    )

    print(f"Stage1 length: {X1.shape[0]}, Stage2 length: {X2.shape[0]}, tau = {tau}, total = {X_full.shape[0]}")

    renderer = PyBulletRenderer3D(env)
    renderer.setup_scene()
    renderer.play_all(
        [X_full],
        [tau],
        g1=g1,
        g2=g2,
        v_target=0.25,
        min_dt=1 / 120.0,
        max_dt=0.12,
    )

    print("Close the PyBullet window to exit.")


if __name__ == "__main__":
    main()
