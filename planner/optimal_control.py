# planner/optimal_control.py
# ------------------------------------------------------------
# Two-stage optimal-control style planner in workspace, MPC-style:
#
#   dynamics:   x_{t+1} = x_t + v_t * dt
#
#   - stage 1:  start  -> subgoal  (with learned distance constraint)
#   - stage 2:  subgoal -> goal    (no distance constraint)
#
# Decision variables (per stage):
#   - velocities v_0, ..., v_{T-1}  in R^3
#   - states x_t obtained by integration (not decision variables)
#
# Constraints (all via penalties, NO projection):
#   - speed bound (soft):      ||v_t|| <= v_max
#   - distance constraint (soft, stage1 only, learned):
#         dist_xy(x_t, obs_center) >= d_safe_min
#
# Objective for a stage with goal x_goal:
#
#   J = w_goal   * sum_t ||x_t - x_goal||           (L2, not squared)
#     + w_vel    * sum_t ||v_t||^2
#     + w_speed  * sum_t max(0, ||v_t|| - v_max)^2
#     + w_obs    * sum_t max(0, d_safe_min - dist_xy(x_t))^2   (stage1 only)
#     + w_term   * ||x_T - x_goal||                 (terminal distance)
#     + w_endv   * ||v_{T-1}||^2                    (encourage slow down at end)
#
# ------------------------------------------------------------

import numpy as np
import torch

from envs.obs_avoid_3d import ObsAvoidEnv3D
from visualization.pybullet_renderer import PyBulletRenderer3D


def sample_random_start_3d(env, rng=None):
    """
    从环境里采一个随机起点：
      - x 在障碍左侧一段范围内
      - y 在小范围均匀
      - z 在 start_z_range 内

    注意：这里用到 obs_radius 只是为了采样起点的几何范围，
    不参与约束建模。
    """
    if rng is None:
        rng = np.random

    cx, cy = env.obs_center_xy
    R = env.obs_radius

    # x 在障碍左边 [cx - 2.5R, cx - 1.5R]
    x_min = cx - 2.5 * R
    x_max = cx - 1.5 * R
    x = rng.uniform(x_min, x_max)

    # y 在 [-0.4R, 0.4R] 一点点偏移
    y = rng.uniform(cy - 0.4 * R, cy + 0.4 * R)

    # z 从 env.start_z_range 里采
    z = rng.uniform(env.start_z_range[0], env.start_z_range[1])

    return np.array([x, y, z], dtype=float)


# ============================================================
# Low-level utilities
# ============================================================
def _forward_integration_torch(x_start, V, dt):
    """
    x_start: (3,) torch tensor
    V: (T,3) torch tensor
    returns:
        X: (T+1,3) torch tensor, with X[0]=x_start and
           X[t+1] = X[t] + V[t] * dt
    """
    T = V.shape[0]
    X = [x_start]
    for t in range(T):
        X.append(X[-1] + V[t] * dt)
    return torch.stack(X, dim=0)  # (T+1,3)


def _plan_single_stage(
    x_start,
    x_goal,
    dt,
    v_max,
    max_steps,
    has_obs=False,
    obs_center_xy=None,
    d_safe_min=None,        # learned distance lower bound (2σ)
    n_iters=100,
    w_goal=1.0,
    w_vel=0.1,
    w_speed=10.0,
    w_obs=50.0,
    w_term=2.0,
    w_endv=1.0,
    verbose=False,
):
    """
    Plan one stage with MPC-style velocity optimization + PyTorch autograd.

    Decision variables: velocities V[0:T-1], each in R^3.
    States X are obtained via integration.

    Objective:
        J = w_goal   * sum_t ||x_t - x_goal||           (L2, not squared)
          + w_vel    * sum_t ||v_t||^2
          + w_speed  * sum_t max(0, ||v_t|| - v_max)^2
          + w_obs    * sum_t max(0, d_safe_min - dist_xy(x_t))^2
          + w_term   * ||x_T - x_goal||
          + w_endv   * ||v_{T-1}||^2

    只用 d_safe_min 做障碍距离约束；obs_radius 不参与规划。
    """

    device = torch.device("cpu")

    x_start_np = np.asarray(x_start, dtype=float)
    x_goal_np = np.asarray(x_goal, dtype=float)

    # -------------------------------
    # Horizon length T based on distance & v_max
    # -------------------------------
    base_dist = np.linalg.norm(x_goal_np - x_start_np)
    min_steps = max(8, int(np.ceil(base_dist / (v_max * dt + 1e-8))))
    T = min(max_steps, min_steps + 8)

    # -------------------------------
    # Initialize velocities as straight-line direction + small noise
    # -------------------------------
    if base_dist < 1e-8:
        direction = np.zeros(3, dtype=float)
    else:
        direction = (x_goal_np - x_start_np) / base_dist

    avg_speed = min(v_max * 0.8, base_dist / (T * dt + 1e-8))
    V_init = np.tile(direction * avg_speed, (T, 1))
    V_init += 0.01 * np.random.randn(T, 3)

    V = torch.tensor(V_init, dtype=torch.float32, device=device, requires_grad=True)
    x_start_t = torch.tensor(x_start_np, dtype=torch.float32, device=device)
    x_goal_t = torch.tensor(x_goal_np, dtype=torch.float32, device=device)

    # -------------------------------
    # Obstacle info (only center + learned d_safe_min)
    # -------------------------------
    use_obs = bool(has_obs and (obs_center_xy is not None) and (d_safe_min is not None))
    if use_obs:
        obs_center_xy_t = torch.tensor(
            np.asarray(obs_center_xy, dtype=float), dtype=torch.float32, device=device
        )
        R_safe = float(d_safe_min)
    else:
        obs_center_xy_t = None
        R_safe = None

    optimizer = torch.optim.Adam([V], lr=0.04)
    eps = 1e-8

    for it in range(n_iters):
        optimizer.zero_grad()

        # ---- forward pass ----
        X = _forward_integration_torch(x_start_t, V, dt)  # (T+1,3)
        v = V                                             # (T,3)
        speed = torch.norm(v, dim=1)                     # (T,)

        # (1) goal proximity (L2, not squared)
        d_goal = torch.norm(X - x_goal_t[None, :], dim=1)   # (T+1,)
        J_goal = d_goal.sum()

        # (2) velocity energy
        J_vel = (v ** 2).sum()

        # (3) speed soft bound
        exceed = torch.clamp(speed - v_max, min=0.0)
        J_speed = (exceed ** 2).sum()

        # (4) obstacle penalty
        J_obs = torch.tensor(0.0, device=device)
        if use_obs:
            xy = X[:, :2]                                             # (T+1,2)
            d_xy = torch.norm(xy - obs_center_xy_t[None, :], dim=1)   # (T+1,)
            viol = torch.clamp(R_safe - d_xy, min=0.0)
            J_obs = (viol ** 2).sum()

        # (5) terminal distance (L2)
        J_term = torch.norm(X[-1] - x_goal_t, p=2)

        # (6) end velocity penalty
        J_endv = (v[-1] ** 2).sum()

        J_total = (
            w_goal * J_goal
            + w_vel * J_vel
            + w_speed * J_speed
            + w_obs * J_obs
            + w_term * J_term
            + w_endv * J_endv
        )

        if not torch.isfinite(J_total):
            print(f"[Stage opt] non-finite J at iter {it}, fallback to init velocity.")
            with torch.no_grad():
                V.copy_(torch.tensor(V_init, dtype=torch.float32, device=device))
            break

        if verbose and it % 50 == 0:
            print(
                f"[Stage opt] iter={it}, J={J_total.item():.4f}, "
                f"goal={w_goal*J_goal.item():.4f}, vel={w_vel*J_vel.item():.4f}, "
                f"speed={w_speed*J_speed.item():.4f}, obs={w_obs*J_obs.item():.4f}, "
                f"term={w_term*J_term.item():.4f}"
            )

        # backward & step
        J_total.backward()

        # gradient clipping on V
        with torch.no_grad():
            g_norm = V.grad.norm(dim=1).max()
            if g_norm > 200.0:
                V.grad *= (200.0 / (g_norm + eps))

        optimizer.step()

        if not torch.isfinite(V).all():
            print(
                f"[Stage opt] numerical issue at iter {it}, "
                "fallback to init velocity."
            )
            with torch.no_grad():
                V.copy_(torch.tensor(V_init, dtype=torch.float32, device=device))
            break

    # 最终 forward 一次得到轨迹并返回 numpy
    with torch.no_grad():
        X_final = _forward_integration_torch(x_start_t, V, dt).cpu().numpy()

    return X_final


def plan_two_stage_trajectory(
    env: ObsAvoidEnv3D,
    x_start=None,
    dt=0.02,
    d_safe_min=None,       # learned stage1 distance 2σ lower bound
    v1_max=0.9,
    v2_max=0.45,           # 阶段2最大速度，直接给数值
    stage1_max_steps=120,
    stage2_max_steps=80,
    verbose=True,
):
    """
    Build a full two-stage trajectory:

        Stage 1: start  -> subgoal (with learned distance constraint)
        Stage 2: subgoal -> goal   (no distance constraint)

    规划中只使用 d_safe_min 做距离约束，不使用 obs_radius。
    """

    # -------- start / subgoal / goal ----------
    if x_start is None:
        x_start = sample_random_start_3d(env)
        if verbose:
            print(f"[Planner] Random start sampled: {x_start}")
    else:
        x_start = np.asarray(x_start, dtype=float)
        if verbose:
            print(f"[Planner] Using provided start: {x_start}")

    x_subgoal = np.array(
        [env.subgoal_xy[0], env.subgoal_xy[1], env.subgoal_z], dtype=float
    )
    x_goal = np.array([env.goal_xy[0], env.goal_xy[1], env.goal_z], dtype=float)

    # -------- Stage 1: start -> subgoal, with learned distance constraint ----------
    if verbose:
        print("[Planner] Stage 1: start -> subgoal, with learned distance constraint")

    X1 = _plan_single_stage(
        x_start=x_start,
        x_goal=x_subgoal,
        dt=dt,
        v_max=v1_max,
        max_steps=stage1_max_steps,
        has_obs=True,
        obs_center_xy=env.obs_center_xy,
        d_safe_min=d_safe_min,   # ★ 只用 learned d_safe_min
        n_iters=100,
        w_goal=0.7,
        w_vel=0.1,
        w_speed=20.0,
        w_obs=80.0,
        w_term=3.0,
        w_endv=1.0,
        verbose=verbose,
    )

    # -------- Stage 2: subgoal -> goal, no distance constraint ----------
    if verbose:
        print(f"[Planner] Stage 2: subgoal -> final goal, v2_max = {v2_max:.3f}")

    # ★ 用 stage1 的真实终点作为 stage2 起点，保证轨迹连续
    x_start_stage2 = X1[-1]

    X2 = _plan_single_stage(
        x_start=x_start_stage2,
        x_goal=x_goal,
        dt=dt,
        v_max=v2_max,
        max_steps=stage2_max_steps,
        has_obs=False,
        obs_center_xy=None,
        d_safe_min=None,
        n_iters=50,
        w_goal=0.7,
        w_vel=0.1,
        w_speed=20.0,
        w_obs=0.0,
        w_term=3.0,
        w_endv=1.0,
        verbose=verbose,
    )

    # ★ 拼接时：保留 X1 全部点 + X2 去掉第一个点（避免重复）
    X_full = np.vstack([X1, X2[1:]])
    tau = X1.shape[0] - 1

    if verbose:
        print(
            f"[Planner] Stage1 length: {X1.shape[0]}, "
            f"Stage2 length: {X2.shape[0]}, "
            f"cutpoint tau={tau}, total={X_full.shape[0]}"
        )

    return X_full, tau, X1, X2, x_subgoal, x_goal

import numpy as np
from envs.obs_avoid_3d import ObsAvoidEnv3D  # 顶部已经有就不用重复

def plan_oscillate_then_descend(
    env: ObsAvoidEnv3D,
    x_start=None,
    dt=0.02,
    d_safe_min=None,     # stage1: learned distance lower bound
    v1_max=0.9,          # stage1 max speed
    v2_max=0.4,          # stage2 max speed (from learner)
    n_round_trips=2,     # 往返次数
    stage1_max_steps_per_leg=80,
    stage2_max_steps=120,
    verbose=True,
):
    """
    Transfer 场景：
      - Stage1: 在两点 A、B 之间往返 n_round_trips 次（AB 连线穿过杯子几何）
      - Stage2: 从最后一点移动到“杯子正上方某个高度”的目标点，
                只受速度约束 v2_max（阶段2约束）。

    返回:
      X_full : (T,3) 全轨迹
      tau    : int，整个 Stage1 结束的 cutpoint index
      X_stage1, X_stage2, goal_above
    """
    # -------- 0. 起点 --------
    if x_start is None:
        # 复用你 main 里的采样逻辑也可以，这里简单给一个在桌面左侧的起点
        cx, cy = env.obs_center_xy
        z0 = np.mean(env.start_z_range)
        x_start = np.array([cx , cy - 1.4, z0], dtype=float)
        if verbose:
            print(f"[Osc] Using default start: {x_start}")
    else:
        x_start = np.asarray(x_start, dtype=float)
        if verbose:
            print(f"[Osc] Using provided start: {x_start}")

    # -------- 1. 定义 A/B 两个往返点 --------
    cx, cy = env.obs_center_xy
    R = env.obs_radius + 0.25  # 稍微绕出杯子一段
    z_mid = env.subgoal_z      # 在 subgoal 高度附近运动

    # 让 AB 大致穿过杯子：取杯子左下 / 右上两个点
    A = np.array([cx - R, cy - 0.4 * R, z_mid], dtype=float)
    B = np.array([cx + R, cy + 0.4 * R, z_mid], dtype=float)

    if verbose:
        print(f"[Osc] A = {A}, B = {B}")

    # -------- 2. Stage1: 起点 -> A -> B -> A -> B ... 往返 --------
    X_pieces = []

    # leg0: x_start -> A
    X0 = _plan_single_stage(
        x_start=x_start,
        x_goal=A,
        dt=dt,
        v_max=v1_max,
        max_steps=stage1_max_steps_per_leg,
        has_obs=True,
        obs_center_xy=env.obs_center_xy,
        d_safe_min=d_safe_min,  # 只允许用 learned 的 distance 下界
        n_iters=400,
        w_vel=1.0,
        w_obs=200.0,
        verbose=verbose,
    )
    X_pieces.append(X0)

    # leg1..: 在 A / B 之间往返
    cur = A
    target = B
    for k in range(2 * n_round_trips - 1):
        Xk = _plan_single_stage(
            x_start=cur,
            x_goal=target,
            dt=dt,
            v_max=v1_max,
            max_steps=stage1_max_steps_per_leg,
            has_obs=True,
            obs_center_xy=env.obs_center_xy,
            d_safe_min=d_safe_min,
            n_iters=350,
            w_vel=1.0,
            w_obs=200.0,
            verbose=False,
        )
        X_pieces.append(Xk)
        # 下一条 leg 的起点 = 当前终点
        cur = Xk[-1]
        target = A if np.allclose(target, B) else B

    # 拼接 stage1 轨迹（去掉前几段的尾点避免重复）
    X_stage1 = X_pieces[0]
    for seg in X_pieces[1:]:
        X_stage1 = np.vstack([X_stage1[:-1], seg])  # 去掉 seg[0]

    # -------- 3. Stage2: 从 Stage1 终点 -> 杯子正上方 --------
    end_stage1 = X_stage1[-1]

    z_above = max(env.subgoal_z, env.goal_z, env.start_z_range[1])
    goal_above = np.array([cx, cy, z_above], dtype=float)

    if verbose:
        print(f"[Osc] Stage2 goal above mug: {goal_above}")

    X_stage2 = _plan_single_stage(
        x_start=end_stage1,
        x_goal=goal_above,
        dt=dt,
        v_max=v2_max,
        max_steps=stage2_max_steps,
        has_obs=False,          # 阶段2不再有 distance constraint，只用速度约束
        obs_center_xy=None,
        d_safe_min=None,
        n_iters=300,
        w_vel=1.0,
        w_obs=0.0,
        verbose=verbose,
    )

    # -------- 4. 拼接 + cutpoint --------
    X_full = np.vstack([X_stage1[:-1], X_stage2])
    tau = X_stage1.shape[0] - 1

    if verbose:
        print(f"[Osc] Stage1 length={X_stage1.shape[0]}, "
              f"Stage2 length={X_stage2.shape[0]}, "
              f"tau={tau}, total={X_full.shape[0]}")

    return X_full, tau, X_stage1, X_stage2, goal_above

# ============================================================
# Local debug main: plan & visualize
# ============================================================
def main():
    # --- build a test 3D env, consistent with your main.py ---
    env = ObsAvoidEnv3D(
        start_xy=(-1.5, 0.0),
        subgoal_xy=(0.5, 0.0),
        goal_xy=(0.2, 0.5),
        obs_center_xy=(-0.5, 0.0),
        obs_radius=0.3,
        start_z_range=(0.2, 0.7),
        subgoal_z=0.4,
        goal_z=0.05,
    )

    # --- plan ---
    X_full, tau, X1, X2, g1, g2 = plan_two_stage_trajectory(
        env,
        dt=0.2,
        v1_max=0.9,
        v2_max=0.4,
        stage1_max_steps=120,
        stage2_max_steps=80,
        d_safe_min=0.3,   # 测试时随便给一个 learned 下界
        verbose=True,
    )

    # --- visualize with your PyBullet renderer ---
    renderer = PyBulletRenderer3D(env)
    renderer.setup_scene()

    demos = [X_full]
    taus = [tau]

    print("[Planner] Launching PyBullet to play planned trajectory...")
    renderer.play_all(
        demos,
        taus,
        g1=g1,
        g2=g2,
        v_target=0.2,      # play speed
        min_dt=1 / 2000.0,
        max_dt=0.5,
    )

    import time
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
