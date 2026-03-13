# planner/rrt_star.py
# ------------------------------------------------------------
# Simple two-stage RRT* planner in 3D workspace:
#
#   Stage 1: start  -> subgoal, with learned distance constraint
#            (collision if dist_xy < d_safe_min)
#
#   Stage 2: subgoal -> goal, no obstacle constraint (straight line)
#
#   - purely geometric planning (no dynamics, no MPC)
#   - only uses learned d_safe_min as obstacle radius; env.obs_radius
#     is NOT used for planning (can still be used for visualization)
#
# Returns:
#   X_full, tau, X1, X2, x_subgoal, x_goal
#
# ------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from envs.obs_avoid_3d import ObsAvoidEnv3D
from visualization.pybullet_renderer import PyBulletRenderer3D
from visualization.io import save_figure


class _Node:
    def __init__(self, x, parent=None, cost=0.0):
        self.x = np.asarray(x, dtype=float)
        self.parent = parent  # parent index
        self.cost = float(cost)
        self.children = []    # list of child indices


# ============================================================
# Utilities
# ============================================================
def _compute_sampling_bounds(env, margin=0.5):
    """
    根据 start / subgoal / goal + 一点 margin，构建采样边界框。
    不使用 obs_radius 做约束，只是定义一个合理的采样区域。
    """
    pts = []

    if hasattr(env, "start_xy"):
        z_min, z_max = env.start_z_range
        pts.append(np.array([env.start_xy[0], env.start_xy[1], z_min], dtype=float))
        pts.append(np.array([env.start_xy[0], env.start_xy[1], z_max], dtype=float))

    if hasattr(env, "subgoal_xy") and hasattr(env, "subgoal_z"):
        pts.append(
            np.array(
                [env.subgoal_xy[0], env.subgoal_xy[1], env.subgoal_z],
                dtype=float,
            )
        )

    if hasattr(env, "goal_xy") and hasattr(env, "goal_z"):
        pts.append(
            np.array(
                [env.goal_xy[0], env.goal_xy[1], env.goal_z],
                dtype=float,
            )
        )

    P = np.stack(pts, axis=0)
    xyz_min = P.min(axis=0) - margin
    xyz_max = P.max(axis=0) + margin

    return xyz_min, xyz_max


def _sample_free(bounds_min, bounds_max, rng):
    """
    在给定边界框里均匀采样一点。
    """
    return rng.uniform(bounds_min, bounds_max)


def _dist(a, b):
    return np.linalg.norm(a - b)


def _in_collision_point(x, env, d_safe_min):
    """
    点碰撞检测（仅在 Stage 1 使用）：
      - collision <=> dist_xy(x, obs_center_xy) < d_safe_min
    """
    if d_safe_min is None:
        return False

    xy = x[:2]
    center = env.obs_center_xy
    d_xy = np.linalg.norm(xy - center)
    return d_xy < d_safe_min


def _in_collision_segment(p1, p2, env, d_safe_min, step=0.03):
    """
    线段碰撞检测：在 p1->p2 上做若干插值点，每个点都用 _in_collision_point 检测。
    """
    if d_safe_min is None:
        return False

    length = _dist(p1, p2)
    if length < 1e-9:
        return _in_collision_point(p1, env, d_safe_min)

    n_steps = max(2, int(np.ceil(length / step)))
    alphas = np.linspace(0.0, 1.0, n_steps)
    for a in alphas:
        p = (1.0 - a) * p1 + a * p2
        if _in_collision_point(p, env, d_safe_min):
            return True
    return False


def _nearest_node(nodes, x_rand):
    """
    在 nodes 中找离 x_rand 最近的节点索引。
    """
    dists = [np.linalg.norm(n.x - x_rand) for n in nodes]
    return int(np.argmin(dists))


def _find_neighbors(nodes, x_new, radius):
    """
    在 nodes 中找在某个半径内的邻居索引列表。
    """
    idxs = []
    for i, n in enumerate(nodes):
        if np.linalg.norm(n.x - x_new) <= radius:
            idxs.append(i)
    return idxs


def _propagate_cost(nodes, idx, delta):
    """
    当节点 idx 的 cost 发生变化（减小），递归更新其所有子孙节点的 cost。
    """
    nodes[idx].cost += delta
    for c in nodes[idx].children:
        _propagate_cost(nodes, c, delta)


def _extract_path(nodes, goal_idx):
    """
    从 goal_idx 回溯到根节点，得到路径点列表（start->...->goal）。
    """
    path = []
    idx = goal_idx
    while idx is not None:
        path.append(nodes[idx].x)
        idx = nodes[idx].parent
    path.reverse()
    return np.stack(path, axis=0)


def _densify_path(path, ds=0.03):
    """
    对一条折线轨迹进行线性插值，使得相邻点之间的距离不超过 ds。
    """
    path = np.asarray(path)
    dense = [path[0]]
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i + 1]
        seg_len = _dist(p0, p1)
        if seg_len < 1e-9:
            continue
        n_steps = max(1, int(np.ceil(seg_len / ds)))
        for k in range(1, n_steps + 1):
            a = k / n_steps
            dense.append((1.0 - a) * p0 + a * p1)
    return np.stack(dense, axis=0)


# ============================================================
# RRT* core (Stage 1)
# ============================================================
def _rrt_star_stage(
    env: ObsAvoidEnv3D,
    x_start,
    x_goal,
    d_safe_min,
    max_iters=3000,
    step_size=0.18,
    goal_radius=0.15,
    neighbor_radius=0.35,
    bounds_margin=0.5,
    goal_sample_rate=0.1,
    ds_resample=0.03,
    rng=None,
    verbose=False,
):
    """
    在 3D 空间中用 RRT* 规划一条从 x_start 到 x_goal 的几何路径。

    - 仅在 stage1 使用障碍约束：collision if dist_xy < d_safe_min
    - 不使用任何速度 / 动力学约束
    """

    if rng is None:
        rng = np.random

    x_start = np.asarray(x_start, dtype=float)
    x_goal = np.asarray(x_goal, dtype=float)

    # 采样边界
    bounds_min, bounds_max = _compute_sampling_bounds(env, margin=bounds_margin)

    # 初始化树
    nodes = []
    root = _Node(x_start, parent=None, cost=0.0)
    nodes.append(root)

    best_goal_idx = None
    best_goal_cost = np.inf

    for it in range(max_iters):
        # ---- 采样 ----
        if rng.rand() < goal_sample_rate:
            x_rand = x_goal.copy()  # goal bias
        else:
            x_rand = _sample_free(bounds_min, bounds_max, rng)

        # ---- 最近邻 ----
        idx_near = _nearest_node(nodes, x_rand)
        x_near = nodes[idx_near].x

        # ---- Steer ----
        direction = x_rand - x_near
        dist_near = np.linalg.norm(direction)
        if dist_near < 1e-9:
            continue
        direction /= dist_near
        if dist_near > step_size:
            x_new = x_near + direction * step_size
        else:
            x_new = x_rand

        # ---- 碰撞检查：线段 (x_near -> x_new) ----
        if _in_collision_segment(x_near, x_new, env, d_safe_min):
            continue

        # ---- 选父节点（RRT*）----
        # 先假设父节点就是最近邻
        new_parent = idx_near
        new_cost = nodes[idx_near].cost + _dist(x_near, x_new)

        # 在 neighbor_radius 内找更优父节点
        neighbor_idxs = _find_neighbors(nodes, x_new, neighbor_radius)
        for j in neighbor_idxs:
            x_j = nodes[j].x
            # 检查 j -> x_new 无碰撞
            if _in_collision_segment(x_j, x_new, env, d_safe_min):
                continue
            cand_cost = nodes[j].cost + _dist(x_j, x_new)
            if cand_cost < new_cost:
                new_cost = cand_cost
                new_parent = j

        # 创建新节点
        new_idx = len(nodes)
        new_node = _Node(x_new, parent=new_parent, cost=new_cost)
        nodes.append(new_node)
        nodes[new_parent].children.append(new_idx)

        # ---- Rewire ----
        for j in neighbor_idxs:
            if j == new_parent:
                continue
            x_j = nodes[j].x
            # 检查 new -> j 是否可行
            if _in_collision_segment(x_new, x_j, env, d_safe_min):
                continue
            cand_cost = new_cost + _dist(x_new, x_j)
            if cand_cost + 1e-9 < nodes[j].cost:
                # 重新挂接 j 到 new_idx
                old_parent = nodes[j].parent
                if old_parent is not None:
                    if j in nodes[old_parent].children:
                        nodes[old_parent].children.remove(j)
                nodes[new_idx].children.append(j)
                delta = cand_cost - nodes[j].cost
                nodes[j].parent = new_idx
                # 递归更新 j 及其子孙的 cost
                _propagate_cost(nodes, j, delta)

        # ---- 检查是否接近 goal ----
        d_to_goal = _dist(x_new, x_goal)
        if d_to_goal <= goal_radius:
            # 尝试直接连到 goal（确保线段可行）
            if not _in_collision_segment(x_new, x_goal, env, d_safe_min):
                goal_total_cost = new_cost + d_to_goal
                if goal_total_cost < best_goal_cost:
                    # 把 goal 当成一个新节点挂到树上
                    goal_idx = len(nodes)
                    goal_node = _Node(x_goal, parent=new_idx, cost=goal_total_cost)
                    nodes.append(goal_node)
                    nodes[new_idx].children.append(goal_idx)
                    best_goal_idx = goal_idx
                    best_goal_cost = goal_total_cost

        if verbose and (it + 1) % 200 == 0:
            print(
                f"[RRT*] iter={it+1}, nodes={len(nodes)}, "
                f"best_goal_cost={best_goal_cost if np.isfinite(best_goal_cost) else 'inf'}"
            )

    # ---- 构造路径 ----
    if best_goal_idx is None:
        # 没有成功连到 goal，用距离 goal 最近的节点收尾（可能略不精确）
        dists_to_goal = [np.linalg.norm(n.x - x_goal) for n in nodes]
        idx_best = int(np.argmin(dists_to_goal))
        path = _extract_path(nodes, idx_best)
        # 如果最后一段到 goal 无碰撞，就补一段
        if not _in_collision_segment(path[-1], x_goal, env, d_safe_min):
            path = np.vstack([path, x_goal[None, :]])
        if verbose:
            print("[RRT*] WARNING: did not reach goal within radius; using nearest node.")
    else:
        path = _extract_path(nodes, best_goal_idx)
        if verbose:
            print(
                f"[RRT*] Reached goal, nodes={len(nodes)}, "
                f"cost={best_goal_cost:.3f}, path_len={len(path)}"
            )

    # densify for smooth visualization
    path_dense = _densify_path(path, ds=ds_resample)
    return path_dense


# ============================================================
# Public two-stage interface (RRT* + straight line)
# ============================================================
def plan_two_stage_trajectory_rrt(
    env: ObsAvoidEnv3D,
    x_start=None,
    dt=0.02,          # 保留接口兼容（这里不会用到）
    d_safe_min=None,  # learned stage1 distance lower bound
    v1_max=0.9,       # 保留接口兼容（RRT* 不用速度）
    v2_max=0.45,      # 保留接口兼容（stage2 仍然不用速度）
    stage1_max_steps=120,
    stage2_max_steps=80,
    verbose=True,
):
    """
    Two-stage geometric planner:

        Stage 1: start  -> subgoal  (RRT*, collision if dist_xy < d_safe_min)
        Stage 2: subgoal -> goal    (straight line interpolation)

    仅使用 d_safe_min 作为障碍物安全距离（learned constraint）。
    不使用 env.obs_radius 做 planning。
    v1_max, v2_max, dt 保留在接口里以便和 optimal_control 版本兼容，
    但在这里不会参与计算。
    """

    rng = np.random.RandomState()

    # -------- start / subgoal / goal ----------
    if x_start is None:
        # 和你 main.py 的 sample_random_start_3d 风格类似
        if hasattr(env, "start_xy"):
            x0 = np.zeros(3, dtype=float)
            x0[0] = env.start_xy[0] + rng.uniform(-0.05, 0.05)
            x0[1] = env.start_xy[1] + rng.uniform(-0.05, 0.05)
            x0[2] = rng.uniform(env.start_z_range[0], env.start_z_range[1])
            x_start = x0
        else:
            raise ValueError("env has no start_xy / start_z_range; please provide x_start explicitly.")
        if verbose:
            print(f"[RRT* Planner] Random start sampled: {x_start}")
    else:
        x_start = np.asarray(x_start, dtype=float)
        if verbose:
            print(f"[RRT* Planner] Using provided start: {x_start}")

    x_subgoal = np.array(
        [env.subgoal_xy[0], env.subgoal_xy[1], env.subgoal_z],
        dtype=float,
    )
    x_goal = np.array(
        [env.goal_xy[0], env.goal_xy[1], env.goal_z],
        dtype=float,
    )

    # -------- Stage 1: RRT* around obstacle (using d_safe_min) ----------
    if verbose:
        print("[RRT* Planner] Stage 1: start -> subgoal, with learned distance constraint")

    X1 = _rrt_star_stage(
        env=env,
        x_start=x_start,
        x_goal=x_subgoal,
        d_safe_min=d_safe_min,          # ★ 只用 learned safety distance
        max_iters=3000,
        step_size=0.18,
        goal_radius=0.12,
        neighbor_radius=0.35,
        bounds_margin=0.6,
        goal_sample_rate=0.15,
        ds_resample=0.03,
        rng=rng,
        verbose=verbose,
    )

    # -------- Stage 2: straight line from end of stage1 to goal ----------
    if verbose:
        print("[RRT* Planner] Stage 2: subgoal -> goal (straight line, no obstacle)")

    x_start_stage2 = X1[-1]
    # 直接线性插值，保持分辨率和 stage1 相近
    X2 = _densify_path(np.stack([x_start_stage2, x_goal], axis=0), ds=0.03)

    # -------- Assemble full trajectory ----------
    X_full = np.vstack([X1, X2[1:]])   # 避免重复连接点
    tau = X1.shape[0] - 1

    if verbose:
        print(
            f"[RRT* Planner] Stage1 length: {X1.shape[0]}, "
            f"Stage2 length: {X2.shape[0]}, "
            f"cutpoint tau={tau}, total={X_full.shape[0]}"
        )

    return X_full, tau, X1, X2, x_subgoal, x_goal


# ============================================================
# Debug visualization (matplotlib) for a single trajectory
# ============================================================
def _debug_plot_single_3d(env, X_full, tau, title="RRT* planned trajectory"):
    """
    用 matplotlib 画出单条 3D 轨迹，
    阶段1(0..tau) 蓝色，阶段2(tau..end) 红色。
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    X = np.asarray(X_full)
    tau = int(tau)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    # ---- 画 obstacle 的投影圈，仅用于可视化 ----
    cx, cy = env.obs_center_xy
    r = env.obs_radius
    theta = np.linspace(0, 2 * np.pi, 100)
    z_ring = env.subgoal_z if hasattr(env, "subgoal_z") else 0.3

    ax.plot(
        cx + r * np.cos(theta),
        cy + r * np.sin(theta),
        z_ring * np.ones_like(theta),
        "gray",
        alpha=0.4,
        label="obstacle (proj.)",
    )

    # ---- 轨迹 ----
    ax.plot(
        X[: tau + 1, 0],
        X[: tau + 1, 1],
        X[: tau + 1, 2],
        "b-",
        lw=2,
        alpha=0.9,
        label="stage 1",
    )
    ax.plot(
        X[tau:, 0],
        X[tau:, 1],
        X[tau:, 2],
        "r-",
        lw=2,
        alpha=0.9,
        label="stage 2",
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
    save_figure(plt.gcf(), Path("outputs/plots/RRTStar/rrt_star_overview.png"))


# ============================================================
# Local debug main: plan & visualize with RRT*
# ============================================================
def main():
    """
    本地测试入口：

      - 创建一个 ObsAvoidEnv3D
      - 用 RRT* 规划两阶段轨迹
      - 用 matplotlib 画出 3D 轨迹
      - 用 PyBulletRenderer3D 播放规划轨迹
    """
    # --- build a test 3D env, consistent with your other scripts ---
    env = ObsAvoidEnv3D(
        start_xy=(-1.5, 0.0),
        subgoal_xy=(0.5, 0.0),
        goal_xy=(0.2, 0.5),
        obs_center_xy=(-0.5, 0.0),
        obs_radius=0.3,          # 仅用于可视化
        start_z_range=(0.2, 0.7),
        subgoal_z=0.4,
        goal_z=0.2,
    )

    # --- choose a test safety distance (learned d_safe_min 的替身) ---
    # 实际实验里，你会从 SegCons 的 get_feature_constraints_from_learner 拿到 d_safe_min_raw
    d_safe_min_test = 0.32

    print("[RRT* main] Planning one two-stage trajectory with RRT*…")
    X_full, tau, X1, X2, g1, g2 = plan_two_stage_trajectory_rrt(
        env,
        x_start=None,
        dt=0.02,
        d_safe_min=d_safe_min_test,
        v1_max=0.9,
        v2_max=0.45,
        stage1_max_steps=120,
        stage2_max_steps=80,
        verbose=True,
    )

    # --- matplotlib debug plot ---
    _debug_plot_single_3d(env, X_full, tau, title="RRT* 2-stage planned trajectory")

    # --- PyBullet render ---
    print("[RRT* main] Launching PyBullet to play planned trajectory...")
    renderer = PyBulletRenderer3D(env)
    renderer.setup_scene()

    demos = [X_full]
    taus = [tau]

    renderer.play_all(
        demos,
        taus,
        g1=g1,
        g2=g2,
        v_target=0.25,      # renderer will adapt dt based on distance
        min_dt=1 / 120.0,
        max_dt=0.12,
    )

    import time
    while True:
        time.sleep(0.1)


if __name__ == "__main__":
    main()
