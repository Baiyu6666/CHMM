# envs/sine_corridor_3d.py
import numpy as np

from .base import TaskBundle


class SineCorridorEnv3D:
    """
    3D 版 Sine Corridor 环境，用于和 SegCons 对接。

    轨迹 X[t] = (x, y, z):
      - 一阶段：沿正弦走廊从随机起点走到固定 subgoal
          * 起点满足约束 manifold: y0 = A sin(ω x0)
          * x0 在给定区间随机；z0 在给定区间随机
          * 生成 demo 时：
              x(t): 线性插值 x0 -> x_sub
              y(t): centerline(x) = A sin(ω x)，可叠加少量噪声
              z(t): 线性插值 z0 -> z_sub，可叠加少量噪声
      - 二阶段（可选）：当前版本不强行生成第二阶段，你后续可以在此基础上扩展
        比如从 subgoal 继续到 goal，并在该阶段施加速度下限约束。

    提供特征：
      feat0: dist_to_centerline = |y - y_c(x)|            (distance-like, stage1 主约束)
      feat1: speed (范数 / dt)                            (用于 stage2 的速度约束)
      feat2: z 坐标（可做额外分析用）
      feat3: 随机噪声特征（便于测试 feature selection）

    与现有代码兼容的属性/方法：
      - self.subgoal: 3D 向量
      - self.goal:    3D 向量（暂时可以设为和 subgoal 相同，或者另一个点）
      - self.obs_center_xy: (cx, cy)，供 3D plot 画一个 cylinder 用
      - self.obs_radius:    半径
      - compute_all_features_matrix(X) -> (T, M_raw)
      - compute_features_all(X) -> (distance_like, speed)
    """

    def __init__(
        self,
        A=0.1,
        omega=10.0,
        bias=0.2,
        phase=0.0,
        x_start_range=(-2.0, -1.0),
        z_start_range=(0, 0.5),
        x_sub=0.0,
        z_sub=0.6,
        goal=(0.5, 0, 0),
        # ---- 二阶段最大速度（约束）----
        corridor_half_width=0.2,
        dt=1,
    ):
        """
        参数:
          A, omega            : 正弦中心线 y_c(x) = A sin(ω x)
          x_start_range       : 起点 x0 的均匀采样区间
          z_start_range       : 起点 z0 的均匀采样区间
          x_sub, z_sub        : 所有 demo 共用的 subgoal.x / subgoal.z
                                subgoal.y = centerline(x_sub)
          corridor_half_width : 仅用于 3D plot 时画“障碍柱”的半径
          dt                  : 时间步长（用于计算速度）
        """
        self.A = float(A)
        self.omega = float(omega)
        self.bias = float(bias)
        self.phase = float(phase)

        self.x_start_range = tuple(x_start_range)
        self.z_start_range = tuple(z_start_range)
        self.dt = float(dt)

        # ---- subgoal / goal ----
        y_sub = self.centerline(x_sub)
        self.subgoal = np.array([x_sub, y_sub, z_sub], dtype=float)

        # 目前先让 goal = subgoal，后面你要加第二阶段时再改成别的点即可
        self.goal = np.array(goal, dtype=float)
        self.true_constraints = None

        # ---- 为了兼容 3D 绘图中的“障碍柱”接口 ----
        # 这里随便放一个 cylinder 在原点附近，半径 = corridor_half_width
        self.obs_center_xy = np.array([0.0, 0.0], dtype=float)
        self.obs_radius = float(corridor_half_width)

    # ==========================================================
    # 几何 & 中心线
    # ==========================================================
    def centerline(self, x):
        """正弦中心线 y_c(x) = A sin(ω x)。x 可以是标量或 ndarray。"""
        return self.A * np.sin(self.omega * x + self.phase) + self.bias

    # ==========================================================
    # Demo 生成
    # ==========================================================
    def rollout_demo(
            self,
            n1=20,
            speed_ratio=0.7,
            speed_noise_std=0.0,
    ):
        """
        生成一条两阶段 demo（和 env2d 的逻辑对齐）：

          - 阶段1: 在正弦走廊上从随机起点走到 subgoal
          - 阶段2: 从 subgoal 直线走到 goal，速度 profile 由 speed_ratio 控制

        返回:
          traj     : ndarray(T, 3)
          true_tau : int，阶段1最后一个 index
        """

        def resample_by_speed_profile(points, step_weights):
            pts = np.asarray(points, float)
            if len(pts) < 2:
                return pts

            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            L = s[-1]

            w = np.asarray(step_weights, float)
            # 给速度 profile 加一点噪声（可选）
            if speed_noise_std > 0:
                w = w * (1.0 + np.random.randn(*w.shape) * speed_noise_std)
            w = np.clip(w, 1e-9, None)

            d = (w / w.sum()) * max(L, 1e-12)
            sp = np.concatenate([[0.0], np.cumsum(d)])

            out, j = [], 0
            for v in sp:
                while j + 1 < len(s) and s[j + 1] < v:
                    j += 1
                if j + 1 >= len(s):
                    out.append(pts[-1])
                else:
                    t = (v - s[j]) / max(s[j + 1] - s[j], 1e-12)
                    out.append(pts[j] * (1 - t) + pts[j + 1] * t)
            return np.asarray(out)

        # ============ (0) 随机起点 ============
        x0 = np.random.uniform(*self.x_start_range)
        z0 = np.random.uniform(*self.z_start_range)
        y0 = self.centerline(x0)
        p0 = np.array([x0, y0, z0], dtype=float)

        x_sub, y_sub, z_sub = self.subgoal
        p_sub = np.array([x_sub, y_sub, z_sub], dtype=float)

        p_goal = self.goal

        # ============ (1) 先生成几何路径 stage1_raw / stage2_raw ============
        # stage1_raw: 沿着正弦中心线从 p0 -> p_sub
        xs1 = np.linspace(x0, x_sub, num=max(n1 + 1, 6))
        ys1_center = self.centerline(xs1)
        zs1 = np.linspace(z0, z_sub, num=len(xs1))
        stage1_raw = np.stack([xs1, ys1_center, zs1], axis=1)  # (N1_raw, 3)

        # stage2_raw: 直线从 subgoal -> goal
        N2_raw = 20
        alphas2 = np.linspace(0.0, 1.0, num=N2_raw)
        stage2_raw = p_sub[None, :] + alphas2[:, None] * (p_goal[None, :] - p_sub[None, :])

        # ============ (2) speed profiles + resample ============
        # ---- 基础步长（用阶段1的几何路径）----
        base_step1 = np.mean(np.linalg.norm(np.diff(stage1_raw, axis=0), axis=1))
        base_step1 = max(base_step1, 1e-6)

        # 二阶段 target 步长：和 2D 一样用 speed_ratio * base_step1
        target_step2 = speed_ratio * base_step1

        # ---- stage1 speed: plateau then slow down ----
        M1 = max(6, len(stage1_raw) - 1)
        u1 = (np.arange(M1) + 0.5) / M1
        eps_step = 0.05 * base_step1
        plateau_frac = 0.40
        r = np.clip((u1 - plateau_frac) / max(1e-6, (1.0 - plateau_frac)), 0.0, 1.0)
        smooth = 3 * r ** 2 - 2 * r ** 3
        v1 = (1.0 - smooth) * base_step1 + smooth * eps_step

        stage1 = resample_by_speed_profile(stage1_raw, v1)  # (T1, 3)
        true_tau = len(stage1) - 1

        # ---- stage2 speed: ramp up to target_step2 then constant ----
        L2 = np.sum(np.linalg.norm(np.diff(stage2_raw, axis=0), axis=1))
        M2 = max(6, int(np.ceil(L2 / max(target_step2, 1e-6))) + 1)
        u2 = (np.arange(M2 - 1) + 0.5) / (M2 - 1)
        ramp_end = 0.25
        rr = np.clip(u2 / ramp_end, 0.0, 1.0)
        smooth2 = 3 * rr ** 2 - 2 * rr ** 3
        v2 = eps_step + (target_step2 - eps_step) * smooth2

        stage2 = resample_by_speed_profile(stage2_raw, v2)  # (T2, 3)

        # ============ (3) 拼接两个阶段 ============
        traj = np.vstack([stage1, stage2[1:]])  # 去掉重复的 subgoal

        # 小噪声（和 env2d 一致）
        if hasattr(self, "noise_std") and self.noise_std > 0.0:
            traj = traj + np.random.randn(*traj.shape) * (self.noise_std * 0.5)

        return traj, int(true_tau)

    def generate_demos(
            self,
            n_demos=12,
            n1=20,
            speed_ratio=0.5,
            speed_noise_std=0.0,
            **kwargs,
    ):
        """
        生成多条两阶段 demo：

          - 阶段1：正弦 corridor → subgoal
          - 阶段2：subgoal 直线匀速 → goal

        为了兼容 main_3d_auto 等上层代码，额外的 kwargs 会被忽略
        （例如 direction 等）。
        """
        demos = []
        true_taus = []
        for _ in range(n_demos):
            X, tau_true = self.rollout_demo(
                n1=n1,
                speed_ratio=speed_ratio,
                speed_noise_std=speed_noise_std,
            )
            demos.append(X)
            true_taus.append(int(tau_true))
        return demos, true_taus

    # ==========================================================
    # Features
    # ==========================================================
    def compute_all_features_matrix(self, X):
        """
        返回 shape = (T, M_raw) 的特征矩阵。

        设计:
          feat0: dist_to_centerline = |y - y_c(x)|           (distance-like)
          feat1: speed = ||X[t+1]-X[t]|| / dt               (用于速度约束)
          feat2: z 坐标
          feat3: noise / dummy feature (高斯噪声)

        这些 feature 都会在 learner 里经过 global 标准化 (z-score)。
        """
        X = np.asarray(X, float)
        T = len(X)
        assert X.shape[1] == 3, "SineCorridorEnv3D expects X shape (T,3)"

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        # 到中心线的距离（只看 xy 平面，与 z 无关）
        y_c = self.centerline(x)
        dist_center = np.abs(y - y_c)

        # 速度 (欧氏距离 / dt)
        V = np.zeros(T, dtype=float)
        if T > 1:
            dX = X[1:] - X[:-1]
            V[:-1] = np.linalg.norm(dX, axis=1) / self.dt
            V[-1] = V[-2]  # 最后一帧复用前一帧速度

        # z 坐标本身
        # z_feat = z.copy()
        far_center_xy=np.array((-0.5, 2.5))
        d_far = np.linalg.norm(X[:, :2] - far_center_xy[None, :], axis=1) * 0.2 # 0.2 only for better plotting

        # 随机噪声特征（用来测试 feature selection）
        noise_feat = np.random.randn(T) * 0.1

        F = np.stack([dist_center, V, d_far, noise_feat], axis=1)  # (T,4)
        return F

    def compute_features_all(self, X):
        F = self.compute_all_features_matrix(X)
        return F[:, 0], F[:, 1]

    # ==========================================================
    # Oracle constraints (仅用于 evaluation，不参与训练)
    # ==========================================================
    def estimate_oracle_constraints(self, demos, true_taus, q_speed=95.0):
        """
        基于一批 demos + true_taus，收集：
          - 阶段1的所有轨迹段 stage1_X_list（用于等式约束误差）
          - 阶段1/2 的速度上限（q_speed 分位数）v1_max, v2_max  （raw 空间）

        存到:
          self.true_constraints = {
              "stage1_X_list": [X1_stage1, X2_stage1, ...],  # 每个 (T1_i, 3)
              "v1_max": float,   # 阶段1速度上限（95% 分位）
              "v2_max": float,   # 阶段2速度上限（95% 分位）
          }
        """
        stage1_X_list = []
        all_v1 = []
        all_v2 = []

        for X, tau_true in zip(demos, true_taus):
            if tau_true is None:
                continue

            X = np.asarray(X, float)
            T = len(X)
            if T <= 1:
                continue

            tau_true = int(np.clip(tau_true, 0, T - 1))

            # ---- 阶段1轨迹 ----
            X_stage1 = X[: tau_true + 1].copy()
            if X_stage1.shape[0] > 0:
                stage1_X_list.append(X_stage1)

            # ---- 速度（raw 空间）----
            dX = X[1:] - X[:-1]  # (T-1, 3)
            speeds_edge = np.linalg.norm(dX, axis=1) / self.dt  # (T-1,)

            # pad 到长度 T（和你其他 env 对齐，虽然这里只要分段就行）
            speeds = np.empty(T, dtype=float)
            speeds[0] = speeds_edge[0]
            speeds[1:] = speeds_edge

            # 阶段1/2 速度切段
            v1_seg = speeds[: tau_true + 1]
            v2_seg = speeds[tau_true + 1:]

            if v1_seg.size > 0:
                all_v1.append(v1_seg)
            if v2_seg.size > 0:
                all_v2.append(v2_seg)

        v1_max = np.nan
        v2_max = np.nan
        if len(all_v1) > 0:
            v1_all = np.concatenate(all_v1, axis=0)
            v1_max = float(np.percentile(v1_all, q_speed))
        if len(all_v2) > 0:
            v2_all = np.concatenate(all_v2, axis=0)
            v2_max = float(np.percentile(v2_all, q_speed))

        self.true_constraints = {
            "stage1_X_list": stage1_X_list,
            "v1_max": v1_max,
            "v2_max": v2_max,
        }


def load_3d_sine_corridor(
    n_demos: int = 12,
    seed: int = 0,
    env_kwargs=None,
    demo_kwargs=None,
) -> TaskBundle:
    env_cfg = {
        "A": 0.1,
        "omega": 10.0,
        "bias": 0.2,
        "phase": 0.0,
        "x_start_range": (-2.0, -1.0),
        "z_start_range": (0.0, 0.5),
        "x_sub": 0.0,
        "z_sub": 0.6,
        "goal": (0.6, 0.0, 0.0),
        "dt": 1.0,
    }
    if env_kwargs:
        env_cfg.update(env_kwargs)

    run_kwargs = {
        "n_demos": n_demos,
        "T_stage1": 120,
        "noise_y_std": 0.0,
        "noise_z_std": 0.0,
        "v2_max": None,
    }
    if demo_kwargs:
        run_kwargs.update(demo_kwargs)

    import random

    np.random.seed(seed)
    random.seed(seed)
    env = SineCorridorEnv3D(**env_cfg)
    demos, true_taus = env.generate_demos(**run_kwargs)
    env.estimate_oracle_constraints(demos, true_taus)
    return TaskBundle(
        name="3DSineCorridor",
        demos=demos,
        env=env,
        true_taus=[int(t) for t in true_taus],
        meta={"seed": seed, "task_name": "3DSineCorridor"},
    )


def main():
    env = SineCorridorEnv3D()
    demos, true_taus = env.generate_demos(n_demos=5, noise_y_std=0.05, noise_z_std=0.05)
    for i, X in enumerate(demos):
        print(f"Demo {i}: shape = {X.shape}, true_tau = {true_taus[i]}")
        F = env.compute_all_features_matrix(X)
        print(f"  Features shape: {F.shape}")
        print(f"  First 5 feature rows:\n{F[:5]}")

if __name__ == "__main__":
    main()
