# env/sine_corridor_env3d.py
import numpy as np


class SineCorridorEnv3D:
    """
    3D 版 Sine Corridor 环境，用于和 GoalHMM3D 对接。

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
        A=0.5,
        omega=1.0,
        x_start_range=(-2.0, -1.0),
        z_start_range=(-0.5, 0.5),
        x_sub=0.0,
        z_sub=0.0,
        corridor_half_width=0.2,
        dt=0.02,
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
        self.x_start_range = tuple(x_start_range)
        self.z_start_range = tuple(z_start_range)
        self.dt = float(dt)

        # ---- subgoal / goal ----
        y_sub = self.centerline(x_sub)
        self.subgoal = np.array([x_sub, y_sub, z_sub], dtype=float)

        # 目前先让 goal = subgoal，后面你要加第二阶段时再改成别的点即可
        self.goal = self.subgoal.copy()

        # ---- 为了兼容 3D 绘图中的“障碍柱”接口 ----
        # 这里随便放一个 cylinder 在原点附近，半径 = corridor_half_width
        self.obs_center_xy = np.array([0.0, 0.0], dtype=float)
        self.obs_radius = float(corridor_half_width)

    # ==========================================================
    # 几何 & 中心线
    # ==========================================================
    def centerline(self, x):
        """正弦中心线 y_c(x) = A sin(ω x)。x 可以是标量或 ndarray。"""
        return self.A * np.sin(self.omega * x)

    # ==========================================================
    # Demo 生成
    # ==========================================================
    def rollout_demo(
        self,
        T_stage1=120,
        noise_y_std=0.0,
        noise_z_std=0.0,
    ):
        """
        生成一条从随机起点到 subgoal 的单阶段 demo。

        起点:
          x0 ~ Uniform(x_start_range)
          y0 = centerline(x0)          (保证在 manifold 上)
          z0 ~ Uniform(z_start_range)

        轨迹构造:
          设 t = 0..T_stage1:
            alpha = t / T_stage1 ∈ [0,1]
            x_t = (1-alpha)*x0 + alpha*x_sub
            y_t = centerline(x_t) + N(0, noise_y_std^2)
            z_t = (1-alpha)*z0 + alpha*z_sub + N(0, noise_z_std^2)

        返回:
          X: shape = (T_stage1+1, 3)
        """
        # ---- 起点在 manifold 上 ----
        x0 = np.random.uniform(*self.x_start_range)
        z0 = np.random.uniform(*self.z_start_range)
        y0 = self.centerline(x0)
        p0 = np.array([x0, y0, z0], dtype=float)

        X = [p0]

        x_sub, y_sub, z_sub = self.subgoal
        T = int(T_stage1)

        for t in range(1, T + 1):
            alpha = t / T  # 线性插值系数 in [0,1]

            # x: 从 x0 -> x_sub 线性插值
            x_t = (1.0 - alpha) * x0 + alpha * x_sub

            # y: 始终贴在中心线上（可加一点噪声）
            y_center = self.centerline(x_t)
            if noise_y_std > 0.0:
                y_t = y_center + np.random.randn() * noise_y_std
            else:
                y_t = y_center

            # z: 起点 z0 -> z_sub 线性插值（可加一点噪声）
            z_t = (1.0 - alpha) * z0 + alpha * z_sub
            if noise_z_std > 0.0:
                z_t = z_t + np.random.randn() * noise_z_std

            X.append(np.array([x_t, y_t, z_t], dtype=float))

        return np.stack(X, axis=0)  # (T_stage1+1, 3)

    def generate_demos(
        self,
        n_demos=12,
        T_stage1=120,
        noise_y_std=0.0,
        noise_z_std=0.0,
    ):
        """
        生成多条 demo，并返回 (demos, true_taus)。

        目前只有一阶段（从起点到 subgoal），
        因此 true_tau 统一设为 T_stage1 （最后一个切点之前的 index）。

        返回:
          demos    : list of ndarray(T_i, 3)
          true_taus: list[int]
        """
        demos = []
        true_taus = []
        for _ in range(n_demos):
            X = self.rollout_demo(
                T_stage1=T_stage1,
                noise_y_std=noise_y_std,
                noise_z_std=noise_z_std,
            )
            demos.append(X)
            # 切点：认为整条轨迹都属于 stage1，tau_true=T-1
            # 你后面如果加 stage2，这里再改。
            tau_true = len(X) - 1
            true_taus.append(tau_true)

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
        z_feat = z.copy()

        # 随机噪声特征（用来测试 feature selection）
        noise_feat = np.random.randn(T) * 0.1

        F = np.stack([dist_center, V, z_feat, noise_feat], axis=1)  # (T,4)
        return F

    def compute_features_all(self, X):
        """
        兼容旧接口：返回 (distance_like, speed)

        在本环境中：
          distance_like = dist_to_centerline
          speed         = speed
        """
        F = self.compute_all_features_matrix(X)
        dist_center = F[:, 0]
        speed = F[:, 1]
        return dist_center, speed


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