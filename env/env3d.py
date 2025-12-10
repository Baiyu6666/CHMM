# env/env3d.py
# ------------------------------------------------------------
# 3D wrapper environment — produces 3D trajectories by:
#   (1) generating 2D XY trajectories using ObsAvoidEnv
#   (2) interpolating Z: start_z → subgoal_z → goal_z
#
# Full features:
#   f0: XY distance to main obstacle
#   f1: 3D speed magnitude
#   f2: XY distance to far cylinder center
#   f3: deterministic noise-like feature (x,y,z)
# ------------------------------------------------------------

import numpy as np
from env.env2d import ObsAvoidEnv


class ObsAvoidEnv3D:
    """
    3D 环境：

    - XY 与 2D env 完全一致（同样绕障碍物）
    - Z 分两段插值：
         0 ~ tau      : start_z → subgoal_z
         tau ~ T-1    : subgoal_z → goal_z

    特征（full set）：
        f0: dist_xy(xy, obstacle_center)
        f1: 3D speed magnitude
        f2: dist_xy(xy, far_center_xy)
        f3: deterministic noise-like feature in (x,y,z)
    """

    def __init__(
        self,
        start_xy=(-1.5, 0.0),
        subgoal_xy=(0.5, 0.0),
        goal_xy=(0.0, 0.3),
        obs_center_xy=(-0.5, 0.0),
        obs_radius=0.3,

        # Z 维度设置
        start_z_range=(0.2, 0.6),
        subgoal_z=0.4,
        goal_z=0.6,

        # base env2d configs
        dt=1.0,
        noise_std=0.01
    ):

        # Store parameters
        self.start_xy = np.array(start_xy, float)
        self.subgoal_xy = np.array(subgoal_xy, float)
        self.goal_xy = np.array(goal_xy, float)

        self.obs_center_xy = np.array(obs_center_xy, float)
        self.obs_radius = float(obs_radius)

        self.start_z_range = tuple(start_z_range)
        self.subgoal_z = float(subgoal_z)
        self.goal_z = float(goal_z)

        self.dt = float(dt)
        self.noise_std = float(noise_std)

        # --- create 3D goal/subgoal for plotting ---
        self.subgoal = np.array([self.subgoal_xy[0], self.subgoal_xy[1], self.subgoal_z], float)
        self.goal = np.array([self.goal_xy[0], self.goal_xy[1], self.goal_z], float)

        # Underlying 2D env (same logic you used)
        # 注意：2D env 里也有一个 far_center，用于 2D feature；
        # 但 3D feature 我们自己重算一遍，以 XY 为主。
        self.env2d = ObsAvoidEnv(
            start=start_xy,
            subgoal=subgoal_xy,
            goal=goal_xy,
            obs_center=obs_center_xy,
            obs_radius=obs_radius,
            dt=dt,
            noise_std=noise_std,
        )

        # 噪声特征用的 3D 向量
        self.noise_vec3 = np.array([0.31, -0.47, 0.62], float)
        self.noise_bias3 = 0.0

    # ------------------------------------------------------------------
    # 3D demo generation
    # ------------------------------------------------------------------
    def generate_demo_3d(self, **kwargs):
        """
        生成一条 3D 轨迹：
            - XY 由 env2d.generate_demo() 得到
            - Z 由插值生成

        返回:
            traj3d: shape (T,3)
            tau: int
        """
        # --- XY from 2D version ---
        traj2d, tau = self.env2d.generate_demo(**kwargs)
        T = len(traj2d)

        # --- Z interpolation ---
        start_z = np.random.uniform(*self.start_z_range)

        # Z for stage1: start_z -> subgoal_z
        z1 = np.linspace(start_z, self.subgoal_z, tau + 1)

        # Z for stage2: subgoal_z -> goal_z
        z2 = np.linspace(self.subgoal_z, self.goal_z, T - tau)

        z_full = np.concatenate([z1, z2[1:]], axis=0)
        assert len(z_full) == T

        # --- Combine into 3D ---
        traj3d = np.column_stack([
            traj2d[:, 0],
            traj2d[:, 1],
            z_full
        ])

        return traj3d, tau

    # ------------------------------------------------------------------
    # New unified multi-feature API
    # ------------------------------------------------------------------
    def compute_all_features_matrix(self, traj3d, feat_ids=None):
        """
        返回 full feature matrix F: shape = (T,4)
            f0: XY distance to main obstacle
            f1: 3D speed magnitude
            f2: XY distance to far cylinder
            f3: deterministic noise-like feature

        feat_ids: None or 列索引列表
        """
        traj3d = np.asarray(traj3d, float)
        xy = traj3d[:, :2]
        T = traj3d.shape[0]

        # f0: XY distance to main obstacle
        d_main = np.linalg.norm(xy - self.obs_center_xy[None, :], axis=1)

        # f1: 3D speed magnitude, padded to length T
        speeds_edge = np.linalg.norm(np.diff(traj3d, axis=0), axis=1) / self.dt
        speeds = np.empty(T, dtype=float)
        if T > 1:
            speeds[0] = speeds_edge[0]
            speeds[1:] = speeds_edge
        else:
            speeds[0] = 0.0

        # f2: XY distance to far cylinder
        far_center_xy=np.array((-0.5, 2.5))
        d_far = np.linalg.norm(xy - far_center_xy[None, :], axis=1)

        # f3: noise-like feature
        t = np.linspace(0, 2 * np.pi, T)
        phase = np.random.uniform(0, 2 * np.pi)
        noise_feat = 0.2 * np.sin(5 * t + phase)

        F = np.stack([d_main, speeds, d_far, noise_feat], axis=1)  # (T,4)

        if feat_ids is None:
            return F
        else:
            return F[:, feat_ids]

    # --------------------------------------------------------
    # Simple Oracle Constraints (3D, final version)
    # --------------------------------------------------------
    def estimate_oracle_constraints(
            self,
            demos,
            true_taus,
            clearance=0.10,
            speed_quantile=0.95,
    ):
        """
        Oracle 定义（最终版）：

          Stage 1:
            - XY obstacle distance: d >= obs_radius + clearance
            - 3D speed upper bound: v <= v1_max

          Stage 2:
            - 3D speed upper bound: v <= v2_max
        """

        # ---------- (1) obstacle distance ----------
        d_safe_raw = float(self.obs_radius + clearance)

        # ---------- (2) speed bounds ----------
        v_stage1_all = []
        v_stage2_all = []

        for X, tau_true in zip(demos, true_taus):
            X = np.asarray(X, float)
            T = len(X)

            tau = int(np.clip(int(tau_true), 0, T - 1))

            # 3D speed
            if T > 1:
                speeds_edge = np.linalg.norm(np.diff(X, axis=0), axis=1) / self.dt
                speeds = np.empty(T, dtype=float)
                speeds[0] = speeds_edge[0]
                speeds[1:] = speeds_edge
            else:
                speeds = np.zeros(1)

            v_stage1_all.append(speeds[: tau + 1])
            v_stage2_all.append(speeds[tau:])

        v_stage1_all = np.concatenate(v_stage1_all).astype(float)
        v_stage2_all = np.concatenate(v_stage2_all).astype(float)

        v1_max_raw = float(
            np.quantile(v_stage1_all, speed_quantile)
        )
        v2_max_raw = float(
            np.quantile(v_stage2_all, speed_quantile)
        )

        self.true_constraints = {
            "d_safe": d_safe_raw,
            "v1_max": v1_max_raw,
            "v2_max": v2_max_raw,
            "speed_quantile": speed_quantile,
        }
        print('True constraints :', self.true_constraints)
