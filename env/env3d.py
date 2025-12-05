# env/env3d.py
# ------------------------------------------------------------
# 3D wrapper environment — produces 3D trajectories by:
#   (1) generating 2D XY trajectories using ObsAvoidEnv
#   (2) interpolating Z: start_z → subgoal_z → goal_z
# Features:
#   f1: XY obstacle distance
#   f2: 3D speed magnitude
# ------------------------------------------------------------

import numpy as np
from env.env2d import ObsAvoidEnv


class ObsAvoidEnv3D:
    """
    3D 环境（严格遵循你的设计）：

    - XY 与 2D env 完全一致（同样绕障碍物）
    - Z 分两段插值：
         0 ~ tau      : start_z → subgoal_z
         tau ~ T-1    : subgoal_z → goal_z
    - 特征：
         f1: dist_xy(xy, obstacle_center)
         f2: ||x[t+1] - x[t]|| / dt  （3D）
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
        self.env2d = ObsAvoidEnv(
            start=start_xy,
            subgoal=subgoal_xy,
            goal=goal_xy,
            obs_center=obs_center_xy,
            obs_radius=obs_radius,
            dt=dt,
            noise_std=noise_std
        )

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
    # Feature extraction
    # ------------------------------------------------------------------
    def compute_features(self, traj3d, tau):
        """
        Compute features for stage1 and stage2, consistent with learner:

            f1 = distance(XY, obstacle_center)
            f2 = 3D speed magnitude

        Return:
            feats1: f1[:tau+1]
            feats2: f2[tau:]
        """
        xy = traj3d[:, :2]
        dists_xy = np.linalg.norm(xy - self.obs_center_xy[None, :], axis=1)

        speeds_3d = np.linalg.norm(np.diff(traj3d, axis=0), axis=1) / self.dt
        feats1 = dists_xy[: tau + 1]
        feats2 = speeds_3d[tau:]

        return feats1, feats2

    def compute_features_all(self, traj3d):
        """
        Return full-series features:
            dists_xy: len T
            speeds_3d: len T-1
        """
        xy = traj3d[:, :2]
        dists = np.linalg.norm(xy - self.obs_center_xy[None, :], axis=1)
        speeds = np.linalg.norm(np.diff(traj3d, axis=0), axis=1) / self.dt
        return dists, speeds
