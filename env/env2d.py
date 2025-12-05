# env/env2d.py
# ------------------------------------------------------------
# 2D obstacle-avoid + speed-constraint demo generator.
# This file keeps the original 2D behavior you used before.
# ------------------------------------------------------------

import numpy as np


class ObsAvoidEnv:
    """
    2D 环境（保留你原来的 2D demo 生成逻辑）：
      - stage1: 绕圆柱障碍（XY平面）
      - stage2: 速度约束（后半段速度更慢）
    特征：
      f1 = distance to obstacle center (2D)
      f2 = speed magnitude (2D)
    """

    def __init__(
        self,
        start=(-1.5, 0.0),
        subgoal=(0.5, 0.0),
        goal=(0.6, 0.6),
        obs_center=(-0.5, 0.0),
        obs_radius=0.3,
        dt=1.0,
        noise_std=0.01,
    ):
        self.start = np.array(start, dtype=float)
        self.subgoal = np.array(subgoal, dtype=float)
        self.goal = np.array(goal, dtype=float)

        self.obs_center = np.array(obs_center, dtype=float)
        self.obs_radius = float(obs_radius)

        self.dt = float(dt)
        self.noise_std = float(noise_std)

    # ----------------------------
    # Demo generation (2D)
    # ----------------------------
    def generate_demo(
        self,
        n1=20,
        direction=None,
        speed_ratio=0.5,
        jitter_frac=0.05,
        arc_early_frac_range=(0.90, 1.00),
        speed_noise_std=0.05,
        goal_jitter_frac=0.05,
    ):
        """
        生成一条 2D demo，并返回 (traj, true_tau)

        保留你原来风格：
          - 段1：直线 -> 绕障碍圆弧（上绕/下绕） -> 可能提前离开圆弧 -> 直线到 subgoal
          - 段2：subgoal -> goal 的轻微摆动直线
          - 速度：段1后半段减速，段2整体慢于段1（speed_ratio）
          - 每条 demo 的 subgoal 和 goal 都有轻微扰动
        """
        if direction is None:
            direction = np.random.choice(["up", "down"])

        # =====(0) trajectory-local goal jitter =====
        seg_len_glob = np.linalg.norm(self.goal - self.subgoal) + 1e-12
        goal_eps = np.random.randn(2) * (goal_jitter_frac * seg_len_glob)
        goal_local = self.goal + goal_eps

        # =====(1) trajectory-local subgoal jitter =====
        subgoal_eps = np.random.randn(2) * (jitter_frac * seg_len_glob)
        subgoal_local = self.subgoal + subgoal_eps

        # ---- stage1: straight -> arc around obstacle -> straight to subgoal ----
        straight1 = np.linspace(self.start, [-1.0, 0.0], 5)

        clearance = 0.1
        R_path = self.obs_radius + clearance

        if direction == "up":
            theta1 = np.linspace(np.pi, 0.0, n1)
        else:
            theta1 = np.linspace(-np.pi, 0.0, n1)

        arc_full = self.obs_center + R_path * np.c_[np.cos(theta1), np.sin(theta1)]
        arc_full[0] = straight1[-1]

        # early exit from arc (random cut)
        early_frac = float(np.random.uniform(*arc_early_frac_range))
        cut_idx = max(1, int(np.round(early_frac * (len(arc_full) - 1))))
        exit_point = arc_full[cut_idx]
        arc = arc_full[: cut_idx + 1]

        stage1_pre = np.vstack([straight1, arc[1:]])

        # connect exit -> subgoal_local with step size similar to arc
        arc_steps = np.linalg.norm(np.diff(arc, axis=0), axis=1)
        mean_arc_step = max(1e-6, np.mean(arc_steps))
        seg_exit2sub = np.linalg.norm(subgoal_local - exit_point)
        n_exit_pts = max(3, int(np.ceil(seg_exit2sub / mean_arc_step)) + 1)

        line_to_subgoal = np.linspace(exit_point, subgoal_local, n_exit_pts)
        stage1_raw = np.vstack([stage1_pre, line_to_subgoal[1:]])

        # ---- stage2: wobble line to goal_local ----
        def resample_by_speed_profile(points, step_weights):
            pts = np.asarray(points, float)
            if len(pts) < 2:
                return pts

            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            s = np.concatenate([[0.0], np.cumsum(seg)])
            L = s[-1]

            w = np.asarray(step_weights, float)
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

        base2 = np.linspace(subgoal_local, goal_local, 200)
        d2 = goal_local - subgoal_local
        L2_geom = np.linalg.norm(d2) + 1e-12
        t_hat = d2 / L2_geom
        n_hat = np.array([-t_hat[1], t_hat[0]])

        u = np.linspace(0.0, 1.0, len(base2))
        k = np.random.choice([1, 2])
        phi = np.random.uniform(0, 2 * np.pi)
        A = 0.02 * L2_geom + 0.04 * L2_geom * np.random.rand()
        window = 0.5 * (1 - np.cos(2 * np.pi * u))
        offset_mag = A * np.sin(2 * np.pi * k * u + phi) * window

        stage2_raw = base2 + (offset_mag[:, None] * n_hat[None, :])

        # =====(2) speed profiles + resample =====
        base_step1 = np.mean(np.linalg.norm(np.diff(stage1_raw, axis=0), axis=1))
        base_step1 = max(base_step1, 1e-6)
        target_step2 = speed_ratio * base_step1

        # stage1 speed: plateau then slow down
        M1 = max(6, len(stage1_raw) - 1)
        u1 = (np.arange(M1) + 0.5) / M1
        eps_step = 0.05 * base_step1
        plateau_frac = 0.40
        r = np.clip((u1 - plateau_frac) / max(1e-6, (1.0 - plateau_frac)), 0.0, 1.0)
        smooth = 3 * r**2 - 2 * r**3
        v1 = (1.0 - smooth) * base_step1 + smooth * eps_step

        stage1 = resample_by_speed_profile(stage1_raw, v1)
        true_tau = len(stage1) - 1

        # stage2 speed: ramp up to target_step2 then constant
        L2 = np.sum(np.linalg.norm(np.diff(stage2_raw, axis=0), axis=1))
        M2 = max(6, int(np.ceil(L2 / max(target_step2, 1e-6))) + 1)
        u2 = (np.arange(M2 - 1) + 0.5) / (M2 - 1)
        ramp_end = 0.25
        rr = np.clip(u2 / ramp_end, 0.0, 1.0)
        smooth2 = 3 * rr**2 - 2 * rr**3
        v2 = eps_step + (target_step2 - eps_step) * smooth2

        stage2 = resample_by_speed_profile(stage2_raw, v2)

        # =====(3) stitch =====
        traj = np.vstack([stage1, stage2[1:]])

        # small measurement noise
        traj = traj + np.random.randn(*traj.shape) * (self.noise_std * 0.5)

        return traj, true_tau

    # ----------------------------
    # Features
    # ----------------------------
    def compute_features(self, traj, tau):
        """
        traj: (T,2)
        tau: cutpoint index
        Returns:
          feats1: distance series for stage1 [0..tau]
          feats2: speed series for stage2 [tau..T-1] (speed defined on edges)
        """
        dists = np.linalg.norm(traj - self.obs_center[None, :], axis=1)
        speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.dt
        feats1 = dists[: tau + 1]
        feats2 = speeds[tau:]
        return feats1, feats2

    def compute_features_all(self, traj):
        """
        Returns:
          dists: length T
          speeds: length T-1 (edge speeds)
        """
        dists = np.linalg.norm(traj - self.obs_center[None, :], axis=1)
        speeds = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.dt
        return dists, speeds
