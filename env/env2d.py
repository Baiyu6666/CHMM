# env/env2d.py
# ------------------------------------------------------------
# 2D obstacle-avoid + speed-constraint demo generator.
# ------------------------------------------------------------

import numpy as np


class ObsAvoidEnv:
    """
    2D 环境：
      - stage1: 绕圆柱障碍（XY平面）
      - stage2: 速度约束（后半段速度更慢）

    特征（full feature set）：
      f0 = distance to main obstacle center (2D)
      f1 = speed magnitude (2D)
      f2 = distance to a far cylinder center (2D)
      f3 = deterministic "noise-like" feature (no physical meaning)
    """

    def __init__(
        self,
        start=(-1.5, 0.0),
        subgoal=(0.5, 0.0),
        goal=(0.6, 0.6),
        obs_center=(-0.5, 0.0),
        obs_radius=0.3,
        clearance=0.1,
        dt=1.0,
        noise_std=0.01,
         ):
        self.start = np.array(start, dtype=float)
        self.subgoal = np.array(subgoal, dtype=float)
        self.goal = np.array(goal, dtype=float)

        self.obs_center = np.array(obs_center, dtype=float)
        self.obs_radius = float(obs_radius)
        self.clearance = float(clearance)

        self.dt = float(dt)
        self.noise_std = float(noise_std)

        # 噪声特征的“方向向量”（固定常数，保证 deterministic）
        # 可以随便选几个不共线的数
        self.noise_vec = np.array([0.37, -0.58], dtype=float)
        self.noise_bias = 0.0  # 可选偏移，先设 0

    def get_true_constraints(self):
        return {
            "d_safe_raw": self.obs_radius + self.clearance,  # 你可以把 clearance 存成成员
            "v2_max_raw": self.v2_max_true,  # 由 speed profile 推出来或预计算
        }

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

        R_path = self.obs_radius + self.clearance

        if direction == "up":
            theta1 = np.linspace(np.pi, 0.0, n1)
        else:
            theta1 = np.linspace(-np.pi, 0.0, n1)

        arc_full = self.obs_center + R_path * np.c_[np.cos(theta1), np.sin(theta1)]
        arc_full[0] = straight1[-1]

        # early exit from arc
        early_frac = float(np.random.uniform(*arc_early_frac_range))
        cut_idx = max(1, int(np.round(early_frac * (len(arc_full) - 1))))
        exit_point = arc_full[cut_idx]
        arc = arc_full[: cut_idx + 1]

        stage1_pre = np.vstack([straight1, arc[1:]])

        # connect exit -> subgoal_local
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
    # New unified multi-feature API
    # ----------------------------
    def compute_all_features_matrix(self, traj, feat_ids=None):
        """
        返回 full feature matrix F: shape = (T, 4)
            f0 = distance to main obstacle center
            f1 = 2D speed magnitude (padded to length T)
            f2 = distance to far cylinder center
            f3 = deterministic noise-like feature

        feat_ids: None or 列索引列表，例如 [0,2,3]
        """
        traj = np.asarray(traj, float)
        T = traj.shape[0]

        # f0: 主障碍距离
        d_main = np.linalg.norm(traj - self.obs_center[None, :], axis=1)

        # f1: 2D 速度模长，pad 到长度 T
        speeds_edge = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.dt
        speeds = np.empty(T, dtype=float)
        if T > 1:
            speeds[0] = speeds_edge[0]
            speeds[1:] = speeds_edge
        else:
            speeds[0] = 0.0

        # f2: 距离远处圆柱
        far_center = np.array((-0.5, 2.5), dtype=float)
        d_far = np.linalg.norm(traj - far_center[None, :], axis=1)

        # f3: "噪声特征"：对 (x,y) 做一个固定的线性投影再加一点非线性
        t = np.linspace(0, 2 * np.pi, T)
        phase = np.random.uniform(0, 2 * np.pi)
        noise_feat = 0.2 * np.sin(5 * t + phase)

        F = np.stack([d_main, speeds, d_far, noise_feat], axis=1)  # (T,4)

        if feat_ids is None:
            return F
        else:
            return F[:, feat_ids]

    def estimate_oracle_constraints(
            self,
            demos,
            true_taus,
            speed_quantile=0.95,
    ):
        """
        Oracle 定义（最终版）：

          Stage 1:
            - obstacle distance: d >= obs_radius + clearance
            - speed upper bound: v <= v1_max   (95% quantile)

          Stage 2:
            - speed upper bound: v <= v2_max   (95% quantile)
        """

        # ---------- (1) obstacle distance (stage1) ----------
        d_safe_raw = float(self.obs_radius + self.clearance)

        # ---------- (2) speed bounds ----------
        v_stage1_all = []
        v_stage2_all = []

        for X, tau_true in zip(demos, true_taus):
            X = np.asarray(X, float)
            T = len(X)

            tau = int(np.clip(int(tau_true), 0, T - 1))

            # 2D speed
            if T > 1:
                speeds_edge = np.linalg.norm(np.diff(X, axis=0), axis=1) / self.dt
                speeds = np.empty(T, dtype=float)
                speeds[0] = speeds_edge[0]
                speeds[1:] = speeds_edge
            else:
                speeds = np.zeros(1)

            # stage1: [0 .. tau]
            v_stage1_all.append(speeds[: tau + 1])

            # stage2: [tau .. T-1]
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
        }
        print('True constraints :', self.true_constraints)
