import numpy as np

from .base import TaskBundle
from planner import optimize_trajectory, repair_trajectory_constraints, resample_polyline


class ObsAvoidEnv:
    """
    2D obstacle-avoidance environment with explicit oracle constraints.

    Stage 1:
      - follow a clearance-respecting path around the obstacle
      - speed is bounded by ``stage1_speed_max``

    Stage 2:
      - move from subgoal to goal
      - speed is bounded by ``stage2_speed_max``
    """

    def __init__(
        self,
        start=(-1.5, 0.0),
        subgoal=(0.5, 0.0),
        goal=(0., 0.3),
        obs_center=(-0.5, 0.0),
        obs_radius=0.3,
        clearance=0.1,
        stage1_speed_max=0.15,
        stage2_speed_max=0.06,
        stage1_accel_max=0.06,
        stage2_accel_max=0.04,
        dt=1.0,
        noise_std=0.01,
    ):
        self.start = np.array(start, dtype=float)
        self.subgoal = np.array(subgoal, dtype=float)
        self.goal = np.array(goal, dtype=float)

        self.obs_center = np.array(obs_center, dtype=float)
        self.obs_radius = float(obs_radius)
        self.clearance = float(clearance)
        self.stage1_speed_max = float(stage1_speed_max)
        self.stage2_speed_max = float(stage2_speed_max)
        self.stage1_accel_max = float(stage1_accel_max)
        self.stage2_accel_max = float(stage2_accel_max)

        self.dt = float(dt)
        self.noise_std = float(noise_std)

        self.noise_vec = np.array([0.37, -0.58], dtype=float)
        self.noise_bias = 0.0
        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self._direct_true_constraints()
        self.constraint_specs = self.get_constraint_specs()

    def _direct_true_constraints(self):
        return {
            "d_safe": float(self.obs_radius + self.clearance),
            "v1_max": float(self.stage1_speed_max),
            "v2_max": float(self.stage2_speed_max),
            "a1_max": float(self.stage1_accel_max),
            "a2_max": float(self.stage2_accel_max),
        }

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "dist_main_obstacle", "description": "Distance to main obstacle center"},
            {"id": 1, "name": "speed", "description": "2D speed magnitude"},
            {"id": 2, "name": "dist_far_obstacle", "description": "Distance to far obstacle center"},
            {"id": 3, "name": "noise_aux", "description": "Deterministic auxiliary noise-like feature"},
        ]

    def get_true_constraints(self):
        return dict(self.true_constraints)

    def get_constraint_specs(self):
        return [
            {
                "metric": "AbsErr_d",
                "feature_name": "dist_main_obstacle",
                "stage": 0,
                "semantics": "lower_bound",
                "oracle_key": "d_safe",
            },
            {
                "metric": "AbsErr_v",
                "feature_name": "speed",
                "stage": 1,
                "semantics": "upper_bound",
                "oracle_key": "v2_max",
            },
        ]

    def _arc_points(self, theta_start: float, theta_end: float, direction: str, radius: float, n_arc: int):
        if direction == "up":
            delta = (theta_end - theta_start) % (2.0 * np.pi)
            if delta <= 1e-8:
                delta += 2.0 * np.pi
        else:
            delta = -((theta_start - theta_end) % (2.0 * np.pi))
            if abs(delta) <= 1e-8:
                delta -= 2.0 * np.pi
        thetas = theta_start + np.linspace(0.0, delta, max(int(n_arc), 8))
        return self.obs_center[None, :] + radius * np.c_[np.cos(thetas), np.sin(thetas)]

    def _smooth_process_noise(self, length: int, dim: int, scale: float, knots: int, rng):
        if int(length) <= 2 or float(scale) <= 0.0:
            return np.zeros((int(length), int(dim)), dtype=float)
        num_knots = max(int(knots), 2)
        knot_t = np.linspace(0.0, 1.0, num_knots)
        knot_vals = rng.randn(num_knots, int(dim)) * float(scale)
        knot_vals[0] = 0.0
        knot_vals[-1] = 0.0
        t = np.linspace(0.0, 1.0, int(length))
        noise = np.empty((int(length), int(dim)), dtype=float)
        for d in range(int(dim)):
            noise[:, d] = np.interp(t, knot_t, knot_vals[:, d])
        return noise

    def generate_demo(
        self,
        n1=24,
        direction=None,
        start_x_range=(-2.5, -0.5),
        start_y_range=(-0.5, 0.5),
        start_safe_scale=1.08,
        start_resample_tries=64,
        start_angle_jitter_deg=22.0,
        start_radius_jitter_frac=0.18,
        start_tangent_jitter_frac=0.18,
        subgoal_jitter_frac=0.1,
        goal_jitter_frac=0.08,
        lateral_stage2_scale=0.08,
        transition_steps=1,
        transition_jitter=2,
        transition_lateral_scale=0.10,
        process_noise_scale=0.035,
        process_noise_knots=6,
        stage1_speed_scale=1.0,
        stage2_speed_scale=1.0,
        rng=None,
    ):
        """Generate one 2D demonstration and return ``(traj, true_tau)``."""
        rng = np.random if rng is None else rng
        if direction is None:
            direction = rng.choice(["up", "down"])

        d_safe = float(self.true_constraints["d_safe"])
        seg_len = np.linalg.norm(self.goal - self.subgoal) + 1e-12
        start_local = None
        if start_x_range is not None and start_y_range is not None:
            x_lo, x_hi = sorted([float(start_x_range[0]), float(start_x_range[1])])
            y_lo, y_hi = sorted([float(start_y_range[0]), float(start_y_range[1])])
            min_start_dist = d_safe * max(float(start_safe_scale), 1.0)
            for _ in range(max(int(start_resample_tries), 1)):
                candidate = np.array(
                    [rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)],
                    dtype=float,
                )
                if np.linalg.norm(candidate - self.obs_center) >= min_start_dist:
                    start_local = candidate
                    break
            if start_local is None:
                corners = np.array(
                    [
                        [x_lo, y_lo],
                        [x_lo, y_hi],
                        [x_hi, y_lo],
                        [x_hi, y_hi],
                    ],
                    dtype=float,
                )
                distances = np.linalg.norm(corners - self.obs_center[None, :], axis=1)
                start_local = corners[int(np.argmax(distances))]

        if start_local is None:
            start_rel_base = self.start - self.obs_center
            start_radius_base = np.linalg.norm(start_rel_base)
            start_theta_base = np.arctan2(start_rel_base[1], start_rel_base[0])
            theta_jitter = np.deg2rad(float(start_angle_jitter_deg))
            start_theta = start_theta_base + rng.uniform(-theta_jitter, theta_jitter)
            start_radius = start_radius_base * (1.0 + rng.uniform(-start_radius_jitter_frac, start_radius_jitter_frac))
            start_radius = max(start_radius, d_safe * 1.15)
            start_dir = np.array([np.cos(start_theta), np.sin(start_theta)], dtype=float)
            tangent_dir = np.array([-start_dir[1], start_dir[0]], dtype=float)
            start_local = self.obs_center + start_radius * start_dir
            start_local = start_local + tangent_dir * (start_tangent_jitter_frac * seg_len * rng.uniform(-1.0, 1.0))
        subgoal_local = self.subgoal + rng.randn(2) * (subgoal_jitter_frac * seg_len)
        goal_local = self.goal + rng.randn(2) * (goal_jitter_frac * seg_len)

        start_rel = start_local - self.obs_center
        subgoal_rel = subgoal_local - self.obs_center
        theta_start = np.arctan2(start_rel[1], start_rel[0])
        theta_end = np.arctan2(subgoal_rel[1], subgoal_rel[0])
        start_anchor = self.obs_center + d_safe * start_rel / max(np.linalg.norm(start_rel), 1e-12)
        end_anchor = self.obs_center + d_safe * subgoal_rel / max(np.linalg.norm(subgoal_rel), 1e-12)

        arc = self._arc_points(theta_start, theta_end, direction, d_safe, n1)
        stage1_ctrl = np.vstack([start_local, start_anchor, arc[1:-1], end_anchor, subgoal_local])
        stage1_ref = resample_polyline(stage1_ctrl, self.stage1_speed_max * stage1_speed_scale * self.dt)

        def stage1_projector(path):
            out = np.asarray(path, dtype=float).copy()
            rel = out - self.obs_center[None, :]
            norms = np.linalg.norm(rel, axis=1, keepdims=True)
            mask = norms[:, 0] < d_safe
            if np.any(mask):
                rel[mask] /= np.maximum(norms[mask], 1e-12)
                out[mask] = self.obs_center[None, :] + d_safe * rel[mask]
            return out

        stage1 = optimize_trajectory(
            stage1_ref,
            dt=self.dt,
            v_max=self.stage1_speed_max * stage1_speed_scale,
            a_max=self.stage1_accel_max,
            projector=stage1_projector,
        )
        boundary_tau = int(len(stage1) - 1)
        repair_tau = int(len(stage1) - 1)

        d_goal = goal_local - subgoal_local
        d_goal_norm = np.linalg.norm(d_goal) + 1e-12
        tangent = d_goal / d_goal_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        u = np.linspace(0.0, 1.0, 48)
        amp = lateral_stage2_scale * d_goal_norm * rng.uniform(0.5, 1.0)
        wobble = amp * np.sin(np.pi * u) * np.sin(2.0 * np.pi * u + rng.uniform(0.0, 2.0 * np.pi))
        stage2_ctrl = subgoal_local[None, :] + u[:, None] * d_goal[None, :] + wobble[:, None] * normal[None, :]
        stage2_ctrl[0] = subgoal_local
        stage2_ctrl[-1] = goal_local
        stage2_ref = resample_polyline(stage2_ctrl, self.stage2_speed_max * stage2_speed_scale * self.dt)
        stage2 = optimize_trajectory(
            stage2_ref,
            dt=self.dt,
            v_max=self.stage2_speed_max * stage2_speed_scale,
            a_max=self.stage2_accel_max,
            projector=None,
        )

        transition_count = max(
            int(transition_steps) + int(rng.randint(-int(transition_jitter), int(transition_jitter) + 1)),
            0,
        )
        transition_count = min(transition_count, max(len(stage2) - 2, 0))
        if transition_count > 0:
            transition_target_idx = min(transition_count + 1, len(stage2) - 1)
            transition_target = np.asarray(stage2[transition_target_idx], dtype=float)
            transition_vec = transition_target - subgoal_local
            transition_len = np.linalg.norm(transition_vec) + 1e-12
            transition_tangent = transition_vec / transition_len
            transition_normal = np.array([-transition_tangent[1], transition_tangent[0]], dtype=float)
            u_tr = np.linspace(0.0, 1.0, transition_count + 2)
            transition_amp = transition_lateral_scale * transition_len * rng.uniform(0.4, 1.2)
            transition_wobble = transition_amp * np.sin(np.pi * u_tr) * np.sin(
                np.pi * u_tr + rng.uniform(0.0, 2.0 * np.pi)
            )
            transition_ctrl = (
                subgoal_local[None, :]
                + u_tr[:, None] * transition_vec[None, :]
                + transition_wobble[:, None] * transition_normal[None, :]
            )
            transition_ctrl[0] = subgoal_local
            transition_ctrl[-1] = transition_target
            transition_ref = resample_polyline(
                transition_ctrl,
                0.5 * (self.stage1_speed_max * stage1_speed_scale + self.stage2_speed_max * stage2_speed_scale) * self.dt,
            )
            transition = optimize_trajectory(
                transition_ref,
                dt=self.dt,
                v_max=max(self.stage1_speed_max * stage1_speed_scale, self.stage2_speed_max * stage2_speed_scale),
                a_max=max(self.stage1_accel_max, self.stage2_accel_max),
                projector=stage1_projector,
            )
            traj = np.vstack([stage1, transition[1:], stage2[transition_target_idx + 1 :]])
            repair_tau = int(len(stage1) - 1 + max((len(transition) - 1) // 2, 0))
        else:
            traj = np.vstack([stage1, stage2[1:]])

        traj = traj + self._smooth_process_noise(
            len(traj),
            traj.shape[1],
            process_noise_scale * seg_len,
            process_noise_knots,
            rng,
        )
        if self.noise_std > 0.0:
            traj = traj + rng.randn(*traj.shape) * (self.noise_std * 0.35)
            stage1_noisy = repair_trajectory_constraints(
                traj[: repair_tau + 1],
                dt=self.dt,
                v_max=self.stage1_speed_max * stage1_speed_scale,
                a_max=self.stage1_accel_max,
                projector=stage1_projector,
            )
            stage2_noisy = repair_trajectory_constraints(
                traj[repair_tau:],
                dt=self.dt,
                v_max=self.stage2_speed_max * stage2_speed_scale,
                a_max=self.stage2_accel_max,
                projector=None,
            )
            traj = np.vstack([stage1_noisy, stage2_noisy[1:]])

        return traj, boundary_tau

    def compute_all_features_matrix(self, traj, feat_ids=None):
        traj = np.asarray(traj, float)
        T = traj.shape[0]

        d_main = np.linalg.norm(traj - self.obs_center[None, :], axis=1)

        speeds_edge = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.dt
        speeds = np.empty(T, dtype=float)
        if T > 1:
            speeds[0] = speeds_edge[0]
            speeds[1:] = speeds_edge
        else:
            speeds[0] = 0.0

        far_center = np.array((-0.5, 2.5), dtype=float)
        d_far = np.linalg.norm(traj - far_center[None, :], axis=1)

        t = np.linspace(0.0, 2.0 * np.pi, T)
        phase = float(np.dot(traj.mean(axis=0), self.noise_vec) + self.noise_bias)
        noise_feat = 0.2 * np.sin(5.0 * t + phase)

        F = np.stack([d_main, speeds, d_far, noise_feat], axis=1)
        return F if feat_ids is None else F[:, feat_ids]

    def compute_features_all(self, traj):
        F = self.compute_all_features_matrix(traj)
        return F[:, 0], F[:, 1]

def load_2d_obs_avoid(
    n_demos: int = 10,
    seed: int = 42,
    env_kwargs=None,
    demo_kwargs=None,
) -> TaskBundle:
    env_cfg = dict(env_kwargs or {})
    run_kwargs = dict(demo_kwargs or {})

    rng = np.random.RandomState(seed)
    env = ObsAvoidEnv(**env_cfg)
    demos, true_taus = [], []
    for _ in range(n_demos):
        X, tau = env.generate_demo(rng=rng, **run_kwargs)
        demos.append(X)
        true_taus.append(int(tau))
    true_cutpoints = [np.asarray([int(tau)], dtype=int) for tau in true_taus]
    return TaskBundle(
        name="2DObsAvoid",
        demos=demos,
        env=env,
        true_taus=true_taus,
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(),
        constraint_specs=env.get_constraint_specs(),
        meta={"seed": seed, "task_name": "2DObsAvoid"},
    )
