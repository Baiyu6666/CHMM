import numpy as np

from .base import TaskBundle
from .obs_avoid_2d import ObsAvoidEnv
from planner import optimize_trajectory, repair_trajectory_constraints, resample_polyline


class ObsAvoidArc3StageEnv(ObsAvoidEnv):
    """
    Three-stage 2D obstacle avoidance environment.

    Stage 1:
      - move around the obstacle while keeping safe clearance

    Stage 2:
      - move from stage 1 end to the entry point of a terminal arc
      - speed is concentrated around a stage-specific value with reduced fluctuation

    Stage 3:
      - follow a terminal circular arc into the goal
      - speed remains constrained at a lower stage-specific value
      - the equality feature is the distance to the arc center, with target value equal to the arc radius
    """

    def __init__(
        self,
        start=(-1.5, 0.0),
        stage1_end=(0.3, 0.0),
        stage2_end=(0.7, 0.0),
        obs_center=(-0.5, 0.0),
        obs_radius=0.3,
        clearance=0.1,
        stage1_speed_max=0.12,
        stage2_speed_max=0.055,
        stage3_speed_max=0.045,
        stage1_accel_max=0.06,
        stage2_accel_max=0.015,
        stage3_accel_max=0.015,
        terminal_arc_center_offset=(0.0, -0.2),
        terminal_arc_radius=0.2,
        terminal_arc_theta_start=0.5 * np.pi,
        terminal_arc_theta_end=-1.05,
        dt=0.8,
        noise_std=0.003,
    ):
        self.stage1_end = np.asarray(stage1_end, dtype=float)
        self.stage2_end = np.asarray(stage2_end, dtype=float)
        self.terminal_arc_center_offset = np.asarray(terminal_arc_center_offset, dtype=float)
        self.terminal_arc_center = self.stage2_end + self.terminal_arc_center_offset
        self.terminal_arc_radius = float(terminal_arc_radius)
        self.terminal_arc_theta_start = float(terminal_arc_theta_start)
        self.terminal_arc_theta_end = float(terminal_arc_theta_end)
        self.stage3_speed_max = float(stage3_speed_max)
        self.stage3_accel_max = float(stage3_accel_max)
        self.arc_entry = self._arc_point(self.terminal_arc_theta_start)
        if np.linalg.norm(self.arc_entry - self.stage2_end) > 1e-6:
            raise ValueError(
                "terminal arc parameters are inconsistent: stage2_end must coincide with the arc start. "
                "Adjust terminal_arc_center_offset / terminal_arc_radius / terminal_arc_theta_start."
            )
        self.stage3_end = self._arc_point(self.terminal_arc_theta_end)

        super().__init__(
            start=start,
            subgoal=self.stage1_end,
            goal=self.stage2_end,
            obs_center=obs_center,
            obs_radius=obs_radius,
            clearance=clearance,
            stage1_speed_max=stage1_speed_max,
            stage2_speed_max=stage2_speed_max,
            stage1_accel_max=stage1_accel_max,
            stage2_accel_max=stage2_accel_max,
            dt=dt,
            noise_std=noise_std,
        )

        self.subgoal = self.stage1_end.copy()
        self.goal = self.stage3_end.copy()
        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self._direct_true_constraints()
        self.constraint_specs = self.get_constraint_specs()

    def _direct_true_constraints(self):
        return {
            "d_safe": float(self.obs_radius + self.clearance),
            "v2_target": float(self.stage2_speed_max),
            "v3_target": float(self.stage3_speed_max),
            "arc_distance_target": float(self.terminal_arc_radius),
        }

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "dist_main_obstacle", "description": "Distance to main obstacle center"},
            {"id": 1, "name": "speed", "description": "2D speed magnitude"},
            {"id": 2, "name": "terminal_arc_distance", "description": "Distance to the terminal arc center"},
            {"id": 3, "name": "dist_far_obstacle", "description": "Distance to far obstacle center"},
            {"id": 4, "name": "noise_aux", "description": "Deterministic auxiliary noise-like feature"},
        ]

    def get_true_constraints(self):
        return dict(self.true_constraints)

    def get_constraint_specs(self):
        return [
            {
                "feature_name": "dist_main_obstacle",
                "stage": 0,
                "semantics": "lower_bound",
                "oracle_key": "d_safe",
            },
            {
                "feature_name": "speed",
                "stage": 1,
                "semantics": "target_value",
                "oracle_key": "v2_target",
            },
            {
                "feature_name": "terminal_arc_distance",
                "stage": 2,
                "semantics": "target_value",
                "oracle_key": "arc_distance_target",
            },
            {
                "feature_name": "speed",
                "stage": 2,
                "semantics": "target_value",
                "oracle_key": "v3_target",
            },
        ]

    def _arc_point(self, theta):
        theta = float(theta)
        return self.terminal_arc_center + self.terminal_arc_radius * np.array([np.cos(theta), np.sin(theta)], dtype=float)

    def _terminal_arc_points(self, num_points=24):
        thetas = np.linspace(self.terminal_arc_theta_start, self.terminal_arc_theta_end, max(int(num_points), 8))
        return self.terminal_arc_center[None, :] + self.terminal_arc_radius * np.c_[np.cos(thetas), np.sin(thetas)]

    def _nearest_terminal_arc_points(self, path, num_points=256):
        pts = np.asarray(path, dtype=float).reshape(-1, 2)
        arc_pts = self._terminal_arc_points(num_points=max(int(num_points), 32))
        d2 = np.sum((pts[:, None, :] - arc_pts[None, :, :]) ** 2, axis=2)
        nearest_idx = np.argmin(d2, axis=1)
        return arc_pts[nearest_idx]

    def _project_to_terminal_arc(self, path):
        return self._nearest_terminal_arc_points(path)

    def _terminal_arc_distance(self, traj):
        pts = np.asarray(traj, dtype=float)
        return np.linalg.norm(pts - self.terminal_arc_center[None, :], axis=1)

    def generate_demo(
        self,
        n1=24,
        direction=None,
        start_x_range=(-2., -0.5),
        start_y_range=(-0.5, 0.5),
        start_safe_scale=1.08,
        start_resample_tries=64,
        start_angle_jitter_deg=22.0,
        start_radius_jitter_frac=0.18,
        start_tangent_jitter_frac=0.18,
        subgoal_jitter_frac=0.05,
        arc_entry_jitter_scale=0.006,
        stage2_lateral_scale=0.003,
        process_noise_scale=0.005,
        process_noise_knots=3,
        stage1_speed_scale=1.0,
        stage2_speed_scale=1.0,
        stage3_speed_scale=1.0,
        rng=None,
    ):
        rng = np.random if rng is None else rng
        if direction is None:
            direction = rng.choice(["up", "down"])

        d_safe = float(self.true_constraints["d_safe"])
        seg_len = np.linalg.norm(self.stage3_end - self.stage1_end) + 1e-12

        start_local = None
        if start_x_range is not None and start_y_range is not None:
            x_lo, x_hi = sorted([float(start_x_range[0]), float(start_x_range[1])])
            y_lo, y_hi = sorted([float(start_y_range[0]), float(start_y_range[1])])
            min_start_dist = d_safe * max(float(start_safe_scale), 1.0)
            for _ in range(max(int(start_resample_tries), 1)):
                candidate = np.array([rng.uniform(x_lo, x_hi), rng.uniform(y_lo, y_hi)], dtype=float)
                if np.linalg.norm(candidate - self.obs_center) >= min_start_dist:
                    start_local = candidate
                    break
            if start_local is None:
                corners = np.array([[x_lo, y_lo], [x_lo, y_hi], [x_hi, y_lo], [x_hi, y_hi]], dtype=float)
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

        stage1_end_local = self.stage1_end + rng.randn(2) * (subgoal_jitter_frac * seg_len)
        arc_entry_local = self.arc_entry.copy()

        start_rel = start_local - self.obs_center
        stage1_end_rel = stage1_end_local - self.obs_center
        theta_start = np.arctan2(start_rel[1], start_rel[0])
        theta_end = np.arctan2(stage1_end_rel[1], stage1_end_rel[0])
        start_anchor = self.obs_center + d_safe * start_rel / max(np.linalg.norm(start_rel), 1e-12)
        end_anchor = self.obs_center + d_safe * stage1_end_rel / max(np.linalg.norm(stage1_end_rel), 1e-12)

        arc = self._arc_points(theta_start, theta_end, direction, d_safe, n1)
        stage1_ctrl = np.vstack([start_local, start_anchor, arc[1:-1], end_anchor, stage1_end_local])
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

        stage2_vec = arc_entry_local - stage1_end_local
        stage2_len = np.linalg.norm(stage2_vec) + 1e-12
        tangent = stage2_vec / stage2_len
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        u2 = np.linspace(0.0, 1.0, 14)
        wobble = (
            stage2_lateral_scale
            * stage2_len
            * rng.uniform(0.5, 1.0)
            * np.sin(np.pi * u2)
            * np.sin(2.0 * np.pi * u2 + rng.uniform(0.0, 2.0 * np.pi))
        )
        stage2_ctrl = stage1_end_local[None, :] + u2[:, None] * stage2_vec[None, :] + wobble[:, None] * normal[None, :]
        stage2_ctrl[0] = stage1_end_local
        stage2_ctrl[-1] = arc_entry_local
        stage2_ref = resample_polyline(stage2_ctrl, self.stage2_speed_max * stage2_speed_scale * self.dt)
        stage2 = optimize_trajectory(
            stage2_ref,
            dt=self.dt,
            v_max=self.stage2_speed_max * stage2_speed_scale,
            a_max=self.stage2_accel_max,
            projector=None,
        )

        stage3_ctrl = self._terminal_arc_points(num_points=36)
        stage3_ctrl[0] = arc_entry_local
        stage3_ctrl[-1] = self.stage3_end
        stage3_ref = resample_polyline(stage3_ctrl, self.stage3_speed_max * stage3_speed_scale * self.dt)
        stage3 = optimize_trajectory(
            stage3_ref,
            dt=self.dt,
            v_max=self.stage3_speed_max * stage3_speed_scale,
            a_max=self.stage3_accel_max,
            projector=self._project_to_terminal_arc,
        )

        tau1 = int(len(stage1) - 1)
        tau2 = int(len(stage1) + len(stage2) - 2)
        traj = np.vstack([stage1, stage2[1:], stage3[1:]])

        traj = traj + self._smooth_process_noise(
            len(traj),
            traj.shape[1],
            process_noise_scale * seg_len,
            process_noise_knots,
            rng,
        )
        if self.noise_std > 0.0:
            traj = traj + rng.randn(*traj.shape) * (self.noise_std * 0.2)
            stage1_noisy = repair_trajectory_constraints(
                traj[: tau1 + 1],
                dt=self.dt,
                v_max=self.stage1_speed_max * stage1_speed_scale,
                a_max=self.stage1_accel_max,
                projector=stage1_projector,
            )
            stage2_noisy = repair_trajectory_constraints(
                traj[tau1 : tau2 + 1],
                dt=self.dt,
                v_max=self.stage2_speed_max * stage2_speed_scale,
                a_max=self.stage2_accel_max,
                projector=None,
            )
            stage3_noisy = repair_trajectory_constraints(
                traj[tau2:],
                dt=self.dt,
                v_max=self.stage3_speed_max * stage3_speed_scale,
                a_max=self.stage3_accel_max,
                projector=self._project_to_terminal_arc,
            )
            traj = np.vstack([stage1_noisy, stage2_noisy[1:], stage3_noisy[1:]])

        return traj, np.asarray([tau1, tau2], dtype=int)

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

        d_arc = self._terminal_arc_distance(traj)

        far_center = np.array((-0.5, 2.5), dtype=float)
        d_far = np.linalg.norm(traj - far_center[None, :], axis=1)

        t = np.linspace(0.0, 2.0 * np.pi, T)
        phase = float(np.dot(traj.mean(axis=0), self.noise_vec) + self.noise_bias)
        noise_feat = 0.2 * np.sin(5.0 * t + phase)

        F = np.stack([d_main, speeds, d_arc, d_far, noise_feat], axis=1)
        return F if feat_ids is None else F[:, feat_ids]

    def compute_features_all(self, traj):
        F = self.compute_all_features_matrix(traj)
        return F[:, 0], F[:, 1]


def load_2d_obs_avoid_arc3(
    n_demos: int = 10,
    seed: int = 42,
    env_kwargs=None,
    demo_kwargs=None,
) -> TaskBundle:
    env_cfg = dict(env_kwargs or {})
    run_kwargs = dict(demo_kwargs or {})

    rng = np.random.RandomState(seed)
    env = ObsAvoidArc3StageEnv(**env_cfg)
    demos, true_cutpoints = [], []
    for _ in range(n_demos):
        X, cuts = env.generate_demo(rng=rng, **run_kwargs)
        demos.append(X)
        true_cutpoints.append(np.asarray(cuts, dtype=int))
    return TaskBundle(
        name="2DObsAvoidArc3",
        demos=demos,
        env=env,
        true_taus=None,
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(),
        constraint_specs=env.get_constraint_specs(),
        meta={"seed": seed, "task_name": "2DObsAvoidArc3"},
    )
