import numpy as np

from .base import TaskBundle
from .obs_avoid_2d import ObsAvoidEnv
from planner import optimize_trajectory, repair_trajectory_constraints, resample_polyline


class ObsAvoidEnv3D:
    """
    3D obstacle-avoidance environment with explicit oracle constraints.

    XY motion follows the 2D obstacle-avoidance path generator. The 3D trajectory
    is then constructed by adding stage-wise Z interpolation and resampling in 3D
    so that the resulting speeds respect the configured stage caps.
    """

    def __init__(
        self,
        start_xy=(-1.5, 0.0),
        subgoal_xy=(0.5, 0.0),
        goal_xy=(0.0, 0.3),
        obs_center_xy=(-0.5, 0.0),
        obs_radius=0.3,
        clearance=0.1,
        stage1_speed_max=0.12,
        stage2_speed_max=0.06,
        stage1_accel_max=0.06,
        stage2_accel_max=0.04,
        start_z_range=(0.2, 0.6),
        subgoal_z=0.4,
        goal_z=0.6,
        dt=1.0,
        noise_std=0.01,
    ):
        self.start_xy = np.array(start_xy, float)
        self.subgoal_xy = np.array(subgoal_xy, float)
        self.goal_xy = np.array(goal_xy, float)

        self.obs_center_xy = np.array(obs_center_xy, float)
        self.obs_radius = float(obs_radius)
        self.clearance = float(clearance)
        self.stage1_speed_max = float(stage1_speed_max)
        self.stage2_speed_max = float(stage2_speed_max)
        self.stage1_accel_max = float(stage1_accel_max)
        self.stage2_accel_max = float(stage2_accel_max)

        self.start_z_range = tuple(start_z_range)
        self.subgoal_z = float(subgoal_z)
        self.goal_z = float(goal_z)

        self.dt = float(dt)
        self.noise_std = float(noise_std)

        self.subgoal = np.array([self.subgoal_xy[0], self.subgoal_xy[1], self.subgoal_z], float)
        self.goal = np.array([self.goal_xy[0], self.goal_xy[1], self.goal_z], float)

        self.env2d = ObsAvoidEnv(
            start=start_xy,
            subgoal=subgoal_xy,
            goal=goal_xy,
            obs_center=obs_center_xy,
            obs_radius=obs_radius,
            clearance=clearance,
            stage1_speed_max=stage1_speed_max,
            stage2_speed_max=stage2_speed_max,
            dt=dt,
            noise_std=noise_std,
        )

        self.noise_vec3 = np.array([0.31, -0.47, 0.62], float)
        self.noise_bias3 = 0.0
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
            {"id": 0, "name": "dist_main_obstacle_xy", "description": "XY distance to main obstacle center"},
            {"id": 1, "name": "speed", "description": "3D speed magnitude"},
            {"id": 2, "name": "dist_far_obstacle_xy", "description": "XY distance to far obstacle center"},
            {"id": 3, "name": "noise_aux", "description": "Deterministic auxiliary noise-like feature"},
        ]

    def get_constraint_specs(self):
        return [
            {
                "metric": "AbsErr_d",
                "feature_name": "dist_main_obstacle_xy",
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

    def generate_demo_3d(self, rng=None, **kwargs):
        rng = np.random if rng is None else rng
        traj2d, tau2d = self.env2d.generate_demo(rng=rng, **kwargs)
        start_z = rng.uniform(*self.start_z_range)

        stage1_xy = traj2d[: tau2d + 1]
        stage2_xy = traj2d[tau2d:]

        z1 = np.linspace(start_z, self.subgoal_z, len(stage1_xy))
        z2 = np.linspace(self.subgoal_z, self.goal_z, len(stage2_xy))
        stage1_ctrl = np.column_stack([stage1_xy, z1])
        stage2_ctrl = np.column_stack([stage2_xy, z2])

        d_safe = float(self.true_constraints["d_safe"])

        def stage1_projector(path):
            out = np.asarray(path, dtype=float).copy()
            rel_xy = out[:, :2] - self.obs_center_xy[None, :]
            norms = np.linalg.norm(rel_xy, axis=1, keepdims=True)
            mask = norms[:, 0] < d_safe
            if np.any(mask):
                rel_xy[mask] /= np.maximum(norms[mask], 1e-12)
                out[mask, :2] = self.obs_center_xy[None, :] + d_safe * rel_xy[mask]
            return out

        stage1_ref = resample_polyline(stage1_ctrl, self.stage1_speed_max * self.dt)
        stage2_ref = resample_polyline(stage2_ctrl, self.stage2_speed_max * self.dt)
        stage1 = optimize_trajectory(
            stage1_ref,
            dt=self.dt,
            v_max=self.stage1_speed_max,
            a_max=self.stage1_accel_max,
            projector=stage1_projector,
        )
        stage2 = optimize_trajectory(
            stage2_ref,
            dt=self.dt,
            v_max=self.stage2_speed_max,
            a_max=self.stage2_accel_max,
            projector=None,
        )
        true_tau = int(len(stage1) - 1)

        traj3d = np.vstack([stage1, stage2[1:]])
        if self.noise_std > 0.0:
            traj3d = traj3d + rng.randn(*traj3d.shape) * (self.noise_std * 0.25)
            stage1_noisy = repair_trajectory_constraints(
                traj3d[: true_tau + 1],
                dt=self.dt,
                v_max=self.stage1_speed_max,
                a_max=self.stage1_accel_max,
                projector=stage1_projector,
            )
            stage2_noisy = repair_trajectory_constraints(
                traj3d[true_tau:],
                dt=self.dt,
                v_max=self.stage2_speed_max,
                a_max=self.stage2_accel_max,
                projector=None,
            )
            traj3d = np.vstack([stage1_noisy, stage2_noisy[1:]])

        return traj3d, true_tau

    def compute_all_features_matrix(self, traj3d, feat_ids=None):
        traj3d = np.asarray(traj3d, float)
        xy = traj3d[:, :2]
        T = traj3d.shape[0]

        d_main = np.linalg.norm(xy - self.obs_center_xy[None, :], axis=1)

        speeds_edge = np.linalg.norm(np.diff(traj3d, axis=0), axis=1) / self.dt
        speeds = np.empty(T, dtype=float)
        if T > 1:
            speeds[0] = speeds_edge[0]
            speeds[1:] = speeds_edge
        else:
            speeds[0] = 0.0

        far_center_xy = np.array((-0.5, 2.5), dtype=float)
        d_far = np.linalg.norm(xy - far_center_xy[None, :], axis=1)

        t = np.linspace(0.0, 2.0 * np.pi, T)
        phase = float(np.dot(traj3d.mean(axis=0), self.noise_vec3) + self.noise_bias3)
        noise_feat = 0.2 * np.sin(5.0 * t + phase)

        F = np.stack([d_main, speeds, d_far, noise_feat], axis=1)
        return F if feat_ids is None else F[:, feat_ids]

    def compute_features_all(self, traj3d):
        F = self.compute_all_features_matrix(traj3d)
        return F[:, 0], F[:, 1]

def load_3d_obs_avoid(
    n_demos: int = 10,
    seed: int = 42,
    env_kwargs=None,
    demo_kwargs=None,
) -> TaskBundle:
    env_cfg = {
        "start_xy": (-1.5, 0.0),
        "subgoal_xy": (0.5, 0.0),
        "goal_xy": (-0.2, 0.5),
        "obs_center_xy": (-0.5, 0.0),
        "obs_radius": 0.3,
        "clearance": 0.1,
        "stage1_speed_max": 0.12,
        "stage2_speed_max": 0.06,
        "stage1_accel_max": 0.06,
        "stage2_accel_max": 0.04,
        "start_z_range": (0.2, 0.7),
        "subgoal_z": 0.4,
        "goal_z": 0.05,
        "dt": 1.0,
        "noise_std": 0.01,
    }
    if env_kwargs:
        env_cfg.update(env_kwargs)

    run_kwargs = {"n1": 24, "direction": None}
    if demo_kwargs:
        run_kwargs.update(demo_kwargs)

    rng = np.random.RandomState(seed)
    env = ObsAvoidEnv3D(**env_cfg)
    demos, true_taus = [], []
    for _ in range(n_demos):
        X, tau = env.generate_demo_3d(rng=rng, **run_kwargs)
        demos.append(X)
        true_taus.append(int(tau))
    return TaskBundle(
        name="3DObsAvoid",
        demos=demos,
        env=env,
        true_taus=true_taus,
        feature_schema=env.get_feature_schema(),
        true_constraints=dict(env.true_constraints),
        constraint_specs=env.get_constraint_specs(),
        meta={"seed": seed, "task_name": "3DObsAvoid"},
    )
