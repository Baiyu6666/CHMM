import numpy as np

from .base import TaskBundle
from planner import optimize_trajectory, repair_trajectory_constraints, resample_polyline


class DockCorridorEnv2D:
    """
    Two-stage docking approach.

    Stage 1:
      stay inside the docking lane, expressed by an upper bound on
      ``lateral_offset``.

    Stage 2:
      approach the dock slowly, expressed by an upper bound on ``speed``.
    """

    def __init__(
        self,
        start=(-1.25, 0.03),
        subgoal=(0.30, 0.01),
        goal=(0.95, 0.0),
        dock_x=0.95,
        lane_center_y=0.0,
        lane_half_width=0.10,
        stage1_speed_max=0.10,
        stage2_speed_max=0.045,
        stage1_accel_max=0.06,
        stage2_accel_max=0.035,
        dt=1.0,
        noise_std=0.007,
    ):
        self.start = np.array(start, dtype=float)
        self.subgoal = np.array(subgoal, dtype=float)
        self.goal = np.array(goal, dtype=float)

        self.dock_x = float(dock_x)
        self.lane_center_y = float(lane_center_y)
        self.lane_half_width = float(lane_half_width)

        self.stage1_speed_max = float(stage1_speed_max)
        self.stage2_speed_max = float(stage2_speed_max)
        self.stage1_accel_max = float(stage1_accel_max)
        self.stage2_accel_max = float(stage2_accel_max)

        self.dt = float(dt)
        self.noise_std = float(noise_std)

        self.noise_vec = np.array([-0.22, 0.51], dtype=float)
        self.noise_bias = -0.2
        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self._direct_true_constraints()
        self.constraint_specs = self.get_constraint_specs()

    def _direct_true_constraints(self):
        return {
            "lateral_offset_max": float(0.7 * self.lane_half_width),
            "v1_max": float(self.stage1_speed_max),
            "v2_max": float(self.stage2_speed_max),
            "a1_max": float(self.stage1_accel_max),
            "a2_max": float(self.stage2_accel_max),
        }

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "lateral_offset", "description": "Absolute offset from docking lane centerline"},
            {"id": 1, "name": "distance_to_dock", "description": "Euclidean distance to dock point"},
            {"id": 2, "name": "speed", "description": "2D speed magnitude"},
            {"id": 3, "name": "noise_aux", "description": "Deterministic auxiliary feature"},
        ]

    def get_true_constraints(self):
        return dict(self.true_constraints)

    def get_constraint_specs(self):
        return [
            {
                "metric": "AbsErr_lateral_offset",
                "feature_name": "lateral_offset",
                "stage": 0,
                "semantics": "upper_bound",
                "oracle_key": "lateral_offset_max",
            },
            {
                "metric": "AbsErr_v",
                "feature_name": "speed",
                "stage": 1,
                "semantics": "upper_bound",
                "oracle_key": "v2_max",
            },
        ]

    def _lane_projector(self, path):
        out = np.asarray(path, dtype=float).copy()
        y_dev = out[:, 1] - self.lane_center_y
        out[:, 1] = self.lane_center_y + np.clip(y_dev, -self.lane_half_width, self.lane_half_width)
        return out

    def generate_demo(
        self,
        lane_jitter=0.03,
        approach_bend_scale=0.06,
        rng=None,
    ):
        rng = np.random if rng is None else rng

        start_local = self.start.copy()
        subgoal_local = self.subgoal.copy()
        start_local[1] = self.lane_center_y + rng.uniform(-lane_jitter, lane_jitter)
        subgoal_local[1] = self.lane_center_y + rng.uniform(-lane_jitter, lane_jitter)
        goal_local = self.goal + np.array([0.0, rng.uniform(-0.01, 0.01)], dtype=float)

        x_ctrl = np.linspace(start_local[0], subgoal_local[0], 10)
        y_ctrl = np.linspace(start_local[1], subgoal_local[1], 10)
        stage1_ctrl = np.c_[x_ctrl, y_ctrl]
        stage1_ref = resample_polyline(stage1_ctrl, self.stage1_speed_max * self.dt)
        stage1 = optimize_trajectory(
            stage1_ref,
            dt=self.dt,
            v_max=self.stage1_speed_max,
            a_max=self.stage1_accel_max,
            projector=self._lane_projector,
        )
        true_tau = int(len(stage1) - 1)

        d_goal = goal_local - subgoal_local
        d_goal_norm = np.linalg.norm(d_goal) + 1e-12
        tangent = d_goal / d_goal_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        u = np.linspace(0.0, 1.0, 24)
        bend = approach_bend_scale * d_goal_norm * rng.uniform(-1.0, 1.0)
        wobble = bend * np.sin(np.pi * u)
        stage2_ctrl = subgoal_local[None, :] + u[:, None] * d_goal[None, :] + wobble[:, None] * normal[None, :]
        stage2_ctrl[0] = subgoal_local
        stage2_ctrl[-1] = goal_local
        stage2_ref = resample_polyline(stage2_ctrl, self.stage2_speed_max * self.dt)
        stage2 = optimize_trajectory(
            stage2_ref,
            dt=self.dt,
            v_max=self.stage2_speed_max,
            a_max=self.stage2_accel_max,
            projector=self._lane_projector,
        )

        traj = np.vstack([stage1, stage2[1:]])
        if self.noise_std > 0.0:
            traj = traj + rng.randn(*traj.shape) * self.noise_std
            stage1_noisy = repair_trajectory_constraints(
                traj[: true_tau + 1],
                dt=self.dt,
                v_max=self.stage1_speed_max,
                a_max=self.stage1_accel_max,
                projector=self._lane_projector,
            )
            stage2_noisy = repair_trajectory_constraints(
                traj[true_tau:],
                dt=self.dt,
                v_max=self.stage2_speed_max,
                a_max=self.stage2_accel_max,
                projector=self._lane_projector,
            )
            traj = np.vstack([stage1_noisy, stage2_noisy[1:]])

        return traj, true_tau

    def compute_all_features_matrix(self, traj, feat_ids=None):
        traj = np.asarray(traj, dtype=float)
        T = traj.shape[0]

        lateral_offset = np.abs(traj[:, 1] - self.lane_center_y)
        dock_point = np.array([self.dock_x, self.lane_center_y], dtype=float)
        distance_to_dock = np.linalg.norm(traj - dock_point[None, :], axis=1)

        speeds_edge = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.dt
        speed = np.empty(T, dtype=float)
        if T > 1:
            speed[0] = speeds_edge[0]
            speed[1:] = speeds_edge
        else:
            speed[0] = 0.0

        t = np.linspace(0.0, 2.0 * np.pi, T)
        phase = float(np.dot(traj.mean(axis=0), self.noise_vec) + self.noise_bias)
        noise_feat = 0.14 * np.cos(4.5 * t + phase)

        F = np.stack([lateral_offset, distance_to_dock, speed, noise_feat], axis=1)
        return F if feat_ids is None else F[:, feat_ids]

    def compute_features_all(self, traj):
        F = self.compute_all_features_matrix(traj)
        return F[:, 0], F[:, 2]


def load_2d_dock_corridor(
    n_demos: int = 10,
    seed: int = 42,
    env_kwargs=None,
    demo_kwargs=None,
) -> TaskBundle:
    env_cfg = {
        "start": (-1.25, 0.03),
        "subgoal": (0.30, 0.01),
        "goal": (0.95, 0.0),
        "dock_x": 0.95,
        "lane_center_y": 0.0,
        "lane_half_width": 0.10,
        "stage1_speed_max": 0.10,
        "stage2_speed_max": 0.045,
        "stage1_accel_max": 0.06,
        "stage2_accel_max": 0.035,
        "dt": 1.0,
        "noise_std": 0.007,
    }
    if env_kwargs:
        env_cfg.update(env_kwargs)

    run_kwargs = {}
    if demo_kwargs:
        run_kwargs.update(demo_kwargs)

    rng = np.random.RandomState(seed)
    env = DockCorridorEnv2D(**env_cfg)
    demos, true_taus = [], []
    for _ in range(n_demos):
        X, tau = env.generate_demo(rng=rng, **run_kwargs)
        demos.append(X)
        true_taus.append(int(tau))
    true_cutpoints = [np.asarray([int(tau)], dtype=int) for tau in true_taus]

    return TaskBundle(
        name="2DDockCorridor",
        demos=demos,
        env=env,
        true_taus=true_taus,
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(),
        constraint_specs=env.get_constraint_specs(),
        meta={"seed": seed, "task_name": "2DDockCorridor"},
    )
