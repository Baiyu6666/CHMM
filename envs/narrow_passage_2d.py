import numpy as np

from .base import TaskBundle
from planner import optimize_trajectory, repair_trajectory_constraints, resample_polyline


class NarrowPassageEnv2D:
    """
    Two-stage narrow-passage task.

    Stage 1:
      stay inside a narrow corridor, expressed by a lower bound on
      ``corridor_margin``.

    Stage 2:
      leave the corridor and move quickly toward the goal, expressed by a
      lower bound on ``speed``.
    """

    def __init__(
        self,
        start=(-1.2, 0.02),
        subgoal=(0.15, 0.0),
        goal=(1.1, 0.45),
        corridor_center_y=0.0,
        corridor_half_width=0.14,
        corridor_x_min=-1.3,
        corridor_x_max=0.25,
        stage1_speed_max=0.08,
        stage2_speed_nominal=0.14,
        stage2_speed_min=0.10,
        stage1_accel_max=0.05,
        stage2_accel_max=0.08,
        dt=1.0,
        noise_std=0.008,
    ):
        self.start = np.array(start, dtype=float)
        self.subgoal = np.array(subgoal, dtype=float)
        self.goal = np.array(goal, dtype=float)

        self.corridor_center_y = float(corridor_center_y)
        self.corridor_half_width = float(corridor_half_width)
        self.corridor_x_min = float(corridor_x_min)
        self.corridor_x_max = float(corridor_x_max)

        self.stage1_speed_max = float(stage1_speed_max)
        self.stage2_speed_nominal = float(stage2_speed_nominal)
        self.stage2_speed_min = float(stage2_speed_min)
        self.stage1_accel_max = float(stage1_accel_max)
        self.stage2_accel_max = float(stage2_accel_max)

        self.dt = float(dt)
        self.noise_std = float(noise_std)

        self.noise_vec = np.array([0.41, -0.33], dtype=float)
        self.noise_bias = 0.15
        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self._direct_true_constraints()
        self.constraint_specs = self.get_constraint_specs()

    def _direct_true_constraints(self):
        return {
            "corridor_margin_min": float(0.6 * self.corridor_half_width),
            "v1_max": float(self.stage1_speed_max),
            "v2_min": float(self.stage2_speed_min),
            "a1_max": float(self.stage1_accel_max),
            "a2_max": float(self.stage2_accel_max),
        }

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "corridor_margin", "description": "Signed margin to corridor boundary"},
            {"id": 1, "name": "speed", "description": "2D speed magnitude"},
            {"id": 2, "name": "progress_along_corridor", "description": "Normalized progress along corridor"},
            {"id": 3, "name": "noise_aux", "description": "Deterministic auxiliary feature"},
        ]

    def get_true_constraints(self):
        return dict(self.true_constraints)

    def get_constraint_specs(self):
        return [
            {
                "metric": "AbsErr_corridor_margin",
                "feature_name": "corridor_margin",
                "stage": 0,
                "semantics": "lower_bound",
                "oracle_key": "corridor_margin_min",
            },
            {
                "metric": "AbsErr_speed_min",
                "feature_name": "speed",
                "stage": 1,
                "semantics": "lower_bound",
                "oracle_key": "v2_min",
            },
        ]

    def _stage1_projector(self, path):
        out = np.asarray(path, dtype=float).copy()
        out[:, 0] = np.clip(out[:, 0], self.corridor_x_min, self.corridor_x_max)
        y_dev = out[:, 1] - self.corridor_center_y
        out[:, 1] = self.corridor_center_y + np.clip(
            y_dev,
            -self.corridor_half_width,
            self.corridor_half_width,
        )
        return out

    def generate_demo(
        self,
        stage1_y_jitter=0.025,
        stage2_bend_scale=0.12,
        stage2_speed_scale=1.0,
        rng=None,
    ):
        rng = np.random if rng is None else rng

        y0 = self.corridor_center_y + rng.uniform(-stage1_y_jitter, stage1_y_jitter)
        y1 = self.corridor_center_y + rng.uniform(-stage1_y_jitter, stage1_y_jitter)
        start_local = np.array([self.start[0], y0], dtype=float)
        subgoal_local = np.array([self.subgoal[0], y1], dtype=float)
        goal_local = self.goal + np.array(
            [rng.uniform(-0.04, 0.04), rng.uniform(-0.05, 0.05)],
            dtype=float,
        )

        x_ctrl = np.linspace(start_local[0], subgoal_local[0], 9)
        y_ctrl = np.linspace(start_local[1], subgoal_local[1], 9)
        stage1_ctrl = np.c_[x_ctrl, y_ctrl]
        stage1_ref = resample_polyline(stage1_ctrl, self.stage1_speed_max * self.dt)
        stage1 = optimize_trajectory(
            stage1_ref,
            dt=self.dt,
            v_max=self.stage1_speed_max,
            a_max=self.stage1_accel_max,
            projector=self._stage1_projector,
        )
        true_tau = int(len(stage1) - 1)

        d_goal = goal_local - subgoal_local
        d_goal_norm = np.linalg.norm(d_goal) + 1e-12
        tangent = d_goal / d_goal_norm
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        u = np.linspace(0.0, 1.0, 26)
        bend = stage2_bend_scale * d_goal_norm * rng.uniform(0.3, 0.8)
        wobble = bend * np.sin(np.pi * u) * np.sin(np.pi * u + rng.uniform(0.0, np.pi))
        stage2_ctrl = subgoal_local[None, :] + u[:, None] * d_goal[None, :] + wobble[:, None] * normal[None, :]
        stage2_ctrl[0] = subgoal_local
        stage2_ctrl[-1] = goal_local
        stage2_ref = resample_polyline(stage2_ctrl, self.stage2_speed_nominal * stage2_speed_scale * self.dt)
        stage2 = optimize_trajectory(
            stage2_ref,
            dt=self.dt,
            v_max=max(self.stage2_speed_nominal * stage2_speed_scale, self.stage2_speed_min * 1.05),
            a_max=self.stage2_accel_max,
            projector=None,
        )

        traj = np.vstack([stage1, stage2[1:]])
        if self.noise_std > 0.0:
            traj = traj + rng.randn(*traj.shape) * self.noise_std
            stage1_noisy = repair_trajectory_constraints(
                traj[: true_tau + 1],
                dt=self.dt,
                v_max=self.stage1_speed_max,
                a_max=self.stage1_accel_max,
                projector=self._stage1_projector,
            )
            stage2_noisy = repair_trajectory_constraints(
                traj[true_tau:],
                dt=self.dt,
                v_max=max(self.stage2_speed_nominal * stage2_speed_scale, self.stage2_speed_min * 1.05),
                a_max=self.stage2_accel_max,
                projector=None,
            )
            traj = np.vstack([stage1_noisy, stage2_noisy[1:]])
        return traj, true_tau

    def compute_all_features_matrix(self, traj, feat_ids=None):
        traj = np.asarray(traj, dtype=float)
        T = traj.shape[0]

        corridor_margin = self.corridor_half_width - np.abs(traj[:, 1] - self.corridor_center_y)

        speeds_edge = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.dt
        speed = np.empty(T, dtype=float)
        if T > 1:
            speed[0] = speeds_edge[0]
            speed[1:] = speeds_edge
        else:
            speed[0] = 0.0

        denom = max(self.corridor_x_max - self.corridor_x_min, 1e-12)
        progress = np.clip((traj[:, 0] - self.corridor_x_min) / denom, 0.0, 1.0)

        t = np.linspace(0.0, 2.0 * np.pi, T)
        phase = float(np.dot(traj.mean(axis=0), self.noise_vec) + self.noise_bias)
        noise_feat = 0.18 * np.sin(4.0 * t + phase)

        F = np.stack([corridor_margin, speed, progress, noise_feat], axis=1)
        return F if feat_ids is None else F[:, feat_ids]

    def compute_features_all(self, traj):
        F = self.compute_all_features_matrix(traj)
        return F[:, 0], F[:, 1]


def load_2d_narrow_passage(
    n_demos: int = 10,
    seed: int = 42,
    env_kwargs=None,
    demo_kwargs=None,
) -> TaskBundle:
    env_cfg = {
        "start": (-1.2, 0.02),
        "subgoal": (0.15, 0.0),
        "goal": (1.1, 0.45),
        "corridor_center_y": 0.0,
        "corridor_half_width": 0.14,
        "corridor_x_min": -1.3,
        "corridor_x_max": 0.25,
        "stage1_speed_max": 0.08,
        "stage2_speed_nominal": 0.14,
        "stage2_speed_min": 0.10,
        "stage1_accel_max": 0.05,
        "stage2_accel_max": 0.08,
        "dt": 1.0,
        "noise_std": 0.008,
    }
    if env_kwargs:
        env_cfg.update(env_kwargs)

    run_kwargs = {}
    if demo_kwargs:
        run_kwargs.update(demo_kwargs)

    rng = np.random.RandomState(seed)
    env = NarrowPassageEnv2D(**env_cfg)
    demos, true_taus = [], []
    for _ in range(n_demos):
        X, tau = env.generate_demo(rng=rng, **run_kwargs)
        demos.append(X)
        true_taus.append(int(tau))
    true_cutpoints = [np.asarray([int(tau)], dtype=int) for tau in true_taus]

    return TaskBundle(
        name="2DNarrowPassage",
        demos=demos,
        env=env,
        true_taus=true_taus,
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(),
        constraint_specs=env.get_constraint_specs(),
        meta={"seed": seed, "task_name": "2DNarrowPassage"},
    )
