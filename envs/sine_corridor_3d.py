import numpy as np

from .base import TaskBundle
from planner import optimize_trajectory, resample_polyline


class SineCorridorEnv3D:
    """
    3D sine-corridor environment with explicit oracle constraints.

    Stage 1:
      - track the sine centerline with bounded deviation
      - speed is bounded by ``stage1_speed_max``

    Stage 2:
      - move toward the final goal
      - speed is bounded by ``stage2_speed_max``
    """

    def __init__(
        self,
        A=0.1,
        omega=10.0,
        bias=0.2,
        phase=0.0,
        x_start_range=(-2.0, -1.0),
        z_start_range=(0.0, 0.5),
        x_sub=0.0,
        z_sub=0.6,
        goal=(0.5, 0.0, 0.0),
        corridor_half_width=0.2,
        centerline_tol=0.03,
        stage1_speed_max=0.08,
        stage2_speed_max=0.05,
        stage1_accel_max=0.04,
        stage2_accel_max=0.03,
        dt=1.0,
    ):
        self.A = float(A)
        self.omega = float(omega)
        self.bias = float(bias)
        self.phase = float(phase)

        self.x_start_range = tuple(x_start_range)
        self.z_start_range = tuple(z_start_range)
        self.corridor_half_width = float(corridor_half_width)
        self.centerline_tol = float(centerline_tol)
        self.stage1_speed_max = float(stage1_speed_max)
        self.stage2_speed_max = float(stage2_speed_max)
        self.stage1_accel_max = float(stage1_accel_max)
        self.stage2_accel_max = float(stage2_accel_max)
        self.dt = float(dt)

        y_sub = self.centerline(x_sub)
        self.subgoal = np.array([x_sub, y_sub, z_sub], dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.true_constraints = self._direct_true_constraints()
        self.feature_schema = self.get_feature_schema()
        self.constraint_specs = self.get_constraint_specs()

    def _direct_true_constraints(self):
        return {
            "centerline_target": 0.0,
            "centerline_tol": float(self.centerline_tol),
            "v1_max": float(self.stage1_speed_max),
            "v2_max": float(self.stage2_speed_max),
            "a1_max": float(self.stage1_accel_max),
            "a2_max": float(self.stage2_accel_max),
        }

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "dist_centerline", "description": "Absolute distance to sine centerline"},
            {"id": 1, "name": "speed", "description": "3D speed magnitude"},
            {"id": 2, "name": "dist_far_reference", "description": "Distance to far XY reference point"},
            {"id": 3, "name": "noise_aux", "description": "Deterministic auxiliary noise-like feature"},
        ]

    def get_constraint_specs(self):
        return [
            {
                "metric": "AbsErr_centerline",
                "feature_name": "dist_centerline",
                "stage": 0,
                "semantics": "target_value",
                "oracle_key": "centerline_target",
                "estimator": "center",
            },
            {
                "metric": "AbsErr_v",
                "feature_name": "speed",
                "stage": 1,
                "semantics": "upper_bound",
                "oracle_key": "v2_max",
            },
        ]

    def centerline(self, x):
        return self.A * np.sin(self.omega * x + self.phase) + self.bias

    def rollout_demo(
        self,
        n1=40,
        lateral_scale=1.0,
        rng=None,
    ):
        rng = np.random if rng is None else rng
        x0 = rng.uniform(*self.x_start_range)
        z0 = rng.uniform(*self.z_start_range)
        y0 = self.centerline(x0)
        p0 = np.array([x0, y0, z0], dtype=float)

        x_sub, _, z_sub = self.subgoal
        p_sub = self.subgoal.copy()
        p_goal = self.goal.copy()

        xs1 = np.linspace(x0, x_sub, num=max(int(n1), 16))
        u1 = np.linspace(0.0, 1.0, len(xs1))
        amp = self.centerline_tol * lateral_scale * rng.uniform(0.25, 0.8)
        amp = min(amp, self.centerline_tol)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        dev = amp * np.sin(np.pi * u1) * np.sin(2.0 * np.pi * u1 + phase)
        dev = np.clip(dev, -self.centerline_tol, self.centerline_tol)
        ys1 = self.centerline(xs1) + dev
        zs1 = np.linspace(z0, z_sub, len(xs1))
        stage1_ctrl = np.stack([xs1, ys1, zs1], axis=1)
        stage1_ctrl[0] = p0
        stage1_ctrl[-1] = p_sub
        def stage1_projector(path):
            out = np.asarray(path, dtype=float).copy()
            x = out[:, 0]
            y_center = self.centerline(x)
            dev = out[:, 1] - y_center
            dev = np.clip(dev, -self.centerline_tol, self.centerline_tol)
            out[:, 1] = y_center + dev
            return out

        stage1_ref = resample_polyline(stage1_ctrl, self.stage1_speed_max * self.dt)
        stage1 = optimize_trajectory(
            stage1_ref,
            dt=self.dt,
            v_max=self.stage1_speed_max,
            a_max=self.stage1_accel_max,
            projector=stage1_projector,
        )
        true_tau = int(len(stage1) - 1)

        u2 = np.linspace(0.0, 1.0, 32)
        base2 = p_sub[None, :] + u2[:, None] * (p_goal - p_sub)[None, :]
        direction = p_goal - p_sub
        xy_dir = direction[:2]
        xy_norm = np.linalg.norm(xy_dir) + 1e-12
        normal_xy = np.array([-xy_dir[1], xy_dir[0]]) / xy_norm
        normal = np.array([normal_xy[0], normal_xy[1], 0.0], dtype=float)
        wobble_amp = 0.05 * np.linalg.norm(direction) * rng.uniform(0.2, 0.7)
        wobble = wobble_amp * np.sin(np.pi * u2) * np.sin(2.0 * np.pi * u2 + rng.uniform(0.0, 2.0 * np.pi))
        stage2_ctrl = base2 + wobble[:, None] * normal[None, :]
        stage2_ctrl[0] = p_sub
        stage2_ctrl[-1] = p_goal
        stage2_ref = resample_polyline(stage2_ctrl, self.stage2_speed_max * self.dt)
        stage2 = optimize_trajectory(
            stage2_ref,
            dt=self.dt,
            v_max=self.stage2_speed_max,
            a_max=self.stage2_accel_max,
            projector=None,
        )

        traj = np.vstack([stage1, stage2[1:]])
        return traj, int(true_tau)

    def generate_demos(
        self,
        n_demos=12,
        n1=40,
        rng=None,
        **kwargs,
    ):
        rng = np.random if rng is None else rng
        demos = []
        true_taus = []
        for _ in range(n_demos):
            X, tau_true = self.rollout_demo(
                n1=n1,
                rng=rng,
            )
            demos.append(X)
            true_taus.append(int(tau_true))
        return demos, true_taus

    def compute_all_features_matrix(self, X):
        X = np.asarray(X, float)
        T = len(X)
        if X.shape[1] != 3:
            raise ValueError("SineCorridorEnv3D expects trajectories with shape (T, 3).")

        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]

        y_c = self.centerline(x)
        dist_center = np.abs(y - y_c)

        V = np.zeros(T, dtype=float)
        if T > 1:
            dX = X[1:] - X[:-1]
            V[:-1] = np.linalg.norm(dX, axis=1) / self.dt
            V[-1] = V[-2]

        far_center_xy = np.array((-0.5, 2.5), dtype=float)
        d_far = np.linalg.norm(X[:, :2] - far_center_xy[None, :], axis=1) * 0.2

        phase = float(0.7 * np.mean(x) - 0.4 * np.mean(y) + 0.2 * np.mean(z))
        noise_feat = 0.1 * np.sin(4.0 * x + 1.5 * z + phase)

        return np.stack([dist_center, V, d_far, noise_feat], axis=1)

    def compute_features_all(self, X):
        F = self.compute_all_features_matrix(X)
        return F[:, 0], F[:, 1]

def load_3d_sine_corridor(
    n_demos: int = 12,
    seed: int = 0,
    env_kwargs=None,
    demo_kwargs=None,
) -> TaskBundle:
    env_cfg = {
        "A": 0.1,
        "omega": 10.0,
        "bias": 0.2,
        "phase": 0.0,
        "x_start_range": (-2.0, -1.0),
        "z_start_range": (0.0, 0.5),
        "x_sub": 0.0,
        "z_sub": 0.6,
        "goal": (0.6, 0.0, 0.0),
        "corridor_half_width": 0.2,
        "centerline_tol": 0.03,
        "stage1_speed_max": 0.08,
        "stage2_speed_max": 0.05,
        "stage1_accel_max": 0.04,
        "stage2_accel_max": 0.03,
        "dt": 1.0,
    }
    if env_kwargs:
        env_cfg.update(env_kwargs)

    run_kwargs = {
        "n_demos": n_demos,
        "n1": 40,
    }
    if demo_kwargs:
        run_kwargs.update(demo_kwargs)

    rng = np.random.RandomState(seed)
    env = SineCorridorEnv3D(**env_cfg)
    demos, true_taus = env.generate_demos(rng=rng, **run_kwargs)
    return TaskBundle(
        name="3DSineCorridor",
        demos=demos,
        env=env,
        true_taus=[int(t) for t in true_taus],
        feature_schema=env.get_feature_schema(),
        true_constraints=dict(env.true_constraints),
        constraint_specs=env.get_constraint_specs(),
        meta={"seed": seed, "task_name": "3DSineCorridor"},
    )


def main():
    env = SineCorridorEnv3D()
    demos, true_taus = env.generate_demos(n_demos=5)
    for i, X in enumerate(demos):
        print(f"Demo {i}: shape = {X.shape}, true_tau = {true_taus[i]}")
        F = env.compute_all_features_matrix(X)
        print(f"  Features shape: {F.shape}")
        print(f"  First 5 feature rows:\n{F[:5]}")


if __name__ == "__main__":
    main()
