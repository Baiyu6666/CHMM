from __future__ import annotations

import numpy as np

from .base import TaskBundle


class Line2DEnv:
    def __init__(
        self,
        horizon: int = 120,
        n_segments: int = 3,
        speed_variability: float = 0.0,
        noise_scale: float = 0.02,
        smooth_coeff: float = 0.4,
        dt: float = 1.0,
    ):
        self.horizon = int(horizon)
        self.n_segments = int(n_segments)
        self.speed_variability = float(speed_variability)
        self.noise_scale = float(noise_scale)
        self.smooth_coeff = float(smooth_coeff)
        self.dt = float(dt)
        self.eval_tag = "2DLine"
        self.subgoal = None
        self.goal = None
        self.true_constraints = None
        self.feature_schema = self.get_feature_schema()

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "x", "description": "x coordinate"},
            {"id": 1, "name": "y", "description": "y coordinate"},
            {"id": 2, "name": "dist_center", "description": "Distance to trajectory mean center"},
            {"id": 3, "name": "speed", "description": "2D speed magnitude"},
            {"id": 4, "name": "noise_aux", "description": "Deterministic auxiliary sinusoidal feature"},
        ]

    def generate_demo(self, seed: int, waypoints=None):
        rng = np.random.RandomState(seed)
        if waypoints is None:
            n_waypoints = self.n_segments + 1
            xs = np.arange(n_waypoints, dtype=float)
            ys = (np.arange(n_waypoints) % 2).astype(float)
            waypoints = np.stack([xs, ys], axis=1)
        else:
            waypoints = np.asarray(waypoints, float)

        n_segments = waypoints.shape[0] - 1
        alpha = 10.0 / (1.0 + max(0.0, self.speed_variability))
        proportions = rng.dirichlet(np.full(n_segments, alpha)) if n_segments > 1 else np.array([1.0])
        seg_lengths = np.maximum(1, np.round(self.horizon * proportions).astype(int))

        diff = self.horizon - int(seg_lengths.sum())
        order = np.argsort(-proportions)
        i = 0
        while diff != 0:
            j = order[i % n_segments]
            if diff > 0:
                seg_lengths[j] += 1
                diff -= 1
            elif seg_lengths[j] > 1:
                seg_lengths[j] -= 1
                diff += 1
            i += 1

        segments = []
        for s in range(n_segments):
            start = waypoints[s]
            end = waypoints[s + 1]
            steps = seg_lengths[s]
            if s < n_segments - 1:
                seg = np.linspace(start, end, steps, endpoint=False)
            else:
                seg = np.linspace(start, end, steps, endpoint=True)
            segments.append(seg)

        traj = np.vstack(segments)
        eps = rng.randn(*traj.shape) * self.noise_scale
        smooth = np.cumsum(eps, axis=0)
        ramp = np.linspace(0, 1, traj.shape[0])[:, None]
        smooth -= ramp * smooth[-1]
        noisy = traj + self.smooth_coeff * smooth
        noisy[0] = traj[0]
        noisy[-1] = traj[-1]

        cuts = np.linspace(0, self.horizon, self.n_segments + 1, dtype=int)[1:] - 1
        labels = np.zeros(self.horizon, int)
        start = 0
        for k, t_end in enumerate(cuts):
            labels[start : t_end + 1] = k
            start = t_end + 1
        return noisy, labels

    def compute_all_features_matrix(self, traj: np.ndarray, feat_ids=None) -> np.ndarray:
        traj = np.asarray(traj, dtype=float)
        T = traj.shape[0]
        speed = np.zeros(T, dtype=float)
        if T > 1:
            speed_edge = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.dt
            speed[0] = speed_edge[0]
            speed[1:] = speed_edge
        dist_to_center = np.linalg.norm(traj - np.mean(traj, axis=0, keepdims=True), axis=1)
        phase = np.linspace(0.0, 2.0 * np.pi, T)
        noise = 0.2 * np.sin(3.0 * phase)
        F = np.concatenate([traj, dist_to_center[:, None], speed[:, None], noise[:, None]], axis=1)
        return F if feat_ids is None else F[:, feat_ids]

    def compute_features_all(self, traj: np.ndarray):
        F = self.compute_all_features_matrix(traj)
        return F[:, -3], F[:, -2]


def load_line_2d(
    n_demos: int = 8,
    horizon: int = 120,
    n_segments: int = 3,
    seed: int = 2025,
    env_kwargs=None,
    demo_kwargs=None,
) -> TaskBundle:
    env_cfg = {
        "horizon": horizon,
        "n_segments": n_segments,
    }
    if env_kwargs:
        env_cfg.update(env_kwargs)

    run_kwargs = {}
    if demo_kwargs:
        run_kwargs.update(demo_kwargs)

    env = Line2DEnv(**env_cfg)
    demos = []
    labels = []
    for i in range(n_demos):
        demo, z = env.generate_demo(seed=seed + i, **run_kwargs)
        demos.append(np.asarray(demo, dtype=float))
        labels.append(np.asarray(z, dtype=int))

    env.subgoal = np.mean([x[len(x) // 2] for x in demos], axis=0)
    env.goal = np.mean([x[-1] for x in demos], axis=0)
    cutpoints = [np.where(np.diff(z) != 0)[0] for z in labels]
    true_taus = [int(c[0]) for c in cutpoints] if n_segments == 2 else None
    return TaskBundle(
        name="2DLine",
        demos=demos,
        env=env,
        true_taus=true_taus,
        true_cutpoints=[np.asarray(c, dtype=int) for c in cutpoints],
        true_labels=labels,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.true_constraints,
        constraint_specs=getattr(env, "constraint_specs", None),
        meta={"seed": seed, "cutpoints": [c.tolist() for c in cutpoints], "task_name": "2DLine"},
    )
