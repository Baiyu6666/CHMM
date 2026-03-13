from __future__ import annotations

import numpy as np

from .base import TaskBundle


class PickPlaceEnv:
    def __init__(
        self,
        seg_lengths=(50, 10, 80, 10, 30),
        noise_pos: float = 0.01,
        noise_misc: float = 0.02,
        dt: float = 1.0,
    ):
        self.seg_lengths = tuple(seg_lengths)
        self.noise_pos = float(noise_pos)
        self.noise_misc = float(noise_misc)
        self.dt = float(dt)
        self.eval_tag = "PickPlace"
        self.subgoal = None
        self.goal = None
        self.true_constraints = None

    def generate_demo(
        self,
        seed: int,
        x_start=0.0,
        z_start=0.5,
        x_pick=1.0,
        z_pick=0.0,
        x_place=2.5,
        z_place=0.0,
        x_retreat=2.2,
        z_retreat=0.5,
    ):
        rng = np.random.RandomState(seed)
        seg_lengths = np.array(self.seg_lengths, int)
        horizon = int(seg_lengths.sum())
        x = np.zeros(horizon)
        z = np.zeros(horizon)
        grip = np.zeros(horizon)
        force = np.zeros(horizon)

        t0 = 0
        length = seg_lengths[0]
        x[t0:t0 + length] = np.linspace(x_start, x_pick, length, endpoint=False)
        z[t0:t0 + length] = np.linspace(z_start, z_pick, length, endpoint=False)
        grip[t0:t0 + length] = 1.0
        t0 += length

        length = seg_lengths[1]
        x[t0:t0 + length] = x_pick
        z[t0:t0 + length] = z_pick
        grip[t0:t0 + length] = np.linspace(1.0, 0.0, length, endpoint=False)
        force[t0:t0 + length] = np.linspace(0.0, 1.0, length)
        t0 += length

        length = seg_lengths[2]
        x[t0:t0 + length] = np.linspace(x_pick, x_place, length, endpoint=False)
        z[t0:t0 + length] = np.linspace(z_pick, z_place, length, endpoint=False)
        grip[t0:t0 + length] = 0.0
        force[t0:t0 + length] = 0.15
        t0 += length

        length = seg_lengths[3]
        x[t0:t0 + length] = x_place
        z[t0:t0 + length] = z_place
        grip[t0:t0 + length] = np.linspace(0.0, 1.0, length, endpoint=False)
        force[t0:t0 + length] = np.linspace(0.15, 0.0, length)
        t0 += length

        length = seg_lengths[4]
        x[t0:t0 + length] = np.linspace(x_place, x_retreat, length, endpoint=True)
        z[t0:t0 + length] = np.linspace(z_place, z_retreat, length, endpoint=True)
        grip[t0:t0 + length] = 1.0
        force[t0:t0 + length] = 0.0

        x += rng.randn(*x.shape) * self.noise_pos
        z += rng.randn(*z.shape) * self.noise_pos
        grip = np.clip(grip + rng.randn(*grip.shape) * self.noise_misc, 0.0, 1.0)
        force = np.clip(force + rng.randn(*force.shape) * self.noise_misc, 0.0, None)

        demo = np.stack([x, z, grip, force], axis=1)
        demo = np.vstack([demo, demo[-1][None, :]])
        labels = np.repeat(np.arange(len(seg_lengths)), seg_lengths)
        return demo, labels

    def compute_all_features_matrix(self, traj: np.ndarray, feat_ids=None) -> np.ndarray:
        traj = np.asarray(traj, dtype=float)
        T = traj.shape[0]
        speed = np.zeros(T, dtype=float)
        if T > 1:
            speed_edge = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1) / self.dt
            speed[0] = speed_edge[0]
            speed[1:] = speed_edge
        position_radius = np.linalg.norm(traj[:, :2], axis=1)
        F = np.concatenate([traj, position_radius[:, None], speed[:, None]], axis=1)
        return F if feat_ids is None else F[:, feat_ids]

    def compute_features_all(self, traj: np.ndarray):
        F = self.compute_all_features_matrix(traj)
        return F[:, -2], F[:, -1]


def load_pick_place(
    n_demos: int = 6,
    seg_lengths=(50, 10, 80, 10, 30),
    seed: int = 100,
) -> TaskBundle:
    env = PickPlaceEnv(seg_lengths=seg_lengths)
    demos = []
    labels = []
    for i in range(n_demos):
        demo, z = env.generate_demo(seed=seed + i)
        demos.append(np.asarray(demo, dtype=float))
        labels.append(np.asarray(z, dtype=int))

    seg_lengths = np.array(seg_lengths, dtype=int)
    pick_idx = int(seg_lengths[0] - 1)
    place_idx = int(seg_lengths[:3].sum() - 1)
    retreat_idx = int(sum(seg_lengths))
    env.pick_point = np.mean([x[pick_idx, :2] for x in demos], axis=0)
    env.place_point = np.mean([x[place_idx, :2] for x in demos], axis=0)
    env.retreat_point = np.mean([x[retreat_idx, :2] for x in demos], axis=0)
    env.subgoal = np.mean([x[len(x) // 2] for x in demos], axis=0)
    env.goal = np.mean([x[-1] for x in demos], axis=0)
    cutpoints = [np.where(np.diff(z) != 0)[0] for z in labels]
    return TaskBundle(
        name="PickPlace",
        demos=demos,
        env=env,
        true_taus=None,
        true_labels=labels,
        meta={"seed": seed, "cutpoints": [c.tolist() for c in cutpoints], "task_name": "PickPlace"},
    )
