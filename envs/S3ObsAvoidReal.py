from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .base import TaskBundle
from .rendering import render_planar_episode


_DEFAULT_DATA = Path(__file__).resolve().parent / "demo_data" / "S3ObsAvoidReal.npz"


class S3ObsAvoidRealEnv:
    """Three-stage planar task reconstructed from the real IIWA recording."""

    def __init__(self, data_path=None, **overrides):
        self.data_path = Path(data_path) if data_path is not None else _DEFAULT_DATA
        with np.load(self.data_path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
        self.metadata = metadata
        self.dt = float(overrides.pop("dt", metadata["dt_s"]))
        obstacle = metadata["obstacle"]
        line1 = metadata["line_1"]
        line2 = metadata["line_2"]
        self.obs_center = np.asarray(overrides.pop("obs_center", obstacle["center"]), dtype=float)
        self.obs_radius = float(overrides.pop("obs_radius", obstacle["radius_m"]))
        self.line1_point = np.asarray(overrides.pop("line1_point", line1["point"]), dtype=float)
        self.line1_direction = self._unit(overrides.pop("line1_direction", line1["direction"]))
        self.line2_point = np.asarray(overrides.pop("line2_point", line2["point"]), dtype=float)
        self.line2_direction = self._unit(overrides.pop("line2_direction", line2["direction"]))
        if overrides:
            raise TypeError(f"Unknown S3ObsAvoidReal options: {sorted(overrides)}")
        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self.hide_true_stage_end_markers = True

    @staticmethod
    def _unit(value):
        vector = np.asarray(value, dtype=float).reshape(2)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    @staticmethod
    def _line_distance(points, point, direction):
        normal = np.array([-direction[1], direction[0]], dtype=float)
        return np.abs((np.asarray(points, dtype=float) - point[None, :]) @ normal)

    def get_feature_schema(self):
        return [
            {"id": 0, "column_idx": 0, "name": "obs_dist", "description": "2D distance to estimated circular obstacle center"},
            {"id": 1, "column_idx": 1, "name": "speed", "description": "2D end-effector speed magnitude"},
            {"id": 2, "column_idx": 2, "name": "line1_dist", "description": "Perpendicular 2D distance to estimated stage-2 line"},
            {"id": 3, "column_idx": 3, "name": "line2_dist", "description": "Perpendicular 2D distance to estimated stage-3 line"},
        ]

    def get_true_constraints(self):
        return {
            "obstacle_radius": float(self.obs_radius),
            "line1_distance_target": 0.0,
            "line2_distance_target": 0.0,
        }

    def get_constraint_specs(self):
        return [
            {"feature_name": "obs_dist", "stage": 0, "semantics": "lower_bound", "oracle_key": "obstacle_radius"},
            {"feature_name": "line1_dist", "stage": 1, "semantics": "target_value", "oracle_key": "line1_distance_target"},
            {"feature_name": "line2_dist", "stage": 2, "semantics": "target_value", "oracle_key": "line2_distance_target"},
        ]

    def get_observation_spec(self):
        return {
            "feature_schema": self.get_feature_schema(),
            "source": "real_rosbag",
            "coordinate_system": self.metadata["coordinate_system"],
            "geometry_estimation_status": self.metadata["geometry_estimation_status"],
        }

    def get_render_camera_presets(self):
        return {"default_planar": {"projection": "orthographic_like_2d", "xlabel": "x [m]", "ylabel": "y [m]", "equal_aspect": True}}

    def get_asset_handles(self):
        return {"estimated_obstacle": {"type": "circle"}, "track_lines": [{"type": "line"}, {"type": "line"}]}

    def get_true_reference_lines(self):
        """True planar track geometry consumed by trajectory/debug plots."""
        return [
            {"name": "true line 1", "point": self.line1_point.copy(),
             "direction": self.line1_direction.copy(), "color": "#2563EB"},
            {"name": "true line 2", "point": self.line2_point.copy(),
             "direction": self.line2_direction.copy(), "color": "#059669"},
        ]

    def sample_scene(self, seed=None, rng=None):
        return {
            "task_name": "S3ObsAvoidReal",
            "geometry": {
                "obs_center": self.obs_center.tolist(), "obs_radius": float(self.obs_radius),
                "line1_point": self.line1_point.tolist(), "line1_direction": self.line1_direction.tolist(),
                "line2_point": self.line2_point.tolist(), "line2_direction": self.line2_direction.tolist(),
            },
            "task": {"dt": float(self.dt), "geometry_is_estimated": True},
        }

    def compute_all_features_matrix(self, traj, feat_ids=None):
        points = np.asarray(traj, dtype=float)[:, :2]
        speed = np.zeros(len(points), dtype=float)
        if len(points) > 1:
            edge_speed = np.linalg.norm(np.diff(points, axis=0), axis=1) / float(self.dt)
            speed[0] = edge_speed[0]
            speed[1:] = edge_speed
        features = np.column_stack([
            np.linalg.norm(points - self.obs_center[None, :], axis=1),
            speed,
            self._line_distance(points, self.line1_point, self.line1_direction),
            self._line_distance(points, self.line2_point, self.line2_direction),
        ])
        return features if feat_ids is None else features[:, feat_ids]

    def compute_features_all(self, traj):
        features = self.compute_all_features_matrix(traj)
        return tuple(features[:, index] for index in range(features.shape[1]))

    def render_episode(self, scene, trajectory, output_path, **kwargs):
        return render_planar_episode(
            trajectory=np.asarray(trajectory, dtype=float)[:, :2], output_path=output_path,
            cutpoints=kwargs.get("cutpoints"), title=kwargs.get("title", "S3ObsAvoidReal episode"),
            obstacles=[{"center": self.obs_center.tolist(), "radius": float(self.obs_radius),
                        "facecolor": "#CBD5E1", "edgecolor": "#475569", "alpha": 0.38}],
            reference_lines=[
                {"point": self.line1_point.tolist(), "direction": self.line1_direction.tolist(), "color": "#64748B"},
                {"point": self.line2_point.tolist(), "direction": self.line2_direction.tolist(), "color": "#64748B"},
            ], xlabel="x [m]", ylabel="y [m]", equal_aspect=True,
        )


def load_S3ObsAvoidReal(n_demos=4, seed=0, env_kwargs=None, demo_kwargs=None, **extra_env_kwargs):
    env_cfg = dict(env_kwargs or {})
    env_cfg.update(extra_env_kwargs)
    env = S3ObsAvoidRealEnv(**env_cfg)
    demos, times, cutpoints, labels = [], [], [], []
    with np.load(env.data_path, allow_pickle=False) as data:
        available = int(data["count"].item())
        if int(n_demos) > available:
            raise ValueError(f"S3ObsAvoidReal contains {available} demonstrations, requested {n_demos}")
        for index in range(int(n_demos)):
            demos.append(np.asarray(data[f"demo_{index}"], dtype=float))
            times.append(np.asarray(data[f"time_{index}"], dtype=float))
            cutpoints.append(np.asarray(data[f"cutpoints_{index}"], dtype=int))
            labels.append(np.asarray(data[f"labels_{index}"], dtype=int))
    scenes = []
    for index in range(len(demos)):
        scene = env.sample_scene()
        scene["demo_index"] = index
        scenes.append(scene)
    return TaskBundle(
        name="S3ObsAvoidReal", demos=demos, env=env, true_cutpoints=cutpoints,
        true_labels=labels, feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(), constraint_specs=env.get_constraint_specs(),
        meta={"seed": int(seed), "task_name": "S3ObsAvoidReal", "times": times,
              "source_metadata": env.metadata, "scene_specs": scenes,
              "observation_specs": env.get_observation_spec(),
              "render_camera_presets": env.get_render_camera_presets(), "asset_handles": env.get_asset_handles()},
    )
