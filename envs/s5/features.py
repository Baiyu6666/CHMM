from __future__ import annotations

import numpy as np


class S5FeatureMixin:
    @staticmethod
    def _traj_cache_key(traj: np.ndarray):
        arr = np.ascontiguousarray(np.asarray(traj, dtype=np.float64))
        return arr.shape, arr.tobytes()

    @staticmethod
    def _normalize_observation_backend(name) -> str:
        text = str(name).lower()
        if text == "analytic":
            return "analytic_raw"
        if text == "raw":
            return "analytic_raw"
        return text

    @staticmethod
    def _normalize_goal_dist_mode(name) -> str:
        text = str(name).strip().lower()
        aliases = {
            "nominal": "nominal_shared",
            "fixed": "nominal_shared",
            "fixed_nominal": "nominal_shared",
            "shared_nominal": "nominal_shared",
            "final": "demo_goal",
            "endpoint": "demo_goal",
            "demo_endpoint": "demo_goal",
            "demo_final": "demo_goal",
            "goal": "demo_goal",
        }
        normalized = aliases.get(text, text)
        if normalized not in {"nominal_shared", "demo_goal"}:
            raise ValueError(
                "Unsupported S5 goal_dist_mode "
                f"'{name}'. Expected 'nominal_shared' or 'demo_goal'."
            )
        return normalized

    def register_tool_axis_trace(self, traj: np.ndarray, tool_axis: np.ndarray):
        self._cached_tool_axis_traces[self._traj_cache_key(traj)] = np.asarray(tool_axis, dtype=float).copy()

    def register_goal_position(self, traj: np.ndarray, goal_position: np.ndarray):
        self._cached_goal_positions[self._traj_cache_key(traj)] = np.asarray(goal_position, dtype=float).reshape(3).copy()

    def register_timestamp_trace(self, traj: np.ndarray, timestamps: np.ndarray):
        values = np.asarray(timestamps, dtype=float).reshape(-1)
        if len(values) != len(traj):
            raise ValueError("S5 timestamps must align with the demonstration trajectory.")
        self._cached_timestamp_traces[self._traj_cache_key(traj)] = values.copy()

    def register_feature_trace(self, traj: np.ndarray, features: np.ndarray):
        matrix = np.asarray(features, dtype=float)
        if matrix.ndim != 2 or len(matrix) != len(traj):
            raise ValueError("Stored S5 features must be a 2D matrix aligned with the demonstration.")
        self._cached_feature_traces[self._traj_cache_key(traj)] = matrix.copy()

    def _lookup_cached_tool_axis_trace(self, traj: np.ndarray):
        axis = self._cached_tool_axis_traces.get(self._traj_cache_key(traj))
        if axis is None:
            return None
        return np.asarray(axis, dtype=float)

    def _lookup_cached_goal_position(self, traj: np.ndarray):
        goal_position = self._cached_goal_positions.get(self._traj_cache_key(traj))
        if goal_position is None:
            return None
        return np.asarray(goal_position, dtype=float)

    def _lookup_cached_timestamp_trace(self, traj: np.ndarray):
        timestamps = self._cached_timestamp_traces.get(self._traj_cache_key(traj))
        if timestamps is None:
            return None
        return np.asarray(timestamps, dtype=float)

    def _lookup_cached_feature_trace(self, traj: np.ndarray):
        features = self._cached_feature_traces.get(self._traj_cache_key(traj))
        if features is None:
            return None
        return np.asarray(features, dtype=float)

    def get_feature_schema(self):
        goal_description = (
            "3D distance to this demonstration's assigned goal position"
            if self.goal_dist_mode == "demo_goal"
            else "3D distance to the shared nominal inspection goal"
        )
        return [
            {"id": 0, "name": "surf_dist", "description": "Absolute radial distance to the sphere surface"},
            {"id": 1, "name": "normal_err", "description": "Angle between tool axis and sphere normal"},
            {"id": 2, "name": "speed", "description": "3D speed magnitude"},
            {"id": 3, "name": "ang_speed", "description": "Tool-axis angular speed magnitude"},
            {"id": 4, "name": "noise", "description": "Deterministic auxiliary irrelevant feature"},
            {"id": 5, "name": "start_dist", "description": "3D distance to the demo start position"},
            {"id": 6, "name": "goal_dist", "description": goal_description},
        ]

    def get_true_constraints(self):
        base = {
            "surface_trace_target": 0.0,
            "surface_near_target": float(self.surface_near_target_ratio * self.shell_thickness),
            "tool_align_max_stage2": float(self.tool_align_max_stage2),
            "v23_max": float(self.stage2_speed_max),
        }
        return base

    def get_constraint_specs(self):
        return [
            {"feature_name": "surf_dist", "stage": 1, "semantics": "target_value", "oracle_key": "surface_trace_target"},
            {"feature_name": "normal_err", "stage": 1, "semantics": "upper_bound", "oracle_key": "tool_align_max_stage2"},
            {"feature_name": "speed", "stage": 1, "semantics": "upper_bound", "oracle_key": "v23_max"},
            {"feature_name": "surf_dist", "stage": 3, "semantics": "target_value", "oracle_key": "surface_near_target"},
            {"feature_name": "speed", "stage": 3, "semantics": "upper_bound", "oracle_key": "v23_max"},
        ]

    def get_observation_spec(self):
        return {
            "feature_schema": self.get_feature_schema(),
            "goal_dist": {
                "mode": str(self.goal_dist_mode),
                "demo_goal_source": "stored per-demo goal_position",
                "nominal_goal": np.asarray(self.goal, dtype=float).tolist(),
            },
            "default_rollout_backend": str(self.rollout_backend),
            "default_observation_backend": str(self.observation_backend),
            "noise_model": {
                "trajectory_noise_std": float(self.noise_std),
                "cached_feature_trace": False,
            },
            "pybullet_rollout": {
                "enabled": bool(self.rollout_backend == "pybullet" or self.observation_backend == "pybullet"),
                "backend": "ur5_ik_position_control",
                "sim_dt": float(self.pybullet_sim_dt),
                "steps_per_sample": None if self.pybullet_steps_per_sample is None else int(self.pybullet_steps_per_sample),
                "world_scale": float(self.pybullet_world_scale),
                "world_center": list(self.pybullet_world_center),
                "ur5_ee_link_index": int(self.pybullet_ur5_ee_link_index),
                "ur5_tool_axis": str(self.pybullet_ur5_tool_axis),
                "ur5_tip_offset": float(self.pybullet_ur5_tip_offset),
                "ur5_base_xyz": list(self.pybullet_ur5_base_xyz),
                "ur5_base_rpy": list(self.pybullet_ur5_base_rpy),
                "suppress_urdf_warnings": bool(self.pybullet_suppress_urdf_warnings),
                "ik_filter": {
                    "enabled": bool(self.pybullet_filter_ik_valid),
                    "max_attempts": int(self.pybullet_filter_max_attempts),
                    "max_position_error": float(self.pybullet_filter_max_position_error),
                    "max_axis_error": float(self.pybullet_filter_max_axis_error),
                    "global_axis_error": bool(self.pybullet_filter_global_axis_error),
                    "constrained_max_axis_error": float(self.pybullet_filter_constrained_max_axis_error),
                    "max_speed_ratio": float(self.pybullet_filter_max_speed_ratio),
                    "precheck_ik_waypoints": bool(self.pybullet_precheck_ik_waypoints),
                    "precheck_points_per_stage": int(self.pybullet_precheck_points_per_stage),
                },
            },
        }

    def get_render_camera_presets(self):
        return {
            "default_3d": {
                "backend": "matplotlib",
                "elev": 24.0,
                "azim": 38.0,
            },
            "paper_orbit": {
                "backend": "pybullet",
                "main_yaw": 42.0,
                "inset_yaw": 205.0,
            },
        }

    def get_asset_handles(self):
        return {
            "sphere_surface": {"type": "sphere"},
            "ur5": {"type": "robot_arm", "model": "UR5+hidden_gripper"},
            "visible_ee": {"type": "urdf_task_tool_link", "normal_axis": "local_-x"},
            "reference_table": {"type": "tabletop"},
        }

    def _assemble_feature_matrix(self, traj, *, tool_axis=None, use_cached=True):
        traj = np.asarray(traj, dtype=float)
        surf_dist, normal_err, speed, ang_speed = self._compute_geometry_feature_traces(traj, tool_axis=tool_axis)

        return {
            "surf_dist": np.asarray(surf_dist, dtype=float),
            "normal_err": np.asarray(normal_err, dtype=float),
            "speed": np.asarray(speed, dtype=float),
            "ang_speed": np.asarray(ang_speed, dtype=float),
        }

    def compute_all_features_matrix(
        self,
        traj,
        feat_ids=None,
        *,
        tool_axis=None,
        goal_position=None,
        use_cached=None,
    ):
        traj = np.asarray(traj, dtype=float)
        if use_cached is not False:
            cached_features = self._lookup_cached_feature_trace(traj)
            if cached_features is not None:
                return cached_features if feat_ids is None else cached_features[:, feat_ids]
        T = len(traj)
        base = self._assemble_feature_matrix(traj, tool_axis=tool_axis, use_cached=False)
        surf_dist = np.asarray(base["surf_dist"], dtype=float)
        normal_err = np.asarray(base["normal_err"], dtype=float)
        speed = np.asarray(base["speed"], dtype=float)
        ang_speed = np.asarray(base["ang_speed"], dtype=float)

        t = np.linspace(0.0, 2.0 * np.pi, T)
        phase = float(0.31 * np.mean(traj[:, 0]) - 0.27 * np.mean(traj[:, 1]) + 0.43 * np.mean(traj[:, 2]))
        noise = 0.15 * np.sin(4.3 * t + phase) + 0.08 * np.cos(1.7 * t - 0.5 * phase)
        start_dist = np.linalg.norm(traj - traj[0:1], axis=1)
        if self.goal_dist_mode == "demo_goal":
            goal_point = goal_position
            if goal_point is None:
                goal_point = self._lookup_cached_goal_position(traj)
            if goal_point is None:
                goal_point = traj[-1]
            goal_point = np.asarray(goal_point, dtype=float).reshape(3)
        else:
            goal_point = self.goal
        goal_dist = np.linalg.norm(traj - goal_point[None, :], axis=1)

        F = np.stack(
            [
                surf_dist,
                normal_err,
                speed,
                ang_speed,
                noise,
                start_dist,
                goal_dist,
            ],
            axis=1,
        )
        return F if feat_ids is None else F[:, feat_ids]

    def compute_observation(self, latent_rollout, scene, backend=None):
        suggested = latent_rollout.get("observation_backend", self.observation_backend) if backend is None else backend
        active_backend = self._normalize_observation_backend(suggested)
        traj = np.asarray(latent_rollout["trajectory"], dtype=float)
        tool_axis = latent_rollout.get("tool_axis")
        if tool_axis is not None:
            tool_axis = np.asarray(tool_axis, dtype=float)
        goal_position = latent_rollout.get("goal_position")
        if goal_position is None:
            reference_trajectory = latent_rollout.get("reference_trajectory")
            if reference_trajectory is not None and len(reference_trajectory) > 0:
                goal_position = np.asarray(reference_trajectory, dtype=float)[-1]
            else:
                goal_position = traj[-1]
        goal_position = np.asarray(goal_position, dtype=float).reshape(3)
        if active_backend == "analytic_raw":
            features = np.asarray(
                self.compute_all_features_matrix(
                    traj,
                    tool_axis=tool_axis,
                    goal_position=goal_position,
                    use_cached=False,
                ),
                dtype=float,
            )
        elif active_backend == "pybullet":
            features = np.asarray(
                self.compute_all_features_matrix(
                    traj,
                    tool_axis=tool_axis,
                    goal_position=goal_position,
                    use_cached=False,
                ),
                dtype=float,
            )
        else:
            raise ValueError(f"Unsupported S5 observation backend '{active_backend}'.")
        timestamps = np.asarray(
            latent_rollout.get("timestamps", np.arange(len(traj), dtype=float) * float(self.dt)),
            dtype=float,
        ).reshape(-1)
        if len(timestamps) != len(traj):
            raise ValueError("S5 rollout timestamps must align with the observed trajectory.")
        observation = {
            "trajectory": traj,
            "timestamps": timestamps,
            "features": features,
            "true_cutpoints": np.asarray(latent_rollout.get("true_cutpoints", []), dtype=int),
            "feature_schema": self.get_feature_schema(),
            "observation_spec": self.get_observation_spec(),
            "tool_axis": tool_axis,
            "goal_position": goal_position,
            "scene": dict(scene or {}),
            "generation_metadata": dict(latent_rollout.get("generation_metadata", {})),
        }
        self.register_goal_position(traj, goal_position)
        self.register_feature_trace(traj, features)
        for key in (
            "quaternions",
            "linear_velocity",
            "angular_velocity",
            "contact_flags",
            "joint_positions",
            "joint_velocities",
            "joint_position_commands",
            "joint_position_commands_nominal",
            "execution_joint_noise",
            "true_labels",
            "sim_dt",
            "steps_per_sample",
            "reference_trajectory",
            "reference_tool_axis",
            "reference_trajectory_world",
            "target_ee_trajectory_world",
            "realized_trajectory_world",
            "realized_ee_trajectory_world",
            "ik_position_error_world",
            "ik_axis_error",
            "ur5_tool_axis",
            "ur5_tip_offset",
            "robot_backend",
            "reference_seed",
            "ik_filter",
        ):
            if key in latent_rollout:
                observation[key] = latent_rollout.get(key)
        return observation

    def _estimate_tool_axis_from_geometry(self, traj):
        pts = np.asarray(traj, dtype=float)
        rel = pts - self.sphere_center[None, :]
        normals = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
        return normals

    def _compute_geometry_feature_traces(self, traj, tool_axis=None):
        traj = np.asarray(traj, dtype=float)
        T = len(traj)
        rel = traj - self.sphere_center[None, :]
        radial_dist = np.linalg.norm(rel, axis=1)
        surf_dist = np.abs(radial_dist - self.sphere_radius)

        if tool_axis is None:
            tool_axis = self._lookup_cached_tool_axis_trace(traj)
        if tool_axis is None:
            tool_axis = self._estimate_tool_axis_from_geometry(traj)
        tool_axis = np.asarray(tool_axis, dtype=float)
        tool_axis = tool_axis / np.maximum(np.linalg.norm(tool_axis, axis=1, keepdims=True), 1e-12)
        normals = rel / np.maximum(radial_dist[:, None], 1e-12)
        cos_align = np.sum(tool_axis * normals, axis=1)
        cos_align = np.clip(cos_align, -1.0, 1.0)
        normal_err = np.arccos(cos_align)

        speed = np.zeros(T, dtype=float)
        if T > 1:
            speed_edge = np.linalg.norm(np.diff(traj, axis=0), axis=1) / self.dt
            speed[0] = speed_edge[0]
            speed[1:] = speed_edge

        ang_speed = np.zeros(T, dtype=float)
        if T > 1:
            dots = np.sum(tool_axis[1:] * tool_axis[:-1], axis=1)
            dots = np.clip(dots, -1.0, 1.0)
            ang = np.arccos(dots) / self.dt
            ang_speed[0] = ang[0]
            ang_speed[1:] = ang

        return surf_dist, normal_err, speed, ang_speed

    def compute_features_all(self, traj):
        F = self.compute_all_features_matrix(traj)
        return F[:, 0], F[:, 2]
