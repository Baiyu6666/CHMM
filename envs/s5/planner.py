from __future__ import annotations

import numpy as np

from .execution import check_s5_reference_waypoints_ik, simulate_s5_demo_from_reference


class S5ConstraintPlannerMixin:
    @staticmethod
    def _lookup_plan_constraint_value(values, stage_idx: int, feature_name: str, default):
        if values is None:
            return float(default)
        if not isinstance(values, dict):
            return float(default)
        keys = (
            f"s{int(stage_idx) + 1}:{feature_name}",
            f"stage{int(stage_idx) + 1}:{feature_name}",
            f"{int(stage_idx) + 1}:{feature_name}",
            f"{int(stage_idx)}:{feature_name}",
        )
        for key in keys:
            if key in values and values[key] is not None:
                try:
                    value = float(values[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    return value
        return float(default)

    def _clean_axis_near_normals(self, normals, *, max_error: float, fraction: float = 0.0, phase: float = 0.0):
        normals = np.asarray(normals, dtype=float)
        out = np.empty_like(normals)
        angle = float(max(max_error, 0.0)) * float(np.clip(fraction, 0.0, 1.0))
        for i, normal in enumerate(normals):
            normal = self._unit(normal)
            if angle <= 1e-10:
                out[i] = normal
                continue
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(normal, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(normal, ref))
            t2 = self._unit(np.cross(normal, t1))
            theta = float(phase) + 2.0 * np.pi * (float(i) / max(len(normals) - 1, 1))
            tangent = self._unit(np.cos(theta) * t1 + np.sin(theta) * t2)
            out[i] = self._unit(np.cos(angle) * normal + np.sin(angle) * tangent)
        return out

    def _clean_shell_arc_by_speed(self, n_start, n_end, *, radius_offset: float, n_points: int):
        raw = self._make_spherical_shell_path(
            n_start,
            n_end,
            max(int(n_points) * 4, 64),
            radius_offset=float(radius_offset),
            detour_angle=0.0,
        )
        return self._time_parameterize_fixed_count(raw, int(n_points)).positions

    @staticmethod
    def _latitude_delta_for_geodesic_angle(phi: float, angle: float, max_delta: float) -> float:
        sin_phi = max(float(np.sin(float(phi))), 1e-6)
        arg = float(np.sin(0.5 * float(angle))) / sin_phi
        if arg < 1.0:
            delta = 2.0 * float(np.arcsin(np.clip(arg, -1.0, 1.0)))
        else:
            delta = float(max_delta)
        return float(np.clip(delta, 0.0, float(max_delta)))

    def plan_episode_from_constraints(
        self,
        scene,
        constraint_values,
        seed=None,
        *,
        stage_lengths=None,
        speed_safety: float = 1.0,
    ):
        rng = np.random.RandomState(0 if seed is None else int(seed))
        values = dict(constraint_values or {})
        true = dict(self.true_constraints)

        s2_surf = max(0.0, self._lookup_plan_constraint_value(values, 1, "surf_dist", true["surface_trace_target"]))
        s2_normal = max(0.0, self._lookup_plan_constraint_value(values, 1, "normal_err", true["tool_align_max_stage2"]))
        s2_speed = max(1e-5, self._lookup_plan_constraint_value(values, 1, "speed", true["v23_max"]))
        s4_surf = max(0.0, self._lookup_plan_constraint_value(values, 3, "surf_dist", true["surface_near_target"]))
        s4_speed = max(1e-5, self._lookup_plan_constraint_value(values, 3, "speed", true["v23_max"]))

        base_lengths = [int(x) for x in self.seg_lengths]
        while len(base_lengths) < 4:
            base_lengths.append(base_lengths[-1] if base_lengths else 18)
        lengths = {
            "stage1": int(base_lengths[0]),
            "stage2": int(base_lengths[1]),
            "stage3": int(base_lengths[2]),
            "stage4": int(base_lengths[3]),
            "stage5": int(base_lengths[3]),
        }
        if stage_lengths is not None:
            for key, value in dict(stage_lengths).items():
                if key in lengths:
                    lengths[key] = int(max(int(value), 4))
        l1 = max(lengths["stage1"], 4)
        l2 = max(lengths["stage2"], 8)
        l3 = max(lengths["stage3"], 8)
        l4 = max(lengths["stage4"], 8)
        l5 = max(lengths["stage5"], 6)

        r2 = float(self.sphere_radius + s2_surf)
        r4 = float(self.sphere_radius + s4_surf)
        speed_safety = float(np.clip(speed_safety, 0.10, 1.0))
        phi_lo, phi_hi = self.contact_phi_range
        phi0 = float(np.clip(0.5 * (phi_lo + phi_hi) + rng.uniform(-0.025 * np.pi, 0.025 * np.pi), phi_lo, phi_hi))
        lateral_sign = -1.0 if float(rng.rand()) < 0.5 else 1.0

        stage2_length = speed_safety * float(s2_speed) * float(self.dt) * max(l2 - 1, 1)
        stage2_angle = float(np.clip(stage2_length / max(r2, 1e-8), 0.28, 1.28))
        theta_center = float(self.stage2_lateral_center_theta)
        delta_theta2 = self._latitude_delta_for_geodesic_angle(phi0, stage2_angle, 0.92 * np.pi)
        theta0 = theta_center - 0.5 * lateral_sign * delta_theta2
        theta1 = theta_center + 0.5 * lateral_sign * delta_theta2
        n0 = self._normal_from_spherical(theta0, phi0)
        n1 = self._normal_from_spherical(theta1, phi0)

        top_phi_lo, top_phi_hi = self.stage345_top_phi_range
        phi_top = float(np.clip(0.5 * (top_phi_lo + top_phi_hi) + rng.uniform(-0.018 * np.pi, 0.018 * np.pi), top_phi_lo, top_phi_hi))
        stage4_length = speed_safety * float(s4_speed) * float(self.dt) * max(l4 - 1, 1)
        stage4_angle = float(np.clip(stage4_length / max(r4, 1e-8), 0.18, 0.72))
        theta_top = theta_center + rng.uniform(-0.045 * np.pi, 0.045 * np.pi)
        delta_theta4 = self._latitude_delta_for_geodesic_angle(phi_top, stage4_angle, 0.72 * np.pi)
        theta4_start = theta_top - 0.5 * lateral_sign * delta_theta4
        theta4_end = theta_top + 0.5 * lateral_sign * delta_theta4
        n4_start = self._normal_from_spherical(theta4_start, phi_top)
        n4_end = self._normal_from_spherical(theta4_end, phi_top)

        p_contact = self.sphere_center + r2 * n0
        p_start = self.sphere_center + (r2 + self.approach_offset) * n0
        stage1_ctrl = np.vstack([p_start, 0.35 * p_start + 0.65 * p_contact, p_contact])
        stage1 = self._time_parameterize_fixed_count(stage1_ctrl, l1).positions
        stage2 = self._clean_shell_arc_by_speed(n0, n1, radius_offset=s2_surf, n_points=l2)

        stage3_normals = self._slerp_unit(n1, n4_start, l3, endpoint=True)
        u3 = np.linspace(0.0, 1.0, l3, endpoint=True)
        stage3_radius = (1.0 - u3) * r2 + u3 * r4
        stage3 = self.sphere_center[None, :] + stage3_radius[:, None] * stage3_normals

        stage4 = self._clean_shell_arc_by_speed(n4_start, n4_end, radius_offset=s4_surf, n_points=l4)
        p_depart = self.sphere_center + (r4 + self.depart_offset) * n4_end
        stage5_ctrl = np.vstack([stage4[-1], 0.55 * stage4[-1] + 0.45 * p_depart, p_depart])
        stage5 = self._time_parameterize_fixed_count(stage5_ctrl, l5).positions

        traj = np.vstack([stage1, stage2[1:], stage3[1:], stage4[1:], stage5[1:]])
        true_cutpoints = np.asarray(
            [
                int(len(stage1) - 1),
                int(len(stage1) + len(stage2) - 2),
                int(len(stage1) + len(stage2) + len(stage3) - 3),
                int(len(stage1) + len(stage2) + len(stage3) + len(stage4) - 4),
            ],
            dtype=int,
        )

        normals = traj - self.sphere_center[None, :]
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        s1 = slice(0, true_cutpoints[0] + 1)
        s2 = slice(true_cutpoints[0] + 1, true_cutpoints[1] + 1)
        s3 = slice(true_cutpoints[1] + 1, true_cutpoints[2] + 1)
        s4 = slice(true_cutpoints[2] + 1, true_cutpoints[3] + 1)
        s5 = slice(true_cutpoints[3] + 1, len(traj))

        axis = np.empty_like(traj)
        axis[s2] = self._clean_axis_near_normals(normals[s2], max_error=s2_normal, fraction=1.0, phase=0.2)
        axis[s1] = self._interpolate_unit_axes(normals[s1][0], axis[s2][0], len(axis[s1]))
        axis[s3] = self._interpolate_unit_axes(axis[s2][-1], normals[s3][-1], len(axis[s3]))
        axis[s4] = self._clean_axis_near_normals(normals[s4], max_error=0.0, fraction=0.0)
        axis[s5] = self._interpolate_unit_axes(axis[s4][-1], normals[s5][-1], len(axis[s5]))
        axis = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)

        self.register_tool_axis_trace(traj, axis)
        return {
            "trajectory": np.asarray(traj, dtype=float),
            "tool_axis": np.asarray(axis, dtype=float),
            "true_cutpoints": true_cutpoints.astype(int),
            "rollout_backend": "geometric_plan",
            "observation_backend": "analytic_raw",
            "planner": "s5_clean_geometric_shell_planner",
            "constraint_values": {
                "s2:surf_dist": float(s2_surf),
                "s2:normal_err": float(s2_normal),
                "s2:speed": float(s2_speed),
                "s4:surf_dist": float(s4_surf),
                "s4:speed": float(s4_speed),
            },
            "stage_lengths": {
                "stage1": int(len(stage1)),
                "stage2": int(len(stage2) - 1),
                "stage3": int(len(stage3) - 1),
                "stage4": int(len(stage4) - 1),
                "stage5": int(len(stage5) - 1),
            },
        }

    def execute_plan_pybullet(
        self,
        scene,
        planned_episode,
        *,
        precheck=None,
        filter_valid=None,
        execution_joint_noise_std: float = 0.0,
        execution_joint_noise_smooth: float = 0.90,
        execution_noise_seed=None,
    ):
        reference = {
            "trajectory": np.asarray(planned_episode["trajectory"], dtype=float),
            "tool_axis": np.asarray(planned_episode["tool_axis"], dtype=float),
            "true_cutpoints": np.asarray(planned_episode["true_cutpoints"], dtype=int),
        }
        do_precheck = bool(self.pybullet_precheck_ik_waypoints if precheck is None else precheck)
        do_filter = bool(self.pybullet_filter_ik_valid if filter_valid is None else filter_valid)
        precheck_report = None
        if do_precheck:
            precheck_report = check_s5_reference_waypoints_ik(
                self,
                scene=scene,
                reference_traj=reference["trajectory"],
                reference_tool_axis=reference["tool_axis"],
                true_cutpoints=reference["true_cutpoints"],
                points_per_stage=int(self.pybullet_precheck_points_per_stage),
            )
            if do_filter and not bool(precheck_report.get("valid", False)):
                raise RuntimeError(f"S5 planned trajectory failed PyBullet IK precheck: {precheck_report}")

        latent = simulate_s5_demo_from_reference(
            self,
            scene=scene,
            reference_traj=reference["trajectory"],
            reference_tool_axis=reference["tool_axis"],
            true_cutpoints=reference["true_cutpoints"],
            execution_joint_noise_std=float(execution_joint_noise_std),
            execution_joint_noise_smooth=float(execution_joint_noise_smooth),
            execution_noise_seed=execution_noise_seed,
        )
        report = self._pybullet_rollout_validity_report(reference, latent)
        if do_filter and not bool(report.get("valid", False)):
            raise RuntimeError(f"S5 planned trajectory failed PyBullet rollout filter: {report}")
        latent["rollout_backend"] = "pybullet_plan"
        latent["observation_backend"] = "pybullet"
        latent["planner"] = str(planned_episode.get("planner", "s5_clean_geometric_shell_planner"))
        latent["planned_constraint_values"] = dict(planned_episode.get("constraint_values", {}))
        latent["ik_filter"] = dict(report)
        if precheck_report is not None:
            latent["ik_filter"]["precheck"] = dict(precheck_report)
        return latent
