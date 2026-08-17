from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from ..pybullet_ur5 import (
    _UR5PoseTracker,
    _axis_from_quat,
    _quat_align_local_axis_to_vec,
    _require_pybullet,
)


def _segment_bounds(true_cutpoints: np.ndarray, length: int) -> list[tuple[int, int]]:
    ends = [int(v) for v in np.asarray(true_cutpoints, dtype=int).reshape(-1)] + [int(length - 1)]
    starts = [0] + [end + 1 for end in ends[:-1]]
    return list(zip(starts, ends))


def _make_stage_labels(true_cutpoints: np.ndarray, length: int) -> np.ndarray:
    labels = np.zeros(int(length), dtype=int)
    for stage_idx, (start, end) in enumerate(_segment_bounds(true_cutpoints, length)):
        labels[int(start) : int(end) + 1] = int(stage_idx)
    return labels



def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if len(vals) <= 1:
        return np.zeros_like(vals)
    grad = np.zeros_like(vals)
    grad[1:] = (vals[1:] - vals[:-1]) / max(float(dt), 1e-12)
    grad[0] = grad[1]
    return grad


def _smooth_command_noise(shape, *, std: float, smooth: float, seed: int | None) -> np.ndarray:
    std = float(std)
    if std <= 0.0:
        return np.zeros(tuple(shape), dtype=float)
    rng = np.random.RandomState(0 if seed is None else int(seed))
    raw = rng.normal(loc=0.0, scale=std, size=tuple(shape))
    if raw.shape[0] > 0:
        raw[0] = 0.0
    alpha = float(np.clip(float(smooth), 0.0, 0.999))
    out = np.zeros_like(raw, dtype=float)
    for i in range(1, raw.shape[0]):
        out[i] = alpha * out[i - 1] + (1.0 - alpha) * raw[i]
    return out


def _smoothstep01(x: float) -> float:
    u = float(np.clip(x, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def _axis_weight_for_index_factory(env, true_cutpoints: np.ndarray, length: int):
    cuts = np.asarray(true_cutpoints, dtype=int).reshape(-1)
    stage1_end = int(cuts[0]) if cuts.size > 0 else min(int(length) - 1, 10)
    default_axis_weight = float(getattr(env, "pybullet_ur5_axis_error_weight", 0.02))
    configured_stage1_axis_weight = getattr(env, "pybullet_ur5_stage1_axis_error_weight", None)
    stage1_axis_weight = None if configured_stage1_axis_weight is None else float(configured_stage1_axis_weight)
    stage1_axis_ramp = int(max(int(getattr(env, "pybullet_ur5_stage1_axis_weight_ramp_points", 5)), 0))

    def axis_weight_for_index(index: int) -> float | None:
        if stage1_axis_weight is None:
            return None
        index_i = int(index)
        if index_i <= stage1_end:
            return stage1_axis_weight
        if stage1_axis_ramp > 0 and index_i <= stage1_end + stage1_axis_ramp:
            alpha = _smoothstep01(float(index_i - stage1_end) / float(stage1_axis_ramp))
            return (1.0 - alpha) * stage1_axis_weight + alpha * default_axis_weight
        return None

    return axis_weight_for_index


def _waypoint_indices_from_cutpoints(true_cutpoints: np.ndarray, length: int, points_per_stage: int) -> np.ndarray:
    idxs: set[int] = set()
    n_per_stage = int(max(int(points_per_stage), 2))
    for start, end in _segment_bounds(np.asarray(true_cutpoints, dtype=int), int(length)):
        if end < start:
            continue
        vals = np.linspace(int(start), int(end), num=min(n_per_stage, int(end - start + 1)))
        idxs.update(int(round(v)) for v in vals)
    idxs.add(0)
    idxs.add(int(length) - 1)
    return np.asarray(sorted(v for v in idxs if 0 <= int(v) < int(length)), dtype=int)


def check_s5_reference_waypoints_ik(
    env,
    *,
    scene: dict[str, Any] | None,
    reference_traj: np.ndarray,
    reference_tool_axis: np.ndarray,
    true_cutpoints: np.ndarray,
    points_per_stage: int = 3,
) -> dict[str, Any]:
    _require_pybullet()

    ref_traj = np.asarray(reference_traj, dtype=float)
    ref_axis = np.asarray(reference_tool_axis, dtype=float)
    if ref_traj.ndim != 2 or ref_traj.shape[1] < 3:
        raise ValueError("reference_traj must have shape (T, 3+).")
    if ref_axis.shape != ref_traj.shape:
        raise ValueError("reference_tool_axis must have the same shape as reference_traj.")
    if len(ref_traj) < 2:
        return {"valid": False, "reason": "empty_or_singleton_reference"}

    geometry = dict((scene or {}).get("geometry", {}))
    sphere_center = np.asarray(geometry.get("sphere_center", env.sphere_center.tolist()), dtype=float)
    sphere_radius = float(geometry.get("sphere_radius", env.sphere_radius))

    tracker = _UR5PoseTracker(env, scene, sphere_center_s5=sphere_center, sphere_radius_s5=sphere_radius)
    try:
        waypoint_idxs = _waypoint_indices_from_cutpoints(
            np.asarray(true_cutpoints, dtype=int),
            len(ref_traj),
            int(points_per_stage),
        )
        target_tip_world = tracker.s5_to_world(ref_traj[waypoint_idxs])
        target_quat = np.asarray(
            [
                _quat_align_local_axis_to_vec(axis, tracker.tool_axis_index, tracker.tool_axis_sign)
                for axis in ref_axis[waypoint_idxs]
            ],
            dtype=float,
        )
        axis_weight_for_index = _axis_weight_for_index_factory(env, np.asarray(true_cutpoints, dtype=int), len(ref_traj))
        stage_labels = _make_stage_labels(np.asarray(true_cutpoints, dtype=int), len(ref_traj))
        normal_stage_ids = {
            int(spec["stage"])
            for spec in env.get_constraint_specs()
            if spec.get("feature_name") == "normal_err"
        }
        q_prev = tracker.home_q.copy()
        max_pos_s5 = 0.0
        max_axis = 0.0
        constrained_max_axis = 0.0
        worst_index = int(waypoint_idxs[0])
        constrained_worst_index = int(waypoint_idxs[0])
        for local_i, traj_i in enumerate(waypoint_idxs):
            q_prev = tracker.solve_ik(
                target_tip_world[local_i],
                target_quat[local_i],
                q_prev,
                axis_weight=axis_weight_for_index(int(traj_i)),
            )
            _, pos_err_world, axis_err = tracker._score_ik_candidate(
                q_prev,
                target_tip_world[local_i],
                ref_axis[int(traj_i)],
                axis_weight=axis_weight_for_index(int(traj_i)),
            )
            pos_err_s5 = float(pos_err_world) / max(float(tracker.world_scale), 1e-12)
            if pos_err_s5 > max_pos_s5 or float(axis_err) > max_axis:
                worst_index = int(traj_i)
            max_pos_s5 = max(max_pos_s5, float(pos_err_s5))
            max_axis = max(max_axis, float(axis_err))
            if int(stage_labels[int(traj_i)]) in normal_stage_ids and float(axis_err) > constrained_max_axis:
                constrained_max_axis = float(axis_err)
                constrained_worst_index = int(traj_i)
    finally:
        tracker.close()

    pos_threshold = float(getattr(env, "pybullet_filter_max_position_error", 0.012))
    axis_threshold = float(getattr(env, "pybullet_filter_max_axis_error", 0.50))
    constrained_axis_threshold = float(getattr(env, "pybullet_filter_constrained_max_axis_error", axis_threshold))
    use_global_axis = bool(getattr(env, "pybullet_filter_global_axis_error", False))
    valid = bool(
        max_pos_s5 <= pos_threshold
        and (not use_global_axis or max_axis <= axis_threshold)
        and constrained_max_axis <= constrained_axis_threshold
    )
    reason = "ok"
    if max_pos_s5 > pos_threshold:
        reason = "precheck_position_error"
    elif use_global_axis and max_axis > axis_threshold:
        reason = "precheck_axis_error"
    elif constrained_max_axis > constrained_axis_threshold:
        reason = "precheck_constrained_axis_error"
        worst_index = int(constrained_worst_index)
    return {
        "valid": valid,
        "reason": reason,
        "max_position_error": float(max_pos_s5),
        "max_axis_error": float(max_axis),
        "constrained_max_axis_error": float(constrained_max_axis),
        "max_speed_ratio": 1.0,
        "waypoint_indices": waypoint_idxs.astype(int).tolist(),
        "worst_index": int(worst_index),
        "thresholds": {
            "max_position_error": pos_threshold,
            "max_axis_error": axis_threshold,
            "global_axis_error": use_global_axis,
            "constrained_max_axis_error": constrained_axis_threshold,
        },
    }


def simulate_s5_demo_from_reference(
    env,
    *,
    scene: dict[str, Any] | None,
    reference_traj: np.ndarray,
    reference_tool_axis: np.ndarray,
    true_cutpoints: np.ndarray,
    execution_joint_noise_std: float = 0.0,
    execution_joint_noise_smooth: float = 0.90,
    execution_noise_seed: int | None = None,
) -> dict[str, Any]:
    _require_pybullet()

    ref_traj = np.asarray(reference_traj, dtype=float)
    ref_axis = np.asarray(reference_tool_axis, dtype=float)
    if ref_traj.ndim != 2 or ref_traj.shape[1] < 3:
        raise ValueError("reference_traj must have shape (T, 3+).")
    if ref_axis.shape != ref_traj.shape:
        raise ValueError("reference_tool_axis must have the same shape as reference_traj.")
    if len(ref_traj) < 2:
        raise ValueError("reference_traj must contain at least two poses.")

    sim_dt = float(getattr(env, "pybullet_sim_dt", 1.0 / 120.0))
    configured_steps = getattr(env, "pybullet_steps_per_sample", None)
    steps_per_sample = max(1, int(round(float(env.dt) / sim_dt))) if configured_steps is None else int(configured_steps)

    geometry = dict((scene or {}).get("geometry", {}))
    sphere_center = np.asarray(geometry.get("sphere_center", env.sphere_center.tolist()), dtype=float)
    sphere_radius = float(geometry.get("sphere_radius", env.sphere_radius))

    tracker = _UR5PoseTracker(env, scene, sphere_center_s5=sphere_center, sphere_radius_s5=sphere_radius)
    try:
        target_tip_world = tracker.s5_to_world(ref_traj)
        target_quat = np.asarray(
            [
                _quat_align_local_axis_to_vec(axis, tracker.tool_axis_index, tracker.tool_axis_sign)
                for axis in ref_axis
            ],
            dtype=float,
        )
        target_axis_world = np.asarray(
            [_axis_from_quat(quat, tracker.tool_axis_index, tracker.tool_axis_sign) for quat in target_quat],
            dtype=float,
        )
        target_ee_world = np.asarray(
            [tracker.target_ee_from_tip(tip, axis) for tip, axis in zip(target_tip_world, target_axis_world)],
            dtype=float,
        )

        q_cmd = np.zeros((len(ref_traj), 6), dtype=float)
        cuts = np.asarray(true_cutpoints, dtype=int).reshape(-1)
        stage1_end = int(cuts[0]) if cuts.size > 0 else min(len(ref_traj) - 1, 10)
        stage1_axis_weight = getattr(env, "pybullet_ur5_stage1_axis_error_weight", None)
        stage1_axis_weight = None if stage1_axis_weight is None else float(stage1_axis_weight)
        axis_weight_for_index = _axis_weight_for_index_factory(env, true_cutpoints, len(ref_traj))

        q_prev = tracker.home_q.copy()
        for i in range(len(ref_traj)):
            axis_weight = axis_weight_for_index(i)
            q_prev = tracker.solve_ik(target_tip_world[i], target_quat[i], q_prev, axis_weight=axis_weight)
            q_cmd[i] = q_prev
        if len(ref_traj) >= 3:
            repair_end = int(np.clip(stage1_end, 1, len(ref_traj) - 1))
            q_next = q_cmd[repair_end].copy()
            for i in range(repair_end - 1, -1, -1):
                q_next = tracker.solve_ik(
                    target_tip_world[i],
                    target_quat[i],
                    q_next,
                    axis_weight=stage1_axis_weight,
                )
                q_cmd[i] = q_next
        q_cmd_nominal = np.asarray(q_cmd, dtype=float).copy()
        execution_noise = _smooth_command_noise(
            q_cmd.shape,
            std=float(execution_joint_noise_std),
            smooth=float(execution_joint_noise_smooth),
            seed=execution_noise_seed,
        )
        if float(execution_joint_noise_std) > 0.0:
            q_cmd = np.clip(q_cmd + execution_noise, tracker.q_lo[None, :], tracker.q_hi[None, :])
        else:
            execution_noise = np.zeros_like(q_cmd, dtype=float)

        realized_tip_world = np.zeros_like(target_tip_world)
        realized_ee_world = np.zeros_like(target_tip_world)
        realized_axis = np.zeros_like(ref_axis)
        quats = np.zeros((len(ref_traj), 4), dtype=float)
        q_meas = np.zeros_like(q_cmd)
        qd_meas = np.zeros_like(q_cmd)
        contact_flags = np.zeros(len(ref_traj), dtype=int)

        tracker.reset_joint_state(q_cmd[0])
        tracker.command_joint_target(q_cmd[0])
        settle_steps = getattr(env, "pybullet_ur5_settle_steps", None)
        if settle_steps is None:
            settle_steps = max(20, steps_per_sample)
        tracker.step(int(settle_steps))

        for i in range(len(ref_traj)):
            tracker.command_joint_target(q_cmd[i])
            tracker.step(steps_per_sample)
            ee_pos, ee_quat = tracker.get_ee_pose()
            tip_pos, tip_axis = tracker.tip_from_ee_pose(ee_pos, ee_quat)
            q_i, qd_i = tracker.get_joint_state()
            pos_s5 = tracker.world_to_s5(tip_pos)
            realized_tip_world[i] = tip_pos
            realized_ee_world[i] = ee_pos
            quats[i] = ee_quat
            realized_axis[i] = tip_axis
            q_meas[i] = q_i
            qd_meas[i] = qd_i

            surf_dist = abs(float(np.linalg.norm(pos_s5 - sphere_center) - sphere_radius))
            geom_contact = surf_dist <= float(getattr(env, "pybullet_contact_surface_tol", 0.025 * sphere_radius))
            contact_flags[i] = int(geom_contact or tracker.has_contact())
    finally:
        tracker.close()

    realized_pos = tracker.world_to_s5(realized_tip_world)
    linear_velocity = _finite_difference(realized_pos, float(env.dt))
    angular_velocity = _finite_difference(realized_axis, float(env.dt))
    ik_pos_err_world = np.linalg.norm(realized_tip_world - target_tip_world, axis=1)
    ik_axis_err = np.arccos(np.clip(np.sum(realized_axis * ref_axis, axis=1), -1.0, 1.0))

    return {
        "trajectory": np.asarray(realized_pos, dtype=float),
        "tool_axis": np.asarray(realized_axis, dtype=float),
        "quaternions": np.asarray(quats, dtype=float),
        "linear_velocity": np.asarray(linear_velocity, dtype=float),
        "angular_velocity": np.asarray(angular_velocity, dtype=float),
        "contact_flags": np.asarray(contact_flags, dtype=int),
        "joint_positions": np.asarray(q_meas, dtype=float),
        "joint_velocities": np.asarray(qd_meas, dtype=float),
        "joint_position_commands": np.asarray(q_cmd, dtype=float),
        "joint_position_commands_nominal": np.asarray(q_cmd_nominal, dtype=float),
        "execution_joint_noise": np.asarray(execution_noise, dtype=float),
        "true_cutpoints": np.asarray(true_cutpoints, dtype=int),
        "true_labels": _make_stage_labels(true_cutpoints, len(realized_pos)),
        "reference_trajectory": np.asarray(ref_traj, dtype=float),
        "reference_tool_axis": np.asarray(ref_axis, dtype=float),
        "reference_trajectory_world": np.asarray(target_tip_world, dtype=float),
        "target_ee_trajectory_world": np.asarray(target_ee_world, dtype=float),
        "realized_trajectory_world": np.asarray(realized_tip_world, dtype=float),
        "realized_ee_trajectory_world": np.asarray(realized_ee_world, dtype=float),
        "ik_position_error_world": np.asarray(ik_pos_err_world, dtype=float),
        "ik_axis_error": np.asarray(ik_axis_err, dtype=float),
        "ur5_tool_axis": str(getattr(env, "pybullet_ur5_tool_axis", "x")),
        "ur5_tip_offset": float(getattr(env, "pybullet_ur5_tip_offset", 0.0)),
        "sim_dt": float(sim_dt),
        "steps_per_sample": int(steps_per_sample),
        "robot_backend": "ur5_pybullet_ik_position_control_virtual_tip",
    }

class S5ExecutionMixin:

    def _pybullet_attempt_seed(self, seed, scene, attempt: int) -> int:
        if seed is not None:
            base = int(seed)
        else:
            base = int((scene or {}).get("rollout_seed", 0))
        return int(base + int(attempt))

    def demo_seed_for_index(self, seed: int, demo_idx: int) -> int:
        if self.rollout_backend == "pybullet" and bool(self.pybullet_filter_ik_valid):
            return int(seed) + int(demo_idx) * int(self.pybullet_filter_max_attempts)
        return int(seed) + int(demo_idx)

    @staticmethod
    def _stage_slices_from_cutpoints(length: int, cutpoints) -> list[slice]:
        T = int(length)
        cuts = np.asarray(cutpoints, dtype=int).reshape(-1)
        cuts = np.sort(cuts[(cuts >= 0) & (cuts < T - 1)])
        ends = cuts.tolist() + [T - 1]
        starts = [0] + [int(v) + 1 for v in ends[:-1]]
        return [slice(int(a), int(b) + 1) for a, b in zip(starts, ends)]

    @staticmethod
    def _axis_error_trace(axis_a, axis_b) -> np.ndarray:
        a = np.asarray(axis_a, dtype=float)
        b = np.asarray(axis_b, dtype=float)
        a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        return np.arccos(np.clip(np.sum(a * b, axis=1), -1.0, 1.0))

    def _pybullet_rollout_validity_report(self, reference: dict, latent: dict) -> dict:
        ref_traj = np.asarray(reference["trajectory"], dtype=float)
        exe_traj = np.asarray(latent["trajectory"], dtype=float)
        ref_axis = np.asarray(reference["tool_axis"], dtype=float)
        exe_axis = np.asarray(latent["tool_axis"], dtype=float)
        T = min(len(ref_traj), len(exe_traj), len(ref_axis), len(exe_axis))
        if T <= 1:
            return {"valid": False, "reason": "empty_or_singleton_rollout"}
        ref_traj = ref_traj[:T]
        exe_traj = exe_traj[:T]
        ref_axis = ref_axis[:T]
        exe_axis = exe_axis[:T]
        cutpoints = np.asarray(reference["true_cutpoints"], dtype=int)

        pos_err = np.linalg.norm(exe_traj - ref_traj, axis=1)
        axis_err = self._axis_error_trace(exe_axis, ref_axis)
        ref_speed = np.zeros(T, dtype=float)
        exe_speed = np.zeros(T, dtype=float)
        ref_speed[1:] = np.linalg.norm(np.diff(ref_traj, axis=0), axis=1) / max(float(self.dt), 1e-12)
        exe_speed[1:] = np.linalg.norm(np.diff(exe_traj, axis=0), axis=1) / max(float(self.dt), 1e-12)
        speed_ratio = float(np.max(exe_speed / np.maximum(ref_speed, 1e-6)))

        stage_slices = self._stage_slices_from_cutpoints(T, cutpoints)
        normal_stage_ids = sorted(
            {
                int(spec["stage"])
                for spec in self.get_constraint_specs()
                if spec.get("feature_name") == "normal_err"
            }
        )
        constrained_axis_max = 0.0
        constrained_stage_axis_max = {}
        for stage_idx in normal_stage_ids:
            if 0 <= stage_idx < len(stage_slices):
                val = float(np.max(axis_err[stage_slices[stage_idx]]))
                constrained_axis_max = max(constrained_axis_max, val)
                constrained_stage_axis_max[str(stage_idx)] = val

        max_pos = float(np.max(pos_err))
        max_axis = float(np.max(axis_err))
        global_axis_ok = (not bool(self.pybullet_filter_global_axis_error)) or (
            max_axis <= float(self.pybullet_filter_max_axis_error)
        )
        valid = (
            max_pos <= float(self.pybullet_filter_max_position_error)
            and global_axis_ok
            and constrained_axis_max <= float(self.pybullet_filter_constrained_max_axis_error)
            and speed_ratio <= float(self.pybullet_filter_max_speed_ratio)
        )
        reason = "ok"
        if max_pos > float(self.pybullet_filter_max_position_error):
            reason = "position_error"
        elif bool(self.pybullet_filter_global_axis_error) and max_axis > float(self.pybullet_filter_max_axis_error):
            reason = "axis_error"
        elif constrained_axis_max > float(self.pybullet_filter_constrained_max_axis_error):
            reason = "constrained_axis_error"
        elif speed_ratio > float(self.pybullet_filter_max_speed_ratio):
            reason = "speed_ratio"
        return {
            "valid": bool(valid),
            "reason": reason,
            "max_position_error": max_pos,
            "max_axis_error": max_axis,
            "constrained_max_axis_error": float(constrained_axis_max),
            "constrained_stage_axis_max": constrained_stage_axis_max,
            "max_speed_ratio": speed_ratio,
            "thresholds": {
                "max_position_error": float(self.pybullet_filter_max_position_error),
                "max_axis_error": float(self.pybullet_filter_max_axis_error),
                "global_axis_error": bool(self.pybullet_filter_global_axis_error),
                "constrained_max_axis_error": float(self.pybullet_filter_constrained_max_axis_error),
                "max_speed_ratio": float(self.pybullet_filter_max_speed_ratio),
            },
        }

    def _rollout_demo_pybullet(self, scene, seed=None, rng=None, **kwargs):
        progress_callback = kwargs.pop("progress_callback", None)
        max_attempts = self.pybullet_filter_max_attempts if bool(self.pybullet_filter_ik_valid) else 1
        last_report = None
        last_seed = None
        for attempt in range(int(max_attempts)):
            if rng is not None:
                reference = self._rollout_demo_analytic(scene, seed=seed, rng=rng, **kwargs)
                attempt_seed = None
            else:
                attempt_seed = self._pybullet_attempt_seed(seed, scene, attempt)
                reference = self._rollout_demo_analytic(scene, seed=attempt_seed, rng=None, **kwargs)
            precheck_report = None
            if bool(self.pybullet_precheck_ik_waypoints):
                precheck_report = check_s5_reference_waypoints_ik(
                    self,
                    scene=scene,
                    reference_traj=np.asarray(reference["trajectory"], dtype=float),
                    reference_tool_axis=np.asarray(reference["tool_axis"], dtype=float),
                    true_cutpoints=np.asarray(reference["true_cutpoints"], dtype=int),
                    points_per_stage=int(self.pybullet_precheck_points_per_stage),
                )
                last_report = precheck_report
                last_seed = attempt_seed
                if not bool(precheck_report.get("valid", False)):
                    if progress_callback is not None:
                        progress_callback(
                            attempt=int(attempt),
                            max_attempts=int(max_attempts),
                            attempt_seed=None if attempt_seed is None else int(attempt_seed),
                            report=dict(precheck_report),
                        )
                    continue
            latent = simulate_s5_demo_from_reference(
                self,
                scene=scene,
                reference_traj=np.asarray(reference["trajectory"], dtype=float),
                reference_tool_axis=np.asarray(reference["tool_axis"], dtype=float),
                true_cutpoints=np.asarray(reference["true_cutpoints"], dtype=int),
            )
            report = self._pybullet_rollout_validity_report(reference, latent)
            last_report = report
            last_seed = attempt_seed
            if progress_callback is not None:
                progress_callback(
                    attempt=int(attempt),
                    max_attempts=int(max_attempts),
                    attempt_seed=None if attempt_seed is None else int(attempt_seed),
                    report=dict(report),
                )
            if report["valid"] or not bool(self.pybullet_filter_ik_valid):
                latent["rollout_backend"] = "pybullet"
                latent["observation_backend"] = "pybullet"
                latent["reference_seed"] = None if attempt_seed is None else int(attempt_seed)
                latent["timestamps"] = np.asarray(reference["timestamps"], dtype=float)
                latent["ik_filter"] = dict(report)
                if precheck_report is not None:
                    latent["ik_filter"]["precheck"] = dict(precheck_report)
                latent["ik_filter"]["attempt"] = int(attempt)
                latent["ik_filter"]["max_attempts"] = int(max_attempts)
                generation_metadata = dict(reference.get("generation_metadata", {}))
                generation_metadata["pybullet_execution"] = {
                    "reference_seed": None if attempt_seed is None else int(attempt_seed),
                    "accepted_attempt": int(attempt),
                    "max_attempts": int(max_attempts),
                    "ik_filter": dict(latent["ik_filter"]),
                }
                latent["generation_metadata"] = generation_metadata
                return latent

        raise RuntimeError(
            "Failed to sample an IK-valid S5 pybullet demo after "
            f"{int(max_attempts)} attempts. Last seed={last_seed}, last_report={last_report}"
        )

    def rollout_demo(self, scene, seed=None, rng=None, backend=None, **kwargs):
        active_backend = str(self.rollout_backend if backend is None else backend).lower()
        if active_backend == "analytic":
            return self._rollout_demo_analytic(scene, seed=seed, rng=rng, **kwargs)
        if active_backend == "pybullet":
            return self._rollout_demo_pybullet(scene, seed=seed, rng=rng, **kwargs)
        raise ValueError(f"Unsupported S5 rollout backend '{active_backend}'.")
