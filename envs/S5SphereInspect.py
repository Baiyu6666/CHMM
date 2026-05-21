from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .base import TaskBundle
from .rendering import render_s5_pybullet_demo_video, render_s5_pybullet_episode, render_sphere_episode
from planner import optimize_trajectory, resample_polyline


# Integrated PyBullet backend helpers for S5SphereInspect.
import contextlib
import math
import os
import re
import tempfile
from typing import Any

import numpy as np

try:
    import pybullet as p
except ModuleNotFoundError:
    p = None


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_UR5_URDF = os.path.join(_THIS_DIR, "assets", "UR5+gripper", "ur5_gripper.urdf")


def _require_pybullet() -> None:
    if p is None:
        raise RuntimeError("pybullet is required for S5 rollout_backend='pybullet'.")


@contextlib.contextmanager
def _suppress_native_output(enabled: bool = True):
    if not enabled:
        yield
        return

    devnull_fd = None
    saved_fds = []
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        for fd in (1, 2):
            saved_fd = os.dup(fd)
            saved_fds.append((fd, saved_fd))
            os.dup2(devnull_fd, fd)
        yield
    finally:
        for fd, saved_fd in reversed(saved_fds):
            try:
                os.dup2(saved_fd, fd)
            finally:
                os.close(saved_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


def _normalize(vec: np.ndarray, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=float).reshape(3)
    return arr / norm


def _segment_bounds(true_cutpoints: np.ndarray, length: int) -> list[tuple[int, int]]:
    ends = [int(v) for v in np.asarray(true_cutpoints, dtype=int).reshape(-1)] + [int(length - 1)]
    starts = [0] + [end + 1 for end in ends[:-1]]
    return list(zip(starts, ends))


def _make_stage_labels(true_cutpoints: np.ndarray, length: int) -> np.ndarray:
    labels = np.zeros(int(length), dtype=int)
    for stage_idx, (start, end) in enumerate(_segment_bounds(true_cutpoints, length)):
        labels[int(start) : int(end) + 1] = int(stage_idx)
    return labels


def _candidate_mesh_paths(urdf_dir: str, basename: str) -> list[str]:
    stem, _ = os.path.splitext(basename)
    names = [
        basename,
        basename.lower(),
        basename.upper(),
        stem + ".stl",
        stem + ".STL",
        stem + ".dae",
        stem + ".DAE",
    ]
    roots = [
        os.path.join(urdf_dir, "mesh"),
        os.path.join(urdf_dir, "mesh", "visual"),
        os.path.join(urdf_dir, "meshes"),
        os.path.join(urdf_dir, "meshes", "visual"),
    ]
    return [os.path.join(root, name) for root in roots for name in names]


def _resolve_mesh_uri(urdf_dir: str, uri: str) -> str:
    if not uri.startswith("package://"):
        return uri
    basename = os.path.basename(uri)
    for candidate in _candidate_mesh_paths(urdf_dir, basename):
        if os.path.exists(candidate):
            return candidate
    return uri


def _make_pybullet_friendly_urdf(urdf_path: str) -> str:
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    with open(urdf_path, "r", encoding="utf-8") as f:
        text = f.read()

    def _sub(match: re.Match[str]) -> str:
        pre, uri, post = match.group(1), match.group(2), match.group(3)
        return f'{pre}{_resolve_mesh_uri(urdf_dir, uri)}{post}'

    patched = re.sub(r'(filename\s*=\s*")([^"]+)(")', _sub, text)
    fd, tmp = tempfile.mkstemp(prefix="s5_ur5_patch_", suffix=".urdf")
    os.close(fd)
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(patched)
    return tmp


def _pick_default_ee_link_index(robot_id: int, fallback_joint_index: int, client_id: int) -> int:
    prefer = ("tcp", "tool0", "ee_link", "gripper", "wrist_3_link", "wrist3_link", "wrist3")
    nj = p.getNumJoints(robot_id, physicsClientId=client_id)
    best = None
    best_rank = 10**9
    for j in range(nj):
        info = p.getJointInfo(robot_id, j, physicsClientId=client_id)
        link_name = info[12].decode("utf-8", errors="ignore").lower()
        for rank, key in enumerate(prefer):
            if key in link_name and rank < best_rank:
                best = int(j)
                best_rank = int(rank)
    return int(fallback_joint_index if best is None else best)


def _quat_from_matrix(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=float).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[0, 0] - r[1, 1] - r[2, 2], 1e-12)) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(max(1.0 + r[1, 1] - r[0, 0] - r[2, 2], 1e-12)) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(max(1.0 + r[2, 2] - r[0, 0] - r[1, 1], 1e-12)) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=float)
    return quat / max(float(np.linalg.norm(quat)), 1e-12)


def _parse_tool_axis(name: str) -> tuple[int, float]:
    text = str(name).strip().lower()
    sign = -1.0 if text.startswith("-") else 1.0
    axis_name = text[1:] if text[:1] in {"+", "-"} else text
    return {"x": 0, "y": 1, "z": 2}.get(axis_name, 0), sign


def _quat_align_local_axis_to_vec(
    vec: np.ndarray,
    axis_index: int,
    axis_sign: float = 1.0,
    up_hint: np.ndarray | None = None,
) -> np.ndarray:
    target = _normalize(vec, fallback=(1.0, 0.0, 0.0))
    local_positive_axis = float(axis_sign) * target
    up = np.asarray([0.0, 0.0, 1.0] if up_hint is None else up_hint, dtype=float).reshape(3)
    if abs(float(np.dot(_normalize(up, fallback=(0.0, 0.0, 1.0)), local_positive_axis))) > 0.94:
        up = np.asarray([0.0, 1.0, 0.0], dtype=float)
    axis_index = int(axis_index)
    if axis_index == 0:
        x_axis = local_positive_axis
        y_axis = _normalize(np.cross(up, x_axis), fallback=(0.0, 1.0, 0.0))
        z_axis = _normalize(np.cross(x_axis, y_axis), fallback=(0.0, 0.0, 1.0))
    elif axis_index == 1:
        y_axis = local_positive_axis
        z_axis = _normalize(np.cross(y_axis, up), fallback=(0.0, 0.0, 1.0))
        x_axis = _normalize(np.cross(y_axis, z_axis), fallback=(1.0, 0.0, 0.0))
    else:
        z_axis = local_positive_axis
        x_axis = _normalize(np.cross(up, z_axis), fallback=(1.0, 0.0, 0.0))
        y_axis = _normalize(np.cross(z_axis, x_axis), fallback=(0.0, 1.0, 0.0))
    rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    return _quat_from_matrix(rot)


def _quat_align_local_x_to_vec(vec: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    return _quat_align_local_axis_to_vec(vec, 0, up_hint=up_hint)


def _axis_from_quat(quat: np.ndarray, axis_index: int = 0, axis_sign: float = 1.0) -> np.ndarray:
    rot = np.asarray(p.getMatrixFromQuaternion(np.asarray(quat, dtype=float).tolist()), dtype=float).reshape(3, 3)
    return _normalize(float(axis_sign) * rot[:, int(axis_index)], fallback=(1.0, 0.0, 0.0))


def _rpy_to_quat(rpy: np.ndarray | tuple[float, float, float]) -> tuple[float, float, float, float]:
    return tuple(float(v) for v in p.getQuaternionFromEuler(np.asarray(rpy, dtype=float).reshape(3).tolist()))


class _UR5PoseTracker:
    def __init__(self, env, scene: dict[str, Any] | None, sphere_center_s5: np.ndarray, sphere_radius_s5: float):
        _require_pybullet()
        try:
            import pybullet_data  # type: ignore
        except ModuleNotFoundError:
            pybullet_data = None

        self.env = env
        self.scene = scene or {}
        self.sim_dt = float(getattr(env, "pybullet_sim_dt", 1.0 / 120.0))
        self.gravity_z = float(getattr(env, "pybullet_gravity_z", -9.81))
        self.solver_iterations = int(getattr(env, "pybullet_solver_iterations", 100))
        self.world_scale = float(getattr(env, "pybullet_world_scale", 1.0))
        self.world_center = np.asarray(getattr(env, "pybullet_world_center", (0.55, 0.0, 0.52)), dtype=float).reshape(3)
        self.tip_offset = float(getattr(env, "pybullet_ur5_tip_offset", 0.0))
        self.sphere_center_s5 = np.asarray(sphere_center_s5, dtype=float).reshape(3)
        self.sphere_radius_s5 = float(sphere_radius_s5)
        self.sphere_center_world = self.s5_to_world(self.sphere_center_s5)
        self.sphere_radius_world = self.world_scale * self.sphere_radius_s5
        self.client_id = p.connect(p.GUI if bool(getattr(env, "pybullet_force_gui", False)) else p.DIRECT)
        self.patched_urdf = None

        p.resetSimulation(physicsClientId=self.client_id)
        if pybullet_data is not None:
            p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        p.setTimeStep(self.sim_dt, physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, self.gravity_z, physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(numSolverIterations=self.solver_iterations, physicsClientId=self.client_id)

        self.sphere_id = self._spawn_sphere()
        self.robot_id = self._load_robot()
        self._init_joints()
        self.home_q = np.clip(
            np.asarray(getattr(env, "pybullet_ur5_home_q", [0.0, -1.25, 1.85, -2.10, -1.57, 0.0]), dtype=float),
            self.q_lo,
            self.q_hi,
        )
        self.reset_joint_state(self.home_q)
        self.disable_default_motors()

    def close(self) -> None:
        if getattr(self, "client_id", None) is not None:
            try:
                p.disconnect(physicsClientId=self.client_id)
            except Exception:
                pass
            self.client_id = None
        if self.patched_urdf:
            try:
                os.remove(self.patched_urdf)
            except OSError:
                pass
            self.patched_urdf = None

    def s5_to_world(self, pos_s5: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos_s5, dtype=float).reshape(-1, 3)
        out = self.world_center[None, :] + self.world_scale * (pos - self.sphere_center_s5[None, :])
        return out.reshape(np.asarray(pos_s5).shape)

    def world_to_s5(self, pos_world: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos_world, dtype=float).reshape(-1, 3)
        out = self.sphere_center_s5[None, :] + (pos - self.world_center[None, :]) / max(self.world_scale, 1e-12)
        return out.reshape(np.asarray(pos_world).shape)

    def _spawn_sphere(self) -> int:
        if bool(getattr(self.env, "pybullet_sphere_collision", False)):
            col_id = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=float(self.sphere_radius_world),
                physicsClientId=self.client_id,
            )
        else:
            col_id = -1
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=float(self.sphere_radius_world),
            rgbaColor=[0.70, 0.82, 0.92, 0.24],
            specularColor=[0.40, 0.40, 0.40],
            physicsClientId=self.client_id,
        )
        return int(
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=self.sphere_center_world.tolist(),
                physicsClientId=self.client_id,
            )
        )

    def _load_robot(self) -> int:
        urdf_path = str(getattr(self.env, "pybullet_ur5_urdf_path", "") or _DEFAULT_UR5_URDF)
        if not os.path.exists(urdf_path):
            raise RuntimeError(f"UR5 URDF not found: {urdf_path}")
        load_path = urdf_path
        with open(urdf_path, "r", encoding="utf-8") as f:
            if "package://" in f.read():
                load_path = _make_pybullet_friendly_urdf(urdf_path)
                self.patched_urdf = load_path
        base_xyz = np.asarray(getattr(self.env, "pybullet_ur5_base_xyz", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        base_rpy = np.asarray(getattr(self.env, "pybullet_ur5_base_rpy", (0.0, 0.0, 0.0)), dtype=float).reshape(3)
        suppress_warnings = bool(getattr(self.env, "pybullet_suppress_urdf_warnings", True))
        with _suppress_native_output(suppress_warnings):
            return int(
                p.loadURDF(
                    load_path,
                    basePosition=base_xyz.tolist(),
                    baseOrientation=_rpy_to_quat(base_rpy),
                    useFixedBase=True,
                    flags=p.URDF_USE_INERTIA_FROM_FILE,
                    physicsClientId=self.client_id,
                )
            )

    def _init_joints(self) -> None:
        self.arm_joint_indices: list[int] = []
        self.ik_joint_indices: list[int] = []
        q_lo = []
        q_hi = []
        nj = p.getNumJoints(self.robot_id, physicsClientId=self.client_id)
        for j in range(nj):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self.client_id)
            joint_type = int(info[2])
            if joint_type != p.JOINT_FIXED:
                self.ik_joint_indices.append(int(j))
            if joint_type == p.JOINT_REVOLUTE:
                self.arm_joint_indices.append(int(j))
                lo = float(info[8])
                hi = float(info[9])
                if (not np.isfinite(lo)) or (not np.isfinite(hi)) or hi <= lo:
                    lo, hi = -math.pi, math.pi
                q_lo.append(lo)
                q_hi.append(hi)
        if len(self.arm_joint_indices) < 6:
            raise RuntimeError(f"UR5 model has fewer than 6 revolute joints: {len(self.arm_joint_indices)}")
        self.arm_joint_indices = self.arm_joint_indices[:6]
        self.q_lo = np.asarray(q_lo[:6], dtype=float)
        self.q_hi = np.asarray(q_hi[:6], dtype=float)
        self.joint_ranges = np.maximum(self.q_hi - self.q_lo, 1e-3)
        self.arm_ik_positions = [int(self.ik_joint_indices.index(j)) for j in self.arm_joint_indices]
        ee_override = int(getattr(self.env, "pybullet_ur5_ee_link_index", -1))
        self.ee_link_index = (
            ee_override
            if ee_override >= 0
            else _pick_default_ee_link_index(self.robot_id, self.arm_joint_indices[-1], self.client_id)
        )
        self.tool_axis_index, self.tool_axis_sign = _parse_tool_axis(
            str(getattr(self.env, "pybullet_ur5_tool_axis", "x"))
        )

    def disable_default_motors(self) -> None:
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * len(self.arm_joint_indices),
            forces=[0.0] * len(self.arm_joint_indices),
            physicsClientId=self.client_id,
        )

    def reset_joint_state(self, q: np.ndarray) -> None:
        qv = np.clip(np.asarray(q, dtype=float).reshape(6), self.q_lo, self.q_hi)
        for i, joint_idx in enumerate(self.arm_joint_indices):
            p.resetJointState(
                self.robot_id,
                int(joint_idx),
                targetValue=float(qv[i]),
                targetVelocity=0.0,
                physicsClientId=self.client_id,
            )

    def get_joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        sts = p.getJointStates(self.robot_id, self.arm_joint_indices, physicsClientId=self.client_id)
        q = np.asarray([float(s[0]) for s in sts], dtype=float)
        qd = np.asarray([float(s[1]) for s in sts], dtype=float)
        return q, qd

    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        ls = p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        pos = np.asarray(ls[4], dtype=float)
        quat = np.asarray(ls[5], dtype=float)
        quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
        return pos, quat

    def tip_from_ee_pose(self, ee_pos: np.ndarray, ee_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        axis = _axis_from_quat(ee_quat, self.tool_axis_index, self.tool_axis_sign)
        tip_pos = np.asarray(ee_pos, dtype=float).reshape(3) - float(self.tip_offset) * axis
        return tip_pos, axis

    def target_ee_from_tip(self, target_tip_pos: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
        return np.asarray(target_tip_pos, dtype=float).reshape(3) + float(self.tip_offset) * _normalize(target_axis)

    def _run_pybullet_ik(
        self,
        target_pos: np.ndarray,
        *,
        target_quat: np.ndarray | None,
        rest_q: np.ndarray,
    ) -> np.ndarray:
        damping = float(getattr(self.env, "pybullet_ur5_ik_damping", 0.02))
        kwargs = {
            "lowerLimits": [float(v) for v in self.q_lo],
            "upperLimits": [float(v) for v in self.q_hi],
            "jointRanges": [float(v) for v in self.joint_ranges],
            "restPoses": [float(v) for v in np.clip(np.asarray(rest_q, dtype=float).reshape(6), self.q_lo, self.q_hi)],
            "jointDamping": [damping] * len(self.ik_joint_indices),
            "solver": p.IK_DLS,
            "maxNumIterations": int(getattr(self.env, "pybullet_ur5_ik_iterations", 120)),
            "residualThreshold": float(getattr(self.env, "pybullet_ur5_ik_residual_threshold", 1e-5)),
            "physicsClientId": self.client_id,
        }
        if target_quat is not None:
            kwargs["targetOrientation"] = [float(v) for v in np.asarray(target_quat, dtype=float).reshape(4)]
        sol_full = p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            targetPosition=[float(v) for v in np.asarray(target_pos, dtype=float).reshape(3)],
            **kwargs,
        )
        sol_full = np.asarray(sol_full, dtype=float).reshape(-1)
        if len(sol_full) < len(self.ik_joint_indices):
            raise RuntimeError(f"pybullet IK returned {len(sol_full)} joints, expected {len(self.ik_joint_indices)}")
        q = np.asarray([sol_full[idx] for idx in self.arm_ik_positions], dtype=float)
        return np.clip(q, self.q_lo, self.q_hi)

    def _score_ik_candidate(
        self,
        q: np.ndarray,
        target_tip_pos: np.ndarray,
        target_axis: np.ndarray,
        *,
        axis_weight: float | None = None,
    ) -> tuple[float, float, float]:
        self.reset_joint_state(q)
        ee_pos, ee_quat = self.get_ee_pose()
        tip_pos, axis = self.tip_from_ee_pose(ee_pos, ee_quat)
        pos_err = float(np.linalg.norm(tip_pos - np.asarray(target_tip_pos, dtype=float).reshape(3)))
        axis_err = float(np.arccos(np.clip(np.dot(axis, _normalize(target_axis)), -1.0, 1.0)))
        axis_weight = float(getattr(self.env, "pybullet_ur5_axis_error_weight", 0.02) if axis_weight is None else axis_weight)
        score = pos_err + axis_weight * axis_err
        return score, pos_err, axis_err

    def _target_quat_roll_candidates(self, target_axis: np.ndarray, seed_quat: np.ndarray) -> list[np.ndarray]:
        target = _normalize(target_axis)
        candidates = [np.asarray(seed_quat, dtype=float).reshape(4)]
        for up_hint in (
            np.asarray([0.0, 0.0, 1.0], dtype=float),
            np.asarray([0.0, 1.0, 0.0], dtype=float),
            np.asarray([1.0, 0.0, 0.0], dtype=float),
            np.asarray([0.0, -1.0, 0.0], dtype=float),
            np.asarray([-1.0, 0.0, 0.0], dtype=float),
        ):
            candidates.append(
                _quat_align_local_axis_to_vec(
                    target,
                    self.tool_axis_index,
                    self.tool_axis_sign,
                    up_hint=up_hint,
                )
            )

        unique: list[np.ndarray] = []
        for quat in candidates:
            quat = np.asarray(quat, dtype=float).reshape(4)
            quat = quat / max(float(np.linalg.norm(quat)), 1e-12)
            if not any(abs(float(np.dot(quat, prev))) > 0.999 for prev in unique):
                unique.append(quat)
        return unique

    def solve_ik(
        self,
        target_tip_pos: np.ndarray,
        target_quat: np.ndarray,
        rest_q: np.ndarray,
        *,
        axis_weight: float | None = None,
    ) -> np.ndarray:
        home_blend = float(np.clip(getattr(self.env, "pybullet_ur5_rest_home_blend", 0.03), 0.0, 1.0))
        rest_prev = np.clip(np.asarray(rest_q, dtype=float).reshape(6), self.q_lo, self.q_hi)
        rest_blend = np.clip((1.0 - home_blend) * rest_prev + home_blend * self.home_q, self.q_lo, self.q_hi)
        target_axis = _axis_from_quat(target_quat, self.tool_axis_index, self.tool_axis_sign)
        target_ee_pos = self.target_ee_from_tip(target_tip_pos, target_axis)
        q_pos = self._run_pybullet_ik(target_ee_pos, target_quat=None, rest_q=rest_blend)
        candidates = [q_pos]
        for quat in self._target_quat_roll_candidates(target_axis, target_quat):
            candidates.extend(
                [
                    self._run_pybullet_ik(target_ee_pos, target_quat=quat, rest_q=rest_blend),
                    self._run_pybullet_ik(target_ee_pos, target_quat=quat, rest_q=q_pos),
                    self._run_pybullet_ik(target_ee_pos, target_quat=quat, rest_q=self.home_q),
                ]
            )
        best_q = candidates[0]
        best_score = float("inf")
        best_pos_err = float("inf")
        active_axis_weight = float(
            getattr(self.env, "pybullet_ur5_axis_error_weight", 0.02) if axis_weight is None else axis_weight
        )
        for q in candidates:
            score, _, _ = self._score_ik_candidate(q, target_tip_pos, target_axis, axis_weight=axis_weight)
            if score < best_score:
                best_score = score
                best_q = q
                _, best_pos_err, _ = self._score_ik_candidate(
                    q,
                    target_tip_pos,
                    target_axis,
                    axis_weight=axis_weight,
                )
        fallback_threshold = float(getattr(self.env, "pybullet_ur5_ik_position_error_fallback_threshold", 0.004))
        fallback_axis_weight = float(getattr(self.env, "pybullet_ur5_ik_fallback_axis_error_weight", 0.005))
        if best_pos_err > fallback_threshold and active_axis_weight > fallback_axis_weight:
            fallback_q = best_q
            fallback_score = float("inf")
            fallback_pos_err = best_pos_err
            for q in candidates:
                score, pos_err, _ = self._score_ik_candidate(
                    q,
                    target_tip_pos,
                    target_axis,
                    axis_weight=fallback_axis_weight,
                )
                if score < fallback_score:
                    fallback_score = score
                    fallback_pos_err = pos_err
                    fallback_q = q
            if fallback_pos_err < best_pos_err:
                best_q = fallback_q
        self.reset_joint_state(best_q)
        return np.asarray(best_q, dtype=float)

    def command_joint_target(self, q_target: np.ndarray) -> None:
        p.setJointMotorControlArray(
            self.robot_id,
            self.arm_joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[float(v) for v in np.asarray(q_target, dtype=float).reshape(6)],
            targetVelocities=[0.0] * len(self.arm_joint_indices),
            positionGains=[float(getattr(self.env, "pybullet_ur5_position_gain", 0.08))] * len(self.arm_joint_indices),
            velocityGains=[float(getattr(self.env, "pybullet_ur5_velocity_gain", 1.0))] * len(self.arm_joint_indices),
            forces=[float(getattr(self.env, "pybullet_ur5_max_force", 500.0))] * len(self.arm_joint_indices),
            physicsClientId=self.client_id,
        )

    def step(self, n_steps: int) -> None:
        for _ in range(int(max(1, n_steps))):
            p.stepSimulation(physicsClientId=self.client_id)

    def has_contact(self) -> bool:
        return bool(p.getContactPoints(self.robot_id, self.sphere_id, physicsClientId=self.client_id))


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

_S5_METRIC_SCALE = 0.18


class S5SphereInspectEnv:
    """
    3D spherical surface inspection task.
    """

    def __init__(
        self,
        sphere_center=(0.0, 0.0, 0.0),
        sphere_radius=_S5_METRIC_SCALE,
        shell_thickness=0.24 * _S5_METRIC_SCALE,
        seg_lengths=(18, 34, 24, 18),
        seg_length_jitter=(3, 5, 5, 3),
        approach_offset=0.42 * _S5_METRIC_SCALE,
        depart_offset=0.50 * _S5_METRIC_SCALE,
        stage1_speed_max=0.12 * _S5_METRIC_SCALE,
        stage2_speed_max=0.05 * _S5_METRIC_SCALE,
        stage3_speed_max=0.06 * _S5_METRIC_SCALE,
        stage4_speed_max=0.09 * _S5_METRIC_SCALE,
        stage1_accel_max=0.08 * _S5_METRIC_SCALE,
        stage2_accel_max=0.03 * _S5_METRIC_SCALE,
        stage3_accel_max=0.07 * _S5_METRIC_SCALE,
        stage4_accel_max=0.06 * _S5_METRIC_SCALE,
        tool_align_max_stage2=0.04,
        ang_speed_max_stage2=0.22,
        ang_speed_max_stage3=0.55,
        dt=0.8,
        noise_std=0.004 * _S5_METRIC_SCALE,
        surface_near_target_ratio=0.75,
        split_stage3_transition=False,
        transition_stage_fraction=1.0 / 3.0,
        contact_theta_range=(-0.12 * np.pi, 0.16 * np.pi),
        contact_phi_range=(0.20 * np.pi, 0.34 * np.pi),
        stage2_trace_angle_range=(1.184, 1.376),
        stage2_robot_lateral_trace=True,
        stage2_lateral_center_theta=0.0,
        stage2_lateral_phi_bump_range=(-0.035 * np.pi, 0.035 * np.pi),
        stage2_surface_detour_angle=0.0,
        stage4_shell_detour_angle=0.10,
        stage2_length_scale_range=(1.0, 1.0),
        stage4_length_scale_range=(1.0, 1.0),
        stage1_target_speed_ratio=0.68,
        stage1_speed_taper_fraction=1.0,
        stage1_speed_taper_end_ratio=0.78,
        stage2_target_speed_ratio=0.99,
        stage3_target_speed_ratio=0.75,
        stage4_target_speed_ratio=0.99,
        stage5_target_speed_ratio=0.62,
        stage2_speed_valley_depths=(0.07, 0.18, 0.07),
        stage2_speed_valley_centers=(0.30, 0.58, 0.80),
        stage2_speed_valley_widths=(0.018, 0.025, 0.018),
        stage3_speed_jitter_std=0.04,
        stage3_speed_jitter_clip=0.09,
        stage3_speed_jitter_kernel=5,
        stage4_speed_valley_depth=0.08,
        stage4_speed_valley_center=0.54,
        stage4_speed_valley_width=0.025,
        stage2_noise_scale=0.28,
        stage4_noise_scale=0.24,
        stage4_tool_normal_max_error=0.30,
        stage5_tool_normal_max_error=0.18,
        trajectory_noise_kernel=9,
        segment_count_slack=0.35,
        repos_angle_range=(0.95, 1.18),
        stage3_shell_blend_range=(0.44, 0.58),
        stage345_top_phi_range=(0.10 * np.pi, 0.18 * np.pi),
        stage345_top_theta_pull=0.45,
        stage345_top_theta_jitter=0.10 * np.pi,
        feature_boundary_ramp_half_windows=None,
        rollout_backend="analytic",
        observation_backend=None,
        pybullet_sim_dt=1.0 / 120.0,
        pybullet_steps_per_sample=None,
        pybullet_gravity_z=0.0,
        pybullet_solver_iterations=80,
        pybullet_world_scale=1.0,
        pybullet_world_center=(0.55, 0.0, 0.52),
        pybullet_ur5_urdf_path=None,
        pybullet_ur5_base_xyz=(0.0, 0.0, 0.0),
        pybullet_ur5_base_rpy=(0.0, 0.0, 0.0),
        pybullet_ur5_ee_link_index=-1,
        pybullet_ur5_tool_axis="-x",
        pybullet_ur5_tip_offset=0.0,
        pybullet_ur5_home_q=(0.0, -1.25, 1.85, -2.10, -1.57, 0.0),
        pybullet_ur5_ik_iterations=120,
        pybullet_ur5_ik_damping=0.02,
        pybullet_ur5_rest_home_blend=0.03,
        pybullet_ur5_axis_error_weight=0.02,
        pybullet_ur5_stage1_axis_error_weight=None,
        pybullet_ur5_stage1_axis_weight_ramp_points=5,
        pybullet_ur5_ik_position_error_fallback_threshold=0.0005,
        pybullet_ur5_ik_fallback_axis_error_weight=0.005,
        pybullet_filter_ik_valid=True,
        pybullet_filter_max_attempts=80,
        pybullet_filter_max_position_error=0.012 * _S5_METRIC_SCALE,
        pybullet_filter_max_axis_error=0.30,
        pybullet_filter_global_axis_error=False,
        pybullet_filter_constrained_max_axis_error=0.45,
        pybullet_filter_max_speed_ratio=1.25,
        pybullet_precheck_ik_waypoints=True,
        pybullet_precheck_points_per_stage=3,
        pybullet_suppress_urdf_warnings=True,
        pybullet_ur5_position_gain=0.08,
        pybullet_ur5_velocity_gain=1.0,
        pybullet_ur5_max_force=500.0,
        pybullet_ur5_settle_steps=None,
        pybullet_contact_surface_tol=None,
        pybullet_sphere_collision=False,
        eval_tag="S5SphereInspect",
    ):
        self.sphere_center = np.asarray(sphere_center, dtype=float)
        self.sphere_radius = float(sphere_radius)
        self.shell_thickness = float(shell_thickness)
        self.seg_lengths = tuple(int(x) for x in seg_lengths)
        self.seg_length_jitter = tuple(int(x) for x in seg_length_jitter)
        self.approach_offset = float(approach_offset)
        self.depart_offset = float(depart_offset)
        self.stage1_speed_max = float(stage1_speed_max)
        self.stage2_speed_max = float(stage2_speed_max)
        self.stage3_speed_max = float(stage3_speed_max)
        self.stage4_speed_max = float(stage4_speed_max)
        self.stage1_accel_max = float(stage1_accel_max)
        self.stage2_accel_max = float(stage2_accel_max)
        self.stage3_accel_max = float(stage3_accel_max)
        self.stage4_accel_max = float(stage4_accel_max)
        self.tool_align_max_stage2 = float(tool_align_max_stage2)
        self.ang_speed_max_stage2 = float(ang_speed_max_stage2)
        self.ang_speed_max_stage3 = float(ang_speed_max_stage3)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.surface_near_target_ratio = float(surface_near_target_ratio)
        self.split_stage3_transition = bool(split_stage3_transition)
        self.transition_stage_fraction = float(transition_stage_fraction)
        theta_lo, theta_hi = contact_theta_range
        phi_lo, phi_hi = contact_phi_range
        self.contact_theta_range = (float(theta_lo), float(theta_hi))
        self.contact_phi_range = (float(phi_lo), float(phi_hi))
        angle_lo, angle_hi = stage2_trace_angle_range
        self.stage2_trace_angle_range = (float(angle_lo), float(angle_hi))
        self.stage2_robot_lateral_trace = bool(stage2_robot_lateral_trace)
        self.stage2_lateral_center_theta = float(stage2_lateral_center_theta)
        bump_lo, bump_hi = stage2_lateral_phi_bump_range
        self.stage2_lateral_phi_bump_range = (float(bump_lo), float(bump_hi))
        self.stage2_surface_detour_angle = float(stage2_surface_detour_angle)
        self.stage4_shell_detour_angle = float(stage4_shell_detour_angle)
        stage2_scale_lo, stage2_scale_hi = stage2_length_scale_range
        stage4_scale_lo, stage4_scale_hi = stage4_length_scale_range
        self.stage2_length_scale_range = (float(stage2_scale_lo), float(stage2_scale_hi))
        self.stage4_length_scale_range = (float(stage4_scale_lo), float(stage4_scale_hi))
        self.stage1_target_speed_ratio = float(stage1_target_speed_ratio)
        self.stage1_speed_taper_fraction = float(stage1_speed_taper_fraction)
        self.stage1_speed_taper_end_ratio = (
            None if stage1_speed_taper_end_ratio is None else float(stage1_speed_taper_end_ratio)
        )
        self.stage2_target_speed_ratio = float(stage2_target_speed_ratio)
        self.stage3_target_speed_ratio = float(stage3_target_speed_ratio)
        self.stage4_target_speed_ratio = float(stage4_target_speed_ratio)
        self.stage5_target_speed_ratio = float(stage5_target_speed_ratio)
        self.stage2_speed_valley_depths = tuple(float(x) for x in np.asarray(stage2_speed_valley_depths, dtype=float).reshape(-1))
        self.stage2_speed_valley_centers = tuple(float(x) for x in np.asarray(stage2_speed_valley_centers, dtype=float).reshape(-1))
        self.stage2_speed_valley_widths = tuple(float(x) for x in np.asarray(stage2_speed_valley_widths, dtype=float).reshape(-1))
        self.stage3_speed_jitter_std = float(stage3_speed_jitter_std)
        self.stage3_speed_jitter_clip = float(stage3_speed_jitter_clip)
        self.stage3_speed_jitter_kernel = int(max(int(stage3_speed_jitter_kernel), 1))
        self.stage4_speed_valley_depth = float(stage4_speed_valley_depth)
        self.stage4_speed_valley_center = float(stage4_speed_valley_center)
        self.stage4_speed_valley_width = float(stage4_speed_valley_width)
        self.stage2_noise_scale = float(stage2_noise_scale)
        self.stage4_noise_scale = float(stage4_noise_scale)
        self.stage4_tool_normal_max_error = float(stage4_tool_normal_max_error)
        self.stage5_tool_normal_max_error = float(stage5_tool_normal_max_error)
        self.trajectory_noise_kernel = int(max(int(trajectory_noise_kernel), 1))
        self.segment_count_slack = float(segment_count_slack)
        repos_lo, repos_hi = repos_angle_range
        shell_blend_lo, shell_blend_hi = stage3_shell_blend_range
        self.repos_angle_range = (float(repos_lo), float(repos_hi))
        self.stage3_shell_blend_range = (float(shell_blend_lo), float(shell_blend_hi))
        top_phi_lo, top_phi_hi = stage345_top_phi_range
        self.stage345_top_phi_range = (float(top_phi_lo), float(top_phi_hi))
        self.stage345_top_theta_pull = float(stage345_top_theta_pull)
        self.stage345_top_theta_jitter = float(stage345_top_theta_jitter)
        self.feature_boundary_ramp_half_windows = feature_boundary_ramp_half_windows
        self.rollout_backend = str(rollout_backend).lower()
        requested_observation_backend = self.rollout_backend if observation_backend is None else observation_backend
        self.observation_backend = self._normalize_observation_backend(requested_observation_backend)
        if self.rollout_backend not in {"analytic", "pybullet"}:
            raise ValueError(f"Unsupported S5 rollout_backend '{self.rollout_backend}'.")
        if self.observation_backend not in {"analytic_raw", "pybullet"}:
            raise ValueError(f"Unsupported S5 observation_backend '{self.observation_backend}'.")
        self.pybullet_sim_dt = float(pybullet_sim_dt)
        self.pybullet_steps_per_sample = None if pybullet_steps_per_sample is None else int(pybullet_steps_per_sample)
        self.pybullet_gravity_z = float(pybullet_gravity_z)
        self.pybullet_solver_iterations = int(pybullet_solver_iterations)
        self.pybullet_world_scale = float(pybullet_world_scale)
        self.pybullet_world_center = tuple(float(x) for x in np.asarray(pybullet_world_center, dtype=float).reshape(3))
        self.pybullet_ur5_urdf_path = pybullet_ur5_urdf_path
        self.pybullet_ur5_base_xyz = tuple(float(x) for x in np.asarray(pybullet_ur5_base_xyz, dtype=float).reshape(3))
        self.pybullet_ur5_base_rpy = tuple(float(x) for x in np.asarray(pybullet_ur5_base_rpy, dtype=float).reshape(3))
        self.pybullet_ur5_ee_link_index = int(pybullet_ur5_ee_link_index)
        self.pybullet_ur5_tool_axis = str(pybullet_ur5_tool_axis)
        self.pybullet_ur5_tip_offset = float(pybullet_ur5_tip_offset)
        self.pybullet_ur5_home_q = tuple(float(x) for x in np.asarray(pybullet_ur5_home_q, dtype=float).reshape(6))
        self.pybullet_ur5_ik_iterations = int(pybullet_ur5_ik_iterations)
        self.pybullet_ur5_ik_damping = float(pybullet_ur5_ik_damping)
        self.pybullet_ur5_rest_home_blend = float(pybullet_ur5_rest_home_blend)
        self.pybullet_ur5_axis_error_weight = float(pybullet_ur5_axis_error_weight)
        self.pybullet_ur5_stage1_axis_error_weight = (
            None if pybullet_ur5_stage1_axis_error_weight is None else float(pybullet_ur5_stage1_axis_error_weight)
        )
        self.pybullet_ur5_stage1_axis_weight_ramp_points = int(max(int(pybullet_ur5_stage1_axis_weight_ramp_points), 0))
        self.pybullet_ur5_ik_position_error_fallback_threshold = float(pybullet_ur5_ik_position_error_fallback_threshold)
        self.pybullet_ur5_ik_fallback_axis_error_weight = float(pybullet_ur5_ik_fallback_axis_error_weight)
        self.pybullet_filter_ik_valid = bool(pybullet_filter_ik_valid)
        self.pybullet_filter_max_attempts = int(max(int(pybullet_filter_max_attempts), 1))
        self.pybullet_filter_max_position_error = float(pybullet_filter_max_position_error)
        self.pybullet_filter_max_axis_error = float(pybullet_filter_max_axis_error)
        self.pybullet_filter_global_axis_error = bool(pybullet_filter_global_axis_error)
        self.pybullet_filter_constrained_max_axis_error = float(pybullet_filter_constrained_max_axis_error)
        self.pybullet_filter_max_speed_ratio = float(pybullet_filter_max_speed_ratio)
        self.pybullet_precheck_ik_waypoints = bool(pybullet_precheck_ik_waypoints)
        self.pybullet_precheck_points_per_stage = int(max(int(pybullet_precheck_points_per_stage), 2))
        self.pybullet_suppress_urdf_warnings = bool(pybullet_suppress_urdf_warnings)
        self.pybullet_ur5_position_gain = float(pybullet_ur5_position_gain)
        self.pybullet_ur5_velocity_gain = float(pybullet_ur5_velocity_gain)
        self.pybullet_ur5_max_force = float(pybullet_ur5_max_force)
        self.pybullet_ur5_settle_steps = None if pybullet_ur5_settle_steps is None else int(pybullet_ur5_settle_steps)
        default_contact_tol = 0.025 * self.sphere_radius
        self.pybullet_contact_surface_tol = (
            float(default_contact_tol) if pybullet_contact_surface_tol is None else float(pybullet_contact_surface_tol)
        )
        self.pybullet_sphere_collision = bool(pybullet_sphere_collision)
        self.eval_tag = str(eval_tag)

        self.feature_schema = self.get_feature_schema()
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self._cached_tool_axis_traces = {}

        nominal_contact = self.sphere_center + np.array([0.0, self.sphere_radius, 0.0], dtype=float)
        nominal_shell = self.sphere_center + np.array(
            [0.0, self.sphere_radius + self.surface_near_target_ratio * self.shell_thickness, 0.0],
            dtype=float,
        )
        self.subgoal = nominal_contact.copy()
        self.goal = nominal_shell.copy()

    @staticmethod
    def _unit(vec):
        arr = np.asarray(vec, dtype=float)
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return arr / norm

    @staticmethod
    def _smoothstep(u):
        u = np.asarray(u, dtype=float)
        return u * u * (3.0 - 2.0 * u)

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
    def _smooth_trace(values: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        k = max(1, int(kernel_size))
        if k <= 1 or len(vals) == 0:
            return vals
        kernel = np.ones(k, dtype=float) / float(k)
        pad_left = k // 2
        pad_right = k - 1 - pad_left
        padded = np.pad(vals, (pad_left, pad_right), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    @staticmethod
    def _blend_segment_boundary(values: np.ndarray, boundary: int, half_window: int = 2) -> np.ndarray:
        out = np.asarray(values, dtype=float).copy()
        squeeze = False
        if out.ndim == 1:
            out = out[:, None]
            squeeze = True
        left = max(0, int(boundary) - int(half_window))
        right = min(len(out) - 1, int(boundary) + int(half_window) + 1)
        if left < 1 or right >= len(out) - 1 or right - left < 2:
            return out[:, 0] if squeeze else out

        p0 = out[left].copy()
        p1 = out[right].copy()
        span = float(right - left)
        m0 = (out[left] - out[left - 1]) * span
        m1 = (out[right + 1] - out[right]) * span

        u = np.linspace(0.0, 1.0, right - left + 1)
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        out[left : right + 1] = (
            h00[:, None] * p0
            + h10[:, None] * m0
            + h01[:, None] * p1
            + h11[:, None] * m1
        )
        return out[:, 0] if squeeze else out

    def _smooth_noise(self, rng, length: int, scale: float, kernel_size: int = 7) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.zeros(0, dtype=float)
        noise = rng.randn(n) * float(scale)
        return self._smooth_trace(noise, kernel_size=kernel_size)

    @staticmethod
    def _half_sine_wave(length: int, cycles: float, phase: float = 0.0) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.zeros(0, dtype=float)
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        base_wave = np.sin(2.0 * np.pi * float(cycles) * u - 0.5 * np.pi + float(phase))
        return np.maximum(base_wave, 0.0)

    def _make_stage_margin_profile(
        self,
        length: int,
        *,
        offset: float,
        amplitude: float,
        cycles: float,
        phase: float = 0.0,
        noise_scale: float = 0.0,
        rng=None,
        kernel_size: int = 5,
    ) -> np.ndarray:
        trace = float(amplitude) * self._half_sine_wave(length, cycles=cycles, phase=phase) - float(offset)
        if rng is not None and float(noise_scale) > 0.0:
            trace = trace + self._smooth_noise(rng, length, scale=float(noise_scale), kernel_size=kernel_size)
        trace = self._smooth_trace(trace, kernel_size=kernel_size)
        return np.asarray(trace, dtype=float)

    def _make_target_stage_trace(
        self,
        length: int,
        *,
        target: float,
        amplitude: float,
        cycles: float,
        phase: float = 0.0,
        noise_scale: float = 0.0,
        rng=None,
        kernel_size: int = 5,
        lower: float | None = 0.0,
        upper: float | None = None,
    ) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.zeros(0, dtype=float)
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        envelope = np.sin(np.pi * u)
        trace = float(target) + float(amplitude) * envelope * np.sin(2.0 * np.pi * float(cycles) * u + float(phase))
        if rng is not None and float(noise_scale) > 0.0:
            trace = trace + self._smooth_noise(rng, n, scale=float(noise_scale), kernel_size=kernel_size)
        trace = self._smooth_trace(trace, kernel_size=kernel_size)
        if lower is not None or upper is not None:
            lo = -np.inf if lower is None else float(lower)
            hi = np.inf if upper is None else float(upper)
            trace = np.clip(trace, lo, hi)
        return np.asarray(trace, dtype=float)

    def _make_irregular_positive_stage_trace(
        self,
        length: int,
        *,
        base: float,
        amplitude: float,
        phase: float = 0.0,
        noise_scale: float = 0.0,
        rng=None,
        kernel_size: int = 5,
        lower: float = 0.0,
        upper: float | None = None,
    ) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.zeros(0, dtype=float)
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        env1 = np.sin(np.pi * u)
        env2 = np.sin(np.pi * np.clip(1.15 * u, 0.0, 1.0)) ** 1.15
        wave = (
            0.55 * env1 * np.sin(4.5 * np.pi * u + float(phase))
            + 0.35 * env2 * np.sin(7.8 * np.pi * u - 0.55 * float(phase))
            + 0.20 * np.sin(11.6 * np.pi * u + 0.7 * float(phase))
        )
        trace = float(base) + float(amplitude) * wave
        if rng is not None and float(noise_scale) > 0.0:
            trace = trace + self._smooth_noise(rng, n, scale=float(noise_scale), kernel_size=kernel_size)
        trace = self._smooth_trace(trace, kernel_size=kernel_size)
        hi = np.inf if upper is None else float(upper)
        return np.clip(trace, float(lower), hi)

    def register_tool_axis_trace(self, traj: np.ndarray, tool_axis: np.ndarray):
        self._cached_tool_axis_traces[self._traj_cache_key(traj)] = np.asarray(tool_axis, dtype=float).copy()

    def _lookup_cached_tool_axis_trace(self, traj: np.ndarray):
        axis = self._cached_tool_axis_traces.get(self._traj_cache_key(traj))
        if axis is None:
            return None
        return np.asarray(axis, dtype=float)

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "surf_dist", "description": "Absolute radial distance to the sphere surface"},
            {"id": 1, "name": "normal_err", "description": "Angle between tool axis and sphere normal"},
            {"id": 2, "name": "speed", "description": "3D speed magnitude"},
            {"id": 3, "name": "ang_speed", "description": "Tool-axis angular speed magnitude"},
            {"id": 4, "name": "noise", "description": "Deterministic auxiliary irrelevant feature"},
            {"id": 5, "name": "start_dist", "description": "3D distance to the demo start position"},
            {"id": 6, "name": "goal_dist", "description": "3D distance to the nominal inspection goal"},
        ]

    def get_true_constraints(self):
        base = {
            "surface_trace_target": 0.0,
            "surface_near_target": float(self.surface_near_target_ratio * self.shell_thickness),
            "surface_trace_max": float(0.018 * self.sphere_radius),
            "surface_near_max": float(self.shell_thickness),
            "tool_align_max_stage2": float(self.tool_align_max_stage2),
            "v23_max": float(self.stage2_speed_max),
        }
        return base

    def get_constraint_specs(self):
        if self.split_stage3_transition:
            return [
                {"feature_name": "surf_dist", "stage": 1, "semantics": "target_value", "oracle_key": "surface_trace_target"},
                {"feature_name": "normal_err", "stage": 1, "semantics": "upper_bound", "oracle_key": "tool_align_max_stage2"},
                {"feature_name": "speed", "stage": 1, "semantics": "upper_bound", "oracle_key": "v23_max"},
                {"feature_name": "surf_dist", "stage": 3, "semantics": "target_value", "oracle_key": "surface_near_target"},
                {"feature_name": "speed", "stage": 3, "semantics": "upper_bound", "oracle_key": "v23_max"},
            ]
        return [
            {"feature_name": "surf_dist", "stage": 1, "semantics": "target_value", "oracle_key": "surface_trace_target"},
            {"feature_name": "normal_err", "stage": 1, "semantics": "upper_bound", "oracle_key": "tool_align_max_stage2"},
            {"feature_name": "speed", "stage": 1, "semantics": "upper_bound", "oracle_key": "v23_max"},
            {"feature_name": "surf_dist", "stage": 2, "semantics": "target_value", "oracle_key": "surface_near_target"},
            {"feature_name": "speed", "stage": 2, "semantics": "upper_bound", "oracle_key": "v23_max"},
        ]

    def get_observation_spec(self):
        return {
            "feature_schema": self.get_feature_schema(),
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

    def sample_scene(self, seed=None, rng=None):
        return {
            "task_name": str(self.eval_tag),
            "geometry": {
                "sphere_center": self.sphere_center.tolist(),
                "sphere_radius": float(self.sphere_radius),
                "shell_thickness": float(self.shell_thickness),
                "surface_near_target_ratio": float(self.surface_near_target_ratio),
            },
            "task": {
                "split_stage3_transition": bool(self.split_stage3_transition),
                "transition_stage_fraction": float(self.transition_stage_fraction),
                "contact_theta_range": list(self.contact_theta_range),
                "contact_phi_range": list(self.contact_phi_range),
                "stage2_trace_angle_range": list(self.stage2_trace_angle_range),
                "stage2_robot_lateral_trace": bool(self.stage2_robot_lateral_trace),
                "stage2_lateral_center_theta": float(self.stage2_lateral_center_theta),
                "stage2_lateral_phi_bump_range": list(self.stage2_lateral_phi_bump_range),
                "repos_angle_range": list(self.repos_angle_range),
                "stage3_shell_blend_range": list(self.stage3_shell_blend_range),
                "stage345_top_phi_range": list(self.stage345_top_phi_range),
                "stage345_top_theta_pull": float(self.stage345_top_theta_pull),
                "stage345_top_theta_jitter": float(self.stage345_top_theta_jitter),
                "stage4_shell_detour_angle": float(self.stage4_shell_detour_angle),
                "stage2_speed_valley": {
                    "depths": list(self.stage2_speed_valley_depths),
                    "centers": list(self.stage2_speed_valley_centers),
                    "widths": list(self.stage2_speed_valley_widths),
                },
                "stage3_speed_jitter": {
                    "std": float(self.stage3_speed_jitter_std),
                    "clip": float(self.stage3_speed_jitter_clip),
                    "kernel": int(self.stage3_speed_jitter_kernel),
                },
                "stage4_speed_valley": {
                    "depth": float(self.stage4_speed_valley_depth),
                    "center": float(self.stage4_speed_valley_center),
                    "width": float(self.stage4_speed_valley_width),
                },
                "tool_axis": {
                    "stage4_normal_max_error": float(self.stage4_tool_normal_max_error),
                    "stage5_normal_max_error": float(self.stage5_tool_normal_max_error),
                },
                "trajectory_noise": {
                    "noise_std": float(self.noise_std),
                    "kernel": int(self.trajectory_noise_kernel),
                    "stage2_scale": float(self.stage2_noise_scale),
                    "stage4_scale": float(self.stage4_noise_scale),
                },
                "stage2_length_scale_range": list(self.stage2_length_scale_range),
                "stage4_length_scale_range": list(self.stage4_length_scale_range),
            },
        }

    def _rollout_demo_analytic(self, scene, seed=None, rng=None, **kwargs):
        if scene is not None and "demo_index" in scene and "demo_index" not in kwargs:
            kwargs["demo_index"] = int(scene["demo_index"])
        if rng is not None:
            traj, cutpoints = self.generate_demo(rng=rng, **kwargs)
        else:
            local_seed = int(seed) if seed is not None else int((scene or {}).get("rollout_seed", 0))
            local_rng = np.random.RandomState(local_seed)
            traj, cutpoints = self.generate_demo(rng=local_rng, **kwargs)
        tool_axis = self._lookup_cached_tool_axis_trace(traj)
        return {
            "trajectory": np.asarray(traj, dtype=float),
            "true_cutpoints": np.asarray(cutpoints, dtype=int),
            "tool_axis": None if tool_axis is None else np.asarray(tool_axis, dtype=float),
            "rollout_backend": "analytic",
            "observation_backend": str(self.observation_backend),
        }

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
                latent["ik_filter"] = dict(report)
                if precheck_report is not None:
                    latent["ik_filter"]["precheck"] = dict(precheck_report)
                latent["ik_filter"]["attempt"] = int(attempt)
                latent["ik_filter"]["max_attempts"] = int(max_attempts)
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
        return self._resample_fixed_count_with_speed_profile(raw, int(n_points))

    def _normal_with_geodesic_angle(self, n_start, *, tangent_hint, angle: float, sign: float = 1.0):
        n_start = self._unit(n_start)
        tangent = np.asarray(tangent_hint, dtype=float).reshape(3)
        tangent = tangent - float(np.dot(tangent, n_start)) * n_start
        if float(np.linalg.norm(tangent)) <= 1e-10:
            _, tangent, _ = self._orthonormal_frame(n_start, np.random.RandomState(0))
        tangent = self._unit(tangent)
        angle = float(np.clip(angle, 0.0, np.pi - 1e-5))
        return self._unit(np.cos(angle) * n_start + float(sign) * np.sin(angle) * tangent)

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
        stage1 = self._resample_fixed_count_with_speed_profile(stage1_ctrl, l1)
        stage2 = self._clean_shell_arc_by_speed(n0, n1, radius_offset=s2_surf, n_points=l2)

        stage3_normals = self._slerp_unit(n1, n4_start, l3, endpoint=True)
        u3 = np.linspace(0.0, 1.0, l3, endpoint=True)
        stage3_radius = (1.0 - u3) * r2 + u3 * r4
        stage3 = self.sphere_center[None, :] + stage3_radius[:, None] * stage3_normals

        stage4 = self._clean_shell_arc_by_speed(n4_start, n4_end, radius_offset=s4_surf, n_points=l4)
        p_depart = self.sphere_center + (r4 + self.depart_offset) * n4_end
        stage5_ctrl = np.vstack([stage4[-1], 0.55 * stage4[-1] + 0.45 * p_depart, p_depart])
        stage5 = self._resample_fixed_count_with_speed_profile(stage5_ctrl, l5)

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

    def _assemble_feature_matrix(self, traj, *, tool_axis=None, use_cached=True):
        traj = np.asarray(traj, dtype=float)
        surf_dist, normal_err, speed, ang_speed = self._compute_geometry_feature_traces(traj, tool_axis=tool_axis)

        return {
            "surf_dist": np.asarray(surf_dist, dtype=float),
            "normal_err": np.asarray(normal_err, dtype=float),
            "speed": np.asarray(speed, dtype=float),
            "ang_speed": np.asarray(ang_speed, dtype=float),
        }

    def compute_all_features_matrix(self, traj, feat_ids=None, *, tool_axis=None, use_cached=None):
        traj = np.asarray(traj, dtype=float)
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
        goal_dist = np.linalg.norm(traj - self.goal[None, :], axis=1)

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
        if active_backend == "analytic_raw":
            features = np.asarray(self.compute_all_features_matrix(traj, tool_axis=tool_axis, use_cached=False), dtype=float)
        elif active_backend == "pybullet":
            features = np.asarray(self.compute_all_features_matrix(traj, tool_axis=tool_axis, use_cached=False), dtype=float)
        else:
            raise ValueError(f"Unsupported S5 observation backend '{active_backend}'.")
        observation = {
            "trajectory": traj,
            "features": features,
            "true_cutpoints": np.asarray(latent_rollout.get("true_cutpoints", []), dtype=int),
            "feature_schema": self.get_feature_schema(),
            "observation_spec": self.get_observation_spec(),
            "tool_axis": tool_axis,
            "scene": dict(scene or {}),
        }
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

    def render_episode(self, scene, trajectory, output_path, **kwargs):
        geometry = dict((scene or {}).get("geometry", {}))
        camera_name = str(kwargs.get("camera", "default_3d"))
        presets = self.get_render_camera_presets()
        preset = dict(presets.get(camera_name, presets["default_3d"]))
        backend = str(kwargs.get("backend", preset.get("backend", "matplotlib"))).lower()
        traj = np.asarray(trajectory, dtype=float)[:, :3]
        sphere_center = geometry.get("sphere_center", self.sphere_center.tolist())
        sphere_radius = float(geometry.get("sphere_radius", self.sphere_radius))
        if backend in {"matplotlib", "mpl"}:
            return render_sphere_episode(
                trajectory=traj,
                output_path=output_path,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                cutpoints=kwargs.get("cutpoints"),
                title=kwargs.get("title", str(self.eval_tag)),
                elev=float(kwargs.get("elev", preset.get("elev", 24.0))),
                azim=float(kwargs.get("azim", preset.get("azim", 38.0))),
            )
        if backend == "pybullet":
            tool_axis = kwargs.get("tool_axis")
            if tool_axis is None:
                tool_axis = self._lookup_cached_tool_axis_trace(traj)
            if tool_axis is None:
                tool_axis = self._estimate_tool_axis_from_geometry(traj)
            return render_s5_pybullet_episode(
                trajectory=traj,
                output_path=output_path,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                cutpoints=kwargs.get("cutpoints"),
                overlay_cutpoints=kwargs.get("overlay_cutpoints"),
                tool_axis=np.asarray(tool_axis, dtype=float),
                title=kwargs.get("title", str(self.eval_tag)),
                center_world=kwargs.get("center_world", self.pybullet_world_center),
                world_scale=float(kwargs.get("world_scale", self.pybullet_world_scale)),
                main_yaw=float(kwargs.get("main_yaw", preset.get("main_yaw", 42.0))),
                inset_yaw=float(kwargs.get("inset_yaw", preset.get("inset_yaw", 205.0))),
                main_pitch=float(kwargs.get("main_pitch", -18.0)),
                inset_pitch=float(kwargs.get("inset_pitch", -16.0)),
                main_distance=float(kwargs.get("main_distance", 1.42)),
                inset_distance=float(kwargs.get("inset_distance", 1.46)),
                tube_radius=float(kwargs.get("tube_radius", 0.0065)),
            )
        if backend in {"pybullet_video", "video"}:
            tool_axis = kwargs.get("tool_axis")
            if tool_axis is None:
                tool_axis = self._lookup_cached_tool_axis_trace(traj)
            if tool_axis is None:
                tool_axis = self._estimate_tool_axis_from_geometry(traj)
            return render_s5_pybullet_demo_video(
                trajectory=traj,
                output_path=output_path,
                sphere_center=sphere_center,
                sphere_radius=sphere_radius,
                cutpoints=kwargs.get("cutpoints"),
                tool_axis=np.asarray(tool_axis, dtype=float),
                joint_positions=kwargs.get("joint_positions"),
                title=kwargs.get("title", str(self.eval_tag)),
                center_world=kwargs.get("center_world", self.pybullet_world_center),
                world_scale=float(kwargs.get("world_scale", self.pybullet_world_scale)),
                urdf_path=kwargs.get("urdf_path", self.pybullet_ur5_urdf_path),
                ur5_base_xyz=kwargs.get("ur5_base_xyz", self.pybullet_ur5_base_xyz),
                ur5_base_rpy=kwargs.get("ur5_base_rpy", self.pybullet_ur5_base_rpy),
                gui=int(kwargs.get("gui", 1)),
                fps=float(kwargs.get("fps", 30.0)),
                width=int(kwargs.get("width", 1024)),
                height=int(kwargs.get("height", 768)),
                render_frame_stride=int(kwargs.get("render_frame_stride", 1)),
                realtime=bool(kwargs.get("realtime", False)),
                gui_hold_seconds=float(kwargs.get("gui_hold_seconds", 0.0)),
                camera_yaw=float(kwargs.get("camera_yaw", preset.get("main_yaw", 90.0))),
                camera_pitch=float(kwargs.get("camera_pitch", -34.0)),
                camera_distance=float(kwargs.get("camera_distance", 1.62)),
                camera_target=kwargs.get("camera_target"),
                camera_fov=float(kwargs.get("camera_fov", 38.0)),
                tube_radius=float(kwargs.get("tube_radius", 0.0055)),
                stage4_shell_offset=float(
                    kwargs.get("stage4_shell_offset", self.get_true_constraints().get("surface_near_target", 0.0))
                ),
                sphere_texture_name=str(kwargs.get("sphere_texture_name", "")),
                trace_stride=int(kwargs.get("trace_stride", 1)),
                draw_stage_trace=bool(kwargs.get("draw_stage_trace", True)),
                draw_executed_trace=bool(kwargs.get("draw_executed_trace", True)),
                trace_width=float(kwargs.get("trace_width", 3.0)),
                draw_current_marker=bool(kwargs.get("draw_current_marker", False)),
                hide_gripper=bool(kwargs.get("hide_gripper", True)),
                draw_tool_bar=bool(kwargs.get("draw_tool_bar", False)),
                tool_bar_length=float(kwargs.get("tool_bar_length", 0.205)),
                tool_bar_radius=float(kwargs.get("tool_bar_radius", 0.005)),
                suppress_urdf_warnings=bool(
                    kwargs.get("suppress_urdf_warnings", self.pybullet_suppress_urdf_warnings)
                ),
                connect_client=bool(kwargs.get("connect_client", True)),
                feature_overlay=bool(kwargs.get("feature_overlay", False)),
                feature_overlay_features=kwargs.get("feature_overlay_features"),
                feature_overlay_names=kwargs.get("feature_overlay_names"),
                feature_overlay_specs=kwargs.get("feature_overlay_specs"),
                feature_overlay_true_constraints=kwargs.get("feature_overlay_true_constraints"),
                feature_overlay_title=kwargs.get("feature_overlay_title"),
                playback_speed=float(kwargs.get("playback_speed", 1.0)),
                playback_label=kwargs.get("playback_label"),
                save_frame_indices=kwargs.get("save_frame_indices"),
                save_frame_dir=kwargs.get("save_frame_dir"),
                save_frame_prefix=str(kwargs.get("save_frame_prefix", "s5_frame")),
            )
        raise ValueError(f"Unsupported S5 render backend '{backend}'.")

    def _sample_segment_lengths(self, rng):
        out = []
        for base, jitter in zip(self.seg_lengths, self.seg_length_jitter):
            delta = 0 if int(jitter) <= 0 else int(rng.randint(-int(jitter), int(jitter) + 1))
            out.append(max(int(base) + delta, 8))
        return tuple(out)

    def _orthonormal_frame(self, normal, rng):
        normal = self._unit(normal)
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(normal, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        t1 = np.cross(normal, ref)
        t1 = self._unit(t1)
        if float(np.linalg.norm(t1)) <= 1e-12:
            ref = np.array([1.0, 0.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(normal, ref))
        phase = float(rng.uniform(-np.pi, np.pi))
        t1 = self._unit(np.cos(phase) * t1 + np.sin(phase) * np.cross(normal, t1))
        t2 = self._unit(np.cross(normal, t1))
        return normal, t1, t2

    def _normal_from_spherical(self, theta, phi):
        theta = float(theta)
        phi = float(phi)
        return self._unit(
            [
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            ]
        )

    def _spherical_from_normal(self, normal):
        normal = self._unit(normal)
        theta = float(np.arctan2(normal[1], normal[0]))
        phi = float(np.arccos(np.clip(normal[2], -1.0, 1.0)))
        return theta, phi

    def _make_latitude_surface_path(
        self,
        theta_start,
        theta_end,
        phi,
        num_points,
        *,
        radius_offset=0.0,
        phi_bump=0.0,
    ):
        theta = np.linspace(float(theta_start), float(theta_end), int(num_points), endpoint=True)
        u = np.linspace(0.0, 1.0, int(num_points), endpoint=True)
        phi = float(phi) + float(phi_bump) * np.sin(np.pi * u)
        phi = np.clip(phi, 0.08 * np.pi, 0.48 * np.pi)
        normals = np.stack(
            [
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            ],
            axis=1,
        )
        normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        radius = float(self.sphere_radius + float(radius_offset))
        return self.sphere_center[None, :] + radius * normals

    def _slerp_unit(self, u0, u1, num_points, endpoint=True):
        u0 = self._unit(u0)
        u1 = self._unit(u1)
        dots = float(np.clip(np.dot(u0, u1), -1.0, 1.0))
        if dots > 0.9995:
            t = np.linspace(0.0, 1.0, int(num_points), endpoint=endpoint)
            out = (1.0 - t)[:, None] * u0[None, :] + t[:, None] * u1[None, :]
            return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
        omega = float(np.arccos(dots))
        sin_omega = float(np.sin(omega))
        t = np.linspace(0.0, 1.0, int(num_points), endpoint=endpoint)
        out = (
            np.sin((1.0 - t) * omega)[:, None] * u0[None, :]
            + np.sin(t * omega)[:, None] * u1[None, :]
        ) / max(sin_omega, 1e-12)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)

    def _make_spherical_shell_path(self, n_start, n_end, num_points, *, radius_offset=0.0, detour_angle=None):
        normals = self._slerp_unit(n_start, n_end, num_points, endpoint=True)
        detour_angle = self.stage2_surface_detour_angle if detour_angle is None else detour_angle
        detour_angle = float(max(detour_angle, 0.0))
        if detour_angle > 1e-8 and len(normals) > 2:
            axis = np.cross(self._unit(n_start), self._unit(n_end))
            if float(np.linalg.norm(axis)) <= 1e-8:
                _, _, detour_dir = self._orthonormal_frame(n_start, np.random.RandomState(0))
            else:
                axis = self._unit(axis)
                detour_dir = np.cross(axis, normals)
                detour_dir = detour_dir / np.maximum(np.linalg.norm(detour_dir, axis=1, keepdims=True), 1e-12)
            u = np.linspace(0.0, 1.0, len(normals), endpoint=True)
            bend = detour_angle * np.sin(np.pi * u)
            normals = (
                np.cos(bend)[:, None] * normals
                + np.sin(bend)[:, None] * detour_dir
            )
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        radius = float(self.sphere_radius + float(radius_offset))
        return self.sphere_center[None, :] + radius * normals

    def _make_surface_path(self, n_start, n_end, num_points):
        return self._make_spherical_shell_path(n_start, n_end, num_points, radius_offset=0.0)

    def _build_stage3_transition(self, n_start, n_end, n_points, shell_offset, rng):
        n_points = int(max(n_points, 8))
        n_start = self._unit(n_start)
        n_end = self._unit(n_end)
        normals = self._slerp_unit(n_start, n_end, n_points, endpoint=True)
        if len(normals) > 2:
            phase = float(rng.uniform(-0.25 * np.pi, 0.25 * np.pi))
            u = np.linspace(0.0, 1.0, len(normals), endpoint=True)
            envelope = np.sin(np.pi * u) ** 1.05
            wiggle = 0.035 * np.sin(2.2 * np.pi * u + phase) * envelope
            ref = np.cross(n_start, n_end)
            if float(np.linalg.norm(ref)) <= 1e-8:
                _, _, ref = self._orthonormal_frame(n_start, rng)
            ref = self._unit(ref)
            detour = np.cross(ref[None, :], normals)
            detour = detour / np.maximum(np.linalg.norm(detour, axis=1, keepdims=True), 1e-12)
            normals = normals + wiggle[:, None] * detour
            normals = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

        u = np.linspace(0.0, 1.0, len(normals), endpoint=True)
        radial_progress = u
        radius = self.sphere_radius + float(shell_offset) * radial_progress
        return self.sphere_center[None, :] + radius[:, None] * normals

    @staticmethod
    def _polyline_length(path) -> float:
        pts = np.asarray(path, dtype=float)
        if len(pts) <= 1:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    @staticmethod
    def _sample_polyline_at_distances(path, sample_distances):
        pts = np.asarray(path, dtype=float)
        distances = np.asarray(sample_distances, dtype=float).reshape(-1)
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=float)
        if len(pts) == 1:
            return np.repeat(pts, len(distances), axis=0)
        edges = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(edges)])
        total = float(cum[-1])
        if total <= 1e-12:
            return np.repeat(pts[:1], len(distances), axis=0)
        d = np.clip(distances, 0.0, total)
        out = np.empty((len(d), pts.shape[1]), dtype=float)
        for i, target in enumerate(d):
            idx = int(np.searchsorted(cum, target, side="right") - 1)
            idx = max(0, min(idx, len(pts) - 2))
            span = float(cum[idx + 1] - cum[idx])
            alpha = 0.0 if span <= 1e-12 else float((target - cum[idx]) / span)
            out[i] = (1.0 - alpha) * pts[idx] + alpha * pts[idx + 1]
        return out

    def _make_cruise_valley_weights(self, num_edges: int, *, depth, center, width) -> np.ndarray:
        n = int(max(num_edges, 1))
        if n == 1:
            return np.ones(1, dtype=float)
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        depths = np.asarray(depth, dtype=float).reshape(-1)
        centers = np.asarray(center, dtype=float).reshape(-1)
        widths = np.asarray(width, dtype=float).reshape(-1)
        count = max(len(depths), len(centers), len(widths))
        if len(depths) == 1 and count > 1:
            depths = np.repeat(depths, count)
        if len(centers) == 1 and count > 1:
            centers = np.repeat(centers, count)
        if len(widths) == 1 and count > 1:
            widths = np.repeat(widths, count)
        if not (len(depths) == len(centers) == len(widths)):
            raise ValueError("depth, center, and width must broadcast to the same number of valleys.")
        valley = np.zeros_like(u)
        for d, c, w in zip(depths, centers, widths):
            c = float(np.clip(c, 0.08, 0.92))
            w = float(max(w, 0.015))
            d = float(np.clip(d, 0.0, 0.45))
            valley = valley + d * np.exp(-0.5 * ((u - c) / w) ** 2)
        weights = 1.0 - valley
        weights = self._smooth_trace(weights, kernel_size=3)
        return np.clip(weights, 0.55, None)

    @staticmethod
    def _stabilize_tail_weights(weights, *, tail_len: int = 3, floor_ratio: float = 0.94):
        arr = np.asarray(weights, dtype=float).copy()
        n = len(arr)
        tail_len = int(max(tail_len, 0))
        if n <= 2 or tail_len <= 0:
            return arr
        start = max(0, n - tail_len)
        anchor_lo = max(0, start - 3)
        anchor = float(np.mean(arr[anchor_lo:start])) if start > anchor_lo else float(arr[start - 1])
        floor = float(floor_ratio) * anchor
        arr[start:] = np.maximum(arr[start:], floor)
        return arr

    def _sample_stage3_speed_profile_weights(self, num_edges: int, rng) -> np.ndarray:
        n = int(max(num_edges, 1))
        if n == 1:
            return np.ones(1, dtype=float)
        local_rng = np.random if rng is None else rng
        noise = np.asarray(local_rng.randn(n), dtype=float)
        noise = self._smooth_trace(noise, kernel_size=self.stage3_speed_jitter_kernel)
        noise = noise - float(np.mean(noise))
        scale = float(np.std(noise))
        if scale > 1e-8:
            noise = noise / scale
        jitter_std = float(max(self.stage3_speed_jitter_std, 0.0))
        clip = float(max(self.stage3_speed_jitter_clip, 0.0))
        weights = 1.0 + jitter_std * noise
        if clip > 0.0:
            weights = np.clip(weights, 1.0 - clip, 1.0 + clip)
        weights = self._smooth_trace(weights, kernel_size=3)
        mean = float(np.mean(weights))
        if mean > 1e-8:
            weights = weights / mean
        return np.clip(weights, 1e-6, None)

    def _stage4_speed_profile_weights(self, num_edges: int) -> np.ndarray:
        return self._stabilize_tail_weights(
            self._make_cruise_valley_weights(
                num_edges,
                depth=self.stage4_speed_valley_depth,
                center=self.stage4_speed_valley_center,
                width=self.stage4_speed_valley_width,
            ),
            tail_len=2,
            floor_ratio=0.98,
        )

    def _resample_fixed_count_with_speed_profile(self, path, num_points: int, *, speed_profile_weights=None):
        pts = np.asarray(path, dtype=float)
        target_count = int(max(int(num_points), 2))
        if len(pts) <= 1 or target_count <= 1:
            return pts.copy()
        path_length = self._polyline_length(pts)
        if path_length <= 1e-10:
            return np.repeat(pts[:1], target_count, axis=0)
        num_edges = target_count - 1
        if speed_profile_weights is None:
            dists = np.linspace(0.0, path_length, target_count, endpoint=True)
        else:
            weights = np.asarray(speed_profile_weights(num_edges), dtype=float).reshape(-1)
            if weights.size != num_edges:
                raise ValueError(f"speed_profile_weights produced {weights.size} values, expected {num_edges}.")
            weights = np.clip(weights, 1e-6, None)
            step_lengths = path_length * (weights / np.sum(weights))
            dists = np.concatenate([[0.0], np.cumsum(step_lengths)])
            dists[-1] = path_length
        return self._sample_polyline_at_distances(pts, dists)

    def _project_to_shell(self, path, *, shell_offset):
        pts = np.asarray(path, dtype=float)
        rel = pts - self.sphere_center[None, :]
        rel_norm = np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
        shell_radius = float(self.sphere_radius + shell_offset)
        return self.sphere_center[None, :] + shell_radius * rel / rel_norm

    def _soften_stage3_radial_profile(self, path, *, shell_offset, blend=0.7):
        pts = np.asarray(path, dtype=float)
        if len(pts) <= 1:
            return pts.copy()
        rel = pts - self.sphere_center[None, :]
        rel_norm = np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
        normals = rel / rel_norm
        u = np.linspace(0.0, 1.0, len(pts), endpoint=True)
        current_radius = rel_norm.reshape(-1)
        linear_radius = self.sphere_radius + float(shell_offset) * u
        blend = float(np.clip(blend, 0.0, 1.0))
        radius = (1.0 - blend) * current_radius + blend * linear_radius
        return self.sphere_center[None, :] + radius[:, None] * normals

    def _regularize_stage3_transition_path(
        self,
        path,
        *,
        shell_offset,
        radial_blend=0.8,
        normal_kernel=5,
        dense_factor=6,
        speed_profile_weights=None,
    ):
        pts = np.asarray(path, dtype=float)
        target_count = len(pts)
        if target_count <= 1:
            return pts.copy()
        if target_count <= 3:
            out = self._soften_stage3_radial_profile(pts, shell_offset=shell_offset, blend=radial_blend)
            return self._resample_fixed_count_with_speed_profile(
                out,
                target_count,
                speed_profile_weights=speed_profile_weights,
            )

        rel = pts - self.sphere_center[None, :]
        rel_norm = np.maximum(np.linalg.norm(rel, axis=1), 1e-12)
        normals = rel / rel_norm[:, None]

        smooth_normals = np.stack(
            [self._smooth_trace(normals[:, dim], kernel_size=normal_kernel) for dim in range(normals.shape[1])],
            axis=1,
        )
        smooth_normals[0] = normals[0]
        smooth_normals[-1] = normals[-1]
        smooth_normals = smooth_normals / np.maximum(np.linalg.norm(smooth_normals, axis=1, keepdims=True), 1e-12)

        u = np.linspace(0.0, 1.0, target_count, endpoint=True)
        linear_radius = self.sphere_radius + float(shell_offset) * u
        blend = float(np.clip(radial_blend, 0.0, 1.0))
        radius = (1.0 - blend) * rel_norm + blend * linear_radius
        radius[0] = self.sphere_radius
        radius[-1] = self.sphere_radius + float(shell_offset)

        dense_count = max(int(target_count) * int(max(dense_factor, 1)), 32)
        dense_u = np.linspace(0.0, 1.0, dense_count, endpoint=True)
        dense_normals = np.stack(
            [np.interp(dense_u, u, smooth_normals[:, dim]) for dim in range(smooth_normals.shape[1])],
            axis=1,
        )
        dense_normals = dense_normals / np.maximum(np.linalg.norm(dense_normals, axis=1, keepdims=True), 1e-12)
        dense_radius = np.interp(dense_u, u, radius)
        dense_path = self.sphere_center[None, :] + dense_radius[:, None] * dense_normals
        return self._resample_fixed_count_with_speed_profile(
            dense_path,
            target_count,
            speed_profile_weights=speed_profile_weights,
        )

    def _regularize_tail_spacing(self, path, *, tail_points: int = 5):
        pts = np.asarray(path, dtype=float).copy()
        tail_points = int(max(tail_points, 0))
        if len(pts) < max(tail_points, 3):
            return pts
        start = len(pts) - tail_points
        tail = pts[start:].copy()
        edges = np.linalg.norm(np.diff(tail, axis=0), axis=1)
        total = float(np.sum(edges))
        if total <= 1e-10:
            return pts
        dists = np.linspace(0.0, total, len(tail), endpoint=True)
        pts[start:] = self._sample_polyline_at_distances(tail, dists)
        return pts

    def _repair_stage4_departure_tail(self, path, departure_target, *, shell_offset, tail_points: int = 9):
        pts = np.asarray(path, dtype=float).copy()
        if len(pts) < 4:
            return pts
        tail_points = int(max(tail_points, 2))
        departure_target = np.asarray(departure_target, dtype=float).reshape(3)
        shell_radius = float(self.sphere_radius + shell_offset)

        rel = pts - self.sphere_center[None, :]
        normals = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
        pts = self.sphere_center[None, :] + shell_radius * normals
        end_normal = self._unit(normals[-1])
        dep = departure_target - pts[-1]
        dep_norm = float(np.linalg.norm(dep))
        if dep_norm <= 1e-10:
            return pts
        dep_u = dep / dep_norm

        check_start = max(0, len(pts) - tail_points)
        tail_edges = np.diff(pts[check_start:], axis=0)
        if len(tail_edges) > 0 and float(np.min(tail_edges @ dep_u)) >= -1e-6:
            return pts

        original_edge_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        nominal_step = float(np.median(original_edge_lengths)) if len(original_edge_lengths) else 0.0
        nominal_step = max(nominal_step, 1e-6)
        best = pts
        best_min_proj = -float("inf")
        first_start = max(1, len(pts) - tail_points)
        for start in range(first_start, 0, -1):
            candidate = pts.copy()
            anchor_normal = self._unit(normals[start - 1])
            repaired_normals = self._slerp_unit(anchor_normal, end_normal, len(pts) - start + 1, endpoint=True)
            candidate[start:] = self.sphere_center[None, :] + shell_radius * repaired_normals[1:]
            candidate[-1] = self.sphere_center + shell_radius * end_normal
            candidate_edges = np.diff(candidate[check_start:], axis=0)
            min_proj = float(np.min(candidate_edges @ dep_u)) if len(candidate_edges) else 0.0
            candidate_step = float(np.max(np.linalg.norm(np.diff(candidate[start - 1 :], axis=0), axis=1)))
            if min_proj > best_min_proj:
                best_min_proj = min_proj
                best = candidate
            if min_proj >= -1e-6:
                return candidate

        tangent = dep_u - float(np.dot(dep_u, end_normal)) * end_normal
        tangent_norm = float(np.linalg.norm(tangent))
        if tangent_norm > 1e-10:
            tangent = tangent / tangent_norm
            for start in range(len(pts) - 2, first_start - 1, -1):
                candidate = pts.copy()
                anchor_normal = self._unit(normals[start - 1])
                span = float(np.dot(end_normal - anchor_normal, tangent))
                if span <= 1e-4:
                    continue
                span = float(np.clip(span, 0.02, 0.18))
                alphas = np.linspace(span, 0.0, len(pts) - start, endpoint=True)
                repaired_normals = end_normal[None, :] - alphas[:, None] * tangent[None, :]
                repaired_normals = repaired_normals / np.maximum(
                    np.linalg.norm(repaired_normals, axis=1, keepdims=True), 1e-12
                )
                candidate[start:] = self.sphere_center[None, :] + shell_radius * repaired_normals
                candidate[-1] = self.sphere_center + shell_radius * end_normal
                candidate_edges = np.diff(candidate[check_start:], axis=0)
                min_proj = float(np.min(candidate_edges @ dep_u)) if len(candidate_edges) else 0.0
                candidate_step = float(np.max(np.linalg.norm(np.diff(candidate[start - 1 :], axis=0), axis=1)))
                if min_proj > best_min_proj:
                    best_min_proj = min_proj
                    best = candidate
                if min_proj >= -1e-6 and candidate_step <= 3.0 * nominal_step:
                    return candidate
        return best

    def _resample_with_speed(
        self,
        path,
        v_max,
        a_max,
        *,
        target_speed=None,
        nominal_count=None,
        use_optimizer=True,
        speed_profile_weights=None,
    ):
        pts = np.asarray(path, dtype=float)
        if len(pts) <= 2:
            return pts.copy()

        v_max = float(v_max)
        if target_speed is None:
            target_speed = 0.78 * v_max
        target_speed = float(np.clip(target_speed, 1e-4, max(1e-4, 0.995 * v_max)))

        path_length = self._polyline_length(pts)
        if path_length <= 1e-10:
            ref = pts.copy()
        else:
            derived_count = int(np.ceil(path_length / max(target_speed * self.dt, 1e-6))) + 1
            if nominal_count is not None:
                nominal = max(int(nominal_count), 2)
                slack = max(float(self.segment_count_slack), 0.0)
                lo = max(2, int(np.floor((1.0 - slack) * nominal)))
                hi = max(lo + 1, int(np.ceil((1.0 + slack) * nominal)))
                target_count = int(np.clip(derived_count, lo, hi))
            else:
                target_count = max(2, int(derived_count))
            num_edges = max(target_count - 1, 1)
            if speed_profile_weights is None:
                max_step = path_length / max(num_edges, 1)
                ref = resample_polyline(pts, max_step=max(max_step, 1e-6))
            else:
                weights = np.asarray(speed_profile_weights(num_edges), dtype=float).reshape(-1)
                if weights.size != num_edges:
                    raise ValueError(f"speed_profile_weights produced {weights.size} values, expected {num_edges}.")
                weights = np.clip(weights, 1e-6, None)
                step_lengths = path_length * (weights / np.sum(weights))
                dists = np.concatenate([[0.0], np.cumsum(step_lengths)])
                dists[-1] = path_length
                ref = self._sample_polyline_at_distances(pts, dists)
        if not bool(use_optimizer):
            return np.asarray(ref, dtype=float)
        return optimize_trajectory(
            ref,
            dt=self.dt,
            v_max=v_max,
            a_max=float(a_max),
            projector=None,
        )

    @staticmethod
    def _sample_range_value(rng, value_range):
        arr = np.asarray(value_range, dtype=float).reshape(-1)
        if arr.size == 0:
            return 1.0
        if arr.size == 1:
            return float(arr[0])
        lo = float(np.min(arr[:2]))
        hi = float(np.max(arr[:2]))
        return float(rng.uniform(lo, hi))

    @staticmethod
    def _split_polyline_by_fraction(path, fraction: float):
        pts = np.asarray(path, dtype=float)
        if len(pts) < 3:
            return pts.copy(), pts.copy()
        frac = float(np.clip(fraction, 0.1, 0.9))
        edges = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(edges)])
        total = float(cum[-1])
        if total <= 1e-9:
            split_idx = max(1, min(len(pts) - 2, int(round(frac * (len(pts) - 1)))))
            return pts[: split_idx + 1].copy(), pts[split_idx:].copy()
        target = frac * total
        split_idx = int(np.searchsorted(cum, target, side="right") - 1)
        split_idx = max(0, min(split_idx, len(pts) - 2))
        edge_len = float(edges[split_idx])
        if edge_len <= 1e-9:
            alpha = 0.0
        else:
            alpha = float(np.clip((target - cum[split_idx]) / edge_len, 0.0, 1.0))
        split_pt = (1.0 - alpha) * pts[split_idx] + alpha * pts[split_idx + 1]
        first = np.vstack([pts[: split_idx + 1], split_pt[None, :]])
        second = np.vstack([split_pt[None, :], pts[split_idx + 1 :]])
        return np.asarray(first, dtype=float), np.asarray(second, dtype=float)

    def _build_stage1(self, p_start, p_contact, n_points, v_max, a_max):
        mid = 0.35 * np.asarray(p_start, dtype=float) + 0.65 * np.asarray(p_contact, dtype=float)
        ctrl = np.vstack([p_start, mid, p_contact])
        return self._resample_with_speed(
            ctrl,
            v_max=v_max,
            a_max=a_max,
            target_speed=self.stage1_target_speed_ratio * float(v_max),
            nominal_count=n_points,
            speed_profile_weights=lambda n: self._stage1_speed_profile_weights(n, v_max=v_max),
        )

    def _stage1_speed_profile_weights(self, num_edges: int, *, v_max=None) -> np.ndarray:
        n = int(max(num_edges, 1))
        if n == 1:
            return np.ones(1, dtype=float)
        taper_fraction = float(np.clip(self.stage1_speed_taper_fraction, 0.0, 1.0))
        if taper_fraction <= 1e-8:
            return np.ones(n, dtype=float)
        if self.stage1_speed_taper_end_ratio is None:
            stage1_v = self.stage1_target_speed_ratio * float(self.stage1_speed_max if v_max is None else v_max)
            stage2_v = self.stage2_target_speed_ratio * float(self.stage2_speed_max)
            end_ratio = stage2_v / max(stage1_v, 1e-8)
            end_ratio = float(np.clip(end_ratio, 0.72, 0.92))
        else:
            end_ratio = float(np.clip(self.stage1_speed_taper_end_ratio, 0.45, 1.0))
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        start = 1.0 - taper_fraction
        alpha = np.clip((u - start) / max(taper_fraction, 1e-8), 0.0, 1.0)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        weights = 1.0 - (1.0 - end_ratio) * smooth
        return np.clip(weights, 0.45, None)

    def _build_stage3(self, n_start, n_end, n_points, rng):
        n_points = int(max(n_points, 8))
        n_start = self._unit(n_start)
        n_end = self._unit(n_end)
        _, t1, t2 = self._orthonormal_frame(n_start, rng)
        bend_sign = -1.0 if float(np.dot(t2, n_end)) < 0.0 else 1.0
        detour_dir = self._unit(0.45 * t1 + bend_sign * 0.55 * t2)
        if self.split_stage3_transition:
            detour_angle = float(rng.uniform(0.78, 1.00))
        else:
            detour_angle = float(rng.uniform(0.68, 0.92))
        n_mid = self._unit(np.cos(detour_angle) * n_start + np.sin(detour_angle) * detour_dir)

        split = max(3, n_points // 2)
        normals_a = self._slerp_unit(n_start, n_mid, split, endpoint=False)
        normals_b = self._slerp_unit(n_mid, n_end, n_points - split + 1, endpoint=True)
        normals = np.vstack([normals_a, normals_b[1:]])
        if len(normals) != n_points:
            normals = self._slerp_unit(n_start, n_end, n_points, endpoint=True)

        u = np.linspace(0.0, 1.0, len(normals))
        envelope = np.sin(np.pi * u) ** 1.15
        phase = float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi))
        if self.split_stage3_transition:
            base = 0.27 + 0.16 * np.sin(1.8 * np.pi * u + phase)
            ripple = 0.06 * np.sin(4.0 * np.pi * u - 0.4 * phase)
            radial_frac = np.clip((base + ripple) * envelope + 0.145, 0.08, 0.76)
        else:
            base = 0.24 + 0.15 * np.sin(1.8 * np.pi * u + phase)
            ripple = 0.05 * np.sin(4.0 * np.pi * u - 0.4 * phase)
            radial_frac = np.clip((base + ripple) * envelope + 0.13, 0.07, 0.72)
        radius = self.sphere_radius + self.shell_thickness * radial_frac
        return self.sphere_center[None, :] + radius[:, None] * normals

    def _build_stage4(self, p_start, n_start, rng):
        normal, t1, t2 = self._orthonormal_frame(n_start, rng)
        depart_radius = self.sphere_radius + self.depart_offset * rng.uniform(0.85, 1.15)
        lateral = 0.10 * self.sphere_radius * rng.uniform(-1.0, 1.0)
        vertical = 0.08 * self.sphere_radius * rng.uniform(-1.0, 1.0)
        p_end = (
            self.sphere_center
            + depart_radius * normal
            + lateral * t1
            + vertical * t2
        )
        ctrl = np.vstack([p_start, 0.55 * p_start + 0.45 * p_end, p_end])
        return ctrl

    def _interpolate_unit_axes(self, axis_start, axis_end, num_points):
        return self._slerp_unit(axis_start, axis_end, num_points, endpoint=True)

    def _make_irregular_axis_transition(self, axis_start, axis_end, num_points, rng, max_tilt):
        base = self._interpolate_unit_axes(axis_start, axis_end, num_points)
        n = len(base)
        if n == 0:
            return base
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        env = np.sin(np.pi * u) ** 0.78
        tilt = (
            0.30
            + 0.46 * np.sin(3.4 * np.pi * u - 0.15 * np.pi)
            + 0.24 * np.sin(7.2 * np.pi * u + 0.33 * np.pi)
            + 0.16 * np.sin(11.4 * np.pi * u - 0.52 * np.pi)
        )
        tilt = env * np.abs(tilt)
        tilt = self._smooth_trace(tilt, kernel_size=3)
        tilt = np.clip(tilt * float(max_tilt), 0.0, float(max_tilt))

        tangent_phase = (
            float(rng.uniform(-np.pi, np.pi))
            + 1.10 * np.sin(4.2 * np.pi * u)
            + 0.75 * np.sin(8.6 * np.pi * u - 0.40)
            + 0.35 * np.sin(14.0 * np.pi * u + 0.25)
        )
        tangent_phase = tangent_phase + self._smooth_noise(rng, n, scale=0.22, kernel_size=3)

        out = np.empty_like(base)
        for i, axis in enumerate(base):
            axis = self._unit(axis)
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(axis, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(axis, ref))
            t2 = self._unit(np.cross(axis, t1))
            tangent = np.cos(tangent_phase[i]) * t1 + np.sin(tangent_phase[i]) * t2
            out[i] = self._unit(np.cos(tilt[i]) * axis + np.sin(tilt[i]) * tangent)
        out[0] = self._unit(axis_start)
        out[-1] = self._unit(axis_end)
        return out

    def _make_aligned_axis_trace(self, normals, rng, max_error):
        normals = np.asarray(normals, dtype=float)
        n = len(normals)
        out = np.empty_like(normals)
        if n == 0:
            return out
        max_error = float(max(max_error, 1e-4))
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        phase0 = float(rng.uniform(-np.pi, np.pi))
        angle_margin = self._make_stage_margin_profile(
            n,
            offset=0.03 * max_error,
            amplitude=0.52 * max_error,
            cycles=4.6,
            phase=0.0,
            noise_scale=0.0,
            rng=None,
            kernel_size=1,
        )
        angle = 1.00 * max_error - angle_margin
        angle = np.clip(angle, 0.48 * max_error, 0.99 * max_error)

        tangent_phase = (
            phase0
            + 1.15 * np.sin(2.9 * np.pi * u + 0.18 * phase0)
            + 0.55 * np.sin(5.8 * np.pi * u - 0.33 * phase0)
        )
        tangent_phase = self._smooth_trace(tangent_phase, kernel_size=7)

        for i, normal in enumerate(normals):
            normal = self._unit(normal)
            ref = np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(np.dot(normal, ref))) > 0.9:
                ref = np.array([0.0, 1.0, 0.0], dtype=float)
            t1 = self._unit(np.cross(normal, ref))
            t2 = self._unit(np.cross(normal, t1))
            tangent = np.cos(tangent_phase[i]) * t1 + np.sin(tangent_phase[i]) * t2
            out[i] = self._unit(np.cos(angle[i]) * normal + np.sin(angle[i]) * tangent)
        return out

    def _sample_axis_near_normal(self, normal, rng, max_error, min_fraction=0.35):
        normal = self._unit(normal)
        max_error = float(max(max_error, 0.0))
        if max_error <= 1e-8:
            return normal
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(normal, ref))) > 0.9:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        t1 = self._unit(np.cross(normal, ref))
        t2 = self._unit(np.cross(normal, t1))
        phase = float(rng.uniform(-np.pi, np.pi))
        tangent = self._unit(np.cos(phase) * t1 + np.sin(phase) * t2)
        lo = float(np.clip(min_fraction, 0.0, 1.0)) * max_error
        angle = float(rng.uniform(lo, max_error))
        return self._unit(np.cos(angle) * normal + np.sin(angle) * tangent)

    def _generate_tool_axis_trace(self, traj, stage_lengths, normals_stage2, normals_stage3, n_contact, rng):
        lengths = [int(x) for x in stage_lengths]
        total = int(sum(stage_lengths))
        axis_start = self._unit(0.76 * self._unit(n_contact) + 0.24 * self._unit(rng.randn(3)))
        l1 = lengths[0]
        stage1 = self._make_irregular_axis_transition(
            axis_start,
            n_contact,
            l1,
            rng=rng,
            max_tilt=0.38 * float(self.tool_align_max_stage2),
        )
        stage2 = self._make_aligned_axis_trace(normals_stage2, rng, max_error=self.tool_align_max_stage2)

        if len(lengths) == 5:
            l1, l2, l3, l4, l5 = lengths
            mid_anchor = self._unit(0.72 * self._unit(normals_stage3[0]) + 0.28 * self._unit(rng.randn(3)))
            stage3 = self._make_irregular_axis_transition(
                stage2[-1],
                mid_anchor,
                l3,
                rng=rng,
                max_tilt=0.46 * float(self.tool_align_max_stage2),
            )
            free_axis = self._sample_axis_near_normal(
                normals_stage3[-1],
                rng,
                max_error=self.stage4_tool_normal_max_error,
                min_fraction=0.45,
            )
            stage4 = self._make_irregular_axis_transition(
                stage3[-1],
                free_axis,
                l4,
                rng=rng,
                max_tilt=0.92 * float(self.tool_align_max_stage2),
            )
            s5_start = int(l1 + l2 + l3 + l4)
            stage5_rel = np.asarray(traj[s5_start : s5_start + l5], dtype=float) - self.sphere_center[None, :]
            if len(stage5_rel) > 0:
                stage5_normals = stage5_rel / np.maximum(np.linalg.norm(stage5_rel, axis=1, keepdims=True), 1e-12)
                stage5_target_normal = stage5_normals[-1]
            else:
                stage5_target_normal = normals_stage3[-1]
            stage5_end = self._sample_axis_near_normal(
                stage5_target_normal,
                rng,
                max_error=self.stage5_tool_normal_max_error,
                min_fraction=0.25,
            )
            stage5 = self._make_irregular_axis_transition(
                stage4[-1],
                stage5_end,
                l5,
                rng=rng,
                max_tilt=0.36 * float(self.tool_align_max_stage2),
            )
            axis = np.vstack([stage1, stage2, stage3, stage4, stage5])
        else:
            _, _, l3, l4 = lengths
            free_axis = self._unit(0.6 * normals_stage3[-1] + 0.4 * self._unit(rng.randn(3)))
            stage3 = self._make_irregular_axis_transition(
                stage2[-1],
                free_axis,
                l3,
                rng=rng,
                max_tilt=0.92 * float(self.tool_align_max_stage2),
            )
            stage4_end = self._unit(0.78 * self._unit(stage3[-1]) + 0.22 * self._unit(rng.randn(3)))
            stage4 = self._make_irregular_axis_transition(
                stage3[-1],
                stage4_end,
                l4,
                rng=rng,
                max_tilt=0.36 * float(self.tool_align_max_stage2),
            )
            axis = np.vstack([stage1, stage2, stage3, stage4])
        if axis.shape[0] != total:
            axis = np.asarray(axis[:total], dtype=float)
        return axis

    def generate_demo(self, rng=None, **kwargs):
        rng = np.random if rng is None else rng
        demo_index = kwargs.get("demo_index", None)
        l1, l2, l3, l4 = self._sample_segment_lengths(rng)

        phi0 = float(rng.uniform(self.contact_phi_range[0], self.contact_phi_range[1]))
        trace_angle = float(rng.uniform(self.stage2_trace_angle_range[0], self.stage2_trace_angle_range[1]))
        stage2_latitude_path = None
        if bool(self.stage2_robot_lateral_trace):
            lateral_sign = -1.0 if float(rng.rand()) < 0.5 else 1.0
            sin_phi = max(float(np.sin(phi0)), 1e-6)
            delta_theta = min(float(trace_angle) / sin_phi, 0.92 * np.pi)
            theta_center = float(self.stage2_lateral_center_theta)
            bump_lo, bump_hi = self.stage2_lateral_phi_bump_range
            phi_bump = float(rng.uniform(bump_lo, bump_hi))
            theta0 = theta_center - 0.5 * lateral_sign * delta_theta
            theta1 = theta_center + 0.5 * lateral_sign * delta_theta
            n_contact = self._normal_from_spherical(theta0, phi0)
            n_trace_end = self._normal_from_spherical(theta1, phi0)
            normal0 = n_contact
            t1 = self._unit(
                [
                    -lateral_sign * np.sin(theta0) * np.sin(phi0),
                    lateral_sign * np.cos(theta0) * np.sin(phi0),
                    0.0,
                ]
            )
            t2 = self._unit(np.cross(normal0, t1))
            stage2_latitude_path = (theta0, theta1, phi0, phi_bump)
        else:
            theta0 = float(rng.uniform(self.contact_theta_range[0], self.contact_theta_range[1]))
            n_contact = self._normal_from_spherical(theta0, phi0)
            normal0, t1, t2 = self._orthonormal_frame(n_contact, rng)
            n_trace_end = self._unit(np.cos(trace_angle) * normal0 + np.sin(trace_angle) * t1)

        theta_trace_end, phi_trace_end = self._spherical_from_normal(n_trace_end)
        top_phi_lo, top_phi_hi = self.stage345_top_phi_range
        phi_cap = float(rng.uniform(min(top_phi_lo, top_phi_hi), max(top_phi_lo, top_phi_hi)))
        theta_pull = float(np.clip(self.stage345_top_theta_pull, 0.0, 1.0))
        theta_repos = (
            (1.0 - theta_pull) * theta_trace_end
            + theta_pull * float(self.stage2_lateral_center_theta)
            + float(rng.uniform(-self.stage345_top_theta_jitter, self.stage345_top_theta_jitter))
        )
        n_repos_end = self._normal_from_spherical(theta_repos, phi_cap)

        p_contact = self.sphere_center + self.sphere_radius * n_contact
        p_precontact = self.sphere_center + (self.sphere_radius + 0.18 * self.sphere_radius) * n_contact
        p_start = (
            self.sphere_center
            + (self.sphere_radius + self.approach_offset * rng.uniform(0.85, 1.15)) * n_contact
            + 0.18 * self.sphere_radius * rng.uniform(-1.0, 1.0) * t1
            + 0.12 * self.sphere_radius * rng.uniform(-1.0, 1.0) * t2
        )

        stage1 = self._build_stage1(p_start, p_contact, l1, v_max=self.stage1_speed_max, a_max=self.stage1_accel_max)
        stage2_length_scale = max(self._sample_range_value(rng, self.stage2_length_scale_range), 1e-3)
        stage4_length_scale = max(self._sample_range_value(rng, self.stage4_length_scale_range), 1e-3)
        if stage2_latitude_path is None:
            stage2_raw = self._make_surface_path(n_contact, n_trace_end, max(int(4 * l2), 96))
        else:
            stage2_raw = self._make_latitude_surface_path(
                stage2_latitude_path[0],
                stage2_latitude_path[1],
                stage2_latitude_path[2],
                max(int(4 * l2), 96),
                phi_bump=stage2_latitude_path[3],
            )
        stage2 = self._resample_with_speed(
            stage2_raw,
            v_max=self.stage2_speed_max / stage2_length_scale,
            a_max=self.stage2_accel_max / stage2_length_scale,
            target_speed=self.stage2_target_speed_ratio * self.stage2_speed_max,
            nominal_count=l2,
            use_optimizer=False,
            speed_profile_weights=lambda n: self._make_cruise_valley_weights(
                n,
                depth=self.stage2_speed_valley_depths,
                center=self.stage2_speed_valley_centers,
                width=self.stage2_speed_valley_widths,
            ),
        )
        if self.split_stage3_transition:
            shell_offset = float(self.true_constraints["surface_near_target"])
            shell_blend = float(rng.uniform(self.stage3_shell_blend_range[0], self.stage3_shell_blend_range[1]))
            n_shell_start = self._unit((1.0 - shell_blend) * n_trace_end + shell_blend * n_repos_end)
            stage3_raw = self._build_stage3_transition(
                n_trace_end,
                n_shell_start,
                max(int(5 * l3), 96),
                shell_offset=shell_offset,
                rng=rng,
            )
            stage3 = self._resample_with_speed(
                stage3_raw,
                v_max=self.stage3_speed_max,
                a_max=self.stage3_accel_max,
                target_speed=self.stage3_target_speed_ratio * self.stage3_speed_max,
                nominal_count=l3,
                use_optimizer=False,
            )
            stage3_speed_weights = self._sample_stage3_speed_profile_weights(max(len(stage3) - 1, 1), rng)
            stage3 = self._regularize_stage3_transition_path(
                stage3,
                shell_offset=shell_offset,
                radial_blend=0.8,
                speed_profile_weights=lambda n, w=stage3_speed_weights: w,
            )
            stage4_raw = self._make_spherical_shell_path(
                n_shell_start,
                n_repos_end,
                max(int(5 * l4), 96),
                radius_offset=shell_offset,
                detour_angle=float(max(self.stage4_shell_detour_angle, 0.0)),
            )
            stage4 = self._resample_with_speed(
                stage4_raw,
                v_max=self.stage2_speed_max / stage4_length_scale,
                a_max=self.stage3_accel_max / stage4_length_scale,
                target_speed=self.stage4_target_speed_ratio * self.stage2_speed_max,
                nominal_count=None,
                use_optimizer=False,
                speed_profile_weights=self._stage4_speed_profile_weights,
            )
            stage4 = self._regularize_tail_spacing(stage4, tail_points=5)
            stage5_ctrl = self._build_stage4(stage4[-1], n_repos_end, rng=rng)
            stage4 = self._repair_stage4_departure_tail(
                stage4,
                stage5_ctrl[-1],
                shell_offset=shell_offset,
                tail_points=min(5, max(2, len(stage4) // 2)),
            )
            stage4 = self._resample_with_speed(
                stage4,
                v_max=self.stage2_speed_max / stage4_length_scale,
                a_max=self.stage3_accel_max / stage4_length_scale,
                target_speed=self.stage4_target_speed_ratio * self.stage2_speed_max,
                nominal_count=None,
                use_optimizer=False,
                speed_profile_weights=self._stage4_speed_profile_weights,
            )
            stage4 = self._project_to_shell(stage4, shell_offset=shell_offset)
            stage5 = self._resample_with_speed(
                stage5_ctrl,
                v_max=self.stage4_speed_max,
                a_max=self.stage4_accel_max,
                target_speed=self.stage5_target_speed_ratio * self.stage4_speed_max,
                nominal_count=l4,
            )
            traj = np.vstack([stage1, stage2[1:], stage3[1:], stage4[1:], stage5[1:]])
        else:
            stage3_raw_full = self._build_stage3(n_trace_end, n_repos_end, l3, rng=rng)
            stage3 = self._resample_with_speed(
                stage3_raw_full,
                v_max=self.stage2_speed_max,
                a_max=self.stage3_accel_max,
                target_speed=self.stage3_target_speed_ratio * self.stage2_speed_max,
                nominal_count=l3,
            )
            stage4_ctrl = self._build_stage4(stage3[-1], n_repos_end, rng=rng)
            stage4 = self._resample_with_speed(
                stage4_ctrl,
                v_max=self.stage4_speed_max,
                a_max=self.stage4_accel_max,
                target_speed=self.stage5_target_speed_ratio * self.stage4_speed_max,
                nominal_count=l4,
            )
            traj = np.vstack([stage1, stage2[1:], stage3[1:], stage4[1:]])
        if self.noise_std > 0.0:
            noise = np.stack(
                [
                    self._smooth_noise(rng, len(traj), scale=self.noise_std, kernel_size=self.trajectory_noise_kernel)
                    for _ in range(traj.shape[1])
                ],
                axis=1,
            )
            stage_noise_scale = np.ones(len(traj), dtype=float)
            if self.split_stage3_transition:
                stage2_slice = slice(len(stage1), len(stage1) + len(stage2) - 1)
                stage4_slice = slice(
                    len(stage1) + len(stage2) + len(stage3) - 2,
                    len(stage1) + len(stage2) + len(stage3) + len(stage4) - 3,
                )
                stage_noise_scale[stage2_slice] *= float(self.stage2_noise_scale)
                stage_noise_scale[stage4_slice] *= float(self.stage4_noise_scale)
            else:
                stage2_slice = slice(len(stage1), len(stage1) + len(stage2) - 1)
                stage_noise_scale[stage2_slice] *= float(self.stage2_noise_scale)
            traj = traj + noise * stage_noise_scale[:, None]
            radii = np.linalg.norm(traj - self.sphere_center[None, :], axis=1)
            contact_mask = slice(len(stage1), len(stage1) + len(stage2) - 1)
            safe = np.maximum(radii[contact_mask], 1e-12)
            traj[contact_mask] = self.sphere_center[None, :] + (
                self.sphere_radius * (traj[contact_mask] - self.sphere_center[None, :]) / safe[:, None]
            )
            if self.split_stage3_transition:
                shell_offset = float(self.true_constraints["surface_near_target"])
                stage3_slice_after_noise = slice(
                    len(stage1) + len(stage2) - 2,
                    len(stage1) + len(stage2) + len(stage3) - 2,
                )
                if (stage3_slice_after_noise.stop - stage3_slice_after_noise.start) >= 4:
                    traj[stage3_slice_after_noise] = self._regularize_stage3_transition_path(
                        traj[stage3_slice_after_noise],
                        shell_offset=shell_offset,
                        radial_blend=0.9,
                        speed_profile_weights=lambda n, w=stage3_speed_weights: w,
                    )
                stage4_slice_after_noise = slice(
                    len(stage1) + len(stage2) + len(stage3) - 2,
                    len(stage1) + len(stage2) + len(stage3) + len(stage4) - 3,
                )
                if (stage4_slice_after_noise.stop - stage4_slice_after_noise.start) >= 5:
                    stage4_tail_fixed = self._regularize_tail_spacing(
                        traj[stage4_slice_after_noise],
                        tail_points=5,
                    )
                    stage4_tail_fixed = self._project_to_shell(stage4_tail_fixed, shell_offset=shell_offset)
                    departure_target = traj[min(stage4_slice_after_noise.stop, len(traj) - 1)]
                    stage4_tail_fixed = self._repair_stage4_departure_tail(
                        stage4_tail_fixed,
                        departure_target,
                        shell_offset=shell_offset,
                        tail_points=min(5, max(2, len(stage4_tail_fixed) // 2)),
                    )
                    stage4_tail_fixed = self._resample_fixed_count_with_speed_profile(
                        stage4_tail_fixed,
                        len(stage4_tail_fixed),
                        speed_profile_weights=self._stage4_speed_profile_weights,
                    )
                    stage4_tail_fixed = self._project_to_shell(stage4_tail_fixed, shell_offset=shell_offset)
                    traj[stage4_slice_after_noise] = stage4_tail_fixed

        if self.split_stage3_transition:
            true_cutpoints = np.asarray(
                [
                    int(len(stage1) - 1),
                    int(len(stage1) + len(stage2) - 2),
                    int(len(stage1) + len(stage2) + len(stage3) - 3),
                    int(len(stage1) + len(stage2) + len(stage3) + len(stage4) - 4),
                ],
                dtype=int,
            )
            stage_lengths = (len(stage1), len(stage2) - 1, len(stage3) - 1, len(stage4) - 1, len(stage5) - 1)
            stage2_slice = slice(len(stage1), len(stage1) + len(stage2) - 1)
            stage4_slice = slice(
                len(stage1) + len(stage2) + len(stage3) - 2,
                len(stage1) + len(stage2) + len(stage3) + len(stage4) - 3,
            )
            normals_stage2 = traj[stage2_slice] - self.sphere_center[None, :]
            normals_stage2 = normals_stage2 / np.maximum(np.linalg.norm(normals_stage2, axis=1, keepdims=True), 1e-12)
            normals_stage4 = traj[stage4_slice] - self.sphere_center[None, :]
            normals_stage4 = normals_stage4 / np.maximum(np.linalg.norm(normals_stage4, axis=1, keepdims=True), 1e-12)
            tool_axis = self._generate_tool_axis_trace(
                traj=traj,
                stage_lengths=stage_lengths,
                normals_stage2=normals_stage2,
                normals_stage3=normals_stage4,
                n_contact=n_contact,
                rng=rng,
            )
        else:
            true_cutpoints = np.asarray(
                [
                    int(len(stage1) - 1),
                    int(len(stage1) + len(stage2) - 2),
                    int(len(stage1) + len(stage2) + len(stage3) - 3),
                ],
                dtype=int,
            )
            stage_lengths = (len(stage1), len(stage2) - 1, len(stage3) - 1, len(stage4) - 1)
            stage2_slice = slice(len(stage1), len(stage1) + len(stage2) - 1)
            stage3_slice = slice(len(stage1) + len(stage2) - 1, len(stage1) + len(stage2) + len(stage3) - 2)
            normals_stage2 = traj[stage2_slice] - self.sphere_center[None, :]
            normals_stage2 = normals_stage2 / np.maximum(np.linalg.norm(normals_stage2, axis=1, keepdims=True), 1e-12)
            normals_stage3 = traj[stage3_slice] - self.sphere_center[None, :]
            normals_stage3 = normals_stage3 / np.maximum(np.linalg.norm(normals_stage3, axis=1, keepdims=True), 1e-12)
            tool_axis = self._generate_tool_axis_trace(
                traj=traj,
                stage_lengths=stage_lengths,
                normals_stage2=normals_stage2,
                normals_stage3=normals_stage3,
                n_contact=n_contact,
                rng=rng,
            )
        if tool_axis.shape[0] != len(traj):
            axis_fixed = np.empty((len(traj), 3), dtype=float)
            axis_fixed[: min(len(tool_axis), len(traj))] = tool_axis[: min(len(tool_axis), len(traj))]
            if len(tool_axis) < len(traj):
                axis_fixed[len(tool_axis) :] = tool_axis[-1]
            tool_axis = axis_fixed
        self.register_tool_axis_trace(traj, tool_axis)
        return traj, true_cutpoints

    def generate_demos(self, n_demos=10, rng=None, **kwargs):
        rng = np.random if rng is None else rng
        demos = []
        true_cutpoints = []
        for _ in range(int(n_demos)):
            traj, cutpoints = self.generate_demo(rng=rng, **kwargs)
            demos.append(np.asarray(traj, dtype=float))
            true_cutpoints.append(np.asarray(cutpoints, dtype=int))
        return demos, true_cutpoints

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


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _s5_demo_cache_path(*, task_name: str, n_seed: int, env_cfg: dict, run_kwargs: dict, cache_dir=None) -> Path:
    root = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parent / "demo_cache"
    payload = {
        "task_name": str(task_name),
        "seed": int(n_seed),
        "env_cfg": _jsonable(env_cfg),
        "run_kwargs": _jsonable(run_kwargs),
        "cache_version": 17,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return root / str(task_name) / f"seed_{int(n_seed)}_{digest}.npz"


def _make_s5_bundle_from_arrays(
    *,
    task_name: str,
    seed: int,
    env: S5SphereInspectEnv,
    demos: list[np.ndarray],
    true_cutpoints: list[np.ndarray],
    scene_specs: list[dict],
    cache_path: Path | None,
    cache_hit: bool,
) -> TaskBundle:
    return TaskBundle(
        name=task_name,
        demos=demos,
        env=env,
        true_taus=[None for _ in demos],
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=dict(env.true_constraints),
        constraint_specs=env.get_constraint_specs(),
        meta={
            "seed": int(seed),
            "task_name": task_name,
            "scene_specs": scene_specs,
            "observation_specs": env.get_observation_spec(),
            "render_camera_presets": env.get_render_camera_presets(),
            "asset_handles": env.get_asset_handles(),
            "demo_cache": None if cache_path is None else {"path": str(cache_path), "hit": bool(cache_hit)},
        },
    )


def _try_load_s5_demo_cache(*, cache_path: Path, task_name: str, n_demos: int, seed: int, env: S5SphereInspectEnv):
    if not cache_path.exists():
        return None
    try:
        with np.load(cache_path, allow_pickle=True) as data:
            count = int(data["count"])
            if count < int(n_demos):
                return None
            demos = [np.asarray(data[f"demo_{i}"], dtype=float) for i in range(int(n_demos))]
            cutpoints = [np.asarray(data[f"cutpoints_{i}"], dtype=int) for i in range(int(n_demos))]
            tool_axes = []
            for i in range(int(n_demos)):
                key = f"tool_axis_{i}"
                tool_axes.append(None if key not in data else np.asarray(data[key], dtype=float))
            scene_specs = json.loads(str(data["scene_specs_json"].item()))[: int(n_demos)]
    except Exception:
        return None

    print(
        f"\033[31m[S5 demo cache] loaded {int(n_demos)}/{int(count)} demos from {cache_path}\033[0m",
        flush=True,
    )
    for traj, axis in zip(demos, tool_axes):
        if axis is not None:
            env.register_tool_axis_trace(traj, axis)
    return _make_s5_bundle_from_arrays(
        task_name=task_name,
        seed=seed,
        env=env,
        demos=demos,
        true_cutpoints=cutpoints,
        scene_specs=scene_specs,
        cache_path=cache_path,
        cache_hit=True,
    )


def _save_s5_demo_cache(*, cache_path: Path, bundle: TaskBundle, tool_axis_traces: list, env_cfg: dict, run_kwargs: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "count": np.asarray(len(bundle.demos), dtype=np.int64),
        "metadata_json": np.asarray(
            json.dumps(
                {
                    "task_name": bundle.name,
                    "seed": bundle.meta.get("seed"),
                    "env_cfg": _jsonable(env_cfg),
                    "run_kwargs": _jsonable(run_kwargs),
                    "cache_version": 17,
                },
                sort_keys=True,
            )
        ),
        "scene_specs_json": np.asarray(json.dumps(_jsonable(bundle.meta.get("scene_specs", [])), sort_keys=True)),
    }
    for i, demo in enumerate(bundle.demos):
        arrays[f"demo_{i}"] = np.asarray(demo, dtype=float)
        arrays[f"cutpoints_{i}"] = np.asarray(bundle.true_cutpoints[i], dtype=int)
        if i < len(tool_axis_traces) and tool_axis_traces[i] is not None:
            arrays[f"tool_axis_{i}"] = np.asarray(tool_axis_traces[i], dtype=float)
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    np.savez_compressed(tmp_path, **arrays)
    written = tmp_path if tmp_path.exists() else tmp_path.with_suffix(tmp_path.suffix + ".npz")
    os.replace(written, cache_path)


def _build_sphere_inspect_bundle(
    *,
    task_name: str,
    n_demos: int,
    seed: int,
    env_kwargs=None,
    demo_kwargs=None,
    **extra_env_kwargs,
) -> TaskBundle:
    env_cfg = dict(env_kwargs or {})
    env_cfg.update(extra_env_kwargs)
    cache_demos = bool(env_cfg.pop("cache_demos", False))
    cache_dir = env_cfg.pop("demo_cache_dir", None)
    run_kwargs = dict(demo_kwargs or {})
    env = S5SphereInspectEnv(**env_cfg)
    cache_path = None
    if cache_demos:
        cache_path = _s5_demo_cache_path(
            task_name=task_name,
            n_seed=int(seed),
            env_cfg=env_cfg,
            run_kwargs=run_kwargs,
            cache_dir=cache_dir,
        )
        cached_bundle = _try_load_s5_demo_cache(
            cache_path=cache_path,
            task_name=task_name,
            n_demos=int(n_demos),
            seed=int(seed),
            env=env,
        )
        if cached_bundle is not None:
            return cached_bundle

    demos = []
    true_cutpoints = []
    scene_specs = []
    tool_axis_traces = []
    for demo_idx in range(int(n_demos)):
        scene = env.sample_scene()
        scene["demo_index"] = int(demo_idx)
        latent = env.rollout_demo(scene, seed=env.demo_seed_for_index(seed, demo_idx), **run_kwargs)
        observation = env.compute_observation(latent, scene)
        traj = np.asarray(observation["trajectory"], dtype=float)
        tool_axis = observation.get("tool_axis")
        if tool_axis is not None:
            tool_axis = np.asarray(tool_axis, dtype=float)
            env.register_tool_axis_trace(traj, tool_axis)
        demos.append(traj)
        true_cutpoints.append(np.asarray(observation["true_cutpoints"], dtype=int))
        scene_specs.append(dict(scene))
        tool_axis_traces.append(None if tool_axis is None else np.asarray(tool_axis, dtype=float))
    true_taus = [None for _ in demos]
    bundle = TaskBundle(
        name=task_name,
        demos=demos,
        env=env,
        true_taus=true_taus,
        true_cutpoints=true_cutpoints,
        feature_schema=env.get_feature_schema(),
        true_constraints=dict(env.true_constraints),
        constraint_specs=env.get_constraint_specs(),
        meta={
            "seed": seed,
            "task_name": task_name,
            "scene_specs": scene_specs,
            "observation_specs": env.get_observation_spec(),
            "render_camera_presets": env.get_render_camera_presets(),
            "asset_handles": env.get_asset_handles(),
            "demo_cache": None if cache_path is None else {"path": str(cache_path), "hit": False},
        },
    )
    if cache_path is not None:
        _save_s5_demo_cache(
            cache_path=cache_path,
            bundle=bundle,
            tool_axis_traces=tool_axis_traces,
            env_cfg=env_cfg,
            run_kwargs=run_kwargs,
        )
    return bundle


def _apply_default_s5_loader_config(env_cfg: dict) -> dict:
    env_cfg = dict(env_cfg)
    env_cfg.setdefault("seg_lengths", (18, 34, 24, 18))
    env_cfg.setdefault("seg_length_jitter", (3, 5, 5, 3))
    env_cfg.setdefault("sphere_radius", 1.0 * _S5_METRIC_SCALE)
    env_cfg.setdefault("shell_thickness", 0.24 * _S5_METRIC_SCALE)
    env_cfg.setdefault("approach_offset", 0.42 * _S5_METRIC_SCALE)
    env_cfg.setdefault("depart_offset", 0.50 * _S5_METRIC_SCALE)
    env_cfg.setdefault("surface_near_target_ratio", 0.75)
    env_cfg.setdefault("split_stage3_transition", True)
    env_cfg.setdefault("transition_stage_fraction", 0.40)
    env_cfg.setdefault("contact_theta_range", (-0.12 * np.pi, 0.16 * np.pi))
    env_cfg.setdefault("contact_phi_range", (0.20 * np.pi, 0.34 * np.pi))
    env_cfg.setdefault("stage1_speed_max", 0.12 * _S5_METRIC_SCALE)
    env_cfg.setdefault("stage2_speed_max", 0.047 * _S5_METRIC_SCALE)
    env_cfg.setdefault("stage3_speed_max", 0.060 * _S5_METRIC_SCALE)
    env_cfg.setdefault("stage4_speed_max", 0.09 * _S5_METRIC_SCALE)
    env_cfg.setdefault("stage1_accel_max", 0.08 * _S5_METRIC_SCALE)
    env_cfg.setdefault("stage2_accel_max", 0.03 * _S5_METRIC_SCALE)
    env_cfg.setdefault("stage3_accel_max", 0.07 * _S5_METRIC_SCALE)
    env_cfg.setdefault("stage4_accel_max", 0.06 * _S5_METRIC_SCALE)
    env_cfg.setdefault("stage2_trace_angle_range", (1.184, 1.376))
    env_cfg.setdefault("stage2_robot_lateral_trace", True)
    env_cfg.setdefault("stage2_lateral_center_theta", 0.0)
    env_cfg.setdefault("stage2_lateral_phi_bump_range", (-0.035 * np.pi, 0.035 * np.pi))
    env_cfg.setdefault("repos_angle_range", (0.95, 1.18))
    env_cfg.setdefault("stage3_shell_blend_range", (0.44, 0.58))
    env_cfg.setdefault("stage345_top_phi_range", (0.10 * np.pi, 0.18 * np.pi))
    env_cfg.setdefault("stage345_top_theta_pull", 0.45)
    env_cfg.setdefault("stage345_top_theta_jitter", 0.10 * np.pi)
    env_cfg.setdefault("stage2_surface_detour_angle", 0.0)
    env_cfg.setdefault("stage4_shell_detour_angle", 0.10)
    env_cfg.setdefault("stage2_length_scale_range", (1.0, 1.0))
    env_cfg.setdefault("stage4_length_scale_range", (1.0, 1.0))
    env_cfg.setdefault("stage1_speed_taper_fraction", 1.0)
    env_cfg.setdefault("stage1_speed_taper_end_ratio", 0.78)
    env_cfg.setdefault("stage2_target_speed_ratio", 0.99)
    env_cfg.setdefault("stage3_target_speed_ratio", 0.75)
    env_cfg.setdefault("stage4_target_speed_ratio", 0.99)
    env_cfg.setdefault("stage2_speed_valley_depths", (0.07, 0.18, 0.07))
    env_cfg.setdefault("stage2_speed_valley_centers", (0.30, 0.58, 0.80))
    env_cfg.setdefault("stage2_speed_valley_widths", (0.018, 0.025, 0.018))
    env_cfg.setdefault("stage3_speed_jitter_std", 0.04)
    env_cfg.setdefault("stage3_speed_jitter_clip", 0.09)
    env_cfg.setdefault("stage3_speed_jitter_kernel", 5)
    env_cfg.setdefault("stage4_speed_valley_depth", 0.08)
    env_cfg.setdefault("stage4_speed_valley_center", 0.54)
    env_cfg.setdefault("stage4_speed_valley_width", 0.025)
    env_cfg.setdefault("noise_std", 0.004 * _S5_METRIC_SCALE)
    env_cfg.setdefault("stage2_noise_scale", 0.28)
    env_cfg.setdefault("stage4_noise_scale", 0.24)
    env_cfg.setdefault("trajectory_noise_kernel", 9)
    env_cfg.setdefault("pybullet_world_scale", 1.0)
    env_cfg.setdefault("pybullet_filter_max_position_error", 0.012 * _S5_METRIC_SCALE)
    return env_cfg


def load_S5SphereInspect(
    n_demos: int = 10,
    seed: int = 0,
    env_kwargs=None,
    demo_kwargs=None,
    **extra_env_kwargs,
) -> TaskBundle:
    env_cfg = dict(env_kwargs or {})
    env_cfg.update(extra_env_kwargs)
    env_cfg = _apply_default_s5_loader_config(env_cfg)
    env_cfg.setdefault("observation_backend", "analytic_raw")
    env_cfg.setdefault("eval_tag", "S5SphereInspect")
    return _build_sphere_inspect_bundle(
        task_name="S5SphereInspect",
        n_demos=n_demos,
        seed=seed,
        env_kwargs=env_cfg,
        demo_kwargs=demo_kwargs,
    )
