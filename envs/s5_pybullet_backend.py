from __future__ import annotations

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
