from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .rendering import _FFmpegVideoWriter, _space_was_triggered
from .s5_pybullet_backend import _UR5PoseTracker, _quat_from_matrix, _require_pybullet

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import pybullet as p
except ModuleNotFoundError:
    p = None


def _normalize(vec: np.ndarray, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.asarray(fallback, dtype=float).reshape(3)
    return arr / norm


def _surface_normal(env) -> np.ndarray:
    normal = np.asarray(
        [
            -float(getattr(env, "surface_tilt_x", 0.0)),
            -float(getattr(env, "surface_tilt_y", 0.0)),
            1.0,
        ],
        dtype=float,
    )
    return _normalize(normal, fallback=(0.0, 0.0, 1.0))


def _surface_frame_axes(env, theta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_axis = _surface_normal(env)
    heading_xy = np.asarray([math.cos(float(theta)), math.sin(float(theta)), 0.0], dtype=float)
    y_axis = heading_xy - float(np.dot(heading_xy, x_axis)) * x_axis
    y_axis = _normalize(y_axis, fallback=(1.0, 0.0, 0.0))
    z_axis = _normalize(np.cross(x_axis, y_axis), fallback=(0.0, 1.0, 0.0))
    y_axis = _normalize(np.cross(z_axis, x_axis), fallback=(1.0, 0.0, 0.0))
    return x_axis, y_axis, z_axis


def _quat_for_slider_theta(env, theta: float) -> np.ndarray:
    x_axis, y_axis, z_axis = _surface_frame_axes(env, theta)
    return _quat_from_matrix(np.stack([x_axis, y_axis, z_axis], axis=1))


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


def _spawn_box(client_id: int, *, half_extents, rgba, position, orientation=None, collision: bool = True, specular=(0.25, 0.25, 0.25)) -> int:
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half_extents), physicsClientId=client_id) if collision else -1
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=list(half_extents),
        rgbaColor=list(rgba),
        specularColor=list(specular),
        physicsClientId=client_id,
    )
    quat = [0.0, 0.0, 0.0, 1.0] if orientation is None else list(orientation)
    return int(
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(position),
            baseOrientation=quat,
            physicsClientId=client_id,
        )
    )


def _spawn_cylinder(client_id: int, *, radius: float, height: float, rgba, position, collision: bool = False, specular=(0.5, 0.5, 0.5)) -> int:
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(radius), height=float(height), physicsClientId=client_id) if collision else -1
    vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(radius),
        length=float(height),
        rgbaColor=list(rgba),
        specularColor=list(specular),
        physicsClientId=client_id,
    )
    return int(
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(position),
            physicsClientId=client_id,
        )
    )


def _yaw_quat(yaw: float):
    return p.getQuaternionFromEuler([0.0, 0.0, float(yaw)])


def _surface_height(env, xy) -> np.ndarray:
    if hasattr(env, "surface_height"):
        return np.asarray(env.surface_height(np.asarray(xy, dtype=float)), dtype=float)
    pts = np.asarray(xy, dtype=float)
    return np.zeros(pts.shape[:-1], dtype=float)


def _surface_quat(env, yaw: float = 0.0):
    sx = float(getattr(env, "surface_tilt_x", 0.0))
    sy = float(getattr(env, "surface_tilt_y", 0.0))
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    x_axis = _normalize([c, s, sx * c + sy * s])
    y_axis = _normalize([-s, c, -sx * s + sy * c])
    z_axis = _normalize(np.cross(x_axis, y_axis), fallback=(0.0, 0.0, 1.0))
    y_axis = _normalize(np.cross(z_axis, x_axis), fallback=(-s, c, 0.0))
    return _quat_from_matrix(np.stack([x_axis, y_axis, z_axis], axis=1))


def _quat_align_local_x_to_vec(vec: np.ndarray, up_hint=(0.0, 0.0, 1.0)):
    x_axis = _normalize(vec)
    up = _normalize(up_hint, fallback=(0.0, 0.0, 1.0))
    if abs(float(np.dot(x_axis, up))) > 0.96:
        up = np.asarray([0.0, 1.0, 0.0], dtype=float)
    y_axis = _normalize(np.cross(up, x_axis), fallback=(0.0, 1.0, 0.0))
    z_axis = _normalize(np.cross(x_axis, y_axis), fallback=(0.0, 0.0, 1.0))
    return _quat_from_matrix(np.stack([x_axis, y_axis, z_axis], axis=1))


def _compose_quat(a, b):
    return p.multiplyTransforms([0.0, 0.0, 0.0], list(a), [0.0, 0.0, 0.0], list(b))[1]


def _quat_between_vectors(src, dst):
    a = _normalize(np.asarray(src, dtype=float), fallback=(0.0, 0.0, 1.0))
    b = _normalize(np.asarray(dst, dtype=float), fallback=(0.0, 0.0, 1.0))
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return [0.0, 0.0, 0.0, 1.0]
    if dot < -1.0 + 1e-10:
        axis = _normalize(np.cross(a, [1.0, 0.0, 0.0]), fallback=(0.0, 1.0, 0.0))
        if float(np.linalg.norm(axis)) <= 1e-8:
            axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
        return p.getQuaternionFromAxisAngle(axis.tolist(), math.pi)
    axis = _normalize(np.cross(a, b), fallback=(0.0, 1.0, 0.0))
    angle = math.acos(dot)
    return p.getQuaternionFromAxisAngle(axis.tolist(), angle)


def _close_gripper_for_visual(tracker: _UR5PoseTracker) -> None:
    gripper_joint_ids = []
    target_positions = []
    for joint_idx in range(p.getNumJoints(tracker.robot_id, physicsClientId=tracker.client_id)):
        info = p.getJointInfo(tracker.robot_id, joint_idx, physicsClientId=tracker.client_id)
        if int(info[2]) == p.JOINT_FIXED:
            continue
        name = info[1].decode("utf-8", errors="ignore")
        if "gripper" not in name and "finger" not in name:
            continue
        lo = float(info[8])
        hi = float(info[9])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = -1.0, 1.0
        if name.endswith("_joint_1"):
            target = lo + 0.92 * (hi - lo)
        elif name.endswith("_joint_2"):
            target = lo + 0.82 * (hi - lo)
        elif name.endswith("_joint_3"):
            target = lo + 0.78 * (hi - lo)
        elif "palm_finger" in name:
            target = 0.0
        else:
            target = lo + 0.50 * (hi - lo)
        target = float(np.clip(target, lo, hi))
        p.resetJointState(
            tracker.robot_id,
            int(joint_idx),
            targetValue=target,
            targetVelocity=0.0,
            physicsClientId=tracker.client_id,
        )
        gripper_joint_ids.append(int(joint_idx))
        target_positions.append(target)
    if gripper_joint_ids:
        p.setJointMotorControlArray(
            tracker.robot_id,
            gripper_joint_ids,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            forces=[60.0] * len(gripper_joint_ids),
            physicsClientId=tracker.client_id,
        )


def _style_gripper_for_s4(tracker: _UR5PoseTracker) -> None:
    for joint_idx in range(p.getNumJoints(tracker.robot_id, physicsClientId=tracker.client_id)):
        info = p.getJointInfo(tracker.robot_id, joint_idx, physicsClientId=tracker.client_id)
        link_name = info[12].decode("utf-8", errors="ignore").lower()
        if (
            "task_tool" in link_name
            or "task_tcp" in link_name
            or "hinge" in link_name
            or "bar" in link_name
            or "actuating" in link_name
            or "underactuated" in link_name
        ):
            p.changeVisualShape(
                tracker.robot_id,
                int(joint_idx),
                rgbaColor=[1.0, 1.0, 1.0, 0.0],
                physicsClientId=tracker.client_id,
            )
        elif "gripper" in link_name or "finger" in link_name or "palm" in link_name:
            p.changeVisualShape(
                tracker.robot_id,
                int(joint_idx),
                rgbaColor=[0.50, 0.52, 0.54, 1.0],
                physicsClientId=tracker.client_id,
            )


def _spawn_s4_scene(env, tracker: _UR5PoseTracker) -> dict[str, object]:
    client_id = tracker.client_id
    origin = np.asarray(getattr(env, 'pybullet_world_center', (0.55, 0.0, 0.52)), dtype=float)
    table_half = np.asarray(getattr(env, 'pybullet_table_half_extents', (0.45, 0.20, 0.015)), dtype=float)
    table_id = _spawn_box(
        client_id,
        half_extents=table_half,
        rgba=(0.64, 0.66, 0.67, 1.0),
        specular=(0.28, 0.28, 0.28),
        position=tracker.s5_to_world(np.asarray([0.0, 0.0, float(_surface_height(env, [[0.0, 0.0]])[0]) - table_half[2]])),
        orientation=_surface_quat(env, 0.0),
        collision=True,
    )
    slot_x = float(getattr(env, 'slot_x', 0.16))
    slot_half_width = float(getattr(env, 'slot_half_width', 0.028))
    wall_len = float(getattr(env, 'slot_wall_length', 0.21))
    forward_ext = float(getattr(env, 'slot_wall_forward_extension', 0.0))
    wall_thick = float(getattr(env, 'slot_wall_thickness', 0.010))
    wall_height = float(getattr(env, 'slot_wall_height', 0.030))
    rail_shape = str(getattr(env, "rail_shape", "straight") or "straight").strip().lower()
    rail_polyline = getattr(env, "rail_polyline", None)
    if rail_shape == "straight" and rail_polyline is None:
        center_y = float(getattr(env, "clearance_target", 0.0))
        rail_poly = np.asarray([[slot_x - wall_len, center_y], [slot_x, center_y]], dtype=float)
    else:
        rail_poly = np.asarray(env.get_rail_polyline(num=48), dtype=float) if hasattr(env, "get_rail_polyline") else np.asarray([[float(getattr(env, "start", [-0.224])[0]), 0.0], [slot_x, 0.0]], dtype=float)
    if forward_ext > 1e-8 and len(rail_poly) >= 2:
        end_vec_for_ext = np.asarray(rail_poly[-1] - rail_poly[-2], dtype=float)
        end_len_for_ext = max(float(np.linalg.norm(end_vec_for_ext)), 1e-8)
        end_tangent_for_ext = end_vec_for_ext / end_len_for_ext
        rail_poly = np.vstack([rail_poly, rail_poly[-1] + float(forward_ext) * end_tangent_for_ext])
    rail_min = np.min(rail_poly, axis=0)
    rail_max = np.max(rail_poly, axis=0)
    fixture_len = max(wall_len + 0.105 + forward_ext, float(rail_max[0] - rail_min[0]) + 0.15)
    fixture_center_x = 0.5 * float(rail_min[0] + rail_max[0]) + 0.025
    fixture_center_y = 0.5 * float(rail_min[1] + rail_max[1])
    fixture_half_y = slot_half_width + wall_thick + 0.030
    fixture_half_y = max(fixture_half_y, 0.5 * float(rail_max[1] - rail_min[1]) + slot_half_width + wall_thick + 0.030)
    fixture_parts = []
    fixture_parts.append(
        _spawn_box(
            client_id,
            half_extents=(0.5 * fixture_len, fixture_half_y, 0.004),
            rgba=(0.24, 0.27, 0.29, 1.0),
            specular=(0.30, 0.30, 0.30),
            position=tracker.s5_to_world(np.asarray([fixture_center_x, fixture_center_y, float(_surface_height(env, [[fixture_center_x, fixture_center_y]])[0]) + 0.004])),
            orientation=_surface_quat(env, 0.0),
            collision=True,
        )
    )
    wall_ids = []
    for a, b in zip(rail_poly[:-1], rail_poly[1:]):
        vec = np.asarray(b - a, dtype=float)
        seg_len = float(np.linalg.norm(vec))
        if seg_len <= 1e-8:
            continue
        tangent = vec / seg_len
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        angle = float(math.atan2(tangent[1], tangent[0]))
        mid = 0.5 * (a + b)
        for sign in (-1.0, 1.0):
            rail_xy = mid + sign * (slot_half_width + wall_thick) * normal
            wall_ids.append(
                _spawn_box(
                    client_id,
                    half_extents=(0.5 * seg_len + 0.001, wall_thick, wall_height),
                    rgba=(0.68, 0.71, 0.72, 1.0),
                    specular=(0.95, 0.95, 0.95),
                    position=tracker.s5_to_world(np.asarray([rail_xy[0], rail_xy[1], float(_surface_height(env, [rail_xy])[0]) + wall_height])),
                    orientation=_surface_quat(env, angle),
                    collision=True,
                )
            )
            trim_xy = mid + sign * (slot_half_width + 0.0012) * normal
            wall_ids.append(
                _spawn_box(
                    client_id,
                    half_extents=(0.5 * seg_len + 0.001, 0.0024, 0.0030),
                    rgba=(0.93, 0.94, 0.92, 1.0),
                    specular=(1.0, 1.0, 1.0),
                    position=tracker.s5_to_world(np.asarray([trim_xy[0], trim_xy[1], float(_surface_height(env, [trim_xy])[0]) + 2.0 * wall_height + 0.003])),
                    orientation=_surface_quat(env, angle),
                    collision=False,
                )
            )
    start_vec = rail_poly[1] - rail_poly[0]
    start_len = max(float(np.linalg.norm(start_vec)), 1e-8)
    start_tangent = start_vec / start_len
    start_normal = np.asarray([-start_tangent[1], start_tangent[0]], dtype=float)
    start_angle = float(math.atan2(start_tangent[1], start_tangent[0]))
    funnel_len = 0.065
    for sign in (-1.0, 1.0):
        funnel_xy = rail_poly[0] - start_tangent * (0.5 * funnel_len - 0.006) + sign * (slot_half_width + 0.020) * start_normal
        wall_ids.append(
            _spawn_box(
                client_id,
                half_extents=(0.5 * funnel_len, 0.006, 0.020),
                rgba=(0.70, 0.73, 0.74, 1.0),
                specular=(0.9, 0.9, 0.9),
                position=tracker.s5_to_world(np.asarray([funnel_xy[0], funnel_xy[1], float(_surface_height(env, [funnel_xy])[0]) + 0.020])),
                orientation=_surface_quat(env, start_angle - sign * 0.22),
                collision=True,
            )
        )
    end_vec = rail_poly[-1] - rail_poly[-2]
    end_len = max(float(np.linalg.norm(end_vec)), 1e-8)
    end_tangent = end_vec / end_len
    end_angle = float(math.atan2(end_tangent[1], end_tangent[0]))
    end_stop = _spawn_box(
        client_id,
        half_extents=(0.010, fixture_half_y, 0.022),
        rgba=(0.34, 0.37, 0.39, 1.0),
        specular=(0.6, 0.6, 0.6),
        position=tracker.s5_to_world(np.asarray([rail_poly[-1, 0] + 0.012 * end_tangent[0], rail_poly[-1, 1] + 0.012 * end_tangent[1], float(_surface_height(env, [[rail_poly[-1, 0], rail_poly[-1, 1]]])[0]) + 0.022])),
        orientation=_surface_quat(env, end_angle),
        collision=True,
    )
    fixture_parts.append(end_stop)
    for bx in (fixture_center_x - 0.5 * fixture_len + 0.025, slot_x - 0.015):
        for by in (fixture_center_y - fixture_half_y + 0.014, fixture_center_y + fixture_half_y - 0.014):
            fixture_parts.append(
                _spawn_cylinder(
                    client_id,
                    radius=0.0045,
                    height=0.0045,
                    rgba=(0.08, 0.085, 0.09, 1.0),
                    specular=(1.0, 1.0, 1.0),
                    position=tracker.s5_to_world(np.asarray([bx, by, float(_surface_height(env, [[bx, by]])[0]) + 0.010])),
                    collision=False,
                )
            )
    slider_half = np.asarray(getattr(env, 'slider_half_extents', (0.080, 0.030, 0.018)), dtype=float)
    slider_parts = []
    tongue_width = max(0.010, min(0.014, 0.45 * float(slot_half_width)))
    slider_parts.append(
        {
            'body': _spawn_box(
                client_id,
                half_extents=(0.016, 0.060, tongue_width),
                rgba=(0.74, 0.76, 0.76, 1.0),
                specular=(1.0, 1.0, 1.0),
                position=tracker.s5_to_world(np.asarray([0.0, 0.0, slider_half[0]])),
                collision=False,
            ),
            'offset': np.asarray([-slider_half[0] + 0.016, -0.006, 0.0], dtype=float),
        }
    )
    slider_parts.append(
        {
            'body': _spawn_box(
                client_id,
                half_extents=(0.032, 0.044, 0.025),
                rgba=(0.68, 0.27, 0.16, 1.0),
                specular=(0.55, 0.36, 0.25),
                position=tracker.s5_to_world(np.asarray([0.0, 0.0, slider_half[0]])),
                collision=False,
            ),
            'offset': np.asarray([-0.034, -0.006, 0.0], dtype=float),
        }
    )
    slider_parts.append(
        {
            'body': _spawn_box(
                client_id,
                half_extents=(0.070, 0.020, 0.020),
                rgba=(0.74, 0.31, 0.18, 1.0),
                specular=(0.55, 0.36, 0.25),
                position=tracker.s5_to_world(np.asarray([0.0, 0.0, slider_half[0]])),
                collision=False,
            ),
            'offset': np.asarray([0.018, -0.006, 0.0], dtype=float),
        }
    )
    slider_parts.append(
        {
            'body': _spawn_box(
                client_id,
                half_extents=(0.016, 0.026, 0.020),
                rgba=(0.035, 0.038, 0.040, 1.0),
                specular=(0.06, 0.06, 0.06),
                position=tracker.s5_to_world(np.asarray([0.0, 0.0, slider_half[0]])),
                collision=False,
            ),
            'offset': np.asarray([slider_half[0] - 0.018, -0.006, 0.0], dtype=float),
        }
    )
    slider_parts.append(
        {
            'body': _spawn_box(
                client_id,
                half_extents=(0.045, 0.0035, 0.003),
                rgba=(0.94, 0.76, 0.22, 1.0),
                specular=(0.35, 0.28, 0.12),
                position=tracker.s5_to_world(np.asarray([0.0, 0.0, slider_half[0]])),
                collision=False,
            ),
            'offset': np.asarray([0.030, -0.006, 0.023], dtype=float),
        }
    )
    slider_parts.append(
        {
            'body': _spawn_box(
                client_id,
                half_extents=(0.014, 0.016, tongue_width),
                rgba=(0.78, 0.80, 0.80, 1.0),
                specular=(1.0, 1.0, 1.0),
                position=tracker.s5_to_world(np.asarray([0.0, 0.0, slider_half[0]])),
                orientation=_yaw_quat(0.24),
                collision=False,
            ),
            'offset': np.asarray([-slider_half[0] + 0.016, -0.058, 0.0], dtype=float),
            'local_quat': _yaw_quat(0.24),
        }
    )
    hidden = origin + np.asarray([0.0, 0.0, -0.8], dtype=float)
    normal_load_arrow = []
    for _ in range(20):
        normal_load_arrow.append(
            _spawn_box(
                client_id,
                half_extents=(0.0040, 0.0013, 0.0013),
                rgba=(0.95, 0.45, 0.08, 1.0),
                specular=(0.4, 0.25, 0.10),
                position=hidden,
                collision=False,
            )
        )
    for _ in range(4):
        normal_load_arrow.append(
            _spawn_box(
                client_id,
                half_extents=(0.0130, 0.0017, 0.0017),
                rgba=(0.95, 0.25, 0.08, 1.0),
                specular=(0.4, 0.18, 0.10),
                position=hidden,
                collision=False,
            )
        )
    return {
        'table': table_id,
        'walls': wall_ids,
        'fixture': fixture_parts,
        'slider_parts': slider_parts,
        'slider': slider_parts[0]['body'],
        'normal_load_arrow': normal_load_arrow,
    }


def _slider_center_state(env, state: np.ndarray) -> np.ndarray:
    st = np.asarray(state, dtype=float).reshape(-1).copy()
    slider_half = np.asarray(getattr(env, 'slider_half_extents', (0.080, 0.030, 0.018)), dtype=float)
    st[2] = float(st[2]) + float(slider_half[0])
    return st


def _slider_part_state(env, state: np.ndarray, local_offset: np.ndarray) -> np.ndarray:
    center = _slider_center_state(env, state)
    theta = float(center[3])
    up, heading, lateral = _surface_frame_axes(env, theta)
    offset = np.asarray(local_offset, dtype=float).reshape(3)
    out = center.copy()
    out[:3] = center[:3] + offset[0] * up + offset[1] * heading + offset[2] * lateral
    return out


def _grasp_state(env, state: np.ndarray) -> np.ndarray:
    st = np.asarray(state, dtype=float).reshape(-1).copy()
    slider_half = np.asarray(getattr(env, 'slider_half_extents', (0.080, 0.030, 0.018)), dtype=float)
    grasp_height = float(getattr(env, 'pybullet_grasp_height', 0.070))
    up = _surface_normal(env)
    st[:3] = st[:3] + up * (float(slider_half[0]) + grasp_height)
    return st


def _set_slider_pose(env, tracker: _UR5PoseTracker, slider_ids, state: np.ndarray) -> None:
    if isinstance(slider_ids, list):
        for part in slider_ids:
            st = _slider_part_state(env, state, np.asarray(part.get('offset', np.zeros(3)), dtype=float))
            base_quat = _quat_for_slider_theta(env, float(st[3]))
            local_quat = part.get('local_quat')
            quat = _compose_quat(base_quat, local_quat) if local_quat is not None else base_quat
            p.resetBasePositionAndOrientation(
                int(part['body']),
                [float(v) for v in tracker.s5_to_world(st[:3])],
                [float(v) for v in quat],
                physicsClientId=tracker.client_id,
            )
        return
    st = _slider_center_state(env, state)
    p.resetBasePositionAndOrientation(
        int(slider_ids),
        [float(v) for v in tracker.s5_to_world(st[:3])],
        [float(v) for v in _quat_for_slider_theta(env, float(st[3]))],
        physicsClientId=tracker.client_id,
    )


def _camera_frame(env, tracker: _UR5PoseTracker, *, width: int, height: int) -> np.ndarray:
    target = np.asarray(getattr(env, 'pybullet_camera_target', (0.55, 0.0, 0.54)), dtype=float)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[float(v) for v in target],
        distance=float(getattr(env, 'pybullet_camera_distance', 0.78)),
        yaw=float(getattr(env, 'pybullet_camera_yaw', 78.0)),
        pitch=float(getattr(env, 'pybullet_camera_pitch', -34.0)),
        roll=0.0,
        upAxisIndex=2,
        physicsClientId=tracker.client_id,
    )
    proj = p.computeProjectionMatrixFOV(fov=float(getattr(env, 'pybullet_camera_fov', 42.0)), aspect=float(width) / float(height), nearVal=0.02, farVal=3.0)
    _, _, rgb, _, _ = p.getCameraImage(int(width), int(height), viewMatrix=view, projectionMatrix=proj, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=tracker.client_id)
    return np.asarray(rgb, dtype=np.uint8).reshape(int(height), int(width), 4)[:, :, :3]


def _compute_ik_commands(env, tracker: _UR5PoseTracker, ref: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grasp_ref = np.asarray([_grasp_state(env, st) for st in ref[:, :4]], dtype=float)
    target_tip_world = tracker.s5_to_world(grasp_ref[:, :3])
    track_orientation = bool(getattr(env, "pybullet_s4_track_orientation", True))
    q_pos = np.zeros((len(ref), 6), dtype=float)
    pos_quat = np.zeros((len(ref), 4), dtype=float)
    q_prev = tracker.home_q.copy()
    for i in range(len(ref)):
        q_prev = tracker._run_pybullet_ik(target_tip_world[i], target_quat=None, rest_q=q_prev)
        tracker.reset_joint_state(q_prev)
        _, ee_quat = tracker.get_ee_pose()
        q_pos[i] = q_prev
        pos_quat[i] = ee_quat
    if not track_orientation:
        return q_pos, target_tip_world, pos_quat

    target_quat = np.zeros_like(pos_quat)
    tilt_quat = _quat_between_vectors([0.0, 0.0, 1.0], _surface_normal(env))
    for i in range(len(ref)):
        theta_now = _theta_from_ee_quat(pos_quat[i], float(ref[i, 3]))
        yaw_delta = float(math.atan2(math.sin(float(ref[i, 3]) - theta_now), math.cos(float(ref[i, 3]) - theta_now)))
        yaw_quat = np.asarray(_compose_quat(_yaw_quat(yaw_delta), pos_quat[i]), dtype=float)
        target_quat[i] = np.asarray(_compose_quat(tilt_quat, yaw_quat), dtype=float)

    q_cmd = np.zeros_like(q_pos)
    q_prev = q_pos[0].copy()
    for i in range(len(ref)):
        q_prev = tracker._run_pybullet_ik(target_tip_world[i], target_quat=target_quat[i], rest_q=q_prev)
        tracker.reset_joint_state(q_prev)
        q_cmd[i] = q_prev
    return q_cmd, target_tip_world, target_quat


def _theta_from_ee_quat(ee_quat: np.ndarray, fallback_theta: float) -> float:
    rot = np.asarray(p.getMatrixFromQuaternion(np.asarray(ee_quat, dtype=float).reshape(4).tolist()), dtype=float).reshape(3, 3)
    heading = np.asarray(rot[:, 1], dtype=float)
    heading[2] = 0.0
    if float(np.linalg.norm(heading)) <= 1e-8:
        return float(fallback_theta)
    heading = heading / max(float(np.linalg.norm(heading)), 1e-12)
    return float(math.atan2(float(heading[1]), float(heading[0])))


def _executed_slider_state_from_ee(env, tracker: _UR5PoseTracker, ee_pos_world: np.ndarray, ee_quat: np.ndarray, fallback_theta: float) -> np.ndarray:
    grasp_s4 = np.asarray(tracker.world_to_s5(np.asarray(ee_pos_world, dtype=float).reshape(3)), dtype=float)
    slider_half = np.asarray(getattr(env, 'slider_half_extents', (0.080, 0.030, 0.018)), dtype=float)
    grasp_height = float(getattr(env, 'pybullet_grasp_height', 0.070))
    out = np.zeros(4, dtype=float)
    out[3] = _theta_from_ee_quat(ee_quat, fallback_theta)
    out[:3] = grasp_s4 - _surface_normal(env) * (float(slider_half[0]) + grasp_height)
    return out


def _remove_debug_items(tracker: _UR5PoseTracker, item_ids: list[int]) -> None:
    for item_id in item_ids:
        try:
            p.removeUserDebugItem(int(item_id), physicsClientId=tracker.client_id)
        except Exception:
            pass


def _normal_load_color(load_frac: float) -> tuple[float, float, float]:
    u = float(np.clip(load_frac, 0.0, 1.0))
    if u < 0.5:
        a = u / 0.5
        return 0.10 + 0.85 * a, 0.42 + 0.34 * a, 0.90 * (1.0 - a)
    a = (u - 0.5) / 0.5
    return 0.95, 0.76 * (1.0 - a) + 0.18 * a, 0.08


def _normal_load_arrow_geometry(env, state: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    st = np.asarray(state, dtype=float).reshape(-1)
    theta = float(st[3])
    lateral = np.asarray([-math.sin(theta), math.cos(theta), 0.0], dtype=float)
    heading = np.asarray([math.cos(theta), math.sin(theta), 0.0], dtype=float)
    slider_half = np.asarray(getattr(env, 'slider_half_extents', (0.080, 0.026, 0.018)), dtype=float)
    length = 0.012 + float(getattr(env, 'pybullet_normal_load_arrow_scale', 0.055)) * float(frac)
    base = st[:3].copy()
    base[2] += float(slider_half[0]) + float(getattr(env, 'pybullet_grasp_height', 0.070)) + 0.045
    start = base + lateral * 0.115 - heading * 0.030
    direction = np.asarray([0.0, 0.0, -1.0], dtype=float)
    end = start + direction * length
    return start, end, direction, lateral, heading


def _project_s4_point_to_pixel(env, tracker: _UR5PoseTracker, point_s4: np.ndarray, *, width: int, height: int) -> tuple[int, int] | None:
    target = np.asarray(getattr(env, 'pybullet_camera_target', (0.55, 0.0, 0.54)), dtype=float)
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[float(v) for v in target],
        distance=float(getattr(env, 'pybullet_camera_distance', 0.78)),
        yaw=float(getattr(env, 'pybullet_camera_yaw', 78.0)),
        pitch=float(getattr(env, 'pybullet_camera_pitch', -34.0)),
        roll=0.0,
        upAxisIndex=2,
        physicsClientId=tracker.client_id,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=float(getattr(env, 'pybullet_camera_fov', 42.0)),
        aspect=float(width) / float(height),
        nearVal=0.02,
        farVal=3.0,
    )
    point_world = tracker.s5_to_world(np.asarray(point_s4, dtype=float).reshape(3))
    view_m = np.asarray(view, dtype=float).reshape(4, 4).T
    proj_m = np.asarray(proj, dtype=float).reshape(4, 4).T
    clip = proj_m @ (view_m @ np.asarray([point_world[0], point_world[1], point_world[2], 1.0], dtype=float))
    if abs(float(clip[3])) <= 1e-9:
        return None
    ndc = clip[:3] / float(clip[3])
    if not np.all(np.isfinite(ndc)):
        return None
    x = int((float(ndc[0]) + 1.0) * 0.5 * int(width))
    y = int((1.0 - float(ndc[1])) * 0.5 * int(height))
    return x, y


def _overlay_normal_force_label(frame: np.ndarray, env, tracker: _UR5PoseTracker, state: np.ndarray, *, load_value: float, load_max: float) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return frame
    frac = float(np.clip(float(load_value) / max(float(load_max), 1e-9), 0.0, 1.0))
    if frac <= 1e-4:
        return frame
    height, width = int(frame.shape[0]), int(frame.shape[1])
    start, _, _, lateral, _ = _normal_load_arrow_geometry(env, state, frac)
    label_point = start + lateral * 0.034 + np.asarray([0.0, 0.0, 0.010], dtype=float)
    pixel = _project_s4_point_to_pixel(env, tracker, label_point, width=width, height=height)
    if pixel is None:
        return frame
    x, y = pixel
    x = int(np.clip(x, 8, width - 170))
    y = int(np.clip(y, 8, height - 34))
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode='RGB')
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default() if ImageFont is not None else None
    text = "normal force"
    draw.text((x + 2, y + 2), text, fill=(230, 230, 230), font=font)
    draw.text((x, y), text, fill=(10, 10, 10), font=font)
    return np.asarray(image, dtype=np.uint8)


def _stage_spans(cutpoints: np.ndarray | None, length: int) -> list[tuple[int, int]]:
    if length <= 0:
        return []
    cuts = []
    if cutpoints is not None:
        cuts = [int(v) for v in np.asarray(cutpoints, dtype=int).reshape(-1).tolist() if 0 <= int(v) < length - 1]
    ends = cuts + [length - 1]
    starts = [0] + [int(v) + 1 for v in ends[:-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _dash_line(draw, xy, *, fill, width=1, dash=4, gap=3) -> None:
    x0, y0, x1, y1 = [float(v) for v in xy]
    dist = math.hypot(x1 - x0, y1 - y0)
    if dist <= 1e-9:
        return
    ux = (x1 - x0) / dist
    uy = (y1 - y0) / dist
    s = 0.0
    while s < dist:
        e = min(s + float(dash), dist)
        draw.line((x0 + ux * s, y0 + uy * s, x0 + ux * e, y0 + uy * e), fill=fill, width=int(width))
        s += float(dash + gap)


def _constraint_feature_specs(env) -> tuple[list[dict[str, object]], list[str], dict[str, int]]:
    schema = list(env.get_feature_schema()) if hasattr(env, "get_feature_schema") else list(getattr(env, "feature_schema", []))
    name_to_idx = {}
    schema_names = []
    for idx, spec in enumerate(schema):
        name = str(spec.get("name", f"feature_{idx}"))
        col = int(spec.get("column_idx", spec.get("id", idx)))
        name_to_idx[name] = col
        schema_names.append(name)
    if hasattr(env, "get_overlay_feature_names"):
        feature_names = [str(name) for name in env.get_overlay_feature_names() if str(name) in name_to_idx]
    else:
        feature_names = []
    seen = set(feature_names)
    for spec in list(env.get_constraint_specs()) if hasattr(env, "get_constraint_specs") else list(getattr(env, "constraint_specs", [])):
        name = str(spec.get("feature_name", ""))
        if name not in name_to_idx:
            continue
        if name not in seen:
            feature_names.append(name)
            seen.add(name)
    if not feature_names:
        feature_names = [name for name in schema_names if name in name_to_idx]
    return list(getattr(env, "constraint_specs", env.get_constraint_specs() if hasattr(env, "get_constraint_specs") else [])), feature_names, name_to_idx


def _constraint_semantics_kind(spec: dict[str, object]) -> str:
    text = str(spec.get("semantics", "")).strip().lower()
    if text in {"target", "target_value", "equality", "eq", "equal"}:
        return "target"
    if text in {"upper", "upper_bound", "max", "maximum", "<=", "leq"}:
        return "upper"
    if text in {"lower", "lower_bound", "min", "minimum", ">=", "geq"}:
        return "lower"
    return text


def _overlay_constraint_feature_panel(
    frame: np.ndarray,
    env,
    executed_prefix: np.ndarray,
    *,
    current_index: int,
    total_length: int,
    true_cutpoints: np.ndarray | None,
    normal_load_trace: np.ndarray | None = None,
    ylim_trajectory: np.ndarray | None = None,
    title: str = "Executed features",
) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return frame
    prefix = np.asarray(executed_prefix, dtype=float)
    if prefix.ndim != 2 or prefix.shape[0] <= 0:
        return frame
    specs, feature_names, name_to_idx = _constraint_feature_specs(env)
    if not feature_names:
        return frame
    ylim_traj = None if ylim_trajectory is None else np.asarray(ylim_trajectory, dtype=float)
    if normal_load_trace is not None and len(normal_load_trace) >= len(prefix) and hasattr(env, "register_normal_load_trace"):
        env.register_normal_load_trace(prefix[:, :4], np.asarray(normal_load_trace[: len(prefix)], dtype=float))
    if (
        ylim_traj is not None
        and ylim_traj.ndim == 2
        and ylim_traj.shape[0] > 0
        and normal_load_trace is not None
        and len(normal_load_trace) >= len(ylim_traj)
        and hasattr(env, "register_normal_load_trace")
    ):
        env.register_normal_load_trace(ylim_traj[:, :4], np.asarray(normal_load_trace[: len(ylim_traj)], dtype=float))
    try:
        F = np.asarray(env.compute_all_features_matrix(prefix[:, :4]), dtype=float)
        F_ylim = None if ylim_traj is None or ylim_traj.ndim != 2 or ylim_traj.shape[0] <= 0 else np.asarray(env.compute_all_features_matrix(ylim_traj[:, :4]), dtype=float)
    except Exception:
        return frame
    true_constraints = dict(getattr(env, "true_constraints", {}) or {})
    spans = _stage_spans(true_cutpoints, int(total_length))
    height, width = int(frame.shape[0]), int(frame.shape[1])
    rows = len(feature_names)
    panel_w = int(min(max(445, 0.40 * width), max(width - 28, 1), 610))
    title_text = str(title or "Executed features")
    if " (planned with " in title_text:
        first, rest = title_text.split(" (planned with ", 1)
        title_lines = [first, "(planned with " + rest]
    else:
        title_lines = [title_text]
    header_h = 66 + 15 * max(0, len(title_lines) - 1)
    pad = 10
    available_h = max(1, height - 2 * 14 - pad)
    row_h = int(np.clip((available_h - header_h) / max(rows, 1), 34, 54))
    panel_h = header_h + rows * row_h + pad
    x0 = max(8, width - panel_w - 14)
    y0 = 14
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 12)
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except Exception:
        font = ImageFont.load_default() if ImageFont is not None else None
        font_bold = font
        font_title = font

    draw.rounded_rectangle((x0, y0, x0 + panel_w, y0 + panel_h), radius=7, fill=(255, 255, 255, 218), outline=(36, 42, 50, 185), width=1)
    for line_idx, line in enumerate(title_lines):
        draw.text((x0 + 9, y0 + 5 + 15 * line_idx), line, fill=(20, 24, 30, 255), font=font_title)
    constraint_orange = (234, 88, 12, 245)
    constraint_orange_text = (194, 65, 12, 255)
    feasible_yellow = (254, 240, 138, 145)
    equality_band = (253, 186, 116, 92)
    legend_y = y0 + 27 + 15 * max(0, len(title_lines) - 1)
    legend_x0 = x0 + 10
    draw.line((legend_x0, legend_y + 6, legend_x0 + 26, legend_y + 6), fill=constraint_orange, width=3)
    draw.text((legend_x0 + 32, legend_y), "GT equality constraint target", fill=constraint_orange_text, font=font)
    legend_y2 = legend_y + 16
    legend_x1 = legend_x0
    draw.rectangle((legend_x1, legend_y2 + 2, legend_x1 + 28, legend_y2 + 12), fill=feasible_yellow)
    _dash_line(draw, (legend_x1, legend_y2 + 6, legend_x1 + 28, legend_y2 + 6), fill=constraint_orange, width=2, dash=6, gap=4)
    draw.text((legend_x1 + 34, legend_y2), "GT inequality constraint bound and feasible region", fill=constraint_orange_text, font=font)
    plot_x0 = x0 + 120
    plot_x1 = x0 + panel_w - 12
    plot_w = max(1, plot_x1 - plot_x0)
    total_den = max(int(total_length) - 1, 1)

    for row, name in enumerate(feature_names):
        feat_idx = int(name_to_idx[name])
        if feat_idx >= F.shape[1]:
            continue
        ry0 = y0 + header_h + row * row_h + 5
        ry1 = ry0 + row_h - 12
        py0 = ry0 + 4
        py1 = ry1
        trace = np.asarray(F[:, feat_idx], dtype=float)
        finite_vals = trace[np.isfinite(trace)]
        if F_ylim is not None and feat_idx < F_ylim.shape[1]:
            ylim_trace = np.asarray(F_ylim[:, feat_idx], dtype=float)
            finite_ylim = ylim_trace[np.isfinite(ylim_trace)]
            if finite_ylim.size:
                finite_vals = np.concatenate([finite_vals, finite_ylim]) if finite_vals.size else finite_ylim
        spec_vals = []
        for spec in specs:
            if str(spec.get("feature_name", "")) != name:
                continue
            key = str(spec.get("oracle_key", ""))
            if key in true_constraints and np.isfinite(float(true_constraints[key])):
                spec_vals.append(float(true_constraints[key]))
        all_vals = np.concatenate([finite_vals, np.asarray(spec_vals, dtype=float)]) if spec_vals else finite_vals
        if all_vals.size == 0:
            continue
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        if abs(vmax - vmin) < 1e-8:
            margin = max(1e-4, abs(vmax) * 0.15 + 1e-4)
            vmin -= margin
            vmax += margin
        else:
            margin = 0.12 * (vmax - vmin)
            vmin -= margin
            vmax += margin

        def x_at(t: int) -> float:
            return float(plot_x0) + float(np.clip(t, 0, total_den)) / float(total_den) * float(plot_w)

        def y_at(v: float) -> float:
            return float(py1) - (float(v) - vmin) / max(vmax - vmin, 1e-12) * float(py1 - py0)

        draw.text((x0 + 9, ry0 + 4), name, fill=(18, 24, 38, 255), font=font_bold)
        if trace.size:
            draw.text((x0 + 9, ry0 + 21), f"{trace[-1]:.3g}", fill=(37, 99, 235, 255), font=font)
        draw.rectangle((plot_x0, py0, plot_x1, py1), outline=(188, 196, 206, 220), fill=(248, 250, 252, 205), width=1)
        for cp in np.asarray(true_cutpoints if true_cutpoints is not None else [], dtype=int).reshape(-1):
            x = x_at(int(cp))
            draw.line((x, py0, x, py1), fill=(150, 158, 170, 150), width=1)
        for spec in specs:
            if str(spec.get("feature_name", "")) != name:
                continue
            stage_idx = int(spec.get("stage", -1))
            if stage_idx < 0 or stage_idx >= len(spans):
                continue
            key = str(spec.get("oracle_key", ""))
            if key not in true_constraints:
                continue
            value = float(true_constraints[key])
            if not np.isfinite(value):
                continue
            a, b = spans[stage_idx]
            y = y_at(value)
            xa, xb = x_at(a), x_at(b)
            kind = _constraint_semantics_kind(spec)
            if kind == "target":
                band = max(1.5, 0.05 * float(py1 - py0))
                draw.rectangle((xa, max(py0, y - band), xb, min(py1, y + band)), fill=equality_band)
                draw.line((xa, y, xb, y), fill=constraint_orange, width=3)
            elif kind == "upper":
                draw.rectangle((xa, max(py0, min(py1, y)), xb, py1), fill=feasible_yellow)
                _dash_line(draw, (xa, y, xb, y), fill=constraint_orange, width=2, dash=6, gap=4)
            elif kind == "lower":
                draw.rectangle((xa, py0, xb, max(py0, min(py1, y))), fill=feasible_yellow)
                _dash_line(draw, (xa, y, xb, y), fill=constraint_orange, width=2, dash=6, gap=4)
            else:
                _dash_line(draw, (xa, y, xb, y), fill=constraint_orange, width=2, dash=6, gap=4)
        if trace.size >= 2:
            pts = [(x_at(i), y_at(float(v))) for i, v in enumerate(trace) if np.isfinite(float(v))]
            if len(pts) >= 2:
                draw.line(pts, fill=(37, 99, 235, 255), width=2)
        elif trace.size == 1 and np.isfinite(float(trace[0])):
            x = x_at(0)
            y = y_at(float(trace[0]))
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(37, 99, 235, 255))
        cx = x_at(int(current_index))
        draw.line((cx, py0, cx, py1), fill=(99, 102, 241, 210), width=1)

    composed = Image.alpha_composite(image, overlay).convert("RGB")
    return np.asarray(composed, dtype=np.uint8)


def _draw_normal_load_arrow(
    env,
    tracker: _UR5PoseTracker,
    state: np.ndarray,
    *,
    load_value: float,
    load_max: float,
    old_item_ids: list[int],
    visual_bodies=None,
) -> list[int]:
    _remove_debug_items(tracker, old_item_ids)
    frac = float(np.clip(float(load_value) / max(float(load_max), 1e-9), 0.0, 1.0))
    visual_bodies = list(visual_bodies or [])
    hidden = np.asarray(getattr(env, 'pybullet_world_center', (0.55, 0.0, 0.52)), dtype=float) + np.asarray([0.0, 0.0, -0.8], dtype=float)
    if visual_bodies:
        for body in visual_bodies:
            p.resetBasePositionAndOrientation(
                int(body),
                hidden.tolist(),
                [0.0, 0.0, 0.0, 1.0],
                physicsClientId=tracker.client_id,
            )
    if frac <= 1e-4:
        return []
    start, end, direction, lateral, heading = _normal_load_arrow_geometry(env, state, frac)
    length = float(np.linalg.norm(end - start))
    head_len = min(0.018, 0.45 * length)
    head_a = end - direction * head_len + lateral * 0.010
    head_b = end - direction * head_len - lateral * 0.010
    color = _normal_load_color(frac)
    width = 2.0 + 4.0 * frac
    if visual_bodies:
        head_count = 4
        shaft_bodies = visual_bodies[:-head_count]
        head_bodies = visual_bodies[-head_count:]
        shaft_quat = _quat_align_local_x_to_vec(direction)
        n_segments = max(1, int(math.ceil(frac * min(len(shaft_bodies), max(len(shaft_bodies), 1)))))
        segment_step = length / max(float(n_segments), 1.0)
        for k, body in enumerate(shaft_bodies):
            if k >= n_segments:
                continue
            pos = start + direction * ((k + 0.5) * segment_step)
            p.resetBasePositionAndOrientation(
                int(body),
                tracker.s5_to_world(pos).tolist(),
                list(shaft_quat),
                physicsClientId=tracker.client_id,
            )
            p.changeVisualShape(int(body), -1, rgbaColor=[*color, 1.0], physicsClientId=tracker.client_id)
        head_span = min(0.018, max(0.011, 0.38 * length))
        head_vecs = [
            -direction * head_span + lateral * 0.012,
            -direction * head_span - lateral * 0.012,
            -direction * head_span + heading * 0.012,
            -direction * head_span - heading * 0.012,
        ]
        for body, vec in zip(head_bodies, head_vecs):
            center = end + 0.5 * vec
            quat = _quat_align_local_x_to_vec(vec)
            p.resetBasePositionAndOrientation(
                int(body),
                tracker.s5_to_world(center).tolist(),
                list(quat),
                physicsClientId=tracker.client_id,
            )
            p.changeVisualShape(int(body), -1, rgbaColor=[*color, 1.0], physicsClientId=tracker.client_id)
    ids = [
        p.addUserDebugLine(
            tracker.s5_to_world(start).tolist(),
            tracker.s5_to_world(end).tolist(),
            lineColorRGB=list(color),
            lineWidth=float(width),
            lifeTime=0.0,
            physicsClientId=tracker.client_id,
        ),
        p.addUserDebugLine(
            tracker.s5_to_world(end).tolist(),
            tracker.s5_to_world(head_a).tolist(),
            lineColorRGB=list(color),
            lineWidth=float(width),
            lifeTime=0.0,
            physicsClientId=tracker.client_id,
        ),
        p.addUserDebugLine(
            tracker.s5_to_world(end).tolist(),
            tracker.s5_to_world(head_b).tolist(),
            lineColorRGB=list(color),
            lineWidth=float(width),
            lifeTime=0.0,
            physicsClientId=tracker.client_id,
        ),
    ]
    return [int(x) for x in ids]


def _play_s4_reference(
    env,
    tracker: _UR5PoseTracker,
    scene_ids: dict[str, object],
    ref: np.ndarray,
    *,
    gui: int,
    video_path: str | Path | None,
    fps: float,
    width: int,
    height: int,
    render_frame_stride: int,
    video_end_hold_seconds: float = 2.0,
    realtime: bool,
    gui_hold_seconds: float,
    normal_load_trace: np.ndarray | None = None,
    visualize_normal_load: bool = True,
    true_cutpoints: np.ndarray | None = None,
    feature_overlay: bool = True,
    feature_overlay_title: str | None = None,
    execution_joint_noise_std: float = 0.0,
    execution_joint_noise_smooth: float = 0.90,
    execution_noise_seed: int | None = None,
) -> dict[str, Any]:
    sim_dt = float(getattr(env, 'pybullet_sim_dt', 1.0 / 120.0))
    configured_steps = getattr(env, 'pybullet_steps_per_sample', None)
    steps_per_sample = max(1, int(round(float(env.dt) / sim_dt))) if configured_steps is None else int(configured_steps)
    q_nominal_cmd, target_tip_world, target_quat = _compute_ik_commands(env, tracker, ref)
    q_noise = _smooth_command_noise(
        q_nominal_cmd.shape,
        std=float(execution_joint_noise_std),
        smooth=float(execution_joint_noise_smooth),
        seed=execution_noise_seed,
    )
    q_cmd = q_nominal_cmd + q_noise
    writer = None
    save_video = int(gui) == 1 and video_path is not None
    use_gui = int(gui) == 2
    if save_video:
        writer = _FFmpegVideoWriter(out_path=video_path, width=width, height=height, fps=float(fps))
    frame_count = 0
    executed = np.zeros((len(ref), 4), dtype=float)
    realized_ee_world = np.zeros((len(ref), 3), dtype=float)
    realized_ee_quat = np.zeros((len(ref), 4), dtype=float)
    q_meas = np.zeros_like(q_cmd)
    qd_meas = np.zeros_like(q_cmd)
    load_trace = None if normal_load_trace is None else np.asarray(normal_load_trace, dtype=float).reshape(-1)
    load_max = float(np.max(load_trace)) if load_trace is not None and load_trace.size > 0 else float(getattr(env, 'normal_load_min', 1.0))
    load_visual_ids: list[int] = []
    try:
        tracker.reset_joint_state(q_cmd[0])
        _close_gripper_for_visual(tracker)
        tracker.command_joint_target(q_cmd[0])
        tracker.step(max(20, steps_per_sample))
        ee_pos, ee_quat = tracker.get_ee_pose()
        executed[0] = _executed_slider_state_from_ee(env, tracker, ee_pos, ee_quat, float(ref[0, 3]))
        realized_ee_world[0] = ee_pos
        realized_ee_quat[0] = ee_quat
        _set_slider_pose(env, tracker, scene_ids.get('slider_parts', scene_ids['slider']), executed[0])
        if bool(visualize_normal_load) and load_trace is not None and len(load_trace) == len(ref):
            load_visual_ids = _draw_normal_load_arrow(
                env,
                tracker,
                executed[0],
                load_value=float(load_trace[0]),
                load_max=load_max,
                old_item_ids=load_visual_ids,
                visual_bodies=scene_ids.get('normal_load_arrow'),
            )
        text_id = None
        if use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=float(getattr(env, 'pybullet_camera_distance', 0.78)),
                cameraYaw=float(getattr(env, 'pybullet_camera_yaw', 78.0)),
                cameraPitch=float(getattr(env, 'pybullet_camera_pitch', -34.0)),
                cameraTargetPosition=list(getattr(env, 'pybullet_camera_target', (0.55, 0.0, 0.54))),
                physicsClientId=tracker.client_id,
            )
        i = 0
        paused = False
        while i < len(ref):
            if use_gui and _space_was_triggered():
                paused = not paused
            if use_gui and paused:
                time.sleep(0.05)
                continue
            _close_gripper_for_visual(tracker)
            tracker.command_joint_target(q_cmd[i])
            tracker.step(steps_per_sample)
            q_i, qd_i = tracker.get_joint_state()
            q_meas[i] = q_i
            qd_meas[i] = qd_i
            ee_pos, ee_quat = tracker.get_ee_pose()
            executed[i] = _executed_slider_state_from_ee(env, tracker, ee_pos, ee_quat, float(ref[i, 3]))
            realized_ee_world[i] = ee_pos
            realized_ee_quat[i] = ee_quat
            _set_slider_pose(env, tracker, scene_ids.get('slider_parts', scene_ids['slider']), executed[i])
            if bool(visualize_normal_load) and load_trace is not None and len(load_trace) == len(ref):
                load_visual_ids = _draw_normal_load_arrow(
                    env,
                    tracker,
                    executed[i],
                    load_value=float(load_trace[i]),
                    load_max=load_max,
                    old_item_ids=load_visual_ids,
                    visual_bodies=scene_ids.get('normal_load_arrow'),
                )
            if writer is not None and i % max(int(render_frame_stride), 1) == 0:
                frame = _camera_frame(env, tracker, width=width, height=height)
                if bool(visualize_normal_load) and load_trace is not None and len(load_trace) == len(ref):
                    frame = _overlay_normal_force_label(frame, env, tracker, executed[i], load_value=float(load_trace[i]), load_max=load_max)
                if bool(feature_overlay):
                    frame = _overlay_constraint_feature_panel(
                        frame,
                        env,
                        executed[: i + 1],
                        current_index=i,
                        total_length=len(ref),
                        true_cutpoints=true_cutpoints,
                        normal_load_trace=load_trace,
                        ylim_trajectory=ref,
                        title=str(feature_overlay_title or "Executed features"),
                    )
                writer.append_data(frame)
                frame_count += 1
            if use_gui and bool(realtime):
                time.sleep(1.0 / max(float(fps), 1e-6))
            i += 1
        if use_gui:
            hold_seconds = float(gui_hold_seconds)
            if hold_seconds < 0.0:
                while True:
                    if _space_was_triggered():
                        break
                    time.sleep(0.1)
            elif hold_seconds > 0.0:
                time.sleep(hold_seconds)
        if writer is not None and float(video_end_hold_seconds) > 0.0 and len(executed) > 0:
            last_idx = len(executed) - 1
            hold_frames = int(round(float(video_end_hold_seconds) * float(fps)))
            for _ in range(max(0, hold_frames)):
                frame = _camera_frame(env, tracker, width=width, height=height)
                if bool(visualize_normal_load) and load_trace is not None and len(load_trace) == len(ref):
                    frame = _overlay_normal_force_label(frame, env, tracker, executed[last_idx], load_value=float(load_trace[last_idx]), load_max=load_max)
                if bool(feature_overlay):
                    frame = _overlay_constraint_feature_panel(
                        frame,
                        env,
                        executed,
                        current_index=last_idx,
                        total_length=len(ref),
                        true_cutpoints=true_cutpoints,
                        normal_load_trace=load_trace,
                        ylim_trajectory=ref,
                        title=str(feature_overlay_title or "Executed features"),
                    )
                writer.append_data(frame)
                frame_count += 1
    finally:
        if writer is not None:
            writer.close()
        _remove_debug_items(tracker, load_visual_ids)
    return {
        'trajectory': np.asarray(executed, dtype=float),
        'linear_velocity': _finite_difference(executed[:, :3], float(env.dt)),
        'angular_velocity': _finite_difference(executed[:, 3], float(env.dt)),
        'joint_positions': np.asarray(q_meas, dtype=float),
        'joint_velocities': np.asarray(qd_meas, dtype=float),
        'joint_position_commands': np.asarray(q_cmd, dtype=float),
        'joint_position_commands_nominal': np.asarray(q_nominal_cmd, dtype=float),
        'execution_joint_noise': np.asarray(q_noise, dtype=float),
        'reference_trajectory': np.asarray(ref[:, :4], dtype=float),
        'reference_trajectory_world': np.asarray(target_tip_world, dtype=float),
        'target_quaternions': np.asarray(target_quat, dtype=float),
        'realized_ee_trajectory_world': np.asarray(realized_ee_world, dtype=float),
        'realized_ee_quaternions': np.asarray(realized_ee_quat, dtype=float),
        'ik_position_error_world': np.linalg.norm(tracker.s5_to_world([_grasp_state(env, st)[:3] for st in executed]) - target_tip_world, axis=1),
        'sim_dt': float(sim_dt),
        'steps_per_sample': int(steps_per_sample),
        'robot_backend': 'ur5_pybullet_ik_position_control_executed_slider_pose',
        'frames': int(frame_count),
    }


class S4PyBulletPlaybackSession:
    def __init__(self, env, scene: dict[str, Any] | None = None, *, force_gui: bool = True):
        _require_pybullet()
        old_force_gui = bool(getattr(env, "pybullet_force_gui", False))
        setattr(env, "pybullet_force_gui", bool(force_gui))
        self.env = env
        self.tracker = _UR5PoseTracker(
            env,
            scene or {},
            sphere_center_s5=np.zeros(3),
            sphere_radius_s5=float(getattr(env, 'pybullet_marker_radius', 0.002)),
        )
        setattr(env, "pybullet_force_gui", old_force_gui)
        _style_gripper_for_s4(self.tracker)
        _close_gripper_for_visual(self.tracker)
        self.scene_ids = _spawn_s4_scene(env, self.tracker)

    def play(self, reference_traj: np.ndarray, **kwargs) -> dict[str, Any]:
        ref = np.asarray(reference_traj, dtype=float)
        gui = int(kwargs.get("gui", 2))
        return _play_s4_reference(
            self.env,
            self.tracker,
            self.scene_ids,
            ref,
            gui=gui,
            video_path=kwargs.get("video_path", None),
            fps=float(kwargs.get("fps", 15.0)),
            width=int(kwargs.get("width", getattr(self.env, "pybullet_render_width", 1280))),
            height=int(kwargs.get("height", getattr(self.env, "pybullet_render_height", 720))),
            render_frame_stride=int(kwargs.get("render_frame_stride", 1)),
            video_end_hold_seconds=float(kwargs.get("video_end_hold_seconds", 2.0)),
            realtime=bool(kwargs.get("realtime", gui == 2)),
            gui_hold_seconds=float(kwargs.get("gui_hold_seconds", -1.0 if gui == 2 else 0.0)),
            normal_load_trace=kwargs.get("normal_load_trace", None),
            visualize_normal_load=bool(kwargs.get("visualize_normal_load", getattr(self.env, "pybullet_visualize_normal_load", True))),
            true_cutpoints=kwargs.get("true_cutpoints", None),
            feature_overlay=bool(kwargs.get("feature_overlay", True)),
            feature_overlay_title=kwargs.get("feature_overlay_title", None),
            execution_joint_noise_std=float(kwargs.get("execution_joint_noise_std", 0.0)),
            execution_joint_noise_smooth=float(kwargs.get("execution_joint_noise_smooth", 0.90)),
            execution_noise_seed=kwargs.get("execution_noise_seed", None),
        )

    def close(self) -> None:
        self.tracker.close()


def simulate_s4_demo_from_reference(
    env,
    *,
    scene: dict[str, Any] | None,
    reference_traj: np.ndarray,
    true_cutpoints: np.ndarray,
    gui: int = 0,
    video_path: str | Path | None = None,
    fps: float = 15.0,
    width: int | None = None,
    height: int | None = None,
    render_frame_stride: int = 1,
    video_end_hold_seconds: float = 2.0,
    realtime: bool = False,
    gui_hold_seconds: float = 0.0,
    normal_load_trace: np.ndarray | None = None,
    visualize_normal_load: bool = True,
    feature_overlay: bool = True,
    feature_overlay_title: str | None = None,
    execution_joint_noise_std: float = 0.0,
    execution_joint_noise_smooth: float = 0.90,
    execution_noise_seed: int | None = None,
) -> dict[str, Any]:
    _require_pybullet()
    ref = np.asarray(reference_traj, dtype=float)
    if ref.ndim != 2 or ref.shape[1] < 4:
        raise ValueError('reference_traj must have shape (T, 4+) for S4 PyBullet tracking.')
    sim_dt = float(getattr(env, 'pybullet_sim_dt', 1.0 / 120.0))
    configured_steps = getattr(env, 'pybullet_steps_per_sample', None)
    steps_per_sample = max(1, int(round(float(env.dt) / sim_dt))) if configured_steps is None else int(configured_steps)
    old_force_gui = bool(getattr(env, "pybullet_force_gui", False))
    if int(gui) == 2:
        setattr(env, "pybullet_force_gui", True)
    tracker = _UR5PoseTracker(env, scene or {}, sphere_center_s5=np.zeros(3), sphere_radius_s5=float(getattr(env, 'pybullet_marker_radius', 0.002)))
    setattr(env, "pybullet_force_gui", old_force_gui)
    _style_gripper_for_s4(tracker)
    _close_gripper_for_visual(tracker)
    scene_ids = _spawn_s4_scene(env, tracker)
    writer = None
    width = int(getattr(env, 'pybullet_render_width', 1280) if width is None else width)
    height = int(getattr(env, 'pybullet_render_height', 720) if height is None else height)
    save_video = int(gui) == 1 and video_path is not None
    use_gui = int(gui) == 2
    if save_video:
        writer = _FFmpegVideoWriter(out_path=video_path, width=width, height=height, fps=float(fps))
    frame_count = 0
    try:
        q_nominal_cmd, target_tip_world, target_quat = _compute_ik_commands(env, tracker, ref)
        q_noise = _smooth_command_noise(
            q_nominal_cmd.shape,
            std=float(execution_joint_noise_std),
            smooth=float(execution_joint_noise_smooth),
            seed=execution_noise_seed,
        )
        q_cmd = q_nominal_cmd + q_noise
        executed = np.zeros((len(ref), 4), dtype=float)
        realized_ee_world = np.zeros((len(ref), 3), dtype=float)
        realized_ee_quat = np.zeros((len(ref), 4), dtype=float)
        q_meas = np.zeros_like(q_cmd)
        qd_meas = np.zeros_like(q_cmd)
        load_trace = None if normal_load_trace is None else np.asarray(normal_load_trace, dtype=float).reshape(-1)
        load_max = float(np.max(load_trace)) if load_trace is not None and load_trace.size > 0 else float(getattr(env, 'normal_load_min', 1.0))
        load_visual_ids: list[int] = []
        tracker.reset_joint_state(q_cmd[0])
        _close_gripper_for_visual(tracker)
        tracker.command_joint_target(q_cmd[0])
        tracker.step(max(20, steps_per_sample))
        ee_pos, ee_quat = tracker.get_ee_pose()
        executed[0] = _executed_slider_state_from_ee(env, tracker, ee_pos, ee_quat, float(ref[0, 3]))
        realized_ee_world[0] = ee_pos
        realized_ee_quat[0] = ee_quat
        _set_slider_pose(env, tracker, scene_ids.get('slider_parts', scene_ids['slider']), executed[0])
        if bool(visualize_normal_load) and load_trace is not None and len(load_trace) == len(ref):
            load_visual_ids = _draw_normal_load_arrow(
                env,
                tracker,
                executed[0],
                load_value=float(load_trace[0]),
                load_max=load_max,
                old_item_ids=load_visual_ids,
                visual_bodies=scene_ids.get('normal_load_arrow'),
            )
        text_id = None
        if use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=float(getattr(env, 'pybullet_camera_distance', 0.78)),
                cameraYaw=float(getattr(env, 'pybullet_camera_yaw', 78.0)),
                cameraPitch=float(getattr(env, 'pybullet_camera_pitch', -34.0)),
                cameraTargetPosition=list(getattr(env, 'pybullet_camera_target', (0.55, 0.0, 0.54))),
                physicsClientId=tracker.client_id,
            )
        i = 0
        paused = False
        while i < len(ref):
            if use_gui and _space_was_triggered():
                paused = not paused
            if use_gui and paused:
                time.sleep(0.05)
                continue
            _close_gripper_for_visual(tracker)
            tracker.command_joint_target(q_cmd[i])
            tracker.step(steps_per_sample)
            q_i, qd_i = tracker.get_joint_state()
            q_meas[i] = q_i
            qd_meas[i] = qd_i
            ee_pos, ee_quat = tracker.get_ee_pose()
            executed[i] = _executed_slider_state_from_ee(env, tracker, ee_pos, ee_quat, float(ref[i, 3]))
            realized_ee_world[i] = ee_pos
            realized_ee_quat[i] = ee_quat
            _set_slider_pose(env, tracker, scene_ids.get('slider_parts', scene_ids['slider']), executed[i])
            if bool(visualize_normal_load) and load_trace is not None and len(load_trace) == len(ref):
                load_visual_ids = _draw_normal_load_arrow(
                    env,
                    tracker,
                    executed[i],
                    load_value=float(load_trace[i]),
                    load_max=load_max,
                    old_item_ids=load_visual_ids,
                    visual_bodies=scene_ids.get('normal_load_arrow'),
                )
            if writer is not None and i % max(int(render_frame_stride), 1) == 0:
                frame = _camera_frame(env, tracker, width=width, height=height)
                if bool(visualize_normal_load) and load_trace is not None and len(load_trace) == len(ref):
                    frame = _overlay_normal_force_label(frame, env, tracker, executed[i], load_value=float(load_trace[i]), load_max=load_max)
                if bool(feature_overlay):
                    frame = _overlay_constraint_feature_panel(
                        frame,
                        env,
                        executed[: i + 1],
                        current_index=i,
                        total_length=len(ref),
                        true_cutpoints=true_cutpoints,
                        normal_load_trace=load_trace,
                        ylim_trajectory=ref,
                        title=str(feature_overlay_title or "Executed features"),
                    )
                writer.append_data(frame)
                frame_count += 1
            if use_gui and bool(realtime):
                time.sleep(1.0 / max(float(fps), 1e-6))
            i += 1
        if use_gui:
            hold_seconds = float(gui_hold_seconds)
            if hold_seconds < 0.0:
                while True:
                    if _space_was_triggered():
                        break
                    time.sleep(0.1)
            elif hold_seconds > 0.0:
                time.sleep(hold_seconds)
        if writer is not None and float(video_end_hold_seconds) > 0.0 and len(executed) > 0:
            last_idx = len(executed) - 1
            hold_frames = int(round(float(video_end_hold_seconds) * float(fps)))
            for _ in range(max(0, hold_frames)):
                frame = _camera_frame(env, tracker, width=width, height=height)
                if bool(visualize_normal_load) and load_trace is not None and len(load_trace) == len(ref):
                    frame = _overlay_normal_force_label(frame, env, tracker, executed[last_idx], load_value=float(load_trace[last_idx]), load_max=load_max)
                if bool(feature_overlay):
                    frame = _overlay_constraint_feature_panel(
                        frame,
                        env,
                        executed,
                        current_index=last_idx,
                        total_length=len(ref),
                        true_cutpoints=true_cutpoints,
                        normal_load_trace=load_trace,
                        ylim_trajectory=ref,
                        title=str(feature_overlay_title or "Executed features"),
                    )
                writer.append_data(frame)
                frame_count += 1
    finally:
        if writer is not None:
            writer.close()
        _remove_debug_items(tracker, load_visual_ids if 'load_visual_ids' in locals() else [])
        tracker.close()
    return {
        'trajectory': np.asarray(executed, dtype=float),
        'linear_velocity': _finite_difference(executed[:, :3], float(env.dt)),
        'angular_velocity': _finite_difference(executed[:, 3], float(env.dt)),
        'joint_positions': np.asarray(q_meas, dtype=float),
        'joint_velocities': np.asarray(qd_meas, dtype=float),
        'joint_position_commands': np.asarray(q_cmd, dtype=float),
        'joint_position_commands_nominal': np.asarray(q_nominal_cmd, dtype=float),
        'execution_joint_noise': np.asarray(q_noise, dtype=float),
        'true_cutpoints': np.asarray(true_cutpoints, dtype=int),
        'reference_trajectory': np.asarray(ref[:, :4], dtype=float),
        'reference_trajectory_world': np.asarray(target_tip_world, dtype=float),
        'target_quaternions': np.asarray(target_quat, dtype=float),
        'realized_ee_trajectory_world': np.asarray(realized_ee_world, dtype=float),
        'realized_ee_quaternions': np.asarray(realized_ee_quat, dtype=float),
        'ik_position_error_world': np.linalg.norm(tracker.s5_to_world([_grasp_state(env, st)[:3] for st in executed]) - target_tip_world, axis=1),
        'sim_dt': float(sim_dt),
        'steps_per_sample': int(steps_per_sample),
        'robot_backend': 'ur5_pybullet_ik_position_control_executed_slider_pose',
        'frames': int(frame_count),
    }
