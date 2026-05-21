from __future__ import annotations

import numpy as np

from .base import TaskBundle
from .rendering import render_planar_episode



# Integrated PyBullet backend helpers for S4SlideInsert.
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .rendering import _FFmpegVideoWriter, _save_rgb_frame, _space_was_triggered
from .S5SphereInspect import _UR5PoseTracker, _quat_from_matrix, _require_pybullet

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


_S4_STAGE_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
_S4_STAGE_SEMANTICS = ("approach", "contact/alignment", "sliding", "insertion")
_S4_STAGE_LABEL_CENTER_FRAC = (0.39, 0.35)


def _rgba255_from_hex(color: str, alpha: float) -> tuple[int, int, int, int]:
    text = str(color).strip().lstrip("#")
    if len(text) != 6:
        return (31, 41, 55, int(np.clip(float(alpha), 0.0, 1.0) * 255.0))
    r = int(text[0:2], 16)
    g = int(text[2:4], 16)
    b = int(text[4:6], 16)
    a = int(np.clip(float(alpha), 0.0, 1.0) * 255.0)
    return (r, g, b, a)


def _stage_index_at_frame(cutpoints: np.ndarray | None, frame_idx: int, length: int) -> int:
    spans = _stage_spans(cutpoints, int(length))
    if not spans:
        return 0
    idx = int(np.clip(int(frame_idx), 0, max(int(length) - 1, 0)))
    for stage_idx, (start, end) in enumerate(spans):
        if int(start) <= idx <= int(end):
            return int(stage_idx)
    return int(len(spans) - 1)


def _overlay_s4_stage_label(
    frame: np.ndarray,
    *,
    current_index: int,
    total_length: int,
    true_cutpoints: np.ndarray | None,
) -> np.ndarray:
    if Image is None or ImageDraw is None:
        return frame
    height, width = int(frame.shape[0]), int(frame.shape[1])
    stage_idx = _stage_index_at_frame(true_cutpoints, int(current_index), int(total_length))
    semantic = _S4_STAGE_SEMANTICS[min(stage_idx, len(_S4_STAGE_SEMANTICS) - 1)]
    title = f"Stage {stage_idx + 1}"
    subtitle = str(semantic)
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8), mode="RGB").convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", int(np.clip(round(height * 0.032), 20, 32)))
        font_body = ImageFont.truetype("DejaVuSans.ttf", int(np.clip(round(height * 0.023), 15, 23)))
    except Exception:
        font_title = ImageFont.load_default() if ImageFont is not None else None
        font_body = font_title
    try:
        title_box = draw.textbbox((0, 0), title, font=font_title)
        body_box = draw.textbbox((0, 0), subtitle, font=font_body)
        title_w = int(title_box[2] - title_box[0])
        title_h = int(title_box[3] - title_box[1])
        body_w = int(body_box[2] - body_box[0])
        body_h = int(body_box[3] - body_box[1])
    except Exception:
        title_w, title_h = draw.textsize(title, font=font_title)
        body_w, body_h = draw.textsize(subtitle, font=font_body)
    pad_x = int(max(14, round(width * 0.012)))
    pad_y = int(max(10, round(height * 0.012)))
    gap = int(max(4, round(height * 0.006)))
    box_w = max(title_w, body_w) + 2 * pad_x
    box_h = title_h + body_h + gap + 2 * pad_y
    center_x = float(width) * float(_S4_STAGE_LABEL_CENTER_FRAC[0])
    center_y = float(height) * float(_S4_STAGE_LABEL_CENTER_FRAC[1])
    x0 = int(np.clip(round(center_x - 0.5 * box_w), 8, max(8, width - box_w - 8)))
    y0 = int(np.clip(round(center_y - 0.5 * box_h), 8, max(8, height - box_h - 8)))
    x1 = min(width - 8, x0 + box_w)
    y1 = min(height - 8, y0 + box_h)
    color = _rgba255_from_hex(_S4_STAGE_COLORS[stage_idx % len(_S4_STAGE_COLORS)], 0.95)
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=max(6, int(round((y1 - y0) * 0.12))),
        fill=(255, 255, 255, 218),
        outline=color,
        width=2,
    )
    accent_w = int(max(5, round(width * 0.004)))
    draw.rounded_rectangle((x0, y0, x0 + accent_w, y1), radius=4, fill=color)
    tx = x0 + pad_x + accent_w
    draw.text((tx, y0 + pad_y - 2), title, fill=(15, 23, 42, 255), font=font_title)
    draw.text((tx, y0 + pad_y + title_h + gap - 1), subtitle, fill=(51, 65, 85, 255), font=font_body)
    return np.asarray(Image.alpha_composite(image, overlay).convert("RGB"), dtype=np.uint8)


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


def _frame_index_set(save_frame_indices, length: int) -> set[int]:
    return {
        int(v)
        for v in ([] if save_frame_indices is None else save_frame_indices)
        if 0 <= int(v) < int(length)
    }


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


def _spawn_ground_grid(
    client_id: int,
    *,
    center_xy=(0.50, 0.0),
    half_extents_xy=(1.15, 0.85),
    spacing: float = 0.10,
    z: float = 0.0,
) -> list[int]:
    floor_id = int(
        p.loadURDF(
            "plane.urdf",
            basePosition=[0.0, 0.0, float(z) - 0.02],
            useFixedBase=True,
            physicsClientId=client_id,
        )
    )
    p.changeVisualShape(floor_id, -1, rgbaColor=[0.96, 0.97, 0.99, 1.0], physicsClientId=client_id)
    return [floor_id]


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
    grid_ids = _spawn_ground_grid(
        client_id,
        center_xy=(float(origin[0]) * 0.45, float(origin[1])),
        half_extents_xy=(1.18, 0.86),
        spacing=0.10,
        z=0.0,
    )
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
        'ground_grid': grid_ids,
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
    stage_header_h = 19 if len(spans) > 1 else 0
    header_h = 66 + 15 * max(0, len(title_lines) - 1) + stage_header_h
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
    if stage_header_h > 0:
        stage_y0 = y0 + header_h - stage_header_h + 2
        for stage_idx, (start, end) in enumerate(spans):
            xa = plot_x0 + float(start) / float(total_den) * float(plot_w)
            xb = plot_x0 + float(end) / float(total_den) * float(plot_w)
            color = _S4_STAGE_COLORS[stage_idx % len(_S4_STAGE_COLORS)]
            draw.rectangle(
                (xa, stage_y0, xb, stage_y0 + 13),
                fill=_rgba255_from_hex(color, 0.18),
                outline=_rgba255_from_hex(color, 0.90),
                width=1,
            )
            label = f"Stage {stage_idx + 1}" if xb - xa >= 54 else f"S{stage_idx + 1}"
            try:
                bbox = draw.textbbox((0, 0), label, font=font_bold)
                tw = float(bbox[2] - bbox[0])
                th = float(bbox[3] - bbox[1])
            except Exception:
                tw, th = 34.0, 9.0
            tx = min(max(xa + 2.0, 0.5 * (xa + xb) - 0.5 * tw), max(xa + 2.0, xb - tw - 2.0))
            draw.text((tx, stage_y0 + max(0.0, 0.5 * (13.0 - th)) - 1.0), label, fill=(18, 24, 38, 245), font=font_bold)

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
    save_frame_indices=None,
    save_frame_dir: str | Path | None = None,
    save_frame_prefix: str = "s4_frame",
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
    frame_indices_to_save = _frame_index_set(save_frame_indices, len(ref))
    frame_save_dir = None if save_frame_dir is None else Path(save_frame_dir)
    saved_frame_paths: list[str] = []
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
            write_frame = writer is not None and i % max(int(render_frame_stride), 1) == 0
            save_still = int(i) in frame_indices_to_save and frame_save_dir is not None
            if write_frame or save_still:
                frame = _camera_frame(env, tracker, width=width, height=height)
                if bool(visualize_normal_load) and load_trace is not None and len(load_trace) == len(ref):
                    frame = _overlay_normal_force_label(frame, env, tracker, executed[i], load_value=float(load_trace[i]), load_max=load_max)
                frame = _overlay_s4_stage_label(
                    frame,
                    current_index=i,
                    total_length=len(ref),
                    true_cutpoints=true_cutpoints,
                )
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
                if save_still:
                    frame_path = _save_rgb_frame(
                        frame,
                        frame_save_dir / f"{str(save_frame_prefix)}_frame_{int(i):06d}.png",
                        crop_aspect=4.0 / 3.0,
                    )
                    saved_frame_paths.append(str(frame_path.resolve()))
                if write_frame:
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
                frame = _overlay_s4_stage_label(
                    frame,
                    current_index=last_idx,
                    total_length=len(ref),
                    true_cutpoints=true_cutpoints,
                )
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
        'saved_frames': saved_frame_paths,
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
            save_frame_indices=kwargs.get("save_frame_indices", None),
            save_frame_dir=kwargs.get("save_frame_dir", None),
            save_frame_prefix=str(kwargs.get("save_frame_prefix", "s4_frame")),
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
    save_frame_indices=None,
    save_frame_dir: str | Path | None = None,
    save_frame_prefix: str = "s4_frame",
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
    frame_indices_to_save = _frame_index_set(save_frame_indices, len(ref))
    frame_save_dir = None if save_frame_dir is None else Path(save_frame_dir)
    saved_frame_paths: list[str] = []
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
            write_frame = writer is not None and i % max(int(render_frame_stride), 1) == 0
            save_still = int(i) in frame_indices_to_save and frame_save_dir is not None
            if write_frame or save_still:
                frame = _camera_frame(env, tracker, width=width, height=height)
                if bool(visualize_normal_load) and load_trace is not None and len(load_trace) == len(ref):
                    frame = _overlay_normal_force_label(frame, env, tracker, executed[i], load_value=float(load_trace[i]), load_max=load_max)
                frame = _overlay_s4_stage_label(
                    frame,
                    current_index=i,
                    total_length=len(ref),
                    true_cutpoints=true_cutpoints,
                )
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
                if save_still:
                    frame_path = _save_rgb_frame(
                        frame,
                        frame_save_dir / f"{str(save_frame_prefix)}_frame_{int(i):06d}.png",
                        crop_aspect=4.0 / 3.0,
                    )
                    saved_frame_paths.append(str(frame_path.resolve()))
                if write_frame:
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
                frame = _overlay_s4_stage_label(
                    frame,
                    current_index=last_idx,
                    total_length=len(ref),
                    true_cutpoints=true_cutpoints,
                )
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
            'saved_frames': saved_frame_paths,
        }



class _S4SlideInsertBase:
    def __init__(
        self,
        seg_lengths=(20, 8, 38, 12),
        start=(-1.4, 0.35),
        start_jitter=(0.10, 0.05),
        stage1_end=(-0.25, 0.014),
        stage2_end=(-0.20, 0.0),
        stage3_end=(0.70, 0.0),
        stage4_end=(1.00, 0.0),
        stage_end_jitter=((0.032, 0.012), (0.024, 0.006), (0.038, 0.006), (0.014, 0.003)),
        stage2_end_x_range=(-0.70, 0.0),
        stage2_end_z_range=(-0.010, 0.010),
        stage2_theta_end_range=(-0.030, 0.085),
        slot_x=1.0,
        slot_theta=0.0,
        theta_start=0.34,
        theta_stage1_end=0.12,
        theta_stage2_end=0.03,
        theta_stage3_end=0.01,
        theta_stage4_end=0.0,
        theta_end_jitter=(0.055, 0.032, 0.018, 0.010),
        theta_start_jitter=0.05,
        v1_target=0.060,
        v2_target=0.007,
        v3_target=0.045,
        v4_target=0.018,
        f_contact_min=0.40,
        f_slide_min=0.72,
        f_insert_min=1.00,
        orient_err_max_stage3=0.06,
        orient_err_max_stage4=0.04,
        transition_half_window: int = 1,
        noise_pos: float = 0.003,
        noise_misc: float = 0.02,
        seg_length_jitter=(5, 3, 6, 4),
        seg_length_scale_range=(0.84, 1.18),
        dt: float = 0.7,
    ):
        self.seg_lengths = tuple(int(x) for x in seg_lengths)
        self.start = np.asarray(start, dtype=float)
        self.start_jitter = np.asarray(start_jitter, dtype=float)
        self.stage1_end = np.asarray(stage1_end, dtype=float)
        self.stage2_end = np.asarray(stage2_end, dtype=float)
        self.stage3_end = np.asarray(stage3_end, dtype=float)
        self.stage4_end = np.asarray(stage4_end, dtype=float)
        self.stage_end_jitter = tuple(np.asarray(x, dtype=float) for x in stage_end_jitter)
        self.stage2_end_x_range = tuple(float(x) for x in stage2_end_x_range)
        self.stage2_end_z_range = tuple(float(x) for x in stage2_end_z_range)
        self.stage2_theta_end_range = tuple(float(x) for x in stage2_theta_end_range)
        self.slot_x = float(slot_x)
        self.slot_theta = float(slot_theta)
        self.theta_start = float(theta_start)
        self.theta_stage1_end = float(theta_stage1_end)
        self.theta_stage2_end = float(theta_stage2_end)
        self.theta_stage3_end = float(theta_stage3_end)
        self.theta_stage4_end = float(theta_stage4_end)
        self.theta_end_jitter = tuple(float(x) for x in theta_end_jitter)
        self.theta_start_jitter = float(theta_start_jitter)
        self.v1_target = float(v1_target)
        self.v2_target = float(v2_target)
        self.v3_target = float(v3_target)
        self.v4_target = float(v4_target)
        self.f_contact_min = float(f_contact_min)
        self.f_slide_min = float(f_slide_min)
        self.f_insert_min = float(f_insert_min)
        self.orient_err_max_stage3 = float(orient_err_max_stage3)
        self.orient_err_max_stage4 = float(orient_err_max_stage4)
        self.transition_half_window = int(transition_half_window)
        self.noise_pos = float(noise_pos)
        self.noise_misc = float(noise_misc)
        self.seg_length_jitter = tuple(int(x) for x in seg_length_jitter)
        self.seg_length_scale_range = tuple(float(x) for x in seg_length_scale_range)
        self.dt = float(dt)
        self.eval_tag = "S4SlideInsertBase"
        self.n_segments = 4
        self._cached_force_traces = {}
        self._cached_speed_traces = {}
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self.feature_schema = self.get_feature_schema()
        self.subgoal = np.array([self.stage2_end[0], self.stage2_end[1], self.theta_stage2_end], dtype=float)
        self.goal = np.array([self.stage4_end[0], self.stage4_end[1], self.theta_stage4_end], dtype=float)
        self.demo_subgoals = None
        self.demo_goals = None
        self.demo_stage_lengths = None

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "surf_dist", "description": "Absolute distance to the contact surface z=0"},
            {"id": 1, "name": "force", "description": "Contact force proxy"},
            {"id": 2, "name": "orient_err", "description": "Absolute angle error between object and slot"},
            {"id": 3, "name": "speed", "description": "Planar speed magnitude"},
            {"id": 4, "name": "noise", "description": "Auxiliary irrelevant feature"},
            {"id": 5, "name": "start_dist", "description": "Distance to the demo start pose in the x-z plane"},
            {"id": 6, "name": "insertion_err", "description": "Remaining x-direction distance to the slot target"},
        ]

    def get_true_constraints(self):
        return {
            "surface_target": 0.0,
            "v2_target": float(self.v2_target),
            "v3_target": float(self.v3_target),
            "v4_target": float(self.v4_target),
            "f_contact_min": float(self.f_contact_min),
            "f_slide_min": float(self.f_slide_min),
            "f_insert_min": float(self.f_insert_min),
            "orient_err_max_stage3": float(self.orient_err_max_stage3),
            "orient_err_max_stage4": float(self.orient_err_max_stage4),
        }

    def get_constraint_specs(self):
        return [
            {"feature_name": "surf_dist", "stage": 1, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "speed", "stage": 1, "semantics": "target_value", "oracle_key": "v2_target"},
            {"feature_name": "force", "stage": 1, "semantics": "lower_bound", "oracle_key": "f_contact_min"},
            {"feature_name": "surf_dist", "stage": 2, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "speed", "stage": 2, "semantics": "target_value", "oracle_key": "v3_target"},
            {"feature_name": "force", "stage": 2, "semantics": "lower_bound", "oracle_key": "f_slide_min"},
            {
                "feature_name": "orient_err",
                "stage": 2,
                "semantics": "upper_bound",
                "oracle_key": "orient_err_max_stage3",
            },
            {"feature_name": "surf_dist", "stage": 3, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "speed", "stage": 3, "semantics": "target_value", "oracle_key": "v4_target"},
            {"feature_name": "force", "stage": 3, "semantics": "lower_bound", "oracle_key": "f_insert_min"},
            {
                "feature_name": "orient_err",
                "stage": 3,
                "semantics": "upper_bound",
                "oracle_key": "orient_err_max_stage4",
            },
        ]

    def get_observation_spec(self):
        return {
            "feature_schema": self.get_feature_schema(),
            "noise_model": {
                "position_noise_std": float(self.noise_pos),
                "misc_noise_std": float(self.noise_misc),
                "force_trace_source": "cached_or_state_estimated",
                "speed_trace_source": "cached_or_finite_difference",
            },
        }

    def get_render_camera_presets(self):
        return {
            "default_planar": {
                "projection": "orthographic_like_2d",
                "xlabel": "x",
                "ylabel": "z",
                "equal_aspect": False,
            }
        }

    def get_asset_handles(self):
        return {
            "surface": {"type": "line", "axis": "x"},
            "slot": {"type": "slot_marker"},
            "object": {"type": "planar_slider"},
        }

    def sample_scene(self, seed=None, rng=None):
        return {
            "task_name": "S4SlideInsertBase",
            "geometry": {
                "start": self.start.tolist(),
                "stage1_end": self.stage1_end.tolist(),
                "stage2_end": self.stage2_end.tolist(),
                "stage3_end": self.stage3_end.tolist(),
                "stage4_end": self.stage4_end.tolist(),
                "slot_x": float(self.slot_x),
                "slot_theta": float(self.slot_theta),
                "surface_z": 0.0,
            },
            "task": {
                "seg_lengths": list(self.seg_lengths),
                "seg_length_jitter": list(self.seg_length_jitter),
                "seg_length_scale_range": list(self.seg_length_scale_range),
            },
        }

    def rollout_demo(self, scene, seed=None, rng=None, **kwargs):
        local_seed = int(seed) if seed is not None else int((scene or {}).get("rollout_seed", 0))
        pos, theta, labels, force, speed = self.generate_demo(seed=local_seed)
        traj = np.c_[pos, theta]
        cutpoints = np.where(np.diff(np.asarray(labels, dtype=int)) != 0)[0].astype(int)
        return {
            "trajectory": np.asarray(traj, dtype=float),
            "true_cutpoints": np.asarray(cutpoints, dtype=int),
            "true_labels": np.asarray(labels, dtype=int),
            "force_trace": np.asarray(force, dtype=float),
            "speed_trace": np.asarray(speed, dtype=float),
        }

    def compute_observation(self, latent_rollout, scene):
        traj = np.asarray(latent_rollout["trajectory"], dtype=float)
        force = np.asarray(latent_rollout.get("force_trace", []), dtype=float)
        speed = np.asarray(latent_rollout.get("speed_trace", []), dtype=float)
        if force.size > 0:
            self.register_force_trace(traj, force)
        if speed.size > 0:
            self.register_speed_trace(traj, speed)
        features = np.asarray(self.compute_all_features_matrix(traj), dtype=float)
        return {
            "trajectory": traj,
            "features": features,
            "true_cutpoints": np.asarray(latent_rollout.get("true_cutpoints", []), dtype=int),
            "true_labels": np.asarray(latent_rollout.get("true_labels", []), dtype=int),
            "feature_schema": self.get_feature_schema(),
            "observation_spec": self.get_observation_spec(),
            "scene": dict(scene or {}),
        }

    def render_episode(self, scene, trajectory, output_path, **kwargs):
        geometry = dict((scene or {}).get("geometry", {}))
        cutpoints = kwargs.get("cutpoints")
        markers = [
            {"point": [geometry.get("slot_x", self.slot_x), geometry.get("surface_z", 0.0)], "color": "#16A34A", "marker": "s", "size": 34},
            {"point": geometry.get("stage2_end", self.stage2_end.tolist()), "color": "#F97316", "marker": "^", "size": 30},
        ]
        reference_lines = [
            {
                "point": [0.0, geometry.get("surface_z", 0.0)],
                "direction": [1.0, 0.0],
                "color": "#64748B",
                "linestyle": "-",
                "linewidth": 1.0,
                "alpha": 0.8,
            }
        ]
        return render_planar_episode(
            trajectory=np.asarray(trajectory, dtype=float)[:, :2],
            output_path=output_path,
            cutpoints=cutpoints,
            title=kwargs.get("title", "S4SlideInsert episode"),
            obstacles=None,
            reference_lines=reference_lines,
            markers=markers,
            xlabel="x",
            ylabel="z",
            equal_aspect=False,
        )

    def _piecewise_segment(self, start, end, length, endpoint=False):
        x = np.linspace(float(start[0]), float(end[0]), int(length), endpoint=endpoint)
        z = np.linspace(float(start[1]), float(end[1]), int(length), endpoint=endpoint)
        return np.c_[x, z]

    @staticmethod
    def _smoothstep(u):
        u = np.asarray(u, dtype=float)
        return u * u * (3.0 - 2.0 * u)

    def _smooth_segment(self, start, end, length, endpoint=False):
        u = np.linspace(0.0, 1.0, int(length), endpoint=endpoint)
        s = self._smoothstep(u)
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        return start[None, :] + s[:, None] * (end - start)[None, :]

    @staticmethod
    def _path_length(path: np.ndarray) -> float:
        pts = np.asarray(path, dtype=float)
        if len(pts) <= 1:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    @staticmethod
    def _resample_fixed_count(path: np.ndarray, num_points: int) -> np.ndarray:
        pts = np.asarray(path, dtype=float)
        n = int(num_points)
        if len(pts) <= 1 or n <= 1:
            return pts.copy()
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= 1e-12:
            out = np.repeat(pts[:1], n, axis=0)
            out[0] = pts[0]
            out[-1] = pts[-1]
            return out
        targets = np.linspace(0.0, total, n)
        out = np.empty((n, pts.shape[1]), dtype=float)
        out[0] = pts[0]
        out[-1] = pts[-1]
        j = 0
        for i, target in enumerate(targets[1:-1], start=1):
            while j + 1 < len(s) and s[j + 1] < target:
                j += 1
            frac = (target - s[j]) / max(s[j + 1] - s[j], 1e-12)
            out[i] = (1.0 - frac) * pts[j] + frac * pts[j + 1]
        return out

    def _timewarp_path(
        self,
        path: np.ndarray,
        strength: float,
        rng: np.random.RandomState,
        cycles: float = 2.0,
    ) -> np.ndarray:
        pts = np.asarray(path, dtype=float)
        if len(pts) <= 2 or float(strength) <= 0.0:
            return pts.copy()
        s = np.linspace(0.0, 1.0, len(pts))
        phase = float(rng.uniform(-np.pi, np.pi))
        envelope = np.sin(np.pi * s) ** 1.1
        weights = 1.0 + float(strength) * envelope * np.sin(float(cycles) * np.pi * s + phase)
        weights = np.clip(weights, 0.35, None)
        targets = np.cumsum(weights)
        targets = (targets - targets[0]) / max(targets[-1] - targets[0], 1e-12)
        out = np.empty_like(pts)
        out[0] = pts[0]
        out[-1] = pts[-1]
        for d in range(pts.shape[1]):
            out[:, d] = np.interp(targets, s, pts[:, d])
        out[0] = pts[0]
        out[-1] = pts[-1]
        return out

    def _timewarp_decelerating_path(
        self,
        path: np.ndarray,
        strength: float,
        rng: np.random.RandomState,
        floor: float = 0.55,
    ) -> np.ndarray:
        pts = np.asarray(path, dtype=float)
        if len(pts) <= 2 or float(strength) <= 0.0:
            return pts.copy()
        s = np.linspace(0.0, 1.0, len(pts))
        exponent = float(rng.uniform(1.2, 1.8))
        front_bias = 1.0 - s**exponent
        ripple_phase = float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi))
        ripple = 0.10 * np.sin(2.2 * np.pi * s + ripple_phase) * np.sin(np.pi * s) ** 1.3
        weights = 1.0 + float(strength) * front_bias + ripple
        weights = np.clip(weights, float(floor), None)
        targets = np.cumsum(weights)
        targets = (targets - targets[0]) / max(targets[-1] - targets[0], 1e-12)
        out = np.empty_like(pts)
        for d in range(pts.shape[1]):
            out[:, d] = np.interp(targets, s, pts[:, d])
        out[0] = pts[0]
        out[-1] = pts[-1]
        return out

    def _make_target_speed_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_points: int,
        target_speed: float,
        rng: np.random.RandomState,
        max_amp: float,
        cycles: float = 1.0,
        vertical_bias: float = 0.0,
    ) -> np.ndarray:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        n = int(num_points)
        desired_length = max(float(target_speed) * self.dt * max(n - 1, 1), float(np.linalg.norm(end - start)))
        u_hr = np.linspace(0.0, 1.0, 256)
        direction = end - start
        dist = float(np.linalg.norm(direction))
        if dist <= 1e-12:
            direction_unit = np.array([1.0, 0.0], dtype=float)
        else:
            direction_unit = direction / dist
        normal = np.array([-direction_unit[1], direction_unit[0]], dtype=float)
        phase = float(rng.uniform(-0.5 * np.pi, 0.5 * np.pi))
        sign = -1.0 if rng.rand() < 0.5 else 1.0
        envelope = np.sin(np.pi * u_hr)
        waveform = envelope * np.sin(float(cycles) * np.pi * u_hr + phase)

        def build_path(amp: float) -> np.ndarray:
            base = start[None, :] + u_hr[:, None] * (end - start)[None, :]
            offset = sign * float(amp) * waveform
            path = base + offset[:, None] * normal[None, :]
            if vertical_bias != 0.0:
                path[:, 1] += float(vertical_bias) * envelope**2
            path[0] = start
            path[-1] = end
            return path

        if self._path_length(build_path(0.0)) >= desired_length - 1e-6:
            return self._resample_fixed_count(build_path(0.0), n)

        lo = 0.0
        hi = max(float(max_amp), 1e-4)
        for _ in range(20):
            if self._path_length(build_path(hi)) >= desired_length:
                break
            hi *= 1.5
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            if self._path_length(build_path(mid)) < desired_length:
                lo = mid
            else:
                hi = mid
        return self._resample_fixed_count(build_path(hi), n)

    def _make_surface_search_segment(
        self,
        start: np.ndarray,
        end: np.ndarray,
        num_points: int,
        target_speed: float,
        rng: np.random.RandomState,
        max_x_amp: float,
        z_amp: float,
        cycles: float = 2.5,
    ) -> np.ndarray:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        n = int(num_points)
        desired_length = max(float(target_speed) * self.dt * max(n - 1, 1), float(np.linalg.norm(end - start)))
        u_hr = np.linspace(0.0, 1.0, 512)
        phase = float(rng.uniform(-0.35 * np.pi, 0.35 * np.pi))
        envelope = np.sin(np.pi * u_hr) ** 1.2
        speed_wave = np.sin(float(cycles) * np.pi * u_hr + phase)
        z_wave = np.sin((float(cycles) + 0.75) * np.pi * u_hr + 0.5 * phase)
        x_weights = np.clip(1.0 + 0.55 * envelope * speed_wave, 0.25, None)
        x_cum = np.cumsum(x_weights)
        x_progress = (x_cum - x_cum[0]) / max(x_cum[-1] - x_cum[0], 1e-12)

        def build_path(amp_z: float) -> np.ndarray:
            x = start[0] + x_progress * (end[0] - start[0])
            z = start[1] + u_hr * (end[1] - start[1]) + float(z_amp) * envelope * z_wave
            z += float(amp_z) * envelope * np.sin((float(cycles) + 1.5) * np.pi * u_hr + phase)
            path = np.c_[x, z]
            path[0] = start
            path[-1] = end
            return path

        if self._path_length(build_path(0.0)) >= desired_length - 1e-6:
            return self._resample_fixed_count(build_path(0.0), n)

        lo = 0.0
        hi = max(float(max_x_amp) * 0.25, 1e-4)
        for _ in range(20):
            if self._path_length(build_path(hi)) >= desired_length:
                break
            hi *= 1.5
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self._path_length(build_path(mid)) < desired_length:
                lo = mid
            else:
                hi = mid
        return self._resample_fixed_count(build_path(hi), n)

    @staticmethod
    def _smooth_noise(rng: np.random.RandomState, length: int, scale: float, kernel_size: int = 7) -> np.ndarray:
        raw = rng.randn(int(length)) * float(scale)
        kernel = np.ones(int(kernel_size), dtype=float) / float(kernel_size)
        return np.convolve(raw, kernel, mode="same")

    @staticmethod
    def _sample_margin_excess(
        rng: np.random.RandomState,
        length: int,
        scale: float,
        max_extra: float,
        near_boundary_prob: float = 0.82,
    ) -> np.ndarray:
        # Most samples stay close to the lower boundary, with a light exponential tail.
        base = rng.exponential(scale=float(scale), size=int(length))
        tail_mask = rng.rand(int(length)) > float(near_boundary_prob)
        if np.any(tail_mask):
            base[tail_mask] += rng.exponential(scale=1.8 * float(scale), size=int(tail_mask.sum()))
        return np.clip(base, 0.0, float(max_extra))

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
    def _smooth_positive_trace(values: np.ndarray, kernel_size: int = 7) -> np.ndarray:
        vals = np.clip(np.asarray(values, dtype=float), 0.0, None)
        return _S4SlideInsertBase._smooth_trace(vals, kernel_size=kernel_size)

    def _make_stage_force_margin_profile(
        self,
        stage_idx: int,
        z: np.ndarray,
        speed: np.ndarray,
        tangential_speed: np.ndarray,
        dz: np.ndarray,
        orient_err: np.ndarray,
        contact_gate: np.ndarray,
        slide_progress: np.ndarray,
        insert_progress: np.ndarray,
        rng: np.random.RandomState | None,
        latents: dict | None,
    ) -> np.ndarray:
        n = int(len(z))
        if n <= 0:
            return np.zeros(0, dtype=float)

        u = np.linspace(0.0, 1.0, n, endpoint=True)
        if int(stage_idx) == 1:
            cycles = 2.40
            offset = 0.028
            amplitude = 0.070
            margin_cap = 0.115
        elif int(stage_idx) == 2:
            cycles = 3.20
            offset = 0.034
            amplitude = 0.082
            margin_cap = 0.135
        else:
            cycles = 2.80
            offset = 0.032
            amplitude = 0.105
            margin_cap = 0.165

        amp_scale = 1.0 if latents is None else float(latents.get("force_excess_scale", 1.0))
        amp_scale = float(np.clip(amp_scale, 0.9, 1.1))
        base_wave = np.sin(2.0 * np.pi * float(cycles) * u - 0.5 * np.pi)
        half_wave = np.maximum(base_wave, 0.0)
        margin = float(amplitude) * amp_scale * half_wave - float(offset)
        margin = self._smooth_trace(margin, kernel_size=3)
        margin_floor = -float(offset) - 0.01
        return np.clip(margin, float(margin_floor), float(margin_cap))

    @staticmethod
    def _sample_sparse_margin_excess(
        rng: np.random.RandomState,
        length: int,
        base_scale: float,
        base_cap: float,
        burst_scale: float,
        max_extra: float,
        max_bursts: int = 2,
        base_activation_prob: float = 0.16,
    ) -> np.ndarray:
        n = int(length)
        if n <= 0:
            return np.zeros(0, dtype=float)
        base = np.zeros(n, dtype=float)
        base_mask = rng.rand(n) < float(base_activation_prob)
        if np.any(base_mask):
            base[base_mask] = np.clip(
                rng.exponential(scale=float(base_scale), size=int(base_mask.sum())),
                0.0,
                float(base_cap),
            )
        excess = base
        num_bursts = int(rng.randint(1, max(int(max_bursts), 1) + 1))
        for _ in range(num_bursts):
            center = int(rng.randint(0, n))
            half_width = int(rng.randint(1, 3))
            amp = min(float(rng.exponential(scale=float(burst_scale))), 0.6 * float(max_extra))
            left = max(0, center - half_width)
            right = min(n, center + half_width + 1)
            window = np.arange(left, right, dtype=float)
            envelope = 1.0 - np.abs(window - float(center)) / float(half_width + 1)
            excess[left:right] += amp * np.clip(envelope, 0.0, None)
        return np.clip(excess, 0.0, float(max_extra))

    @staticmethod
    def _blend_segment_boundary(values: np.ndarray, boundary: int, half_window: int = 2) -> np.ndarray:
        out = np.asarray(values, dtype=float).copy()
        left = max(0, int(boundary) - int(half_window))
        right = min(len(out) - 1, int(boundary) + int(half_window) + 1)
        if left < 1 or right >= len(out) - 1 or right - left < 2:
            return out

        p0 = out[left].copy()
        p1 = out[right].copy()
        span = float(right - left)
        # Match incoming and outgoing finite-difference slopes so the transition
        # smooths the kink without forcing a dip/spike at the boundary.
        m0 = (out[left] - out[left - 1]) * span
        m1 = (out[right + 1] - out[right]) * span

        u = np.linspace(0.0, 1.0, right - left + 1)
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        out[left:right + 1] = (
            h00[:, None] * p0
            + h10[:, None] * m0
            + h01[:, None] * p1
            + h11[:, None] * m1
        )
        return out

    def _sample_demo_latents(self, rng: np.random.RandomState):
        style = float(rng.uniform(0.85, 1.18))
        return {
            "style": style,
            "phase": float(rng.uniform(0.0, 2.0 * np.pi)),
            "force_excess_scale": float(rng.uniform(0.95, 1.12)),
            "force_bias": float(rng.uniform(-0.004, 0.008)),
            "precontact_force_mean": float(rng.uniform(0.0065, 0.0095)),
            "precontact_force_sigma": float(rng.uniform(0.0010, 0.0018)),
            "contact_force_coupling": float(rng.uniform(0.006, 0.016)),
            "slide_force_coupling": float(rng.uniform(0.010, 0.024)),
            "insert_force_coupling": float(rng.uniform(0.012, 0.028)),
            "micro_wobble": float(rng.uniform(0.004, 0.012)),
            "surface_wobble": float(rng.uniform(0.0015, 0.005)),
            "theta_wobble": float(rng.uniform(0.004, 0.014)),
        }

    def _sample_segment_lengths(self, rng: np.random.RandomState):
        global_scale = float(rng.uniform(*self.seg_length_scale_range))
        min_lengths = [max(6, int(np.floor(0.65 * base))) for base in self.seg_lengths]
        lengths = []
        for base, jitter, min_len in zip(self.seg_lengths, self.seg_length_jitter, min_lengths):
            local_scale = global_scale * float(rng.uniform(0.90, 1.12))
            length = int(round(float(base) * local_scale)) + int(rng.randint(-int(jitter), int(jitter) + 1))
            lengths.append(max(int(min_len), length))
        return tuple(int(x) for x in lengths)

    def _compute_force_signal(
        self,
        pos: np.ndarray,
        theta: np.ndarray,
        stage3_end_x: float,
        labels: np.ndarray,
        rng: np.random.RandomState,
        latents: dict,
    ) -> np.ndarray:
        pos = np.asarray(pos, dtype=float)
        theta = np.asarray(theta, dtype=float)
        labels = np.asarray(labels, dtype=int)
        T = len(pos)
        speed = np.zeros(T, dtype=float)
        tangential_speed = np.zeros(T, dtype=float)
        dz = np.zeros(T, dtype=float)
        if T > 1:
            delta = np.diff(pos, axis=0) / self.dt
            speed[1:] = np.linalg.norm(delta, axis=1)
            speed[0] = speed[1]
            tangential_speed[1:] = np.abs(delta[:, 0])
            tangential_speed[0] = tangential_speed[1]
            dz[1:] = np.abs(delta[:, 1])
            dz[0] = dz[1]
        orient_err = np.abs(self._wrap_to_pi(theta - self.slot_theta))
        z = pos[:, 1]
        x = pos[:, 0]
        contact_gate = 1.0 / (1.0 + np.exp((z - 0.012) / 0.006))
        slide_progress = 1.0 / (1.0 + np.exp(-(x - self.stage2_end[0]) / 0.055))
        insert_progress = 1.0 / (1.0 + np.exp(-(x - float(stage3_end_x)) / 0.045))

        stage_lower_bounds = np.take(
            np.array([0.0, self.f_contact_min, self.f_slide_min, self.f_insert_min], dtype=float),
            labels,
        )
        force = stage_lower_bounds.copy()
        precontact_mask = labels == 0
        if np.any(precontact_mask):
            n0 = int(precontact_mask.sum())
            precontact_force = (
                latents["precontact_force_mean"]
                + self._smooth_noise(rng, n0, latents["precontact_force_sigma"], kernel_size=7)
            )
            precontact_force += 0.0007 * np.sin(np.linspace(0.0, 2.0 * np.pi, n0) + latents["phase"])
            force[precontact_mask] = np.clip(precontact_force, 0.0, 0.02)

        for stage_idx in (1, 2, 3):
            mask = labels == stage_idx
            if not np.any(mask):
                continue
            raw_stage_force = stage_lower_bounds[mask] + self._make_stage_force_margin_profile(
                stage_idx=stage_idx,
                z=z[mask],
                speed=speed[mask],
                tangential_speed=tangential_speed[mask],
                dz=dz[mask],
                orient_err=orient_err[mask],
                contact_gate=contact_gate[mask],
                slide_progress=slide_progress[mask],
                insert_progress=insert_progress[mask],
                rng=rng,
                latents=latents,
            )
            force[mask] = np.maximum(stage_lower_bounds[mask], raw_stage_force)

        for boundary in np.where(np.diff(labels) != 0)[0]:
            force = self._blend_segment_boundary(force[:, None], boundary=int(boundary), half_window=max(2, self.transition_half_window + 1)).ravel()

        if np.any(precontact_mask):
            pre_idx = np.where(precontact_mask)[0]
            force[pre_idx] += self._smooth_noise(rng, len(pre_idx), 0.003, kernel_size=11)
        constrained_mask = labels >= 1
        if np.any(constrained_mask):
            constrained_idx = np.where(constrained_mask)[0]
            force[constrained_idx] = np.asarray(force[constrained_idx], dtype=float)

        force[constrained_mask] = np.maximum(force[constrained_mask], stage_lower_bounds[constrained_mask])
        force_out = force.copy()
        force_out[~constrained_mask] = np.clip(force_out[~constrained_mask] + latents["force_bias"], 0.0, 0.02)
        return force_out

    def generate_demo(self, seed: int):
        rng = np.random.RandomState(seed)
        l1, l2, l3, l4 = self._sample_segment_lengths(rng)
        latents = self._sample_demo_latents(rng)
        start_local = self.start + rng.randn(2) * self.start_jitter
        start_local[0] = float(np.clip(start_local[0], -1.65, -1.12))
        start_local[1] = float(np.clip(start_local[1], 0.24, 0.46))

        stage2_end_local = np.array(
            [
                rng.uniform(*self.stage2_end_x_range),
                rng.uniform(*self.stage2_end_z_range),
            ],
            dtype=float,
        )
        stage1_end_local = self.stage1_end + rng.randn(2) * self.stage_end_jitter[0]
        stage4_end_local = self.stage4_end + rng.randn(2) * self.stage_end_jitter[3]

        stage1_end_local[0] = float(np.clip(stage2_end_local[0] - rng.uniform(0.004, 0.015), -0.82, -0.08))
        stage1_end_local[1] = float(np.clip(rng.uniform(0.004, 0.014), 0.003, 0.020))
        stage2_end_local[0] = float(np.clip(max(stage2_end_local[0], stage1_end_local[0] + 0.003), -0.70, 0.0))
        stage2_end_local[1] = float(np.clip(stage2_end_local[1], *self.stage2_end_z_range))
        stage4_end_local[0] = float(np.clip(stage4_end_local[0], 0.96, 1.03))
        stage4_end_local[1] = float(np.clip(stage4_end_local[1], -0.008, 0.008))
        stage3_end_local = self.stage3_end + rng.randn(2) * self.stage_end_jitter[2]
        stage3_end_local[0] = float(np.clip(max(stage2_end_local[0] + rng.uniform(0.88, 1.10), stage4_end_local[0] - rng.uniform(0.09, 0.13)), 0.78, 0.93))
        stage3_end_local[1] = float(np.clip(stage3_end_local[1], -0.010, 0.010))

        v1_demo = self.v1_target * rng.uniform(0.94, 1.06)
        v2_demo = self.v2_target * rng.uniform(0.98, 1.02)
        v3_demo = self.v3_target * rng.uniform(1.08, 1.14)
        v4_demo = self.v4_target * rng.uniform(0.98, 1.02)

        seg1 = self._make_target_speed_segment(
            start_local,
            stage1_end_local,
            l1 + 1,
            v1_demo,
            rng,
            max_amp=0.12,
            cycles=1.0,
            vertical_bias=0.05,
        )[:-1]

        seg2 = self._make_target_speed_segment(
            stage1_end_local,
            stage2_end_local,
            l2 + 1,
            v2_demo,
            rng,
            max_amp=0.008,
            cycles=1.0,
            vertical_bias=-0.0015,
        )[:-1]
        u2 = np.linspace(0.0, 1.0, l2, endpoint=False)
        seg2[:, 0] += 0.004 * latents["style"] * np.sin(2.0 * np.pi * u2 + latents["phase"]) * np.sin(np.pi * u2)
        seg2[:, 1] += (
            0.0025 * np.exp(-2.8 * u2) * np.sin(3.2 * np.pi * u2 + latents["phase"])
            - 0.0012 * np.sin(np.pi * u2) ** 2
        )
        seg2 = self._resample_fixed_count(seg2, l2)
        seg2[0] = stage1_end_local
        seg2[-1] = stage2_end_local

        seg3 = self._make_surface_search_segment(
            stage2_end_local,
            stage3_end_local,
            l3 + 1,
            v3_demo,
            rng,
            max_x_amp=0.13,
            z_amp=0.005,
            cycles=2.6,
        )[:-1]
        u3 = np.linspace(0.0, 1.0, l3, endpoint=False)
        seg3[:, 1] += 0.35 * latents["surface_wobble"] * np.sin(3.0 * np.pi * u3 + 0.5 * latents["phase"]) * np.sin(np.pi * u3)

        seg4 = self._make_target_speed_segment(
            stage3_end_local,
            stage4_end_local,
            l4,
            v4_demo,
            rng,
            max_amp=0.010,
            cycles=1.1,
            vertical_bias=0.0,
        )
        u4 = np.linspace(0.0, 1.0, l4, endpoint=True)
        seg4[:, 1] += 0.12 * latents["surface_wobble"] * np.sin(2.0 * np.pi * u4 + latents["phase"]) * np.sin(np.pi * u4)

        seg1 = self._timewarp_decelerating_path(seg1, strength=0.85, rng=rng, floor=0.46)
        seg2 = self._timewarp_path(seg2, strength=0.05, rng=rng, cycles=1.4)
        seg3 = self._timewarp_path(seg3, strength=0.10, rng=rng, cycles=2.0)
        seg4 = self._timewarp_path(seg4, strength=0.04, rng=rng, cycles=1.3)

        pos = np.vstack([seg1, seg2, seg3, seg4])
        labels = np.repeat(np.arange(4), [l1, l2, l3, l4])
        theta_start_local = self.theta_start + self.theta_start_jitter * rng.randn()
        theta_stage1_end = self.theta_stage1_end + self.theta_end_jitter[0] * rng.randn()
        theta_stage2_end = float(rng.uniform(*self.stage2_theta_end_range))
        theta_stage3_end = self.theta_stage3_end + self.theta_end_jitter[2] * rng.randn()
        theta_stage4_end = self.theta_stage4_end + self.theta_end_jitter[3] * rng.randn()

        theta1 = np.linspace(theta_start_local, theta_stage1_end, l1, endpoint=False)
        theta2 = np.linspace(theta_stage1_end, theta_stage2_end, l2, endpoint=False)
        theta3 = np.zeros(l3, dtype=float)
        theta4 = np.zeros(l4, dtype=float)
        sign3 = -1.0 if float(theta_stage2_end) < 0.0 else 1.0
        sign4 = sign3 if abs(float(theta_stage4_end)) < 1e-6 else (1.0 if float(theta_stage4_end) >= 0.0 else -1.0)
        if l3 > 0:
            u3_theta = np.linspace(0.0, 1.0, l3, endpoint=False)
            half_wave3 = np.maximum(np.sin(2.35 * np.pi * u3_theta - 0.5 * np.pi + 0.20 * latents["phase"]), 0.0)
            margin3 = 0.62 * self.orient_err_max_stage3 * half_wave3 - 0.18 * self.orient_err_max_stage3
            abs_theta3 = np.clip(self.orient_err_max_stage3 - margin3, 0.0, 0.96 * self.orient_err_max_stage3)
            theta3 = sign3 * self._smooth_trace(abs_theta3, kernel_size=3)
        if l4 > 0:
            u4_theta = np.linspace(0.0, 1.0, l4, endpoint=True)
            half_wave4 = np.maximum(np.sin(1.95 * np.pi * u4_theta - 0.5 * np.pi + 0.16 * latents["phase"]), 0.0)
            margin4 = 0.58 * self.orient_err_max_stage4 * half_wave4 - 0.16 * self.orient_err_max_stage4
            abs_theta4 = np.clip(self.orient_err_max_stage4 - margin4, 0.0, 0.96 * self.orient_err_max_stage4)
            theta4 = sign4 * self._smooth_trace(abs_theta4, kernel_size=3)
        theta = np.concatenate([theta1, theta2, theta3, theta4])
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 - 1, half_window=self.transition_half_window).ravel()
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 + l2 - 1, half_window=self.transition_half_window).ravel()
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 + l2 + l3 - 1, half_window=self.transition_half_window).ravel()

        pos_noise_scale_x = np.take(np.array([1.0, 0.22, 0.32, 0.18], dtype=float), labels)
        pos_noise_scale_z = np.take(np.array([1.0, 0.20, 0.28, 0.16], dtype=float), labels)
        pos[:, 0] += self._smooth_noise(rng, len(pos), 1.35 * self.noise_pos, kernel_size=11) * pos_noise_scale_x
        pos[:, 1] += self._smooth_noise(rng, len(pos), 0.95 * self.noise_pos, kernel_size=9) * pos_noise_scale_z
        stage1_floor = np.linspace(0.045, 0.012, l1)
        pos[:l1, 1] = np.maximum(pos[:l1, 1], stage1_floor)
        contact_smooth = self._smooth_noise(rng, len(pos), 0.0025, kernel_size=13)
        pos[l1:, 1] += 0.6 * contact_smooth[l1:]
        pos[l1:, 1] = np.clip(pos[l1:, 1], -0.018, 0.018)
        stage_bounds = [(0, l1), (l1, l1 + l2), (l1 + l2, l1 + l2 + l3), (l1 + l2 + l3, len(pos))]
        for stage_idx in (1, 2, 3):
            start_i, end_i = stage_bounds[stage_idx]
            pos[start_i:end_i] = self._resample_fixed_count(pos[start_i:end_i], end_i - start_i)
        for boundary in (l1 - 1, l1 + l2 - 1, l1 + l2 + l3 - 1):
            pos = self._blend_segment_boundary(pos, boundary=boundary, half_window=self.transition_half_window)
        theta_noise_scale = np.take(np.array([1.0, 0.30, 0.08, 0.05], dtype=float), labels)
        theta += self._smooth_noise(rng, len(theta), 0.28 * self.noise_misc, kernel_size=11) * theta_noise_scale
        theta += latents["theta_wobble"] * np.sin(np.linspace(0.0, 4.5 * np.pi, len(theta)) + latents["phase"]) * np.r_[
            np.linspace(0.3, 1.0, l1 + l2),
            np.linspace(0.18, 0.08, l3 + l4),
        ] * theta_noise_scale
        theta[l1 + l2:l1 + l2 + l3] = np.clip(
            theta[l1 + l2:l1 + l2 + l3],
            -0.95 * self.orient_err_max_stage3,
            0.95 * self.orient_err_max_stage3,
        )
        theta[l1 + l2 + l3:] = np.clip(
            theta[l1 + l2 + l3:],
            -0.95 * self.orient_err_max_stage4,
            0.95 * self.orient_err_max_stage4,
        )
        for boundary in (l1 - 1, l1 + l2 - 1, l1 + l2 + l3 - 1):
            theta = self._blend_segment_boundary(theta[:, None], boundary=boundary, half_window=self.transition_half_window).ravel()
        theta[l1 + l2:l1 + l2 + l3] = np.clip(
            theta[l1 + l2:l1 + l2 + l3],
            -0.98 * self.orient_err_max_stage3,
            0.98 * self.orient_err_max_stage3,
        )
        theta[l1 + l2 + l3:] = np.clip(
            theta[l1 + l2 + l3:],
            -0.98 * self.orient_err_max_stage4,
            0.98 * self.orient_err_max_stage4,
        )

        force = self._compute_force_signal(pos, theta, stage3_end_local[0], labels, rng, latents)
        speed_trace = np.zeros(len(pos), dtype=float)
        if len(pos) > 1:
            delta = np.diff(pos, axis=0) / self.dt
            speed_trace[1:] = np.linalg.norm(delta, axis=1)
            speed_trace[0] = speed_trace[1]
        stage_targets = {
            1: 0.96 * float(self.v2_target),
            2: 1.00 * float(self.v3_target),
            3: 1.00 * float(self.v4_target),
        }
        stage_amplitudes = {
            1: 0.10 * float(self.v2_target),
            2: 0.10 * float(self.v3_target),
            3: 0.08 * float(self.v4_target),
        }
        stage_noise = {
            1: 0.03 * float(self.v2_target),
            2: 0.04 * float(self.v3_target),
            3: 0.03 * float(self.v4_target),
        }
        stage_bounds = [(0, l1), (l1, l1 + l2), (l1 + l2, l1 + l2 + l3), (l1 + l2 + l3, len(pos))]
        for stage_idx, (start_i, end_i) in enumerate(stage_bounds[1:], start=1):
            n = int(end_i - start_i)
            if n <= 0:
                continue
            u = np.linspace(0.0, 1.0, n, endpoint=True)
            profile = (
                stage_targets[stage_idx]
                + stage_amplitudes[stage_idx]
                * np.sin(np.pi * u)
                * np.sin((1.00 + 0.10 * stage_idx) * np.pi * u + float(latents["phase"]) + 0.20 * stage_idx)
            )
            profile += self._smooth_noise(rng, n, stage_noise[stage_idx], kernel_size=5)
            speed_trace[start_i:end_i] = np.clip(profile, 0.0, None)

        return np.asarray(pos, dtype=float), np.asarray(theta, dtype=float), np.asarray(labels, dtype=int), force, speed_trace

    @staticmethod
    def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
        return (np.asarray(angle, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _traj_cache_key(traj: np.ndarray):
        arr = np.ascontiguousarray(np.asarray(traj, dtype=np.float64))
        return arr.shape, arr.tobytes()

    def register_force_trace(self, traj: np.ndarray, force: np.ndarray):
        self._cached_force_traces[self._traj_cache_key(traj)] = np.asarray(force, dtype=float).copy()

    def register_speed_trace(self, traj: np.ndarray, speed: np.ndarray):
        self._cached_speed_traces[self._traj_cache_key(traj)] = np.asarray(speed, dtype=float).copy()

    def _lookup_cached_force_trace(self, traj: np.ndarray):
        key = self._traj_cache_key(traj)
        force = self._cached_force_traces.get(key)
        if force is None:
            return None
        return np.asarray(force, dtype=float)

    def _lookup_cached_speed_trace(self, traj: np.ndarray):
        key = self._traj_cache_key(traj)
        speed = self._cached_speed_traces.get(key)
        if speed is None:
            return None
        return np.asarray(speed, dtype=float)

    def _estimate_force_from_state(self, traj: np.ndarray) -> np.ndarray:
        traj = np.asarray(traj, dtype=float)
        pos = traj[:, :2]
        theta = traj[:, 2]
        T = len(pos)
        speed = np.zeros(T, dtype=float)
        tangential_speed = np.zeros(T, dtype=float)
        dz = np.zeros(T, dtype=float)
        if T > 1:
            delta = np.diff(pos, axis=0) / self.dt
            speed[1:] = np.linalg.norm(delta, axis=1)
            speed[0] = speed[1]
            tangential_speed[1:] = np.abs(delta[:, 0])
            tangential_speed[0] = tangential_speed[1]
            dz[1:] = np.abs(delta[:, 1])
            dz[0] = dz[1]
        orient_err = np.abs(self._wrap_to_pi(theta - self.slot_theta))
        z = pos[:, 1]
        x = pos[:, 0]
        contact_gate = 1.0 / (1.0 + np.exp((z - 0.012) / 0.006))
        slide_progress = 1.0 / (1.0 + np.exp(-(x - self.stage2_end[0]) / 0.055))
        insert_progress = 1.0 / (1.0 + np.exp(-(x - self.stage3_end[0]) / 0.045))

        contact_weight = np.clip(contact_gate * (1.0 - slide_progress), 0.0, 1.0)
        slide_weight = np.clip(contact_gate * slide_progress * (1.0 - insert_progress), 0.0, 1.0)
        insert_weight = np.clip(contact_gate * insert_progress, 0.0, 1.0)
        weight_sum = np.maximum(contact_weight + slide_weight + insert_weight, 1e-6)
        contact_weight = contact_weight / weight_sum
        slide_weight = slide_weight / weight_sum
        insert_weight = insert_weight / weight_sum
        precontact_gate = 1.0 / (1.0 + np.exp((x - self.stage1_end[0]) / 0.05))
        precontact_mean = 0.010
        base_lower = (
            contact_weight * self.f_contact_min
            + slide_weight * self.f_slide_min
            + insert_weight * self.f_insert_min
        )
        contact_margin = self._make_stage_force_margin_profile(
            stage_idx=1,
            z=z,
            speed=speed,
            tangential_speed=tangential_speed,
            dz=dz,
            orient_err=orient_err,
            contact_gate=contact_gate,
            slide_progress=slide_progress,
            insert_progress=insert_progress,
            rng=None,
            latents=None,
        )
        slide_margin = self._make_stage_force_margin_profile(
            stage_idx=2,
            z=z,
            speed=speed,
            tangential_speed=tangential_speed,
            dz=dz,
            orient_err=orient_err,
            contact_gate=contact_gate,
            slide_progress=slide_progress,
            insert_progress=insert_progress,
            rng=None,
            latents=None,
        )
        insert_margin = self._make_stage_force_margin_profile(
            stage_idx=3,
            z=z,
            speed=speed,
            tangential_speed=tangential_speed,
            dz=dz,
            orient_err=orient_err,
            contact_gate=contact_gate,
            slide_progress=slide_progress,
            insert_progress=insert_progress,
            rng=None,
            latents=None,
        )
        raw_force = (
            contact_weight * (self.f_contact_min + contact_margin)
            + slide_weight * (self.f_slide_min + slide_margin)
            + insert_weight * (self.f_insert_min + insert_margin)
        )
        force = precontact_gate * precontact_mean + (1.0 - precontact_gate) * np.maximum(base_lower, raw_force)
        force[precontact_gate > 0.5] = np.clip(force[precontact_gate > 0.5], 0.0, 0.03)
        return np.clip(force, 0.0, None)

    def compute_all_features_matrix(self, traj: np.ndarray, feat_ids=None) -> np.ndarray:
        traj = np.asarray(traj, dtype=float)
        T = traj.shape[0]
        speed_cached = self._lookup_cached_speed_trace(traj)
        speed = np.zeros(T, dtype=float) if speed_cached is None else np.asarray(speed_cached, dtype=float)
        if speed_cached is None and T > 1:
            speed_edge = np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1) / self.dt
            speed[0] = speed_edge[0]
            speed[1:] = speed_edge
        surf_dist = np.abs(traj[:, 1])
        orient_err = np.abs(self._wrap_to_pi(traj[:, 2] - self.slot_theta))
        if traj.shape[1] >= 4:
            force = np.asarray(traj[:, 3], dtype=float)
        else:
            force = self._lookup_cached_force_trace(traj)
            if force is None:
                force = self._estimate_force_from_state(traj)
        start_xy = np.asarray(traj[0, :2], dtype=float)
        start_dist = np.linalg.norm(np.asarray(traj[:, :2], dtype=float) - start_xy[None, :], axis=1)
        insertion_err = float(self.slot_x) - np.asarray(traj[:, 0], dtype=float)
        noise = 0.35 * np.sin(0.19 * np.arange(T)) + 0.15 * np.cos(0.07 * np.arange(T))
        F = np.c_[surf_dist, force, orient_err, speed, noise, start_dist, insertion_err]
        return F if feat_ids is None else F[:, feat_ids]






class S4SlideInsertEnv(_S4SlideInsertBase):
    """Robot-friendly S4 copy with a lateral clearance dimension.

    State is [x, y, z, theta] in the robot-friendly units used by this task.
    """

    def __init__(
        self,
        slot_half_width: float = 0.032,
        clearance_target: float = 0.0,
        clearance_align_max: float = 0.006,
        clearance_insert_max: float = 0.0035,
        normal_load_min: float = 0.40,
        rollout_backend: str = "analytic",
        observation_backend: str | None = None,
        pybullet_world_center=(0.55, 0.0, 0.52),
        pybullet_sim_dt: float = 1.0 / 120.0,
        pybullet_steps_per_sample=None,
        pybullet_gravity_z: float = -9.81,
        pybullet_solver_iterations: int = 80,
        pybullet_ur5_home_q=(0.0, -1.35, 1.85, -2.05, -1.57, 0.0),
        pybullet_ur5_tool_axis: str = "-x",
        pybullet_ur5_tip_offset: float = 0.0,
        pybullet_ur5_ee_link_index: int = -1,
        pybullet_ur5_urdf_path=None,
        pybullet_ur5_base_xyz=(0.0, 0.0, 0.0),
        pybullet_ur5_base_rpy=(0.0, 0.0, 0.0),
        pybullet_ur5_ik_iterations: int = 120,
        pybullet_ur5_ik_damping: float = 0.02,
        pybullet_ur5_position_gain: float = 0.08,
        pybullet_ur5_velocity_gain: float = 1.0,
        pybullet_ur5_max_force: float = 500.0,
        pybullet_s4_track_orientation: bool = True,
        pybullet_visualize_normal_load: bool = False,
        pybullet_normal_load_arrow_scale: float = 0.055,
        pybullet_suppress_urdf_warnings: bool = True,
        pybullet_marker_radius: float = 0.002,
        pybullet_table_half_extents=(0.45, 0.20, 0.015),
        slider_half_extents=(0.080, 0.026, 0.018),
        pybullet_grasp_height: float = 0.070,
        slot_wall_length: float = 0.19,
        slot_wall_forward_extension: float = 0.065,
        slot_wall_thickness: float = 0.010,
        slot_wall_height: float = 0.030,
        rail_shape: str = "straight",
        rail_polyline=None,
        rail_bend_amp: float = 0.012,
        surface_tilt_x: float = 0.0,
        surface_tilt_y: float = 0.0,
        surface_z0: float = 0.0,
        pybullet_render_width: int = 1280,
        pybullet_render_height: int = 900,
        pybullet_camera_target=(0.59, 0.08, 0.54),
        pybullet_camera_distance: float = 0.90,
        pybullet_camera_yaw: float = 38.0,
        pybullet_camera_pitch: float = -33.0,
        pybullet_camera_fov: float = 42.0,
        **kwargs,
    ):
        realistic_defaults = {
            "dt": 0.40,
            "seg_lengths": (35, 14, 67, 21),
            "seg_length_jitter": (9, 5, 11, 7),
            "start": (-0.2240, 0.0560),
            "start_jitter": (0.0160, 0.0080),
            "stage1_end": (-0.0400, 0.00224),
            "stage2_end": (-0.0320, 0.0),
            "stage3_end": (0.1120, 0.0),
            "stage4_end": (0.1600, 0.0),
            "stage_end_jitter": (
                (0.00512, 0.00192),
                (0.00384, 0.00096),
                (0.00608, 0.00096),
                (0.00224, 0.00048),
            ),
            "stage2_end_x_range": (-0.1120, 0.0),
            "stage2_end_z_range": (-0.00160, 0.00160),
            "slot_x": 0.1600,
            "noise_pos": 0.00048,
            "v1_target": 0.00960,
            "v2_target": 0.00112,
            "v3_target": 0.00720,
            "v4_target": 0.00288,
        }
        base_kwargs = dict(realistic_defaults)
        base_kwargs.update(kwargs)
        self.slot_half_width = float(slot_half_width)
        self.clearance_target = float(clearance_target)
        self.clearance_align_max = float(clearance_align_max)
        self.clearance_insert_max = float(clearance_insert_max)
        self.normal_load_min = float(normal_load_min)
        self.v_align_max = float(base_kwargs["v2_target"])
        self.v_insert_max = float(base_kwargs["v3_target"])
        self.v_seat_max = float(base_kwargs["v4_target"])
        super().__init__(**base_kwargs)
        self.v_align_max = float(self.v2_target)
        self.v_insert_max = float(self.v3_target)
        self.v_seat_max = float(self.v4_target)
        self.slot_half_width = float(slot_half_width)
        self.clearance_target = float(clearance_target)
        self.clearance_align_max = float(clearance_align_max)
        self.clearance_insert_max = float(clearance_insert_max)
        self.normal_load_min = float(normal_load_min)
        self.rollout_backend = str(rollout_backend).lower()
        self.observation_backend = str(observation_backend or self.rollout_backend).lower()
        self.pybullet_world_center = tuple(float(x) for x in np.asarray(pybullet_world_center, dtype=float).reshape(3))
        self.pybullet_sim_dt = float(pybullet_sim_dt)
        self.pybullet_steps_per_sample = None if pybullet_steps_per_sample is None else int(pybullet_steps_per_sample)
        self.pybullet_gravity_z = float(pybullet_gravity_z)
        self.pybullet_solver_iterations = int(pybullet_solver_iterations)
        self.pybullet_ur5_home_q = tuple(float(x) for x in pybullet_ur5_home_q)
        self.pybullet_ur5_tool_axis = str(pybullet_ur5_tool_axis)
        self.pybullet_ur5_tip_offset = float(pybullet_ur5_tip_offset)
        self.pybullet_ur5_ee_link_index = int(pybullet_ur5_ee_link_index)
        self.pybullet_ur5_urdf_path = pybullet_ur5_urdf_path
        self.pybullet_ur5_base_xyz = tuple(float(x) for x in pybullet_ur5_base_xyz)
        self.pybullet_ur5_base_rpy = tuple(float(x) for x in pybullet_ur5_base_rpy)
        self.pybullet_ur5_ik_iterations = int(pybullet_ur5_ik_iterations)
        self.pybullet_ur5_ik_damping = float(pybullet_ur5_ik_damping)
        self.pybullet_ur5_position_gain = float(pybullet_ur5_position_gain)
        self.pybullet_ur5_velocity_gain = float(pybullet_ur5_velocity_gain)
        self.pybullet_ur5_max_force = float(pybullet_ur5_max_force)
        self.pybullet_s4_track_orientation = bool(pybullet_s4_track_orientation)
        self.pybullet_visualize_normal_load = bool(pybullet_visualize_normal_load)
        self.pybullet_normal_load_arrow_scale = float(pybullet_normal_load_arrow_scale)
        self.pybullet_suppress_urdf_warnings = bool(pybullet_suppress_urdf_warnings)
        self.pybullet_marker_radius = float(pybullet_marker_radius)
        self.pybullet_table_half_extents = tuple(float(x) for x in pybullet_table_half_extents)
        self.slider_half_extents = tuple(float(x) for x in slider_half_extents)
        self.pybullet_grasp_height = float(pybullet_grasp_height)
        self.slot_wall_length = float(slot_wall_length)
        self.slot_wall_forward_extension = float(slot_wall_forward_extension)
        self.slot_wall_thickness = float(slot_wall_thickness)
        self.slot_wall_height = float(slot_wall_height)
        self.rail_shape = str(rail_shape).strip().lower()
        self.rail_polyline = self._coerce_rail_polyline(rail_polyline)
        self.rail_bend_amp = float(rail_bend_amp)
        self.surface_tilt_x = float(surface_tilt_x)
        self.surface_tilt_y = float(surface_tilt_y)
        self.surface_z0 = float(surface_z0)
        self.pybullet_render_width = int(pybullet_render_width)
        self.pybullet_render_height = int(pybullet_render_height)
        self.pybullet_camera_target = tuple(float(x) for x in pybullet_camera_target)
        self.pybullet_camera_distance = float(pybullet_camera_distance)
        self.pybullet_camera_yaw = float(pybullet_camera_yaw)
        self.pybullet_camera_pitch = float(pybullet_camera_pitch)
        self.pybullet_camera_fov = float(pybullet_camera_fov)
        self.eval_tag = "S4SlideInsert"
        self.subgoal = np.asarray([self.stage2_end[0], self.clearance_target, self.stage2_end[1], self.theta_stage2_end], dtype=float)
        self.goal = np.asarray([self.stage4_end[0], self.clearance_target, self.stage4_end[1], self.theta_stage4_end], dtype=float)
        self._cached_normal_load_traces = {}
        self.true_constraints = self.get_true_constraints()
        self.constraint_specs = self.get_constraint_specs()
        self.feature_schema = self.get_feature_schema()

    def get_feature_schema(self):
        return [
            {"id": 0, "name": "surf_dist", "description": "Distance to the table/guide surface"},
            {"id": 1, "name": "centerline_dist", "description": "Absolute lateral distance from the slot centerline"},
            {"id": 2, "name": "orient_err", "description": "Absolute angle error relative to the slot"},
            {"id": 3, "name": "speed", "description": "3D translational speed"},
            {"id": 4, "name": "angular_speed", "description": "Absolute angular speed"},
            {"id": 5, "name": "normal_load", "description": "Normal preload applied against the guide"},
            {"id": 6, "name": "noise", "description": "Auxiliary irrelevant feature"},
            {"id": 7, "name": "start_dist", "description": "Distance to the demo start pose"},
            {"id": 8, "name": "insertion_err", "description": "Remaining x-direction distance to the slot target"},
        ]

    def get_overlay_feature_names(self):
        return [
            "surf_dist",
            "centerline_dist",
            "orient_err",
            "speed",
            "normal_load",
            "start_dist",
            "insertion_err",
        ]

    def get_true_constraints(self):
        return {
            "surface_target": 0.0,
            "clearance_target": float(self.clearance_target),
            "clearance_align_max": float(self.clearance_align_max),
            "clearance_insert_max": float(self.clearance_insert_max),
            "normal_load_stage2_min": float(self.f_contact_min),
            "normal_load_stage3_min": float(self.f_slide_min),
            "normal_load_stage4_min": float(self.f_insert_min),
            "v_align_max": float(self.v_align_max),
            "v_insert_max": float(self.v_insert_max),
            "v_seat_max": float(self.v_seat_max),
            "orient_aligned_max": float(self.orient_err_max_stage3),
        }

    def get_constraint_specs(self):
        return [
            {"feature_name": "surf_dist", "stage": 1, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "centerline_dist", "stage": 1, "semantics": "target_value", "oracle_key": "clearance_target"},
            {"feature_name": "normal_load", "stage": 1, "semantics": "lower_bound", "oracle_key": "normal_load_stage2_min"},
            {"feature_name": "surf_dist", "stage": 2, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "centerline_dist", "stage": 2, "semantics": "target_value", "oracle_key": "clearance_target"},
            {"feature_name": "orient_err", "stage": 2, "semantics": "upper_bound", "oracle_key": "orient_aligned_max"},
            {"feature_name": "normal_load", "stage": 2, "semantics": "lower_bound", "oracle_key": "normal_load_stage3_min"},
            {"feature_name": "speed", "stage": 2, "semantics": "upper_bound", "oracle_key": "v_insert_max"},
            {"feature_name": "surf_dist", "stage": 3, "semantics": "target_value", "oracle_key": "surface_target"},
            {"feature_name": "centerline_dist", "stage": 3, "semantics": "target_value", "oracle_key": "clearance_target"},
            {"feature_name": "orient_err", "stage": 3, "semantics": "upper_bound", "oracle_key": "orient_aligned_max"},
            {"feature_name": "normal_load", "stage": 3, "semantics": "lower_bound", "oracle_key": "normal_load_stage4_min"},
            {"feature_name": "speed", "stage": 3, "semantics": "upper_bound", "oracle_key": "v_seat_max"},
        ]

    def get_observation_spec(self):
        return {
            "feature_schema": self.get_feature_schema(),
            "default_rollout_backend": self.rollout_backend,
            "default_observation_backend": self.observation_backend,
            "state_schema": ["x", "y", "z", "theta"],
            "normal_load_semantics": "contact preload; controller/debug signal, not insertion push effort",
        }

    def sample_scene(self, seed=None, rng=None):
        scene = super().sample_scene(seed=seed, rng=rng)
        scene["task_name"] = "S4SlideInsert"
        scene["geometry"].update(
            {
                "start": self.start.tolist(),
                "stage1_end": self.stage1_end.tolist(),
                "stage2_end": self.stage2_end.tolist(),
                "stage3_end": self.stage3_end.tolist(),
                "stage4_end": self.stage4_end.tolist(),
                "slot_x": float(self.slot_x),
                "slot_center_y": float(self.clearance_target),
                "slot_half_width": float(self.slot_half_width),
                "rail_shape": str(self.rail_shape),
                "rail_polyline": self.get_rail_polyline(num=64).tolist(),
                "surface_z": float(self.surface_z0),
                "surface_tilt_x": float(self.surface_tilt_x),
                "surface_tilt_y": float(self.surface_tilt_y),
            }
        )
        return scene

    def surface_height(self, xy: np.ndarray) -> np.ndarray:
        pts = np.asarray(xy, dtype=float)
        flat = pts.reshape(-1, 2)
        z = (
            float(self.surface_z0)
            + float(self.surface_tilt_x) * (flat[:, 0] - float(self.slot_x))
            + float(self.surface_tilt_y) * (flat[:, 1] - float(self.clearance_target))
        )
        return z.reshape(pts.shape[:-1])

    @staticmethod
    def _coerce_rail_polyline(polyline):
        if polyline is None:
            return None
        if isinstance(polyline, str):
            text = polyline.strip()
            if not text:
                return None
            pts = []
            for item in text.split(";"):
                xy = [float(v.strip()) for v in item.split(",") if v.strip()]
                if len(xy) != 2:
                    raise ValueError(f"Invalid rail polyline point {item!r}; expected 'x,y'.")
                pts.append(xy)
            arr = np.asarray(pts, dtype=float)
        else:
            arr = np.asarray(polyline, dtype=float)
        arr = np.asarray(arr, dtype=float).reshape(-1, 2)
        if arr.shape[0] < 2:
            raise ValueError("rail_polyline must contain at least two 2D points.")
        return arr

    def get_rail_polyline(self, num: int = 96) -> np.ndarray:
        if self.rail_polyline is not None:
            return np.asarray(self.rail_polyline, dtype=float).reshape(-1, 2).copy()
        x0 = float(self.slot_x) - float(self.slot_wall_length)
        x1 = float(self.slot_x)
        y0 = float(self.clearance_target)
        shape = str(self.rail_shape or "straight").strip().lower()
        if shape == "polyline":
            xm = 0.5 * (x0 + x1)
            return np.asarray([[x0, y0], [xm, y0 + float(self.rail_bend_amp)], [x1, y0]], dtype=float)
        n = max(int(num), 2)
        u = np.linspace(0.0, 1.0, n)
        x = x0 + (x1 - x0) * u
        if shape in {"sine", "curve", "curved"}:
            y = y0 + float(self.rail_bend_amp) * np.sin(np.pi * u)
        else:
            y = np.full_like(x, y0)
        return np.c_[x, y]

    def _rail_segments(self):
        pts = self.get_rail_polyline(num=128)
        seg = pts[1:] - pts[:-1]
        lengths = np.linalg.norm(seg, axis=1)
        keep = lengths > 1e-10
        if not np.any(keep):
            pts = np.asarray([[float(self.start[0]), float(self.clearance_target)], [float(self.slot_x), float(self.clearance_target)]], dtype=float)
            seg = pts[1:] - pts[:-1]
            lengths = np.linalg.norm(seg, axis=1)
            keep = lengths > 1e-10
        seg = seg[keep]
        starts = pts[:-1][keep]
        lengths = lengths[keep]
        tangents = seg / lengths[:, None]
        normals = np.c_[-tangents[:, 1], tangents[:, 0]]
        cum = np.r_[0.0, np.cumsum(lengths)]
        return starts, lengths, tangents, normals, cum

    def rail_total_length(self) -> float:
        *_, cum = self._rail_segments()
        return float(cum[-1])

    def rail_pose_at_s(self, s):
        starts, lengths, tangents, normals, cum = self._rail_segments()
        s_arr = np.asarray(s, dtype=float)
        flat = np.clip(s_arr.reshape(-1), 0.0, float(cum[-1]))
        idx = np.searchsorted(cum[1:], flat, side="right")
        idx = np.clip(idx, 0, len(lengths) - 1)
        local = (flat - cum[idx]) / np.maximum(lengths[idx], 1e-12)
        points = starts[idx] + tangents[idx] * (local[:, None] * lengths[idx, None])
        angles = np.arctan2(tangents[idx, 1], tangents[idx, 0])
        shape = s_arr.shape
        return (
            points.reshape(shape + (2,)),
            tangents[idx].reshape(shape + (2,)),
            normals[idx].reshape(shape + (2,)),
            angles.reshape(shape),
        )

    def project_to_rail(self, xy: np.ndarray) -> dict[str, np.ndarray]:
        pts = np.asarray(xy, dtype=float).reshape(-1, 2)
        starts, lengths, tangents, normals, cum = self._rail_segments()
        best_d2 = np.full(pts.shape[0], np.inf, dtype=float)
        best_s = np.zeros(pts.shape[0], dtype=float)
        best_signed = np.zeros(pts.shape[0], dtype=float)
        best_angle = np.zeros(pts.shape[0], dtype=float)
        for i in range(len(lengths)):
            rel = pts - starts[i]
            t = np.clip(rel @ tangents[i] / max(lengths[i], 1e-12), 0.0, lengths[i])
            proj = starts[i] + t[:, None] * tangents[i]
            delta = pts - proj
            d2 = np.sum(delta * delta, axis=1)
            take = d2 < best_d2
            if np.any(take):
                best_d2[take] = d2[take]
                best_s[take] = cum[i] + t[take]
                best_signed[take] = delta[take] @ normals[i]
                best_angle[take] = np.arctan2(tangents[i, 1], tangents[i, 0])
        total = float(cum[-1])
        return {
            "s": best_s,
            "signed_dist": best_signed,
            "dist": np.abs(best_signed),
            "angle": best_angle,
            "remaining": total - best_s,
        }

    @staticmethod
    def _traj_cache_key(traj: np.ndarray):
        arr = np.ascontiguousarray(np.asarray(traj, dtype=np.float64))
        return arr.shape, arr.tobytes()

    def register_normal_load_trace(self, traj: np.ndarray, load: np.ndarray):
        self._cached_normal_load_traces[self._traj_cache_key(traj)] = np.asarray(load, dtype=float).copy()

    def _lookup_cached_normal_load_trace(self, traj: np.ndarray):
        load = self._cached_normal_load_traces.get(self._traj_cache_key(traj))
        return None if load is None else np.asarray(load, dtype=float)

    def _normal_load_profile(self, labels: np.ndarray, rng: np.random.RandomState | None = None):
        labels = np.asarray(labels, dtype=int)
        load = np.zeros(len(labels), dtype=float)
        for stage_idx, base in [(1, self.normal_load_min), (2, 1.08 * self.normal_load_min), (3, 1.05 * self.normal_load_min)]:
            mask = labels == stage_idx
            if not np.any(mask):
                continue
            n = int(mask.sum())
            u = np.linspace(0.0, 1.0, n)
            profile = base * (1.0 + 0.08 * np.sin(2.0 * np.pi * u - 0.5 * np.pi) * np.sin(np.pi * u))
            if rng is not None:
                profile += self._smooth_noise(rng, n, 0.025 * base, kernel_size=5)
            load[mask] = np.maximum(0.92 * base, profile)
        return load

    def apply_execution_normal_load_noise(self, normal_load: np.ndarray, *, noise_std: float = 0.0, noise_smooth: float = 0.85, seed=None):
        load = np.asarray(normal_load, dtype=float).reshape(-1)
        executed = load.copy()
        std = float(noise_std)
        if std <= 0.0 or executed.size == 0:
            return executed, np.zeros_like(executed)
        rng = np.random.RandomState(0 if seed is None else int(seed) + 9173)
        raw = rng.normal(0.0, std, size=executed.shape)
        smooth = float(np.clip(float(noise_smooth), 0.0, 0.999))
        noise = np.zeros_like(raw, dtype=float)
        for i in range(1, raw.size):
            noise[i] = smooth * noise[i - 1] + (1.0 - smooth) * raw[i]
        active = load > 1e-9
        executed[active] = np.maximum(0.0, load[active] + noise[active])
        return executed, executed - load

    def _lift_planar_demo_to_4d(self, pos_planar: np.ndarray, theta: np.ndarray, labels: np.ndarray, rng: np.random.RandomState):
        pos_planar = np.asarray(pos_planar, dtype=float)
        labels = np.asarray(labels, dtype=int)
        x = pos_planar[:, 0]
        z = pos_planar[:, 1]
        y = np.zeros(len(x), dtype=float)
        stage0 = labels == 0
        if np.any(stage0):
            n0 = int(stage0.sum())
            y0_start = float(rng.uniform(-0.055, 0.055))
            y[stage0] = np.linspace(y0_start, 0.0, n0, endpoint=True)
        for stage_idx, amp in [(1, 0.00018), (2, 0.00100), (3, 0.00055)]:
            mask = labels == stage_idx
            if not np.any(mask):
                continue
            n = int(mask.sum())
            u = np.linspace(0.0, 1.0, n)
            y_wave = amp * np.sin(2.0 * np.pi * u + float(rng.uniform(-np.pi, np.pi))) * np.sin(np.pi * u)
            y_noise = self._smooth_noise(rng, n, 0.22 * amp, kernel_size=5) * np.sin(np.pi * u)
            y[mask] = np.clip(y_wave + y_noise, -1.25 * amp, 1.25 * amp)
        z[labels >= 1] = np.clip(z[labels >= 1], -0.002, 0.002)
        return np.c_[x, y, z, np.asarray(theta, dtype=float)]

    @staticmethod
    def _sample_polyline_by_edge_weights(path: np.ndarray, num_points: int, edge_weights: np.ndarray) -> np.ndarray:
        pts = np.asarray(path, dtype=float)
        n = int(num_points)
        if len(pts) <= 1 or n <= 1:
            return np.repeat(pts[:1], max(n, 1), axis=0)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= 1e-12:
            out = np.repeat(pts[:1], n, axis=0)
            out[0] = pts[0]
            out[-1] = pts[-1]
            return out
        weights = np.asarray(edge_weights, dtype=float).reshape(-1)
        expected = max(n - 1, 1)
        if weights.size != expected:
            raise ValueError(f"edge_weights produced {weights.size} values, expected {expected}.")
        weights = np.clip(weights, 1e-6, None)
        targets = np.concatenate([[0.0], np.cumsum(weights)])
        targets = targets / float(targets[-1]) * total
        out = np.empty((n, pts.shape[1]), dtype=float)
        for d in range(pts.shape[1]):
            out[:, d] = np.interp(targets, s, pts[:, d])
        out[0] = pts[0]
        out[-1] = pts[-1]
        return out

    def _speed_profile_weights(self, stage_idx: int, num_edges: int, rng: np.random.RandomState, phase: float) -> np.ndarray:
        n = int(num_edges)
        if n <= 0:
            return np.zeros(0, dtype=float)
        u = np.linspace(0.0, 1.0, n, endpoint=True)
        if int(stage_idx) == 0:
            weights = 1.38 - 0.60 * u + 0.05 * np.sin(2.0 * np.pi * u + phase) * np.sin(np.pi * u)
        elif int(stage_idx) == 1:
            valley = np.exp(-0.5 * ((u - 0.58) / 0.12) ** 2)
            weights = 1.0 - 0.16 * valley + 0.035 * np.sin(2.5 * np.pi * u + phase) * np.sin(np.pi * u)
        elif int(stage_idx) == 2:
            valleys = (
                0.10 * np.exp(-0.5 * ((u - 0.32) / 0.040) ** 2)
                + 0.07 * np.exp(-0.5 * ((u - 0.61) / 0.050) ** 2)
                + 0.06 * np.exp(-0.5 * ((u - 0.80) / 0.035) ** 2)
            )
            micro_slowdown = 0.005 * np.abs(self._smooth_noise(rng, n, 1.0, kernel_size=5)) * np.sin(np.pi * u)
            weights = 1.0 - valleys - micro_slowdown
        else:
            valleys = (
                0.115 * np.exp(-0.5 * ((u - 0.45) / 0.055) ** 2)
                + 0.085 * np.exp(-0.5 * ((u - 0.72) / 0.040) ** 2)
            )
            micro_slowdown = 0.005 * np.abs(self._smooth_noise(rng, n, 1.0, kernel_size=3)) * np.sin(np.pi * u)
            weights = 1.0 - valleys - micro_slowdown
        if int(stage_idx) == 2:
            weights += self._smooth_noise(rng, n, 0.002, kernel_size=5)
            weights = np.minimum(weights, 1.0)
        elif int(stage_idx) == 3:
            weights += self._smooth_noise(rng, n, 0.0015, kernel_size=3)
            weights = np.minimum(weights, 1.0)
        else:
            weights += self._smooth_noise(rng, n, 0.012, kernel_size=5)
        return np.clip(weights, 0.45, None)

    @staticmethod
    def _planar_curve(
        start: np.ndarray,
        end: np.ndarray,
        rng: np.random.RandomState,
        *,
        amp: float,
        cycles: float,
        z_bias: float = 0.0,
        n: int = 512,
    ) -> np.ndarray:
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        u = np.linspace(0.0, 1.0, int(n), endpoint=True)
        base = start[None, :] + u[:, None] * (end - start)[None, :]
        direction = end - start
        dist = float(np.linalg.norm(direction))
        if dist <= 1e-12:
            normal = np.array([0.0, 1.0], dtype=float)
        else:
            tangent = direction / dist
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
        phase = float(rng.uniform(-0.5 * np.pi, 0.5 * np.pi))
        sign = -1.0 if rng.rand() < 0.5 else 1.0
        envelope = np.sin(np.pi * u)
        curve = base + sign * float(amp) * envelope[:, None] * np.sin(float(cycles) * np.pi * u + phase)[:, None] * normal[None, :]
        if z_bias != 0.0:
            curve[:, 1] += float(z_bias) * envelope**2
        curve[0] = start
        curve[-1] = end
        return curve

    def _resample_planar_segment(
        self,
        path: np.ndarray,
        num_points: int,
        stage_idx: int,
        rng: np.random.RandomState,
        phase: float,
    ) -> np.ndarray:
        n = int(num_points)
        weights = self._speed_profile_weights(stage_idx, max(n - 1, 1), rng, phase)
        return self._sample_polyline_by_edge_weights(path, n, weights)

    def generate_demo(self, seed: int):
        rng = np.random.RandomState(seed)
        l1, l2, l3, l4 = self._sample_segment_lengths(rng)
        latents = self._sample_demo_latents(rng)
        latents["surface_wobble"] = float(rng.uniform(0.00024, 0.00080))
        phase = float(latents["phase"])

        v1_demo = self.v1_target * rng.uniform(0.94, 1.06)
        v2_demo = self.v2_target * rng.uniform(0.82, 0.90)
        v3_demo = self.v3_target * rng.uniform(0.95, 0.98)
        v4_demo = self.v4_target * rng.uniform(0.92, 0.97)

        start_local = self.start + rng.randn(2) * self.start_jitter
        start_local[0] = float(np.clip(start_local[0], -0.2640, -0.1792))
        start_local[1] = float(np.clip(start_local[1], 0.0384, 0.0736))

        stage4_end_local = self.stage4_end + rng.randn(2) * self.stage_end_jitter[3]
        stage4_end_local[0] = float(np.clip(stage4_end_local[0], 0.1536, 0.1648))
        stage4_end_local[1] = float(np.clip(stage4_end_local[1], -0.00128, 0.00128))

        seat_len = v4_demo * self.dt * max(l4 - 1, 1) * rng.uniform(0.94, 1.04)
        stage3_end_local = self.stage3_end + rng.randn(2) * self.stage_end_jitter[2]
        stage3_end_local[0] = float(np.clip(stage4_end_local[0] - seat_len, 0.1248, 0.1488))
        stage3_end_local[1] = float(np.clip(stage3_end_local[1], -0.00160, 0.00160))

        insert_len = v3_demo * self.dt * max(l3, 1) * rng.uniform(0.96, 1.03)
        stage2_end_local = np.array(
            [
                stage3_end_local[0] - insert_len,
                rng.uniform(*self.stage2_end_z_range),
            ],
            dtype=float,
        )
        stage2_end_local[0] = float(np.clip(stage2_end_local[0], -0.0960, 0.0060))

        align_len = v2_demo * self.dt * max(l2, 1) * rng.uniform(0.92, 1.08)
        stage1_end_local = self.stage1_end + rng.randn(2) * self.stage_end_jitter[0]
        stage1_end_local[0] = float(np.clip(stage2_end_local[0] - align_len, -0.1120, stage2_end_local[0] - 0.00045))
        stage1_end_local[1] = float(np.clip(rng.uniform(0.00064, 0.00224), 0.00048, 0.00320))

        seg1_path = self._planar_curve(start_local, stage1_end_local, rng, amp=0.0060, cycles=1.0, z_bias=0.0060)
        seg2_path = self._planar_curve(stage1_end_local, stage2_end_local, rng, amp=0.00025, cycles=1.1, z_bias=-0.00012)
        seg3_path = self._planar_curve(stage2_end_local, stage3_end_local, rng, amp=0.00120, cycles=2.3, z_bias=0.0000)
        seg4_path = self._planar_curve(stage3_end_local, stage4_end_local, rng, amp=0.00055, cycles=1.2, z_bias=0.0)

        seg1 = self._resample_planar_segment(seg1_path, l1, 0, rng, phase)
        seg2 = self._resample_planar_segment(seg2_path, l2 + 1, 1, rng, phase + 0.4)[1:]
        seg3 = self._resample_planar_segment(seg3_path, l3 + 1, 2, rng, phase + 0.8)[1:]
        seg4 = self._resample_planar_segment(seg4_path, l4 + 1, 3, rng, phase + 1.2)[1:]

        pos = np.vstack([seg1, seg2, seg3, seg4])
        labels = np.repeat(np.arange(4), [l1, l2, l3, l4])

        theta_start_local = self.theta_start + self.theta_start_jitter * rng.randn()
        theta_stage1_end = self.theta_stage1_end + self.theta_end_jitter[0] * rng.randn()
        theta_stage2_end = float(rng.uniform(*self.stage2_theta_end_range))
        theta_stage3_end = self.theta_stage3_end + self.theta_end_jitter[2] * rng.randn()
        theta_stage4_end = self.theta_stage4_end + self.theta_end_jitter[3] * rng.randn()

        theta1 = np.linspace(theta_start_local, theta_stage1_end, l1, endpoint=False)
        theta2 = np.linspace(theta_stage1_end, theta_stage2_end, l2, endpoint=False)
        theta3 = np.zeros(l3, dtype=float)
        theta4 = np.zeros(l4, dtype=float)
        sign3 = -1.0 if float(theta_stage2_end) < 0.0 else 1.0
        sign4 = sign3 if abs(float(theta_stage4_end)) < 1e-6 else (1.0 if float(theta_stage4_end) >= 0.0 else -1.0)
        if l3 > 0:
            u3_theta = np.linspace(0.0, 1.0, l3, endpoint=False)
            half_wave3 = np.maximum(np.sin(2.35 * np.pi * u3_theta - 0.5 * np.pi + 0.20 * latents["phase"]), 0.0)
            margin3 = 0.62 * self.orient_err_max_stage3 * half_wave3 - 0.18 * self.orient_err_max_stage3
            abs_theta3 = np.clip(self.orient_err_max_stage3 - margin3, 0.0, 0.96 * self.orient_err_max_stage3)
            theta3 = sign3 * self._smooth_trace(abs_theta3, kernel_size=3)
        if l4 > 0:
            u4_theta = np.linspace(0.0, 1.0, l4, endpoint=True)
            half_wave4 = np.maximum(np.sin(1.95 * np.pi * u4_theta - 0.5 * np.pi + 0.16 * latents["phase"]), 0.0)
            margin4 = 0.58 * self.orient_err_max_stage4 * half_wave4 - 0.16 * self.orient_err_max_stage4
            abs_theta4 = np.clip(self.orient_err_max_stage4 - margin4, 0.0, 0.96 * self.orient_err_max_stage4)
            theta4 = sign4 * self._smooth_trace(abs_theta4, kernel_size=3)
        theta = np.concatenate([theta1, theta2, theta3, theta4])
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 - 1, half_window=self.transition_half_window).ravel()
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 + l2 - 1, half_window=self.transition_half_window).ravel()
        theta = self._blend_segment_boundary(theta[:, None], boundary=l1 + l2 + l3 - 1, half_window=self.transition_half_window).ravel()

        theta_noise_scale = np.take(np.array([1.0, 0.30, 0.08, 0.05], dtype=float), labels)
        theta += self._smooth_noise(rng, len(theta), 0.28 * self.noise_misc, kernel_size=11) * theta_noise_scale
        theta += latents["theta_wobble"] * np.sin(np.linspace(0.0, 4.5 * np.pi, len(theta)) + latents["phase"]) * np.r_[
            np.linspace(0.3, 1.0, l1 + l2),
            np.linspace(0.18, 0.08, l3 + l4),
        ] * theta_noise_scale
        theta[l1 + l2:l1 + l2 + l3] = np.clip(
            theta[l1 + l2:l1 + l2 + l3],
            -0.95 * self.orient_err_max_stage3,
            0.95 * self.orient_err_max_stage3,
        )
        theta[l1 + l2 + l3:] = np.clip(
            theta[l1 + l2 + l3:],
            -0.95 * self.orient_err_max_stage4,
            0.95 * self.orient_err_max_stage4,
        )
        for boundary in (l1 - 1, l1 + l2 - 1, l1 + l2 + l3 - 1):
            theta = self._blend_segment_boundary(theta[:, None], boundary=boundary, half_window=self.transition_half_window).ravel()
        theta[l1 + l2:l1 + l2 + l3] = np.clip(
            theta[l1 + l2:l1 + l2 + l3],
            -0.98 * self.orient_err_max_stage3,
            0.98 * self.orient_err_max_stage3,
        )
        theta[l1 + l2 + l3:] = np.clip(
            theta[l1 + l2 + l3:],
            -0.98 * self.orient_err_max_stage4,
            0.98 * self.orient_err_max_stage4,
        )

        normal_load = self._compute_force_signal(pos, theta, stage3_end_local[0], labels, rng, latents)
        rng = np.random.RandomState(int(seed) + 100003)
        traj4 = self._lift_planar_demo_to_4d(pos, theta, labels, rng)
        return traj4, np.asarray(labels, dtype=int), normal_load

    def rollout_demo(self, scene, seed=None, rng=None, backend=None, **kwargs):
        local_seed = int(seed) if seed is not None else int((scene or {}).get("rollout_seed", 0))
        traj4, labels, normal_load = self.generate_demo(seed=local_seed)
        cutpoints = np.where(np.diff(labels) != 0)[0].astype(int)
        active_backend = str(backend or self.rollout_backend).lower()
        if active_backend == "pybullet":
            execution_normal_load, load_noise = self.apply_execution_normal_load_noise(
                normal_load,
                noise_std=float(kwargs.get("execution_normal_load_noise_std", 0.0)),
                noise_smooth=float(kwargs.get("execution_normal_load_noise_smooth", 0.85)),
                seed=kwargs.get("execution_normal_load_noise_seed", kwargs.get("execution_noise_seed", None)),
            )
            sim = simulate_s4_demo_from_reference(
                self,
                scene=scene,
                reference_traj=traj4,
                true_cutpoints=cutpoints,
                gui=int(kwargs.get("gui", 0)),
                video_path=kwargs.get("video_path"),
                fps=float(kwargs.get("fps", 15.0)),
                width=kwargs.get("width"),
                height=kwargs.get("height"),
                render_frame_stride=int(kwargs.get("render_frame_stride", 1)),
                video_end_hold_seconds=float(kwargs.get("video_end_hold_seconds", 2.0)),
                realtime=bool(kwargs.get("realtime", False)),
                gui_hold_seconds=float(kwargs.get("gui_hold_seconds", 0.0)),
                normal_load_trace=execution_normal_load,
                visualize_normal_load=bool(kwargs.get("visualize_normal_load", self.pybullet_visualize_normal_load)),
                feature_overlay=bool(kwargs.get("feature_overlay", True)),
                feature_overlay_title=kwargs.get("feature_overlay_title", None),
                save_frame_indices=kwargs.get("save_frame_indices", None),
                save_frame_dir=kwargs.get("save_frame_dir", None),
                save_frame_prefix=str(kwargs.get("save_frame_prefix", "s4_demo")),
            )
            sim["planned_trajectory"] = traj4
            sim["normal_load_trace"] = execution_normal_load
            sim["planned_normal_load_trace"] = normal_load
            sim["execution_normal_load_noise"] = load_noise
            sim["true_labels"] = labels
            return sim
        return {
            "trajectory": traj4,
            "planned_trajectory": traj4,
            "true_cutpoints": cutpoints,
            "true_labels": labels,
            "normal_load_trace": normal_load,
        }

    def plan_episode_from_constraints(self, scene, constraint_values: dict, *, seed: int = 0, stage_lengths=None, speed_safety: float = 1.0):
        lengths = [int(x) for x in self.seg_lengths]
        for key, value in dict(stage_lengths or {}).items():
            text = str(key).strip().lower()
            if text.startswith("stage"):
                idx = int(text.replace("stage", "")) - 1
            elif text.startswith("s"):
                idx = int(text.replace("s", "")) - 1
            else:
                idx = int(text)
            if 0 <= idx < len(lengths):
                lengths[idx] = max(int(value), 3)
        l1, l2, l3, l4 = lengths

        def cv(key: str, default: float) -> float:
            value = dict(constraint_values or {}).get(key)
            if value is None or not np.isfinite(float(value)):
                return float(default)
            return float(value)

        surf2 = cv("s2:surf_dist", float(self.true_constraints.get("surface_target", 0.0)))
        surf3 = cv("s3:surf_dist", surf2)
        surf4 = cv("s4:surf_dist", surf3)
        center2 = cv("s2:centerline_dist", float(self.clearance_target))
        center3 = cv("s3:centerline_dist", center2)
        center4 = cv("s4:centerline_dist", center3)
        theta2 = cv("s2:orient_err", float(self.theta_stage2_end))
        theta3 = cv("s3:orient_err", theta2)
        theta4 = cv("s4:orient_err", theta3)
        v2 = max(cv("s2:speed", float(self.v_align_max)) * float(speed_safety), 1e-5)
        v3 = max(cv("s3:speed", float(self.v_insert_max)) * float(speed_safety), 1e-5)
        v4 = max(cv("s4:speed", float(self.v_seat_max)) * float(speed_safety), 1e-5)

        rail_total = float(self.rail_total_length())
        s4 = rail_total
        s3 = max(0.0, s4 - v4 * float(self.dt) * max(l4 - 1, 1))
        s2 = max(0.0, s3 - v3 * float(self.dt) * max(l3, 1))
        s1 = max(0.0, s2 - v2 * float(self.dt) * max(l2, 1))

        start_y = 0.75 * float(self.slot_half_width)
        w0 = np.asarray([float(self.start[0]), start_y, float(self.start[1]), float(self.theta_start)], dtype=float)

        def line(a, b, n, *, endpoint=False):
            return np.linspace(np.asarray(a, dtype=float), np.asarray(b, dtype=float), int(n), endpoint=bool(endpoint))

        def rail_segment(sa, sb, n, center, surf, theta_err, *, endpoint=False):
            ss = np.linspace(float(sa), float(sb), int(n), endpoint=bool(endpoint))
            points, _tangents, normals, angles = self.rail_pose_at_s(ss)
            traj = np.zeros((int(n), 4), dtype=float)
            traj[:, :2] = points + normals * float(center)
            traj[:, 2] = self.surface_height(traj[:, :2]) + float(surf)
            traj[:, 3] = angles + float(theta_err)
            return traj

        w1 = rail_segment(s1, s1, 1, center2, surf2, float(self.theta_stage1_end), endpoint=True)[0]
        seg1 = line(w0, w1, l1, endpoint=False)
        seg2 = rail_segment(s1, s2, l2, center2, surf2, theta2, endpoint=False)
        seg3 = rail_segment(s2, s3, l3, center3, surf3, theta3, endpoint=False)
        seg4 = rail_segment(s3, s4, l4, center4, surf4, theta4, endpoint=True)
        traj = np.vstack([seg1, seg2, seg3, seg4])
        labels = np.repeat(np.arange(4), [l1, l2, l3, l4])
        cutpoints = np.where(np.diff(labels) != 0)[0].astype(int)

        normal_load = np.zeros(len(traj), dtype=float)
        for stage_idx, key in [(1, "s2:normal_load"), (2, "s3:normal_load"), (3, "s4:normal_load")]:
            normal_load[labels == int(stage_idx)] = max(cv(key, float(self.normal_load_min)), 0.0)

        return {
            "trajectory": traj,
            "planned_trajectory": traj,
            "true_cutpoints": cutpoints,
            "true_labels": labels,
            "normal_load_trace": normal_load,
            "constraint_values": dict(constraint_values or {}),
            "stage_lengths": {"stage1": int(l1), "stage2": int(l2), "stage3": int(l3), "stage4": int(l4)},
            "planner": "s4_clean_geometric_insert_planner",
            "scene": dict(scene or {}),
            "seed": int(seed),
        }

    def execute_plan_pybullet(self, scene, planned_episode, **kwargs):
        traj = np.asarray(planned_episode["trajectory"], dtype=float)
        cutpoints = np.asarray(planned_episode.get("true_cutpoints", []), dtype=int)
        normal_load = np.asarray(planned_episode.get("normal_load_trace", np.zeros(len(traj))), dtype=float)
        execution_normal_load, load_noise = self.apply_execution_normal_load_noise(
            normal_load,
            noise_std=float(kwargs.get("execution_normal_load_noise_std", 0.0)),
            noise_smooth=float(kwargs.get("execution_normal_load_noise_smooth", 0.85)),
            seed=kwargs.get("execution_normal_load_noise_seed", kwargs.get("execution_noise_seed", None)),
        )
        sim = simulate_s4_demo_from_reference(
            self,
            scene=scene,
            reference_traj=traj,
            true_cutpoints=cutpoints,
            gui=int(kwargs.get("gui", 0)),
            video_path=kwargs.get("video_path"),
            fps=float(kwargs.get("fps", 15.0)),
            width=kwargs.get("width"),
            height=kwargs.get("height"),
            render_frame_stride=int(kwargs.get("render_frame_stride", 1)),
            video_end_hold_seconds=float(kwargs.get("video_end_hold_seconds", 2.0)),
            realtime=bool(kwargs.get("realtime", False)),
            gui_hold_seconds=float(kwargs.get("gui_hold_seconds", 0.0)),
            normal_load_trace=execution_normal_load,
            visualize_normal_load=bool(kwargs.get("visualize_normal_load", self.pybullet_visualize_normal_load)),
            feature_overlay=bool(kwargs.get("feature_overlay", True)),
            feature_overlay_title=kwargs.get("feature_overlay_title", None),
            execution_joint_noise_std=float(kwargs.get("execution_joint_noise_std", 0.0)),
            execution_joint_noise_smooth=float(kwargs.get("execution_joint_noise_smooth", 0.90)),
            execution_noise_seed=kwargs.get("execution_noise_seed", None),
            save_frame_indices=kwargs.get("save_frame_indices", None),
            save_frame_dir=kwargs.get("save_frame_dir", None),
            save_frame_prefix=str(kwargs.get("save_frame_prefix", "s4_planned")),
        )
        sim["planned_trajectory"] = traj
        sim["normal_load_trace"] = execution_normal_load
        sim["planned_normal_load_trace"] = normal_load
        sim["execution_normal_load_noise"] = load_noise
        sim["true_labels"] = np.asarray(planned_episode.get("true_labels", []), dtype=int)
        sim["planner"] = str(planned_episode.get("planner", "s4_clean_geometric_insert_planner"))
        sim["planned_constraint_values"] = dict(planned_episode.get("constraint_values", {}))
        return sim

    def compute_observation(self, latent_rollout, scene, backend=None):
        traj = np.asarray(latent_rollout["trajectory"], dtype=float)
        load = latent_rollout.get("normal_load_trace")
        if load is not None:
            self.register_normal_load_trace(traj, np.asarray(load, dtype=float))
        features = np.asarray(self.compute_all_features_matrix(traj), dtype=float)
        out = {
            "trajectory": traj,
            "features": features,
            "true_cutpoints": np.asarray(latent_rollout.get("true_cutpoints", []), dtype=int),
            "true_labels": np.asarray(latent_rollout.get("true_labels", []), dtype=int),
            "feature_schema": self.get_feature_schema(),
            "observation_spec": self.get_observation_spec(),
            "scene": dict(scene or {}),
        }
        for key in (
            "planned_trajectory",
            "reference_trajectory",
            "normal_load_trace",
            "joint_positions",
            "joint_velocities",
            "joint_position_commands",
            "joint_position_commands_nominal",
            "execution_joint_noise",
            "planned_normal_load_trace",
            "execution_normal_load_noise",
            "reference_trajectory_world",
            "realized_ee_trajectory_world",
            "realized_ee_quaternions",
            "target_quaternions",
            "ik_position_error_world",
            "robot_backend",
            "sim_dt",
            "steps_per_sample",
        ):
            if key in latent_rollout:
                out[key] = latent_rollout[key]
        return out

    def compute_all_features_matrix(self, traj: np.ndarray, feat_ids=None) -> np.ndarray:
        traj = np.asarray(traj, dtype=float)
        T = traj.shape[0]
        xyz = traj[:, :3]
        theta = traj[:, 3]
        vel = np.zeros_like(xyz)
        omega = np.zeros(T, dtype=float)
        if T > 1:
            vel[:-1] = np.diff(xyz, axis=0) / max(self.dt, 1e-12)
            vel[-1] = vel[-2]
            dtheta = self._wrap_to_pi(np.diff(theta)) / max(self.dt, 1e-12)
            omega[:-1] = dtheta
            omega[-1] = omega[-2]
        speed = np.linalg.norm(vel, axis=1)
        angular_speed = np.abs(omega)
        surf_dist = np.abs(xyz[:, 2] - self.surface_height(xyz[:, :2]))
        rail_proj = self.project_to_rail(xyz[:, :2])
        centerline_dist = np.asarray(rail_proj["dist"], dtype=float)
        orient_err = np.abs(self._wrap_to_pi(theta - np.asarray(rail_proj["angle"], dtype=float)))
        normal_load = self._lookup_cached_normal_load_trace(traj)
        if normal_load is None:
            normal_load = np.where(surf_dist < 0.004, self.normal_load_min, 0.0)
        start_dist = np.linalg.norm(xyz - xyz[0][None, :], axis=1)
        insertion_err = np.maximum(float(self.slot_x) - xyz[:, 0], 0.0)
        noise = 0.35 * np.sin(0.19 * np.arange(T)) + 0.15 * np.cos(0.07 * np.arange(T))
        F = np.c_[
            surf_dist,
            centerline_dist,
            orient_err,
            speed,
            angular_speed,
            normal_load,
            noise,
            start_dist,
            insertion_err,
        ]
        return F if feat_ids is None else F[:, feat_ids]

    def render_episode(self, scene, trajectory, output_path, **kwargs):
        cutpoints = kwargs.get("cutpoints")
        return render_planar_episode(
            trajectory=np.asarray(trajectory, dtype=float)[:, :2],
            output_path=output_path,
            cutpoints=cutpoints,
            title=kwargs.get("title", "S4SlideInsert top view"),
            obstacles=None,
            reference_lines=[{"point": [0.0, self.clearance_target], "direction": [1.0, 0.0], "color": "#64748B"}],
            markers=[{"point": [self.slot_x, self.clearance_target], "color": "#16A34A", "marker": "s", "size": 34}],
            xlabel="x",
            ylabel="y / clearance",
            equal_aspect=True,
        )


def load_S4SlideInsert(n_demos: int = 10, seed: int = 123, env_kwargs=None, demo_kwargs=None, **extra_env_kwargs):
    env_cfg = dict(env_kwargs or {})
    env_cfg.update(extra_env_kwargs)
    env = S4SlideInsertEnv(**env_cfg)
    run_kwargs = dict(demo_kwargs or {})
    demos = []
    labels = []
    cutpoints = []
    scene_specs = []
    for i in range(int(n_demos)):
        scene = env.sample_scene()
        scene["demo_index"] = int(i)
        latent = env.rollout_demo(scene, seed=int(seed) + int(i), **run_kwargs)
        observation = env.compute_observation(latent, scene)
        demo = np.asarray(observation["trajectory"], dtype=float)
        demos.append(demo)
        labels.append(np.asarray(observation["true_labels"], dtype=int))
        cutpoints.append(np.asarray(observation["true_cutpoints"], dtype=int))
        scene_specs.append(dict(scene))
    env.demo_subgoals = [np.asarray(x[int(c[1]), :4], dtype=float).copy() for x, c in zip(demos, cutpoints)]
    env.demo_goals = [np.asarray(x[-1, :4], dtype=float).copy() for x in demos]
    env.demo_stage_lengths = [np.bincount(np.asarray(z, dtype=int), minlength=env.n_segments).astype(int) for z in labels]
    env.subgoal = np.mean(np.stack(env.demo_subgoals, axis=0), axis=0)
    env.goal = np.mean(np.stack(env.demo_goals, axis=0), axis=0)
    return TaskBundle(
        name="S4SlideInsert",
        demos=demos,
        env=env,
        true_taus=None,
        true_cutpoints=[np.asarray(c, dtype=int) for c in cutpoints],
        true_labels=labels,
        feature_schema=env.get_feature_schema(),
        true_constraints=env.get_true_constraints(),
        constraint_specs=env.get_constraint_specs(),
        meta={
            "seed": int(seed),
            "cutpoints": [c.tolist() for c in cutpoints],
            "task_name": "S4SlideInsert",
            "scene_specs": scene_specs,
            "observation_specs": env.get_observation_spec(),
            "render_camera_presets": env.get_render_camera_presets(),
            "asset_handles": env.get_asset_handles(),
        },
    )
