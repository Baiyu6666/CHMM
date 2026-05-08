from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Iterable, Sequence
import xml.etree.ElementTree as ET

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

try:
    import pybullet as p
    import pybullet_data
except ModuleNotFoundError:
    p = None
    pybullet_data = None


STAGE_COLORS = ["#D55E00", "#0072B2", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _save_figure(fig, path: str | Path, *, dpi: int = 220) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    if plt is not None:
        plt.close(fig)
    return out_path


def stage_segments(length: int, cutpoints: Sequence[int] | None = None) -> list[tuple[int, int]]:
    T = max(int(length), 0)
    if T <= 0:
        return []
    if cutpoints is None:
        return [(0, T - 1)]
    cuts = np.asarray(cutpoints, dtype=int).reshape(-1)
    if cuts.size == 0:
        return [(0, T - 1)]
    cuts = np.sort(cuts[(cuts >= 0) & (cuts < T - 1)])
    ends = cuts.tolist() + [T - 1]
    starts = [0] + [int(v) + 1 for v in ends[:-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _require_matplotlib() -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for env.render_episode().")


def _require_pybullet() -> None:
    if p is None or pybullet_data is None:
        raise RuntimeError("pybullet is required for env.render_episode(..., backend='pybullet').")


class _FFmpegVideoWriter:
    def __init__(self, *, out_path: str | Path, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg binary not found in PATH")
        self.out_path = str(Path(out_path).resolve())
        self.width = int(width)
        self.height = int(height)
        self.fps = float(max(float(fps), 0.1))
        Path(self.out_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{self.fps:.6f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            self.out_path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def append_data(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        arr = np.asarray(frame, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if arr.shape != expected:
            raise ValueError(f"video frame has shape {arr.shape}, expected {expected}")
        self.proc.stdin.write(arr.tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        self.proc.wait(timeout=30)
        if self.proc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg exited with code {self.proc.returncode}")


def render_planar_episode(
    *,
    trajectory: np.ndarray,
    output_path: str | Path,
    cutpoints: Sequence[int] | None = None,
    title: str | None = None,
    obstacles: Iterable[dict] | None = None,
    reference_lines: Iterable[dict] | None = None,
    markers: Iterable[dict] | None = None,
    xlabel: str = "x",
    ylabel: str = "y",
    equal_aspect: bool = True,
) -> Path:
    _require_matplotlib()
    pts = np.asarray(trajectory, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("trajectory must have shape (T, 2+) for planar rendering.")

    fig, ax = plt.subplots(figsize=(4.6, 3.6), constrained_layout=False)

    if obstacles is not None:
        for obs in obstacles:
            center = np.asarray(obs.get("center", [0.0, 0.0]), dtype=float).reshape(2)
            radius = float(obs.get("radius", 0.1))
            facecolor = str(obs.get("facecolor", "#CBD5E1"))
            edgecolor = str(obs.get("edgecolor", "#475569"))
            alpha = float(obs.get("alpha", 0.32))
            circle = plt.Circle(center, radius, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, linewidth=1.2)
            ax.add_patch(circle)

    if reference_lines is not None:
        for line in reference_lines:
            point = np.asarray(line.get("point", [0.0, 0.0]), dtype=float).reshape(2)
            direction = np.asarray(line.get("direction", [1.0, 0.0]), dtype=float).reshape(2)
            span = float(line.get("span", 4.0))
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            direction = direction / norm
            endpoints = np.vstack([point - span * direction, point + span * direction])
            ax.plot(
                endpoints[:, 0],
                endpoints[:, 1],
                linestyle=str(line.get("linestyle", "--")),
                linewidth=float(line.get("linewidth", 1.0)),
                color=str(line.get("color", "#64748B")),
                alpha=float(line.get("alpha", 0.7)),
            )

    segments = stage_segments(len(pts), cutpoints=cutpoints)
    for stage_idx, (start, end) in enumerate(segments):
        seg = pts[start : end + 1, :2]
        color = STAGE_COLORS[stage_idx % len(STAGE_COLORS)]
        ax.plot(seg[:, 0], seg[:, 1], color=color, linewidth=1.8, alpha=0.96)
        ax.scatter(seg[:, 0], seg[:, 1], color=color, s=10, alpha=0.24)

    ax.scatter(pts[0, 0], pts[0, 1], color="#111827", marker="o", s=24, zorder=6)
    ax.scatter(pts[-1, 0], pts[-1, 1], color="#111827", marker="s", s=24, zorder=6)

    if cutpoints is not None:
        for cp in np.asarray(cutpoints, dtype=int).reshape(-1):
            if 0 <= int(cp) < len(pts):
                ax.scatter(pts[int(cp), 0], pts[int(cp), 1], color="#111827", marker="x", s=36, linewidths=1.4, zorder=7)

    if markers is not None:
        for marker in markers:
            point = np.asarray(marker.get("point", [0.0, 0.0]), dtype=float).reshape(2)
            ax.scatter(
                point[0],
                point[1],
                color=str(marker.get("color", "#1D4ED8")),
                marker=str(marker.get("marker", "o")),
                s=float(marker.get("size", 26.0)),
                alpha=float(marker.get("alpha", 0.95)),
                zorder=8,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(str(title), fontsize=10)
    ax.grid(alpha=0.18)
    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")

    mins = np.min(pts[:, :2], axis=0)
    maxs = np.max(pts[:, :2], axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    margin = 0.18 * span
    ax.set_xlim(float(mins[0] - margin[0]), float(maxs[0] + margin[0]))
    ax.set_ylim(float(mins[1] - margin[1]), float(maxs[1] + margin[1]))

    fig.tight_layout(pad=0.25)
    return _save_figure(fig, output_path, dpi=220)


def render_sphere_episode(
    *,
    trajectory: np.ndarray,
    output_path: str | Path,
    sphere_center: Sequence[float],
    sphere_radius: float,
    cutpoints: Sequence[int] | None = None,
    title: str | None = None,
    elev: float = 24.0,
    azim: float = 38.0,
) -> Path:
    _require_matplotlib()
    pts = np.asarray(trajectory, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("trajectory must have shape (T, 3+) for sphere rendering.")

    fig = plt.figure(figsize=(4.6, 3.9), constrained_layout=False)
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    center = np.asarray(sphere_center, dtype=float).reshape(3)
    radius = float(sphere_radius)
    th = np.linspace(0.0, 2.0 * np.pi, 28)
    ph = np.linspace(0.0, np.pi, 18)
    th, ph = np.meshgrid(th, ph)
    xx = center[0] + radius * np.cos(th) * np.sin(ph)
    yy = center[1] + radius * np.sin(th) * np.sin(ph)
    zz = center[2] + radius * np.cos(ph)
    ax.plot_wireframe(xx, yy, zz, color="#94A3B8", alpha=0.26, linewidth=0.6, rstride=1, cstride=1)

    segments = stage_segments(len(pts), cutpoints=cutpoints)
    for stage_idx, (start, end) in enumerate(segments):
        seg = pts[start : end + 1, :3]
        color = STAGE_COLORS[stage_idx % len(STAGE_COLORS)]
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=1.8, alpha=0.98)
        ax.scatter(seg[:, 0], seg[:, 1], seg[:, 2], color=color, s=8.0, alpha=0.22, depthshade=False)

    ax.scatter(pts[0, 0], pts[0, 1], pts[0, 2], color="#111827", marker="o", s=26, depthshade=False)
    ax.scatter(pts[-1, 0], pts[-1, 1], pts[-1, 2], color="#111827", marker="s", s=26, depthshade=False)
    if cutpoints is not None:
        for cp in np.asarray(cutpoints, dtype=int).reshape(-1):
            if 0 <= int(cp) < len(pts):
                ax.scatter(
                    pts[int(cp), 0],
                    pts[int(cp), 1],
                    pts[int(cp), 2],
                    color="#111827",
                    marker="x",
                    s=36,
                    linewidths=1.4,
                    depthshade=False,
                )

    corners = np.array(
        [
            center + np.array([sx, sy, sz], dtype=float) * radius
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ],
        dtype=float,
    )
    all_pts = np.vstack([pts[:, :3], corners])
    mins = np.min(all_pts, axis=0)
    maxs = np.max(all_pts, axis=0)
    center_box = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.55 * max(span, 1e-3)
    ax.set_xlim(center_box[0] - half, center_box[0] + half)
    ax.set_ylim(center_box[1] - half, center_box[1] + half)
    ax.set_zlim(center_box[2] - half, center_box[2] + half)
    ax.view_init(elev=float(elev), azim=float(azim))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(str(title), fontsize=10)

    fig.tight_layout(pad=0.25)
    return _save_figure(fig, output_path, dpi=220)


def _normalize3(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _hex_to_rgba(color: str, alpha: float = 1.0) -> tuple[float, float, float, float]:
    text = str(color).lstrip("#")
    if len(text) != 6:
        raise ValueError(f"Expected 6-digit hex color, got '{color}'.")
    rgb = tuple(int(text[i : i + 2], 16) / 255.0 for i in (0, 2, 4))
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(alpha))


def _quat_align_z_to_vec(vec: np.ndarray) -> tuple[float, float, float, float]:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    target = _normalize3(vec)
    dot = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))
    if dot >= 1.0 - 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    if dot <= -1.0 + 1e-8:
        return tuple(p.getQuaternionFromEuler((np.pi, 0.0, 0.0)))
    axis = _normalize3(np.cross(z_axis, target))
    angle = float(np.arccos(dot))
    return tuple(p.getQuaternionFromAxisAngle(axis.tolist(), angle))


def _env_to_world(points: np.ndarray, sphere_center: np.ndarray, center_world: np.ndarray, scale: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return center_world[None, :] + float(scale) * (pts - np.asarray(sphere_center, dtype=float)[None, :])


def _spawn_table(table_top_z: float) -> None:
    half_extents = [0.54, 0.54, 0.028]
    col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    vis_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
        specularColor=[0.10, 0.08, 0.06],
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=[0.0, 0.0, table_top_z - half_extents[2]],
    )
    tex_path = Path(pybullet_data.getDataPath()) / "table" / "table.png"
    if tex_path.exists():
        tex_id = p.loadTexture(str(tex_path))
        p.changeVisualShape(body_id, -1, textureUniqueId=tex_id)

    leg_half = [0.03, 0.03, 0.30]
    leg_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=leg_half)
    leg_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=leg_half,
        rgbaColor=[0.30, 0.27, 0.24, 1.0],
        specularColor=[0.05, 0.05, 0.05],
    )
    for sx in (-0.44, 0.44):
        for sy in (-0.44, 0.44):
            p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=leg_col,
                baseVisualShapeIndex=leg_vis,
                basePosition=[sx, sy, table_top_z - 2.0 * half_extents[2] - leg_half[2]],
            )


def _spawn_sphere(center_world: np.ndarray, radius_world: float) -> None:
    data_root = Path(pybullet_data.getDataPath())
    sphere_mesh = data_root / "sphere_smooth.obj"
    col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius_world)
    outer_vis = p.createVisualShape(
        p.GEOM_MESH,
        fileName=str(sphere_mesh),
        meshScale=[radius_world, radius_world, radius_world],
        rgbaColor=[0.90, 0.95, 0.99, 0.40],
        specularColor=[0.98, 0.99, 1.00],
    )
    inner_vis = p.createVisualShape(
        p.GEOM_MESH,
        fileName=str(sphere_mesh),
        meshScale=[0.986 * radius_world, 0.986 * radius_world, 0.986 * radius_world],
        rgbaColor=[0.82, 0.88, 0.94, 0.13],
        specularColor=[0.55, 0.58, 0.62],
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=outer_vis,
        basePosition=center_world.tolist(),
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=inner_vis,
        basePosition=center_world.tolist(),
    )


def _spawn_marker(pos_world: np.ndarray, radius: float, color: tuple[float, float, float, float]) -> None:
    vis_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=list(color),
        specularColor=[0.2, 0.2, 0.2],
    )
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=vis_id,
        basePosition=np.asarray(pos_world, dtype=float).tolist(),
    )


def _spawn_oriented_cylinder(
    pos_world: np.ndarray,
    axis_world: np.ndarray,
    length: float,
    radius: float,
    color: tuple[float, float, float, float],
) -> None:
    vis_id = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(radius),
        length=float(length),
        rgbaColor=list(color),
        specularColor=[0.20, 0.20, 0.20],
    )
    orn = _quat_align_z_to_vec(axis_world)
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=vis_id,
        basePosition=np.asarray(pos_world, dtype=float).tolist(),
        baseOrientation=orn,
    )


def _spawn_probe_pose(
    pos_world: np.ndarray,
    axis_world: np.ndarray,
    shaft_len: float = 0.080,
    shaft_radius: float = 0.0045,
    tip_len: float = 0.020,
    tip_radius: float = 0.0075,
) -> None:
    axis = _normalize3(axis_world)
    shaft_center = np.asarray(pos_world, dtype=float) - 0.5 * float(shaft_len) * axis
    _spawn_oriented_cylinder(
        pos_world=shaft_center,
        axis_world=axis,
        length=shaft_len,
        radius=shaft_radius,
        color=(0.18, 0.20, 0.24, 1.0),
    )
    collar_center = np.asarray(pos_world, dtype=float) - 0.12 * float(shaft_len) * axis
    _spawn_oriented_cylinder(
        pos_world=collar_center,
        axis_world=axis,
        length=0.018,
        radius=0.0065,
        color=(0.12, 0.46, 0.84, 1.0),
    )
    tip_center = np.asarray(pos_world, dtype=float) + 0.5 * float(tip_len) * axis
    _spawn_oriented_cylinder(
        pos_world=tip_center,
        axis_world=axis,
        length=tip_len,
        radius=tip_radius,
        color=(0.88, 0.64, 0.18, 1.0),
    )


def _spawn_capsule_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    radius: float,
    color: tuple[float, float, float, float],
) -> None:
    vec = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    seg_len = float(np.linalg.norm(vec))
    if seg_len <= 1e-8:
        return
    cyl_len = max(seg_len - 2.0 * float(radius), 1e-4)
    vis_id = p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=float(radius),
        length=cyl_len,
        rgbaColor=list(color),
        specularColor=[0.18, 0.18, 0.18],
    )
    midpoint = 0.5 * (np.asarray(p0, dtype=float) + np.asarray(p1, dtype=float))
    orn = _quat_align_z_to_vec(vec)
    p.createMultiBody(
        baseMass=0.0,
        baseVisualShapeIndex=vis_id,
        basePosition=midpoint.tolist(),
        baseOrientation=orn,
    )


def _render_rgb(
    *,
    yaw_deg: float,
    target: np.ndarray,
    distance: float,
    width: int,
    height: int,
    pitch_deg: float = -23.0,
) -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=np.asarray(target, dtype=float).tolist(),
        distance=float(distance),
        yaw=float(yaw_deg),
        pitch=float(pitch_deg),
        roll=0.0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=37.0,
        aspect=float(width) / float(height),
        nearVal=0.05,
        farVal=8.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_TINY_RENDERER,
        lightDirection=[1.8, -1.1, 2.8],
        shadow=1,
    )
    rgba = np.asarray(rgba, dtype=np.uint8).reshape(height, width, 4)
    return rgba[:, :, :3]


def _load_ur5_render_robot(
    *,
    urdf_path: str | None,
    base_xyz: Sequence[float],
    base_rpy: Sequence[float],
    hide_link_geometry_patterns: Sequence[str] | None = None,
    suppress_urdf_warnings: bool = True,
) -> tuple[int, list[int], str | None]:
    from .s5_pybullet_backend import _DEFAULT_UR5_URDF, _make_pybullet_friendly_urdf, _suppress_native_output

    path = str(urdf_path or _DEFAULT_UR5_URDF)
    if not os.path.exists(path):
        raise RuntimeError(f"UR5 URDF not found: {path}")
    load_path = path
    patched_path = None
    with open(path, "r", encoding="utf-8") as f:
        if "package://" in f.read():
            patched_path = _make_pybullet_friendly_urdf(path)
            load_path = patched_path
    patterns = [str(v).lower() for v in (hide_link_geometry_patterns or []) if str(v).strip()]
    if patterns:
        stripped_path = _make_urdf_without_link_geometry(load_path, patterns)
        if patched_path:
            try:
                os.remove(patched_path)
            except OSError:
                pass
        patched_path = stripped_path
        load_path = stripped_path
    with _suppress_native_output(bool(suppress_urdf_warnings)):
        robot_id = int(
            p.loadURDF(
                load_path,
                basePosition=np.asarray(base_xyz, dtype=float).reshape(3).tolist(),
                baseOrientation=p.getQuaternionFromEuler(np.asarray(base_rpy, dtype=float).reshape(3).tolist()),
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE,
            )
        )
    arm_joint_indices: list[int] = []
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        if int(info[2]) == p.JOINT_REVOLUTE:
            arm_joint_indices.append(int(j))
    if len(arm_joint_indices) < 6:
        raise RuntimeError(f"UR5 model has fewer than 6 revolute joints: {len(arm_joint_indices)}")
    return robot_id, arm_joint_indices[:6], patched_path


def _make_urdf_without_link_geometry(urdf_path: str, name_patterns: Sequence[str]) -> str:
    patterns = [str(v).lower() for v in name_patterns if str(v).strip()]
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    for link in root.findall("link"):
        name = str(link.attrib.get("name", "")).lower()
        if not any(pattern in name for pattern in patterns):
            continue
        for child in list(link):
            if child.tag in {"visual", "collision"}:
                link.remove(child)
    fd, tmp = tempfile.mkstemp(prefix="s5_ur5_render_hidden_", suffix=".urdf")
    os.close(fd)
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    return tmp


def _hide_robot_links_by_name(robot_id: int, name_patterns: Sequence[str]) -> None:
    patterns = [str(v).lower() for v in name_patterns]
    if not patterns:
        return
    for j in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, j)
        link_name = info[12].decode("utf-8", errors="ignore").lower()
        if not any(pattern in link_name for pattern in patterns):
            continue
        try:
            p.changeVisualShape(robot_id, j, rgbaColor=[0.0, 0.0, 0.0, 0.0])
        except Exception:
            pass
        try:
            p.setCollisionFilterGroupMask(robot_id, j, collisionFilterGroup=0, collisionFilterMask=0)
        except Exception:
            pass


def _set_robot_q(robot_id: int, joint_indices: Sequence[int], q: np.ndarray) -> None:
    q_arr = np.asarray(q, dtype=float).reshape(-1)
    if q_arr.size < len(joint_indices):
        raise ValueError(f"joint_positions row has {q_arr.size} values, expected at least {len(joint_indices)}")
    for i, joint_idx in enumerate(joint_indices):
        p.resetJointState(robot_id, int(joint_idx), float(q_arr[i]), targetVelocity=0.0)


def _spawn_current_marker(radius: float, color: tuple[float, float, float, float]) -> int:
    vis_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=float(radius),
        rgbaColor=list(color),
        specularColor=[0.15, 0.15, 0.15],
    )
    return int(p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis_id, basePosition=[0.0, 0.0, 0.0]))


def _spawn_tool_bar(length: float, radius: float, color: tuple[float, float, float, float]) -> int:
    vis_id = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(radius),
        length=float(length),
        rgbaColor=list(color),
        specularColor=[0.18, 0.18, 0.18],
    )
    return int(p.createMultiBody(baseMass=0.0, baseVisualShapeIndex=vis_id, basePosition=[0.0, 0.0, 0.0]))


def _set_tool_bar_pose(body_id: int, tip_pos_world: np.ndarray, axis_world: np.ndarray, length: float) -> None:
    axis = _normalize3(axis_world)
    center = np.asarray(tip_pos_world, dtype=float).reshape(3) + 0.5 * float(length) * axis
    p.resetBasePositionAndOrientation(
        int(body_id),
        center.tolist(),
        _quat_align_z_to_vec(axis),
    )


def _space_was_triggered() -> bool:
    events = p.getKeyboardEvents()
    state = int(events.get(ord(" "), 0))
    return bool(state & p.KEY_WAS_TRIGGERED)


def render_s5_pybullet_demo_video(
    *,
    trajectory: np.ndarray,
    output_path: str | Path | None,
    sphere_center: Sequence[float],
    sphere_radius: float,
    cutpoints: Sequence[int] | None = None,
    tool_axis: np.ndarray | None = None,
    joint_positions: np.ndarray | None = None,
    title: str | None = None,
    center_world: Sequence[float] = (0.0, 0.0, 0.98),
    world_scale: float = 1.0,
    urdf_path: str | None = None,
    ur5_base_xyz: Sequence[float] = (0.0, 0.0, 0.0),
    ur5_base_rpy: Sequence[float] = (0.0, 0.0, 0.0),
    gui: int = 1,
    fps: float = 30.0,
    width: int = 1024,
    height: int = 768,
    render_frame_stride: int = 1,
    realtime: bool = False,
    gui_hold_seconds: float = 0.0,
    camera_yaw: float = 90.0,
    camera_pitch: float = -34.0,
    camera_distance: float = 1.62,
    camera_target: Sequence[float] | None = None,
    camera_fov: float = 42.0,
    tube_radius: float = 0.0055,
    trace_stride: int = 1,
    draw_stage_trace: bool = True,
    hide_gripper: bool = True,
    draw_tool_bar: bool = False,
    tool_bar_length: float = 0.105,
    tool_bar_radius: float = 0.005,
    suppress_urdf_warnings: bool = True,
    connect_client: bool = True,
) -> dict:
    _require_pybullet()
    pts = np.asarray(trajectory, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("trajectory must have shape (T, 3+) for S5 pybullet video rendering.")
    if len(pts) < 2:
        raise ValueError("trajectory must contain at least two points.")
    if tool_axis is None:
        axis = np.zeros((len(pts), 3), dtype=float)
        axis[:, 2] = 1.0
    else:
        axis = np.asarray(tool_axis, dtype=float)
        if axis.shape != pts[:, :3].shape:
            raise ValueError("tool_axis must have the same shape as trajectory[:, :3].")
    axis = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)

    q_path = None if joint_positions is None else np.asarray(joint_positions, dtype=float)
    if q_path is not None and (q_path.ndim != 2 or q_path.shape[0] != len(pts) or q_path.shape[1] < 6):
        raise ValueError("joint_positions must have shape (T, >=6), matching trajectory length.")

    gui_mode = int(gui)
    if gui_mode not in {0, 1, 2}:
        raise ValueError("gui must be one of 0, 1, 2.")
    save_video = gui_mode == 1 and output_path is not None
    use_gui = gui_mode == 2
    if gui_mode == 1 and output_path is None:
        raise ValueError("output_path is required when gui=1.")

    center_world = np.asarray(center_world, dtype=float).reshape(3)
    sphere_center = np.asarray(sphere_center, dtype=float).reshape(3)
    radius_world = float(world_scale) * float(sphere_radius)
    table_top_z = float(center_world[2] - 1.03 * radius_world)
    traj_world = _env_to_world(pts[:, :3], sphere_center=sphere_center, center_world=center_world, scale=world_scale)
    bounds = stage_segments(len(pts), cutpoints=cutpoints)
    trace_stride = int(max(1, trace_stride))
    render_frame_stride = int(max(1, render_frame_stride))

    client = p.connect(p.GUI if use_gui else p.DIRECT) if bool(connect_client) else None
    writer = None
    patched_urdf = None
    frames_written = 0
    t0 = time.time()
    try:
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0, 0.0, -9.81)
        if not use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        _spawn_table(table_top_z=table_top_z)
        _spawn_sphere(center_world=center_world, radius_world=radius_world)

        if bool(draw_stage_trace):
            for stage_idx, (start, end) in enumerate(bounds):
                idx = np.arange(int(start), int(end) + 1, trace_stride, dtype=int)
                if int(idx[-1]) != int(end):
                    idx = np.concatenate([idx, np.asarray([int(end)], dtype=int)])
                seg = traj_world[idx]
                color = _hex_to_rgba(STAGE_COLORS[stage_idx % len(STAGE_COLORS)], alpha=0.95)
                for a, b in zip(seg[:-1], seg[1:]):
                    _spawn_capsule_segment(a, b, radius=float(tube_radius), color=color)

        _spawn_marker(traj_world[0], radius=0.014, color=(0.10, 0.65, 0.25, 1.0))
        _spawn_marker(traj_world[-1], radius=0.014, color=(0.86, 0.18, 0.18, 1.0))
        if cutpoints is not None:
            for cp in np.asarray(cutpoints, dtype=int).reshape(-1):
                if 0 <= int(cp) < len(traj_world):
                    _spawn_marker(traj_world[int(cp)], radius=0.010, color=(0.08, 0.08, 0.08, 1.0))

        robot_id = None
        joint_indices: list[int] = []
        if q_path is not None:
            hidden_link_patterns: list[str] = []
            if bool(hide_gripper):
                hidden_link_patterns.extend(["gripper", "finger", "palm"])
            if bool(draw_tool_bar):
                hidden_link_patterns.extend(["task_tool", "task_tcp"])
            robot_id, joint_indices, patched_urdf = _load_ur5_render_robot(
                urdf_path=urdf_path,
                base_xyz=ur5_base_xyz,
                base_rpy=ur5_base_rpy,
                hide_link_geometry_patterns=hidden_link_patterns,
                suppress_urdf_warnings=bool(suppress_urdf_warnings),
            )
            if bool(hide_gripper):
                _hide_robot_links_by_name(robot_id, ("gripper", "finger", "palm"))
            _set_robot_q(robot_id, joint_indices, q_path[0])

        current_marker_id = _spawn_current_marker(radius=0.010, color=(0.98, 0.78, 0.16, 1.0))
        tool_bar_id = None
        if bool(draw_tool_bar):
            tool_bar_id = _spawn_tool_bar(
                length=float(tool_bar_length),
                radius=float(tool_bar_radius),
                color=(0.92, 0.50, 0.10, 1.0),
            )
            _set_tool_bar_pose(tool_bar_id, traj_world[0], axis[0], float(tool_bar_length))
        elif q_path is None:
            _spawn_probe_pose(traj_world[0], axis[0], shaft_len=0.080, shaft_radius=0.0045)

        target = (
            np.asarray(camera_target, dtype=float).reshape(3)
            if camera_target is not None
            else center_world + np.array([0.0, 0.0, 0.035], dtype=float)
        )
        p.resetDebugVisualizerCamera(
            cameraDistance=float(camera_distance),
            cameraYaw=float(camera_yaw),
            cameraPitch=float(camera_pitch),
            cameraTargetPosition=target.tolist(),
        )
        if save_video:
            writer = _FFmpegVideoWriter(out_path=Path(output_path), width=int(width), height=int(height), fps=float(fps))

        pause_text_id = None
        if use_gui:
            pause_text_id = p.addUserDebugText(
                "SPACE: pause/resume. At end, press SPACE for next demo.",
                textPosition=(target + np.array([-0.34, -0.30, 0.34], dtype=float)).tolist(),
                textColorRGB=(0.05, 0.05, 0.05),
                textSize=1.15,
                lifeTime=0.0,
            )

        i = 0
        paused = False
        while i < len(pts):
            if use_gui and _space_was_triggered():
                paused = not paused
                if pause_text_id is not None:
                    p.removeUserDebugItem(pause_text_id)
                pause_text_id = p.addUserDebugText(
                    ("Paused. SPACE: resume." if paused else "SPACE: pause/resume. At end, press SPACE for next demo."),
                    textPosition=(target + np.array([-0.34, -0.30, 0.34], dtype=float)).tolist(),
                    textColorRGB=(0.05, 0.05, 0.05),
                    textSize=1.15,
                    lifeTime=0.0,
                )
            if use_gui and paused:
                time.sleep(0.05)
                continue

            if robot_id is not None and q_path is not None:
                _set_robot_q(robot_id, joint_indices, q_path[i])
            p.resetBasePositionAndOrientation(current_marker_id, traj_world[i].tolist(), [0.0, 0.0, 0.0, 1.0])
            if tool_bar_id is not None:
                _set_tool_bar_pose(tool_bar_id, traj_world[i], axis[i], float(tool_bar_length))
            p.stepSimulation()
            write_frame = (i % render_frame_stride == 0) or (i == len(pts) - 1)
            if save_video and writer is not None and write_frame:
                frame = _render_rgb(
                    yaw_deg=float(camera_yaw),
                    target=target,
                    distance=float(camera_distance),
                    width=int(width),
                    height=int(height),
                    pitch_deg=float(camera_pitch),
                )
                if abs(float(camera_fov) - 37.0) > 1e-8:
                    view = p.computeViewMatrixFromYawPitchRoll(
                        cameraTargetPosition=target.tolist(),
                        distance=float(camera_distance),
                        yaw=float(camera_yaw),
                        pitch=float(camera_pitch),
                        roll=0.0,
                        upAxisIndex=2,
                    )
                    proj = p.computeProjectionMatrixFOV(
                        fov=float(camera_fov),
                        aspect=float(width) / float(height),
                        nearVal=0.05,
                        farVal=8.0,
                    )
                    _, _, rgba, _, _ = p.getCameraImage(
                        width=int(width),
                        height=int(height),
                        viewMatrix=view,
                        projectionMatrix=proj,
                        renderer=p.ER_TINY_RENDERER,
                        lightDirection=[1.8, -1.1, 2.8],
                        shadow=1,
                    )
                    frame = np.asarray(rgba, dtype=np.uint8).reshape(int(height), int(width), 4)[:, :, :3]
                writer.append_data(frame)
                frames_written += 1
            i += 1
            if use_gui and bool(realtime):
                time.sleep(1.0 / max(float(fps), 1e-6))
        if use_gui:
            hold_seconds = float(gui_hold_seconds)
            if hold_seconds < 0.0:
                if pause_text_id is not None:
                    p.removeUserDebugItem(pause_text_id)
                p.addUserDebugText(
                    "Demo finished. Press SPACE for next demo.",
                    textPosition=(target + np.array([-0.30, -0.30, 0.34], dtype=float)).tolist(),
                    textColorRGB=(0.05, 0.05, 0.05),
                    textSize=1.2,
                    lifeTime=0.0,
                )
                try:
                    while True:
                        if _space_was_triggered():
                            break
                        time.sleep(0.1)
                except KeyboardInterrupt:
                    pass
            elif hold_seconds > 0.0:
                time.sleep(hold_seconds)
    finally:
        if writer is not None:
            writer.close()
        if bool(connect_client) and client is not None:
            p.disconnect(client)
        if patched_urdf:
            try:
                os.remove(patched_urdf)
            except OSError:
                pass

    return {
        "video_path": None if not save_video else str(Path(output_path).resolve()),
        "frames_written": int(frames_written),
        "source_frames": int(len(pts)),
        "fps": float(fps),
        "gui": int(gui_mode),
        "wall_seconds": float(time.time() - t0),
        "has_robot_joint_playback": bool(q_path is not None),
        "hide_gripper": bool(hide_gripper),
        "draw_tool_bar": bool(draw_tool_bar),
        "title": title,
    }


def _compose_paper_view(main_img: np.ndarray, inset_img: np.ndarray, output_path: str | Path, title: str | None) -> Path:
    _require_matplotlib()
    fig = plt.figure(figsize=(5.7, 3.35), dpi=240)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.92])
    ax.imshow(np.asarray(main_img, dtype=np.uint8))
    ax.set_axis_off()
    if title:
        ax.set_title(str(title), fontsize=10, pad=2.0)

    inset_ax = fig.add_axes([0.67, 0.58, 0.28, 0.28])
    inset_ax.imshow(np.asarray(inset_img, dtype=np.uint8))
    inset_ax.set_axis_off()
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_edgecolor((0.15, 0.15, 0.15, 0.95))

    return _save_figure(fig, output_path, dpi=240)


def render_s5_pybullet_episode(
    *,
    trajectory: np.ndarray,
    output_path: str | Path,
    sphere_center: Sequence[float],
    sphere_radius: float,
    cutpoints: Sequence[int] | None = None,
    overlay_cutpoints: Sequence[int] | None = None,
    tool_axis: np.ndarray | None = None,
    title: str | None = None,
    center_world: Sequence[float] = (0.0, 0.0, 0.98),
    world_scale: float = 1.0,
    main_yaw: float = 42.0,
    inset_yaw: float = 205.0,
    main_pitch: float = -18.0,
    inset_pitch: float = -16.0,
    main_distance: float = 1.42,
    inset_distance: float = 1.46,
    tube_radius: float = 0.0065,
) -> Path:
    _require_pybullet()
    pts = np.asarray(trajectory, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("trajectory must have shape (T, 3+) for S5 pybullet rendering.")
    if tool_axis is None:
        raise ValueError("tool_axis is required for S5 pybullet rendering.")

    axis = np.asarray(tool_axis, dtype=float)
    if axis.shape != pts.shape:
        raise ValueError("tool_axis must have the same shape as trajectory.")

    center_world = np.asarray(center_world, dtype=float).reshape(3)
    sphere_center = np.asarray(sphere_center, dtype=float).reshape(3)
    radius_world = float(world_scale) * float(sphere_radius)
    table_top_z = float(center_world[2] - 1.03 * radius_world)
    traj_world = _env_to_world(pts[:, :3], sphere_center=sphere_center, center_world=center_world, scale=world_scale)
    bounds = stage_segments(len(pts), cutpoints=cutpoints)
    overlay_cutpoints = [] if overlay_cutpoints is None else [int(v) for v in np.asarray(overlay_cutpoints, dtype=int).reshape(-1)]

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.2,
            cameraYaw=35.0,
            cameraPitch=-20.0,
            cameraTargetPosition=[0.0, 0.0, 1.0],
        )

        _spawn_table(table_top_z=table_top_z)
        _spawn_sphere(center_world=center_world, radius_world=radius_world)

        for stage_idx, (start, end) in enumerate(bounds):
            seg = traj_world[start : end + 1]
            color = _hex_to_rgba(STAGE_COLORS[stage_idx % len(STAGE_COLORS)], alpha=1.0)
            for idx in range(len(seg) - 1):
                _spawn_capsule_segment(seg[idx], seg[idx + 1], radius=tube_radius, color=color)

        _spawn_marker(traj_world[0], radius=0.015, color=(0.10, 0.65, 0.25, 1.0))
        _spawn_marker(traj_world[-1], radius=0.014, color=(0.86, 0.18, 0.18, 1.0))
        for cp in overlay_cutpoints:
            if 0 <= int(cp) < len(traj_world):
                _spawn_marker(traj_world[int(cp)], radius=0.011, color=(0.08, 0.08, 0.08, 1.0))

        for start, end in bounds:
            mid = int(round(0.5 * (int(start) + int(end))))
            if 0 <= mid < len(traj_world):
                _spawn_probe_pose(traj_world[mid], axis[mid])

        for _ in range(8):
            p.stepSimulation()

        main_target = center_world + np.array([0.0, 0.0, -0.06], dtype=float)
        inset_target = center_world + np.array([0.0, 0.0, -0.04], dtype=float)
        main_img = _render_rgb(
            yaw_deg=float(main_yaw),
            target=main_target,
            distance=float(main_distance),
            width=1300,
            height=980,
            pitch_deg=float(main_pitch),
        )
        inset_img = _render_rgb(
            yaw_deg=float(inset_yaw),
            target=inset_target,
            distance=float(inset_distance),
            width=720,
            height=720,
            pitch_deg=float(inset_pitch),
        )
    finally:
        p.disconnect(client)

    return _compose_paper_view(main_img=main_img, inset_img=inset_img, output_path=output_path, title=title)
