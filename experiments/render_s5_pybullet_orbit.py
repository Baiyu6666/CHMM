from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import load_env


STAGE_COLORS = [
    (0.84, 0.29, 0.04, 1.0),
    (0.00, 0.45, 0.70, 1.0),
    (0.00, 0.62, 0.45, 1.0),
    (0.80, 0.47, 0.65, 1.0),
    (0.90, 0.62, 0.00, 1.0),
]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return arr / norm


def _quat_align_z_to_vec(vec: np.ndarray) -> tuple[float, float, float, float]:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    target = _normalize(vec)
    dot = float(np.clip(np.dot(z_axis, target), -1.0, 1.0))
    if dot >= 1.0 - 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    if dot <= -1.0 + 1e-8:
        return tuple(p.getQuaternionFromEuler((np.pi, 0.0, 0.0)))
    axis = _normalize(np.cross(z_axis, target))
    angle = float(np.arccos(dot))
    return tuple(p.getQuaternionFromAxisAngle(axis.tolist(), angle))


def _segment_bounds(true_cutpoints: list[int], length: int) -> list[tuple[int, int]]:
    ends = [int(v) for v in true_cutpoints] + [int(length - 1)]
    starts = [0] + [end + 1 for end in ends[:-1]]
    return list(zip(starts, ends))


def _env_to_world(points: np.ndarray, env, center_world: np.ndarray, scale: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return center_world[None, :] + scale * (pts - np.asarray(env.sphere_center, dtype=float)[None, :])


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
        basePosition=pos_world.tolist(),
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
    axis = _normalize(axis_world)
    shaft_center = np.asarray(pos_world, dtype=float) - 0.5 * shaft_len * axis
    _spawn_oriented_cylinder(
        pos_world=shaft_center,
        axis_world=axis,
        length=shaft_len,
        radius=shaft_radius,
        color=(0.18, 0.20, 0.24, 1.0),
    )
    collar_center = np.asarray(pos_world, dtype=float) - 0.12 * shaft_len * axis
    _spawn_oriented_cylinder(
        pos_world=collar_center,
        axis_world=axis,
        length=0.018,
        radius=0.0065,
        color=(0.12, 0.46, 0.84, 1.0),
    )
    tip_center = np.asarray(pos_world, dtype=float) + 0.5 * tip_len * axis
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
    cyl_len = max(seg_len - 2.0 * radius, 1e-4)
    vis_id = p.createVisualShape(
        p.GEOM_CAPSULE,
        radius=radius,
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
    yaw_deg: float,
    target: np.ndarray,
    distance: float,
    width: int,
    height: int,
    pitch_deg: float = -23.0,
) -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target.tolist(),
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


def _compose_paper_view(main_img: np.ndarray, inset_img: np.ndarray, output_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(5.7, 3.35), dpi=240)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.92])
    ax.imshow(main_img)
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, pad=2.0)

    inset_ax = fig.add_axes([0.67, 0.58, 0.28, 0.28])
    inset_ax.imshow(inset_img)
    inset_ax.set_axis_off()
    for spine in inset_ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.1)
        spine.set_edgecolor((0.15, 0.15, 0.15, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def render_s5_orbit_from_run(
    run_dir: Path,
    demo_idx: int = 0,
    yaws: tuple[float, ...] = (42.0, 205.0),
    output_path: Path | None = None,
) -> Path:
    run_dir = run_dir.resolve()
    metadata = _read_json(run_dir / "metadata.json")
    cfg = _read_json(run_dir / "config_snapshot.json")
    segmentation = _read_json(run_dir / "segmentation.json")

    dataset_name = str(metadata["dataset_name"])
    if dataset_name != "S5SphereInspect":
        raise ValueError(f"This renderer currently only supports S5SphereInspect, got '{dataset_name}'.")

    dataset_kwargs = dict(cfg.get("dataset_kwargs", {}))
    bundle = load_env(dataset_name, **dataset_kwargs)

    demo = np.asarray(bundle.demos[int(demo_idx)], dtype=float)
    true_cutpoints = [int(v) for v in segmentation["true_cutpoints"][int(demo_idx)]]
    learned_cutpoints = [int(v) for v in segmentation["predicted_cutpoints"][int(demo_idx)]]
    env = bundle.env
    tool_axis = env._lookup_cached_tool_axis_trace(demo)
    if tool_axis is None:
        tool_axis = env._estimate_tool_axis_from_geometry(demo)
    tool_axis = np.asarray(tool_axis, dtype=float)

    if output_path is None:
        output_path = run_dir / f"paper_pybullet_orbit_demo_{int(demo_idx):02d}.png"
    output_path = output_path.resolve()

    client = p.connect(p.DIRECT)
    try:
        p.resetSimulation()
        p.setGravity(0.0, 0.0, -9.81)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=35.0, cameraPitch=-20.0, cameraTargetPosition=[0, 0, 1.0])

        center_world = np.array([0.0, 0.0, 0.98], dtype=float)
        world_scale = 0.18
        sphere_radius_world = world_scale * float(env.sphere_radius)
        table_top_z = float(center_world[2] - 1.03 * sphere_radius_world)

        _spawn_table(table_top_z=table_top_z)
        _spawn_sphere(center_world=center_world, radius_world=sphere_radius_world)

        traj_world = _env_to_world(demo, env=env, center_world=center_world, scale=world_scale)
        bounds = _segment_bounds(true_cutpoints, len(demo))
        tube_radius = 0.0065

        for stage_idx, (start, end) in enumerate(bounds):
            color = STAGE_COLORS[stage_idx % len(STAGE_COLORS)]
            seg = traj_world[start : end + 1]
            for i in range(len(seg) - 1):
                _spawn_capsule_segment(seg[i], seg[i + 1], radius=tube_radius, color=color)

        _spawn_marker(traj_world[0], radius=0.015, color=(0.10, 0.65, 0.25, 1.0))
        _spawn_marker(traj_world[-1], radius=0.014, color=(0.86, 0.18, 0.18, 1.0))
        for cp in learned_cutpoints:
            if 0 <= int(cp) < len(traj_world):
                _spawn_marker(traj_world[int(cp)], radius=0.011, color=(0.08, 0.08, 0.08, 1.0))

        for start, end in bounds:
            mid = int(round(0.5 * (start + end)))
            if 0 <= mid < len(traj_world):
                _spawn_probe_pose(traj_world[mid], tool_axis[mid])

        for _ in range(8):
            p.stepSimulation()

        if len(yaws) < 2:
            raise ValueError("Expected at least two camera yaws: one for main view and one for inset view.")
        main_target = center_world + np.array([0.0, 0.0, -0.06], dtype=float)
        inset_target = center_world + np.array([0.0, 0.0, -0.04], dtype=float)
        main_img = _render_rgb(
            yaw_deg=float(yaws[0]),
            target=main_target,
            distance=1.42,
            width=1300,
            height=980,
            pitch_deg=-18.0,
        )
        inset_img = _render_rgb(
            yaw_deg=float(yaws[1]),
            target=inset_target,
            distance=1.46,
            width=720,
            height=720,
            pitch_deg=-16.0,
        )
    finally:
        p.disconnect(client)

    _compose_paper_view(
        main_img=main_img,
        inset_img=inset_img,
        output_path=output_path,
        title=f"S5SphereInspect demo {int(demo_idx)}",
    )
    return output_path


def _parse_yaws(text: str) -> tuple[float, ...]:
    parts = [item.strip() for item in str(text).split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of camera yaws.")
    return tuple(float(v) for v in parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an S5 PyBullet orbit figure from a saved SWCL run.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing metadata/config_snapshot/segmentation.")
    parser.add_argument("--demo-idx", type=int, default=0, help="Demo index to render.")
    parser.add_argument(
        "--camera-yaws",
        type=_parse_yaws,
        default=(35.0, 150.0, 265.0),
        help="Comma-separated camera yaws for the orbit strip.",
    )
    parser.add_argument("--output", default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    output_path = None if args.output is None else Path(args.output)
    saved = render_s5_orbit_from_run(
        run_dir=Path(args.run_dir),
        demo_idx=int(args.demo_idx),
        yaws=tuple(args.camera_yaws),
        output_path=output_path,
    )
    print(saved)


if __name__ == "__main__":
    main()
