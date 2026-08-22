from __future__ import annotations

import numpy as np

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None


def top_view_scene_geometry(env, demo_index=None):
    getter = getattr(env, "get_top_view_scene_geometry", None)
    if not callable(getter):
        return None
    geometry = getter(demo_index=demo_index)
    if not isinstance(geometry, dict):
        return None
    return geometry


def _top_view_scene_geometries(env, demo_index=None, all_demos=False):
    if all_demos and getattr(env, "demo_scenes", None):
        return [
            top_view_scene_geometry(env, demo_index=index)
            for index in range(len(env.demo_scenes))
        ]
    return [top_view_scene_geometry(env, demo_index=demo_index)]


def draw_top_view_scene(ax, env, demo_index=None, all_demos=False):
    geometries = [
        geometry
        for geometry in _top_view_scene_geometries(
            env,
            demo_index=demo_index,
            all_demos=all_demos,
        )
        if geometry is not None
    ]
    if not geometries or plt is None:
        return False

    geometry = geometries[0]
    bar_corners = np.asarray(geometry["bar_corners_xy"], dtype=float).reshape(-1, 2)
    ax.add_patch(
        plt.Polygon(
            bar_corners,
            closed=True,
            facecolor="#C58B4E",
            edgecolor="#7C4A1D",
            linewidth=1.1,
            alpha=0.30,
            label="tracked bar",
            zorder=0,
        )
    )
    for index, geometry in enumerate(geometries):
        obstacle_center = np.asarray(
            geometry["obstacle_center_xy"],
            dtype=float,
        ).reshape(2)
        obstacle_radius = float(geometry["obstacle_radius"])
        ax.add_patch(
            plt.Circle(
                obstacle_center,
                obstacle_radius,
                facecolor="#DC2626",
                edgecolor="#991B1B",
                linewidth=0.9 if all_demos else 1.1,
                alpha=0.08 if all_demos else 0.18,
                label=(
                    "per-demo tracked obstacles"
                    if all_demos and index == 0
                    else "tracked obstacle" if index == 0 else None
                ),
                zorder=0,
            )
        )
    return True


def top_view_scene_limit_points(env, demo_index=None, all_demos=False):
    geometries = [
        geometry
        for geometry in _top_view_scene_geometries(
            env,
            demo_index=demo_index,
            all_demos=all_demos,
        )
        if geometry is not None
    ]
    if not geometries:
        return np.empty((0, 2), dtype=float)
    geometry = geometries[0]
    bar_corners = np.asarray(geometry["bar_corners_xy"], dtype=float).reshape(-1, 2)
    obstacle_bounds = []
    for geometry in geometries:
        obstacle_center = np.asarray(
            geometry["obstacle_center_xy"],
            dtype=float,
        ).reshape(2)
        obstacle_radius = float(geometry["obstacle_radius"])
        obstacle_bounds.extend(
            [
                obstacle_center - obstacle_radius,
                obstacle_center + obstacle_radius,
            ]
        )
    return np.vstack([bar_corners, np.asarray(obstacle_bounds, dtype=float)])


__all__ = [
    "draw_top_view_scene",
    "top_view_scene_geometry",
    "top_view_scene_limit_points",
]
