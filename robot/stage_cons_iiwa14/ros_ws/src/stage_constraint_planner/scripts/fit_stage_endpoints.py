#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np

from stage_constraint_planner import quaternion_to_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("processed_npz")
    parser.add_argument("--scene-config", required=True)
    parser.add_argument("--trajectory-key", default="trajectory")
    parser.add_argument("--demo-id-key", default="demo_id")
    parser.add_argument("--stage-label-key", default="coarse_stage_labels")
    parser.add_argument("--bar-pose-key", default="locked_bar_pose")
    parser.add_argument("--output")
    args = parser.parse_args()

    scene_config = json.loads(Path(args.scene_config).read_text())
    optitrack_to_robot = scene_config["optitrack_to_robot"]
    transform_rotation = np.asarray(optitrack_to_robot["rotation"], dtype=float).reshape(3, 3)
    transform_translation = np.asarray(optitrack_to_robot["translation"], dtype=float).reshape(3)

    with np.load(args.processed_npz) as data:
        trajectory = np.asarray(data[args.trajectory_key], dtype=float)
        demo_ids = np.asarray(data[args.demo_id_key], dtype=int)
        stage_labels = np.asarray(data[args.stage_label_key], dtype=int)
        bar_poses = np.asarray(data[args.bar_pose_key], dtype=float)

    endpoints_by_stage = [[] for _ in range(3)]
    valid_demo_ids = [int(value) for value in np.unique(demo_ids) if int(value) >= 0]
    for demo_id in valid_demo_ids:
        for stage in range(3):
            indices = np.flatnonzero((demo_ids == demo_id) & (stage_labels == stage))
            if not len(indices):
                raise ValueError("Demo {} has no samples for stage {}".format(demo_id, stage))
            endpoint_index = int(indices[-1])
            bar_pose = bar_poses if bar_poses.shape == (7,) else bar_poses[endpoint_index]
            bar_rotation_robot = transform_rotation @ quaternion_to_matrix(bar_pose[3:7])
            bar_position_robot = transform_rotation @ bar_pose[:3] + transform_translation
            endpoint_robot = trajectory[endpoint_index, :3]
            endpoint_bar = bar_rotation_robot.T @ (endpoint_robot - bar_position_robot)
            endpoints_by_stage[stage].append(endpoint_bar)

    fitted = np.asarray(
        [np.mean(np.asarray(stage_values, dtype=float), axis=0) for stage_values in endpoints_by_stage]
    )
    payload = {
        "fit": "arithmetic_mean_xyz",
        "demo_count": len(valid_demo_ids),
        "stage_endpoint_positions_bar": fitted.tolist(),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        Path(args.output).write_text(serialized)
    else:
        print(serialized, end="")


if __name__ == "__main__":
    main()
