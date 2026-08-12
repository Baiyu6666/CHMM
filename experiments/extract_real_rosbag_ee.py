#!/usr/bin/env python3
"""Extract the IIWA end-effector pose from a ROS1 bag into CSV."""

import argparse
import csv
import math

import numpy as np
import rosbag


def transform_matrix(transform):
    q = transform.transform.rotation
    p = transform.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [p.x, p.y, p.z]
    return matrix


def rotation_quaternion(rotation):
    qw = math.sqrt(max(0.0, 1.0 + np.trace(rotation))) / 2.0
    if qw <= 1e-9:
        return 0.0, 0.0, 0.0, qw
    return (
        (rotation[2, 1] - rotation[1, 2]) / (4 * qw),
        (rotation[0, 2] - rotation[2, 0]) / (4 * qw),
        (rotation[1, 0] - rotation[0, 1]) / (4 * qw),
        qw,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("output")
    args = parser.parse_args()
    rows = []
    first_stamp = None
    with rosbag.Bag(args.bag) as bag:
        for _, message, bag_stamp in bag.read_messages(topics=["/tf"]):
            links = {item.child_frame_id: item for item in message.transforms}
            if not all(f"iiwa_link_{index}" in links for index in range(1, 8)):
                continue
            matrix = np.eye(4)
            for index in range(1, 8):
                matrix = matrix @ transform_matrix(links[f"iiwa_link_{index}"])
            stamp = message.transforms[0].header.stamp.to_sec() or bag_stamp.to_sec()
            first_stamp = stamp if first_stamp is None else first_stamp
            rows.append([stamp - first_stamp, *matrix[:3, 3], *rotation_quaternion(matrix[:3, :3])])
    with open(args.output, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["time_s", "x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"])
        writer.writerows(rows)
    times = np.asarray(rows)[:, 0]
    print(f"frames={len(rows)} duration={times[-1]:.3f}s median_dt={np.median(np.diff(times)):.6f}s")


if __name__ == "__main__":
    main()
