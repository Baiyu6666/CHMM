#!/usr/bin/env python3
"""Stream one TF pose as compact, full-precision JSON lines."""

import json
import math
import sys

import rospy
import tf


def main() -> None:
    arguments = rospy.myargv(argv=sys.argv)
    if len(arguments) != 4:
        raise SystemExit("usage: stream_tf_pose.py BASE_FRAME TIP_FRAME RATE_HZ")
    base_frame, tip_frame = arguments[1:3]
    rate_hz = float(arguments[3])
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise SystemExit("RATE_HZ must be positive and finite")

    rospy.init_node("stage_tf_pose_stream")
    listener = tf.TransformListener()
    rate = rospy.Rate(rate_hz)
    while not rospy.is_shutdown():
        try:
            translation, quaternion = listener.lookupTransform(
                base_frame, tip_frame, rospy.Time(0)
            )
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rate.sleep()
            continue
        values = [*translation, *quaternion]
        if all(math.isfinite(value) for value in values):
            print(
                json.dumps(
                    dict(
                        zip(
                            ("x", "y", "z", "qx", "qy", "qz", "qw"),
                            values,
                        )
                    ),
                    separators=(",", ":"),
                ),
                flush=True,
            )
        rate.sleep()


if __name__ == "__main__":
    main()
