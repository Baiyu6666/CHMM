# stage_optitrack

This package integrates the ROS1 workflow from
`epfl-lasa/optitrack_ros_interface` into the single stage-constraint runtime.
The topic contract is retained:

- `/vrpn_client_node/<rigid_body>/pose` is the Motive/world pose;
- `/vrpn_client_node/<object>/pose_from_<base>` is the object pose relative to
  the tracked robot base.

The local transformer is a small defensive rewrite of the original EPFL LASA
node: it waits for both poses, rejects invalid quaternions, and reports stale
base data. The lab configuration currently uses base rigid body `iiwa14`,
object rigid bodies `baiyu_bar` and `baiyu_obs_ball`, and VRPN server
`128.178.145.104`.

The upstream project is Apache-2.0 licensed:
<https://github.com/epfl-lasa/optitrack_ros_interface>.
