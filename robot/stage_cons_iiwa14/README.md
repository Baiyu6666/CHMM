# stage_cons_iiwa14

This directory is a self-contained, single-arm ROS Noetic workspace for the
`iiwa14` inspection-demonstration setup. It does not mount, source, or run code
from the old `dual_iiwa_docker` directory.

It contains:

- the EPFL `iiwa_driver` and iiwa14 description required for FRI communication;
- a minimal `SafeTorqueController` with a command watchdog, effort clamp, and
  joint-position guard;
- `inspection_virtual_fixture`, which provides orientation assistance and
  vertical damping for kinesthetic demonstrations;
- the lab `iiwa14`/`baiyu_bar` OptiTrack-to-base transform in the same
  container;
- a validated, service-controlled rosbag demonstration recorder;
- a local web console for demo-mode checks, assistance toggles, rosbag control,
  and a live end-effector XY trace;
- a non-executing Cartesian straight-line placeholder planner;
- one dedicated Docker Compose service using host networking for Motive/VRPN.

## Repository and runtime layers

Both runtime layers are versioned together in this repository:

- `host_gui/` and `systemd/` form the lightweight host layer. It stays running
  while the Docker image or container is replaced and owns lifecycle actions
  only.
- `ros_ws/src/` is the container layer. All ROS robot, OptiTrack, recording,
  assistance, planning, and control behavior remains here.

The host page is `http://127.0.0.1:8080`; it embeds the container-side Demo page
from `http://127.0.0.1:8081`. Rebuilding the image briefly interrupts only the
embedded panel, not the host page.

For the one-time installation of the login service:

```bash
./scripts/install_host_gui_service.sh
```

After that, open <http://127.0.0.1:8080> and use **构建并启动工作站** after
code changes. Login starts only the host page; Docker and `iiwa_driver` still
require explicit buttons, and driver startup additionally requires the typed
`START DRIVER` confirmation.

## Important safety boundary

This is experimental research software, not a certified KUKA safety function.
The software guards supplement, but do not replace, the cabinet safety system,
workspace configuration, mastering, joint limits, an accessible E-stop, and a
trained operator.

**Do not connect this container to the robot while the SmartPAD reports
`Hardware limit exceeded`.** This project cannot recover that fault. Have the
robot returned to a valid mastered configuration using the approved KUKA
service procedure before any FRI or torque-mode test.

The network script and ROS controller are deliberately separate. Starting the
container does not configure the robot-facing Ethernet interface, start the FRI
driver, start OptiTrack, record data, or command the robot.

## Requirements

- Docker Engine with Docker Compose v2 and BuildKit;
- an SSH agent that can read `git@github.com:epfl-lasa/kuka_fri.git`;
- the compatible Sunrise FRI application already installed on the cabinet;
- a dedicated host Ethernet interface for the cabinet;
- a fault-free, correctly mastered robot before real-hardware use.

The FRI source is fetched during image construction and pinned by default to
commit `59af82a8524b37e08a1424a50605e7fe70f11ec9`. Credentials are forwarded only
to the BuildKit clone step and are not copied into the image.

## Build and start without robot access

From this directory:

```bash
cp .env.example .env
# Review .env, especially IIWA14_IFACE and both robot-network IP addresses.
./scripts/start.sh
docker compose exec iiwa14 bash
```

`start.sh` refuses to run if the old container named `kuka14` is active. It
builds and starts only `stage_cons_iiwa14`. Host networking is intentional: the
lab OptiTrack workflow uses VRPN/Motive UDP, which was also the reason the
colleague-provided OptiTrack container ran with `--net host`. Because all ROS
nodes now live in one container, no cross-container ROS routing is needed. The
default command is only `roscore`.

Stop the stack with:

```bash
./scripts/stop.sh
```

## Collect a demonstration

The copied lab configuration is:

- Motive/VRPN server: `128.178.145.104:3883`;
- robot-base rigid body: `iiwa14`;
- tracked objects: `baiyu_bar` and `baiyu_obs_ball`;
- relative bar topic:
  `/vrpn_client_node/baiyu_bar/pose_from_iiwa14`;
- relative obstacle topic:
  `/vrpn_client_node/baiyu_obs_ball/pose_from_iiwa14`.

These values live in `.env` and can be changed without rebuilding the image.
The implementation is based on the colleague-provided
`epfl-lasa/optitrack_ros_interface` ROS1 package, but is integrated directly
into this workspace. Its transformer also rejects invalid poses and stale base
measurements instead of using uninitialized data.

### Recommended: web GUI

Install the host supervisor once as described above, then open
<http://127.0.0.1:8080>. Use **构建并启动工作站** to build/start the container
and its integrated OptiTrack, recorder, virtual-fixture, and Demo panel. This
does not connect to the robot.

Only when real-hardware operation is intended and the robot is safe, use the
separate **配置机器人网络** and **启动 iiwa_driver** controls. Driver startup
requires typing `START DRIVER`. Wait until all required connection indicators
are green, then use this sequence:

```bash
cd /home/baiyu/LearnStageConstraints/robot/stage_cons_iiwa14
./scripts/start_demo_gui.sh
```

1. Click **激活 Demo 采集模式**. This checks joint state, EE TF, both OptiTrack objects,
   virtual-fixture, recorder, FRI commanding, FRI torque mode, and the driver
   motion gate. It clears the XY trace and explicitly starts with both
   assistance channels off.
2. If needed, independently enable **Orientation 保持** and/or **竖直阻尼**.
3. Set the demo label and optional notes, then click **开始 Record Demo**.
4. Move the robot through the demonstration while watching the EE path in the
   XY panel.
5. Click **停止并保存 Rosbag**, then click **退出 Demo 采集模式**. Exiting
   disables both assistance channels and stops an unfinished recording.

**Position-reference hold has been removed.** Its hardware tests produced a
strong return force toward an unexpected pose with the installed Sunrise
impedance application. The driver now unconditionally keeps the FRI position
reference at the measured joints in both GUI states; there is no configuration
switch that can restore the old behavior. Demo mode is only a gate for
recording, tracing, and optional assistance and does not mechanically lock the
arm outside Demo mode.

Both pages bind to loopback only. The host supervisor is on port 8080 and the
container Demo page is on port 8081. Neither bypasses SmartPAD safety state.
Do not start `start_demo_collection.sh` or a second virtual-fixture launch at
the same time; the Demo station already owns those nodes.

For manual troubleshooting without the host supervisor, start the container
Demo page from a terminal with `./scripts/start_demo_gui.sh` and open
<http://127.0.0.1:8081>.

### Fast assistance tuning

Assistance gains can be reloaded while the stack is running, without rebuilding
the image or restarting the driver. Both assistance channels must first be off;
the reload is rejected otherwise. For example:

```bash
# orientation: stiffness, damping, moment limit, recovery speed (deg/s)
./scripts/tune_assistance.sh orientation 1.0 1.0 0.5 10.0

# vertical: damping, force limit
./scripts/tune_assistance.sh vertical 10.0 2.0
```

The node rejects missing, non-finite, or out-of-range values and leaves both
channels disabled after a successful reload. These software bounds are tuning
guard rails, not certified robot safety limits. The position-reference hold is
not touched by this workflow and remains disabled.

### Command-line alternative

Without the GUI, start OptiTrack and the recorder coordinator in a second host
terminal:

```bash
./scripts/start_demo_collection.sh
```

This still does **not** begin recording. Confirm that the four required data
streams are live:

```bash
docker compose exec iiwa14 bash -lc \
  'rostopic hz /iiwa14/joint_states /tf /vrpn_client_node/baiyu_bar/pose_from_iiwa14 /vrpn_client_node/baiyu_obs_ball/pose_from_iiwa14'
```

Then start one demonstration:

```bash
docker compose exec iiwa14 rosservice call /demo_recorder/start
```

Mark stage transitions while moving the robot. Use stable names that the
host-side preprocessing can map to stage indices:

```bash
docker compose exec iiwa14 rostopic pub -1 /stage_cons/demo_marker \
  std_msgs/String "data: 'stage_1_start'"
docker compose exec iiwa14 rostopic pub -1 /stage_cons/demo_marker \
  std_msgs/String "data: 'stage_2_start'"
```

Stop and finalize the bag:

```bash
docker compose exec iiwa14 rosservice call /demo_recorder/stop
```

Every run is written on the host under `data/demos/<timestamp>_<label>/` as
`demo.bag` plus `metadata.json`. The recorder refuses to start if joint states,
TF, or the relative `baiyu_bar` pose is missing, and it also checks free disk
space. Raw bags and exported models are ignored by Git; only code and configs
are versioned.

Change the label or experiment notes before the next run with ROS parameters:

```bash
docker compose exec iiwa14 rosparam set /demo_recorder/label demo_03
docker compose exec iiwa14 rosparam set /demo_recorder/operator_notes \
  'clockwise inspection, orientation assistance enabled'
```

The host-side `experiments/extract_real_rosbag_ee.py` reconstructs the
end-effector pose from `/tf` and auto-detects the `iiwa` or `iiwa14` link-name
prefix. The bag additionally keeps joint state, external torque/controller
signals, raw OptiTrack poses, the base-relative object pose, virtual-fixture
status, stage markers, and any placeholder plan that is published.

## Connect to the cabinet

Run this section only after the hardware-limit fault has been professionally
cleared and the values in `.env` have been checked:

```bash
./scripts/connect_robot_network.sh
```

Because the container uses host networking for Motive, this explicit script
adds the configured FRI host address to the dedicated host Ethernet interface
and checks the cabinet with `ping`. It may prompt for `sudo`. It still does not
start the driver or a torque controller. Do not use the robot-facing interface
for general lab or internet traffic.

With the compatible Sunrise FRI application ready on the cabinet, open a shell:

```bash
docker compose exec iiwa14 bash
roslaunch iiwa_driver iiwa14_bringup.launch
```

The bringup defaults to `SafeTorqueController` and the iiwa14 robot name. To use
a different cabinet address:

```bash
roslaunch iiwa_driver iiwa14_bringup.launch robot_ip:=192.170.10.4
```

The SmartPAD/Sunrise sequence must match the installed cabinet application.
Selecting torque mode on the SmartPAD alone does not replace that application.

## Inspection assistance

`start_demo_gui.sh` starts the virtual fixture and exposes its two switches in
the web page. For command-line-only operation, start it in a second container
shell after the driver is running and valid joint states are visible:

```bash
roslaunch inspection_virtual_fixture inspection_virtual_fixture_iiwa14.launch
```

Both assistance channels start disabled. The current configuration uses the
live `baiyu_bar` OptiTrack orientation and defines that rigid body's local `+X`
axis as the long direction of the bar. If the rigid-body definition is changed
in Motive, verify this axis again before enabling assistance.

Enable only the channel needed for the current demonstration phase:

```bash
rosservice call /iiwa14/demo_virtual_fixture/enable_orientation "data: true"
rosservice call /iiwa14/demo_virtual_fixture/enable_vertical_damping "data: true"
```

Enable or disable both together:

```bash
rosservice call /iiwa14/demo_virtual_fixture/enable_all "data: true"
rosservice call /iiwa14/demo_virtual_fixture/enable_all "data: false"
```

Orientation assistance leaves one rotational degree of freedom: downward tilt
along the bar direction. It restores/damps the other orientation components.
Vertical assistance is velocity damping only: it makes vertical movement feel
harder without defining a fixed height, target height, or allowed interval.

The checked-in gains remain at the original conservative signal-chain test
tier. A higher assistance tier was prepared but rolled back without hardware
activation after an unrelated position-reference hold test produced an unsafe
return force. Tune one channel at a time only after the baseline behavior has
been restored and disable it before changing configurations.

Tune the fixture in
[`inspection_virtual_fixture_iiwa14.yaml`](ros_ws/src/inspection_virtual_fixture/config/inspection_virtual_fixture_iiwa14.yaml).
Begin with the conservative limits already present, verify coordinate signs and
tool axes at low assistance, and change one parameter at a time.

## Placeholder planning boundary

The initial planner exists only to stabilize the interface while the learned
planner is being researched:

```bash
docker compose exec iiwa14 roslaunch stage_placeholder_planner straight_line_planner.launch
```

It consumes `geometry_msgs/PoseStamped` on
`/stage_cons/planner/start` and `/stage_cons/planner/goal`. Calling
`/straight_line_planner/plan` publishes an interpolated `nav_msgs/Path` on
`/stage_cons/plan`. It performs no IK, collision checking, constraint checking,
time parameterization, controller switching, or hardware execution. Keeping
this boundary non-executing prevents a placeholder from accidentally becoming
a robot command path.

The intended later data/model boundary is similarly explicit: training remains
in the host `LearnStageConstraints` environment, and a reviewed deployment
bundle is exported to `data/models/`, mounted read-only at `/models` in the
container. Training dependencies and notebooks are therefore not installed in
the real-time robot image.

## Guards and diagnostics

The fixture and effort controller each independently force zero assist torque
within 5 degrees of any URDF joint limit. The fixture also disables both
channels when this guard is entered. Other zero-output conditions include stale
joint states, stale live bar poses, malformed input, non-finite calculations,
or a command older than 50 ms.

These guards do not actively move a joint away from a limit. They also cannot
prevent motion caused by gravity compensation, the cabinet application, an
incorrect robot model, or other software commanding the robot.

Inspect the output and status with:

```bash
rostopic echo /iiwa14/SafeTorqueController/command
rostopic echo /iiwa14/demo_virtual_fixture/status
```

The 13 status entries are, in order:

1. assistance active;
2. orientation requested;
3. vertical damping requested;
4. joint-state age in seconds;
5. bar-pose age in seconds;
6. vertical TCP velocity;
7. current allowed-axis tilt in radians;
8. constrained orientation error in radians;
9. vertical damping force;
10. norm of the seven-joint assist torque command;
11. enable ramp from 0 to 1;
12. minimum distance to a URDF joint limit in radians;
13. joint-limit guard active.

## Source layout and provenance

- `ros_ws/src/iiwa_driver`, `iiwa_description`, and `force_sensor` were extracted
  from EPFL LASA `iiwa_ros` commit
  `99d488b2bcb4d28c0aafd91a9f657964e8657726` and retain their upstream notices.
- Their GPLv3 license text is stored in `IIWA_ROS_LICENSE`.
- `ros_ws/src/iiwa_control` is the project-local minimal watchdog effort
  controller; it is licensed GPL-3.0-or-later because it derives from the
  earlier GPL controller implementation.
- `ros_ws/src/inspection_virtual_fixture` is project-local code and uses ROS KDL
  directly. It does not depend on `iiwa_tools`, RBDyn, Corrade, or the old dual
  arm workspace.
- `ros_ws/src/stage_optitrack` retains the topic contract and transform used by
  the Apache-2.0 `epfl-lasa/optitrack_ros_interface` project copied by the lab,
  with local input validation and freshness checks.
- `ros_ws/src/stage_demo_recorder` owns only acquisition and recording. It does
  not infer task stages; the operator's marker events are recorded verbatim.
- `ros_ws/src/stage_placeholder_planner` publishes visualization/integration
  paths only and has no execution connection.
- KUKA FRI is not vendored. Its own license and distribution terms apply to the
  source fetched during the Docker build.

The resulting Docker image is `baiyu/stage_cons_iiwa14:noetic`, and the only
container created by this project is `stage_cons_iiwa14`.
