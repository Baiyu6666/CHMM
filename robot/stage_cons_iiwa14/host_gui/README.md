# Host workstation layer

This directory is intentionally outside the Docker image. The supervisor binds
only to `127.0.0.1:8080` and exposes a fixed allow-list of workstation lifecycle
actions. It can build/start the container, launch the container-side Demo
station, configure the dedicated robot NIC through an explicit graphical
authorization prompt, and start/stop `iiwa_driver` after a typed confirmation.

Robot logic does not belong here. OptiTrack, rosbag recording, virtual fixtures,
planning, control, and their safety checks remain ROS packages under
`ros_ws/src/` and run inside the container. The container-side Demo page binds
to `127.0.0.1:8081` and is embedded in the host page.

Rebuilding or replacing the container therefore does not stop this supervisor.
The embedded Demo page is temporarily unavailable and returns after the new
container and ROS master are ready.

Install the user service once from the repository root:

```bash
./scripts/install_host_gui_service.sh
```

The service starts on login, but deliberately does not start Docker, configure
the robot network, or connect to the robot without explicit UI actions.
