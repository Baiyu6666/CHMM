# Configuration ownership

Machine-specific workstation values are copied from the versioned
`.env.example` to the ignored root `.env`. This includes the physical Ethernet
interface, FRI addresses, and Motive rigid-body names.

Configuration owned by one ROS package stays beside that package so catkin and
roslaunch can resolve it normally:

- `ros_ws/src/iiwa_control/config/`: controller guards and limits;
- `ros_ws/src/inspection_virtual_fixture/config/`: assistance gains and bounds;
- `ros_ws/src/stage_demo_recorder/config/`: recorded topic set.

This top-level directory is reserved for future configuration that is shared by
multiple packages. Do not place secrets, local IP overrides, rosbag data, or
trained model weights here.
