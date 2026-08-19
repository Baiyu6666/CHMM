cmake_minimum_required(VERSION 3.0.2)
project(stage_demo_gui)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf2_ros
)

catkin_package(CATKIN_DEPENDS
  geometry_msgs rospy sensor_msgs std_msgs std_srvs tf2_ros
)

catkin_install_python(PROGRAMS scripts/demo_gui.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch web
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
