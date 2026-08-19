cmake_minimum_required(VERSION 3.0.2)
project(stage_placeholder_planner)

find_package(catkin REQUIRED COMPONENTS geometry_msgs nav_msgs rospy std_srvs)

catkin_package(CATKIN_DEPENDS geometry_msgs nav_msgs rospy std_srvs)

catkin_install_python(PROGRAMS scripts/straight_line_planner.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
