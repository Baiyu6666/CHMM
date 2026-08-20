cmake_minimum_required(VERSION 3.0.2)
project(stage_iiwa_sim)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  stage_cartesian_trajectory
  std_msgs
  std_srvs
)

catkin_package(CATKIN_DEPENDS
  geometry_msgs
  nav_msgs
  rospy
  sensor_msgs
  stage_cartesian_trajectory
  std_msgs
  std_srvs
)

catkin_install_python(PROGRAMS scripts/iiwa_pybullet_sim.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
