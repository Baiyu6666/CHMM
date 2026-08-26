cmake_minimum_required(VERSION 3.0.2)
project(stage_real_executor)

find_package(catkin REQUIRED COMPONENTS
  actionlib
  actionlib_msgs
  control_msgs
  geometry_msgs
  iiwa_driver
  nav_msgs
  rospy
  sensor_msgs
  stage_cartesian_trajectory
  std_msgs
  std_srvs
  tf
  trajectory_msgs
)

catkin_package(CATKIN_DEPENDS
  actionlib actionlib_msgs control_msgs geometry_msgs iiwa_driver nav_msgs rospy sensor_msgs
  stage_cartesian_trajectory std_msgs std_srvs trajectory_msgs
  tf
)

catkin_install_python(PROGRAMS
  scripts/real_executor.py
  scripts/stream_tf_pose.py
  scripts/submit_real_plan.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
