cmake_minimum_required(VERSION 3.0.2)
project(stage_demo_recorder)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs std_srvs)

catkin_package(CATKIN_DEPENDS rospy std_msgs std_srvs)

catkin_install_python(PROGRAMS scripts/demo_recorder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
