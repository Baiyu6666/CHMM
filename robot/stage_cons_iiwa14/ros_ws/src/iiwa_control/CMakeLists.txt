cmake_minimum_required(VERSION 3.0.2)
project(iiwa_control)

find_package(catkin REQUIRED COMPONENTS
  controller_interface
  hardware_interface
  pluginlib
  realtime_tools
  roscpp
  std_msgs
  urdf
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES iiwa_safe_effort_controller
  CATKIN_DEPENDS controller_interface hardware_interface pluginlib realtime_tools roscpp std_msgs urdf
)

add_library(iiwa_safe_effort_controller
  src/watchdog_effort_controller.cpp
)
target_compile_features(iiwa_safe_effort_controller PUBLIC cxx_std_14)
target_include_directories(iiwa_safe_effort_controller PUBLIC include ${catkin_INCLUDE_DIRS})
target_link_libraries(iiwa_safe_effort_controller ${catkin_LIBRARIES})

install(TARGETS iiwa_safe_effort_controller
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
install(FILES plugins/watchdog_effort_controller.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/plugins
)
