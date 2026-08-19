cmake_minimum_required(VERSION 3.0.2)
project(stage_optitrack)

find_package(catkin REQUIRED COMPONENTS geometry_msgs roscpp)
find_package(Eigen3 REQUIRED)

catkin_package(CATKIN_DEPENDS geometry_msgs roscpp)

add_executable(optitrack_base_transform src/optitrack_base_transform.cpp)
target_compile_features(optitrack_base_transform PUBLIC cxx_std_14)
target_include_directories(optitrack_base_transform PRIVATE
  ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(optitrack_base_transform ${catkin_LIBRARIES})

install(TARGETS optitrack_base_transform
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
