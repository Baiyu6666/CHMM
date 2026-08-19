cmake_minimum_required(VERSION 3.0.2)
project(inspection_virtual_fixture)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  kdl_parser
  roscpp
  sensor_msgs
  std_msgs
  std_srvs
  urdf
)
find_package(Eigen3 REQUIRED)
find_package(orocos_kdl REQUIRED)

catkin_package(
  CATKIN_DEPENDS geometry_msgs kdl_parser roscpp sensor_msgs std_msgs std_srvs urdf
  DEPENDS orocos_kdl
)

add_executable(inspection_virtual_fixture src/inspection_virtual_fixture.cpp)
target_compile_features(inspection_virtual_fixture PUBLIC cxx_std_14)
target_include_directories(inspection_virtual_fixture PRIVATE
  ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${orocos_kdl_INCLUDE_DIRS}
)
target_link_libraries(inspection_virtual_fixture ${catkin_LIBRARIES} ${orocos_kdl_LIBRARIES})

install(TARGETS inspection_virtual_fixture
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
