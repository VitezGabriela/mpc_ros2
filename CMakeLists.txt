cmake_minimum_required(VERSION 3.10)
project(mpc_ros2)

# -----------------------------
# C++ standard
# -----------------------------
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -----------------------------
# ROS2 dependencies
# -----------------------------
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
# -----------------------------
# System libraries
# -----------------------------
# CppAD
find_path(CPPAD_INCLUDE_DIR NAMES cppad/cppad.hpp)
if(NOT CPPAD_INCLUDE_DIR)
  message(FATAL_ERROR "CppAD headers not found")
endif()

# IPOPT
find_library(IPOPT_LIB NAMES ipopt PATHS /usr/local/lib)
if(NOT IPOPT_LIB)
  message(FATAL_ERROR "Ipopt library not found")
endif()
message(STATUS "Ipopt library: ${IPOPT_LIB}")

# -----------------------------
# MPC library
# -----------------------------
add_library(mpc SHARED src/mpc.cpp)
target_include_directories(mpc PUBLIC
    include
    ${CPPAD_INCLUDE_DIR}
    /usr/local/include/coin-or   # IPOPT headers
    ${EIGEN3_INCLUDE_DIRS}
)
target_link_libraries(mpc ${IPOPT_LIB})

# -----------------------------
# Node executable
# -----------------------------
add_executable(mpc_ros2 src/mpc_ros2_node.cpp)
ament_target_dependencies(mpc_ros2
    rclcpp
    geometry_msgs
    sensor_msgs
    nav_msgs
    visualization_msgs
    tf2
    tf2_ros
    tf2_geometry_msgs
    moveit_core
    moveit_ros_planning_interface
)
target_link_libraries(mpc_ros2 mpc)

# -----------------------------
# Installation
# -----------------------------
install(TARGETS mpc
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(TARGETS mpc_ros2
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY params/ launch/ DESTINATION share/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

ament_package()
