cmake_minimum_required(VERSION 3.16)
project(motion_studio_hardware_controller LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -Wpedantic)

set(OPENARM_CAN_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/openarm_can")
add_library(openarm_can STATIC
  ${OPENARM_CAN_DIR}/src/openarm/can/socket/arm_component.cpp
  ${OPENARM_CAN_DIR}/src/openarm/can/socket/gripper_component.cpp
  ${OPENARM_CAN_DIR}/src/openarm/can/socket/openarm.cpp
  ${OPENARM_CAN_DIR}/src/openarm/canbus/can_device_collection.cpp
  ${OPENARM_CAN_DIR}/src/openarm/canbus/can_socket.cpp
  ${OPENARM_CAN_DIR}/src/openarm/damiao_motor/dm_motor.cpp
  ${OPENARM_CAN_DIR}/src/openarm/damiao_motor/dm_motor_control.cpp
  ${OPENARM_CAN_DIR}/src/openarm/damiao_motor/dm_motor_device.cpp
  ${OPENARM_CAN_DIR}/src/openarm/damiao_motor/dm_motor_device_collection.cpp)
target_include_directories(openarm_can PUBLIC ${OPENARM_CAN_DIR}/include)

add_executable(motion_studio_hardware_service src/main.cpp)
target_link_libraries(motion_studio_hardware_service PRIVATE openarm_can pthread)
