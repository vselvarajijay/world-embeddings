cmake_minimum_required(VERSION 3.14)
project(world_embeddings CXX)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs dnn)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(onnxruntime REQUIRED)

# nlohmann/json via FetchContent
include(FetchContent)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(json)

# Core library
add_library(world_embeddings SHARED
  src/association.cpp
  src/entity_store.cpp
  src/snapshot.cpp
  src/encoder_clip.cpp
  src/detector_yolo.cpp
)
target_include_directories(world_embeddings
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(world_embeddings
  PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    opencv_dnn
  PRIVATE
    onnxruntime::onnxruntime
)

# Node executable
add_executable(world_embeddings_node src/world_embeddings_node.cpp)
target_link_libraries(world_embeddings_node world_embeddings)
ament_target_dependencies(world_embeddings_node rclcpp sensor_msgs std_msgs cv_bridge)

# Install
install(TARGETS
  world_embeddings
  world_embeddings_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)
install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# Tests
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  add_subdirectory(test)
endif()

ament_package()
