cmake_minimum_required(VERSION 3.18)
project(ember LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 配置
find_package(CUDAToolkit REQUIRED)

add_executable(ember main.cpp)

install(TARGETS ember DESTINATION bin)

message(STATUS "")
message(STATUS "Ember Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
