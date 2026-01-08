cmake_minimum_required(VERSION 3.18)
project(ember LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 配置
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_executable(ember main.cpp)
target_link_libraries(ember PRIVATE
    CUDA::cudart
    CUDA::cublas
)

install(TARGETS ember DESTINATION bin)

message(STATUS "")
message(STATUS "Ember Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
