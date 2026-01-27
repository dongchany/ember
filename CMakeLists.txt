cmake_minimum_required(VERSION 3.18)
project(ember LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# =============================================================================
# CUDA Kernels static lib
# =============================================================================
set(KERNEL_SOURCES backends/cuda/kernels/rmsnorm.cu
        backends/cuda/kernels/rope.cu
        backends/cuda/kernels/softmax.cu
        backends/cuda/kernels/ops.cu
        backends/cuda/kernels/attention.cu)

add_library(ember_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(ember_kernels PUBLIC ${CMAKE_SOURCE_DIR})
set_target_properties(ember_kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# =============================================================================
# Core lib
# =============================================================================
set(CORE_SOURCES core/config_loader.cpp formats/safetensors.cpp)

add_library(ember_core STATIC ${CORE_SOURCES})
target_include_directories(ember_core PUBLIC ${CMAKE_SOURCE_DIR})

add_executable(ember
        apps/ember_cli/main.cpp
        cli/args.cpp
)
target_link_libraries(ember PRIVATE
        ember_core
        CUDA::cudart
        CUDA::cublas
)

# =============================================================================
# CPU/CUDA Tests (no external deps)
# =============================================================================
add_executable(ember_tests tests/test_cpu.cpp)
target_link_libraries(ember_tests PRIVATE ember_core)

enable_testing()
add_test(NAME ember_tests COMMAND ember_tests)
set_tests_properties(ember_tests PROPERTIES LABELS "cpu")

add_executable(ember_cuda_kernels_smoke tests/test_cuda_kernels.cu)
target_link_libraries(ember_cuda_kernels_smoke PRIVATE ember_kernels CUDA::cudart CUDA::cublas
)

add_test(NAME ember_cuda_kernels_smoke COMMAND ember_cuda_kernels_smoke)
set_tests_properties(ember_cuda_kernels_smoke PROPERTIES LABELS "gpu")


install(TARGETS ember DESTINATION bin)

message(STATUS "")
message(STATUS "Ember Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
