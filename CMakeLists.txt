cmake_minimum_required(VERSION 3.18)
project(ember LANGUAGES CXX CUDA)

# Training module (optional)
option(EMBER_ENABLE_TRAIN "Build training module" OFF)

# C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA 设置
find_package(CUDAToolkit REQUIRED)

# CUDA 架构（根据需要调整）
# 80 = A100, 86 = RTX 3090/3080, 89 = RTX 4090
set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# 调试模式
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -DDEBUG")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -DDEBUG")
endif()

# 包含目录
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# =============================================================================
# CUDA Common (shared by infer/train)
# =============================================================================
add_library(ember_cuda_common INTERFACE)
target_include_directories(ember_cuda_common INTERFACE
    ${CMAKE_SOURCE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(ember_cuda_common INTERFACE
    CUDA::cudart
    CUDA::cublas
)

# =============================================================================
# CUDA Kernels 静态库
# =============================================================================
set(KERNEL_SOURCES
    backends/cuda/kernels/rmsnorm.cu
    backends/cuda/kernels/rope.cu
    backends/cuda/kernels/softmax.cu
    backends/cuda/kernels/ops.cu
    backends/cuda/kernels/attention.cu
)

add_library(ember_kernels_infer STATIC ${KERNEL_SOURCES})
target_include_directories(ember_kernels_infer PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(ember_kernels_infer PUBLIC ember_cuda_common)
set_target_properties(ember_kernels_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_library(ember_kernels ALIAS ember_kernels_infer)

# =============================================================================
# Core 库（纯 C++）
# =============================================================================
set(CORE_SOURCES
    formats/safetensors.cpp
    core/config_loader.cpp
)

add_library(ember_core STATIC ${CORE_SOURCES})
target_include_directories(ember_core PUBLIC ${CMAKE_SOURCE_DIR})

# =============================================================================
# Inference Runtime 库
# =============================================================================
set(CUDA_RUNTIME_SOURCES
    backends/cuda/cuda_runtime.cpp
    runtime/scheduler.cpp
)

add_library(ember_infer STATIC ${CUDA_RUNTIME_SOURCES})
target_include_directories(ember_infer PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(ember_infer PUBLIC
    ember_core
    ember_kernels_infer
    ember_cuda_common
)
set_target_properties(ember_infer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_library(ember_cuda_runtime ALIAS ember_infer)

# =============================================================================
# 主程序
# =============================================================================
add_executable(ember
    apps/ember_cli/main.cpp
    cli/args.cpp
)
target_link_libraries(ember PRIVATE
    ember_infer
)

# =============================================================================
# CPU Tests (no external deps)
# =============================================================================
add_executable(ember_tests
    tests/test_cpu.cpp
)
target_link_libraries(ember_tests PRIVATE
    ember_core
)
enable_testing()
add_test(NAME ember_tests COMMAND ember_tests)
set_tests_properties(ember_tests PROPERTIES LABELS "cpu")

add_executable(ember_cuda_runtime_smoke
    tests/test_cuda_runtime.cpp
)
target_link_libraries(ember_cuda_runtime_smoke PRIVATE
    ember_infer
)
add_test(NAME ember_cuda_runtime_smoke COMMAND ember_cuda_runtime_smoke)
set_tests_properties(ember_cuda_runtime_smoke PROPERTIES LABELS "gpu")

add_executable(ember_dual_gpu_smoke
    tests/test_dual_gpu.cpp
)
target_link_libraries(ember_dual_gpu_smoke PRIVATE
    ember_infer
)
add_test(NAME ember_dual_gpu_smoke COMMAND ember_dual_gpu_smoke)
set_tests_properties(ember_dual_gpu_smoke PROPERTIES LABELS "gpu")

add_executable(ember_varpos_batch_smoke
    tests/test_varpos_batch.cpp
)
target_link_libraries(ember_varpos_batch_smoke PRIVATE
    ember_infer
)
add_test(NAME ember_varpos_batch_smoke COMMAND ember_varpos_batch_smoke)
set_tests_properties(ember_varpos_batch_smoke PROPERTIES LABELS "gpu")

add_executable(ember_cuda_kernels_smoke
    tests/test_cuda_kernels.cu
)
target_link_libraries(ember_cuda_kernels_smoke PRIVATE
    ember_kernels_infer
    ember_cuda_common
)
add_test(NAME ember_cuda_kernels_smoke COMMAND ember_cuda_kernels_smoke)
set_tests_properties(ember_cuda_kernels_smoke PROPERTIES LABELS "gpu")

add_test(NAME ember_layer_check COMMAND ${CMAKE_SOURCE_DIR}/scripts/ci/layer_check.sh)
set_tests_properties(ember_layer_check PROPERTIES
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    LABELS "gpu;debug"
)

# =============================================================================
# Examples
# =============================================================================
add_executable(ember_example_minimal
    examples/minimal_runtime.cpp
)
target_link_libraries(ember_example_minimal PRIVATE
    ember_infer
)

add_executable(ember_example_decode_loop
    examples/decode_loop.cpp
)
target_link_libraries(ember_example_decode_loop PRIVATE
    ember_infer
)

add_executable(ember_example_check_mode
    examples/check_mode.cpp
)
target_link_libraries(ember_example_check_mode PRIVATE
    ember_infer
)

# =============================================================================
# Benchmarks
# =============================================================================
add_executable(ember_p2p_bandwidth
    benchmarks/p2p_bandwidth.cpp
)
target_link_libraries(ember_p2p_bandwidth PRIVATE
    CUDA::cudart
)

add_executable(ember_kernel_bench
    benchmarks/kernel_bench.cu
)
target_link_libraries(ember_kernel_bench PRIVATE
    ember_kernels_infer
    ember_cuda_common
)

add_executable(ember_phase_analysis
    benchmarks/phase_analysis.cpp
)
target_link_libraries(ember_phase_analysis PRIVATE
    ember_infer
)

add_executable(ember_benchmark
    benchmarks/e2e_benchmark.cpp
)
target_link_libraries(ember_benchmark PRIVATE
    ember_infer
)

add_executable(ember_stage_breakdown
    benchmarks/stage_breakdown.cpp
)
target_link_libraries(ember_stage_breakdown PRIVATE
    ember_infer
)

add_executable(ember_serve_benchmark
    benchmarks/serve_benchmark.cpp
)
target_link_libraries(ember_serve_benchmark PRIVATE
    ember_infer
)

# =============================================================================
# 安装
# =============================================================================
install(TARGETS ember DESTINATION bin)

# =============================================================================
# 打印配置信息
# =============================================================================
message(STATUS "")
message(STATUS "Ember Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA Include: ${CUDAToolkit_INCLUDE_DIRS}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
