cmake_minimum_required(VERSION 3.18)
project(ember LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA 配置
find_package(CUDAToolkit REQUIRED)

# 检测 GPU 架构，默认 SM86 (RTX 3080Ti)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "86")
endif()

# 编译选项
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --extended-lambda -Xcompiler -fPIC")

# 定义宏
add_definitions(-DGGML_USE_CUDA)
add_definitions(-DGGML_CUDA_USE_GRAPHS)

# 包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml
    ${PROJECT_SOURCE_DIR}/ggml/cuda
    ${CUDAToolkit_INCLUDE_DIRS}
)

# 收集 CUDA kernel 源文件
file(GLOB GGML_CUDA_SOURCES 
    "${PROJECT_SOURCE_DIR}/ggml/cuda/*.cu"
)

# ggml-cuda 库（包含所有 CUDA kernel）
add_library(ggml-cuda STATIC
    ${PROJECT_SOURCE_DIR}/ggml/cuda/ggml-cuda.cu
    ${GGML_CUDA_SOURCES}
)
target_link_libraries(ggml-cuda PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
)
set_target_properties(ggml-cuda PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_compile_definitions(ggml-cuda PRIVATE GGML_USE_CUDA GGML_CUDA_USE_GRAPHS)

# ggml 核心库
add_library(ggml STATIC
    ${PROJECT_SOURCE_DIR}/ggml/ggml.c
    ${PROJECT_SOURCE_DIR}/ggml/ggml-alloc.c
    ${PROJECT_SOURCE_DIR}/ggml/ggml-backend.cpp
    ${PROJECT_SOURCE_DIR}/ggml/ggml-quants.c
    ${PROJECT_SOURCE_DIR}/ggml/gguf.cpp
)
target_compile_definitions(ggml PRIVATE GGML_USE_CUDA)
target_link_libraries(ggml PUBLIC ggml-cuda)
set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ember 主程序
add_executable(ember
    src/main.cpp
    src/cli.cpp
    src/model.cpp
    src/inference.cpp
    src/utils.cpp
)
target_link_libraries(ember PRIVATE
    ggml
    ggml-cuda
    CUDA::cudart
    CUDA::cublas
    pthread
    dl
)

# 安装
install(TARGETS ember RUNTIME DESTINATION bin)

# 测试（可选）
option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_executable(test_qwen3 tests/test_qwen3.cpp)
    target_link_libraries(test_qwen3 PRIVATE ggml ggml-cuda CUDA::cudart)
    add_test(NAME test_qwen3 COMMAND test_qwen3)
endif()

# 打印配置信息
message(STATUS "")
message(STATUS "=== Ember Build Configuration ===")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA toolkit: ${CUDAToolkit_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "=================================")
message(STATUS "")