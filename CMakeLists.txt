cmake_minimum_required(VERSION 3.18)
project(ember LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA 配置
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "86")  # RTX 3080Ti = SM86

# 编译选项
add_compile_options(-Wall -Wextra -O3)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# 包含目录
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ggml
)

# ggml 核心库（从 llama.cpp 提取）
add_library(ggml STATIC
    ggml/ggml.c
    ggml/ggml-alloc.c
    ggml/ggml-backend.c
    ggml/ggml-quants.c
    ggml/gguf.c
)
target_compile_definitions(ggml PRIVATE GGML_USE_CUDA)

# ggml-cuda 库
add_library(ggml-cuda STATIC
    ggml/ggml-cuda.cu
)
target_link_libraries(ggml-cuda PRIVATE 
    CUDA::cudart 
    CUDA::cublas
)
set_target_properties(ggml-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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
)

# 测试
enable_testing()
add_executable(test_qwen3 tests/test_qwen3.cpp)
target_link_libraries(test_qwen3 PRIVATE ggml ggml-cuda CUDA::cudart)
add_test(NAME test_qwen3 COMMAND test_qwen3)
