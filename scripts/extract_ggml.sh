#!/bin/bash
# 从 llama.cpp 提取 GGML 核心文件
# 用法: ./extract_ggml.sh /path/to/llama.cpp

set -e

LLAMA_CPP_DIR=${1:-"$HOME/llama.cpp"}
GGML_DIR="$(dirname "$0")/../ggml"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    echo "错误: llama.cpp 目录不存在: $LLAMA_CPP_DIR"
    echo "用法: $0 /path/to/llama.cpp"
    exit 1
fi

echo "从 $LLAMA_CPP_DIR 提取 GGML 文件..."

# 创建目标目录
mkdir -p "$GGML_DIR"

# 核心文件列表
CORE_FILES=(
    # GGML 核心
    "ggml/src/ggml.c"
    "ggml/include/ggml.h"
    "ggml/src/ggml-alloc.c"
    "ggml/include/ggml-alloc.h"
    "ggml/src/ggml-backend.cpp"
    "ggml/include/ggml-backend.h"
    "ggml/src/ggml-quants.c"
    "ggml/src/ggml-quants.h"
    
    # GGUF 格式
    "ggml/src/gguf.cpp"
    "ggml/include/gguf.h"
    
    # CUDA 后端
    "ggml/src/ggml-cuda/ggml-cuda.cu"
    "ggml/include/ggml-cuda.h"
)

# 复制文件
for file in "${CORE_FILES[@]}"; do
    src="$LLAMA_CPP_DIR/$file"
    if [ -f "$src" ]; then
        # 获取文件名
        basename=$(basename "$file")
        dst="$GGML_DIR/$basename"
        echo "  复制: $file -> $basename"
        cp "$src" "$dst"
    else
        echo "  警告: 未找到 $file"
    fi
done

# 复制 CUDA kernel 文件（如果存在子目录）
CUDA_DIR="$LLAMA_CPP_DIR/ggml/src/ggml-cuda"
if [ -d "$CUDA_DIR" ]; then
    echo "复制 CUDA kernel 文件..."
    mkdir -p "$GGML_DIR/cuda"
    for cu_file in "$CUDA_DIR"/*.cu "$CUDA_DIR"/*.cuh; do
        if [ -f "$cu_file" ]; then
            basename=$(basename "$cu_file")
            echo "  复制: $basename"
            cp "$cu_file" "$GGML_DIR/cuda/"
        fi
    done
fi

echo ""
echo "提取完成！文件位于: $GGML_DIR"
echo ""
echo "注意事项:"
echo "1. 可能需要手动调整 #include 路径"
echo "2. 删除不需要的后端 (Metal, Vulkan, etc.)"
echo "3. 检查 CMakeLists.txt 是否正确引用这些文件"
