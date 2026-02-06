# Ember - Qwen3 CUDA Inference Engine

轻量级 Qwen3 CUDA 推理引擎，专为消费级多 GPU（如双卡 RTX 3080Ti）设计。

## 特性

- **原生 CUDA 实现**：完全控制计算流程，不依赖 ggml/llama.cpp
- **Safetensors 直接加载**：原生解析 HuggingFace 格式，无需转换
- **Pipeline Parallel**：支持多卡层级切分，自动根据显存分配
- **FP16 计算**：权重和激活使用 FP16，cuBLAS GEMM 用于矩阵运算
- **自定义 Kernels**：RMSNorm、RoPE、Softmax、SiLU、Attention 等

## 项目结构

```
ember/
├── CMakeLists.txt          # 构建配置
├── main.cpp                # CLI 入口
├── core/                   # 核心抽象（纯 C++，不依赖 CUDA）
│   ├── types.h             # DType 枚举、工具函数
│   ├── error.h             # Error 类、Result<T> 模板
│   ├── tensor.h            # 轻量 Tensor 视图
│   ├── config.h            # ModelConfig、RuntimeConfig
│   ├── session.h           # KV Cache、推理会话
│   ├── tokenizer.h         # ITokenizer 接口
│   └── sampler.h           # 采样器（temperature/top-k/top-p）
├── runtime/
│   └── iruntime.h          # IRuntime 接口、DeviceMap、MemoryEstimate
├── formats/
│   ├── safetensors.h       # Safetensors 加载器
│   └── safetensors.cpp
└── backends/cuda/
    ├── cuda_utils.h        # CUDA 工具函数
    ├── cuda_runtime.h      # CUDA Runtime 声明
    ├── cuda_runtime.cpp    # CUDA Runtime 实现
    └── kernels/
        ├── kernels.h       # Kernel 函数声明
        ├── rmsnorm.cu      # RMSNorm kernel
        ├── rope.cu         # RoPE kernel
        ├── softmax.cu      # Softmax kernel
        ├── ops.cu          # SiLU、逐元素操作、Embedding
        └── attention.cu    # Attention、KV Cache 更新
```

## 构建

### 依赖

- CMake 3.18+
- CUDA Toolkit 11.0+（推荐 12.x）
- C++17 编译器

### 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86  # RTX 3080Ti
make -j$(nproc)
```

CUDA 架构参数：
- `86` - RTX 3080/3090
- `89` - RTX 4090
- `80` - A100

## 使用

### 基本用法

```bash
# 单卡推理
./ember -m /path/to/qwen3-4b -p "Hello, my name is"

# 双卡推理
./ember -m /path/to/qwen3-14b --devices 0,1 -p "Explain quantum computing"

# 交互模式
./ember -m /path/to/qwen3-4b -i
```

### 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | 模型目录（包含 safetensors 和 config.json） | 必填 |
| `-p, --prompt` | 输入提示 | "Hello, my name is" |
| `--devices` | GPU 设备列表 | 0 |
| `-c, --ctx-size` | 上下文长度 | 2048 |
| `-n, --n-predict` | 生成 token 数 | 128 |
| `--temp` | 温度 | 0.7 |
| `--top-p` | Top-P | 0.9 |
| `--top-k` | Top-K | 40 |
| `-i, --interactive` | 交互模式 | false |
| `-v, --verbose` | 详细输出 | false |

## 支持的模型

- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B
- Qwen3-14B（需双卡）

模型需从 HuggingFace 下载 safetensors 格式：

```bash
# 使用 huggingface-cli
huggingface-cli download Qwen/Qwen3-4B --local-dir ./qwen3-4b
```

## 架构设计

### 核心抽象

1. **Tensor**: 轻量视图（shape + dtype + data 指针），不管理生命周期
2. **Session**: 推理会话，管理 KV Cache 和生成状态
3. **IRuntime**: 后端接口，支持 CUDA（未来可扩展 CPU）
4. **DeviceMap**: 层级设备映射，支持 Pipeline Parallel

### 计算流程

```
Input IDs
    ↓
Embedding Lookup (GPU 0)
    ↓
┌─────────────────────────────┐
│ Layer 0-N (可跨多卡)         │
│   ├── Input LayerNorm       │
│   ├── QKV Projection        │
│   ├── RoPE                  │
│   ├── KV Cache Update       │
│   ├── Attention (Q@K^T→V)   │
│   ├── O Projection          │
│   ├── Residual Add          │
│   ├── Post-Attn LayerNorm   │
│   ├── MLP (SwiGLU)          │
│   └── Residual Add          │
└─────────────────────────────┘
    ↓
Final LayerNorm
    ↓
LM Head → Logits
    ↓
Sampling → Next Token
```

### 多卡策略

采用 Pipeline Parallel，按层切分：

```
GPU 0: Embedding + Layers 0-13
GPU 1: Layers 14-27 + LM Head
```

层间通过 `cudaMemcpyPeer` 传输 hidden_states。

## 性能

（实际性能取决于硬件和模型大小）

| 模型 | 显存占用 | 预期速度 |
|------|----------|----------|
| Qwen3-4B (FP16) | ~8 GB | ~40 tok/s |
| Qwen3-8B (FP16) | ~16 GB | ~25 tok/s |
| Qwen3-14B (FP16) | ~28 GB | ~15 tok/s (双卡) |

## 路线图

- [x] M0: 单卡推理基础框架
- [ ] M1: 双卡 Pipeline Parallel
- [ ] M2: 量化支持 (INT8/INT4)
- [ ] M3: FlashAttention 优化

## 引用

如果你在论文/报告中使用 Ember，可引用 Zenodo 归档：

- DOI: https://doi.org/10.5281/zenodo.18477269

## 许可证

Apache-2.0 

## 致谢

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 架构参考
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - 模型格式
- [Qwen](https://github.com/QwenLM/Qwen) - 模型权重
