# Ember 🔥

> *燃尽繁杂，留下精华*

从 llama.cpp 10万行代码中提炼出的 Qwen3 CUDA 推理引擎。

## 特性

- **专注 Qwen3 Dense**: 不支持 MoE、多模态
- **CUDA 优先**: 仅支持 NVIDIA GPU
- **多 GPU**: Layer split / Row split 支持
- **精简代码**: 目标 < 15000 行

## 快速开始

### 编译

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### 运行

```bash
# 单 GPU
./ember -m /path/to/qwen3.gguf -p "你好"

# 多 GPU (双卡，按层分配)
./ember -m /path/to/qwen3.gguf -ngl -1 --split-mode layer --tensor-split 0.6,0.4 -p "你好"

# 交互模式
./ember -m /path/to/qwen3.gguf -i
```

## CLI 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-m, --model` | GGUF 模型路径 | 必需 |
| `-ngl, --n-gpu-layers` | GPU 层数 (-1=全部) | -1 |
| `--main-gpu` | 主 GPU 索引 | 0 |
| `--split-mode` | 分配模式 (none/layer/row) | layer |
| `--tensor-split` | GPU 显存比例 | 自动 |
| `-c, --ctx-size` | 上下文长度 | 2048 |
| `-n, --n-predict` | 生成 token 数 | 128 |
| `--temp` | Temperature | 0.7 |
| `-i, --interactive` | 交互模式 | false |
| `-v, --verbose` | 详细日志 | false |

## 项目结构

```
ember/
├── include/          # 头文件
│   ├── cli.h         # CLI 参数
│   ├── model.h       # 模型加载
│   ├── inference.h   # 推理逻辑
│   └── utils.h       # 工具函数
├── src/              # 实现
│   ├── main.cpp      # 入口
│   ├── cli.cpp
│   ├── model.cpp
│   ├── inference.cpp
│   └── utils.cpp
├── ggml/             # GGML 依赖 (从 llama.cpp 提取)
└── tests/            # 测试
```

## 开发状态

### M1: 基础框架 ✅
- [x] CLI 参数解析
- [x] GGUF 文件头解析
- [x] 模型架构验证
- [x] 多 GPU 层分配计算
- [x] 日志和错误处理

### M2: 核心推理 🚧
- [ ] 集成 GGML 库
- [ ] 实现 Qwen3 前向计算
- [ ] KV Cache 管理
- [ ] CUDA Kernel 调用

### M3: 完善功能
- [ ] 完整 Tokenizer
- [ ] 性能优化
- [ ] 稳定性测试

## 与 llama.cpp 的关系

Ember 从 llama.cpp 提取核心组件：
- `ggml.c/h`: 张量基础设施
- `ggml-cuda.cu`: CUDA 后端
- `gguf.c/h`: 模型格式解析

去除了：
- CPU/Metal/Vulkan 等非 CUDA 后端
- 非 Qwen3 模型支持
- Server HTTP 接口
- 大量兼容性代码

## License

MIT
