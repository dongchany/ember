# Ember - Qwen3 CUDA Inference Engine

A lightweight CUDA inference engine for Qwen3 models, designed for consumer multi-GPU setups (for example, dual RTX 3080 Ti).

## Roadmap

As of **March 12, 2026**, Ember is focused on a narrow goal: high-performance Qwen inference on consumer NVIDIA GPUs.

Current:
- Stable Qwen3 dense CUDA inference
- Native safetensors loading
- Multi-GPU pipeline-parallel runtime
- Minimal CLI and server path

Next:
- Qwen3.5 hybrid architecture support
- DeltaNet + Gated Attention runtime
- HF-aligned correctness and regression harness

Then:
- Qwen3.5 35B-A3B MoE inference
- Dual-GPU + CPU offload execution path
- Reproducible benchmarks and OpenAI-compatible serving

## Quick Start

1) Build (example: RTX 3080 Ti, `SM=86`):

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

By default, Ember builds only the core inference target (`ember`).
To also build legacy tests/examples/benchmarks:

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DEMBER_BUILD_TESTS=ON \
  -DEMBER_BUILD_EXAMPLES=ON \
  -DEMBER_BUILD_BENCHMARKS=ON
```

2) Download a model (safetensors format; first run may require `huggingface-cli login`):

```bash
# If huggingface-cli is missing: pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-0.6B --local-dir ./qwen3-0.6b
```

3) Run:

```bash
./build/ember -m ./qwen3-0.6b -p "Hello, my name is"
```

## Documentation

English (default):
- [Contributing](CONTRIBUTING.md)
- [Development Guide](docs/development.md)
- [Testing and Regression](docs/testing.md)
- [Architecture Overview](docs/architecture.md)
- [Sampler Deep Dive](docs/sampler_explanation.md)
- [Benchmark Handbook](benchmarks/README.md)
- [Legacy Archive](legacy/README.md)

Chinese:
- [README (Chinese)](README.zh.md)
- [Sampler Deep Dive (Chinese)](docs/sampler_explanation.zh.md)
- [Benchmark Handbook (Chinese)](benchmarks/README.zh.md)

## Features

- **Native CUDA implementation**: full control over compute flow, without ggml/llama.cpp.
- **Direct safetensors loading**: loads HuggingFace format natively, no conversion step.
- **Pipeline parallelism**: multi-GPU layer split with memory-aware allocation.
- **FP16 compute**: FP16 weights/activations, GEMM via cuBLAS.
- **Custom kernels**: RMSNorm, RoPE, Softmax, SiLU, Attention, and more.

## Project Structure

```text
ember/
|-- apps/ember_cli/         # CLI entry point (main.cpp)
|-- cli/                    # CLI argument parsing
|-- core/                   # Core abstractions (pure C++, no CUDA dependency)
|-- runtime/                # Scheduling and device mapping runtime logic
|-- formats/                # safetensors/config loaders
|-- backends/cuda/          # CUDA runtime + kernels
|-- examples/               # Minimal runnable examples
|-- benchmarks/             # Performance and interconnect benchmarks
|-- tests/                  # Unit tests and smoke tests
|-- scripts/                # CI and alignment scripts
|-- docs/                   # Current design and testing docs
`-- legacy/                 # Archived old-stage docs and scripts
```

## Build

### Requirements

- CMake 3.18+
- CUDA Toolkit 11.0+ (12.x recommended)
- C++17 compiler

### Compile

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release  # RTX 3080 Ti
cmake --build build --parallel
```

Common CUDA architecture values:
- `86` - RTX 3080/3090
- `89` - RTX 4090
- `80` - A100

## Usage

### Basic Examples

```bash
# Single-GPU inference
./build/ember -m /path/to/qwen3-4b -p "Hello, my name is"

# Dual-GPU inference
./build/ember -m /path/to/qwen3-14b --devices 0,1 -p "Explain quantum computing"

# Interactive mode
./build/ember -m /path/to/qwen3-4b -i
```

### CLI Arguments

| Argument | Description | Default |
|------|------|--------|
| `-m, --model` | Model directory (must contain safetensors and `config.json`) | Required |
| `-p, --prompt` | Input prompt | `"Hello, my name is"` |
| `--devices` | GPU device list | `0` |
| `-c, --ctx-size` | Context length | `2048` |
| `-n, --n-predict` | Number of generated tokens | `128` |
| `--temp` | Temperature | `0.7` |
| `--top-p` | Top-P | `0.9` |
| `--top-k` | Top-K | `40` |
| `-i, --interactive` | Interactive mode | `false` |
| `-v, --verbose` | Verbose output | `false` |

## Supported Models

- Qwen3-0.6B
- Qwen3-1.7B
- Qwen3-4B
- Qwen3-8B
- Qwen3-14B (typically needs dual GPUs)

Models should be downloaded from HuggingFace in safetensors format:

```bash
# If huggingface-cli is missing: pip install -U huggingface_hub
huggingface-cli download Qwen/Qwen3-4B --local-dir ./qwen3-4b
```

## Architecture

### Core Abstractions

1. **Tensor**: lightweight view (`shape + dtype + data ptr`), no ownership.
2. **Session**: inference session state including KV cache.
3. **IRuntime**: backend interface (CUDA now, extensible in future).
4. **DeviceMap**: layer-to-device mapping for pipeline parallelism.

### Compute Flow

```text
Input IDs
    |
    v
Embedding Lookup (GPU 0)
    |
    v
+-----------------------------+
| Layer 0-N (may span GPUs)   |
|   |- Input LayerNorm        |
|   |- QKV Projection         |
|   |- RoPE                   |
|   |- KV Cache Update        |
|   |- Attention (Q@K^T -> V) |
|   |- O Projection           |
|   |- Residual Add           |
|   |- Post-Attn LayerNorm    |
|   |- MLP (SwiGLU)           |
|   `- Residual Add           |
+-----------------------------+
    |
    v
Final LayerNorm
    |
    v
LM Head -> Logits
    |
    v
Sampling -> Next Token
```

### Multi-GPU Strategy

Ember uses layer-wise pipeline parallelism. Example split:

```text
GPU 0: Embedding + Layers 0-13
GPU 1: Layers 14-27 + LM Head
```

Hidden states are transferred with `cudaMemcpyPeer`.

## Performance

Actual throughput depends on hardware and model size.

| Model | Approx. VRAM | Expected Speed |
|------|----------|----------|
| Qwen3-4B (FP16) | ~8 GB | ~40 tok/s |
| Qwen3-8B (FP16) | ~16 GB | ~25 tok/s |
| Qwen3-14B (FP16) | ~28 GB | ~15 tok/s (dual GPU) |

## Roadmap

- [x] M0: Single-GPU inference baseline
- [x] M1: Dual-GPU pipeline parallelism
- [ ] M2: Quantization support (INT8/INT4)
- [ ] M3: FlashAttention optimization

## Citation

If you use Ember in a paper or report, cite the Zenodo archive:

- DOI: https://doi.org/10.5281/zenodo.18477269

## License

Apache-2.0

## Acknowledgements

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - architecture inspiration
- [HuggingFace Transformers](https://github.com/huggingface/transformers) - model format ecosystem
- [Qwen](https://github.com/QwenLM/Qwen) - model weights
