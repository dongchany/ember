# Development Guide

This guide helps contributors build, run, and modify Ember quickly.

## Build

Requirements:
- CMake 3.18+
- CUDA Toolkit 11.0+ (12.x recommended)
- C++17 compiler

Build:
```
mkdir -p build
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

CI-style build (isolated dir):
```
scripts/ci/build.sh
```

## Run

```
./build/ember -m /path/to/qwen3-0.6b -p "Hello, my name is"
```

## Code layout

- `main.cpp`: CLI entry point and sampling loop.
- `core/`: configs, session, sampler, tokenizer.
- `runtime/`: runtime interface, device map.
- `formats/`: safetensors loader.
- `backends/cuda/`: CUDA runtime and kernels.

## How to add features

1. Add CLI flags in `main.cpp`.
2. Add runtime config fields in `core/config.h`.
3. Implement kernel changes in `backends/cuda/kernels/`.
4. Wire into `backends/cuda/cuda_runtime.cpp`.
5. Update docs and tests.

## Local debug workflow

- Use `--check` to dump logits/hidden states:
```
./build/ember -m /path/to/model --check --dump-layer 2 -p "Hello"
```
- Compare with HuggingFace:
```
python3 scripts/compare_logits.py --model /path/to/model --debug-dir debug/check_...
python3 scripts/compare_hidden.py --model /path/to/model --debug-dir debug/check_... --layer 2
```

## First tasks for new contributors

- Change a CLI default (e.g., sampling) and update README.
- Add a small check to `compare_logits.py`.
- Improve a kernel comment or error message.
