# Contributing to Ember

Thanks for your interest in Ember.

Contributions of all kinds are welcome, including bug reports, docs fixes, tests, benchmarks, and kernel or runtime work.

## Project Focus

As of March 8, 2026, Ember is production-focused on Qwen3 FP16/BF16 inference.

Active contribution areas include:

- Bug reports and GPU compatibility results
- Documentation and examples
- CUDA kernel and backend improvements
- Tests and benchmarks
- Roadmap work such as quantization, FlashAttention, and Qwen3.5 support

## Ways to Contribute

### Bug Reports and Compatibility Notes

Testing Ember on different hardware is useful even if you are not sending a code change.

Useful details to include:

- GPU model and GPU count
- CUDA toolkit version
- NVIDIA driver version
- Build flags such as `CMAKE_CUDA_ARCHITECTURES`
- Model used
- Exact command line
- What you expected and what happened instead

### Documentation and Examples

You can improve existing docs, add usage examples, clarify error cases, or translate content.

### Kernel and Runtime Work

CUDA kernels live under `backends/cuda/`.
Core C++ abstractions live under `core/`.
Runtime scheduling and device mapping logic live under `runtime/`.

If you are making a non-trivial runtime or kernel change, open an issue first so the approach can be discussed before implementation.

## Getting Started

1. Fork the repository and clone your fork.
2. Build Ember by following the [README quick start](README.md#quick-start) or the [development guide](docs/development.md).
3. If you need tests, configure CMake with `-DEMBER_BUILD_TESTS=ON`.
4. Create a branch for your change: `git checkout -b my-change`.
5. Make your changes and test locally.
6. Open a pull request against `main`.

Example build with tests enabled:

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_BUILD_TYPE=Release \
  -DEMBER_BUILD_TESTS=ON
cmake --build build --parallel
```

## Local Validation

For the current validation ladder, see [docs/testing.md](docs/testing.md).

Common local checks:

```bash
./build/ember_tests
ctest --test-dir build
./build/ember_cuda_kernels_smoke
MODEL_PATH="/path/to/model" ./build/ember_cuda_runtime_smoke
```

Not every change needs every check, but PRs should run the relevant tests for the area they modify.

## Code Style

- Use C++17.
- Match the existing code style and file organization.
- Keep commits focused to one logical change when possible.
- Add tests for new behavior when practical.

General layout:

- `core/`: pure C++ abstractions and configuration
- `runtime/`: scheduling and runtime coordination
- `backends/cuda/`: CUDA runtime and kernels
- `tests/`: unit tests and smoke tests

## Pull Request Guidelines

- Describe what the change does and why it is needed.
- Keep PRs small and reviewable when possible.
- Make sure existing tests still pass for the area you touched.
- If you change runtime or kernel behavior, run the relevant checks from [docs/testing.md](docs/testing.md).
- Update documentation when user-visible behavior changes.

## Communication

- GitHub Issues: bug reports, feature requests, design discussions
- Pull Requests: code and documentation contributions

## License

By contributing, you agree that your contributions will be licensed under the [Apache-2.0 License](LICENSE).
