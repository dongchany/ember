# Ember Architecture Decisions

## AD-001: No custom programming language
- **Decision**: Use Triton + Python + C++ instead of building a language like Mojo
- **Why**: 8-13 person-years to build a language; Triton covers 90%+ of kernel needs
- **Trade-off**: Lose automatic kernel introspection; mitigate with structured kernel DSL (Phase 2)

## AD-002: IREE as graph compiler (not TVM, not torch.compile)
- **Decision**: Fork IREE, add custom MLIR passes for LLM optimization
- **Why**: MLIR-native (interops with Triton/Torch-MLIR natively); best HAL abstraction; compiler+runtime integrated
- **Not TVM**: TVM's IR (Relay/Relax) is not MLIR; needs conversion layers
- **Not torch.compile**: Coupled to PyTorch runtime; can't deploy independently

## AD-003: SGLang as serving base (not vLLM)
- **Decision**: Fork SGLang, replace model execution layer with Ember Pipeline
- **Why**: Cleaner codebase; RadixAttention; easier to swap out the backend
- **Not vLLM**: PyTorch deeply coupled; model_runner/worker/executor all bound to PyTorch

## AD-004: Bazel build system (not CMake)
- **Decision**: Use Bazel for multi-language build (C++, CUDA, Python, Triton)
- **Why**: Modular uses Bazel; IREE supports Bazel; best multi-language support

## AD-005: nanobind for C++/Python binding (not pybind11)
- **Decision**: Use nanobind
- **Why**: Faster compile, smaller binaries, better performance; Modular also uses it

## AD-006: AI-driven Auto-Evolve system
- **Decision**: Build an AI optimization loop that continuously improves kernels/passes
- **Why**: Kernel optimization is "easy to generate, easy to verify" - perfect for AI
- **Key**: The system is model-agnostic; as LLMs improve, Ember auto-improves
