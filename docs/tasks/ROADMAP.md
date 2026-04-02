# Ember Roadmap

## Milestone 1: Kernel Layer
- [ ] Triton GEMM kernel (FP16/BF16) with autotune
- [ ] Triton GEMM benchmark vs cuBLAS on H100
- [ ] Fused RMSNorm + Residual Triton kernel
- [ ] Fused RoPE Triton kernel
- [ ] Fused Sampling kernel (top-k/top-p + temperature)
- [ ] Verify same kernel runs on NVIDIA and AMD

## Milestone 2: Graph Compiler
- [ ] IREE fork setup (ember-iree, stripped backends)
- [ ] Torch-MLIR import: Llama-3-8B → StableHLO → IREE → correct tokens
- [ ] First custom MLIR Pass: RMSNorm + Residual fusion
- [ ] Attention pattern → FlashAttention dispatch Pass
- [ ] Compilation cache (content hash)
- [ ] Benchmark: compiled vs PyTorch eager (target: +20% speedup)

## Milestone 3: Graph API + NN Module
- [ ] SymbolicShape system
- [ ] GraphBuilder core (input, matmul, custom_op, compile)
- [ ] Custom op mechanism (Triton kernel registration)
- [ ] NN Module: Transformer, Attention, RoPE, RMSNorm, FeedForward
- [ ] WeightSource: from_huggingface, from_safetensors
- [ ] Build Llama-3-8B with Graph API + NN Module, verify output

## Milestone 4: KV Cache + Pipeline
- [ ] PagedCacheStrategy + CacheManager + CacheCollection
- [ ] Multi-tier eviction (GPU → CPU)
- [ ] TextGenerationPipeline (tokenize → prefill → decode → detokenize)
- [ ] PipelineRegistry (HuggingFace architecture → pipeline mapping)
- [ ] SpeculativeDecodingPipeline
- [ ] ConstrainedDecodingPipeline (XGrammar)

## Milestone 5: Serve
- [ ] SGLang fork setup (ember-sglang)
- [ ] Replace model execution backend with Ember Pipeline
- [ ] 3-process architecture (API / Model / Agent workers)
- [ ] OpenAI-compatible API verification
- [ ] Benchmark vs SGLang native (target: < 10% gap)

## Milestone 6: Auto-Evolve
- [ ] Profile collector (kernel-level timing)
- [ ] Codex integration for kernel optimization
- [ ] Auto-verification pipeline (numerical + performance)
- [ ] Auto-PR creation for accepted optimizations
- [ ] Scheduled weekly optimization runs

## Milestone 7: End-to-End Validation
- [ ] Llama-3-8B (dense) full pipeline
- [ ] Mixtral-8x7B (MoE) full pipeline
- [ ] LLaVA (multimodal) full pipeline
- [ ] A/B benchmark vs vLLM and SGLang
