# Benchmark Handbook

This document covers six benchmark binaries:

- `benchmarks/kernel_bench.cu` -> `build/ember_kernel_bench`
- `benchmarks/p2p_bandwidth.cpp` -> `build/ember_p2p_bandwidth`
- `benchmarks/phase_analysis.cpp` -> `build/ember_phase_analysis`
- `benchmarks/e2e_benchmark.cpp` -> `build/ember_benchmark`
- `benchmarks/stage_breakdown.cpp` -> `build/ember_stage_breakdown`
- `benchmarks/serve_benchmark.cpp` -> `build/ember_serve_benchmark`

## Why this handbook exists

1. Reproducibility: benchmark conclusions are sensitive to settings like
   prompt/gen length, chunk size, warmup/iters, and GPU split.
2. Coverage: each benchmark isolates a different bottleneck class
   (kernel efficiency, P2P transfer, layer timing, end-to-end throughput,
   stage-level attribution, and serving-like continuous batching).
3. Lower iteration cost: avoid common mistakes (wrong binary, wrong flags,
   misread CSV columns, OOM due to batch shape).

## Build and prerequisites

### Build (Release recommended)

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

Binaries are generated under `build/`.

### Model directory requirement (`--model`)

`--model <dir>` should point to a loadable snapshot directory containing:

- `config.json`
- one or more `*.safetensors` files in the same directory

## 0) `ember_kernel_bench` (kernel microbenchmark / roofline input)

Purpose: microbenchmark CUDA kernels and output:

- `elapsed_us`: median single-launch latency (default warmup=10, iters=100)
- `bytes_moved`: estimated data movement bytes
- `effective_gbps`: `bytes_moved / elapsed`
- `efficiency_pct`: efficiency relative to `--hw-bw` (default 912 GB/s)

Common commands:

```bash
./build/ember_kernel_bench --dtype f16
./build/ember_kernel_bench --dtype bf16
./build/ember_kernel_bench --dtype f16 --csv /tmp/kernel_bench.csv
```

Note: embedding microbench performs a large memory allocation and is disabled by
default. Enable it with `--include-embedding`.

## 1) `ember_p2p_bandwidth` (cross-GPU transfer bandwidth/latency)

Purpose: measure GPU<->GPU transfer using `cudaMemcpyPeerAsync`, and compare
with host-staging fallback (`D2H + H2D`). Useful for:

- estimating interconnect upper bound (NVLink/PCIe)
- estimating pipeline hidden-state transfer costs

Common command:

```bash
./build/ember_p2p_bandwidth --gpus 0,1 --sizes 1k,10k,100k,1m,10m,100m --method both --direction both
```

Write CSV:

```bash
./build/ember_p2p_bandwidth --gpus 0,1 --sizes 1m,10m,100m --csv /tmp/p2p.csv
```

Key options:

- `--gpus A,B`: two GPU ids (default `0,1`)
- `--sizes LIST`: comma-separated sizes with `k/m/g` suffixes
- `--iters`, `--warmup`: timing iterations and warmup rounds
- `--method both|p2p|staged`: compare direct P2P vs staged path
- `--direction both|a2b|b2a`: bidirectional or single direction
- `--hidden-sizes LIST`: report per-token FP16 activation bytes (`hidden * 2`)

CSV header:

`data_size_bytes,transfer_time_us,bandwidth_gbps,direction,method`

## 2) `ember_phase_analysis` (per-layer profile: prefill vs decode)

Purpose: profile per-layer runtime for single-GPU `prefill()` and `decode()`,
and report simplified effective TFLOPs/bandwidth estimates.

Use cases:

- identify whether attention or FFN dominates
- support dual-GPU layer split decisions
- provide layer-latency inputs for bubble/utilization models

Common command:

```bash
./build/ember_phase_analysis --model /path/to/model --prompt-lens 128,512,1024,2048 --decode-steps 100 --device 0
```

Write CSV:

```bash
./build/ember_phase_analysis --model /path/to/model --prompt-lens 128,512,1024 --output /tmp/phase.csv
```

Key options:

- `--prompt-lens LIST`: required; each length runs full prefill + decode steps
- `--decode-steps N`: decode steps per prompt length (default 100)
- `--warmup N`: warmup rounds per prompt length (default 1)
- `--device ID`: single GPU id (default 0)

CSV header:

`prompt_len,layer_id,prefill_time_ms,decode_time_ms,prefill_tflops,decode_tflops,prefill_bandwidth,decode_bandwidth`

Note: TFLOPs/bandwidth are simplified comparative estimates, not hardware peak
measurements.

## 3) `ember_benchmark` (end-to-end: TTFT / prefill / decode throughput)

Purpose: run end-to-end prefill + decode timing and emit one CSV row for A/B
comparisons (`overlap`, `chunk_len`, `split`, `decode_batch`, etc.).

Common commands:

Single request (`decode_batch=1`):

```bash
./build/ember_benchmark --model /path/to/model --gpus 0,1 --prompt-len 1024 --gen-len 100 --chunk-len 128 --iters 3 --overlap
```

Force 2-GPU chunked pipeline even without overlap:

```bash
./build/ember_benchmark --model /path/to/model --gpus 0,1 --chunk-len 128 --iters 3 --no-overlap --pipeline
```

Key options:

- `--gpus 0` or `--gpus 0,1`: currently supports 1 or 2 GPUs
- `--split A,B`: dual-GPU split (`A+B=num_layers`)
- `--prompt-len`, `--gen-len`: prefill tokens and decode tokens
- `--chunk-len`: prefill chunk size
- `--overlap`, `--no-overlap`: overlap prefill chunks or not
- `--pipeline`, `--no-pipeline`: force chunked pipeline path on/off
- `--decode-batch N`: decode batch size (default 1)
- `--phase-aware`: use `PhaseAwareScheduler` for prefill

CSV columns:

`mode,prompt_len,gen_len,chunk_len,batch_size,ttft_ms,prefill_ms,decode_ms,decode_tok_s`

Notes:

- `ttft_ms` is meaningful mainly for `decode_batch=1`.
- `decode_batch>1` is a batch experiment path; long prompts are more likely OOM.

## 4) `ember_stage_breakdown` (stage-level bottleneck attribution)

Purpose: enable runtime stage profiling and split prefill/decode into:

- `embedding`
- `rmsnorm`
- `attention`
- `ffn`
- `p2p` (cross-GPU transfer/sync)
- `memcpy_h2d`, `memcpy_d2h`
- `sampling` (with `--decode-with-sampling`)
- `lm_head`

Use this to answer:

- where time is actually spent
- whether overlap really hides P2P costs

Common command:

```bash
./build/ember_stage_breakdown --model /path/to/model --gpus 0,1 --prompt-len 2048 --decode-steps 256 --chunk-len 512 --iters 3 --overlap
```

Write CSV:

```bash
./build/ember_stage_breakdown --model /path/to/model --gpus 0,1 --csv /tmp/stage.csv
```

Include sampling (closer to interactive serving behavior):

```bash
./build/ember_stage_breakdown --model /path/to/model --gpus 0,1 --decode-with-sampling --csv /tmp/stage_rollout.csv
```

CSV output has two (or three) rows:

- `phase=prefill`: average stage latency over `iters`
- `phase=decode_per_token`: average per-token stage latency over
  `iters * decode_steps`

CSV header:

`phase,mode,gpus,split,prompt_len,decode_steps,chunk_len,overlap,decode_sampling,wall_ms,embedding_ms,rmsnorm_ms,attention_ms,ffn_ms,p2p_ms,memcpy_h2d_ms,memcpy_d2h_ms,sampling_ms,lm_head_ms,profile_total_ms`

## 5) `ember_serve_benchmark` (continuous batching service simulation)

Purpose: simulate continuously arriving requests with fixed `batch_size`
(max active slots) using `PhaseAwareBatchScheduler`, then report overall
throughput.

This is closer to serving behavior than `ember_benchmark`.

Common command:

```bash
./build/ember_serve_benchmark --model /path/to/model --gpus 0,1 --batch-size 8 --num-req 32 --prompt-len 1024 --gen-len 64
```

Disable per-request `gen_len` jitter for strict fixed-length comparisons:

```bash
./build/ember_serve_benchmark --model /path/to/model --no-vary-gen
```

CSV columns:

`mode,num_reqs,batch_size,prompt_len,gen_len,vary_gen,prefill_ms,decode_ms,gen_tokens,decode_tok_s`

Meaning:

- `prefill_ms`: cumulative submit/admission-side cost for accepted requests
- `decode_ms`: cumulative `step()` loop time
- `decode_tok_s = gen_tokens / decode_ms`

## Recommended: one-command report pipeline

If you want unified parameters, auto-saved CSVs, and a generated Markdown
summary, run:

```bash
python3 scripts/report/run_report.py --hub-root ~/huggingface/hub --gpus 0,1 --model-b Qwen3-8B
```

Outputs are written to `reports/<timestamp>/`.

## Stage 1.1 milestone full matrix

To run the full Stage 1.1 matrix
(prefill/decode/sampling/memcpy breakdown + summary table):

```bash
scripts/report/run_stage1_milestone.sh --model /path/to/model --gpus 0,1
```

You can also pass a HuggingFace model id (resolved from local cache only,
without downloading):

```bash
HF_HOME=~/huggingface scripts/report/run_stage1_milestone.sh --model Qwen/Qwen3-8B --gpus 0,1
```

Default matrix includes:

- `prompt_lens = 512,1024,2048,4096`
- `decode_steps = 64,128,256`
- for 2 GPUs, both `overlap=0` and `overlap=1`
- for 2 GPUs, pipeline path is forced by default (`--pipeline`)
- sampling enabled by default (`--decode-with-sampling`)

## Chinese version

The original Chinese handbook is preserved at:

- `benchmarks/README.zh.md`
