# Benchmarks 操作手册

这份文档覆盖以下 6 个 benchmark：

- `benchmarks/kernel_bench.cu` → `build/ember_kernel_bench`
- `benchmarks/p2p_bandwidth.cpp` → `build/ember_p2p_bandwidth`
- `benchmarks/phase_analysis.cpp` → `build/ember_phase_analysis`
- `benchmarks/e2e_benchmark.cpp` → `build/ember_benchmark`
- `benchmarks/stage_breakdown.cpp` → `build/ember_stage_breakdown`
- `benchmarks/serve_benchmark.cpp` → `build/ember_serve_benchmark`

## 为什么要写这份手册（以及为什么要这样做）

1) **让结果可复现**：benchmark 的参数（prompt/gen/chunk、GPU 切分、warmup/iters）会显著影响结论；写清“怎么跑”和“输出代表什么”，才能让不同机器/不同人跑出来的结果可对比。  
2) **把“测什么”与“为什么测”对齐**：这 5 个 benchmark 分别覆盖 *跨 GPU 传输*、*按层耗时*、*端到端 TTFT/吞吐*、*分 stage 的瓶颈定位*、*连续 batch 的服务形态*。组合起来才足以解释双卡 pipeline 的收益/瓶颈。  
3) **减少试错成本**：避免“跑错 binary / 参数没对 / CSV 列意义误读 / batch>1 OOM”这类反复踩坑。

## 构建与运行前准备

### 构建（Release 推荐）

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j
```

构建后可执行文件在 `build/` 下。

### 模型目录要求（`--model`）

`--model <dir>` 需要是一个可直接加载的 snapshot 目录：

- 存在 `config.json`
- 同目录下存在 `*.safetensors`

## 0) `ember_kernel_bench`（Kernel Microbenchmark / Roofline 输入）

用途：对 **CUDA kernels** 做 microbenchmark，输出每个测试的：

- `elapsed_us`：单次 kernel 的中位数耗时（默认 warmup=10，iters=100）
- `bytes_moved`：按理论 IO 口径估算的搬运字节数（用于带宽/roofline 估算）
- `effective_gbps`：`bytes_moved / elapsed`
- `efficiency_pct`：相对 `--hw-bw`（默认 912 GB/s）的带宽利用率

常用命令：

```bash
./build/ember_kernel_bench --dtype f16
./build/ember_kernel_bench --dtype bf16
./build/ember_kernel_bench --dtype f16 --csv /tmp/kernel_bench.csv
```

注意：embedding microbench 会做大显存分配，默认关闭；需要时加 `--include-embedding`。

## 1) `ember_p2p_bandwidth`（跨 GPU 传输带宽/延迟）

用途：测量 GPU↔GPU 的 P2P copy 能力（`cudaMemcpyPeerAsync`），并对比“经由 Host staging（D2H+H2D）”的退化路径。用于：

- 判断 NVLink/PCIe 拓扑下的传输上限
- 为 pipeline 模型估算“激活跨卡”成本提供数据（见 `scripts/report/run_report.py` 的 bubble sim）

### 常用命令

```bash
./build/ember_p2p_bandwidth --gpus 0,1 --sizes 1k,10k,100k,1m,10m,100m --method both --direction both
```

写入 CSV：

```bash
./build/ember_p2p_bandwidth --gpus 0,1 --sizes 1m,10m,100m --csv /tmp/p2p.csv
```

### 关键参数

- `--gpus A,B`：必须是两个 GPU id（默认 `0,1`）
- `--sizes LIST`：逗号分隔，支持 `k/m/g` 后缀（必填）
- `--iters/--warmup`：计时迭代/预热次数
- `--method both|p2p|staged`：对比 P2P 与 staging
- `--direction both|a2b|b2a`：双向或单向
- `--hidden-sizes LIST`：额外输出 “FP16 激活每 token 传输字节数 = hidden_size * 2”

### 输出（CSV）

表头：`data_size_bytes,transfer_time_us,bandwidth_gbps,direction,method`

## 2) `ember_phase_analysis`（按层 profile：prefill vs decode）

用途：对单卡的 `prefill()` 与 `decode()` 做**逐层耗时**统计，并给出基于简化公式的“有效 TFLOPs / 带宽”估算，常用于：

- 判断瓶颈是 attention 还是 FFN
- 为双卡 layer split 选择提供依据（哪一段层更“重”）
- 为 pipeline bubble/利用率模型提供 per-layer 时延输入

### 常用命令

```bash
./build/ember_phase_analysis --model /path/to/model --prompt-lens 128,512,1024,2048 --decode-steps 100 --device 0
```

写入 CSV：

```bash
./build/ember_phase_analysis --model /path/to/model --prompt-lens 128,512,1024 --output /tmp/phase.csv
```

### 关键参数

- `--prompt-lens LIST`：必填；每个长度都会完整跑一次 prefill + 多步 decode
- `--decode-steps N`：每个 prompt_len 跑多少步 decode（默认 100）
- `--warmup N`：每个 prompt_len 的预热轮数（默认 1）
- `--device ID`：单卡设备 id（默认 0）

### 输出（CSV）

表头：`prompt_len,layer_id,prefill_time_ms,decode_time_ms,prefill_tflops,decode_tflops,prefill_bandwidth,decode_bandwidth`

注意：`*_tflops/*_bandwidth` 来自简化估算（用于横向对比/定位瓶颈），不等价于硬件理论峰值测量。

## 3) `ember_benchmark`（端到端：TTFT / prefill / decode 吞吐）

用途：做 *prefill + decode* 的端到端计时，输出一行 CSV，适合做 A/B 对比（overlap、chunk_len、split、decode_batch 等）。

### 常用命令

单请求（decode_batch=1）：

```bash
./build/ember_benchmark --model /path/to/model --gpus 0,1 --prompt-len 1024 --gen-len 100 --chunk-len 128 --iters 3 --overlap
```

强制使用 2-GPU chunked pipeline（即使不开 overlap）：

```bash
./build/ember_benchmark --model /path/to/model --gpus 0,1 --chunk-len 128 --iters 3 --no-overlap --pipeline
```

### 关键参数

- `--gpus 0` 或 `--gpus 0,1`：当前仅支持 1/2 卡
- `--split A,B`：两卡切分层数（默认平均切分，且要求 `A+B=num_layers`）
- `--prompt-len/--gen-len`：prefill token 数 / decode 生成 token 数
- `--chunk-len`：prefill chunk 大小（与 overlap/pipeline 联动）
- `--overlap/--no-overlap`：是否启用 prefill chunk overlap
- `--pipeline/--no-pipeline`：是否强制走 chunked pipeline 路径
- `--decode-batch N`：decode batch（默认 1）
- `--phase-aware`：prefill 使用 `PhaseAwareScheduler`

### 输出（单行 CSV）

列：`mode,prompt_len,gen_len,chunk_len,batch_size,ttft_ms,prefill_ms,decode_ms,decode_tok_s`

备注：

- `ttft_ms` 仅对 `decode_batch=1` 有意义（benchmark 内部也这样定义）。
- `decode_batch>1` 时走 batch 实验路径；长 prompt 下可能更容易 OOM（`scripts/report/run_report.py` 也默认避免把它当作 “serve 形态”）。

## 4) `ember_stage_breakdown`（分 stage 的瓶颈定位）

用途：在 CUDA runtime 内启用 stage profiling，把 prefill 与 decode 分别拆成：

- `embedding`
- `rmsnorm`
- `attention`
- `ffn`
- `p2p`（跨 GPU 传输/同步开销）
- `memcpy_h2d` / `memcpy_d2h`（Host<->Device 拷贝）
- `sampling`（仅在 `--decode-with-sampling` 时统计）
- `lm_head`

适合回答“时间到底花在了哪里？”以及“overlap 是否真的把 p2p 藏起来了？”。

### 常用命令

```bash
./build/ember_stage_breakdown --model /path/to/model --gpus 0,1 --prompt-len 2048 --decode-steps 256 --chunk-len 512 --iters 3 --overlap
```

写入 CSV：

```bash
./build/ember_stage_breakdown --model /path/to/model --gpus 0,1 --csv /tmp/stage.csv
```

包含 sampling（更接近真实 rollout 路径）：

```bash
./build/ember_stage_breakdown --model /path/to/model --gpus 0,1 --decode-with-sampling --csv /tmp/stage_rollout.csv
```

### 输出（CSV）

两行（或三行）输出：

- `phase=prefill`：prefill 阶段各 stage 的平均耗时（按 `iters` 平均）
- `phase=decode_per_token`：decode 每 token 的平均 stage 耗时（按 `iters * decode_steps` 平均）

表头：

`phase,mode,gpus,split,prompt_len,decode_steps,chunk_len,overlap,decode_sampling,wall_ms,embedding_ms,rmsnorm_ms,attention_ms,ffn_ms,p2p_ms,memcpy_h2d_ms,memcpy_d2h_ms,sampling_ms,lm_head_ms,profile_total_ms`

## 5) `ember_serve_benchmark`（连续 batching 服务形态模拟）

用途：模拟持续到达的请求，在固定 `batch_size`（最多并发 slot）下，用 `PhaseAwareBatchScheduler` 做 prefill admission + decode step，输出整体吞吐。

它更接近“服务端连续 batching”的负载形态，区别于 `ember_benchmark` 的“固定 prompt + 固定 gen 的单/小 batch 微基准”。

### 常用命令

```bash
./build/ember_serve_benchmark --model /path/to/model --gpus 0,1 --batch-size 8 --num-req 32 --prompt-len 1024 --gen-len 64
```

关闭每个请求 gen_len 的扰动（方便对齐固定长度对比）：

```bash
./build/ember_serve_benchmark --model /path/to/model --no-vary-gen
```

### 输出（单行 CSV）

列：`mode,num_reqs,batch_size,prompt_len,gen_len,vary_gen,prefill_ms,decode_ms,gen_tokens,decode_tok_s`

说明：

- `prefill_ms` 是所有成功 `submit()` 的累计耗时（包含 admission/可能的预处理）
- `decode_ms` 是 `step()` 循环累计耗时
- `decode_tok_s = gen_tokens / decode_ms`

## 推荐：用脚本一键跑并生成报告

如果你想要“参数统一 + 自动落盘 CSV + 生成 Markdown 报告”，直接用：

```bash
python3 scripts/report/run_report.py --hub-root ~/huggingface/hub --gpus 0,1 --model-b Qwen3-8B
```

它会自动调用上述 benchmarks（并额外做一些 A/B/C/D 阶段实验组合），输出到 `reports/<timestamp>/`。
