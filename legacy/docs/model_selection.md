# Ember Model Selection (2x RTX 3080 Ti)

This note defines the model selection baseline for Ember experiments on:

- GPU: `2 x RTX 3080 Ti (12 GiB each)`
- Interconnect: `PCIe 4.0`
- Effective constraint: per-GPU memory ceiling is `12 GiB` (not pooled VRAM)

## Local Snapshot Ground Truth

The following values are read from local HuggingFace snapshots under
`~/xilinx/huggingface/hub` (`config.json` + `.safetensors` total bytes):

| model | hidden | layers | attn_heads | kv_heads | intermediate | head_dim | max_position_embeddings | tie_word_embeddings | dtype | weight size (GiB, safetensors) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| Qwen3-0.6B | 1024 | 28 | 16 | 8 | 3072 | 128 | 40960 | true | bf16 | 1.40 |
| Qwen3-1.7B | 2048 | 28 | 16 | 8 | 6144 | 128 | 40960 | true | bf16 | 3.78 |
| Qwen3-4B-Instruct-2507 | 2560 | 36 | 32 | 8 | 9728 | 128 | 262144 | true | bf16 | 7.49 |
| Qwen3-8B | 4096 | 36 | 32 | 8 | 12288 | 128 | 40960 | false | bf16 | 15.26 |
| Qwen3-14B | 5120 | 40 | 40 | 8 | 17408 | 128 | 40960 | false | bf16 | 27.51 |

Notes:

- Qwen3 KV shape depends on `layers / kv_heads / head_dim`, not hidden size directly.
- Therefore, 4B and 8B have very similar KV cache footprint at same sequence length.

## KV Cache Formula

Per request (batch = 1), fp16/bf16:

`KV bytes = layers * seq_len * kv_heads * head_dim * 2 (K+V) * 2 bytes`

Reference values:

| model | seq=2048 | seq=4096 |
| --- | ---: | ---: |
| Qwen3-0.6B / 1.7B (28 layers) | 224 MiB | 448 MiB |
| Qwen3-4B / 8B (36 layers) | 288 MiB | 576 MiB |
| Qwen3-14B (40 layers) | 320 MiB | 640 MiB |

## Recommendation

### Primary: `Qwen/Qwen3-4B-Instruct-2507`

- Best fit for current hardware and experiment goals.
- Enough model capacity for extraction quality while keeping 2-GPU runs practical.
- 36 layers are useful for update-locality sweeps.

### Secondary: `Qwen/Qwen3-1.7B`

- Fast debug model for Stage-0 pipeline bring-up and iteration.
- Suitable as scaling companion in tables/appendix.

### Inference-only reference: `Qwen/Qwen3-8B`

- Keep as upper-bound inference baseline.
- On this hardware, long-context combinations can hit memory boundary, especially no-overlap paths.

## What This Means For Current Ember Milestones

1. Stage profiling and cache-policy mainline should use 4B as default.
2. 1.7B should be used for quick regression checks before expensive 4B reruns.
3. 8B should be limited to selected inference-only points.

## Suggested IDs

- Primary: `Qwen/Qwen3-4B-Instruct-2507`
- Secondary: `Qwen/Qwen3-1.7B`
- Reference: `Qwen/Qwen3-8B`

## Repro Commands

Use local HF cache only:

```bash
HF_HOME=/home/dong/xilinx/huggingface \
scripts/report/run_stage1_milestone.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --gpus 0,1
```

```bash
HF_HOME=/home/dong/xilinx/huggingface \
scripts/report/run_stage1_milestone.sh \
  --model Qwen/Qwen3-1.7B \
  --gpus 0,1
```

## Stage 1.2 Split Sweep (with Delta/P2 Output)

```bash
HF_HOME=/home/dong/xilinx/huggingface \
scripts/report/run_stage1_split_profile.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --gpus 0,1 \
  --include-llama \
  --llama-bin-dir /path/to/llama.cpp/build/bin \
  --llama-gguf /path/to/model.gguf \
  --baseline-summary /path/to/prev/stage12_split_summary.csv \
  --anchor-summary /path/to/anchor/stage12_split_summary.csv \
  --p2-note "one-line implementation note"
```

Outputs (in `reports/stage1_split_profile_<timestamp>/`):

- `stage12_delta_vs_<baseline>.csv/md` (per-split old->new delta)
- `stage12_p2_input.md` (best config + delta summary for stage1.2 doc input)
