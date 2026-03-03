# LoRA Adapter Quickstart

目标：快速得到一个**真实 PEFT LoRA adapter**，用于 Ember 的热更新 benchmark。

## 1) 训练最小 LoRA adapter

### 依赖安装（仅首次）

```bash
pip install torch transformers peft accelerate sentencepiece
```

### 方式 A：直接用内置合成数据（最快）

```bash
scripts/train/run_min_lora.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --output-dir reports/adapters/qwen3_4b_min \
  --max-steps 50 \
  --max-length 256 \
  --per-device-batch-size 1 \
  --grad-acc-steps 8
```

### 方式 B：使用你自己的 `jsonl` 数据

`jsonl` 每行至少包含一组 prompt/response（字段名支持多种别名）：

- prompt: `prompt` / `instruction` / `input` / `question`
- response: `response` / `output` / `answer`

```bash
scripts/train/run_min_lora.sh \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --dataset-jsonl /abs/path/train.jsonl \
  --output-dir reports/adapters/qwen3_4b_real \
  --max-steps 100 \
  --max-length 512
```

训练完成后，`--output-dir` 下应有：

- `adapter_config.json`
- `adapter_model.safetensors`

## 2) 跑 Ember LoRA 热更新 benchmark

```bash
python3 scripts/report/run_stage1_lora_hot_update.py \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --adapter reports/adapters/qwen3_4b_min \
  --gpus 0,1 \
  --split 9,27 \
  --iters 3 \
  --warmup 1 \
  --out-dir reports/stage31_lora_hot_update_qwen3_4b_real
```

## 3) 常见注意事项

- Qwen3-4B 训练仍有显存压力，建议先用较小 `max-length`（256）和较少 `max-steps`。
- 如果你的数据质量一般，先追求“能产出 adapter 并打通流程”，再逐步提升训练设置。
