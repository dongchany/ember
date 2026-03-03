# Tutorial #15 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 15 篇。

## 项目简介
Ember (https://github.com/dongchany/ember) 是一个从零手写的 Qwen3 CUDA 推理引擎，
纯 C++ + CUDA，不依赖 ggml/llama.cpp。支持消费级多 GPU Pipeline Parallel（如双卡 RTX 3080Ti）。
除了推理，Ember 还支持完整的 RL 训练闭环：多候选 Rollout → Verifier/Reward → LoRA 热更新 →
Cache 策略复用，实现了统一后端（推理和训练共享同份权重），相比双栈方案节省 50% 显存。

## 项目 5 层结构
Layer 1: 推理引擎（CUDA kernels, Transformer forward, Pipeline Parallel）
Layer 2: Rollout 能力（多候选、logprobs、stop sequences）
Layer 3: LoRA 热更新 + Cache 策略（UpdateLocality / Prefix / Periodic / Hybrid）
Layer 4: 验证器 + Reward（Extraction / SQL verifier，字段级打分）
Layer 5: 训练闭环（SFT → Best-of-N → DPO → GRPO 可选）+ 统一后端 vs 双栈

## 写作硬性要求
1. 目标读者：想了解 LLM 内部原理的开发者，数学基础较弱也能看懂
2. 数学四步法：直觉 → 小例子手算 → 公式 → 对应 CUDA/训练代码
3. 语言：中文为主，术语和代码注释保留英文
4. 必须引用我提供的真实源码与真实报告，不得编造实验数字
5. 每篇开头必须写：源文件路径、前置知识、下一篇链接
6. 每篇结尾自然放 GitHub 链接：https://github.com/dongchany/ember
7. 风格：友好、像学长讲解，不要居高临下
8. 不要只列 bullet；以叙述为主

## 输出质量要求（必须遵守）
- 你只能使用我提供的“完整代码片段”和“完整报告片段”作为事实来源
- 所有结论都要标注来自哪个文件
- 任何数字都要能在报告中定位到
- 如果某结论缺证据，明确写“当前资料不足”

## 数学深度加严（额外要求）
- 在不影响可读性的前提下，尽量给出更详细的数学推导
- 对每个关键公式都解释“它在数值稳定性/并行实现上的意义”
- 允许在附录给出更完整推导（正文保持循序渐进）
```

---

## 1) 写作任务

```text
请写第 15 篇：SFT 基线 — 最简单的监督微调（QLoRA on 11GB）。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 推导 cross-entropy loss（先直觉后公式，尽量详细）
- 明确解释为何 FP16 SFT 在 11GB 上 OOM，而 QLoRA 可行
```

---

## 2) 代码上下文（完整/相关段落）

### File: scripts/train/run_stage52_sft_min.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from common_train import die, load_json, load_jsonl


def to_sft_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, r in enumerate(rows):
        rid = str(r.get("id", f"row_{i:06d}"))
        prompt = str(r.get("prompt", "")).strip()
        if not prompt:
            continue

        if isinstance(r.get("gold"), dict):
            response = json.dumps(r["gold"], ensure_ascii=False)
        elif "response" in r:
            response = str(r.get("response", ""))
        elif "output" in r:
            response = str(r.get("output", ""))
        else:
            continue
        response = response.strip()
        if not response:
            continue
        out.append({"id": rid, "prompt": prompt, "response": response})
    return out


def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_summary_md(path: Path, data: Dict[str, Any]) -> None:
    lines = [
        "# Stage 5.2 SFT Baseline (Minimal)",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Model: `{data.get('model', '')}`",
        f"- Dataset input: `{data.get('dataset_input', '')}`",
        f"- SFT train rows: `{data.get('num_train_rows', 0)}`",
        f"- Max steps: `{data.get('max_steps', 0)}`",
        f"- Max length: `{data.get('max_length', 0)}`",
        f"- LoRA r/alpha/dropout: `{data.get('lora_r', 0)}/{data.get('lora_alpha', 0)}/{data.get('lora_dropout', 0)}`",
        f"- load_in_4bit: `{data.get('load_in_4bit', False)}`",
        f"- bnb4bit: dtype=`{data.get('bnb_4bit_compute_dtype', '')}`, quant=`{data.get('bnb_4bit_quant_type', '')}`, double_quant=`{data.get('bnb_4bit_use_double_quant', True)}`",
        f"- Training loss: `{data.get('training_loss', 'n/a')}`",
        "",
        f"- Adapter dir: `{data.get('adapter_dir', '')}`",
        f"- Train log: `{data.get('train_log', '')}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5.2 minimal SFT baseline wrapper (LoRA).")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--dataset-jsonl", type=str, required=True, help='rows: {"id","prompt","gold"} or {"prompt","response"}')
    ap.add_argument("--python-bin", type=str, default="python3", help="python interpreter used to run train_min_lora_adapter.py")
    ap.add_argument("--max-train-samples", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-acc-steps", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--load-in-4bit", action="store_true", default=False)
    ap.add_argument("--bnb-4bit-compute-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb-4bit-use-double-quant", action="store_true", default=True)
    ap.add_argument("--no-bnb-4bit-use-double-quant", dest="bnb_4bit_use_double_quant", action="store_false")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    ds_path = Path(args.dataset_jsonl).expanduser().resolve()
    if not ds_path.exists():
        die(f"dataset not found: {ds_path}")
    if args.max_train_samples <= 0 or args.max_steps <= 0 or args.max_length <= 0:
        die("max_train_samples/max_steps/max_length must be > 0")

    rows = load_jsonl(ds_path)
    sft_rows = to_sft_rows(rows)
    if not sft_rows:
        die("no usable rows converted to SFT format")
    sft_rows = sft_rows[: args.max_train_samples]

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage52_sft_min_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    sft_jsonl = out_dir / "stage52_sft_train.jsonl"
    write_jsonl(sft_jsonl, sft_rows)

    adapter_dir = out_dir / "adapter"
    trainer_script = (Path(__file__).resolve().parent / "train_min_lora_adapter.py").resolve()
    cmd = [
        args.python_bin,
        str(trainer_script),
        "--model",
        args.model,
        "--dataset-jsonl",
        str(sft_jsonl),
        "--output-dir",
        str(adapter_dir),
        "--max-train-samples",
        str(len(sft_rows)),
        "--max-length",
        str(args.max_length),
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--grad-acc-steps",
        str(args.grad_acc_steps),
        "--max-steps",
        str(args.max_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--logging-steps",
        str(args.logging_steps),
        "--seed",
        str(args.seed),
        "--lora-r",
        str(args.lora_r),
        "--lora-alpha",
        str(args.lora_alpha),
        "--lora-dropout",
        str(args.lora_dropout),
        "--target-modules",
        args.target_modules,
        "--save-tokenizer",
    ]
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
        cmd += ["--bnb-4bit-compute-dtype", args.bnb_4bit_compute_dtype]
        cmd += ["--bnb-4bit-quant-type", args.bnb_4bit_quant_type]
        if args.bnb_4bit_use_double_quant:
            cmd.append("--bnb-4bit-use-double-quant")
        else:
            cmd.append("--no-bnb-4bit-use-double-quant")
    train_log = logs_dir / "train.log"
    with train_log.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        # Stream child logs directly into file to avoid pipe-buffer deadlock with tqdm progress output.
        p = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode != 0:
        die(f"sft training failed rc={p.returncode}; see {train_log}")

    train_meta = adapter_dir / "train_meta.json"
    if not train_meta.exists():
        die(f"missing train_meta.json: {train_meta}")
    meta = load_json(train_meta)
    loss = meta.get("training_loss", None)

    summary_json = out_dir / "stage52_sft_summary.json"
    summary_md = out_dir / "stage52_sft_summary.md"
    summary = {
        "model": args.model,
        "dataset_input": str(ds_path),
        "num_train_rows": len(sft_rows),
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "load_in_4bit": args.load_in_4bit,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
        "training_loss": loss,
        "adapter_dir": str(adapter_dir),
        "train_log": str(train_log),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary_md(summary_md, summary)

    print("[done] stage5.2 sft baseline")
    print(f"- out_dir: {out_dir}")
    print(f"- sft_jsonl: {sft_jsonl}")
    print(f"- adapter_dir: {adapter_dir}")
    print(f"- summary_md: {summary_md}")
    print(f"- training_loss: {loss}")


if __name__ == "__main__":
    main()

````

### File: scripts/train/common_train.py

````py
#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                o = json.loads(s)
            except Exception as e:
                die(f"{path}:{ln}: invalid json: {e}")
            if not isinstance(o, dict):
                die(f"{path}:{ln}: row must be object")
            out.append(o)
    return out


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_torch():
    try:
        import torch
    except Exception as e:
        die(f"torch is required for dtype resolution, but import failed: {e}")
    return torch


def dtype_from_name(name: str) -> Any:
    torch = _load_torch()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    die(f"unsupported dtype: {name}")
    raise AssertionError("unreachable")


def render_chat(tokenizer, prompt: str, response: Optional[str], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        msgs = [{"role": "user", "content": prompt}]
        if response is not None:
            msgs.append({"role": "assistant", "content": response})
        return tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    if response is None:
        return f"User: {prompt}\nAssistant:"
    return f"User: {prompt}\nAssistant: {response}"

````

### File: scripts/train/train_min_lora_adapter.py

````py
#!/usr/bin/env python3
"""
Train a minimal PEFT LoRA adapter for Qwen-style causal LMs.

Recommended invocation:
  python scripts/train/train_min_lora_adapter.py --output-dir reports/adapters/qwen3_4b_min
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Minimal LoRA finetune for producing a real adapter.")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--dataset-jsonl", type=str, default="", help="Optional jsonl with prompt/response fields.")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--max-train-samples", type=int, default=128)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--per-device-batch-size", type=int, default=1)
    ap.add_argument("--grad-acc-steps", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    ap.add_argument("--load-in-4bit", action="store_true", default=False, help="enable QLoRA-style 4-bit base loading")
    ap.add_argument("--bnb-4bit-compute-dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--bnb-4bit-quant-type", type=str, default="nf4", choices=["nf4", "fp4"])
    ap.add_argument("--bnb-4bit-use-double-quant", action="store_true", default=True)
    ap.add_argument("--no-bnb-4bit-use-double-quant", dest="bnb_4bit_use_double_quant", action="store_false")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    ap.add_argument("--save-tokenizer", action="store_true", default=True)
    ap.add_argument("--no-save-tokenizer", dest="save_tokenizer", action="store_false")
    return ap.parse_args()


def import_stack():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments, set_seed
    except Exception as ex:
        die(
            "missing Python deps. Install first, e.g.\n"
            "  pip install torch transformers peft accelerate sentencepiece bitsandbytes\n"
            f"details: {ex}"
        )
    return (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    )


def load_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            prompt = (
                obj.get("prompt")
                or obj.get("instruction")
                or obj.get("input")
                or obj.get("question")
                or ""
            )
            response = (
                obj.get("response")
                or obj.get("output")
                or obj.get("answer")
                or ""
            )
            if prompt and response:
                rows.append({"prompt": prompt, "response": response})
    return rows


def synth_examples(n: int, seed: int) -> List[Dict[str, str]]:
    random.seed(seed)
    templates = [
        (
            "从这段文本抽取 company 和 date，输出 JSON：{text}",
            '{{"company":"{company}","date":"{date}"}}',
        ),
        (
            "提取以下内容中的金额和币种，输出 JSON：{text}",
            '{{"amount":"{amount}","currency":"{currency}"}}',
        ),
        (
            "把下面句子改写成更正式的一句话：{text}",
            "{formal}",
        ),
        (
            "回答问题：{text}",
            "{answer}",
        ),
    ]
    companies = ["OpenAI", "NVIDIA", "Qwen Labs", "DeepSeek", "Anthropic"]
    dates = ["2026-02-10", "2025-12-01", "2024-07-16", "2023-11-02"]
    amounts = ["1200", "99.9", "5000000", "42"]
    currencies = ["USD", "CNY", "EUR", "JPY"]
    examples: List[Dict[str, str]] = []
    for i in range(n):
        t_idx = i % len(templates)
        t_prompt, t_resp = templates[t_idx]
        payload = {
            "text": f"样例文本 {i}",
            "company": random.choice(companies),
            "date": random.choice(dates),
            "amount": random.choice(amounts),
            "currency": random.choice(currencies),
            "formal": f"这是第 {i} 条正式改写句子。",
            "answer": f"这是问题 {i} 的简短回答。",
        }
        examples.append(
            {
                "prompt": t_prompt.format(**payload),
                "response": t_resp.format(**payload),
            }
        )
    return examples


def render_chat(tokenizer, prompt: str, response: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    return f"User: {prompt}\nAssistant: {response}"


def main() -> None:
    args = parse_args()
    if args.max_train_samples <= 0:
        die("--max-train-samples must be > 0")
    if args.max_length <= 0:
        die("--max-length must be > 0")
    if args.max_steps <= 0:
        die("--max-steps must be > 0")
    if args.per_device_batch_size <= 0 or args.grad_acc_steps <= 0:
        die("batch/grad-acc must be > 0")

    (
        torch,
        LoraConfig,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
    ) = import_stack()
    set_seed(args.seed)

    if args.dataset_jsonl:
        data = load_jsonl(Path(args.dataset_jsonl).expanduser().resolve())
        if not data:
            die(f"no usable rows in dataset: {args.dataset_jsonl}")
    else:
        data = synth_examples(max(args.max_train_samples, 32), seed=args.seed)
    data = data[: args.max_train_samples]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.load_in_4bit:
        if not torch.cuda.is_available():
            die("--load-in-4bit requires CUDA")
        if args.bnb_4bit_compute_dtype == "float16":
            bnb_compute_dtype = torch.float16
            use_bf16 = False
        elif args.bnb_4bit_compute_dtype == "bfloat16":
            bnb_compute_dtype = torch.bfloat16
            use_bf16 = True
        else:
            bnb_compute_dtype = torch.float32
            use_bf16 = False
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        if torch.cuda.is_available():
            use_bf16 = torch.cuda.is_bf16_supported()
            dtype = torch.bfloat16 if use_bf16 else torch.float16
        else:
            use_bf16 = False
            dtype = torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=dtype,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    rows: List[Dict[str, List[int]]] = []
    for ex in data:
        text = render_chat(tokenizer, ex["prompt"], ex["response"])
        tok = tokenizer(
            text,
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=True,
        )
        if not tok["input_ids"]:
            continue
        rows.append(
            {
                "input_ids": tok["input_ids"],
                "attention_mask": tok["attention_mask"],
                "labels": list(tok["input_ids"]),
            }
        )
    if not rows:
        die("empty tokenized dataset")

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]

    class DataCollator:
        def __init__(self, pad_token_id: int):
            self.pad_token_id = pad_token_id

        def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, "torch.Tensor"]:
            max_len = max(len(x["input_ids"]) for x in features)
            batch_ids = []
            batch_attn = []
            batch_lbls = []
            for x in features:
                n = len(x["input_ids"])
                pad_n = max_len - n
                batch_ids.append(x["input_ids"] + [self.pad_token_id] * pad_n)
                batch_attn.append(x["attention_mask"] + [0] * pad_n)
                batch_lbls.append(x["labels"] + [-100] * pad_n)
            return {
                "input_ids": torch.tensor(batch_ids, dtype=torch.long),
                "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
                "labels": torch.tensor(batch_lbls, dtype=torch.long),
            }

    run_dir = Path(args.output_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    work_dir = run_dir / "_trainer_workdir"
    work_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(work_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_acc_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to=[],
        bf16=use_bf16,
        fp16=(torch.cuda.is_available() and not use_bf16),
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=SimpleDataset(rows),
        data_collator=DataCollator(tokenizer.pad_token_id),
    )
    result = trainer.train()

    model.save_pretrained(str(run_dir), safe_serialization=True)
    if args.save_tokenizer:
        tokenizer.save_pretrained(str(run_dir))

    meta = {
        "model": args.model,
        "dataset_jsonl": args.dataset_jsonl,
        "num_examples": len(rows),
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": [x.strip() for x in args.target_modules.split(",") if x.strip()],
        "load_in_4bit": args.load_in_4bit,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
        "training_loss": float(result.training_loss) if result is not None else None,
    }
    (run_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[done] adapter saved to: {run_dir}")
    print(f"[done] examples={len(rows)} max_steps={args.max_steps} loss={meta['training_loss']}")


if __name__ == "__main__":
    main()

````

---

## 3) 报告上下文（完整）

### Report: reports/stage52_sft_min_4b_20260225_external_zip22_qlora_v1/stage52_sft_summary.md

````md
# Stage 5.2 SFT Baseline (Minimal)

- Generated at: `2026-02-26T00:02:09`
- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Dataset input: `/home/dong/workspace/ember/reports/stage52_external_from_zip22_20260225_235040/train.jsonl`
- SFT train rows: `400`
- Max steps: `60`
- Max length: `128`
- LoRA r/alpha/dropout: `8/16/0.05`
- load_in_4bit: `True`
- bnb4bit: dtype=`float16`, quant=`nf4`, double_quant=`True`
- Training loss: `2.230076281229655`

- Adapter dir: `/home/dong/workspace/ember/reports/stage52_sft_min_4b_20260225_external_zip22_qlora_v1/adapter`
- Train log: `/home/dong/workspace/ember/reports/stage52_sft_min_4b_20260225_external_zip22_qlora_v1/logs/train.log`

````

### Report: reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_all_dpo_variants_v2/stage52_baseline_compare.md

````md
# Stage 5.2 Baseline Compare

- Generated at: `2026-02-26T13:30:14`
- Primary metric: `mean_reward` (weighted/partial-credit). Secondary: `pass@1`.

| label | samples | N | reward_first | reward_best | weighted_first | weighted_best | pass@1 | pass@N | best_exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_n4 | 100 | 4 | 0.181452 | 0.199667 | 0.181452 | 0.199667 | 0.020000 | 0.020000 | 0.020000 |
| sft_n4 | 100 | 4 | 0.326786 | 0.468024 | 0.326786 | 0.468024 | 0.010000 | 0.030000 | 0.030000 |
| dpo_v1_n4 | 100 | 4 | 0.181452 | 0.199667 | 0.181452 | 0.199667 | 0.020000 | 0.020000 | 0.020000 |
| dpo_v2_refcpu_n4 | 100 | 4 | 0.181452 | 0.199667 | 0.181452 | 0.199667 | 0.020000 | 0.020000 | 0.020000 |
| dpo_v3_tuned_n4 | 100 | 4 | 0.204381 | 0.221524 | 0.204381 | 0.221524 | 0.010000 | 0.020000 | 0.020000 |
| dpo_v4_hardpair_n4_gpu1 | 100 | 4 | 0.261024 | 0.280667 | 0.261024 | 0.280667 | 0.020000 | 0.020000 | 0.020000 |

````

### Report: reports/stage52_dataset_validation_external_zip22_v1/stage52_dataset_validation_summary.md

````md
# Stage 5.2 Dataset Validation Summary

- Generated at: `2026-02-25T23:50:51`
- Schema: `/home/dong/workspace/ember/reports/stage52_external_from_zip22_20260225_235040/schema.json`
- Splits: `train, val, test`
- Suspicious rows saved: `120`

## Per-split

| split | rows | missing_keys | non_obj_gold | missing_required | type_mismatch | dup_id | dup_prompt | dup_row | p95_tokens_est | rule_ratio | tie_break_ratio | multi_record_ratio | leak_kw_ratio | exact_gold_json_ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 400 | 0 | 0 | 80 | 0 | 0 | 0 | 0 | 105.0 | 0.110000 | 0.090000 | 0.267500 | 0.000000 | 0.000000 |
| val | 100 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 102.1 | 0.150000 | 0.170000 | 0.230000 | 0.000000 | 0.000000 |
| test | 100 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 104.1 | 0.110000 | 0.070000 | 0.230000 | 0.000000 | 0.000000 |

## Cross-split overlap

| split_a | split_b | id_overlap | prompt_overlap | row_overlap | gold_overlap |
| --- | --- | --- | --- | --- | --- |
| train | val | 0 | 0 | 0 | 0 |
| train | test | 0 | 0 | 0 | 0 |
| val | test | 0 | 0 | 0 | 0 |

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`
