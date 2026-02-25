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
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    ap.add_argument("--save-tokenizer", action="store_true", default=True)
    ap.add_argument("--no-save-tokenizer", dest="save_tokenizer", action="store_false")
    return ap.parse_args()


def import_stack():
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    except Exception as ex:
        die(
            "missing Python deps. Install first, e.g.\n"
            "  pip install torch transformers peft accelerate sentencepiece\n"
            f"details: {ex}"
        )
    return torch, LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed


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
        AutoModelForCausalLM,
        AutoTokenizer,
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
        "training_loss": float(result.training_loss) if result is not None else None,
    }
    (run_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[done] adapter saved to: {run_dir}")
    print(f"[done] examples={len(rows)} max_steps={args.max_steps} loss={meta['training_loss']}")


if __name__ == "__main__":
    main()
