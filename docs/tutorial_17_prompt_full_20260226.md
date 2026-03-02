# Tutorial #17 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 17 篇。

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
请写第 17 篇：DPO — 不需要单独训练 reward model 的偏好学习。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 从 Bradley-Terry 出发推导 DPO loss（尽量详细）
- 讲清 pair 质量、margin 阈值、reference_mode 对结果的影响
```

---

## 2) 代码上下文（完整/相关段落）

### File: scripts/train/run_stage52_build_dpo_pairs.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

from common_train import die, load_jsonl


def load_csv(path: Path) -> List[Dict[str, str]]:
    import csv

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def main() -> None:
    ap = argparse.ArgumentParser(description="Build DPO chosen/rejected pairs from Best-of-N candidates.")
    ap.add_argument("--dataset-jsonl", type=str, required=True, help='rows: {"id","prompt",...}')
    ap.add_argument("--candidates-csv", type=str, required=True, help="stage52_candidates.csv")
    ap.add_argument("--min-margin", type=float, default=0.05, help="min reward margin for valid pair")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    ds_path = Path(args.dataset_jsonl).expanduser().resolve()
    cand_path = Path(args.candidates_csv).expanduser().resolve()
    for p in [ds_path, cand_path]:
        if not p.exists():
            die(f"missing file: {p}")

    ds = load_jsonl(ds_path)
    cands = load_csv(cand_path)
    prompt_by_id: Dict[str, str] = {}
    for r in ds:
        rid = str(r.get("id", ""))
        if not rid:
            continue
        prompt_by_id[rid] = str(r.get("prompt", ""))

    by_id: Dict[str, List[Dict[str, str]]] = {}
    for r in cands:
        rid = str(r.get("id", ""))
        if not rid:
            continue
        by_id.setdefault(rid, []).append(r)

    pair_rows: List[Dict[str, Any]] = []
    skipped_low_margin = 0
    skipped_missing_prompt = 0
    margins: List[float] = []

    for rid, rows in by_id.items():
        if rid not in prompt_by_id:
            skipped_missing_prompt += 1
            continue
        if len(rows) < 2:
            continue
        rows_sorted = sorted(
            rows,
            key=lambda r: (
                parse_float(r.get("reward", "0")),
                parse_float(r.get("field_acc", "0")),
                int(r.get("exact_all", "0") or "0"),
            ),
            reverse=True,
        )
        best = rows_sorted[0]
        worst = rows_sorted[-1]
        rb = parse_float(best.get("reward", "0"))
        rw = parse_float(worst.get("reward", "0"))
        margin = rb - rw
        if margin < args.min_margin:
            skipped_low_margin += 1
            continue
        margins.append(margin)
        pair_rows.append(
            {
                "id": rid,
                "prompt": prompt_by_id[rid],
                "chosen": str(best.get("output", "")).replace("\\n", "\n"),
                "rejected": str(worst.get("output", "")).replace("\\n", "\n"),
                "reward_chosen": rb,
                "reward_rejected": rw,
                "reward_margin": margin,
                "chosen_exact_all": int(best.get("exact_all", "0") or "0"),
                "rejected_exact_all": int(worst.get("exact_all", "0") or "0"),
            }
        )

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage52_dpo_pairs_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "stage52_dpo_pairs.jsonl"
    out_md = out_dir / "stage52_dpo_pairs_summary.md"
    write_jsonl(out_jsonl, pair_rows)

    num_pairs = len(pair_rows)
    avg_margin = (sum(margins) / num_pairs) if num_pairs > 0 else 0.0
    min_margin = min(margins) if margins else 0.0
    max_margin = max(margins) if margins else 0.0

    lines = [
        "# Stage 5.2 DPO Pair Builder",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Candidate rows: `{len(cands)}`",
        f"- Unique sample ids: `{len(by_id)}`",
        f"- Valid pairs: `{num_pairs}`",
        f"- Skipped (low margin): `{skipped_low_margin}`",
        f"- Skipped (missing prompt): `{skipped_missing_prompt}`",
        f"- Margin threshold: `{args.min_margin:.6f}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| avg_reward_margin | {avg_margin:.6f} |",
        f"| min_reward_margin | {min_margin:.6f} |",
        f"| max_reward_margin | {max_margin:.6f} |",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.2 dpo pair builder")
    print(f"- pairs jsonl: {out_jsonl}")
    print(f"- summary md: {out_md}")


if __name__ == "__main__":
    main()

````

### File: scripts/train/run_stage52_build_dpo_pairs_oracle_exact.py

````py
#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common_train import die, load_jsonl


def parse_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def parse_int(v: str, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def load_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def choose_rejected(rows: List[Dict[str, str]]) -> Tuple[Dict[str, str], bool]:
    # Prefer the strongest non-exact candidate as hard negative.
    non_exact = [r for r in rows if parse_int(r.get("exact_all", "0")) == 0]
    if non_exact:
        ranked = sorted(
            non_exact,
            key=lambda r: (
                parse_float(r.get("reward", "0")),
                parse_float(r.get("field_acc", "0")),
                parse_int(r.get("parse_ok", "0")),
                parse_int(r.get("schema_ok", "0")),
            ),
            reverse=True,
        )
        return ranked[0], False
    # Fallback: if every candidate is exact, use the weakest exact as rejected.
    ranked = sorted(
        rows,
        key=lambda r: (
            parse_float(r.get("reward", "0")),
            parse_float(r.get("field_acc", "0")),
        ),
    )
    return ranked[0], True


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build DPO pairs with oracle chosen (gold JSON) and hard rejected from candidate outputs."
    )
    ap.add_argument("--dataset-jsonl", type=str, required=True, help='rows: {"id","prompt","gold"}')
    ap.add_argument("--candidates-csv", type=str, required=True, help="stage52_candidates.csv")
    ap.add_argument("--min-margin", type=float, default=0.05, help="minimum (1.0 - rejected_reward) margin")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    ds_path = Path(args.dataset_jsonl).expanduser().resolve()
    cand_path = Path(args.candidates_csv).expanduser().resolve()
    for p in [ds_path, cand_path]:
        if not p.exists():
            die(f"missing file: {p}")

    ds = load_jsonl(ds_path)
    cands = load_csv(cand_path)

    prompt_by_id: Dict[str, str] = {}
    gold_by_id: Dict[str, Dict[str, Any]] = {}
    for r in ds:
        rid = str(r.get("id", ""))
        prompt = str(r.get("prompt", ""))
        gold = r.get("gold")
        if not rid or not prompt or not isinstance(gold, dict):
            continue
        prompt_by_id[rid] = prompt
        gold_by_id[rid] = gold

    by_id: Dict[str, List[Dict[str, str]]] = {}
    for r in cands:
        rid = str(r.get("id", ""))
        if not rid:
            continue
        by_id.setdefault(rid, []).append(r)

    pair_rows: List[Dict[str, Any]] = []
    skipped_missing_meta = 0
    skipped_no_candidates = 0
    skipped_low_margin = 0
    used_exact_fallback = 0
    margins: List[float] = []

    for rid, gold in gold_by_id.items():
        prompt = prompt_by_id.get(rid, "")
        rows = by_id.get(rid, [])
        if not prompt:
            skipped_missing_meta += 1
            continue
        if not rows:
            skipped_no_candidates += 1
            continue

        rejected, is_fallback = choose_rejected(rows)
        rejected_reward = parse_float(rejected.get("reward", "0.0"))
        chosen_reward = 1.0
        margin = chosen_reward - rejected_reward
        if margin < args.min_margin:
            skipped_low_margin += 1
            continue
        if is_fallback:
            used_exact_fallback += 1

        pair_rows.append(
            {
                "id": rid,
                "prompt": prompt,
                "chosen": json.dumps(gold, ensure_ascii=False),
                "rejected": str(rejected.get("output", "")).replace("\\n", "\n"),
                "reward_chosen": chosen_reward,
                "reward_rejected": rejected_reward,
                "reward_margin": margin,
                "chosen_exact_all": 1,
                "rejected_exact_all": parse_int(rejected.get("exact_all", "0")),
                "source": "oracle_exact_hard_negative",
            }
        )
        margins.append(margin)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        Path.cwd() / "reports" / f"stage52_dpo_pairs_oracle_exact_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "stage52_dpo_pairs.jsonl"
    out_md = out_dir / "stage52_dpo_pairs_summary.md"

    write_jsonl(out_jsonl, pair_rows)

    num_pairs = len(pair_rows)
    avg_margin = (sum(margins) / num_pairs) if num_pairs > 0 else 0.0
    min_margin = min(margins) if margins else 0.0
    max_margin = max(margins) if margins else 0.0

    lines = [
        "# Stage 5.2 DPO Pair Builder (Oracle Exact)",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Candidate rows: `{len(cands)}`",
        f"- Unique sample ids in dataset: `{len(gold_by_id)}`",
        f"- Unique sample ids in candidates: `{len(by_id)}`",
        f"- Valid pairs: `{num_pairs}`",
        f"- Skipped (missing prompt/meta): `{skipped_missing_meta}`",
        f"- Skipped (no candidates): `{skipped_no_candidates}`",
        f"- Skipped (low margin): `{skipped_low_margin}`",
        f"- Rejected exact fallback used: `{used_exact_fallback}`",
        f"- Margin threshold: `{args.min_margin:.6f}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| avg_reward_margin | {avg_margin:.6f} |",
        f"| min_reward_margin | {min_margin:.6f} |",
        f"| max_reward_margin | {max_margin:.6f} |",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.2 oracle-exact dpo pair builder")
    print(f"- pairs jsonl: {out_jsonl}")
    print(f"- summary md: {out_md}")


if __name__ == "__main__":
    main()

````

### File: scripts/train/run_stage52_dpo_min.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from common_train import die, dtype_from_name, load_jsonl, render_chat, write_csv


def tokenize_pair(
    tokenizer,
    prompt: str,
    response: str,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    prompt_text = render_chat(tokenizer, prompt=prompt, response=None, add_generation_prompt=True)
    full_text = render_chat(tokenizer, prompt=prompt, response=response, add_generation_prompt=False)
    p = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    f = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(f) > max_length:
        drop = len(f) - max_length
        f = f[drop:]
        # Prompt tokens that remain after left truncation.
        prompt_len = max(0, len(p) - drop)
    else:
        prompt_len = len(p)
    prompt_len = min(prompt_len, len(f))
    # Response mask over token positions in f (before shift): 1 means token belongs to assistant response segment.
    mask = [0] * prompt_len + [1] * (len(f) - prompt_len)
    return f, mask


def collate_side(
    seqs: List[List[int]],
    masks: List[List[int]],
    pad_id: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = len(seqs)
    max_len = max(len(x) for x in seqs)
    ids = torch.full((bsz, max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bsz, max_len), dtype=torch.long, device=device)
    msk = torch.zeros((bsz, max_len), dtype=torch.float32, device=device)
    for i, (s, m) in enumerate(zip(seqs, masks)):
        n = len(s)
        ids[i, :n] = torch.tensor(s, dtype=torch.long, device=device)
        attn[i, :n] = 1
        msk[i, :n] = torch.tensor(m, dtype=torch.float32, device=device)
    return ids, attn, msk


def seq_logp(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    rmask = response_mask[:, 1:]
    logp = F.log_softmax(logits, dim=-1).gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    # Ensure each sample has at least one response token in loss.
    denom = rmask.sum(dim=-1).clamp_min(1.0)
    return (logp * rmask).sum(dim=-1) / denom


def maybe_load_ref_model(
    mode: str,
    model_name: str,
    dtype: torch.dtype,
    policy_device: str,
):
    if mode == "none":
        return None, ""
    if mode == "cpu":
        ref_device = "cpu"
    elif mode == "same_device":
        ref_device = policy_device
    else:
        die(f"unsupported --reference-mode: {mode}")
    ref = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype if ref_device != "cpu" else torch.float32,
    )
    ref.eval().to(ref_device)
    for p in ref.parameters():
        p.requires_grad = False
    return ref, ref_device


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5.2 minimal DPO loop (LoRA policy).")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--pairs-jsonl", type=str, required=True, help='rows: {"id","prompt","chosen","rejected"}')
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--init-adapter", type=str, default="", help="optional PEFT adapter dir to continue DPO training from")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--reference-mode", type=str, default="none", choices=["none", "cpu", "same_device"])
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-acc-steps", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--target-modules", type=str, default="q_proj,k_proj,v_proj,o_proj")
    args = ap.parse_args()

    if args.max_length <= 0 or args.batch_size <= 0 or args.grad_acc_steps <= 0 or args.max_steps <= 0:
        die("invalid training hyper-parameters")
    if args.beta <= 0:
        die("--beta must be > 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_dir = Path(args.model).expanduser().resolve()
    init_adapter_dir = Path(args.init_adapter).expanduser().resolve() if args.init_adapter.strip() else None
    pairs_path = Path(args.pairs_jsonl).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    required_paths = [model_dir, pairs_path]
    if init_adapter_dir is not None:
        required_paths.append(init_adapter_dir)
    for p in required_paths:
        if not p.exists():
            die(f"missing path: {p}")
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_jsonl(pairs_path)
    if not pairs:
        die("empty pairs jsonl")
    for i, r in enumerate(pairs):
        for k in ["id", "prompt", "chosen", "rejected"]:
            if k not in r:
                die(f"pairs row {i} missing key: {k}")

    tdtype = dtype_from_name(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        die("tokenizer pad_token_id is None")

    base = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=tdtype,
    )
    base.config.use_cache = False
    base.to(args.device)
    base.train()

    if init_adapter_dir is not None:
        policy = PeftModel.from_pretrained(base, str(init_adapter_dir), is_trainable=True)
    else:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
            bias="none",
        )
        policy = get_peft_model(base, lora_cfg)
    policy.train()

    ref_model, ref_device = maybe_load_ref_model(
        mode=args.reference_mode,
        model_name=str(model_dir),
        dtype=tdtype,
        policy_device=args.device,
    )

    tokenized: List[Dict[str, Any]] = []
    for r in pairs:
        cid, cm = tokenize_pair(tokenizer, str(r["prompt"]), str(r["chosen"]), args.max_length)
        rid, rm = tokenize_pair(tokenizer, str(r["prompt"]), str(r["rejected"]), args.max_length)
        if sum(cm) <= 0 or sum(rm) <= 0:
            continue
        tokenized.append(
            {
                "id": str(r["id"]),
                "chosen_ids": cid,
                "chosen_mask": cm,
                "rejected_ids": rid,
                "rejected_mask": rm,
            }
        )
    if not tokenized:
        die("all pairs filtered out after tokenization/masking")

    opt = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr)

    log_rows: List[Dict[str, str]] = []
    ptr = 0
    policy.zero_grad(set_to_none=True)
    for step in range(1, args.max_steps + 1):
        batch = []
        for _ in range(args.batch_size):
            batch.append(tokenized[ptr % len(tokenized)])
            ptr += 1

        ch_ids, ch_attn, ch_mask = collate_side(
            [x["chosen_ids"] for x in batch],
            [x["chosen_mask"] for x in batch],
            pad_id=pad_id,
            device=args.device,
        )
        rj_ids, rj_attn, rj_mask = collate_side(
            [x["rejected_ids"] for x in batch],
            [x["rejected_mask"] for x in batch],
            pad_id=pad_id,
            device=args.device,
        )

        pol_ch = seq_logp(policy, ch_ids, ch_attn, ch_mask)
        pol_rj = seq_logp(policy, rj_ids, rj_attn, rj_mask)
        pi_logratio = pol_ch - pol_rj

        if ref_model is not None:
            with torch.no_grad():
                if ref_device == args.device:
                    ref_ch = seq_logp(ref_model, ch_ids, ch_attn, ch_mask)
                    ref_rj = seq_logp(ref_model, rj_ids, rj_attn, rj_mask)
                else:
                    ref_ch = seq_logp(
                        ref_model,
                        ch_ids.to(ref_device),
                        ch_attn.to(ref_device),
                        ch_mask.to(ref_device),
                    ).to(args.device)
                    ref_rj = seq_logp(
                        ref_model,
                        rj_ids.to(ref_device),
                        rj_attn.to(ref_device),
                        rj_mask.to(ref_device),
                    ).to(args.device)
            ref_logratio = ref_ch - ref_rj
        else:
            ref_logratio = torch.zeros_like(pi_logratio)

        logits = args.beta * (pi_logratio - ref_logratio)
        loss = -F.logsigmoid(logits).mean()
        (loss / args.grad_acc_steps).backward()

        if step % args.grad_acc_steps == 0:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        win_rate = float((pi_logratio > 0).float().mean().item())
        mean_pi_margin = float(pi_logratio.mean().item())
        mean_ref_margin = float(ref_logratio.mean().item())
        log_rows.append(
            {
                "step": str(step),
                "loss": f"{float(loss.item()):.8f}",
                "win_rate": f"{win_rate:.6f}",
                "mean_pi_margin": f"{mean_pi_margin:.8f}",
                "mean_ref_margin": f"{mean_ref_margin:.8f}",
            }
        )
        if args.log_every > 0 and step % args.log_every == 0:
            print(
                f"[step {step}] loss={float(loss.item()):.6f} "
                f"win_rate={win_rate:.4f} pi_margin={mean_pi_margin:.6f}"
            )

    adapter_dir = out_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    log_csv = out_dir / "stage52_dpo_train_log.csv"
    write_csv(log_csv, log_rows)

    last = log_rows[-1]
    first = log_rows[0]
    summary = {
        "num_pairs": len(tokenized),
        "max_steps": args.max_steps,
        "reference_mode": args.reference_mode,
        "init_adapter": str(init_adapter_dir) if init_adapter_dir is not None else "",
        "loss_start": float(first["loss"]),
        "loss_end": float(last["loss"]),
        "win_rate_start": float(first["win_rate"]),
        "win_rate_end": float(last["win_rate"]),
        "mean_pi_margin_end": float(last["mean_pi_margin"]),
    }
    summary_json = out_dir / "stage52_dpo_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary_md = out_dir / "stage52_dpo_summary.md"
    lines = [
        "# Stage 5.2 Minimal DPO Loop",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- num_pairs: `{summary['num_pairs']}`",
        f"- steps: `{summary['max_steps']}`",
        f"- reference_mode: `{summary['reference_mode']}`",
        f"- init_adapter: `{summary['init_adapter']}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| loss_start | {summary['loss_start']:.8f} |",
        f"| loss_end | {summary['loss_end']:.8f} |",
        f"| win_rate_start | {summary['win_rate_start']:.6f} |",
        f"| win_rate_end | {summary['win_rate_end']:.6f} |",
        f"| mean_pi_margin_end | {summary['mean_pi_margin_end']:.8f} |",
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.2 minimal dpo loop")
    print(f"- adapter dir: {adapter_dir}")
    print(f"- train log: {log_csv}")
    print(f"- summary md: {summary_md}")
    print(f"- summary json: {summary_json}")


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

---

## 3) 报告上下文（完整）

### Report: reports/stage52_dpo_pairs_4b_20260226_external_zip22_train200_sftn4_margin008_v1/stage52_dpo_pairs_summary.md

````md
# Stage 5.2 DPO Pair Builder

- Generated at: `2026-02-26T13:05:32`
- Candidate rows: `800`
- Unique sample ids: `200`
- Valid pairs: `141`
- Skipped (low margin): `59`
- Skipped (missing prompt): `0`
- Margin threshold: `0.080000`

| metric | value |
| --- | --- |
| avg_reward_margin | 0.403732 |
| min_reward_margin | 0.142857 |
| max_reward_margin | 1.000000 |

````

### Report: reports/stage52_dpo_min_4b_20260226_external_zip22_v4_hardpair_train200_len96_gpu1/stage52_dpo_summary.md

````md
# Stage 5.2 Minimal DPO Loop

- Generated at: `2026-02-26T13:06:40`
- num_pairs: `141`
- steps: `120`
- reference_mode: `none`

| metric | value |
| --- | --- |
| loss_start | 0.67896372 |
| loss_end | 0.66031241 |
| win_rate_start | 1.000000 |
| win_rate_end | 1.000000 |
| mean_pi_margin_end | 0.66784382 |

````

### Report: reports/stage52_baseline_compare_4b_20260226_external_zip22_n1_dpo_v3_v4_v2/stage52_baseline_compare.md

````md
# Stage 5.2 Baseline Compare

- Generated at: `2026-02-26T13:30:14`
- Primary metric: `mean_reward` (weighted/partial-credit). Secondary: `pass@1`.

| label | samples | N | reward_first | reward_best | weighted_first | weighted_best | pass@1 | pass@N | best_exact |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_n1 | 100 | 1 | 0.181452 | 0.181452 | 0.181452 | 0.181452 | 0.020000 | 0.020000 | 0.020000 |
| dpo_v3_tuned_n1 | 100 | 1 | 0.204381 | 0.204381 | 0.204381 | 0.204381 | 0.010000 | 0.010000 | 0.010000 |
| dpo_v4_hardpair_n1_gpu1 | 100 | 1 | 0.261024 | 0.261024 | 0.261024 | 0.261024 | 0.020000 | 0.020000 | 0.020000 |

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

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`
