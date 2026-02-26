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
