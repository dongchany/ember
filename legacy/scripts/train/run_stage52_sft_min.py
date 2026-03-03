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
