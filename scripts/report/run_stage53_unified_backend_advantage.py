#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
from pathlib import Path
from typing import Dict, List

from common_report import die, read_csv


def size_of_safetensors_bytes(dir_path: Path) -> int:
    total = 0
    for p in dir_path.glob("*.safetensors"):
        if p.is_file():
            total += p.stat().st_size
    return total


def to_gib(x: int) -> float:
    return float(x) / (1024.0 ** 3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5.3 unified-backend advantage summary.")
    ap.add_argument("--model-dir", type=str, required=True)
    ap.add_argument("--hot-update-csv", type=str, required=True, help="stage31_lora_hot_update.csv")
    ap.add_argument("--adapter-dir", type=str, default="")
    ap.add_argument("--num-rounds", type=int, default=30)
    ap.add_argument("--pcie-bandwidth-gbps", type=float, default=24.0, help="effective host<->device GB/s")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.num_rounds <= 0:
        die("--num-rounds must be > 0")
    if args.pcie_bandwidth_gbps <= 0:
        die("--pcie-bandwidth-gbps must be > 0")

    model_dir = Path(args.model_dir).expanduser().resolve()
    hot_csv = Path(args.hot_update_csv).expanduser().resolve()
    if not model_dir.exists():
        die(f"model-dir not found: {model_dir}")
    if not hot_csv.exists():
        die(f"hot-update-csv not found: {hot_csv}")

    rows = read_csv(hot_csv)
    if not rows:
        die(f"empty hot-update csv: {hot_csv}")
    r0 = rows[0]
    measured_hot_update_ms = float(r0.get("apply_ms_ext", "0") or "0")

    adapter_dir = Path(args.adapter_dir).expanduser().resolve() if args.adapter_dir.strip() else None
    if adapter_dir is None:
        raw = r0.get("adapter_dir", "")
        if raw:
            adapter_dir = Path(raw).expanduser().resolve()
    adapter_bytes = 0
    if adapter_dir is not None and adapter_dir.exists():
        adapter_bytes = size_of_safetensors_bytes(adapter_dir)

    model_bytes = size_of_safetensors_bytes(model_dir)
    if model_bytes <= 0:
        die(f"no *.safetensors under {model_dir}")

    dual_model_gib = to_gib(model_bytes * 2)
    unified_model_gib = to_gib(model_bytes)
    model_mem_saved_gib = dual_model_gib - unified_model_gib
    model_mem_saved_pct = (model_mem_saved_gib / dual_model_gib * 100.0) if dual_model_gib > 0 else 0.0

    bw_bytes_per_s = args.pcie_bandwidth_gbps * (1024.0 ** 3)
    full_sync_ms = float(model_bytes) / bw_bytes_per_s * 1000.0
    lora_sync_ms = float(adapter_bytes) / bw_bytes_per_s * 1000.0 if adapter_bytes > 0 else 0.0

    total_full_sync_ms = full_sync_ms * args.num_rounds
    total_lora_sync_ms = lora_sync_ms * args.num_rounds
    total_hot_update_ms = measured_hot_update_ms * args.num_rounds

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage53_unified_backend_advantage_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_md = out_dir / "stage53_summary.md"
    out_csv = out_dir / "stage53_summary.csv"

    summary_rows = [
        {
            "metric": "model_weights_gib",
            "value": f"{to_gib(model_bytes):.6f}",
            "notes": "sum of model *.safetensors",
        },
        {
            "metric": "adapter_weights_gib",
            "value": f"{to_gib(adapter_bytes):.6f}",
            "notes": "sum of adapter *.safetensors",
        },
        {
            "metric": "dual_stack_model_only_gib",
            "value": f"{dual_model_gib:.6f}",
            "notes": "2x model copy (train + infer process)",
        },
        {
            "metric": "unified_model_only_gib",
            "value": f"{unified_model_gib:.6f}",
            "notes": "single model copy",
        },
        {
            "metric": "model_memory_saved_gib",
            "value": f"{model_mem_saved_gib:.6f}",
            "notes": f"{model_mem_saved_pct:.3f}% vs dual-stack model-only footprint",
        },
        {
            "metric": "sync_full_model_ms_per_round_est",
            "value": f"{full_sync_ms:.6f}",
            "notes": f"assume {args.pcie_bandwidth_gbps:.2f} GiB/s",
        },
        {
            "metric": "sync_lora_ms_per_round_est",
            "value": f"{lora_sync_ms:.6f}",
            "notes": f"assume {args.pcie_bandwidth_gbps:.2f} GiB/s",
        },
        {
            "metric": "hot_update_ms_per_round_measured",
            "value": f"{measured_hot_update_ms:.6f}",
            "notes": f"from {hot_csv.name}",
        },
        {
            "metric": f"sync_full_model_ms_{args.num_rounds}round_est",
            "value": f"{total_full_sync_ms:.6f}",
            "notes": "transfer-only estimate",
        },
        {
            "metric": f"sync_lora_ms_{args.num_rounds}round_est",
            "value": f"{total_lora_sync_ms:.6f}",
            "notes": "transfer-only estimate",
        },
        {
            "metric": f"hot_update_ms_{args.num_rounds}round_measured",
            "value": f"{total_hot_update_ms:.6f}",
            "notes": "in-process adapter merge",
        },
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "value", "notes"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    lines = [
        "# Stage 5.3 Unified Backend Advantage (Memory + Sync)",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Model dir: `{model_dir}`",
        f"- Adapter dir: `{adapter_dir if adapter_dir else ''}`",
        f"- Hot-update source: `{hot_csv}`",
        "",
        "| metric | value | notes |",
        "| --- | --- | --- |",
    ]
    for r in summary_rows:
        lines.append(f"| {r['metric']} | {r['value']} | {r['notes']} |")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.3 unified backend advantage")
    print(f"- summary md: {out_md}")
    print(f"- summary csv: {out_csv}")


if __name__ == "__main__":
    main()
