#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


def split_ints(text: str) -> List[int]:
    out: List[int] = []
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return out


def hf_hub_root() -> Path:
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "hub")
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def resolve_snapshot_dir(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if (path / "config.json").exists() and list(path.glob("*.safetensors")):
        return path
    snap_root = path / "snapshots"
    if not snap_root.exists():
        return None
    candidates = [
        p for p in snap_root.iterdir()
        if p.is_dir() and (p / "config.json").exists() and list(p.glob("*.safetensors"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_model_dir(model_arg: str) -> Path:
    raw = model_arg.strip()
    if not raw:
        die("--model is empty")

    p = Path(raw).expanduser().resolve()
    resolved = resolve_snapshot_dir(p)
    if resolved is not None:
        return resolved

    hub_root = hf_hub_root()
    model_cache_dir = hub_root / ("models--" + raw.replace("/", "--"))
    resolved = resolve_snapshot_dir(model_cache_dir)
    if resolved is not None:
        return resolved

    die(
        "failed to resolve model from local cache: "
        f"{raw}. Checked path='{p}' and HF cache='{model_cache_dir}'. "
        "Set HF_HOME/HUGGINGFACE_HUB_CACHE correctly, or pass a local snapshot path."
    )
    raise AssertionError("unreachable")


def read_num_layers(model_dir: Path) -> int:
    cfg_path = model_dir / "config.json"
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as ex:
        die(f"failed to read {cfg_path}: {ex}")
    for key in ["num_hidden_layers", "n_layer", "num_layers"]:
        v = data.get(key)
        if isinstance(v, int) and v > 0:
            return v
    die("failed to infer num layers from config.json")
    raise AssertionError("unreachable")


def parse_splits(text: str, num_layers: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    seen = set()
    if text.strip() == "":
        return out
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        vals = split_ints(item)
        if len(vals) != 2:
            die(f"invalid split item '{item}', expected A,B")
        a, b = vals
        if a <= 0 or b <= 0 or a + b != num_layers:
            die(f"invalid split '{item}', expected A+B={num_layers}")
        key = (a, b)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def auto_splits(num_layers: int) -> List[Tuple[int, int]]:
    # Cover left-heavy, balanced, right-heavy partitions.
    ratios = [0.25, 0.33, 0.50, 0.67, 0.75]
    vals = set()
    for r in ratios:
        a = int(round(num_layers * r))
        if a <= 0:
            a = 1
        if a >= num_layers:
            a = num_layers - 1
        vals.add(a)
    vals.add(num_layers // 2)
    out = [(a, num_layers - a) for a in sorted(vals) if 0 < a < num_layers]
    if not out:
        die("failed to construct split sweep")
    return out


def read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Optional[List[str]] = None) -> None:
    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def looks_like_oom(text: str) -> bool:
    t = text.lower()
    return (
        "out of memory" in t
        or "cuda_error_out_of_memory" in t
        or "cuda error: out of memory" in t
        or "cudamalloc" in t
    )


def looks_like_runtime_unavailable(text: str) -> bool:
    t = text.lower()
    keys = [
        "cuda runtime not available",
        "no cuda device available",
        "cuda_error_no_device",
        "cuda_error_system_not_ready",
        "cudasetdevice",
        "operation not supported on this os",
        "os call failed",
    ]
    return any(k in t for k in keys)


def detect_error_hint(text: str) -> str:
    if looks_like_oom(text):
        return "oom"
    if looks_like_runtime_unavailable(text):
        return "runtime_unavailable"
    return "runtime_error"


def run_cmd(cmd: List[str], cwd: Path, log_path: Path, env: Optional[Dict[str, str]] = None) -> Tuple[int, str, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=env)
        f.write(p.stdout)
        if p.stderr:
            f.write("\n[stderr]\n")
            f.write(p.stderr)
    return p.returncode, p.stdout, p.stderr


def summarize(raw_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str], Dict[str, Dict[str, str]]] = {}
    for r in raw_rows:
        split = r.get("split", "")
        mode = r.get("mode", "")
        key = (split, mode)
        grouped.setdefault(key, {})[r.get("phase", "")] = r

    out: List[Dict[str, str]] = []
    for key, phases in sorted(grouped.items()):
        pre = phases.get("prefill")
        dec = phases.get("decode_per_token")
        if pre is None:
            continue

        split = key[0]
        mode = key[1]
        prompt_len = pre.get("prompt_len", "0")
        decode_steps = pre.get("decode_steps", "0")

        prefill_wall = safe_float(pre.get("wall_ms", "0"))
        dec_tok = safe_float(dec.get("wall_ms", "0")) if dec else 0.0
        total_ms = prefill_wall + dec_tok * safe_float(decode_steps)
        prefill_share = (prefill_wall / total_ms * 100.0) if total_ms > 0.0 else 0.0
        tok_s = (safe_float(decode_steps) * 1000.0 / total_ms) if total_ms > 0.0 else 0.0

        pre_compute = (
            safe_float(pre.get("embedding_ms", "0"))
            + safe_float(pre.get("rmsnorm_ms", "0"))
            + safe_float(pre.get("attention_ms", "0"))
            + safe_float(pre.get("ffn_ms", "0"))
            + safe_float(pre.get("lm_head_ms", "0"))
        )
        pre_transfer = (
            safe_float(pre.get("p2p_ms", "0"))
            + safe_float(pre.get("memcpy_h2d_ms", "0"))
            + safe_float(pre.get("memcpy_d2h_ms", "0"))
        )
        dec_compute = (
            safe_float((dec or {}).get("embedding_ms", "0"))
            + safe_float((dec or {}).get("rmsnorm_ms", "0"))
            + safe_float((dec or {}).get("attention_ms", "0"))
            + safe_float((dec or {}).get("ffn_ms", "0"))
            + safe_float((dec or {}).get("lm_head_ms", "0"))
        )
        dec_transfer = (
            safe_float((dec or {}).get("p2p_ms", "0"))
            + safe_float((dec or {}).get("memcpy_h2d_ms", "0"))
            + safe_float((dec or {}).get("memcpy_d2h_ms", "0"))
        )

        out.append(
            {
                "split": split,
                "mode": mode,
                "prompt_len": prompt_len,
                "decode_steps": decode_steps,
                "prefill_wall_ms": f"{prefill_wall:.3f}",
                "decode_per_token_ms": f"{dec_tok:.3f}",
                "rollout_total_ms_est": f"{total_ms:.3f}",
                "rollout_tok_s_est": f"{tok_s:.3f}",
                "prefill_share_pct": f"{prefill_share:.2f}",
                "prefill_compute_ms": f"{pre_compute:.3f}",
                "prefill_transfer_ms": f"{pre_transfer:.3f}",
                "decode_compute_per_tok_ms": f"{dec_compute:.3f}",
                "decode_transfer_per_tok_ms": f"{dec_transfer:.3f}",
            }
        )
    return out


def build_transfer_vs_compute(summary_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in summary_rows:
        pre_compute = safe_float(r.get("prefill_compute_ms", "0"))
        pre_transfer = safe_float(r.get("prefill_transfer_ms", "0"))
        dec_compute = safe_float(r.get("decode_compute_per_tok_ms", "0"))
        dec_transfer = safe_float(r.get("decode_transfer_per_tok_ms", "0"))
        pre_ratio = (pre_transfer / pre_compute) if pre_compute > 0.0 else 0.0
        dec_ratio = (dec_transfer / dec_compute) if dec_compute > 0.0 else 0.0
        out.append(
            {
                "split": r.get("split", ""),
                "mode": r.get("mode", ""),
                "prefill_transfer_compute_ratio": f"{pre_ratio:.4f}",
                "decode_transfer_compute_ratio": f"{dec_ratio:.4f}",
                "prefill_transfer_ms": r.get("prefill_transfer_ms", "0"),
                "prefill_compute_ms": r.get("prefill_compute_ms", "0"),
                "decode_transfer_per_tok_ms": r.get("decode_transfer_per_tok_ms", "0"),
                "decode_compute_per_tok_ms": r.get("decode_compute_per_tok_ms", "0"),
            }
        )
    out.sort(key=lambda x: (x["split"], x["mode"]))
    return out


def build_bubble_vs_split(summary_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_split: Dict[str, Dict[str, Dict[str, str]]] = {}
    for r in summary_rows:
        by_split.setdefault(r.get("split", ""), {})[r.get("mode", "")] = r

    out: List[Dict[str, str]] = []
    for split, modes in sorted(by_split.items()):
        no_ov = modes.get("no_overlap")
        ov = modes.get("overlap")
        if not no_ov or not ov:
            continue
        t0 = safe_float(no_ov.get("rollout_total_ms_est", "0"))
        t1 = safe_float(ov.get("rollout_total_ms_est", "0"))
        if t0 <= 0.0 or t1 <= 0.0:
            continue
        speedup = t0 / t1
        hidden = max(0.0, (t0 - t1) / t0 * 100.0)
        out.append(
            {
                "split": split,
                "no_overlap_total_ms": f"{t0:.3f}",
                "overlap_total_ms": f"{t1:.3f}",
                "overlap_speedup_x": f"{speedup:.4f}",
                "bubble_hidden_pct_est": f"{hidden:.2f}",
            }
        )
    return out


def write_markdown(path: Path, model_dir: Path, summary_rows: List[Dict[str, str]], bubble_rows: List[Dict[str, str]]) -> None:
    lines: List[str] = []
    lines.append("# Stage 1.2 Pipeline Split Profiling")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")

    if not summary_rows:
        lines.append("_No summary rows generated._")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.append("## Throughput vs Split")
    lines.append("| split | mode | total_ms | tok/s_est | prefill_share_% |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in summary_rows:
        lines.append(
            "| " + " | ".join([
                r["split"],
                r["mode"],
                r["rollout_total_ms_est"],
                r["rollout_tok_s_est"],
                r["prefill_share_pct"],
            ]) + " |"
        )

    best = max(summary_rows, key=lambda x: safe_float(x.get("rollout_tok_s_est", "0")))
    lines.append("")
    lines.append("## Key Point")
    lines.append(
        f"- Best rollout tok/s split: `{best['split']}` in `{best['mode']}` mode "
        f"(`{best['rollout_tok_s_est']} tok/s`, total `{best['rollout_total_ms_est']} ms`)."
    )

    lines.append("")
    lines.append("## Bubble vs Split")
    if not bubble_rows:
        lines.append("- missing overlap/no_overlap pairs, cannot estimate bubble")
    else:
        lines.append("| split | no_overlap_ms | overlap_ms | speedup_x | hidden_% |")
        lines.append("| --- | --- | --- | --- | --- |")
        for r in bubble_rows:
            lines.append(
                "| " + " | ".join([
                    r["split"],
                    r["no_overlap_total_ms"],
                    r["overlap_total_ms"],
                    r["overlap_speedup_x"],
                    r["bubble_hidden_pct_est"],
                ]) + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def extract_json_array_from_mixed_output(text: str) -> List[Dict[str, object]]:
    lb = text.find("[")
    rb = text.rfind("]")
    if lb < 0 or rb < 0 or rb <= lb:
        raise ValueError("failed to locate JSON payload in llama-bench output")
    payload = text[lb:rb + 1]
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("unexpected llama-bench JSON: non-list payload")
    return data


def run_llama_bench_once(
    repo: Path,
    llama_bin_dir: Path,
    gguf_path: Path,
    prompt_len: int,
    decode_steps: int,
    device: str,
    split_mode: str,
    tensor_split: str,
    log_path: Path,
) -> Dict[str, float]:
    llama_bench = llama_bin_dir / "llama-bench"
    if not llama_bench.exists():
        raise FileNotFoundError(f"missing llama-bench: {llama_bench}")
    if not gguf_path.exists():
        raise FileNotFoundError(f"missing gguf: {gguf_path}")

    cmd = [
        str(llama_bench),
        "-m", str(gguf_path),
        "-p", str(prompt_len),
        "-n", str(decode_steps),
        "-r", "1",
        "-ngl", "999",
        "-o", "json",
        "-dev", device,
        "-sm", split_mode,
    ]
    if tensor_split.strip():
        cmd += ["-ts", tensor_split]

    env = os.environ.copy()
    ld = env.get("LD_LIBRARY_PATH", "")
    prefix = str(llama_bin_dir)
    env["LD_LIBRARY_PATH"] = f"{prefix}:{ld}" if ld else prefix

    rc, out, err = run_cmd(cmd, cwd=repo, log_path=log_path, env=env)
    if rc != 0:
        raise RuntimeError(f"llama-bench failed rc={rc}; see {log_path}")

    arr = extract_json_array_from_mixed_output((out or "") + "\n" + (err or ""))
    pre = None
    dec = None
    for item in arr:
        n_prompt = int(item.get("n_prompt", 0))
        n_gen = int(item.get("n_gen", 0))
        if n_prompt > 0 and n_gen == 0:
            pre = item
        if n_prompt == 0 and n_gen > 0:
            dec = item
    if pre is None or dec is None:
        raise RuntimeError(f"llama-bench output missing prompt/gen entries; see {log_path}")

    prefill_ms = float(pre.get("avg_ns", 0.0)) / 1e6
    decode_total_ms = float(dec.get("avg_ns", 0.0)) / 1e6
    decode_tok_s = float(dec.get("avg_ts", 0.0))
    decode_per_tok_ms = (decode_total_ms / float(decode_steps)) if decode_steps > 0 else 0.0
    total_ms = prefill_ms + decode_total_ms
    rollout_tok_s = (float(decode_steps) * 1000.0 / total_ms) if total_ms > 0.0 else 0.0
    return {
        "prefill_ms": prefill_ms,
        "decode_total_ms": decode_total_ms,
        "decode_tok_s": decode_tok_s,
        "decode_per_token_ms": decode_per_tok_ms,
        "rollout_total_ms": total_ms,
        "rollout_tok_s": rollout_tok_s,
    }


def write_vs_llama_outputs(
    out_dir: Path,
    ember_best: Dict[str, str],
    llama_rows: List[Dict[str, str]],
    gguf_path: Path,
    split_mode: str,
    tensor_split: str,
) -> None:
    out_rows: List[Dict[str, str]] = []
    out_rows.append(
        {
            "engine": "ember",
            "config": f"split={ember_best.get('split','')} mode={ember_best.get('mode','')}",
            "prefill_ms": f"{safe_float(ember_best.get('prefill_wall_ms', '0')):.3f}",
            "decode_tok_s": f"{(1000.0 / safe_float(ember_best.get('decode_per_token_ms', '0'))) if safe_float(ember_best.get('decode_per_token_ms', '0')) > 0 else 0.0:.3f}",
            "decode_per_token_ms": f"{safe_float(ember_best.get('decode_per_token_ms', '0')):.3f}",
            "rollout_total_ms": f"{safe_float(ember_best.get('rollout_total_ms_est', '0')):.3f}",
            "rollout_tok_s": f"{safe_float(ember_best.get('rollout_tok_s_est', '0')):.3f}",
            "notes": "best split from stage12 sweep",
        }
    )
    out_rows.extend(llama_rows)

    out_csv = out_dir / "stage12_vs_llama.csv"
    write_csv(out_csv, out_rows)

    md_lines: List[str] = []
    md_lines.append("# Stage 1.2 Ember vs llama.cpp")
    md_lines.append("")
    md_lines.append(f"- GGUF: `{gguf_path}`")
    md_lines.append(f"- llama split_mode: `{split_mode}`")
    md_lines.append(f"- llama tensor_split: `{tensor_split if tensor_split.strip() else 'auto/none'}`")
    md_lines.append("")
    md_lines.append("| engine | config | prefill_ms | decode_tok_s | decode_per_token_ms | rollout_total_ms | rollout_tok_s | notes |")
    md_lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in out_rows:
        md_lines.append(
            "| "
            + " | ".join(
                [
                    r["engine"],
                    r["config"],
                    r["prefill_ms"],
                    r["decode_tok_s"],
                    r["decode_per_token_ms"],
                    r["rollout_total_ms"],
                    r["rollout_tok_s"],
                    r["notes"],
                ]
            )
            + " |"
        )
    md_lines.append("")
    md_lines.append("- Note: llama-bench and ember_stage_breakdown are different harnesses; compare as practical baseline, not strict apples-to-apples kernel microbenchmark.")
    (out_dir / "stage12_vs_llama.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def infer_label_from_summary(path: Path) -> str:
    name = path.parent.name.strip()
    if name:
        return name
    stem = path.stem.strip()
    return stem if stem else "baseline"


def pretty_model_name(model_dir: Path) -> str:
    if model_dir.parent.name == "snapshots":
        cache_dir = model_dir.parent.parent.name
        if cache_dir.startswith("models--"):
            return cache_dir[len("models--"):].replace("--", "/")
        if cache_dir:
            return cache_dir
    return model_dir.name


def slugify_label(text: str) -> str:
    raw = text.strip().lower()
    if not raw:
        return "baseline"
    slug = re.sub(r"[^a-z0-9_-]+", "_", raw)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug if slug else "baseline"


def pick_best_row(summary_rows: List[Dict[str, str]]) -> Optional[Dict[str, str]]:
    if not summary_rows:
        return None
    return max(summary_rows, key=lambda r: safe_float(r.get("rollout_tok_s_est", "0")))


def summary_key(row: Dict[str, str]) -> Tuple[str, str]:
    return row.get("split", ""), row.get("mode", "")


def summary_to_map(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str], Dict[str, str]]:
    out: Dict[Tuple[str, str], Dict[str, str]] = {}
    for r in rows:
        out[summary_key(r)] = r
    return out


def pct_delta(old: float, new: float) -> Optional[float]:
    if old == 0.0:
        return None
    return (new - old) / old * 100.0


def fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.2f}%"


def resolve_existing_csv(path_arg: str, default_path: Optional[Path] = None) -> Optional[Path]:
    if path_arg.strip():
        p = Path(path_arg).expanduser().resolve()
        if not p.exists():
            die(f"csv not found: {p}")
        return p
    if default_path is not None and default_path.exists():
        return default_path
    return None


def read_rollout_vs_llama(vs_path: Optional[Path]) -> Optional[Tuple[float, float, float]]:
    if vs_path is None:
        return None
    rows = read_csv(vs_path)
    ember_rollout = None
    llama_dual_rollout = None
    for r in rows:
        engine = r.get("engine", "").strip().lower()
        config = r.get("config", "").strip().lower()
        rollout = safe_float(r.get("rollout_tok_s", "0"))
        if rollout <= 0.0:
            continue
        if engine == "ember":
            ember_rollout = rollout
        if engine == "llama.cpp" and "dual" in config:
            llama_dual_rollout = rollout
    if ember_rollout is None or llama_dual_rollout is None or llama_dual_rollout <= 0.0:
        return None
    ratio = ember_rollout / llama_dual_rollout * 100.0
    return ember_rollout, llama_dual_rollout, ratio


def write_delta_outputs(
    out_dir: Path,
    current_summary: List[Dict[str, str]],
    baseline_summary: List[Dict[str, str]],
    baseline_label: str,
    current_vs_llama_csv: Optional[Path],
    baseline_vs_llama_csv: Optional[Path],
) -> Tuple[Path, Path]:
    cur_map = summary_to_map(current_summary)
    base_map = summary_to_map(baseline_summary)
    keys = sorted(set(cur_map.keys()) & set(base_map.keys()))

    delta_rows: List[Dict[str, str]] = []
    for key in keys:
        old = base_map[key]
        new = cur_map[key]
        old_rollout = safe_float(old.get("rollout_tok_s_est", "0"))
        new_rollout = safe_float(new.get("rollout_tok_s_est", "0"))
        old_decode = safe_float(old.get("decode_per_token_ms", "0"))
        new_decode = safe_float(new.get("decode_per_token_ms", "0"))
        delta_rows.append(
            {
                "split": key[0],
                "mode": key[1],
                "rollout_tok_s_old": f"{old_rollout:.3f}",
                "rollout_tok_s_new": f"{new_rollout:.3f}",
                "rollout_delta_pct": fmt_pct(pct_delta(old_rollout, new_rollout)),
                "decode_ms_old": f"{old_decode:.3f}",
                "decode_ms_new": f"{new_decode:.3f}",
                "decode_delta_pct": fmt_pct(pct_delta(old_decode, new_decode)),
            }
        )

    label_slug = slugify_label(baseline_label)
    out_csv = out_dir / f"stage12_delta_vs_{label_slug}.csv"
    out_md = out_dir / f"stage12_delta_vs_{label_slug}.md"
    write_csv(out_csv, delta_rows)

    md: List[str] = []
    md.append(f"# stage1.2 delta vs {baseline_label}")
    md.append("")
    md.append("## Per-split deltas (old -> new)")
    md.append("| split | mode | rollout_tok/s | delta | decode_ms/token | delta |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for r in delta_rows:
        md.append(
            "| "
            + " | ".join(
                [
                    r["split"],
                    r["mode"],
                    f"{r['rollout_tok_s_old']} -> {r['rollout_tok_s_new']}",
                    r["rollout_delta_pct"],
                    f"{r['decode_ms_old']} -> {r['decode_ms_new']}",
                    r["decode_delta_pct"],
                ]
            )
            + " |"
        )
    if not delta_rows:
        md.append("| - | - | - | - | - | - |")

    old_best = pick_best_row(baseline_summary)
    new_best = pick_best_row(current_summary)
    if old_best is not None and new_best is not None:
        old_best_roll = safe_float(old_best.get("rollout_tok_s_est", "0"))
        new_best_roll = safe_float(new_best.get("rollout_tok_s_est", "0"))
        md.append("")
        md.append("## Best config")
        md.append(
            f"- old best: `{old_best.get('split','')} {old_best.get('mode','')}` rollout "
            f"`{old_best_roll:.3f}` tok/s, decode `{safe_float(old_best.get('decode_per_token_ms','0')):.3f}` ms/token"
        )
        md.append(
            f"- new best: `{new_best.get('split','')} {new_best.get('mode','')}` rollout "
            f"`{new_best_roll:.3f}` tok/s, decode `{safe_float(new_best.get('decode_per_token_ms','0')):.3f}` ms/token"
        )
        md.append(f"- best-to-best rollout change: `{fmt_pct(pct_delta(old_best_roll, new_best_roll))}`")
        md.append(
            f"- best-to-best decode change: "
            f"`{fmt_pct(pct_delta(safe_float(old_best.get('decode_per_token_ms','0')), safe_float(new_best.get('decode_per_token_ms','0'))))}`"
        )

    old_ratio = read_rollout_vs_llama(baseline_vs_llama_csv)
    new_ratio = read_rollout_vs_llama(current_vs_llama_csv)
    if old_ratio is not None and new_ratio is not None:
        md.append("")
        md.append("## Ember vs llama dual (from each report)")
        md.append(
            f"- old: ember `{old_ratio[0]:.3f}` / llama dual `{old_ratio[1]:.3f}` = `{old_ratio[2]:.2f}%`"
        )
        md.append(
            f"- new: ember `{new_ratio[0]:.3f}` / llama dual `{new_ratio[1]:.3f}` = `{new_ratio[2]:.2f}%`"
        )
        md.append("")
        md.append("## Note")
        md.append("- llama numbers are rerun-dependent; cross-date ratio change includes llama-side variance.")

    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    return out_csv, out_md


def write_p2_input(
    out_path: Path,
    model_dir: Path,
    prompt_len: int,
    decode_steps: int,
    current_summary: List[Dict[str, str]],
    baseline_summary: Optional[List[Dict[str, str]]],
    baseline_label: str,
    anchor_summary: Optional[List[Dict[str, str]]],
    anchor_label: str,
    current_vs_llama_csv: Optional[Path],
    note: str,
) -> None:
    new_best = pick_best_row(current_summary)
    if new_best is None:
        return

    lines: List[str] = []
    lines.append(f"# Stage1.2 P2 Input ({dt.date.today().isoformat()})")
    lines.append("")
    lines.append(f"Model: {pretty_model_name(model_dir)} (2x3080Ti)")
    lines.append(f"Setting: prompt_len={prompt_len}, decode_steps={decode_steps}, full split sweep")
    lines.append("")
    lines.append("## Best config (new)")
    lines.append(f"- split/mode: {new_best.get('split','')} {new_best.get('mode','')}")
    lines.append(f"- rollout: {safe_float(new_best.get('rollout_tok_s_est','0')):.3f} tok/s")
    lines.append(f"- decode: {safe_float(new_best.get('decode_per_token_ms','0')):.3f} ms/token")
    lines.append(f"- prefill: {safe_float(new_best.get('prefill_wall_ms','0')):.3f} ms")

    if baseline_summary:
        old_best = pick_best_row(baseline_summary)
        if old_best is not None:
            old_roll = safe_float(old_best.get("rollout_tok_s_est", "0"))
            new_roll = safe_float(new_best.get("rollout_tok_s_est", "0"))
            old_dec = safe_float(old_best.get("decode_per_token_ms", "0"))
            new_dec = safe_float(new_best.get("decode_per_token_ms", "0"))
            lines.append("")
            lines.append(f"## Delta vs previous full sweep ({baseline_label})")
            lines.append(
                f"- old best: {old_best.get('split','')} {old_best.get('mode','')}, "
                f"rollout {old_roll:.3f} tok/s, decode {old_dec:.3f} ms/token"
            )
            lines.append(
                f"- new best: {new_best.get('split','')} {new_best.get('mode','')}, "
                f"rollout {new_roll:.3f} tok/s, decode {new_dec:.3f} ms/token"
            )
            lines.append(f"- best rollout delta: {fmt_pct(pct_delta(old_roll, new_roll))}")
            lines.append(f"- best decode delta: {fmt_pct(pct_delta(old_dec, new_dec))}")

    if anchor_summary:
        anchor_best = pick_best_row(anchor_summary)
        if anchor_best is not None:
            old_roll = safe_float(anchor_best.get("rollout_tok_s_est", "0"))
            new_roll = safe_float(new_best.get("rollout_tok_s_est", "0"))
            old_dec = safe_float(anchor_best.get("decode_per_token_ms", "0"))
            new_dec = safe_float(new_best.get("decode_per_token_ms", "0"))
            lines.append("")
            lines.append(f"## Delta vs {anchor_label} (historical anchor)")
            lines.append(
                f"- old anchor best: {anchor_best.get('split','')} {anchor_best.get('mode','')}, "
                f"rollout {old_roll:.3f} tok/s, decode {old_dec:.3f} ms/token"
            )
            lines.append(
                f"- new best: {new_best.get('split','')} {new_best.get('mode','')}, "
                f"rollout {new_roll:.3f} tok/s, decode {new_dec:.3f} ms/token"
            )
            lines.append(f"- rollout change: {fmt_pct(pct_delta(old_roll, new_roll))}")
            lines.append(f"- decode change: {fmt_pct(pct_delta(old_dec, new_dec))}")

    ratio = read_rollout_vs_llama(current_vs_llama_csv)
    if ratio is not None:
        lines.append("")
        lines.append("## Ember vs llama dual (same run)")
        lines.append(f"- llama dual rollout: {ratio[1]:.3f} tok/s")
        lines.append(f"- ember best rollout: {ratio[0]:.3f} tok/s")
        lines.append(f"- ember/llama dual: {ratio[2]:.2f}%")

    if note.strip():
        lines.append("")
        lines.append("## Implementation note")
        lines.append(f"- {note.strip()}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 1.2 split sweep profiling in one command.")
    ap.add_argument("--model", type=str, default=os.environ.get("MODEL_PATH", ""), help="model path or HF model id")
    ap.add_argument("--gpus", type=str, default="0,1", help="GPU ids, expected 2 GPUs for split profiling")
    ap.add_argument("--splits", type=str, default="", help="explicit split list, e.g. '12,24;18,18;24,12'")
    ap.add_argument("--prompt-len", type=int, default=2048)
    ap.add_argument("--decode-steps", type=int, default=128)
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--decode-with-sampling", action="store_true", default=True)
    ap.add_argument("--decode-no-sampling", dest="decode_with_sampling", action="store_false")
    ap.add_argument("--pipeline-2gpu", action="store_true", default=True)
    ap.add_argument("--no-pipeline-2gpu", dest="pipeline_2gpu", action="store_false")
    ap.add_argument("--retry-oom", action="store_true", default=True)
    ap.add_argument("--no-retry-oom", dest="retry_oom", action="store_false")
    ap.add_argument("--min-chunk-len", type=int, default=64)
    ap.add_argument("--retry-runtime-unavailable", type=int, default=2)
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    ap.add_argument("--stop-on-error", dest="continue_on_error", action="store_false")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    ap.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    ap.add_argument("--include-llama", action="store_true", help="also run llama-bench baseline and output stage12_vs_llama.csv/md")
    ap.add_argument("--llama-bin-dir", type=str, default="", help="path to llama.cpp build/bin (must contain llama-bench)")
    ap.add_argument("--llama-gguf", type=str, default="", help="path to GGUF model for llama-bench")
    ap.add_argument("--llama-single-device", type=str, default="CUDA0")
    ap.add_argument("--llama-dual-device", type=str, default="CUDA0/CUDA1")
    ap.add_argument("--llama-dual-split-mode", type=str, default="layer", choices=["none", "layer", "row"])
    ap.add_argument("--llama-dual-tensor-split", type=str, default="", help="optional tensor split ratios, e.g. 0.5/0.5")
    ap.add_argument("--baseline-summary", type=str, default="", help="optional prior stage12_split_summary.csv for delta report")
    ap.add_argument("--baseline-label", type=str, default="", help="optional label for --baseline-summary")
    ap.add_argument("--baseline-vs-llama", type=str, default="", help="optional prior stage12_vs_llama.csv")
    ap.add_argument("--anchor-summary", type=str, default="", help="optional historical stage12_split_summary.csv for P2 note")
    ap.add_argument("--anchor-label", type=str, default="", help="optional label for --anchor-summary")
    ap.add_argument("--p2-note", type=str, default="", help="optional implementation note appended to stage12_p2_input.md")
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if not args.model:
        die("--model is required (or set MODEL_PATH)")
    model_dir = resolve_model_dir(args.model)

    if args.prompt_len <= 0:
        die("--prompt-len must be > 0")
    if args.decode_steps < 0:
        die("--decode-steps must be >= 0")
    if args.min_chunk_len <= 0:
        die("--min-chunk-len must be > 0")

    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if len(gpus) != 2:
        die("--gpus must specify exactly 2 GPUs for stage 1.2 split profiling")
    gpus_str = ",".join(gpus)

    num_layers = read_num_layers(model_dir)
    splits = parse_splits(args.splits, num_layers) if args.splits.strip() else auto_splits(num_layers)

    repo = Path.cwd()
    build_dir = (repo / args.build_dir).resolve()
    bin_stage = build_dir / "ember_stage_breakdown"
    if not bin_stage.exists():
        die(f"binary not found: {bin_stage} (build first)")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage1_split_profile_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: List[Dict[str, str]] = []
    failed_rows: List[Dict[str, str]] = []

    run_id = 0
    for (a, b) in splits:
        for ov in [0, 1]:
            run_id += 1
            mode = "overlap" if ov == 1 else "no_overlap"
            run_csv = out_dir / f"stage12_raw_{run_id:03d}_split_{a}_{b}_{mode}.csv"

            if args.skip_existing and run_csv.exists() and run_csv.stat().st_size > 0:
                rows = read_csv(run_csv)
                for r in rows:
                    r["run_id"] = str(run_id)
                    r["split"] = f"{a}+{b}"
                raw_rows.extend(rows)
                print(f"[skip {run_id}] reuse existing {run_csv.name}")
                continue

            cmd = [
                str(bin_stage),
                "--model", str(model_dir),
                "--gpus", gpus_str,
                "--split", f"{a},{b}",
                "--prompt-len", str(args.prompt_len),
                "--decode-steps", str(args.decode_steps),
                "--chunk-len", str(args.chunk_len),
                "--iters", str(args.iters),
                "--warmup", str(args.warmup),
                "--csv", str(run_csv),
            ]
            if args.pipeline_2gpu:
                cmd += ["--pipeline"]
            cmd += ["--overlap"] if ov == 1 else ["--no-overlap"]
            cmd += ["--decode-with-sampling"] if args.decode_with_sampling else ["--decode-no-sampling"]

            chunk_len = args.chunk_len
            attempt = 0
            saw_oom = False
            saw_runtime = False
            min_oom_chunk: Optional[int] = None
            success = False

            while True:
                attempt += 1
                run_log = logs_dir / f"run_{run_id:03d}_try{attempt}.log"
                run_cmdline = list(cmd)
                # keep chunk_len dynamic for OOM retries
                ci = run_cmdline.index("--chunk-len")
                run_cmdline[ci + 1] = str(chunk_len)

                print(f"[run {run_id}] split={a}+{b} mode={mode} chunk={chunk_len} try={attempt}")
                rc, out, err = run_cmd(run_cmdline, cwd=repo, log_path=run_log)
                if rc == 0:
                    success = True
                    break

                merged = (out or "") + "\n" + (err or "")
                hint = detect_error_hint(merged)
                if hint == "oom":
                    saw_oom = True
                    min_oom_chunk = chunk_len if min_oom_chunk is None else min(min_oom_chunk, chunk_len)
                elif hint == "runtime_unavailable":
                    saw_runtime = True

                if hint == "runtime_unavailable" and attempt <= args.retry_runtime_unavailable:
                    print(f"[retry {run_id}] CUDA runtime unavailable, retrying in 2s")
                    time.sleep(2.0)
                    continue

                if args.retry_oom and hint == "oom" and chunk_len > args.min_chunk_len:
                    next_chunk = max(args.min_chunk_len, chunk_len // 2)
                    if next_chunk == chunk_len:
                        break
                    print(f"[retry {run_id}] OOM detected, chunk_len {chunk_len} -> {next_chunk}")
                    chunk_len = next_chunk
                    continue

                final_hint = "oom" if saw_oom else ("runtime_unavailable" if saw_runtime else hint)
                final_chunk = min_oom_chunk if (saw_oom and min_oom_chunk is not None) else chunk_len
                failed_rows.append(
                    {
                        "run_id": str(run_id),
                        "split": f"{a}+{b}",
                        "overlap": str(ov),
                        "prompt_len": str(args.prompt_len),
                        "decode_steps": str(args.decode_steps),
                        "chunk_len": str(final_chunk),
                        "return_code": str(rc),
                        "error_hint": final_hint,
                        "log_path": str(run_log),
                    }
                )
                if not args.continue_on_error:
                    die("run failed: " + " ".join(run_cmdline) + f" (log: {run_log})")
                print(f"[warn] run {run_id} failed; continue-on-error enabled")
                break

            if success:
                rows = read_csv(run_csv)
                for r in rows:
                    r["run_id"] = str(run_id)
                    r["split"] = f"{a}+{b}"
                raw_rows.extend(rows)

    # Sort stable for downstream consumption.
    raw_rows.sort(key=lambda r: (r.get("split", ""), r.get("mode", ""), r.get("phase", "")))

    summary_rows = summarize(raw_rows)
    summary_rows.sort(key=lambda r: (r.get("split", ""), r.get("mode", "")))

    transfer_rows = build_transfer_vs_compute(summary_rows)
    bubble_rows = build_bubble_vs_split(summary_rows)

    raw_csv = out_dir / "stage12_raw_rows.csv"
    summary_csv = out_dir / "stage12_split_summary.csv"
    transfer_csv = out_dir / "stage12_transfer_vs_compute.csv"
    bubble_csv = out_dir / "stage12_bubble_vs_split.csv"
    summary_md = out_dir / "stage12_summary.md"
    failures_csv = out_dir / "stage12_failures.csv"

    write_csv(raw_csv, raw_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(transfer_csv, transfer_rows)
    write_csv(bubble_csv, bubble_rows)
    write_markdown(summary_md, model_dir, summary_rows, bubble_rows)

    write_csv(
        failures_csv,
        failed_rows,
        fieldnames=[
            "run_id",
            "split",
            "overlap",
            "prompt_len",
            "decode_steps",
            "chunk_len",
            "return_code",
            "error_hint",
            "log_path",
        ],
    )

    current_vs_llama_csv: Optional[Path] = None
    if args.include_llama:
        if not args.llama_bin_dir.strip():
            die("--llama-bin-dir is required with --include-llama")
        if not args.llama_gguf.strip():
            die("--llama-gguf is required with --include-llama")
        if not summary_rows:
            die("cannot compare against llama: stage12 summary has no usable rows")

        llama_bin_dir = Path(args.llama_bin_dir).expanduser().resolve()
        gguf_path = Path(args.llama_gguf).expanduser().resolve()
        ember_best = max(summary_rows, key=lambda r: safe_float(r.get("rollout_tok_s_est", "0")))

        llama_rows: List[Dict[str, str]] = []
        try:
            single = run_llama_bench_once(
                repo=repo,
                llama_bin_dir=llama_bin_dir,
                gguf_path=gguf_path,
                prompt_len=args.prompt_len,
                decode_steps=args.decode_steps,
                device=args.llama_single_device,
                split_mode="layer",
                tensor_split="",
                log_path=logs_dir / "llama_single.log",
            )
            llama_rows.append(
                {
                    "engine": "llama.cpp",
                    "config": f"{args.llama_single_device} single",
                    "prefill_ms": f"{single['prefill_ms']:.3f}",
                    "decode_tok_s": f"{single['decode_tok_s']:.3f}",
                    "decode_per_token_ms": f"{single['decode_per_token_ms']:.3f}",
                    "rollout_total_ms": f"{single['rollout_total_ms']:.3f}",
                    "rollout_tok_s": f"{single['rollout_tok_s']:.3f}",
                    "notes": "llama-bench",
                }
            )
        except Exception as ex:
            llama_rows.append(
                {
                    "engine": "llama.cpp",
                    "config": f"{args.llama_single_device} single",
                    "prefill_ms": "",
                    "decode_tok_s": "",
                    "decode_per_token_ms": "",
                    "rollout_total_ms": "",
                    "rollout_tok_s": "",
                    "notes": f"failed: {ex}",
                }
            )

        try:
            dual = run_llama_bench_once(
                repo=repo,
                llama_bin_dir=llama_bin_dir,
                gguf_path=gguf_path,
                prompt_len=args.prompt_len,
                decode_steps=args.decode_steps,
                device=args.llama_dual_device,
                split_mode=args.llama_dual_split_mode,
                tensor_split=args.llama_dual_tensor_split,
                log_path=logs_dir / "llama_dual.log",
            )
            llama_rows.append(
                {
                    "engine": "llama.cpp",
                    "config": f"{args.llama_dual_device} dual",
                    "prefill_ms": f"{dual['prefill_ms']:.3f}",
                    "decode_tok_s": f"{dual['decode_tok_s']:.3f}",
                    "decode_per_token_ms": f"{dual['decode_per_token_ms']:.3f}",
                    "rollout_total_ms": f"{dual['rollout_total_ms']:.3f}",
                    "rollout_tok_s": f"{dual['rollout_tok_s']:.3f}",
                    "notes": f"llama-bench split_mode={args.llama_dual_split_mode}",
                }
            )
        except Exception as ex:
            llama_rows.append(
                {
                    "engine": "llama.cpp",
                    "config": f"{args.llama_dual_device} dual",
                    "prefill_ms": "",
                    "decode_tok_s": "",
                    "decode_per_token_ms": "",
                    "rollout_total_ms": "",
                    "rollout_tok_s": "",
                    "notes": f"failed: {ex}",
                }
            )

        write_vs_llama_outputs(
            out_dir=out_dir,
            ember_best=ember_best,
            llama_rows=llama_rows,
            gguf_path=gguf_path,
            split_mode=args.llama_dual_split_mode,
            tensor_split=args.llama_dual_tensor_split,
        )
        current_vs_llama_csv = out_dir / "stage12_vs_llama.csv"

    if current_vs_llama_csv is None:
        current_vs_llama_csv = resolve_existing_csv("", out_dir / "stage12_vs_llama.csv")

    baseline_summary_rows: Optional[List[Dict[str, str]]] = None
    baseline_label = ""
    baseline_vs_llama_csv: Optional[Path] = None
    delta_csv: Optional[Path] = None
    delta_md: Optional[Path] = None
    if args.baseline_summary.strip():
        baseline_summary_path = resolve_existing_csv(args.baseline_summary)
        if baseline_summary_path is None:
            die("invalid --baseline-summary")
        baseline_summary_rows = read_csv(baseline_summary_path)
        baseline_label = args.baseline_label.strip() or infer_label_from_summary(baseline_summary_path)
        baseline_vs_llama_csv = resolve_existing_csv(
            args.baseline_vs_llama,
            baseline_summary_path.parent / "stage12_vs_llama.csv",
        )
        delta_csv, delta_md = write_delta_outputs(
            out_dir=out_dir,
            current_summary=summary_rows,
            baseline_summary=baseline_summary_rows,
            baseline_label=baseline_label,
            current_vs_llama_csv=current_vs_llama_csv,
            baseline_vs_llama_csv=baseline_vs_llama_csv,
        )

    anchor_summary_rows: Optional[List[Dict[str, str]]] = None
    anchor_label = ""
    if args.anchor_summary.strip():
        anchor_summary_path = resolve_existing_csv(args.anchor_summary)
        if anchor_summary_path is None:
            die("invalid --anchor-summary")
        anchor_summary_rows = read_csv(anchor_summary_path)
        anchor_label = args.anchor_label.strip() or infer_label_from_summary(anchor_summary_path)

    p2_md: Optional[Path] = None
    if baseline_summary_rows is not None or anchor_summary_rows is not None or args.p2_note.strip() or current_vs_llama_csv is not None:
        p2_md = out_dir / "stage12_p2_input.md"
        write_p2_input(
            out_path=p2_md,
            model_dir=model_dir,
            prompt_len=args.prompt_len,
            decode_steps=args.decode_steps,
            current_summary=summary_rows,
            baseline_summary=baseline_summary_rows,
            baseline_label=baseline_label,
            anchor_summary=anchor_summary_rows,
            anchor_label=anchor_label,
            current_vs_llama_csv=current_vs_llama_csv,
            note=args.p2_note,
        )
        if not p2_md.exists():
            p2_md = None

    print("")
    print("[done] stage1.2 split profiling completed")
    print(f"- raw rows: {raw_csv}")
    print(f"- split summary: {summary_csv}")
    print(f"- transfer vs compute: {transfer_csv}")
    print(f"- bubble vs split: {bubble_csv}")
    print(f"- summary md: {summary_md}")
    if args.include_llama:
        print(f"- vs llama csv: {out_dir / 'stage12_vs_llama.csv'}")
        print(f"- vs llama md: {out_dir / 'stage12_vs_llama.md'}")
    if delta_csv is not None and delta_md is not None:
        print(f"- delta csv: {delta_csv}")
        print(f"- delta md: {delta_md}")
    if p2_md is not None:
        print(f"- p2 input md: {p2_md}")
    if failed_rows:
        print(f"- failures: {failures_csv} ({len(failed_rows)} failed runs)")
    else:
        print(f"- failures: {failures_csv} (0 failed runs)")
    print(f"- logs: {logs_dir}")


if __name__ == "__main__":
    main()
