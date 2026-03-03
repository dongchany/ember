#!/usr/bin/env python3
import argparse
import csv
import datetime as dt
import re
import subprocess
from pathlib import Path
from typing import Dict, List


SECTION_PATTERNS = {
    "attn_out": re.compile(r"^Attn out vs HF self_attn output$"),
    "attn_residual": re.compile(r"^Attn residual vs HF \(residual \+ attn_out\)$"),
    "post_attn_norm": re.compile(r"^Post-attn norm vs HF layernorm$"),
    "mlp_out": re.compile(r"^MLP out vs HF mlp output$"),
    "gate_proj": re.compile(r"^Gate proj vs HF gate_proj$"),
    "up_proj": re.compile(r"^Up proj vs HF up_proj$"),
    "gate_proj_ember_norm": re.compile(r"^Gate proj vs HF gate_proj \(Ember norm input\)$"),
    "up_proj_ember_norm": re.compile(r"^Up proj vs HF up_proj \(Ember norm input\)$"),
}

LAYER_RE = re.compile(r"^([0-9]+)\t([0-9eE+\-.]+)\t([0-9eE+\-.]+)$")
MAX_RE = re.compile(r"^max_abs_diff:\s*([0-9eE+\-.]+)$")
MEAN_RE = re.compile(r"^mean_abs_diff:\s*([0-9eE+\-.]+)$")


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def parse_layers(raw: str) -> List[int]:
    vals: List[int] = []
    for x in raw.split(","):
        v = x.strip()
        if not v:
            continue
        vals.append(int(v))
    if not vals:
        die("empty layers")
    return sorted(set(vals))


def run_cmd(cmd: List[str], cwd: Path) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"command failed rc={p.returncode}: {' '.join(cmd)}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    return (p.stdout or "") + ("\n[stderr]\n" + p.stderr if p.stderr else "")


def parse_compare_hidden_output(text: str, layer: int) -> Dict[str, str]:
    row: Dict[str, str] = {
        "layer": str(layer),
        "layer_max_abs_diff": "",
        "layer_mean_abs_diff": "",
        "attn_out_max_abs_diff": "",
        "attn_residual_max_abs_diff": "",
        "post_attn_norm_max_abs_diff": "",
        "mlp_out_max_abs_diff": "",
        "gate_proj_max_abs_diff": "",
        "up_proj_max_abs_diff": "",
        "gate_proj_ember_norm_max_abs_diff": "",
        "up_proj_ember_norm_max_abs_diff": "",
    }

    current_key = None
    lines = [ln.strip() for ln in text.splitlines()]
    for i, ln in enumerate(lines):
        m = LAYER_RE.match(ln)
        if m and int(m.group(1)) == layer:
            row["layer_max_abs_diff"] = m.group(2)
            row["layer_mean_abs_diff"] = m.group(3)
            continue
        for key, pat in SECTION_PATTERNS.items():
            if pat.match(ln):
                current_key = key
                break
        if current_key is None:
            continue
        mmax = MAX_RE.match(ln)
        if mmax:
            out_key = f"{current_key}_max_abs_diff"
            if out_key in row:
                row[out_key] = mmax.group(1)
            current_key = None
            continue
        # fallback: max is typically next line
        if i + 1 < len(lines):
            mmax2 = MAX_RE.match(lines[i + 1])
            if mmax2:
                out_key = f"{current_key}_max_abs_diff"
                if out_key in row and not row[out_key]:
                    row[out_key] = mmax2.group(1)
                current_key = None
                continue
        if MEAN_RE.match(ln):
            current_key = None
    return row


def write_md(path: Path, rows: List[Dict[str, str]], model: Path, debug_root: Path) -> None:
    lines: List[str] = []
    lines.append("# Stage31 Base Operator Spotcheck")
    lines.append("")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Model: `{model}`")
    lines.append(f"- Debug root: `{debug_root}`")
    lines.append("")
    lines.append("| layer | layer_max | attn_out | attn_residual | post_attn_norm | mlp_out | gate_proj | gate_proj(ember_norm) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        lines.append(
            f"| {r['layer']} | {r['layer_max_abs_diff']} | {r['attn_out_max_abs_diff']} | "
            f"{r['attn_residual_max_abs_diff']} | {r['post_attn_norm_max_abs_diff']} | "
            f"{r['mlp_out_max_abs_diff']} | {r['gate_proj_max_abs_diff']} | {r['gate_proj_ember_norm_max_abs_diff']} |"
        )
    lines.append("")
    lines.append("## Key Point")
    for r in rows:
        if r["gate_proj_max_abs_diff"] and r["gate_proj_ember_norm_max_abs_diff"]:
            lines.append(
                f"- layer {r['layer']}: gate_proj raw={r['gate_proj_max_abs_diff']} vs "
                f"ember_norm_input={r['gate_proj_ember_norm_max_abs_diff']}."
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run compare_hidden spotchecks for selected layers.")
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--debug-root", type=str, required=True, help="Root directory containing layer_<N> subdirs.")
    ap.add_argument("--layers", type=str, default="0,1")
    ap.add_argument("--python-bin", type=str, default="./sglang-env/bin/python")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    repo = Path.cwd()
    model = Path(args.model).expanduser().resolve()
    debug_root = Path(args.debug_root).expanduser().resolve()
    python_bin = Path(args.python_bin).expanduser()
    if not model.exists():
        die(f"model not found: {model}")
    if not debug_root.exists():
        die(f"debug-root not found: {debug_root}")
    if not python_bin.exists():
        die(f"python-bin not found: {python_bin}")

    layers = parse_layers(args.layers)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (
        repo / "reports" / f"stage31_base_operator_spotcheck_{ts}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, str]] = []
    for layer in layers:
        dbg = debug_root / f"layer_{layer}"
        if not dbg.exists():
            die(f"missing layer debug dir: {dbg}")
        cmd = [
            str(python_bin),
            "scripts/compare_hidden.py",
            "--model", str(model),
            "--debug-dir", str(dbg),
            "--layer", str(layer),
        ]
        out = run_cmd(cmd, cwd=repo)
        (logs_dir / f"layer_{layer}.log").write_text(out, encoding="utf-8")
        rows.append(parse_compare_hidden_output(out, layer=layer))

    csv_path = out_dir / "stage31_base_operator_spotcheck.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_path = out_dir / "stage31_summary.md"
    write_md(md_path, rows=rows, model=model, debug_root=debug_root)

    print(f"[done] out_dir={out_dir}")
    print(f"[done] csv={csv_path}")
    print(f"[done] md={md_path}")


if __name__ == "__main__":
    main()
