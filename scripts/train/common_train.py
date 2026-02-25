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
