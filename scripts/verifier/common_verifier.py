#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception as e:
                die(f"{path}:{ln}: invalid jsonl line: {e}")
            if not isinstance(obj, dict):
                die(f"{path}:{ln}: each line must be a JSON object")
            rows.append(obj)
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
