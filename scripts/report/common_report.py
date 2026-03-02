#!/usr/bin/env python3
import csv
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


def die(msg: str) -> None:
    raise SystemExit(f"error: {msg}")


def safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def split_ints(raw: str) -> List[int]:
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        die("empty int list")
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
        p
        for p in snap_root.iterdir()
        if p.is_dir() and (p / "config.json").exists() and list(p.glob("*.safetensors"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_model_dir(model_arg: str, hub_root: Optional[Path] = None) -> Path:
    raw = model_arg.strip()
    if not raw:
        die("--model is empty")
    path = Path(raw).expanduser().resolve()
    resolved = resolve_snapshot_dir(path)
    if resolved is not None:
        return resolved
    cache_root = hub_root.expanduser().resolve() if hub_root else hf_hub_root()
    model_cache_dir = cache_root / ("models--" + raw.replace("/", "--"))
    resolved = resolve_snapshot_dir(model_cache_dir)
    if resolved is not None:
        return resolved
    die(
        "failed to resolve model from local cache: "
        f"{raw}. Checked path='{path}' and HF cache='{model_cache_dir}'."
    )
    raise AssertionError("unreachable")


def resolve_model_arg(model_arg: str, hub_root: Optional[Path] = None) -> Path:
    """
    Resolve a model argument to a concrete snapshot dir.

    Resolution order:
    1) Existing local path (snapshot dir or HF cache model dir with snapshots/*)
    2) Exact HF cache-id mapping under provided hub_root (if given)
    3) Name-substring match under provided hub_root (if given)
    4) Fallback to default local HF cache resolution
    """
    raw = model_arg.strip()
    if not raw:
        raise FileNotFoundError("model argument is empty")

    path = Path(raw).expanduser()
    if path.exists():
        resolved = resolve_snapshot_dir(path.resolve())
        if resolved is None:
            raise FileNotFoundError(
                "cannot resolve model dir (need config.json + *.safetensors or snapshots/*): "
                f"{path.resolve()}"
            )
        return resolved

    if hub_root is not None:
        root = hub_root.expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"hub root not found: {root}")

        # Try exact HF cache-id mapping first.
        exact_cache_dir = root / ("models--" + raw.replace("/", "--"))
        resolved = resolve_snapshot_dir(exact_cache_dir)
        if resolved is not None:
            return resolved

        # Then allow name-substring search (e.g. "Qwen3-8B").
        candidates: List[Path] = []
        needle = raw.lower()
        for d in root.iterdir():
            if not d.is_dir():
                continue
            if needle not in d.name.lower():
                continue
            snap = resolve_snapshot_dir(d)
            if snap is not None:
                candidates.append(snap)
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]
        raise FileNotFoundError(f"cannot resolve '{raw}' under hub root: {root}")

    # Fallback to default HF cache id resolution.
    default_root = hf_hub_root()
    cache_dir = default_root / ("models--" + raw.replace("/", "--"))
    resolved = resolve_snapshot_dir(cache_dir)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(
        "failed to resolve model from local cache: "
        f"{raw}. Checked path='{path.resolve()}' and HF cache='{cache_dir}'."
    )


def read_csv(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        return
    fn = fieldnames if fieldnames else list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def format_md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_(no data)_\n"
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def run_logged_cmd(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
    progress: bool = False,
) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if progress:
        print(f"[run] {' '.join(cmd)}", flush=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=merged_env)
        f.write(p.stdout)
        if p.stderr:
            f.write("\n[stderr]\n")
            f.write(p.stderr)
    if progress:
        dt_s = time.time() - t0
        if p.returncode == 0:
            tail = ""
            lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
            if lines:
                tail = lines[-1]
            msg = f"[ok ] rc=0 ({dt_s:.1f}s)"
            if tail:
                msg += f" | {tail}"
            print(msg, flush=True)
        else:
            err_tail = ""
            elines = [ln.strip() for ln in p.stderr.splitlines() if ln.strip()]
            if elines:
                err_tail = elines[-1]
            msg = f"[err] rc={p.returncode} ({dt_s:.1f}s)"
            if err_tail:
                msg += f" | {err_tail}"
            msg += f" | log={log_path}"
            print(msg, flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(cmd)} (see {log_path})")
    return p


def run_cmd(
    cmd: List[str],
    cwd: Path,
    log_path: Path,
    env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, env=merged_env)
        f.write(p.stdout)
        if p.stderr:
            f.write("\n[stderr]\n")
            f.write(p.stderr)
    if check and p.returncode != 0:
        die(f"command failed rc={p.returncode}: {' '.join(cmd)} (see {log_path})")
    return p
