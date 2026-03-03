#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List

from common_train import die


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def count_jsonl_rows(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def add_file(
    *,
    src: Path,
    dst_dir: Path,
    out_name: str,
    manifest_files: List[Dict[str, object]],
) -> None:
    if not src.exists():
        die(f"missing input file: {src}")
    dst = dst_dir / out_name
    shutil.copy2(src, dst)
    item: Dict[str, object] = {
        "name": out_name,
        "source_path": str(src),
        "sha256": sha256_file(dst),
        "bytes": int(dst.stat().st_size),
    }
    if out_name.endswith(".jsonl"):
        item["rows"] = count_jsonl_rows(dst)
    manifest_files.append(item)


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze train/val/test + schema files into a reproducible snapshot.")
    ap.add_argument("--train-jsonl", type=str, required=True)
    ap.add_argument("--val-jsonl", type=str, required=True)
    ap.add_argument("--test-jsonl", type=str, required=True)
    ap.add_argument("--schema-json", type=str, required=True, help="primary schema used in training/eval")
    ap.add_argument("--schema-alt-json", type=str, default="", help="optional secondary schema (e.g. optional-fields)")
    ap.add_argument("--source-zip", type=str, default="", help="optional source archive path")
    ap.add_argument("--tag", type=str, default="external")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    train = Path(args.train_jsonl).expanduser().resolve()
    val = Path(args.val_jsonl).expanduser().resolve()
    test = Path(args.test_jsonl).expanduser().resolve()
    schema = Path(args.schema_json).expanduser().resolve()
    schema_alt = Path(args.schema_alt_json).expanduser().resolve() if args.schema_alt_json.strip() else None
    source_zip = Path(args.source_zip).expanduser().resolve() if args.source_zip.strip() else None

    for p in [train, val, test, schema]:
        if not p.exists():
            die(f"missing path: {p}")
    if schema_alt is not None and not schema_alt.exists():
        die(f"missing path: {schema_alt}")
    if source_zip is not None and not source_zip.exists():
        die(f"source zip not found: {source_zip}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (Path.cwd() / "reports" / f"stage52_dataset_snapshot_{args.tag}_{ts}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_files: List[Dict[str, object]] = []
    add_file(src=train, dst_dir=out_dir, out_name="train.jsonl", manifest_files=manifest_files)
    add_file(src=val, dst_dir=out_dir, out_name="val.jsonl", manifest_files=manifest_files)
    add_file(src=test, dst_dir=out_dir, out_name="test.jsonl", manifest_files=manifest_files)
    add_file(src=schema, dst_dir=out_dir, out_name="schema.json", manifest_files=manifest_files)
    if schema_alt is not None:
        add_file(src=schema_alt, dst_dir=out_dir, out_name="schema_alt.json", manifest_files=manifest_files)

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "source_zip": str(source_zip) if source_zip is not None else "",
        "files": manifest_files,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_lines = [
        "# Stage 5.2 Dataset Snapshot",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Tag: `{args.tag}`",
        f"- Source zip: `{manifest['source_zip']}`",
        f"- Out dir: `{out_dir}`",
        "",
        "| file | rows | bytes | sha256 |",
        "| --- | --- | --- | --- |",
    ]
    for f in manifest_files:
        rows = str(f.get("rows", ""))
        summary_lines.append(f"| {f['name']} | {rows} | {f['bytes']} | {f['sha256']} |")
    summary_path = out_dir / "stage52_dataset_snapshot_summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    print(f"[done] manifest={manifest_path}")
    print(f"[done] summary={summary_path}")


if __name__ == "__main__":
    main()
