#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple

from common_verifier import die, load_jsonl, write_csv


def normalize_rows(rows: List[Tuple[Any, ...]]) -> List[Tuple[str, ...]]:
    norm: List[Tuple[str, ...]] = []
    for row in rows:
        norm.append(tuple("null" if v is None else str(v) for v in row))
    norm.sort()
    return norm


def is_select_like(sql: str) -> bool:
    s = sql.strip().lower()
    return s.startswith("select") or s.startswith("with")


def run_query(db_path: Path, sql: str) -> Tuple[bool, List[Tuple[Any, ...]], str]:
    if not is_select_like(sql):
        return False, [], "non_select_query_blocked"
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        con.close()
        return True, rows, ""
    except Exception as e:
        return False, [], str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="SQL verifier (SQLite result-set comparison).")
    ap.add_argument("--pred-jsonl", type=str, required=True, help='jsonl rows: {"id","sql"}')
    ap.add_argument("--gold-jsonl", type=str, required=True, help='jsonl rows: {"id","db_path","gold_sql"}')
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    pred_path = Path(args.pred_jsonl).expanduser().resolve()
    gold_path = Path(args.gold_jsonl).expanduser().resolve()
    for p in [pred_path, gold_path]:
        if not p.exists():
            die(f"missing file: {p}")

    preds = load_jsonl(pred_path)
    golds = load_jsonl(gold_path)
    pred_by_id = {str(r.get("id", "")): r for r in preds}
    gold_by_id = {str(r.get("id", "")): r for r in golds}
    ids = sorted(set(pred_by_id.keys()) & set(gold_by_id.keys()))
    if not ids:
        die("no overlapping ids")

    per_rows: List[Dict[str, str]] = []
    exec_ok_cnt = 0
    match_cnt = 0
    rewards: List[float] = []

    for rid in ids:
        pr = pred_by_id[rid]
        gr = gold_by_id[rid]
        if "sql" not in pr:
            die(f"pred id={rid} missing sql")
        if "db_path" not in gr or "gold_sql" not in gr:
            die(f"gold id={rid} missing db_path/gold_sql")

        db_path = Path(str(gr["db_path"])).expanduser().resolve()
        if not db_path.exists():
            die(f"gold id={rid} db not found: {db_path}")

        pred_sql = str(pr["sql"])
        gold_sql = str(gr["gold_sql"])
        ok_p, rows_p, err_p = run_query(db_path, pred_sql)
        ok_g, rows_g, err_g = run_query(db_path, gold_sql)
        if not ok_g:
            die(f"gold sql failed id={rid}: {err_g}")

        exec_ok = 1 if ok_p else 0
        match = 0
        if ok_p:
            match = 1 if normalize_rows(rows_p) == normalize_rows(rows_g) else 0

        reward = float(match)
        exec_ok_cnt += exec_ok
        match_cnt += match
        rewards.append(reward)

        per_rows.append(
            {
                "id": rid,
                "exec_ok": str(exec_ok),
                "result_match": str(match),
                "pred_row_count": str(len(rows_p) if ok_p else 0),
                "gold_row_count": str(len(rows_g)),
                "reward": f"{reward:.6f}",
                "error": err_p,
            }
        )

    n = len(ids)
    summary = {
        "num_samples": n,
        "exec_ok_rate": exec_ok_cnt / n,
        "result_match_rate": match_cnt / n,
        "mean_reward": sum(rewards) / n,
    }

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage51_sql_verifier_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_csv = out_dir / "stage51_sql_per_sample.csv"
    summary_md = out_dir / "stage51_sql_summary.md"
    summary_json = out_dir / "stage51_sql_summary.json"
    write_csv(per_csv, per_rows)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Stage 5.1 SQL Verifier",
        "",
        f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`",
        f"- Samples: `{n}`",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| exec_ok_rate | {summary['exec_ok_rate']:.6f} |",
        f"| result_match_rate | {summary['result_match_rate']:.6f} |",
        f"| mean_reward | {summary['mean_reward']:.6f} |",
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("[done] stage5.1 sql verifier")
    print(f"- per-sample csv: {per_csv}")
    print(f"- summary md: {summary_md}")
    print(f"- summary json: {summary_json}")


if __name__ == "__main__":
    main()
