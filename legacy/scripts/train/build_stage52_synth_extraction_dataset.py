#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from common_train import die


NAMES = [
    "Alice",
    "Bob",
    "Cara",
    "David",
    "Emma",
    "Frank",
    "Grace",
    "Henry",
    "Iris",
    "Jack",
]

CITIES = ["Shanghai", "Beijing", "Shenzhen", "Hangzhou", "Chengdu", "Nanjing", "Wuhan", "Suzhou"]


def rand_date(rng: random.Random) -> str:
    y = rng.randint(2019, 2026)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def make_doc(rng: random.Random, idx: int) -> Dict[str, Any]:
    name = rng.choice(NAMES)
    age = rng.randint(20, 45)
    active = rng.choice([True, False])
    city = rng.choice(CITIES)
    score = round(rng.uniform(60.0, 99.9), 1)
    join_date = rand_date(rng)
    dept = rng.choice(["AI", "Infra", "Ops", "Research"])
    project = rng.choice(["Ember", "Falcon", "Orion", "Nova"])
    noisy_line = f"note#{idx}: latency={rng.randint(12, 200)}ms, mem={rng.randint(1,24)}GB"
    text = (
        f"员工记录 {idx}\n"
        f"name: {name}\n"
        f"age: {age}\n"
        f"active: {'true' if active else 'false'}\n"
        f"city: {city}\n"
        f"score: {score}\n"
        f"join_date: {join_date}\n"
        f"department: {dept}\n"
        f"project: {project}\n"
        f"{noisy_line}\n"
        f"备注：如字段缺失请返回 null。"
    )
    gold = {
        "name": name,
        "age": age,
        "active": active,
        "city": city,
        "score": score,
        "join_date": join_date,
    }
    prompt = (
        "请从下方文本抽取结构化字段，且只输出 JSON（不要输出解释文字）。\n"
        "目标字段: name(string), age(integer), active(boolean), city(string), score(number), join_date(string)\n"
        "文本如下：\n"
        f"{text}"
    )
    return {
        "id": f"synth_{idx:04d}",
        "prompt": prompt,
        "gold": gold,
    }


def make_doc_hard(rng: random.Random, idx: int) -> Dict[str, Any]:
    num_records = 3
    records: List[Dict[str, Any]] = []
    for j in range(num_records):
        rec_id = f"E{idx:04d}{j+1}"
        name = rng.choice(NAMES)
        age = rng.randint(20, 45)
        active = rng.choice([True, False])
        city = rng.choice(CITIES)
        score = round(rng.uniform(60.0, 99.9), 1)
        join_date = rand_date(rng)
        records.append(
            {
                "employee_id": rec_id,
                "name": name,
                "age": age,
                "active": active,
                "city": city,
                "score": score,
                "join_date": join_date,
            }
        )

    target_idx = rng.randint(0, num_records - 1)
    target = records[target_idx]
    # Shuffle display order so target is not always first.
    order = list(range(num_records))
    rng.shuffle(order)

    lines: List[str] = []
    lines.append(f"案件 #{idx}")
    lines.append("以下是候选员工记录，请只抽取指定 employee_id 对应的一条：")
    for oi in order:
        r = records[oi]
        lines.append(
            f"- employee_id={r['employee_id']}; name={r['name']}; age={r['age']}; "
            f"active={'true' if r['active'] else 'false'}; city={r['city']}; "
            f"score={r['score']}; join_date={r['join_date']}"
        )
    lines.append("注意：可能有同名员工，必须按 employee_id 精确匹配。")
    text = "\n".join(lines)

    prompt = (
        "请从下方文本抽取结构化字段，且只输出 JSON（不要输出解释文字）。\n"
        "目标字段: employee_id(string), name(string), age(integer), active(boolean), city(string), score(number), join_date(string)\n"
        f"目标 employee_id: {target['employee_id']}\n"
        "文本如下：\n"
        f"{text}"
    )
    gold = dict(target)
    return {
        "id": f"synth_hard_{idx:04d}",
        "prompt": prompt,
        "gold": gold,
    }


def make_doc_hard_rule(rng: random.Random, idx: int) -> Dict[str, Any]:
    num_records = 4
    records: List[Dict[str, Any]] = []
    for j in range(num_records):
        rec_id = f"R{idx:04d}{j+1}"
        name = rng.choice(NAMES)
        age = rng.randint(20, 45)
        active = rng.choice([True, False])
        city = rng.choice(CITIES)
        score = round(rng.uniform(60.0, 99.9), 1)
        join_date = rand_date(rng)
        records.append(
            {
                "employee_id": rec_id,
                "name": name,
                "age": age,
                "active": active,
                "city": city,
                "score": score,
                "join_date": join_date,
            }
        )

    # Rule target: among active=true records, choose max(score); tie-breaker earliest join_date.
    actives = [r for r in records if r["active"] is True]
    if not actives:
        # Ensure at least one active record exists.
        pick = rng.randrange(num_records)
        records[pick]["active"] = True
        actives = [records[pick]]
    target = sorted(actives, key=lambda r: (-float(r["score"]), str(r["join_date"])))[0]

    order = list(range(num_records))
    rng.shuffle(order)
    lines: List[str] = []
    lines.append(f"任务 #{idx}")
    lines.append("候选记录如下：")
    for oi in order:
        r = records[oi]
        lines.append(
            f"- employee_id={r['employee_id']}; name={r['name']}; age={r['age']}; "
            f"active={'true' if r['active'] else 'false'}; city={r['city']}; "
            f"score={r['score']}; join_date={r['join_date']}"
        )
    lines.append("规则：仅在 active=true 的记录中选择 score 最高的一条；若并列，选择 join_date 最早的一条。")
    text = "\n".join(lines)

    prompt = (
        "请按照给定规则从候选记录中选择唯一目标，并仅输出其 JSON。\n"
        "输出字段: employee_id(string), name(string), age(integer), active(boolean), city(string), score(number), join_date(string)\n"
        "文本如下：\n"
        f"{text}"
    )
    gold = dict(target)
    return {
        "id": f"synth_rule_{idx:04d}",
        "prompt": prompt,
        "gold": gold,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build synthetic extraction dataset for stage5.2.")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--difficulty", type=str, default="easy", choices=["easy", "hard", "hard_rule"])
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.num_samples <= 0:
        die("--num-samples must be > 0")

    rng = random.Random(args.seed)
    if args.difficulty == "hard":
        rows = [make_doc_hard(rng, i + 1) for i in range(args.num_samples)]
        schema = {
            "required": ["employee_id", "name", "age", "active", "city", "score", "join_date"],
            "fields": {
                "employee_id": "string",
                "name": "string",
                "age": "integer",
                "active": "boolean",
                "city": "string",
                "score": "number",
                "join_date": "string",
            },
        }
    elif args.difficulty == "hard_rule":
        rows = [make_doc_hard_rule(rng, i + 1) for i in range(args.num_samples)]
        schema = {
            "required": ["employee_id", "name", "age", "active", "city", "score", "join_date"],
            "fields": {
                "employee_id": "string",
                "name": "string",
                "age": "integer",
                "active": "boolean",
                "city": "string",
                "score": "number",
                "join_date": "string",
            },
        }
    else:
        rows = [make_doc(rng, i + 1) for i in range(args.num_samples)]
        schema = {
            "required": ["name", "age", "active", "city", "score", "join_date"],
            "fields": {
                "name": "string",
                "age": "integer",
                "active": "boolean",
                "city": "string",
                "score": "number",
                "join_date": "string",
            },
        }

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path.cwd() / "reports" / f"stage52_synth_dataset_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_jsonl = out_dir / "dataset.jsonl"
    schema_json = out_dir / "schema.json"
    meta_json = out_dir / "meta.json"

    with ds_jsonl.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    schema_json.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta = {
        "num_samples": args.num_samples,
        "seed": args.seed,
        "difficulty": args.difficulty,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("[done] stage5.2 synth dataset")
    print(f"- dataset: {ds_jsonl}")
    print(f"- schema: {schema_json}")
    print(f"- meta: {meta_json}")


if __name__ == "__main__":
    main()
