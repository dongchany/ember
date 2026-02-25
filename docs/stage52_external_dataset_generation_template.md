# Stage5.2 外部 LLM 数据生成模板

目标：让外部模型（GPT-5.3 / 其他 LLM）生成**可训练、可评估、不过拟合**的信息抽取数据，直接接入 Ember 的 stage5.2 脚本。

---

## 1) 产物格式（必须）

请生成 4 个文件：

1. `train.jsonl`
2. `val.jsonl`
3. `test.jsonl`
4. `schema.json`

`jsonl` 每行格式：

```json
{"id":"...", "prompt":"...", "gold":{"employee_id":"...", "name":"...", "age":30, "active":true, "city":"...", "score":88.5, "join_date":"2024-07-01"}}
```

`schema.json` 示例：

```json
{
  "required": [],
  "fields": {
    "employee_id": "string",
    "name": "string",
    "age": "integer",
    "active": "boolean",
    "city": "string",
    "score": "number",
    "join_date": "string"
  },
  "field_weights": {
    "employee_id": 1.0,
    "name": 1.0,
    "age": 1.0,
    "active": 1.0,
    "city": 1.0,
    "score": 1.0,
    "join_date": 1.0
  }
}
```

---

## 2) 复制给外部模型的 Prompt（生成器）

```text
你是一个数据集构建器。请生成用于“结构化信息抽取”的 train/val/test 数据。

严格要求：
1) 输出 3 份 JSONL 数据（train/val/test）和 1 份 schema.json。
2) 每条样本字段必须是: id, prompt, gold。
3) gold 必须严格符合 schema 类型，且只保留“可确定字段”（无法确定字段直接省略，不填 null）。
4) 不允许把 gold 直接写进 prompt（不能出现“标准答案/参考答案/gold/expected output”等泄漏词）。
5) 任务必须有难度：
   - 多候选干扰（同名/近似字段）
   - 规则选择（过滤、排序、tie-break）
   - 噪声文本
   - 格式扰动（布尔/日期/数字表述变化）
6) 约 20% 样本为“无有效目标”场景：gold 必须是部分字段（1~6 个键），不能输出空对象 `{}`。
7) train/val/test 的 id、prompt、(prompt+gold) 不能重复或泄漏。

规模要求：
- train: 400
- val: 100
- test: 100

输出要求：
- 先输出 schema.json
- 再分别输出 train.jsonl / val.jsonl / test.jsonl 的完整内容
- 只输出 JSON/JSONL，不要解释文字。
```

---

## 3) 复制给外部模型的 Prompt（二次质检器，可选）

```text
请检查我提供的 train/val/test JSONL 和 schema.json，按以下维度给出问题清单：
1) schema 与 gold 类型不一致
2) 缺字段
3) prompt 泄漏答案（包含 gold/reference answer/标准答案/expected output 等）
4) split 间重复（id、prompt、prompt+gold）
5) 难度不足（缺少规则/干扰/噪声）

输出格式：
- 一个 JSON 对象，字段包括:
  - summary
  - errors (数组)
  - warnings (数组)
  - suggested_fixes (数组)
```

---

## 4) 本地验收命令（接入 Ember）

```bash
python3 scripts/train/run_stage52_validate_dataset.py \
  --schema-json /abs/path/schema.json \
  --train-jsonl /abs/path/train.jsonl \
  --val-jsonl /abs/path/val.jsonl \
  --test-jsonl /abs/path/test.jsonl \
  --out-dir reports/stage52_dataset_validation_external_v1
```

通过后再进入训练（评估默认使用 schema 中的 `field_weights` 计算 `mean_reward`）：

```bash
./sglang-env/bin/python scripts/train/run_stage52_sft_min.py \
  --model /abs/path/qwen3-4b-snapshot \
  --dataset-jsonl /abs/path/train.jsonl \
  --python-bin ./sglang-env/bin/python \
  --load-in-4bit \
  --max-train-samples 400 \
  --max-length 128 \
  --max-steps 50
```
