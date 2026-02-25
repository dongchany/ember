# Ember Task Checklist v3 (2026-02-25)

> 基于 v2 重排清单更新。标注已完成项、当前进度、近期里程碑（talk）和主线优先级。
> 原则：**一切服务 P1 论文主线**，talk 是 P1 叙事的口头预演。

---

## 状态图例

- [x] 已完成
- [~] 进行中
- [ ] 待做
- [—] 降优先级 / 延后

---

## 0. Hard Gates

- [x] `ember_stage_breakdown` 和 `ember_benchmark` 在单卡 / 双卡通过 smoke run
- [x] 主力模型固定：Qwen3-4B-Instruct-2507（snapshot 已记录）
- [x] GGUF 路径固定：`reports/gguf/Qwen3-4B-BF16.gguf`
- [x] 输出目录约定固定：`reports/<experiment_name>_<date>`

---

## 1. 已完成：Profiling 基础数据

### 1.1 Rollout 时间分解（P1 Fig 2）✅

- [x] Prefill / decode / sampling 分阶段计时导出
- [x] 跑完 context × gen_len 矩阵（512/1024/2048/4096 × 64/128/256，overlap + no_overlap）
- [x] 导出 `p1_fig2_prefill_share.csv`
- [x] 决策结论：prefill share 在 2048/128 下约 16%，4096/64 下达 45.7%。**叙事策略：不夸大单次占比，强调 RL 多轮 × 多候选的累积放大效应**

**产出文件：** `stage1_summary.md`, `p1_fig2_prefill_share.csv`

### 1.2 Pipeline Parallel Profiling（P2 核心数据）✅

- [x] 2×3080Ti 全 split sweep（9+27, 12+24, 18+18, 24+12, 27+9 × overlap/no_overlap）
- [x] 找到最优配置：9+27 overlap, 46.456 tok/s
- [x] 与历史 anchor 对比 delta 记录
- [x] Ember vs llama.cpp 对比：Ember 68.30% of llama.cpp dual

**产出文件：** `stage12_p2_input.md`, `stage12_delta_vs_20260224_opt_decode_residual_full.md`, `stage12_vs_llama.csv`

### 1.3 跨框架对比表（已完成）✅

- [x] Ember single + dual 实测数据
- [x] llama.cpp single + dual 实测数据
- [x] Transformers single(cuda:0) 实测数据（外部隔离 Python 环境）
- [x] vLLM single(tp=2) 实测数据（独立 env）
- [x] SGLang single(tp=1) 实测数据（独立 env）
- [x] `run_framework_compare.py` 已支持 transformers / vLLM / SGLang 实测分支

**产出文件：**
- `reports/framework_compare_4b_20260225_mainline/framework_compare.csv`
- `reports/framework_compare_4b_20260225_mainline/framework_compare.md`
- `reports/framework_compare_4b_20260225_uv_mainline/framework_compare.csv`
- `reports/framework_compare_4b_20260225_envs_tp2_mainline_v2/framework_compare.csv`
- `reports/framework_compare_4b_20260225_envs_tp2_mainline_v2/framework_compare.md`
- `reports/framework_compare_4b_20260225_envs_tp2_stable_mainline/framework_compare.csv`
- `reports/framework_compare_4b_20260225_envs_tp2_stable_run2/framework_compare.csv`
- `reports/framework_compare_4b_20260225_envs_tp2_stable_run3/framework_compare.csv`
- `reports/framework_compare_4b_20260225_envs_tp2_stable_repeats/framework_compare_repeat_summary.csv`
- `reports/framework_compare_4b_20260225_envs_tp2_stable_repeats/framework_compare_repeat_summary.md`
- `scripts/report/bench_transformers_rollout.py`
- `scripts/report/bench_vllm_rollout.py`
- `scripts/report/bench_sglang_rollout.py`
- `scripts/report/run_framework_compare.py`
- `scripts/report/summarize_framework_compare_repeats.py`

**当前可引用数字（2048/128，stable: iters=8, warmup=2）：**
- Ember single(0): `46.020 tok/s`
- Ember dual(0,1) split=18+18 overlap: `46.520 tok/s`
- vLLM single(tp=2): `47.956 tok/s`
- SGLang single(tp=1): `62.917 tok/s`
- Transformers single(cuda:0): `36.467 tok/s`
- llama.cpp dual(CUDA0/CUDA1): `69.639 tok/s`

**稳定性统计（3 次重复，same setting）：**
- Ember dual(0,1): mean `47.122`, std `0.554`, CV `1.17%`
- vLLM single(tp=2): mean `48.166`, std `0.643`, CV `1.33%`
- SGLang single(tp=1): mean `65.145`, std `1.941`, CV `2.98%`
- llama.cpp dual(CUDA0/CUDA1): mean `71.142`, std `1.350`, CV `1.90%`

---

## 2. Talk 准备里程碑 🎯

> 目标：完成 talk 所需的**最低可行数据集**，让稿子中所有 `XX%` 占位符都有实数填入。

### 2.1 [当前] Prefix Cache 实现 + 实测（P1 Sec 4.4）[~]

这是 talk 前唯一的必做工程任务。

- [~] 实现基础 prefix KV cache 机制（当前完成 benchmark 路径的复用机制；runtime 通用 cache manager 仍待做）
- [x] 构造 shared-prefix workload：长度受控的 shared-prefix + suffix 请求集（100 docs）
- [x] 测量 有 prefix cache vs 无 prefix cache 的 prefill 时间（100 docs）
- [x] 导出 savings vs prefix 长度曲线
- [ ] 将实测数字填入 talk 稿和 P1 Section 4.4

**新增产出（2026-02-25）：**
- `reports/stage1_prefix_cache_4b_20260225_mainline/stage13_prefix_cache_sweep.csv`
- `reports/stage1_prefix_cache_4b_20260225_mainline/stage13_prefix_cache_summary.md`
- `reports/stage1_prefix_cache_4b_20260225_mainline/stage13_p1_input.md`
- `reports/stage1_prefix_cache_4b_4096_20260225_mainline/stage13_prefix_cache_summary.md`

**当前可引用数字：**
- prompt_len=2048, prefix_len=1024: savings `39.342%`, speedup `1.649x`
- prompt_len=4096, prefix_len=1024: savings `16.604%`, speedup `1.199x`

**服务：** Talk 第二部分策略 2 的数据 + P1 论文

### 2.2 Talk 稿定稿

- [ ] 用 1.1 实测数据替换 talk 稿中的 `XX%` 占位符（prefill share）
- [ ] 用 2.1 实测数据替换 prefix cache 节省比例占位符
- [ ] 内部走一遍 60 分钟计时，确认节奏
- [ ] 准备几张关键图表（prefill share 曲线、prefix cache 对比、架构图）

**里程碑：Talk 就绪** 🏁

---

## 3. P1 引擎功能（Talk 之后推进，按依赖顺序）

> 以下功能是 P1 核心实验（多轮累积对比、策略 sweep）的前置依赖。

### 3.1 LoRA Adapter 加载与热更新

- [x] 支持加载 PEFT 格式 LoRA（A/B 矩阵）
- [x] 推理注入（merge 到投影权重）：`W <- W + scale * (B @ A)`
- [x] 热替换（不重载 base model；支持 `replace_existing` 先回滚后应用）
- [x] `ember --check` 支持 `--adapter/--lora-scale`（可直接导出 LoRA 后 logits）
- [x] 权重空间数值校验：`W_after - W_before` 对齐 `B @ A * scale`（误差 ~2e-4）
- [ ] 数值验证：和 HF PEFT 推理结果对齐（atol < 1e-4）
- [x] 导出热更新延迟

**新增产出（2026-02-25）：**
- `benchmarks/lora_hot_update_benchmark.cpp`
- `scripts/report/run_stage1_lora_hot_update.py`
- `scripts/report/run_stage31_lora_numeric_align.py`
- `benchmarks/lora_weight_merge_check.cpp`
- `scripts/report/run_stage31_lora_weight_merge_check.py`
- `reports/stage31_lora_hot_update_4b_20260225_mainline/stage31_summary.md`
- `reports/stage31_lora_hot_update_4b_20260225_mainline_avg/stage31_lora_hot_update.csv`
- `reports/stage31_lora_hot_update_4b_20260225_replace_mainline/stage31_summary.md`
- `reports/stage31_lora_numeric_align_4b_20260225_synth_mainline/stage31_lora_numeric_align.csv`
- `reports/stage31_lora_numeric_align_4b_20260225_synth_bf16/stage31_lora_numeric_align.csv`
- `reports/adapters/qwen3_4b_peft_init_r8_20260225/`（真实 PEFT init adapter，zero-step）
- `reports/adapters/qwen3_4b_peft_perturb_r8_20260225/`（真实 PEFT 非零扰动 adapter）
- `reports/stage31_lora_numeric_align_4b_20260225_peft_init_mainline/stage31_lora_numeric_align.csv`
- `reports/stage31_lora_numeric_align_4b_20260225_peft_perturb_peftref/stage31_lora_numeric_align.csv`
- `reports/stage31_lora_numeric_align_4b_20260225_peft_diag/stage31_diag_summary.csv`
- `reports/stage31_lora_weight_merge_check_4b_20260225_peft_perturb_layer0_mainline/stage31_lora_weight_merge_check.csv`
- `reports/stage31_lora_weight_merge_check_4b_20260225_peft_perturb_layer35_q_mainline/stage31_lora_weight_merge_check.csv`
- `reports/synthetic_lora_qwen3_4b_r8/`（形状匹配的 synthetic adapter，用于路径验证）

**当前可引用数字（Qwen3-4B, 2x3080Ti, split=9+27）：**
- 冷启动首轮（iters=1, warmup=0）：`353.980 ms`
- 稳态（iters=3, warmup=1）：`28.206 ms`
- 热替换稳态（iters=3, warmup=1, replace_existing=1）：`51.538 ms`
- 本次更新矩阵数：增量 merge `144`；热替换（回滚+应用）`288`
- LoRA numeric align（真实 PEFT init adapter, zero-update）：`delta_max_abs_diff=0.00000000`（通过 `1e-4`）
- LoRA numeric align（真实 PEFT 非零扰动 adapter）：`delta_max_abs_diff=0.26039124`（未通过 `1e-4`，该项继续 pending）
- 单模块扰动诊断：`q=0.35316205`, `k=0.24981344`, `v=0.31881905`, `o=0.38927269`（均未通过）
- LoRA 权重空间校验（真实 PEFT 非零扰动）：layer0 `q/k/v/o` 的 `delta_max_abs_diff ≈ 2.43e-4`；layer35 `q_proj=3.03e-4`

**解锁：** 3.3 cache 策略接口中的 UpdateLocality、多轮累积实验

### 3.2 批量多候选生成 + Logprobs

- [x] `generate(prompts, num_candidates, sampling_params)` 支持 N=4/8/16（已实测 smoke：N=4/8/16）
- [x] 支持 stop sequences
- [x] 导出 token-level logprobs
- [x] 数值一致性校验（同 seed 重跑一致性）

**新增产出（2026-02-25）：**
- `benchmarks/multi_candidate_rollout.cpp`
- `scripts/report/run_stage2_multi_candidate.py`
- `reports/stage21_multi_candidate_4b_20260225_smoke/stage21_multi_candidate.csv`
- `reports/stage21_multi_candidate_4b_20260225_smoke/stage21_candidates.jsonl`
- `reports/stage21_multi_candidate_4b_20260225_smoke/stage21_summary.md`
- `reports/stage21_multi_candidate_4b_20260225_mainline/stage21_multi_candidate.csv`（N=8 主线配置）
- `reports/stage21_multi_candidate_4b_20260225_n16_smoke/stage21_multi_candidate.csv`（N=16 验证）
- `reports/stage21_multi_candidate_4b_20260225_stopseq_smoke/stage21_candidates.jsonl`（`finish_reason=stop_seq` 验证）
- `scripts/report/run_stage2_numeric_consistency.py`
- `reports/stage22_numeric_consistency_4b_20260225_mainline/stage22_numeric_consistency.csv`

**当前可引用数字：**
- smoke (128/32, N=4): total_gen_tokens=`128`, total_ms=`1838.459`, gen_tok_s=`69.624`
- smoke (128/32, N=16): total_gen_tokens=`512`, total_ms=`4798.022`, gen_tok_s=`106.711`
- mainline (2048/128, N=8): total_gen_tokens=`1024`, total_ms=`14028.646`, gen_tok_s=`72.994`
- token-level logprobs 已导出到 `stage21_candidates.jsonl`
- numeric consistency: same-seed 重跑 `token_mismatch_candidates=0`, `max_abs_logprob_diff=0.0`

**解锁：** P1 多轮实验（100 prompt × 8 candidates）、P4 Best-of-N 基线

### 3.3 Cache Policy 接口 + 策略实现

- [x] 设计 `CachePolicy` 抽象接口（已落地 `runtime/cache_policy.h` 策略引擎）
- [x] 实现 `Naive`（全失效）
- [x] 实现 `UpdateLocality(N)`（冻结前 N 层）
- [x] 实现 `PeriodicRefresh(k)`（每 k 步全刷新）
- [x] 每种策略的 stats 导出（已导出每轮 `recompute/reuse/full_refresh` 与汇总 `hit/miss/recompute`）

**新增产出（2026-02-25）：**
- `runtime/cache_policy.h`
- `benchmarks/cache_policy_sim.cpp`
- `scripts/report/run_stage33_cache_policy.py`
- `reports/stage33_cache_policy_20260225_mainline/stage33_policy_summary.csv`
- `reports/stage33_cache_policy_20260225_mainline/stage33_policy_per_round.csv`

**当前可引用数字（num_layers=36, rounds=30, freeze_layers=18, k=10）：**
- `naive`: avg_recompute_ratio=`1.000000`
- `update_locality`: avg_recompute_ratio=`0.516667`
- `periodic_refresh`: avg_recompute_ratio=`0.566667`

**解锁：** 所有 P1 核心实验

---

## 4. P1 核心实验（引擎功能就绪后）

### 4.1 多轮累积成本对比（P1 Fig 3 — 论文最重要的图）[~]

- [x] 模拟 10-50 轮 policy update（当前用参数化 locality 模型）
- [x] 每轮 100 prompt × 8 candidates rollout（成本建模）
- [x] 对比 Naive / Prefix-only / UpdateLocality 三种策略的累积 GPU 时间
- [x] 导出每轮时间 + 累积曲线（CSV/MD）
- [ ] 真实训练闭环版本复跑并替换模拟假设（3.1+3.2+3.3 完成后）

**依赖：** 3.1 + 3.2 + 3.3 全部完成

**新增产出（2026-02-25）：**
- `scripts/report/run_stage1_cumulative_profile.py`
- `reports/stage14_cumulative_profile_4b_20260225_mainline/stage14_per_round.csv`
- `reports/stage14_cumulative_profile_4b_20260225_mainline/stage14_summary.md`
- `reports/stage14_cumulative_profile_4b_4096_20260225_mainline/stage14_summary.md`
- `reports/stage14_cumulative_profile_4b_20260225_policy_mainline/stage14_summary.md`
- `reports/stage14_cumulative_profile_4b_4096_20260225_policy_mainline/stage14_summary.md`

**当前可引用数字（30 轮，100 prompts × 8 candidates，2 GPUs）：**
- 2048/128（policy-per-round=update_locality）: Prefix-only 相对 Naive 降 `7.275%`；UpdateLocality 降 `7.489%`
- 4096/64（base-profile-csv + policy-per-round=update_locality）: Prefix-only 相对 Naive 降 `15.538%`；UpdateLocality 降 `21.938%`

### 4.2 Update Locality N Sweep（P1 Fig 4 — 关键 ablation）

- [ ] N = 全冻结 / 75% / 50% / 25% / 全可训练
- [ ] 每个 N：测量 prefill 加速比
- [ ] 结合质量评估（如果训练闭环已通，测 F1；否则用 KV cache L2 error 作代理指标）
- [ ] 输出推荐 N 范围和失败边界

**依赖：** 3.1 + 3.3

**新增产出（2026-02-25，模拟版）：**
- `scripts/report/run_stage1_locality_sweep.py`
- `reports/stage42_locality_sweep_4b_20260225_mainline/stage42_locality_sweep.md`

**当前可引用数字（30 轮，2048/128，periodic_refresh_k=10）：**
- freeze_proxy=50%（recompute_ratio=0.5）: 相对 Naive 降 `6.764%`
- freeze_proxy=75%（recompute_ratio=0.25）: 相对 Naive 降 `10.146%`
- freeze_proxy=100%（recompute_ratio=0.0）: 相对 Naive 降 `13.527%`

### 4.3 策略谱系全面对比（P1 Table 1 — 论文主表）

- [ ] 5 种 cache 策略在信息抽取任务上全面对比
- [ ] 报告：累积 GPU-hours / 最终质量 / prefill 节省率 / cache 内存

**依赖：** 4.1 + 4.2 + UpdatableKV 实现（如果纳入）

### 4.4 UpdatableKV Sweep（决定 P5 是否独立）

- [ ] Sweep LoRA rank r = 8/16/32/64 × refresh interval k = 1/5/10/20/50
- [ ] 测量逐层修正误差（L2 norm）
- [ ] 门控决策：多层误差可控 → P5 独立成文；否则并入 P1 一个 section

**依赖：** 3.1 + 3.3 + UpdatableKV 策略实现

---

## 5. 训练闭环（P1 端到端证明 + P4 核心数据）

### 5.1 验证器

- [ ] 信息抽取验证器（JSON 校验 + schema 校验 + 字段匹配 + scalar reward 聚合）
- [ ] SQL 验证器（SQLite 执行 + 结果集比对）
- [ ] Reward 设计变体（Binary / Weighted / Field-level decomposed）

### 5.2 训练基线

- [ ] SFT 基线（HF + PEFT LoRA SFT）
- [ ] Best-of-N 基线
- [ ] DPO 闭环（主实验）
- [ ] GRPO 对比（次要）

### 5.3 统一后端优势证明

- [ ] 双栈 vs 统一后端显存对比
- [ ] 权重同步开销 vs 原地热更新延迟
- [ ] 端到端 rollout+update 吞吐对比

---

## 6. 延后项（P1 数据锁定后再做）

### 6.1 跨框架对比补全（P2 需要）

- [x] `run_framework_compare.py` 补 vLLM 实测分支
- [x] `run_framework_compare.py` 补 Transformers 实测分支
- [x] SGLang 实测（独立 env）
- **备注：** 当前已具备跨框架基线数据，后续仅在参数统一（batch / TP / cache policy）或新增量化后再刷新。

### 6.2 P2 引擎论文

- [—] 补全 loading time 对比（safetensors vs GGUF vs HF transformers）
- [—] Pipeline 执行时序图
- [—] Kernel 优化描述补完
- **前置：** 1.2 数据已齐，可随时启动，但优先级低于 P1

### 6.3 P3 训练 Kernel 论文

- [—] LoRA forward + backward CUDA kernel 替换
- [—] Fused cross-entropy loss
- [—] AdamW optimizer step（LoRA params only）
- [—] 双栈消除实验深度数据
- **前置：** 需要训练闭环（5.2）完成

### 6.4 P4 信息抽取 Recipe 论文

- [—] API baseline 采集（Claude Sonnet / GPT-4o）
- [—] 小模型能力摸底（Qwen3-1.7B / 4B few-shot F1）
- [—] Reward 设计 ablation
- [—] $/F1 Frontier 图
- **前置：** 需要训练闭环（5.2）完成

### 6.5 P5 UpdatableKV 论文

- [—] 完整 theorem 推导
- [—] Tightness 分析
- **前置：** 4.4 的门控决策通过

### 6.6 开源工程化

- [—] 项目结构新增模块（cache-policy / train / verifier / profiling）
- [—] Profiling 工具集（`ember_profile_rollout`, `ember_cache_sweep`, `ember_memory_budget`）
- [—] 可复现实验脚本 + YAML 配置
- [—] 文档更新

---

## 执行路径总览

```
已完成 ✅
─────────────────────────────
  0. Hard Gates              ✅
  1.1 Rollout 时间分解         ✅  → P1 Fig 2 数据就绪
  1.2 Pipeline Parallel       ✅  → P2 核心数据就绪
  1.3 跨框架对比（llama.cpp）   ✅  → 部分就绪

当前 → Talk 准备 🎯
─────────────────────────────
  2.1 Prefix Cache 实现+实测   ← 你在这里
  2.2 Talk 稿定稿
  ────── Talk 就绪 🏁 ──────

Talk 之后 → P1 引擎功能
─────────────────────────────
  3.1 LoRA 热更新
  3.2 批量候选生成 + Logprobs
  3.3 Cache Policy 接口

P1 核心实验
─────────────────────────────
  4.1 多轮累积对比             → P1 Fig 3（最重要的图）
  4.2 Update Locality Sweep   → P1 Fig 4
  4.3 策略谱系全面对比          → P1 Table 1
  4.4 UpdatableKV Sweep       → P5 门控

训练闭环
─────────────────────────────
  5.1 验证器
  5.2 训练基线（SFT / DPO / GRPO）
  5.3 统一后端优势证明

延后
─────────────────────────────
  6.x 跨框架补全 / P2 / P3 / P4 / P5 / 开源工程化
```

---

## Done Criteria: "P1-ready"

- [ ] P1 Fig 2（prefill share 曲线）：数据就绪 ✅，图待生成
- [ ] P1 Fig 3（多轮累积对比）：待 4.1
- [ ] P1 Fig 4（Update Locality sweep）：待 4.2
- [ ] P1 Fig 5（UpdatableKV ablation）：待 4.4
- [ ] P1 Table 1（策略谱系主表）：待 4.3
- [ ] P1 Table 2（Baseline 对比表）：待 5.2 + 5.3
- [ ] P1 Sec 4.4（Prefix cache 收益）：待 2.1
- [ ] P1 Sec 5.6（权重同步零开销）：待 5.3
- [ ] 所有 `XX%` 占位符替换为实测数字
- [ ] 至少一次完整策略对比 run，命令可复现
