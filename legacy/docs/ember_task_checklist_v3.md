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

**论文 / talk 统一引用口径（避免 run 混用）：**
- 主表默认引用：`reports/framework_compare_4b_20260225_envs_tp2_stable_repeats/framework_compare_repeat_summary.csv`
- 主指标写法：`mean ± std`（n=3），并在表注里标注 setting：`prompt=2048, decode=128, Ember split=18+18 overlap, SGLang tp=1, vLLM tp=2`
- `reports/framework_compare_4b_20260225_envs_tp2_mainline_v2/framework_compare.csv` 中的 `llama.cpp=72.313 / SGLang=65.561 / Ember=47.586` 仅作为**单次快照**，不作为主结论
- `reports/framework_compare_4b_20260225_envs_mainline/framework_compare.csv`（SGLang `31.673`）与 `reports/framework_compare_4b_20260225_envs_tp2_mainline/framework_compare.csv`（SGLang `77.107`, prefill `3.454ms`）视为异常 run，正文不直接引用

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
- [~] 数值验证：和 HF PEFT 推理结果对齐（atol < 1e-4，受 base forward 偏差阻塞）
- [x] 导出热更新延迟

**新增产出（2026-02-25）：**
- `benchmarks/lora_hot_update_benchmark.cpp`
- `scripts/report/run_stage1_lora_hot_update.py`
- `scripts/report/run_stage31_lora_numeric_align.py`
- `benchmarks/lora_weight_merge_check.cpp`
- `scripts/report/run_stage31_lora_weight_merge_check.py`
- `scripts/report/run_stage31_lora_delta_profile.py`
- `scripts/report/run_stage31_block_align_profile.py`
- `scripts/report/run_stage31_base_operator_spotcheck.py`
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
- `reports/stage31_lora_delta_profile_4b_20260225_peft_perturb_mainline/stage31_lora_delta_profile.csv`
- `reports/stage31_lora_delta_profile_4b_20260225_peft_perturb_mainline_v2/stage31_lora_delta_profile.csv`
- `reports/stage31_lora_delta_profile_4b_20260225_peft_perturb_mainline_v2/stage31_lora_delta_freeze_summary.csv`
- `reports/stage31_lora_delta_profile_4b_20260225_peft_perturb_mainline_v2/stage31_lora_delta_thresholds.csv`
- `reports/stage31_lora_delta_profile_4b_20260225_peft_init_mainline_v2/stage31_lora_delta_profile.csv`
- `reports/stage31_lora_delta_profile_4b_20260225_peft_init_mainline_v2/stage31_lora_delta_thresholds.csv`
- `reports/stage31_block_align_profile_4b_20260225_peft_perturb_mainline/stage31_block_align_profile.csv`
- `reports/stage31_block_align_profile_4b_20260225_peft_init_mainline/stage31_block_align_profile.csv`
- `reports/stage31_block_align_profile_4b_20260225_peft_perturb_mainline_v2/stage31_block_align_profile.csv`
- `reports/stage31_block_align_profile_4b_20260225_peft_perturb_mainline_v2/stage31_attn_residual_decomp.csv`
- `reports/stage31_block_align_profile_4b_20260225_peft_init_mainline_v2/stage31_attn_residual_decomp.csv`
- `reports/stage31_block_align_profile_4b_20260225_peft_perturb_mainline_v4/stage31_attn_residual_decomp.csv`
- `reports/stage31_block_align_profile_4b_20260225_peft_init_mainline_v4/stage31_attn_residual_decomp.csv`
- `reports/stage31_lora_numeric_align_dtype_sweep_4b_20260225_mainline/stage31_dtype_sweep.csv`
- `reports/stage31_lora_numeric_align_4b_20260225_peft_perturb_peft_forward_sweep/stage31_lora_numeric_align.csv`
- `reports/stage31_lora_numeric_align_4b_20260225_peft_perturb_manual_merge_sweep/stage31_lora_numeric_align.csv`
- `reports/stage31_base_operator_spotcheck_4b_20260225_mainline/stage31_base_operator_spotcheck.csv`
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
- LoRA 权重空间校验（双卡 split=18+18 抽检）：layer18 `q_proj delta_max_abs_diff=2.597e-4`
- LoRA delta 逐层剖析（非零扰动）：`layer_0=0.03125`，`layer_24=0.25`，`layer_32=1.75`，`layer_35=2.000267`（误差随深度放大）
- Base/Lora hidden 对齐逐层剖析（非零扰动，v2）：`base_max(layer_35)=18.96875`，`lora_max(layer_35)=19.21875`，`delta_max(layer_35)=2.000267`
- Delta 阈值穿越层位（v2）：`>=0.1 @ layer_4`，`>=0.25 @ layer_20`，`>=0.5 @ layer_28`，`>=1.0 @ layer_31`
- Freeze 前缀风险摘要（v2，delta_max）：`freeze=18 -> 0.234375`，`freeze=24 -> 0.28125`，`freeze=30 -> 0.5625`
- Zero-step PEFT init adapter 的逐层剖析（v2）：所有层 `delta_max_abs_diff=0.0`，但 `base_max(layer_35)=18.96875` 仍存在，说明 LoRA 路径与 PEFT 语义一致，剩余偏差主要来自 base forward 对齐
- Block 级剖析（layers=31-35, perturb adapter）：最大 `delta` 出现在 `attn_residual`（layer32/33/34 达 `1.75`），而同层 `attn_out/post_attn_norm` 显著更小，指向“偏差主要由上游 residual 路径带入并放大”，而非单点 LoRA merge 公式错误
- Block 级剖析（layers=31-35, init adapter）：全 block `delta_max_abs_diff=0.0`，进一步确认 LoRA 注入与 PEFT 语义一致
- `attn_residual = layer_input + attn_out` 来源分解（v2）：
  - perturb adapter: `max(delta_residual)=1.75`, `max(delta_gap)=0.9609375`
  - init adapter: `max(delta_residual)=0.0`, `max(delta_gap)=0.0`
  - `delta_input_max` 在 layer33/35 与 `delta_residual_max` 等幅（share=1.0），而 `delta_attn_max` 显著更小，说明 LoRA-delta 偏差主来源是 layer input 路径（上游累积）而不是当前层 attn_out
- HF dtype sweep（float16/bfloat16/float32@cpu）：
  - init adapter：`delta_max_abs_diff=0.0`（全部通过）
  - perturb adapter：`delta_max_abs_diff` 仍为 `0.26039124`（float16）, `0.51296234`（bfloat16）, `0.26371694`（float32@cpu）
  - 结论：非零 adapter 的端到端 delta 不对齐并非单纯由 HF dtype 选择造成
- HF LoRA reference path sweep（float16, perturb adapter）：
  - `peft_forward`: `delta_max_abs_diff=0.26039124`
  - `manual_merge`: `delta_max_abs_diff=0.25601959`
  - 结论：两条参考路径一致，偏差并非 PEFT 封装语义导致
- Base operator spotcheck（layer0/1）：
  - layer1: `post_attn_norm_max_abs_diff=4.198868`, `gate_proj_max_abs_diff=6.213398`
  - 但在 `Ember norm input` 下：`gate_proj_max_abs_diff=0.031280`
  - 说明主要偏差来自上游 `norm/residual` 输入路径被后续线性层放大，非 gate/up GEMM 或 LoRA merge 本身

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
- `reports/stage14_cumulative_profile_4b_20260225_policy_mainline_v2/stage14_summary.md`（含 Hybrid）
- `reports/stage14_cumulative_profile_4b_4096_20260225_policy_mainline_v2/stage14_summary.md`（含 Hybrid）

**当前可引用数字（30 轮，100 prompts × 8 candidates，2 GPUs）：**
- 2048/128（policy-per-round=update_locality）: Prefix-only 相对 Naive 降 `7.441%`；UpdateLocality 降 `7.544%`
- 4096/64（base-profile-csv + policy-per-round=update_locality）: Prefix-only 相对 Naive 降 `15.538%`；UpdateLocality 降 `21.938%`

### 4.2 Update Locality N Sweep（P1 Fig 4 — 关键 ablation）[~]

- [x] N = 全冻结 / 75% / 50% / 25% / 全可训练（用 recompute_ratio 代理）
- [x] 每个 N：测量 prefill 加速比（累积 GPU-hours / speedup）
- [x] 结合质量评估（当前用 LoRA delta freeze 风险代理；训练闭环后替换为任务 F1）
- [x] 输出推荐 N 范围和失败边界（基于 quality_threshold）

**依赖：** 3.1 + 3.3

**新增产出（2026-02-25，模拟版 + 质量代理）：**
- `scripts/report/run_stage1_locality_sweep.py`
- `reports/stage42_locality_sweep_4b_20260225_mainline/stage42_locality_sweep.md`
- `reports/stage42_locality_sweep_4b_20260225_mainline_v2/stage42_locality_sweep.md`
- `reports/stage42_locality_sweep_4b_20260225_mainline_v2/stage42_p1_input.md`

**当前可引用数字（30 轮，2048/128，periodic_refresh_k=10，quality_threshold=0.3）：**
- recompute_ratio=0.50（freeze_layers≈18）: 降 `6.764%`，quality_proxy=`0.234375`（可接受）
- recompute_ratio=0.25（freeze_layers≈27）: 降 `10.146%`，quality_proxy=`0.421875`（超阈值）
- 推荐点：recompute_ratio=`0.50`（速度与质量约束平衡）

### 4.3 策略谱系全面对比（P1 Table 1 — 论文主表）[~]

- [x] 5 种 cache 策略（Naive/Prefix/Update/Periodic/Hybrid）模拟汇总对比
- [ ] 报告：累积 GPU-hours / 最终质量 / prefill 节省率 / cache 内存（质量与内存仍待真实闭环补齐）

**依赖：** 4.1 + 4.2 + UpdatableKV 实现（如果纳入）

**新增产出（2026-02-25，模拟汇总版）：**
- `scripts/report/run_stage43_strategy_table.py`
- `reports/stage14_cumulative_profile_4b_20260225_policy_mainline_v2/stage14_summary.csv`
- `reports/stage14_cumulative_profile_4b_4096_20260225_policy_mainline_v2/stage14_summary.csv`
- `reports/stage14_cumulative_profile_4b_20260225_periodic_mainline_v2/stage14_summary.csv`
- `reports/stage14_cumulative_profile_4b_4096_20260225_periodic_mainline_v2/stage14_summary.csv`
- `reports/stage43_strategy_table_4b_20260225_mainline_v1/stage43_strategy_table.md`
- `reports/stage43_strategy_table_4b_20260225_mainline_v1/stage43_p1_input.md`

**当前可引用数字（30 轮，100×8，2 GPUs）：**
- 2048/128：Hybrid 相对 Naive 降 `11.389%`（Update=`7.544%`, Periodic=`6.764%`, Prefix=`7.441%`）
- 4096/64：Hybrid 相对 Naive 降 `29.966%`（Update=`21.938%`, Periodic=`19.668%`, Prefix=`15.538%`）

### 4.4 UpdatableKV Sweep（决定 P5 是否独立）[~]

- [x] Sweep LoRA rank r = 8/16/32/64 × refresh interval k = 1/5/10/20/50（代理版）
- [x] 测量逐层修正误差（当前用 `ΔK` 的低秩近似相对 Fro 误差代理）
- [x] 门控决策（代理）：给出 rank-k 可行区域
- [ ] 真实闭环版本复跑并替换代理假设（训练轮间真实漂移 + 任务质量）

**依赖：** 3.1 + 3.3 + UpdatableKV 策略实现

**新增产出（2026-02-25，代理版）：**
- `scripts/report/run_stage44_updatablekv_sweep.py`
- `reports/stage44_updatablekv_sweep_4b_20260225_peft_perturb_proxy_v1/stage44_summary.md`
- `reports/stage44_updatablekv_sweep_4b_20260225_peft_perturb_proxy_seq1024_step2_v1/stage44_summary.md`
- `reports/stage44_updatablekv_sweep_4b_20260225_kproj_only_proxy_v1/stage44_summary.md`
- `reports/stage44_updatablekv_sweep_4b_20260225_peft_init_proxy_v1/stage44_summary.md`

**当前可引用数字（seq_len=512, quality_threshold=0.3, 代理门控）：**
- `peft_perturb_r8`：rank64 在 `k=1` 可行（proxy p95=`0.2423`）；`k>=5` 不可行
- `peft_perturb_r8`（seq1024, step2）结论一致：rank64 仅 `k=1` 可行（proxy p95=`0.2569`）
- `k_proj_only_r8`：rank64 在 `k=1` 接近阈值但仍超出（proxy p95=`0.3035`）
- `peft_init_r8`：全 0（用于 sanity check）

---

## 5. 训练闭环（P1 端到端证明 + P4 核心数据）

### 5.1 验证器

- [x] 信息抽取验证器（JSON 校验 + schema 校验 + 字段匹配 + scalar reward 聚合）
- [x] SQL 验证器（SQLite 执行 + 结果集比对）
- [x] Reward 设计变体（Binary / Weighted / Field-level decomposed）

**新增产出（2026-02-25）：**
- `scripts/verifier/extraction_verifier.py`
- `reports/stage51_extraction_verifier_smoke_20260225/out/stage51_summary.md`
- `scripts/verifier/sql_verifier.py`
- `reports/stage51_sql_verifier_smoke_20260225/out/stage51_sql_summary.md`

### 5.2 训练基线

- [x] SFT 基线（HF + PEFT LoRA SFT；4B QLoRA 最小环已跑通）
- [x] Best-of-N 基线（extraction，脚本与 smoke 跑通）
- [~] DPO 闭环（最小训练环已打通，主实验待真实数据）
- [ ] GRPO 对比（次要）

**新增产出（2026-02-25）：**
- `scripts/train/build_stage52_synth_extraction_dataset.py`
- `scripts/train/run_stage52_sft_min.py`
- `scripts/train/run_stage52_best_of_n_extraction.py`
- `scripts/train/run_stage52_snapshot_dataset.py`
- `scripts/report/run_stage52_baseline_compare.py`
- `scripts/train/run_stage52_build_dpo_pairs.py`
- `scripts/train/run_stage52_build_synthetic_pairs_from_gold.py`
- `scripts/train/run_stage52_build_dpo_pairs_oracle_exact.py`
- `scripts/train/run_stage52_dpo_min.py`
- `reports/stage52_sft_min_06b_20260225_synth_v4/stage52_sft_summary.md`
- `reports/stage52_best_of_n_smoke_20260225/out/stage52_summary.md`
- `reports/stage52_best_of_n_smoke_20260225/out_short/stage52_summary.md`
- `reports/stage52_dpo_pairs_smoke_20260225/out/stage52_dpo_pairs_summary.md`
- `reports/stage52_dpo_min_smoke_20260225/out/stage52_dpo_summary.md`
- `reports/stage52_synth_dataset_4b_20260225_v1/dataset.jsonl`
- `reports/stage52_synth_pairs_4b_20260225_v1/stage52_synth_pairs_summary.md`
- `reports/stage52_dpo_min_4b_20260225_synth_v1_len128/stage52_dpo_summary.md`
- `reports/stage52_sft_min_4b_20260225_qlora_v1/stage52_sft_summary.md`
- `reports/stage52_best_of_n_4b_base_20260225_synth_v2_small/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260225_synth_v2_small/stage52_summary.json`
- `reports/stage52_synth_dataset_4b_20260226_hardrule_v1/dataset.jsonl`
- `reports/stage52_sft_min_4b_20260226_hardrule_qlora_v2/stage52_sft_summary.md`
- `reports/stage52_best_of_n_4b_base_20260226_hardrule_v1_n1_32/stage52_summary.json`
- `reports/stage52_best_of_n_4b_base_20260226_hardrule_v1_n4_32/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_hardrule_v1_n1_32/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_hardrule_v1_n4_32/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_hardrule_v2_n1_32/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_hardrule_v2_n4_32/stage52_summary.json`
- `reports/stage52_synth_pairs_4b_20260226_hardrule_v1/stage52_synth_pairs_summary.md`
- `reports/stage52_dpo_min_4b_20260226_hardrule_v1_len96/stage52_dpo_summary.md`
- `reports/stage52_best_of_n_4b_dpo_20260226_hardrule_v1_n1_32/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_hardrule_v1_n4_32/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260226_hardrule_v1/stage52_baseline_compare.md`
- `reports/stage52_dataset_validation_external_zip22_v1/stage52_dataset_validation_summary.md`
- `reports/stage52_dataset_validation_external_zip22_optional_v1/stage52_dataset_validation_summary.md`
- `reports/stage52_sft_min_4b_20260225_external_zip22_qlora_v1/stage52_sft_summary.md`
- `reports/stage52_best_of_n_4b_base_20260225_external_zip22_n1_100/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260225_external_zip22_v1_n1_100/stage52_summary.json`
- `reports/stage52_best_of_n_4b_base_20260225_external_zip22_n1_100_greedyjson/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260225_external_zip22_v1_n1_100_greedyjson/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260225_external_zip22_v1/stage52_baseline_compare.md`
- `reports/stage52_dataset_snapshot_external_zip22_20260226_013459/manifest.json`
- `reports/stage52_synth_pairs_4b_20260226_external_zip22_v1/stage52_synth_pairs_summary.md`
- `reports/stage52_dpo_min_4b_20260226_external_zip22_v1_len96/stage52_dpo_summary.md`
- `reports/stage52_best_of_n_4b_base_20260226_external_zip22_n4_40_forcejson/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_external_zip22_v1_n4_40_forcejson/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v1_n4_40_forcejson/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_with_dpo_v1/stage52_baseline_compare.md`
- `reports/stage52_dataset_snapshot_external_zip22_v2_20260226_021128/manifest.json`
- `reports/stage52_best_of_n_4b_base_20260226_external_zip22_n4_100_forcejson/stage52_summary.json`
- `reports/stage52_best_of_n_4b_base_20260226_external_zip22_n1_100_forcejson_sample/stage52_summary.json`
- `reports/stage52_best_of_n_4b_base_20260226_external_zip22_n2_100_forcejson_sample/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_external_zip22_v1_n4_100_forcejson/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_external_zip22_v1_n1_100_forcejson_sample/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_external_zip22_v1_n2_100_forcejson_sample/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v1_n4_100_forcejson/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_100_with_dpo_v1/stage52_baseline_compare.md`
- `reports/stage52_dpo_min_4b_20260226_external_zip22_v2_refcpu_len96/stage52_dpo_summary.md`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v2_refcpu_n4_100_forcejson/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_100_with_dpo_v2/stage52_baseline_compare.md`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_bestofn_curve_v1/stage52_baseline_compare.md`
- `reports/stage52_dpo_min_4b_20260226_external_zip22_v3_tuned_len96/stage52_dpo_summary.md`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v3_tuned_n1_100_forcejson_sample/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v3_tuned_n4_100_forcejson_sample/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v3_tuned_n2_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n1_with_dpo_v3_v2/stage52_baseline_compare.md`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_all_dpo_variants_v1/stage52_baseline_compare.md`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_external_zip22_train200_n4_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_dpo_pairs_4b_20260226_external_zip22_train200_sftn4_margin008_v1/stage52_dpo_pairs_summary.md`
- `reports/stage52_dpo_min_4b_20260226_external_zip22_v4_hardpair_train200_len96_gpu1/stage52_dpo_summary.md`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v4_hardpair_n1_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v4_hardpair_n4_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_all_dpo_variants_v2/stage52_baseline_compare.md`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n1_dpo_v3_v4_v2/stage52_baseline_compare.md`
- `reports/stage52_dpo_pairs_4b_20260226_external_zip22_train400_sftn4_margin008_v1/stage52_dpo_pairs_summary.md`
- `reports/stage52_dpo_min_4b_20260226_external_zip22_v5_hardpair_train400_len96_gpu1/stage52_dpo_summary.md`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v5_hardpair_n1_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v5_hardpair_n2_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v5_hardpair_n4_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n1_dpo_v3_v4_v5_v1/stage52_baseline_compare.md`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_all_dpo_variants_v3/stage52_baseline_compare.md`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_bestofn_curve_with_dpo_v5_v1/stage52_baseline_compare.md`
- `reports/stage52_dpo_pairs_4b_20260226_external_zip22_train400_oracle_exact_sftn4_v1/stage52_dpo_pairs_summary.md`
- `reports/stage52_dpo_min_4b_20260226_external_zip22_v6_oracle_exact_train400_len96_gpu1/stage52_dpo_summary.md`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v6_oracle_exact_n1_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v6_oracle_exact_n4_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_dpo_min_4b_20260226_external_zip22_v7_initv5_oracle_exact_train400_len96_gpu1/stage52_dpo_summary.md`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v7_initv5_oracle_exact_n1_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v7_initv5_oracle_exact_n2_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v7_initv5_oracle_exact_n4_100_forcejson_sample_gpu1/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n1_dpo_v3_v4_v5_v6_v7_v1/stage52_baseline_compare.md`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_all_dpo_variants_v5/stage52_baseline_compare.md`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_bestofn_curve_with_dpo_v5_v7_v1/stage52_baseline_compare.md`
- `reports/stage52_best_of_n_4b_base_20260226_external_zip22_n1_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_external_zip22_v1_n1_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v5_n1_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v7_n1_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n1_schemahint_effect_v1/stage52_baseline_compare.md`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_n1_schemahint_core_v1/stage52_baseline_compare.md`
- `reports/stage52_best_of_n_4b_base_20260226_external_zip22_n2_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_base_20260226_external_zip22_n4_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_external_zip22_v1_n2_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_sftqlora_20260226_external_zip22_v1_n4_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v5_n2_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v5_n4_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v7_n2_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v7_n4_100_forcejson_schemahint_sample_gpu1/stage52_summary.json`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_base_sft_schemahint_curve_v1/stage52_baseline_compare.md`
- `reports/stage52_baseline_compare_4b_20260226_external_zip22_schemahint_full_curve_v1/stage52_baseline_compare.md`

**备注：**
- SFT 最小基线（Qwen3-0.6B）已跑通：`max_steps=2`, `max_length=64`, `training_loss=3.5876`。
- Qwen3-4B FP16 SFT 在 11GB 显存卡上 OOM；已通过 QLoRA（4bit）路径跑通最小环：`max_steps=1`, `max_length=64`, `training_loss=4.0052`。
- 当前 synthetic extraction 数据难度偏低，4B base 与 4B+SFT(QLoRA) 在 `max_samples=8, best-of-4` 下指标均为 `1.0`，暂不具区分度；后续需更难数据或更严格 verifier 才能形成有效 baseline 对比。
- 已引入 `hard_rule` 难度（active 过滤 + score 排序 + 日期 tie-break）后，4B base 在 `max_samples=32` 上降到 `pass@1=0.84375`（不再饱和），`best-of-4` 升至 `pass@4=0.875`。
- 新的 QLoRA 训练（v2：`max_steps=12`, `max_length=128`, `max_train_samples=32`）得到 `training_loss=2.2477`，并将 `pass@1` 提升到 `0.875`（相对 base `+3.125pp`）；该配置下 `N=4` 未进一步提升（仍 `0.875`）。
- DPO 最小环（hard_rule synthetic pairs，`max_steps=12`, `max_length=96`, `reference_mode=none`）已复跑并成功收敛训练日志，但在同口径评估下暂未超过 base（`N=1: 0.84375`, `N=4: 0.875` 与 base 持平），后续需提升 pair 难度/质量与 reference 配置。
- 外部数据（`files(22).zip`, 400/100/100）验收：原始 schema 下 `missing_required=120, type_mismatch=0`；采用 optional-fields schema（`required=[]`）后 validator 全通过（suspicious=0, cross-split overlap=0）。
- 外部数据快照已冻结并记录校验和（`manifest.json`），用于后续论文/talk 回溯复现。
- 已补充通用快照脚本 `run_stage52_snapshot_dataset.py`，支持 train/val/test/schema 一键归档（含 SHA256 + row count）。
- 外部数据基线（test=100）显示解码策略影响显著：`sample N=1` 下 base/SFT 都接近 0；启用 `greedy + force-json-output` 后，base `pass@1=0.02, mean_reward=0.1797`，SFT `pass@1=0.03, mean_reward=0.3619`，SFT 相对 base 有明显提升。
- 外部数据 `N=4 + force-json`（quick, test=40）：
  - base: `mean_reward_best=0.1923`, `pass@N=0.0000`
  - SFT: `mean_reward_best=0.5006`, `pass@N=0.0250`
  - DPO(v1, reference=none): `mean_reward_best=0.1923`, `pass@N=0.0000`
  - 结论：当前配置下 SFT 有效，DPO 仍未带来增益（需改 pair 质量或 reference 配置）。
- 外部数据 `N=4 + force-json`（stable, test=100）：
  - base: `mean_reward_best=0.1997`, `pass@N=0.0200`
  - SFT: `mean_reward_best=0.4680`, `pass@N=0.0300`
  - DPO(v1, reference=none): `mean_reward_best=0.1997`, `pass@N=0.0200`
  - 结论：SFT 相对 base 有稳定提升；DPO(v1) 与 base 持平，未体现增益。
- DPO(v2, `reference_mode=cpu`, steps=12, len=96) 复跑外部数据同口径评估（N=4, test=100）：
  - `mean_reward_best=0.1997`, `pass@N=0.0200`（与 base、DPO(v1) 持平）
  - 结论：当前 pair 构造与训练预算下，是否启用 reference 尚未带来可见收益；下一步优先提升 pair 质量/难度，而不是继续堆同配置训练步数。
- Best-of-N 曲线（外部数据，sample + force-json, test=100）已补齐：
  - base: `N=1/2/4 -> mean_reward_best 0.1815 / 0.1915 / 0.1997`（平缓上升，`pass@N` 固定 0.02）
  - SFT: `N=1/2/4 -> mean_reward_best 0.3268 / 0.3952 / 0.4680`（显著上升，`pass@N` 0.01 -> 0.02 -> 0.03）
  - 结论：SFT 在 best-of-n 维度有明显可挖掘余量，适合用于“pass@k 转 pass@1”叙事支撑。
- DPO(v3, tuned: `steps=80`, `lr=5e-5`, `reference=none`)：
  - N=1: `mean_reward_best=0.2044`, `pass@1=0.01`
  - N=2: `mean_reward_best=0.2058`, `pass@N=0.01`（GPU1 复跑）
  - N=4: `mean_reward_best=0.2215`, `pass@N=0.02`
  - 对比：`mean_reward` 高于 base（0.1997），但 `pass@N` 未超 base；说明“字段级部分正确率”改善，但“全字段完全正确”尚未突破。
- DPO(v4, hard-pair mined from SFT train candidates)：
  - 先在 external train(200) 上跑 SFT `N=4` 候选，再用 `min_margin=0.08` 构造 DPO pairs（`141` 对，avg margin=`0.4037`）
  - 训练：`steps=120`, `lr=2e-5`, `reference=none`, `max_length=96`（GPU1）
  - N=1(test100): `mean_reward_best=0.2610`, `pass@1=0.02`
  - N=4(test100): `mean_reward_best=0.2807`, `pass@N=0.02`
  - 结论：hard-pair 显著提升字段级 reward（较 v3/base 均提升），并保持 `pass@1` 与 base 持平；目前仍未超过 `pass@N` 的绝对上限。
- DPO(v5, hard-pair mined from train400 SFT candidates)：
  - 在 external train(400) 上汇总 SFT `N=4` 候选，`min_margin=0.08` 构造 DPO pairs（`271` 对，avg margin=`0.3923`）
  - 训练：`steps=180`, `lr=2e-5`, `reference=none`, `max_length=96`（GPU1）
  - N=1(test100): `mean_reward_best=0.3501`, `pass@1=0.02`
  - N=2(test100): `mean_reward_best=0.3788`, `pass@N=0.02`
  - N=4(test100): `mean_reward_best=0.3873`, `pass@N=0.02`
  - 结论：在 `pass@1/pass@N` 不变的情况下，字段级 reward 再次显著提升（相对 v4 +0.0891 / +0.1067），说明 pair 质量/覆盖度提升有效。
- DPO(v6, oracle-exact from base model + hard negative from SFT candidates)：
  - `chosen=gold`，`rejected=该样本最高分 non-exact 候选`，train400 构造 `399` 对，avg margin=`0.5750`
  - 训练：`steps=240`, `lr=1e-5`, `reference=none`, `max_length=96`（GPU1）
  - N=1(test100): `mean_reward_best=0.2022`, `pass@1=0.02`
  - N=4(test100): `mean_reward_best=0.2437`, `pass@N=0.02`
  - 结论：纯 oracle-exact 路线在该设置下不优（reward 明显低于 v5），不建议单独采用。
- DPO(v7, warm-start from v5 + oracle-exact continuation)：
  - 训练实现新增 `run_stage52_dpo_min.py --init-adapter`（支持从已有 LoRA adapter 继续 DPO）
  - 用 v5 adapter 热启动，在 v6 的 `399` 对 oracle-exact pairs 上续训：`steps=120`, `lr=5e-6`, `reference=none`
  - N=1(test100): `mean_reward_best=0.3919`, `pass@1=0.02`
  - N=2(test100): `mean_reward_best=0.4068`, `pass@N=0.02`
  - N=4(test100): `mean_reward_best=0.4196`, `pass@N=0.02`
  - 结论：v7 相对 v5 在字段级 reward 再提升（N=1 `+0.0418`，N=4 `+0.0323`），但 `pass@1/pass@N` 仍未突破 0.02。
- 评估契约修正（schema-key-hint）：
  - `run_stage52_best_of_n_extraction.py` 新增 `--schema-key-hint`，在 `force-json-output` 下把 schema 字段名契约显式写入 prompt（约束不要输出同义字段名）
  - N=1(test100) 在同模型/同解码下出现显著提升：
    - base: `pass@1 0.02 -> 0.23`, `mean_reward 0.1815 -> 0.6950`
    - sft: `pass@1 0.01 -> 0.32`, `mean_reward 0.3268 -> 0.6960`
    - dpo v5: `pass@1 0.02 -> 0.26`, `mean_reward 0.3501 -> 0.7124`
    - dpo v7: `pass@1 0.02 -> 0.27`, `mean_reward 0.3919 -> 0.7150`
  - N=1/2/4 曲线（schemahint, test100）已补齐：
    - base: `pass@N 0.23 -> 0.25 -> 0.28`, `reward_best 0.6950 -> 0.7153 -> 0.7281`
    - sft: `pass@N 0.32 -> 0.40 -> 0.51`, `reward_best 0.6960 -> 0.7481 -> 0.8026`
    - dpo v5: `pass@N 0.26 -> 0.29 -> 0.34`, `reward_best 0.7124 -> 0.7239 -> 0.7441`
    - dpo v7: `pass@N 0.27 -> 0.30 -> 0.31`, `reward_best 0.7150 -> 0.7277 -> 0.7405`
  - 结论：此前 `pass` 长期卡在 `0.02` 的主要原因是“输出键名与评估 schema 键名不一致”，而非纯模型能力瓶颈；在 schemahint 口径下，当前最优方案是 SFT（pass 与 reward 均最高），DPO(v5/v7)在 reward 上提升但未超过 SFT。
- 当前模型在简单 extraction 上候选高度一致（Best-of-N margin≈0），已引入基于 gold 扰动的 synthetic pair 生成，保障 DPO 训练数据可用。
- `run_stage52_dpo_min.py` 当前默认 `reference_mode=none`（DPO-lite）；完整 DPO 可切到 `cpu/same_device` reference 模式。
- 11GB 显存卡下 DPO 训练建议 `max_length<=128`；`>=192` 容易在 vocab log-softmax 阶段 OOM。

### 5.3 统一后端优势证明

- [x] 双栈 vs 统一后端显存对比（model-only 视角）
- [x] 权重同步开销 vs 原地热更新延迟（transfer estimate + measured hot-update）
- [~] 端到端 rollout+update 吞吐对比（unified 闭环实测已完成；dual-stack 侧仍为 sync-sim）

**新增产出（2026-02-25）：**
- `benchmarks/rollout_update_loop_benchmark.cpp`
- `scripts/report/run_stage53_e2e_loop_compare.py`
- `scripts/report/run_stage53_unified_backend_advantage.py`
- `reports/stage53_unified_backend_advantage_4b_20260225_mainline_v1/stage53_summary.md`
- `reports/stage53_unified_backend_advantage_4b_20260225_mainline_v2/stage53_summary.md`
- `reports/stage53_e2e_loop_compare_4b_20260225_mainline_v1/stage53_e2e_compare.md`

**当前可引用数字（Qwen3-4B, 30 轮）：**
- 模型权重 footprint：dual-stack `14.985 GiB` vs unified `7.492 GiB`（节省 `50%`）
- 全量权重同步估算：`312.186 ms/round`（30 轮 `9365.592 ms`）
- 实测原地热更新：`28.206 ms/round`（30 轮 `846.180 ms`）
- E2E 吞吐估算（rollout-heavy: 100×8×128, rollout tok/s=47.586）：unified vs dual-fullsync `1.000132x`（几乎无差异，说明该配置下同步并非吞吐主瓶颈）
- E2E 闭环实测（512/64, candidates=4, rounds=6,warmup=2）：
  - `unified_apply`: `update_ms_ext_avg=42.666`, `e2e_tok_s=84.720`
  - `dual_fullsync_sim`: `round_ms_avg=3301.325`, `e2e_tok_s=77.545`
  - `dual_lora_sync_sim`: `round_ms_avg=2982.843`, `e2e_tok_s=85.824`
  - 结论：unified vs dual_fullsync(sim) `1.0925x`；vs dual_lora_sync(sim) `0.9871x`

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
- [ ] P1 Fig 3（多轮累积对比）：模拟版已就绪，待真实训练闭环复跑
- [ ] P1 Fig 4（Update Locality sweep）：模拟+质量代理版已就绪，待闭环质量替换
- [ ] P1 Fig 5（UpdatableKV ablation）：代理版已就绪，待真实闭环复跑
- [ ] P1 Table 1（策略谱系主表）：模拟汇总版已就绪，待真实质量/内存列补齐
- [ ] P1 Table 2（Baseline 对比表）：待 5.2 + 5.3
- [ ] P1 Sec 4.4（Prefix cache 收益）：待 2.1
- [ ] P1 Sec 5.6（权重同步零开销）：待 5.3
- [ ] 所有 `XX%` 占位符替换为实测数字
- [ ] 至少一次完整策略对比 run，命令可复现
