# Ember 教程系列逐篇 Prompt（0-19）

更新时间：2026-02-26

## 1) 每次新 Chat 先贴的统一系统 Prompt

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 N 篇。

## 项目简介
Ember (https://github.com/dongchany/ember) 是一个从零手写的 Qwen3 CUDA 推理引擎，
纯 C++ + CUDA，不依赖 ggml/llama.cpp。支持消费级多 GPU Pipeline Parallel（如双卡 RTX 3080Ti）。
除了推理，Ember 还支持完整的 RL 训练闭环：多候选 Rollout → Verifier/Reward → LoRA 热更新 →
Cache 策略复用，实现了统一后端（推理和训练共享同份权重），相比双栈方案节省 50% 显存。

## 项目 5 层结构
Layer 1: 推理引擎（CUDA kernels, Transformer forward, Pipeline Parallel）
Layer 2: Rollout 能力（多候选、logprobs、stop sequences）
Layer 3: LoRA 热更新 + Cache 策略（UpdateLocality / Prefix / Periodic / Hybrid）
Layer 4: 验证器 + Reward（Extraction / SQL verifier，字段级打分）
Layer 5: 训练闭环（SFT → Best-of-N → DPO → GRPO 可选）+ 统一后端 vs 双栈

## 教程系列大纲
第一阶段：
- #0: 5 分钟跑通 Ember
- #1: 一个 Token 的一生（全局概览）

第二阶段：
- #2: Tensor 与内存布局
- #3: RMSNorm
- #4: RoPE
- #5: Attention + Softmax
- #6: SwiGLU MLP
- #7: Safetensors 加载与权重映射

第三阶段：
- #8: KV Cache
- #9: Pipeline Parallel
- #10: Sampling 全家桶
- #11: 多候选 Rollout
- #12: LoRA 注入与热更新
- #13: Cache 策略

第四阶段：
- #14: Verifier 与 Reward 设计
- #15: SFT 基线
- #16: Best-of-N
- #17: DPO
- #18: 统一后端 vs 双栈
- #19: GRPO（可选）

## 写作硬性要求
1. 目标读者：想了解 LLM 内部原理的开发者，数学基础较弱也能看懂
2. 数学四步法：直觉 → 小例子手算 → 公式 → 对应 CUDA/训练代码
3. 语言：中文为主，术语和代码注释保留英文
4. 必须引用我提供的真实源码与真实报告，不得编造实验数字
5. 每篇开头必须写：源文件路径、前置知识、下一篇链接
6. 每篇结尾自然放 GitHub 链接：https://github.com/dongchany/ember
7. 风格：友好、像学长讲解，不要居高临下
8. 不要只列 bullet；以叙述为主

## 输出质量要求（必须遵守）
- 你只能使用我提供的“完整代码片段”和“完整报告片段”作为事实来源
- 所有结论都要标注来自哪个文件
- 任何数字都要能在报告中定位到
- 如果某结论缺证据，明确写“当前资料不足”

## 数学深度加严（额外要求）
- 在不影响可读性的前提下，尽量给出更详细的数学推导
- 对每个关键公式都解释“它在数值稳定性/并行实现上的意义”
- 允许在附录给出更完整推导（正文保持循序渐进）
```

## 2) 每篇单独 Prompt 模板

使用方式：
- 第一步：先贴上面“统一系统 Prompt”（把 N 改成对应篇号）
- 第二步：再贴下面对应篇号模板
- 第三步：把“代码文件/报告文件”内容完整贴入（不要摘要）
- 第四步：附件上传 `tutorial_01_life_of_a_token.md` 作为风格参考

---

## #0 5 分钟跑通 Ember

```text
请写第 0 篇：5 分钟跑通 Ember（编译、下载模型、生成第一个 token）。

## 本篇必须完整粘贴的代码文件
1. README.md
2. docs/development.md
3. apps/ember_cli/main.cpp
4. scripts/ci/build.sh

## 本篇必须完整粘贴的报告文件
1. reports/stage1_milestone_4b_20260225_mainline/stage1_mainline_ready.md
2. reports/stage1_milestone_4b_20260225_mainline/stage1_summary.md

## 补充要求
- 用一台 11GB 显存卡用户也能照做的口径写
- 明确区分“最小可跑通路径”和“推荐性能路径”
- 给出常见报错排查（模型路径、CUDA 架构、显存不足）
```

---

## #1 一个 Token 的一生（如需重写）

```text
请写第 1 篇：一个 Token 的一生（全局数据流概览）。

## 本篇必须完整粘贴的代码文件
1. apps/ember_cli/main.cpp
2. backends/cuda/cuda_runtime.cpp
3. core/session.h
4. core/sampler.h

## 本篇必须完整粘贴的报告文件
1. reports/stage1_milestone_4b_20260225_mainline/stage1_summary.md
2. reports/stage1_split_profile_4b_20260225_mainline/stage12_summary.md

## 补充要求
- 强调 prefill vs decode 的差异
- 给一张“数据如何在模块间流动”的 ASCII 图
```

---

## #2 Tensor 与内存布局

```text
请写第 2 篇：Tensor 与内存布局。

## 本篇必须完整粘贴的代码文件
1. core/tensor.h
2. core/types.h
3. backends/cuda/cuda_runtime.cpp（涉及 tensor shape/stride/alloc 的相关段落）

## 本篇必须完整粘贴的报告文件
1. reports/stage1_milestone_4b_20260225_mainline/p2_stage_latency_components.csv
2. reports/stage1_split_profile_4b_20260225_mainline/stage12_transfer_vs_compute.csv
3. reports/stage1_split_profile_4b_20260225_mainline/stage12_summary.md

## 补充要求
- 用 2x3x4 的小张量举例，讲清线性地址映射
- 解释为什么布局会影响带宽和 kernel 吞吐
```

---

## #3 RMSNorm

```text
请写第 3 篇：RMSNorm — 最简单的 CUDA Kernel 入门。

## 本篇必须完整粘贴的代码文件
1. backends/cuda/kernels/rmsnorm.cu
2. backends/cuda/kernels/kernels.h（RMSNorm 声明）
3. backends/cuda/cuda_runtime.cpp（RMSNorm 调用处）

## 本篇必须完整粘贴的报告文件
1. reports/stage31_base_operator_spotcheck_4b_20260225_mainline/stage31_base_operator_spotcheck.csv
2. reports/stage1_split_profile_4b_20260225_mainline/stage12_summary.md

## 补充要求
- 从 thread/block 概念开始讲
- 用 4 元素向量手算 RMSNorm
- 明确说明 epsilon 的作用
```

---

## #4 RoPE

```text
请写第 4 篇：RoPE — 旋转位置编码的几何直觉。

## 本篇必须完整粘贴的代码文件
1. backends/cuda/kernels/rope.cu
2. backends/cuda/kernels/kernels.h（RoPE 声明）
3. backends/cuda/cuda_runtime.cpp（RoPE 调用上下文）

## 本篇必须完整粘贴的报告文件
1. reports/stage1_milestone_4b_20260225_mainline/stage1_summary.md
2. reports/stage1_split_profile_4b_20260225_mainline/stage12_summary.md

## 补充要求
- 用二维旋转矩阵先讲直觉，再推广到高维偶奇位
- 给一个 pos=0/1/2 的最小数值例子
```

---

## #5 Attention + Softmax

```text
请写第 5 篇：Attention + Softmax — 从 Q·K^T 到加权求和。

## 本篇必须完整粘贴的代码文件
1. backends/cuda/kernels/attention.cu
2. backends/cuda/kernels/softmax.cu
3. backends/cuda/kernels/kernels.h（attention/softmax 声明）
4. backends/cuda/cuda_runtime.cpp（QKV 与 attention 调用处）

## 本篇必须完整粘贴的报告文件
1. reports/stage1_milestone_4b_20260225_mainline/p2_stage_latency_components.csv
2. reports/stage1_split_profile_4b_20260225_mainline/stage12_summary.md
3. reports/stage1_split_profile_4b_20260225_mainline/stage12_vs_llama.md

## 补充要求
- 手算一个 2-token 的 attention 例子
- 解释为什么要除以 sqrt(d_k)
- 指出数值稳定性（max-trick）在 softmax 中的作用
```

---

## #6 SwiGLU MLP

```text
请写第 6 篇：SwiGLU MLP — 为什么不是简单 ReLU。

## 本篇必须完整粘贴的代码文件
1. backends/cuda/kernels/ops.cu（MLP 相关 kernel）
2. backends/cuda/cuda_runtime.cpp（compute_mlp 与调用链）

## 本篇必须完整粘贴的报告文件
1. reports/stage31_base_operator_spotcheck_4b_20260225_mainline/stage31_base_operator_spotcheck.csv
2. reports/stage1_milestone_4b_20260225_mainline/stage1_summary.md

## 补充要求
- 讲清 gate/up/down 三条支路
- 用小向量演示 SiLU(x)*gate 的非线性效果
```

---

## #7 Safetensors 加载与权重映射

```text
请写第 7 篇：Safetensors 加载与权重映射。

## 本篇必须完整粘贴的代码文件
1. formats/safetensors.cpp
2. formats/safetensors.h
3. backends/cuda/cuda_runtime.cpp（权重绑定/加载相关段落）

## 本篇必须完整粘贴的报告文件
1. reports/stage1_milestone_4b_20260225_mainline/stage1_mainline_ready.md
2. reports/stage31_lora_hot_update_4b_20260225_mainline/stage31_summary.md

## 补充要求
- 重点讲“磁盘 tensor 名称 -> 运行时权重指针”的映射链路
- 提醒读者常见的 shape mismatch 问题
```

---

## #8 KV Cache

```text
请写第 8 篇：KV Cache — 为什么 decode 每次只算一个 token。

## 本篇必须完整粘贴的代码文件
1. core/session.h
2. backends/cuda/cuda_runtime.cpp（KV cache 分配/写入/读取相关段落）

## 本篇必须完整粘贴的报告文件
1. reports/stage1_prefix_cache_4b_20260225_mainline/stage13_prefix_cache_summary.md
2. reports/stage1_prefix_cache_4b_20260225_mainline/stage13_prefix_cache_sweep.csv
3. reports/stage14_cumulative_profile_4b_20260225_mainline/stage14_summary.md

## 补充要求
- 给出 KV cache 显存占用估算公式
- 解释为什么 RL 多轮场景会放大 prefill 成本
```

---

## #9 Pipeline Parallel

```text
请写第 9 篇：Pipeline Parallel — 两张消费卡怎么协作。

## 本篇必须完整粘贴的代码文件
1. backends/cuda/cuda_runtime.cpp（forward_layer 跨卡调度段落）
2. runtime/iruntime.h（DeviceMap）
3. runtime/runtime_setup.h（如涉及设备初始化）

## 本篇必须完整粘贴的报告文件
1. reports/stage1_split_profile_4b_20260225_mainline/stage12_summary.md
2. reports/stage1_split_profile_4b_20260225_mainline/stage12_vs_llama.md
3. reports/stage1_split_profile_4b_20260225_mainline/stage12_bubble_vs_split.csv

## 补充要求
- 讲清 9+27 / 18+18 这类 split 的含义
- 解释 overlap/no-overlap 对吞吐的影响
```

---

## #10 Sampling 全家桶

```text
请写第 10 篇：Sampling 全家桶 — Temperature / Top-K / Top-P 的概率推导。

## 本篇必须完整粘贴的代码文件
1. core/sampler.h
2. docs/sampler_explanation.md
3. apps/ember_cli/main.cpp（采样调用点）

## 本篇必须完整粘贴的报告文件
1. reports/stage21_multi_candidate_4b_20260225_mainline/stage21_summary.md
2. reports/stage22_numeric_consistency_4b_20260225_mainline/stage22_summary.md

## 补充要求
- 用一组 logits 做温度缩放和 top-k/top-p 手算
- 解释“随机性 vs 稳定性”的工程取舍
```

---

## #11 多候选 Rollout

```text
请写第 11 篇：多候选 Rollout — 为 RL 生成 N 条候选。

## 本篇必须完整粘贴的代码文件
1. benchmarks/multi_candidate_rollout.cpp
2. runtime/batch_runtime.h
3. scripts/report/run_stage2_multi_candidate.py
4. scripts/report/run_stage2_numeric_consistency.py

## 本篇必须完整粘贴的报告文件
1. reports/stage21_multi_candidate_4b_20260225_mainline/stage21_summary.md
2. reports/stage21_multi_candidate_4b_20260225_mainline/stage21_multi_candidate.csv
3. reports/stage22_numeric_consistency_4b_20260225_mainline/stage22_numeric_consistency.csv

## 补充要求
- 强调这篇是 Best-of-N / DPO 的前置
- 解释 stop sequences、logprobs 导出和 same-seed 一致性验证
```

---

## #12 LoRA 注入与热更新

```text
请写第 12 篇：LoRA 注入与热更新 — W <- W + scale*(B@A)。

## 本篇必须完整粘贴的代码文件
1. backends/cuda/cuda_runtime.cpp（apply_lora_adapter / rollback / replace 相关段落）
2. docs/lora_adapter_quickstart.md
3. benchmarks/lora_hot_update_benchmark.cpp
4. scripts/report/run_stage1_lora_hot_update.py

## 本篇必须完整粘贴的报告文件
1. reports/stage31_lora_hot_update_4b_20260225_mainline/stage31_summary.md
2. reports/stage31_lora_hot_update_4b_20260225_mainline/stage31_lora_hot_update.csv
3. reports/stage31_lora_weight_merge_check_4b_20260225_peft_perturb_layer0_mainline/stage31_lora_weight_merge_check.csv
4. reports/stage31_lora_delta_profile_4b_20260225_peft_perturb_mainline_v2/stage31_lora_delta_freeze_summary.csv

## 补充要求
- 数学从低秩分解直觉出发，不要默认读者线代很强
- 讲清 merge/rollback/replace 三种路径的差异
```

---

## #13 Cache 策略

```text
请写第 13 篇：Cache 策略 — UpdateLocality / Prefix / Periodic / Hybrid。

## 本篇必须完整粘贴的代码文件
1. runtime/cache_policy.h
2. benchmarks/cache_policy_sim.cpp
3. scripts/report/run_stage33_cache_policy.py
4. scripts/report/run_stage1_prefix_cache.py

## 本篇必须完整粘贴的报告文件
1. reports/stage33_cache_policy_20260225_mainline/stage33_summary.md
2. reports/stage42_locality_sweep_4b_20260225_mainline/stage42_locality_sweep.md
3. reports/stage43_strategy_table_4b_20260225_mainline_v2_refactor/stage43_strategy_table.md
4. reports/stage1_prefix_cache_4b_20260225_mainline/stage13_prefix_cache_summary.md

## 补充要求
- 明确解释“为什么多轮 RL 放大 prefill 成本”
- 给出策略选择建议（何时选 UpdateLocality / Prefix / Hybrid）
```

---

## #14 Verifier 与 Reward 设计

```text
请写第 14 篇：Verifier 与 Reward 设计 — 怎么给模型打分。

## 本篇必须完整粘贴的代码文件
1. scripts/verifier/extraction_verifier.py
2. scripts/verifier/sql_verifier.py
3. scripts/verifier/README.md

## 本篇必须完整粘贴的报告文件
1. reports/stage51_extraction_verifier_smoke_20260225/out/stage51_summary.md
2. reports/stage51_sql_verifier_smoke_20260225/out/stage51_sql_summary.md
3. reports/stage51_extraction_verifier_smoke_20260225/out/stage51_per_sample.csv

## 补充要求
- 数学只要求 precision/recall/F1 级别
- 讲清 binary / weighted / field-level reward 的差异与场景
```

---

## #15 SFT 基线

```text
请写第 15 篇：SFT 基线 — 最简单的监督微调（QLoRA on 11GB）。

## 本篇必须完整粘贴的代码文件
1. scripts/train/run_stage52_sft_min.py
2. scripts/train/common_train.py
3. scripts/train/train_min_lora_adapter.py

## 本篇必须完整粘贴的报告文件
1. reports/stage52_sft_min_4b_20260225_external_zip22_qlora_v1/stage52_sft_summary.md
2. reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_all_dpo_variants_v2/stage52_baseline_compare.md
3. reports/stage52_dataset_validation_external_zip22_v1/stage52_dataset_validation_summary.md

## 补充要求
- 推导 cross-entropy loss（先直觉后公式）
- 明确解释为何 FP16 SFT 在 11GB 上 OOM，而 QLoRA 可行
```

---

## #16 Best-of-N

```text
请写第 16 篇：Best-of-N — 不训练也能提升，pass@k 的数学。

## 本篇必须完整粘贴的代码文件
1. scripts/train/run_stage52_best_of_n_extraction.py
2. scripts/report/run_stage52_baseline_compare.py
3. scripts/verifier/extraction_verifier.py

## 本篇必须完整粘贴的报告文件
1. reports/stage52_best_of_n_4b_base_20260226_external_zip22_n4_100_forcejson/stage52_summary.md
2. reports/stage52_best_of_n_4b_sftqlora_20260226_external_zip22_v1_n4_100_forcejson/stage52_summary.md
3. reports/stage52_best_of_n_4b_dpo_20260226_external_zip22_v4_hardpair_n4_100_forcejson_sample_gpu1/stage52_summary.md
4. reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_all_dpo_variants_v2/stage52_baseline_compare.md

## 补充要求
- 从组合概率角度推导 pass@k 与 N 的关系
- 解释 first vs best 两套指标为何都要看
```

---

## #17 DPO

```text
请写第 17 篇：DPO — 不需要单独训练 reward model 的偏好学习。

## 本篇必须完整粘贴的代码文件
1. scripts/train/run_stage52_dpo_min.py
2. scripts/train/run_stage52_build_dpo_pairs.py
3. scripts/train/run_stage52_build_dpo_pairs_oracle_exact.py
4. scripts/train/common_train.py

## 本篇必须完整粘贴的报告文件
1. reports/stage52_dpo_pairs_4b_20260226_external_zip22_train200_sftn4_margin008_v1/stage52_dpo_pairs_summary.md
2. reports/stage52_dpo_min_4b_20260226_external_zip22_v4_hardpair_train200_len96_gpu1/stage52_dpo_summary.md
3. reports/stage52_baseline_compare_4b_20260226_external_zip22_n1_dpo_v3_v4_v2/stage52_baseline_compare.md
4. reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_all_dpo_variants_v2/stage52_baseline_compare.md

## 补充要求
- 从 Bradley-Terry 出发推导 DPO loss
- 讲清 pair 质量、margin 阈值、reference_mode 对结果的影响
- 不要回避“reward 提升但 pass 指标受限”的现象
```

---

## #18 统一后端 vs 双栈

```text
请写第 18 篇：统一后端 vs 双栈 — 为什么 Ember 把推理和训练放在一起。

## 本篇必须完整粘贴的代码文件
1. benchmarks/rollout_update_loop_benchmark.cpp
2. scripts/report/run_stage53_unified_backend_advantage.py
3. scripts/report/run_stage53_e2e_loop_compare.py
4. docs/ember_task_checklist_v3.md（5.3 节）

## 本篇必须完整粘贴的报告文件
1. reports/stage53_unified_backend_advantage_4b_20260225_mainline_v2/stage53_summary.md
2. reports/stage53_e2e_loop_compare_4b_20260225_mainline_v1/stage53_e2e_compare.md
3. reports/stage31_lora_hot_update_4b_20260225_mainline/stage31_summary.md

## 补充要求
- 必须明确写出关键数字：14.985 GiB vs 7.492 GiB；28 ms vs 312 ms
- 区分“显存优势”“更新延迟优势”“端到端吞吐优势”三条证据链
```

---

## #19（可选）GRPO

```text
请写第 19 篇（可选）：GRPO — 如果 DPO 之后还想走更远。

## 本篇必须完整粘贴的代码文件
1. scripts/train/README.md
2. scripts/train/common_train.py
3. scripts/verifier/extraction_verifier.py
4. scripts/report/run_stage52_baseline_compare.py

## 本篇必须完整粘贴的报告文件
1. reports/stage52_baseline_compare_4b_20260226_external_zip22_n4_all_dpo_variants_v2/stage52_baseline_compare.md
2. reports/stage53_e2e_loop_compare_4b_20260225_mainline_v1/stage53_e2e_compare.md

## 补充要求
- 先明确“当前仓库尚未形成完整 GRPO 主线代码”的边界
- 重点写成“设计路线 + 实现计划 + 实验协议”而不是伪造结果
```

---

## 3) 建议的上下文拼接格式（每篇都用）

按这个顺序组织消息即可：

1. [统一系统 Prompt]
2. [本篇 Prompt]
3. 代码上下文（完整）
   - `### File: <path>`
   - ` ```cpp ...完整内容... ``` ` 或 ` ```python ...完整内容... ``` `
4. 报告上下文（完整）
   - `### Report: <path>`
   - ` ```md ...完整内容... ``` ` 或 ` ```csv ...完整内容... ``` `
5. 风格参考附件：`tutorial_01_life_of_a_token.md`
