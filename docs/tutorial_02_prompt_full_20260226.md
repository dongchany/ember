# Tutorial #2 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 2 篇。

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

---

## 1) 写作任务

```text
请写第 2 篇：Tensor 与内存布局。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 用 2x3x4 的小张量举例，讲清线性地址映射
- 解释为什么布局会影响带宽和 kernel 吞吐
- 结合报告里的 p2p/compute 拆分解释“访存 vs 计算”
```

---

## 2) 代码上下文（完整/相关段落）

### File: core/tensor.h

````h
#pragma once

#include "types.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <numeric>
#include <string>
#include <sstream>

namespace ember {

// 轻量级 Tensor 结构
// 不管理内存生命周期，只是一个视图
struct Tensor {
    std::vector<int64_t> shape;
    DType dtype = DType::F32;
    void* data = nullptr;
    int device_id = DEVICE_CPU;  // -1 = CPU, 0+ = GPU
    
    // 默认构造
    Tensor() = default;
    
    // 完整构造
    Tensor(std::vector<int64_t> shape_, DType dtype_, void* data_, int device_ = DEVICE_CPU)
        : shape(std::move(shape_)), dtype(dtype_), data(data_), device_id(device_) {}
    
    // 元素数量
    size_t numel() const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 
                               int64_t(1), std::multiplies<int64_t>());
    }
    
    // 字节大小
    size_t size_bytes() const {
        return numel() * dtype_size(dtype);
    }
    
    // 维度数
    size_t ndim() const { return shape.size(); }
    
    // 是否为空
    bool empty() const { return data == nullptr || numel() == 0; }
    
    // 是否在 CPU
    bool is_cpu() const { return device_id == DEVICE_CPU; }
    
    // 是否在 GPU
    bool is_cuda() const { return device_id >= 0; }
    
    // 获取指定维度大小
    int64_t dim(int i) const {
        if (i < 0) i += static_cast<int>(shape.size());
        assert(i >= 0 && i < static_cast<int>(shape.size()));
        return shape[i];
    }
    
    // 形状字符串
    std::string shape_str() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << shape[i];
        }
        oss << "]";
        return oss.str();
    }
    
    // 完整信息字符串
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Tensor(" << shape_str() << ", " << dtype_name(dtype);
        if (is_cuda()) {
            oss << ", cuda:" << device_id;
        } else {
            oss << ", cpu";
        }
        oss << ")";
        return oss.str();
    }
    
    // 类型转换访问（仅 CPU）
    template<typename T>
    T* data_ptr() {
        assert(is_cpu() && "data_ptr() only works for CPU tensors");
        return static_cast<T*>(data);
    }
    
    template<typename T>
    const T* data_ptr() const {
        assert(is_cpu() && "data_ptr() only works for CPU tensors");
        return static_cast<const T*>(data);
    }
};

// 计算 strides（row-major）
inline std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;
    
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// 检查两个形状是否匹配
inline bool shapes_match(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    return a == b;
}

// 广播形状计算（简化版，只支持末尾对齐）
inline bool shapes_broadcastable(const std::vector<int64_t>& a, 
                                  const std::vector<int64_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t max_dim = std::max(na, nb);
    
    for (size_t i = 0; i < max_dim; ++i) {
        int64_t da = (i < na) ? a[na - 1 - i] : 1;
        int64_t db = (i < nb) ? b[nb - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) {
            return false;
        }
    }
    return true;
}

}  // namespace ember

````

### File: core/types.h

````h
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace ember {

// 数据类型枚举
enum class DType : uint8_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    INT8 = 3,
    INT4 = 4,
    UNKNOWN = 255
};

// 获取数据类型的字节大小
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::INT8: return 1;
        case DType::INT4: return 1;  // 实际是 0.5，但按 1 处理
        default: return 0;
    }
}

// 数据类型名称
inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::F32:  return "f32";
        case DType::F16:  return "f16";
        case DType::BF16: return "bf16";
        case DType::INT8: return "int8";
        case DType::INT4: return "int4";
        default: return "unknown";
    }
}

// 从字符串解析数据类型
inline DType dtype_from_string(const std::string& s) {
    if (s == "float32" || s == "f32" || s == "F32") return DType::F32;
    if (s == "float16" || s == "f16" || s == "F16") return DType::F16;
    if (s == "bfloat16" || s == "bf16" || s == "BF16") return DType::BF16;
    if (s == "int8" || s == "INT8") return DType::INT8;
    if (s == "int4" || s == "INT4") return DType::INT4;
    return DType::UNKNOWN;
}

// 设备常量
constexpr int DEVICE_CPU = -1;

}  // namespace ember

````

### File: backends/cuda/cuda_runtime.cpp (attention strides)

````cpp
    int seq_k = start_pos + seq_len;  // KV cache 中的有效长度
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::attention_bf16(
                static_cast<__nv_bfloat16*>(act.attn_out),
                static_cast<const __nv_bfloat16*>(act.q_proj_out),
                static_cast<const __nv_bfloat16*>(layer_kv.key_cache.data),
                static_cast<const __nv_bfloat16*>(layer_kv.value_cache.data),
                static_cast<float*>(act.attn_scores),
                static_cast<__nv_bfloat16*>(act.attn_probs),
                batch_size, seq_len, seq_k, max_seq, start_pos,
                num_heads, num_kv_heads, head_dim,
                scale, cublas.get(), stream
            );
        } else {
            kernels::attention_f16(
                static_cast<half*>(act.attn_out),
                static_cast<const half*>(act.q_proj_out),
                static_cast<const half*>(layer_kv.key_cache.data),
                static_cast<const half*>(layer_kv.value_cache.data),
                static_cast<float*>(act.attn_scores),
                static_cast<half*>(act.attn_probs),
                batch_size, seq_len, seq_k, max_seq, start_pos,
                num_heads, num_kv_heads, head_dim,
                scale, cublas.get(), stream
            );
        }
    } else {
        const size_t attn_out_batch_stride =
            static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) * static_cast<size_t>(head_dim);
        const size_t cache_batch_stride =
            static_cast<size_t>(num_kv_heads) * static_cast<size_t>(max_seq) * static_cast<size_t>(head_dim);
        const size_t score_batch_stride =
            static_cast<size_t>(num_heads) * act.attn_q_max * act.attn_k_max;

        for (int b = 0; b < batch_size; ++b) {
            const int sp = start_pos_by_batch[b];

            const size_t attn_off = static_cast<size_t>(b) * attn_out_batch_stride;
            const size_t q_off =
                static_cast<size_t>(b) * static_cast<size_t>(seq_len) * static_cast<size_t>(num_heads) *
                static_cast<size_t>(head_dim);
            const size_t cache_off = static_cast<size_t>(b) * cache_batch_stride;
            const size_t score_off = static_cast<size_t>(b) * score_batch_stride;

            if (sp < 0) {
                CUDA_CHECK(cudaMemsetAsync(
                    static_cast<char*>(act.attn_out) + attn_off * elem_size,
                    0,
                    attn_out_batch_stride * elem_size,
                    stream));
                continue;
            }

            const int seq_k_b = sp + seq_len;
            if (seq_k_b <= 0) {
                CUDA_CHECK(cudaMemsetAsync(
                    static_cast<char*>(act.attn_out) + attn_off * elem_size,
                    0,
                    attn_out_batch_stride * elem_size,
                    stream));
                continue;
            }

            if (compute_dtype == DType::BF16) {
                auto* out_ptr = static_cast<__nv_bfloat16*>(act.attn_out) + attn_off;
                auto* q_ptr = static_cast<const __nv_bfloat16*>(act.q_proj_out) + q_off;
                auto* k_cache_ptr = static_cast<const __nv_bfloat16*>(layer_kv.key_cache.data) + cache_off;
                auto* v_cache_ptr = static_cast<const __nv_bfloat16*>(layer_kv.value_cache.data) + cache_off;
                auto* scores_ptr = static_cast<float*>(act.attn_scores) + score_off;
                auto* probs_ptr = static_cast<__nv_bfloat16*>(act.attn_probs) + score_off;

                kernels::attention_bf16(
                    out_ptr,
                    q_ptr,
                    k_cache_ptr,
                    v_cache_ptr,
                    scores_ptr,
                    probs_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    seq_k_b,
                    max_seq,
                    sp,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    cublas.get(),
                    stream
                );
            } else {
                auto* out_ptr = static_cast<half*>(act.attn_out) + attn_off;
                auto* q_ptr = static_cast<const half*>(act.q_proj_out) + q_off;
                auto* k_cache_ptr = static_cast<const half*>(layer_kv.key_cache.data) + cache_off;
                auto* v_cache_ptr = static_cast<const half*>(layer_kv.value_cache.data) + cache_off;
                auto* scores_ptr = static_cast<float*>(act.attn_scores) + score_off;
                auto* probs_ptr = static_cast<half*>(act.attn_probs) + score_off;

                kernels::attention_f16(
                    out_ptr,
                    q_ptr,
                    k_cache_ptr,
                    v_cache_ptr,
                    scores_ptr,
                    probs_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    seq_k_b,
                    max_seq,
                    sp,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale,
                    cublas.get(),
                    stream
                );
            }

````

### File: backends/cuda/cuda_runtime.cpp (MLP strided GEMM strides)

````cpp
            static_cast<const half*>(act.hidden_states),
            static_cast<const half*>(layer.post_attention_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    }
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.post_norm_end, stream));
        CUDA_CHECK(cudaEventRecord(sev.ffn_start, stream));
    }

    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                     "layer_" + std::to_string(layer_idx) + "_post_attn_norm",
                                     device_id, act.norm_out,
                                     seq_len, hidden_size, compute_dtype, stream);
        if (err) return err;
    }
    
    // =====================================================================
    // MLP: SwiGLU
    // gate = SiLU(norm_out @ W_gate)
    // up = norm_out @ W_up
    // down = (gate * up) @ W_down
    // =====================================================================
    
    if (seq_len == 1 && layer.gate_up_proj_packed && act.mlp_gate_up_packed) {
        const int64_t stride_a = static_cast<int64_t>(intermediate_size) * hidden_size;
        const int64_t stride_c = static_cast<int64_t>(intermediate_size) *
                                 static_cast<int64_t>(act.batch_size * act.max_seq_len);
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate_size, M, hidden_size,
            &alpha_one,
            layer.gate_proj_weight, cuda_dtype, hidden_size, stride_a,
            act.norm_out, cuda_dtype, hidden_size, 0,
            &beta_zero,
            act.mlp_gate, cuda_dtype, intermediate_size, stride_c,
            2,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    } else {
        // Gate projection: [M, hidden] @ [intermediate, hidden]^T = [M, intermediate]
        CUBLAS_CHECK(cublasGemmEx(
            cublas.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate_size, M, hidden_size,
            &alpha_one,
            layer.gate_proj_weight, cuda_dtype, hidden_size,
            act.norm_out, cuda_dtype, hidden_size,
            &beta_zero,
            act.mlp_gate, cuda_dtype, intermediate_size,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));

        // Up projection
        CUBLAS_CHECK(cublasGemmEx(
            cublas.get(),
            CUBLAS_OP_T, CUBLAS_OP_N,
            intermediate_size, M, hidden_size,
            &alpha_one,
            layer.up_proj_weight, cuda_dtype, hidden_size,
            act.norm_out, cuda_dtype, hidden_size,
            &beta_zero,
            act.mlp_up, cuda_dtype, intermediate_size,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        ));
    }

    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                  "layer_" + std::to_string(layer_idx) + "_mlp_gate_pre",
                                  device_id, act.mlp_gate,
                                  seq_len, intermediate_size, compute_dtype, stream);
        if (err) return err;
    }
    
    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                  "layer_" + std::to_string(layer_idx) + "_mlp_up",
                                  device_id, act.mlp_up,
                                  seq_len, intermediate_size, compute_dtype, stream);
        if (err) return err;
    }
    
    const bool dump_mlp_intermediates =
        session.runtime_config().check_correctness && session.runtime_config().dump_layer == layer_idx;
    if (dump_mlp_intermediates) {
        // Debug path: keep intermediate tensors materialized for compare_hidden.py.
        if (compute_dtype == DType::BF16) {
            kernels::silu_bf16(
                static_cast<__nv_bfloat16*>(act.mlp_gate),
                static_cast<const __nv_bfloat16*>(act.mlp_gate),
                batch_size * seq_len * intermediate_size,
                stream
            );
        } else {
            kernels::silu_f16(
                static_cast<half*>(act.mlp_gate),
                static_cast<const half*>(act.mlp_gate),
                batch_size * seq_len * intermediate_size,
                stream
            );
        }

        Error err = dump_last_row(session.runtime_config().dump_dir,
                                  "layer_" + std::to_string(layer_idx) + "_mlp_gate_act",
                                  device_id, act.mlp_gate,
                                  seq_len, intermediate_size, compute_dtype, stream);
        if (err) return err;

        if (compute_dtype == DType::BF16) {

````

---

## 3) 报告上下文（完整）

### Report: reports/stage1_milestone_4b_20260225_mainline/p2_stage_latency_components.csv

````csv
run_id,phase,mode,prompt_len,decode_steps,wall_ms,embedding_ms,rmsnorm_ms,attention_ms,ffn_ms,p2p_ms,memcpy_h2d_ms,memcpy_d2h_ms,sampling_ms,lm_head_ms,profile_total_ms
1,prefill,no_overlap,512,64,88.663,0.020,0.801,42.313,41.721,0.033,0.000,0.000,0.000,0.000,88.212
1,decode_per_token,no_overlap,512,64,16.593,0.016,0.438,7.005,7.250,0.016,0.028,0.271,0.144,0.910,16.415
2,prefill,overlap,512,64,88.055,0.019,0.766,42.045,41.643,1.304,0.000,0.000,0.000,0.000,87.565
2,decode_per_token,overlap,512,64,16.610,0.018,0.441,7.070,7.230,0.014,0.025,0.266,0.144,0.910,16.436
3,prefill,no_overlap,512,128,88.459,0.019,0.784,42.161,41.668,0.026,0.000,0.000,0.000,0.000,87.938
3,decode_per_token,no_overlap,512,128,16.664,0.015,0.435,6.849,7.046,0.015,0.222,0.270,0.146,0.910,16.283
4,prefill,overlap,512,128,88.449,0.025,0.793,42.366,41.200,1.320,0.000,0.000,0.000,0.000,87.947
4,decode_per_token,overlap,512,128,16.684,0.016,0.449,6.956,7.067,0.017,0.141,0.284,0.146,0.910,16.384
5,prefill,no_overlap,512,256,88.575,0.019,0.782,42.572,41.386,0.029,0.000,0.000,0.000,0.000,88.052
5,decode_per_token,no_overlap,512,256,16.666,0.023,0.443,7.094,7.047,0.016,0.077,0.270,0.145,0.910,16.432
6,prefill,overlap,512,256,91.693,0.016,0.861,42.724,41.720,1.322,0.000,0.000,0.000,0.000,89.889
6,decode_per_token,overlap,512,256,16.691,0.018,0.443,7.179,7.125,0.016,0.050,0.273,0.145,0.910,16.487
7,prefill,no_overlap,1024,64,187.029,0.035,1.605,93.428,85.208,0.057,0.000,0.000,0.000,0.000,186.566
7,decode_per_token,no_overlap,1024,64,17.171,0.016,0.458,7.777,7.085,0.017,0.013,0.267,0.145,0.910,17.009
8,prefill,overlap,1024,64,185.929,0.039,1.609,93.892,85.406,2.591,0.000,0.000,0.000,0.000,185.487
8,decode_per_token,overlap,1024,64,17.217,0.015,0.456,7.792,7.115,0.017,0.019,0.265,0.144,0.910,17.049
9,prefill,no_overlap,1024,128,189.054,0.034,1.659,94.190,85.099,0.061,0.000,0.000,0.000,0.000,188.397
9,decode_per_token,no_overlap,1024,128,17.173,0.017,0.453,7.770,7.082,0.019,0.020,0.269,0.145,0.911,17.003
10,prefill,overlap,1024,128,190.202,0.039,1.705,94.700,87.704,2.731,0.000,0.000,0.000,0.000,189.632
10,decode_per_token,overlap,1024,128,17.192,0.016,0.464,7.762,7.103,0.018,0.021,0.270,0.144,0.910,17.024
11,prefill,no_overlap,1024,256,190.418,0.038,1.658,94.279,86.114,0.058,0.000,0.000,0.000,0.000,189.963
11,decode_per_token,no_overlap,1024,256,17.172,0.015,0.457,7.781,7.087,0.017,0.016,0.267,0.144,0.910,17.008
12,prefill,overlap,1024,256,193.988,0.041,1.687,97.174,88.994,2.607,0.000,0.000,0.000,0.000,193.523
12,decode_per_token,overlap,1024,256,17.265,0.016,0.463,7.809,7.141,0.017,0.016,0.266,0.145,0.910,17.101
13,prefill,no_overlap,2048,64,448.461,0.059,3.493,255.727,171.999,0.110,0.000,0.000,0.000,0.000,447.966
13,decode_per_token,no_overlap,2048,64,18.185,0.019,0.455,8.576,7.282,0.016,0.021,0.270,0.144,0.910,18.015
14,prefill,overlap,2048,64,448.556,0.068,3.299,258.070,175.206,5.272,0.000,0.000,0.000,0.000,447.934
14,decode_per_token,overlap,2048,64,18.344,0.018,0.456,8.646,7.380,0.018,0.021,0.265,0.144,0.910,18.175
15,prefill,no_overlap,2048,128,454.479,0.061,3.813,257.010,174.576,0.115,0.000,0.000,0.000,0.000,454.020
15,decode_per_token,no_overlap,2048,128,18.476,0.016,0.454,8.830,7.326,0.017,0.023,0.266,0.145,0.910,18.304
16,prefill,overlap,2048,128,447.763,0.067,3.434,258.270,173.253,5.225,0.000,0.000,0.000,0.000,447.321
16,decode_per_token,overlap,2048,128,18.501,0.016,0.461,8.853,7.317,0.018,0.021,0.267,0.145,0.910,18.331
17,prefill,no_overlap,2048,256,452.585,0.060,3.520,256.649,173.212,0.120,0.000,0.000,0.000,0.000,452.160
17,decode_per_token,no_overlap,2048,256,18.538,0.017,0.465,8.876,7.320,0.017,0.026,0.267,0.144,0.910,18.364
18,prefill,overlap,2048,256,450.038,0.064,4.169,258.874,174.046,5.222,0.000,0.000,0.000,0.000,448.535
18,decode_per_token,overlap,2048,256,18.578,0.016,0.466,8.908,7.330,0.017,0.024,0.267,0.144,0.910,18.405
19,prefill,no_overlap,4096,64,1085.963,0.114,7.145,693.672,348.913,0.238,0.000,0.000,0.000,0.000,1085.509
19,decode_per_token,no_overlap,4096,64,20.162,0.015,0.470,10.428,7.387,0.017,0.030,0.268,0.146,0.910,19.982
20,prefill,overlap,4096,64,1087.917,0.117,6.985,696.946,351.495,10.348,0.000,0.000,0.000,0.000,1087.484
20,decode_per_token,overlap,4096,64,20.453,0.019,0.482,10.576,7.488,0.017,0.031,0.269,0.144,0.910,20.274
21,prefill,no_overlap,4096,128,1084.414,0.115,6.779,695.419,345.939,0.228,0.000,0.000,0.000,0.000,1083.976
21,decode_per_token,no_overlap,4096,128,20.495,0.015,0.466,10.713,7.423,0.017,0.031,0.268,0.145,0.910,20.314
22,prefill,overlap,4096,128,1085.979,0.121,6.917,696.809,350.978,10.586,0.000,0.000,0.000,0.000,1085.542
22,decode_per_token,overlap,4096,128,20.138,0.018,0.463,10.463,7.326,0.018,0.026,0.270,0.145,0.910,19.963
23,prefill,no_overlap,4096,256,1095.467,0.118,6.907,699.859,349.586,0.256,0.000,0.000,0.000,0.000,1094.851
23,decode_per_token,no_overlap,4096,256,20.289,0.017,0.463,10.640,7.307,0.018,0.024,0.269,0.145,0.910,20.117
24,prefill,overlap,4096,256,1078.841,0.121,7.354,697.152,348.734,10.458,0.000,0.000,0.000,0.000,1078.361
24,decode_per_token,overlap,4096,256,20.312,0.016,0.464,10.633,7.326,0.018,0.027,0.270,0.145,0.910,20.136

````

### Report: reports/stage1_split_profile_4b_20260225_mainline/stage12_transfer_vs_compute.csv

````csv
split,mode,prefill_transfer_compute_ratio,decode_transfer_compute_ratio,prefill_transfer_ms,prefill_compute_ms,decode_transfer_per_tok_ms,decode_compute_per_tok_ms
12+24,no_overlap,0.0003,0.0180,0.128,420.433,0.311,17.258
12+24,overlap,0.0123,0.0175,5.186,422.623,0.302,17.278
18+18,no_overlap,0.0003,0.0173,0.121,437.801,0.305,17.612
18+18,overlap,0.0119,0.0173,5.216,438.862,0.304,17.543
24+12,no_overlap,0.0003,0.0169,0.125,476.104,0.309,18.282
24+12,overlap,0.0112,0.0173,5.242,467.837,0.312,18.051
27+9,no_overlap,0.0002,0.0162,0.114,479.594,0.303,18.701
27+9,overlap,0.0108,0.0171,5.203,481.037,0.317,18.508
9+27,no_overlap,0.0003,0.0183,0.112,424.016,0.314,17.205
9+27,overlap,0.0125,0.0180,5.260,421.981,0.309,17.173

````

### Report: reports/stage1_split_profile_4b_20260225_mainline/stage12_summary.md

````md
# Stage 1.2 Pipeline Split Profiling

- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Generated at: `2026-02-25T01:32:46`

## Throughput vs Split
| split | mode | total_ms | tok/s_est | prefill_share_% |
| --- | --- | --- | --- | --- |
| 12+24 | no_overlap | 2770.230 | 46.206 | 15.73 |
| 12+24 | overlap | 2767.109 | 46.258 | 15.61 |
| 18+18 | no_overlap | 2830.028 | 45.229 | 16.08 |
| 18+18 | overlap | 2817.737 | 45.427 | 16.07 |
| 24+12 | no_overlap | 2957.505 | 43.280 | 16.86 |
| 24+12 | overlap | 2915.003 | 43.911 | 16.63 |
| 27+9 | no_overlap | 3014.120 | 42.467 | 16.69 |
| 27+9 | overlap | 2987.478 | 42.846 | 16.74 |
| 9+27 | no_overlap | 2769.227 | 46.222 | 15.83 |
| 9+27 | overlap | 2755.312 | 46.456 | 15.61 |

## Key Point
- Best rollout tok/s split: `9+27` in `overlap` mode (`46.456 tok/s`, total `2755.312 ms`).

## Bubble vs Split
| split | no_overlap_ms | overlap_ms | speedup_x | hidden_% |
| --- | --- | --- | --- | --- |
| 12+24 | 2770.230 | 2767.109 | 1.0011 | 0.11 |
| 18+18 | 2830.028 | 2817.737 | 1.0044 | 0.43 |
| 24+12 | 2957.505 | 2915.003 | 1.0146 | 1.44 |
| 27+9 | 3014.120 | 2987.478 | 1.0089 | 0.88 |
| 9+27 | 2769.227 | 2755.312 | 1.0051 | 0.50 |

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`
