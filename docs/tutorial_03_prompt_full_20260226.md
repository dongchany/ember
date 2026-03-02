# Tutorial #3 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 3 篇。

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
请写第 3 篇：RMSNorm — 最简单的 CUDA Kernel 入门。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 从 thread/block 概念开始讲
- 用 4 元素向量手算 RMSNorm
- 明确说明 epsilon 的作用
- 解释 warp reduce + shared memory cross-warp reduce 的正确性
```

---

## 2) 代码上下文（完整/相关段落）

### File: backends/cuda/kernels/rmsnorm.cu

````cpp
#include "kernels.h"

namespace ember {
namespace cuda {
namespace kernels {

// =============================================================================
// Warp-level 归约
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum_rmsnorm(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// =============================================================================
// RMSNorm Kernel (F16)
// =============================================================================

template<int BLOCK_SIZE>
__global__ void rms_norm_kernel_f16(
    half* output,
    const half* input,
    const half* weight,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int num_warps = BLOCK_SIZE / 32;
    
    const half* row_input = input + row * hidden_size;
    half* row_output = output + row * hidden_size;
    
    // Step 1: 计算平方和
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = __half2float(row_input[i]);
        sum_sq += val * val;
    }
    
    // Warp-level sum
    sum_sq = warp_reduce_sum_rmsnorm(sum_sq);
    
    // Cross-warp sum
    __shared__ float s_warp_sum[32];
    if (lane_id == 0) {
        s_warp_sum[warp_id] = sum_sq;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? s_warp_sum[lane_id] : 0.0f;
        v = warp_reduce_sum_rmsnorm(v);
        if (lane_id == 0) {
            s_warp_sum[0] = v;
        }
    }
    __syncthreads();
    
    // 计算 RMS inverse
    float rms_inv = rsqrtf(s_warp_sum[0] / hidden_size + eps);
    
    // Step 2: 应用归一化和权重
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = __half2float(row_input[i]);
        float w = __half2float(weight[i]);
        float normed = val * rms_inv * w;
        row_output[i] = __float2half(normed);
    }
}

// =============================================================================
// RMSNorm Kernel (F32)
// =============================================================================

template<int BLOCK_SIZE>
__global__ void rms_norm_kernel_f32(
    float* output,
    const float* input,
    const float* weight,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int num_warps = BLOCK_SIZE / 32;
    
    const float* row_input = input + row * hidden_size;
    float* row_output = output + row * hidden_size;
    
    // Step 1: 计算平方和
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = row_input[i];
        sum_sq += val * val;
    }
    
    // Warp-level sum
    sum_sq = warp_reduce_sum_rmsnorm(sum_sq);
    
    // Cross-warp sum
    __shared__ float s_warp_sum[32];
    if (lane_id == 0) {
        s_warp_sum[warp_id] = sum_sq;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? s_warp_sum[lane_id] : 0.0f;
        v = warp_reduce_sum_rmsnorm(v);
        if (lane_id == 0) {
            s_warp_sum[0] = v;
        }
    }
    __syncthreads();
    
    // 计算 RMS inverse
    float rms_inv = rsqrtf(s_warp_sum[0] / hidden_size + eps);
    
    // Step 2: 应用归一化和权重
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = row_input[i];
        float w = weight[i];
        row_output[i] = val * rms_inv * w;
    }
}

// =============================================================================
// Host functions
// =============================================================================

void rms_norm_f16(
    half* output,
    const half* input,
    const half* weight,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    const int num_rows = batch_size * seq_len;
    
    if (hidden_size <= 1024) {
        rms_norm_kernel_f16<256><<<num_rows, 256, 0, stream>>>(
            output, input, weight, hidden_size, eps);
    } else if (hidden_size <= 4096) {
        rms_norm_kernel_f16<512><<<num_rows, 512, 0, stream>>>(
            output, input, weight, hidden_size, eps);
    } else {
        rms_norm_kernel_f16<1024><<<num_rows, 1024, 0, stream>>>(
            output, input, weight, hidden_size, eps);
    }
}

void rms_norm_f32(
    float* output,
    const float* input,
    const float* weight,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    const int num_rows = batch_size * seq_len;
    
    if (hidden_size <= 1024) {
        rms_norm_kernel_f32<256><<<num_rows, 256, 0, stream>>>(
            output, input, weight, hidden_size, eps);
    } else if (hidden_size <= 4096) {
        rms_norm_kernel_f32<512><<<num_rows, 512, 0, stream>>>(
            output, input, weight, hidden_size, eps);
    } else {
        rms_norm_kernel_f32<1024><<<num_rows, 1024, 0, stream>>>(
            output, input, weight, hidden_size, eps);
    }
}

// =============================================================================
// RMSNorm Kernel (BF16)
// =============================================================================

template<int BLOCK_SIZE>
__global__ void rms_norm_kernel_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    int hidden_size,
    float eps
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int num_warps = BLOCK_SIZE / 32;
    
    const __nv_bfloat16* row_input = input + row * hidden_size;
    __nv_bfloat16* row_output = output + row * hidden_size;
    
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = __bfloat162float(row_input[i]);
        sum_sq += val * val;
    }
    
    sum_sq = warp_reduce_sum_rmsnorm(sum_sq);
    
    __shared__ float s_warp_sum[32];
    if (lane_id == 0) {
        s_warp_sum[warp_id] = sum_sq;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? s_warp_sum[lane_id] : 0.0f;
        v = warp_reduce_sum_rmsnorm(v);
        if (lane_id == 0) {
            s_warp_sum[0] = v;
        }
    }
    __syncthreads();
    
    float rms_inv = rsqrtf(s_warp_sum[0] / hidden_size + eps);
    
    for (int i = tid; i < hidden_size; i += BLOCK_SIZE) {
        float val = __bfloat162float(row_input[i]);
        float w = __bfloat162float(weight[i]);
        row_output[i] = __float2bfloat16_rn(val * rms_inv * w);
    }
}

void rms_norm_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps,
    cudaStream_t stream
) {
    const int num_rows = batch_size * seq_len;
    
    if (hidden_size <= 1024) {
        rms_norm_kernel_bf16<256><<<num_rows, 256, 0, stream>>>(
            output, input, weight, hidden_size, eps);
    } else if (hidden_size <= 4096) {
        rms_norm_kernel_bf16<512><<<num_rows, 512, 0, stream>>>(
            output, input, weight, hidden_size, eps);
    } else {
        rms_norm_kernel_bf16<1024><<<num_rows, 1024, 0, stream>>>(
            output, input, weight, hidden_size, eps);
    }
}

}  // namespace kernels
}  // namespace cuda
}  // namespace ember

````

### File: backends/cuda/kernels/kernels.h

````h
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>

namespace ember {
namespace cuda {
namespace kernels {

// =============================================================================
// RMSNorm
// =============================================================================
// output[i] = x[i] / sqrt(mean(x^2) + eps) * weight[i]
void rms_norm_f16(
    half* output,           // [batch, seq_len, hidden_size]
    const half* input,      // [batch, seq_len, hidden_size]
    const half* weight,     // [hidden_size]
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps,
    cudaStream_t stream = nullptr
);

void rms_norm_f32(
    float* output,
    const float* input,
    const float* weight,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps,
    cudaStream_t stream = nullptr
);

void rms_norm_bf16(
    __nv_bfloat16* output,  // [batch, seq_len, hidden_size]
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    int batch_size,
    int seq_len,
    int hidden_size,
    float eps,
    cudaStream_t stream = nullptr
);

// =============================================================================
// RoPE (Rotary Position Embedding)
// =============================================================================
// 原地应用 RoPE 到 Q 和 K
void apply_rope_f16(
    half* q,                // [batch, seq_len, num_heads, head_dim]
    half* k,                // [batch, seq_len, num_kv_heads, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,          // KV cache 中的起始位置
    float theta,            // RoPE theta (base frequency)
    cudaStream_t stream = nullptr
);

void apply_rope_bf16(
    __nv_bfloat16* q,       // [batch, seq_len, num_heads, head_dim]
    __nv_bfloat16* k,       // [batch, seq_len, num_kv_heads, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    float theta,
    cudaStream_t stream = nullptr
);

// =============================================================================
// Softmax
// =============================================================================
void softmax_f16(
    half* output,           // [batch, heads, seq_q, seq_k]
    const half* input,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    float scale,            // 通常是 1/sqrt(head_dim)
    cudaStream_t stream = nullptr
);

void softmax_f32(
    float* output,
    const float* input,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    float scale,
    cudaStream_t stream = nullptr
);

void softmax_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    float scale,
    cudaStream_t stream = nullptr
);

// =============================================================================
// SiLU (Swish) activation
// =============================================================================
// output = x * sigmoid(x)
void silu_f16(
    half* output,
    const half* input,
    int64_t size,
    cudaStream_t stream = nullptr
);

void silu_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int64_t size,
    cudaStream_t stream = nullptr
);

// =============================================================================
// Fused SiLU * Mul (SwiGLU helper)
// =============================================================================
// gate = silu(gate) * up (in-place on gate)
void silu_mul_fused_f16(
    half* gate_inout,
    const half* up,
    int64_t size,
    cudaStream_t stream = nullptr
);

void silu_mul_fused_bf16(
    __nv_bfloat16* gate_inout,
    const __nv_bfloat16* up,
    int64_t size,
    cudaStream_t stream = nullptr
);

// =============================================================================
// Element-wise operations
// =============================================================================
// output = a * b (element-wise)
void elementwise_mul_f16(
    half* output,
    const half* a,
    const half* b,
    int64_t size,
    cudaStream_t stream = nullptr
);

void elementwise_mul_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    int64_t size,
    cudaStream_t stream = nullptr
);

// output = a + b
void elementwise_add_f16(
    half* output,
    const half* a,
    const half* b,
    int64_t size,
    cudaStream_t stream = nullptr
);

void elementwise_add_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    int64_t size,
    cudaStream_t stream = nullptr
);

// =============================================================================
// Embedding lookup
// =============================================================================
void embedding_lookup_f16(
    half* output,           // [batch, seq_len, hidden_size]
    const half* embedding,  // [vocab_size, hidden_size]
    const int* input_ids,   // [batch, seq_len]
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream = nullptr
);

void embedding_lookup_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* embedding,
    const int* input_ids,
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream = nullptr
);

// =============================================================================
// Copy last token hidden state
// =============================================================================
void copy_last_hidden_f16(
    half* output,           // [batch, hidden_size]
    const half* input,      // [batch, seq_len, hidden_size]
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream = nullptr
);

void copy_last_hidden_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream = nullptr
);

// =============================================================================
// Type conversion
// =============================================================================
void convert_f32_to_f16(
    half* output,
    const float* input,
    int64_t size,
    cudaStream_t stream = nullptr
);

void convert_f16_to_f32(
    float* output,
    const half* input,
    int64_t size,
    cudaStream_t stream = nullptr
);

void convert_f32_to_bf16(
    __nv_bfloat16* output,
    const float* input,
    int64_t size,
    cudaStream_t stream = nullptr
);

void convert_bf16_to_f32(
    float* output,
    const __nv_bfloat16* input,
    int64_t size,
    cudaStream_t stream = nullptr
);

// =============================================================================
// Sampling helpers
// =============================================================================
// Greedy argmax over logits (FP32).
void argmax_f32(
    int* output_ids,        // [batch]
    const float* logits,    // [batch, vocab_size]
    int batch_size,
    int vocab_size,
    cudaStream_t stream = nullptr
);

// =============================================================================
// Attention
// =============================================================================
// Standard attention without FlashAttention
// Q @ K^T -> Softmax -> @ V
void attention_f16(
    half* output,           // [batch, seq_q, num_heads, head_dim]
    const half* q,          // [batch, seq_q, num_heads, head_dim]
    const half* k_cache,    // [batch, num_kv_heads, max_seq, head_dim]
    const half* v_cache,    // [batch, num_kv_heads, max_seq, head_dim]
    float* attn_workspace,  // [batch, num_heads, seq_q, seq_k] workspace (FP32)
    half* attn_probs,       // [batch, num_heads, seq_q, seq_k] probs (compute dtype)
    int batch_size,
    int seq_q,              // 当前 query 长度
    int seq_k,              // KV cache 中的有效长度
    int max_seq,            // KV cache 的最大长度 (用于计算 stride)
    int start_pos,          // 当前序列在 KV cache 中的起始位置（用于 causal mask）
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,            // 1/sqrt(head_dim)
    cublasHandle_t cublas,
    cudaStream_t stream = nullptr
);

void attention_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k_cache,
    const __nv_bfloat16* v_cache,
    float* attn_workspace,
    __nv_bfloat16* attn_probs,
    int batch_size,
    int seq_q,
    int seq_k,
    int max_seq,
    int start_pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    cublasHandle_t cublas,
    cudaStream_t stream = nullptr
);

// Update KV cache
void update_kv_cache_f16(
    half* k_cache,          // [batch, num_kv_heads, max_seq, head_dim]
    half* v_cache,          // [batch, num_kv_heads, max_seq, head_dim]
    const half* k_new,      // [batch, seq_len, num_kv_heads, head_dim]
    const half* v_new,      // [batch, seq_len, num_kv_heads, head_dim]
    int batch_size,
    int seq_len,            // 新增的 token 数
    int num_kv_heads,
    int head_dim,
    int start_pos,          // 写入位置
    int max_seq,            // cache 最大长度
    cudaStream_t stream = nullptr
);

void update_kv_cache_bf16(
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    const __nv_bfloat16* k_new,
    const __nv_bfloat16* v_new,
    int batch_size,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int max_seq,
    cudaStream_t stream = nullptr
);

}  // namespace kernels
}  // namespace cuda
}  // namespace ember

````

### File: backends/cuda/cuda_runtime.cpp (RMSNorm call: input)

````cpp
        }
    }

    if (profile_layers_ && static_cast<size_t>(layer_idx) < last_layer_profile_ms_.size()) {
        auto& ev = profile_events_[device_id];
        CUDA_CHECK(cudaEventRecord(ev.start, stream));
    }

    if (session.runtime_config().check_correctness) {
        int target = session.runtime_config().dump_layer;
        if (target < 0 || target == layer_idx) {
            Error err = dump_last_row(session.runtime_config().dump_dir,
                                      "layer_" + std::to_string(layer_idx) + "_layer_input",
                                      device_id, act.hidden_states,
                                      seq_len, hidden_size, compute_dtype, stream);
            if (err) return err;
        }
    }
    
    // =====================================================================
    // Input LayerNorm
    // =====================================================================
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_start, stream));
    }
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.norm_out),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.norm_out),
            static_cast<const half*>(act.hidden_states),
            static_cast<const half*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    }
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_end, stream));
        CUDA_CHECK(cudaEventRecord(sev.attn_start, stream));
    }
    
    // =====================================================================
    // QKV Projection: norm_out @ W_q/k/v -> q/k/v_proj_out
    // 权重布局: [out_features, in_features] (row-major, Qwen3 safetensors格式)
    // 输入: [batch*seq, hidden_size]
    // 输出: [batch*seq, out_features]
    // =====================================================================
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    
    int M = batch_size * seq_len;  // 批次*序列长度
    
    // Q projection: [M, hidden] @ [hidden, num_heads*head_dim]^T = [M, num_heads*head_dim]
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,  // W^T @ X
        num_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.q_proj_weight, cuda_dtype, hidden_size,  // W: [num_heads*head_dim, hidden] -> W^T
        act.norm_out, cuda_dtype, hidden_size,         // X: [M, hidden]
        &beta_zero,
        act.q_proj_out, cuda_dtype, num_heads * head_dim,  // Y: [M, num_heads*head_dim]
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // K projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.k_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.k_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // V projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.v_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.v_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // =====================================================================
    // Q/K per-head RMSNorm (Qwen3)
    // =====================================================================
    int q_rows = batch_size * seq_len * static_cast<int>(num_heads);
    int k_rows = batch_size * seq_len * static_cast<int>(num_kv_heads);
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.q_proj_out),
            static_cast<const half*>(act.q_proj_out),
            static_cast<const half*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_f16(
            static_cast<half*>(act.k_proj_out),
            static_cast<const half*>(act.k_proj_out),
            static_cast<const half*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,

````

### File: backends/cuda/cuda_runtime.cpp (RMSNorm call: per-head)

````cpp
        &alpha_one,
        layer.v_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.v_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // =====================================================================
    // Q/K per-head RMSNorm (Qwen3)
    // =====================================================================
    int q_rows = batch_size * seq_len * static_cast<int>(num_heads);
    int k_rows = batch_size * seq_len * static_cast<int>(num_kv_heads);
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.q_proj_out),
            static_cast<const half*>(act.q_proj_out),
            static_cast<const half*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_f16(
            static_cast<half*>(act.k_proj_out),
            static_cast<const half*>(act.k_proj_out),
            static_cast<const half*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    }
    
    // =====================================================================
    // Apply RoPE to Q and K
    // =====================================================================
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::apply_rope_bf16(
                static_cast<__nv_bfloat16*>(act.q_proj_out),
                static_cast<__nv_bfloat16*>(act.k_proj_out),
                batch_size, seq_len, num_heads, num_kv_heads, head_dim,
                start_pos, config_.rope_theta,
                stream
            );
        } else {
            kernels::apply_rope_f16(
                static_cast<half*>(act.q_proj_out),
                static_cast<half*>(act.k_proj_out),
                batch_size, seq_len, num_heads, num_kv_heads, head_dim,
                start_pos, config_.rope_theta,
                stream
            );
        }
    }
    
    // =====================================================================
    // Update KV Cache
    // =====================================================================
    auto& kv_cache = session.kv_cache();
    auto& layer_kv = kv_cache.layer(layer_idx);
    
    const int max_seq = session.runtime_config().max_ctx_len;
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::update_kv_cache_bf16(
                static_cast<__nv_bfloat16*>(layer_kv.key_cache.data),
                static_cast<__nv_bfloat16*>(layer_kv.value_cache.data),
                static_cast<const __nv_bfloat16*>(act.k_proj_out),
                static_cast<const __nv_bfloat16*>(act.v_proj_out),
                batch_size, seq_len, num_kv_heads, head_dim,
                start_pos, max_seq,
                stream
            );
        } else {
            kernels::update_kv_cache_f16(
                static_cast<half*>(layer_kv.key_cache.data),
                static_cast<half*>(layer_kv.value_cache.data),
                static_cast<const half*>(act.k_proj_out),
                static_cast<const half*>(act.v_proj_out),
                batch_size, seq_len, num_kv_heads, head_dim,
                start_pos, max_seq,
                stream
            );
        }
    } else {
        for (int b = 0; b < batch_size; ++b) {
            const int sp = start_pos_by_batch[b];
            if (sp < 0) continue;
            if (sp + seq_len > max_seq) {
                return Error(ErrorCode::CONTEXT_TOO_LONG, "Context full (varpos batch)");
            }

            const size_t q_off = static_cast<size_t>(b) * static_cast<size_t>(seq_len) *
                                 static_cast<size_t>(num_heads) * static_cast<size_t>(head_dim);
            const size_t kv_off = static_cast<size_t>(b) * static_cast<size_t>(seq_len) *
                                  static_cast<size_t>(num_kv_heads) * static_cast<size_t>(head_dim);
            const size_t cache_off = static_cast<size_t>(b) * static_cast<size_t>(num_kv_heads) *
                                     static_cast<size_t>(max_seq) * static_cast<size_t>(head_dim);

            if (compute_dtype == DType::BF16) {
                auto* q_ptr = static_cast<__nv_bfloat16*>(act.q_proj_out) + q_off;
                auto* k_ptr = static_cast<__nv_bfloat16*>(act.k_proj_out) + kv_off;
                auto* v_ptr = static_cast<__nv_bfloat16*>(act.v_proj_out) + kv_off;
                auto* k_cache_ptr = static_cast<__nv_bfloat16*>(layer_kv.key_cache.data) + cache_off;
                auto* v_cache_ptr = static_cast<__nv_bfloat16*>(layer_kv.value_cache.data) + cache_off;

                kernels::apply_rope_bf16(
                    q_ptr,
                    k_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_heads,
                    num_kv_heads,

````

### File: backends/cuda/cuda_runtime.cpp (RMSNorm call: post-attn)

````cpp
                                     seq_len, hidden_size, compute_dtype, stream);
        if (err) return err;
    }
    
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.attn_end, stream));
    }
    
    // =====================================================================
    // Post-Attention LayerNorm
    // =====================================================================
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.post_norm_start, stream));
    }
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.norm_out),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(layer.post_attention_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.norm_out),
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

````

### File: backends/cuda/cuda_runtime.cpp (RMSNorm call: final_norm)

````cpp
        CUDA_CHECK(cudaEventRecord(ev.end, stream));
        CUDA_CHECK(cudaEventSynchronize(ev.end));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.start, ev.end));
        last_layer_profile_ms_[static_cast<size_t>(layer_idx)] = ms;
    }
    
    return Error::success();
}

Error CudaRuntime::forward_final_norm(int batch_size, int seq_len, Session& session) {
    int lm_device = device_map_.lm_head_device;
    auto& act = activations_[lm_device];
    auto stream = streams_[lm_device];
    DType compute_dtype = weights_.dtype;
    
    CUDA_CHECK(cudaSetDevice(lm_device));
    
    // 如果最后一层不在 lm_head 设备上，需要拷贝
    int last_layer_device = device_map_.layer_to_device[config_.num_layers - 1];
    if (last_layer_device != lm_device) {
        size_t size = batch_size * seq_len * config_.hidden_size * dtype_size(compute_dtype);
        auto t0 = std::chrono::high_resolution_clock::now();
        Error err = copy_bytes_peer_or_staged(act.hidden_states, lm_device,
                                              activations_[last_layer_device].hidden_states, last_layer_device,
                                              size);
        if (err) return err;
        if (profile_stages_) {
            auto t1 = std::chrono::high_resolution_clock::now();
            last_stage_profile_ms_.p2p_ms += static_cast<float>(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
    }
    
    if (profile_stages_) {
        ensure_simple_stage_events_(final_norm_events_, lm_device);
        CUDA_CHECK(cudaEventRecord(final_norm_events_.start, stream));
    }
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.norm_out),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(weights_.final_norm),
            batch_size, seq_len, config_.hidden_size,
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.norm_out),
            static_cast<const half*>(act.hidden_states),
            static_cast<const half*>(weights_.final_norm),
            batch_size, seq_len, config_.hidden_size,
            config_.rms_norm_eps,
            stream
        );
    }
    if (profile_stages_) {
        CUDA_CHECK(cudaEventRecord(final_norm_events_.end, stream));
    }

    if (session.runtime_config().check_correctness) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                     "final_norm_last_hidden",
                                     lm_device, act.norm_out,
                                     seq_len, config_.hidden_size, compute_dtype, stream);
        if (err) return err;
    }
    
    return Error::success();
}

Error CudaRuntime::forward_lm_head(int batch_size, int seq_len) {
    int lm_device = device_map_.lm_head_device;
    auto& act = activations_[lm_device];
    auto stream = streams_[lm_device];
    auto& cublas = cublas_handles_[lm_device];
    DType compute_dtype = weights_.dtype;
    cudaDataType_t cuda_dtype = to_cuda_dtype(compute_dtype);
    
    CUDA_CHECK(cudaSetDevice(lm_device));
    cublasSetStream(cublas.get(), stream);

    if (profile_stages_) {
        ensure_simple_stage_events_(lm_head_events_, lm_device);
        CUDA_CHECK(cudaEventRecord(lm_head_events_.start, stream));
    }
    
    // 取每个 batch 的最后一个 token hidden state，并打包为 [hidden_size, batch]（列主序列=batch）
    const int hidden_size = static_cast<int>(config_.hidden_size);
    const int vocab_size = static_cast<int>(config_.vocab_size);

    if (compute_dtype == DType::BF16) {
        kernels::copy_last_hidden_bf16(
            static_cast<__nv_bfloat16*>(act.last_hidden),
            static_cast<const __nv_bfloat16*>(act.norm_out),
            batch_size, seq_len, hidden_size,
            stream
        );
    } else {
        kernels::copy_last_hidden_f16(
            static_cast<half*>(act.last_hidden),
            static_cast<const half*>(act.norm_out),
            batch_size, seq_len, hidden_size,
            stream
        );
    }

    const void* hidden_ptr = static_cast<const void*>(act.last_hidden);
    const void* lm_head_ptr = static_cast<const void*>(weights_.lm_head);
    float* logits_ptr = static_cast<float*>(act.logits);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // GEMM: C = alpha * A * B + beta * C
    // A: [vocab_size, hidden_size], B: [hidden_size, batch], C: [vocab_size, batch]
    // 由于 cuBLAS 是列主序，需要调整参数
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,      // A^T * B
        vocab_size, batch_size, hidden_size,
        &alpha,
        lm_head_ptr, cuda_dtype, hidden_size,
        hidden_ptr, cuda_dtype, hidden_size,
        &beta,
        logits_ptr, CUDA_R_32F, vocab_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));


````

### File: backends/cuda/cuda_runtime.cpp (pipeline last-token RMSNorm path)

````cpp
    if (ev.in_norm_start) cudaEventDestroy(ev.in_norm_start);
    if (ev.in_norm_end) cudaEventDestroy(ev.in_norm_end);
    if (ev.attn_start) cudaEventDestroy(ev.attn_start);
    if (ev.attn_end) cudaEventDestroy(ev.attn_end);
    if (ev.post_norm_start) cudaEventDestroy(ev.post_norm_start);
    if (ev.post_norm_end) cudaEventDestroy(ev.post_norm_end);
    if (ev.ffn_start) cudaEventDestroy(ev.ffn_start);
    if (ev.ffn_end) cudaEventDestroy(ev.ffn_end);
    ev = {};

    ev.device_id = device_id;
    cudaSetDevice(device_id);
    cudaEventCreate(&ev.in_norm_start);
    cudaEventCreate(&ev.in_norm_end);
    cudaEventCreate(&ev.attn_start);
    cudaEventCreate(&ev.attn_end);
    cudaEventCreate(&ev.post_norm_start);
    cudaEventCreate(&ev.post_norm_end);
    cudaEventCreate(&ev.ffn_start);
    cudaEventCreate(&ev.ffn_end);
}

void CudaRuntime::ensure_stage_profiling_events_() {
    if (!profile_stages_) return;
    if (layer_stage_events_.size() != static_cast<size_t>(config_.num_layers)) {
        layer_stage_events_.assign(static_cast<size_t>(config_.num_layers), {});
    }
    for (int l = 0; l < config_.num_layers; ++l) {
        ensure_layer_stage_events_(layer_stage_events_[static_cast<size_t>(l)], device_map_.layer_to_device[l]);
    }
    ensure_simple_stage_events_(embedding_events_, device_map_.embedding_device);
    ensure_simple_stage_events_(final_norm_events_, device_map_.lm_head_device);
    ensure_simple_stage_events_(lm_head_events_, device_map_.lm_head_device);
    ensure_simple_stage_events_(total_events_, device_map_.embedding_device);
}

Error CudaRuntime::prefill_chunked_pipeline(const std::vector<int>& tokens,
                                            Session& session,
                                            int chunk_len,
                                            bool overlap,
                                            std::vector<float>* out_logits) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    if (chunk_len <= 0) {
        return Error::invalid_argument("chunk_len must be > 0");
    }
    if (tokens.empty()) {
        return Error::invalid_argument("tokens is empty");
    }

    const int batch_size = 1;
    const int total_len = static_cast<int>(tokens.size());
    if (session.cur_pos() + total_len > session.runtime_config().max_ctx_len) {
        return Error(ErrorCode::CONTEXT_TOO_LONG, "Input too long");
    }

    // B 阶段 baseline：仅支持 2-GPU、连续切分的 pipeline。
    if (device_map_.num_devices != 2) {
        return Error::invalid_argument("prefill_chunked_pipeline supports exactly 2 GPUs");
    }

    const int gpu0 = device_map_.embedding_device;
    const int gpu1 = device_map_.lm_head_device;
    if (gpu0 == gpu1) {
        return prefill(tokens, session);
    }

    // Find a single boundary: gpu0 layers first, then gpu1 layers.
    int boundary = -1;
    for (int i = 1; i < config_.num_layers; ++i) {
        if (device_map_.layer_to_device[i - 1] == gpu0 && device_map_.layer_to_device[i] == gpu1) {
            boundary = i;
            break;
        }
    }
    if (boundary <= 0 || boundary >= config_.num_layers) {
        return Error::invalid_argument("device_map must be a 2-stage contiguous split for pipeline prefill");
    }
    for (int i = 0; i < boundary; ++i) {
        if (device_map_.layer_to_device[i] != gpu0) {
            return Error::invalid_argument("non-contiguous split before boundary");
        }
    }
    for (int i = boundary; i < config_.num_layers; ++i) {
        if (device_map_.layer_to_device[i] != gpu1) {
            return Error::invalid_argument("non-contiguous split after boundary");
        }
    }

    const int max_ctx = session.runtime_config().max_ctx_len;
    // Token buffers only need to hold one chunk; attention workspace needs [seq_q=chunk_len, seq_k<=max_ctx].
    Error err = allocate_activation_buffers(chunk_len, batch_size, /*attn_q_max=*/chunk_len, /*attn_k_max=*/max_ctx);
    if (err) return err;

    EMBER_RETURN_IF_ERROR(begin_stage_profile_());

    auto accumulate_layer_stage_ms = [&](int layer_idx) -> Error {
        if (!profile_stages_) return Error::success();
        const auto& ev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        float ms = 0.0f;
        CUDA_CHECK(cudaSetDevice(ev.device_id));
        // Ensure the recorded end event has completed so elapsed-time queries are valid.
        CUDA_CHECK(cudaEventSynchronize(ev.ffn_end));
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.in_norm_start, ev.in_norm_end));
        last_stage_profile_ms_.rmsnorm_ms += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.post_norm_start, ev.post_norm_end));
        last_stage_profile_ms_.rmsnorm_ms += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.attn_start, ev.attn_end));
        last_stage_profile_ms_.attention_ms += ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev.ffn_start, ev.ffn_end));
        last_stage_profile_ms_.ffn_ms += ms;
        return Error::success();
    };

    const DType compute_dtype = weights_.dtype;
    const size_t elem = dtype_size(compute_dtype);
    const size_t hidden = static_cast<size_t>(config_.hidden_size);
    const size_t max_chunk_bytes = static_cast<size_t>(chunk_len) * hidden * elem;

    int* d_input_ids = nullptr;
    void* stage0_out[2] = {nullptr, nullptr};
    void* gpu1_io[2] = {nullptr, nullptr};
    cudaEvent_t stage0_ready[2] = {nullptr, nullptr};
    cudaEvent_t xfer_done[2] = {nullptr, nullptr};
    cudaEvent_t stage1_done[2] = {nullptr, nullptr};

    auto cleanup = [&]() {
        cudaSetDevice(gpu0);
        if (d_input_ids) cudaFree(d_input_ids);
        if (stage0_out[0]) cudaFree(stage0_out[0]);
        if (stage0_out[1]) cudaFree(stage0_out[1]);
        if (stage0_ready[0]) cudaEventDestroy(stage0_ready[0]);
        if (stage0_ready[1]) cudaEventDestroy(stage0_ready[1]);
        if (xfer_done[0]) cudaEventDestroy(xfer_done[0]);
        if (xfer_done[1]) cudaEventDestroy(xfer_done[1]);

        cudaSetDevice(gpu1);
        if (gpu1_io[0]) cudaFree(gpu1_io[0]);
        if (gpu1_io[1]) cudaFree(gpu1_io[1]);
        if (stage1_done[0]) cudaEventDestroy(stage1_done[0]);
        if (stage1_done[1]) cudaEventDestroy(stage1_done[1]);
    };

    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaMalloc(&d_input_ids, static_cast<size_t>(chunk_len) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&stage0_out[0], max_chunk_bytes));
    CUDA_CHECK(cudaMalloc(&stage0_out[1], max_chunk_bytes));
    CUDA_CHECK(cudaEventCreateWithFlags(&stage0_ready[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&stage0_ready[1], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&xfer_done[0]));
    CUDA_CHECK(cudaEventCreate(&xfer_done[1]));

    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaMalloc(&gpu1_io[0], max_chunk_bytes));
    CUDA_CHECK(cudaMalloc(&gpu1_io[1], max_chunk_bytes));
    CUDA_CHECK(cudaEventCreateWithFlags(&stage1_done[0], cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&stage1_done[1], cudaEventDisableTiming));

    const int num_chunks = (total_len + chunk_len - 1) / chunk_len;
    const int start_pos0 = session.cur_pos();

    struct TimedEvents {
        int device_id = -1;
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
    };
    std::vector<TimedEvents> copy_events;
    std::vector<TimedEvents> embed_events;

    auto destroy_copy_events = [&]() {
        for (auto& ce : copy_events) {
            if (ce.start) cudaEventDestroy(ce.start);
            if (ce.end) cudaEventDestroy(ce.end);
            ce = {};
        }
        copy_events.clear();
        for (auto& ee : embed_events) {
            if (ee.start) cudaEventDestroy(ee.start);
            if (ee.end) cudaEventDestroy(ee.end);
            ee = {};
        }
        embed_events.clear();
    };

    auto run_stage1 = [&](int chunk_index) -> Error {
        const int off = chunk_index * chunk_len;
        const int clen = std::min(chunk_len, total_len - off);
        const int slot = chunk_index & 1;

        // Ensure transfer is done (event recorded on gpu0).
        CUDA_CHECK(cudaSetDevice(gpu0));
        CUDA_CHECK(cudaEventSynchronize(xfer_done[slot]));

        CUDA_CHECK(cudaSetDevice(gpu1));
        void* saved_hidden = activations_[gpu1].hidden_states;
        activations_[gpu1].hidden_states = gpu1_io[slot];

        const int chunk_start_pos = start_pos0 + off;
        for (int layer = boundary; layer < config_.num_layers; ++layer) {
            const bool skip = (layer == boundary);

````

---

## 3) 报告上下文（完整）

### Report: reports/stage31_base_operator_spotcheck_4b_20260225_mainline/stage31_base_operator_spotcheck.csv

````csv
layer,layer_max_abs_diff,layer_mean_abs_diff,attn_out_max_abs_diff,attn_residual_max_abs_diff,post_attn_norm_max_abs_diff,mlp_out_max_abs_diff,gate_proj_max_abs_diff,up_proj_max_abs_diff,gate_proj_ember_norm_max_abs_diff,up_proj_ember_norm_max_abs_diff
0,0.205282,0.017445,0.147710,0.149907,0.113919,0.370814,0.957237,0.715941,0.003892,0.001909
1,0.427129,0.031933,0.180284,0.364644,4.198868,0.287741,6.213398,5.650882,0.031280,0.030877

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
