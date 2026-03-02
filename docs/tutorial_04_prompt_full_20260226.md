# Tutorial #4 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 4 篇。

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
请写第 4 篇：RoPE — 旋转位置编码的几何直觉。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 用二维旋转矩阵先讲直觉，再推广到高维偶奇位
- 给一个 pos=0/1/2 的最小数值例子
```

---

## 2) 代码上下文（完整/相关段落）

### File: backends/cuda/kernels/rope.cu

````cpp
#include "kernels.h"
#include <cmath>

namespace ember {
namespace cuda {
namespace kernels {

// =============================================================================
// RoPE (Rotary Position Embedding) Kernel
// =============================================================================

// 每个线程处理一个 (batch, seq, head, dim_pair)
__global__ void rope_kernel_f16(
    half* q,                // [batch, seq_len, num_heads, head_dim]
    half* k,                // [batch, seq_len, num_kv_heads, head_dim]
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    float theta
) {
    // 计算当前处理的位置
    const int batch = blockIdx.z;
    const int seq = blockIdx.y;
    const int head = blockIdx.x;
    const int dim_pair = threadIdx.x;  // 处理 dim_pair*2 和 dim_pair*2+1
    
    const int half_head_dim = head_dim / 2;
    if (dim_pair >= half_head_dim) return;
    
    // 计算位置编码频率
    // freq = 1.0 / (theta ^ (2i / head_dim))
    const int pos = start_pos + seq;
    const float freq = 1.0f / powf(theta, float(dim_pair * 2) / float(head_dim));
    const float angle = pos * freq;
    
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    
    // 处理 Q
    if (head < num_heads) {
        const int q_offset = ((batch * seq_len + seq) * num_heads + head) * head_dim;
        
        float q0 = __half2float(q[q_offset + dim_pair * 2]);
        float q1 = __half2float(q[q_offset + dim_pair * 2 + 1]);
        
        // 旋转: (q0, q1) -> (q0*cos - q1*sin, q0*sin + q1*cos)
        float new_q0 = q0 * cos_val - q1 * sin_val;
        float new_q1 = q0 * sin_val + q1 * cos_val;
        
        q[q_offset + dim_pair * 2] = __float2half(new_q0);
        q[q_offset + dim_pair * 2 + 1] = __float2half(new_q1);
    }
    
    // 处理 K（只有 num_kv_heads 个）
    if (head < num_kv_heads) {
        const int k_offset = ((batch * seq_len + seq) * num_kv_heads + head) * head_dim;
        
        float k0 = __half2float(k[k_offset + dim_pair * 2]);
        float k1 = __half2float(k[k_offset + dim_pair * 2 + 1]);
        
        float new_k0 = k0 * cos_val - k1 * sin_val;
        float new_k1 = k0 * sin_val + k1 * cos_val;
        
        k[k_offset + dim_pair * 2] = __float2half(new_k0);
        k[k_offset + dim_pair * 2 + 1] = __float2half(new_k1);
    }
}

void apply_rope_f16(
    half* q,
    half* k,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    float theta,
    cudaStream_t stream
) {
    // 每个 block 处理一个 (batch, seq, head)
    // 每个线程处理一对维度
    dim3 grid(max(num_heads, num_kv_heads), seq_len, batch_size);
    dim3 block(head_dim / 2);
    
    rope_kernel_f16<<<grid, block, 0, stream>>>(
        q, k, batch_size, seq_len, num_heads, num_kv_heads, 
        head_dim, start_pos, theta
    );
}

// =============================================================================
// RoPE (BF16)
// =============================================================================

__global__ void rope_kernel_bf16(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    float theta
) {
    const int batch = blockIdx.z;
    const int seq = blockIdx.y;
    const int head = blockIdx.x;
    const int dim_pair = threadIdx.x;
    
    const int half_head_dim = head_dim / 2;
    if (dim_pair >= half_head_dim) return;
    
    const int pos = start_pos + seq;
    const float freq = 1.0f / powf(theta, float(dim_pair * 2) / float(head_dim));
    const float angle = pos * freq;
    
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    
    if (head < num_heads) {
        const int q_offset = ((batch * seq_len + seq) * num_heads + head) * head_dim;
        
        float q0 = __bfloat162float(q[q_offset + dim_pair * 2]);
        float q1 = __bfloat162float(q[q_offset + dim_pair * 2 + 1]);
        
        float new_q0 = q0 * cos_val - q1 * sin_val;
        float new_q1 = q0 * sin_val + q1 * cos_val;
        
        q[q_offset + dim_pair * 2] = __float2bfloat16_rn(new_q0);
        q[q_offset + dim_pair * 2 + 1] = __float2bfloat16_rn(new_q1);
    }
    
    if (head < num_kv_heads) {
        const int k_offset = ((batch * seq_len + seq) * num_kv_heads + head) * head_dim;
        
        float k0 = __bfloat162float(k[k_offset + dim_pair * 2]);
        float k1 = __bfloat162float(k[k_offset + dim_pair * 2 + 1]);
        
        float new_k0 = k0 * cos_val - k1 * sin_val;
        float new_k1 = k0 * sin_val + k1 * cos_val;
        
        k[k_offset + dim_pair * 2] = __float2bfloat16_rn(new_k0);
        k[k_offset + dim_pair * 2 + 1] = __float2bfloat16_rn(new_k1);
    }
}

void apply_rope_bf16(
    __nv_bfloat16* q,
    __nv_bfloat16* k,
    int batch_size,
    int seq_len,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    float theta,
    cudaStream_t stream
) {
    dim3 grid(max(num_heads, num_kv_heads), seq_len, batch_size);
    dim3 block(head_dim / 2);
    
    rope_kernel_bf16<<<grid, block, 0, stream>>>(
        q, k, batch_size, seq_len, num_heads, num_kv_heads, 
        head_dim, start_pos, theta
    );
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

### File: backends/cuda/cuda_runtime.cpp (RoPE call sites (forward_layer))

````cpp
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
                    head_dim,
                    sp,
                    config_.rope_theta,
                    stream
                );
                kernels::update_kv_cache_bf16(
                    k_cache_ptr,
                    v_cache_ptr,
                    k_ptr,
                    v_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_kv_heads,
                    head_dim,
                    sp,
                    max_seq,
                    stream
                );
            } else {
                auto* q_ptr = static_cast<half*>(act.q_proj_out) + q_off;
                auto* k_ptr = static_cast<half*>(act.k_proj_out) + kv_off;
                auto* v_ptr = static_cast<half*>(act.v_proj_out) + kv_off;
                auto* k_cache_ptr = static_cast<half*>(layer_kv.key_cache.data) + cache_off;
                auto* v_cache_ptr = static_cast<half*>(layer_kv.value_cache.data) + cache_off;

                kernels::apply_rope_f16(
                    q_ptr,
                    k_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    sp,
                    config_.rope_theta,
                    stream
                );
                kernels::update_kv_cache_f16(
                    k_cache_ptr,
                    v_cache_ptr,
                    k_ptr,

````

---

## 3) 报告上下文（完整）

### Report: reports/stage1_milestone_4b_20260225_mainline/stage1_summary.md

````md
# Stage 1.1 Milestone Summary

- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Generated at: `2026-02-25T01:29:40`

| gpus | split | mode | prompt_len | decode_steps | prefill_ms | decode_per_token_ms | prefill_share_% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0+1 | 18+18 | no_overlap | 1024 | 128 | 189.054 | 17.173 | 7.92 |
| 0+1 | 18+18 | no_overlap | 1024 | 256 | 190.418 | 17.172 | 4.15 |
| 0+1 | 18+18 | no_overlap | 1024 | 64 | 187.029 | 17.171 | 14.54 |
| 0+1 | 18+18 | no_overlap | 2048 | 128 | 454.479 | 18.476 | 16.12 |
| 0+1 | 18+18 | no_overlap | 2048 | 256 | 452.585 | 18.538 | 8.71 |
| 0+1 | 18+18 | no_overlap | 2048 | 64 | 448.461 | 18.185 | 27.81 |
| 0+1 | 18+18 | no_overlap | 4096 | 128 | 1084.414 | 20.495 | 29.25 |
| 0+1 | 18+18 | no_overlap | 4096 | 256 | 1095.467 | 20.289 | 17.42 |
| 0+1 | 18+18 | no_overlap | 4096 | 64 | 1085.963 | 20.162 | 45.70 |
| 0+1 | 18+18 | no_overlap | 512 | 128 | 88.459 | 16.664 | 3.98 |
| 0+1 | 18+18 | no_overlap | 512 | 256 | 88.575 | 16.666 | 2.03 |
| 0+1 | 18+18 | no_overlap | 512 | 64 | 88.663 | 16.593 | 7.71 |
| 0+1 | 18+18 | overlap | 1024 | 128 | 190.202 | 17.192 | 7.96 |
| 0+1 | 18+18 | overlap | 1024 | 256 | 193.988 | 17.265 | 4.20 |
| 0+1 | 18+18 | overlap | 1024 | 64 | 185.929 | 17.217 | 14.44 |
| 0+1 | 18+18 | overlap | 2048 | 128 | 447.763 | 18.501 | 15.90 |
| 0+1 | 18+18 | overlap | 2048 | 256 | 450.038 | 18.578 | 8.64 |
| 0+1 | 18+18 | overlap | 2048 | 64 | 448.556 | 18.344 | 27.64 |
| 0+1 | 18+18 | overlap | 4096 | 128 | 1085.979 | 20.138 | 29.64 |
| 0+1 | 18+18 | overlap | 4096 | 256 | 1078.841 | 20.312 | 17.18 |
| 0+1 | 18+18 | overlap | 4096 | 64 | 1087.917 | 20.453 | 45.39 |
| 0+1 | 18+18 | overlap | 512 | 128 | 88.449 | 16.684 | 3.98 |
| 0+1 | 18+18 | overlap | 512 | 256 | 91.693 | 16.691 | 2.10 |
| 0+1 | 18+18 | overlap | 512 | 64 | 88.055 | 16.610 | 7.65 |

## Key Point
- Highest prefill share: `45.70%` (prompt_len=4096, decode_steps=64, gpus=0+1, mode=no_overlap).

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
