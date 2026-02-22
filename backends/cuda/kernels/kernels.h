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
