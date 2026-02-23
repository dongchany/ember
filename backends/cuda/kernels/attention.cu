#include "kernels.h"
#include <cublas_v2.h>

namespace ember {
namespace cuda {
namespace kernels {

// =============================================================================
// KV Cache Update
// =============================================================================

// 将新的 K/V 写入 cache 的指定位置
// k_new: [batch, seq_len, num_kv_heads, head_dim] -> k_cache: [batch, num_kv_heads, max_seq, head_dim]
__global__ void update_kv_cache_kernel_f16(
    half* k_cache,
    half* v_cache,
    const half* k_new,
    const half* v_new,
    int batch_size,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int max_seq
) {
    // 每个线程处理一个元素
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int seq = blockIdx.x;
    const int dim = threadIdx.x;
    
    if (batch >= batch_size || head >= num_kv_heads || seq >= seq_len || dim >= head_dim) {
        return;
    }
    
    // 源索引: [batch, seq, head, dim]
    const int src_idx = ((batch * seq_len + seq) * num_kv_heads + head) * head_dim + dim;
    
    // 目标索引: [batch, head, pos, dim]
    const int cache_pos = start_pos + seq;
    const int dst_idx = ((batch * num_kv_heads + head) * max_seq + cache_pos) * head_dim + dim;
    
    k_cache[dst_idx] = k_new[src_idx];
    v_cache[dst_idx] = v_new[src_idx];
}

void update_kv_cache_f16(
    half* k_cache,
    half* v_cache,
    const half* k_new,
    const half* v_new,
    int batch_size,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int max_seq,
    cudaStream_t stream
) {
    dim3 grid(seq_len, num_kv_heads, batch_size);
    dim3 block(head_dim);
    
    update_kv_cache_kernel_f16<<<grid, block, 0, stream>>>(
        k_cache, v_cache, k_new, v_new,
        batch_size, seq_len, num_kv_heads, head_dim, start_pos, max_seq
    );
}

__global__ void causal_mask_kernel_f16(half* scores, int seq_q, int seq_k, int start_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_q * seq_k;
    if (idx >= total) return;
    int sq = idx / seq_k;
    int k = idx - sq * seq_k;
    int max_k = start_pos + sq;
    if (k > max_k) {
        scores[idx] = __float2half_rn(-1e4f);
    }
}

__global__ void causal_mask_kernel_bf16(__nv_bfloat16* scores, int seq_q, int seq_k, int start_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_q * seq_k;
    if (idx >= total) return;
    int sq = idx / seq_k;
    int k = idx - sq * seq_k;
    int max_k = start_pos + sq;
    if (k > max_k) {
        scores[idx] = __float2bfloat16_rn(-1e4f);
    }
}

__global__ void causal_mask_kernel_f32(float* scores, int seq_q, int seq_k, int start_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_q * seq_k;
    if (idx >= total) return;
    int sq = idx / seq_k;
    int k = idx - sq * seq_k;
    int max_k = start_pos + sq;
    if (k > max_k) {
        scores[idx] = -1e4f;
    }
}

__global__ void update_kv_cache_kernel_bf16(
    __nv_bfloat16* k_cache,
    __nv_bfloat16* v_cache,
    const __nv_bfloat16* k_new,
    const __nv_bfloat16* v_new,
    int batch_size,
    int seq_len,
    int num_kv_heads,
    int head_dim,
    int start_pos,
    int max_seq
) {
    const int batch = blockIdx.z;
    const int head = blockIdx.y;
    const int seq = blockIdx.x;
    const int dim = threadIdx.x;
    
    if (batch >= batch_size || head >= num_kv_heads || seq >= seq_len || dim >= head_dim) {
        return;
    }
    
    const int src_idx = ((batch * seq_len + seq) * num_kv_heads + head) * head_dim + dim;
    const int cache_pos = start_pos + seq;
    const int dst_idx = ((batch * num_kv_heads + head) * max_seq + cache_pos) * head_dim + dim;
    
    k_cache[dst_idx] = k_new[src_idx];
    v_cache[dst_idx] = v_new[src_idx];
}

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
    cudaStream_t stream
) {
    dim3 grid(seq_len, num_kv_heads, batch_size);
    dim3 block(head_dim);
    
    update_kv_cache_kernel_bf16<<<grid, block, 0, stream>>>(
        k_cache, v_cache, k_new, v_new,
        batch_size, seq_len, num_kv_heads, head_dim, start_pos, max_seq
    );
}

// =============================================================================
// GQA Attention using cuBLAS
// =============================================================================

// GQA: 每个 KV head 对应多个 Q heads
// Q: [batch, seq_q, num_heads, head_dim]
// K_cache: [batch, num_kv_heads, seq_k, head_dim]
// V_cache: [batch, num_kv_heads, seq_k, head_dim]
// Output: [batch, seq_q, num_heads, head_dim]

void attention_f16(
    half* output,
    const half* q,
    const half* k_cache,
    const half* v_cache,
    float* attn_workspace,  // [batch, num_heads, seq_q, seq_k]
    half* attn_probs,       // [batch, num_heads, seq_q, seq_k]
    int batch_size,
    int seq_q,
    int seq_k,
    int max_seq,            // KV cache 的 stride
    int start_pos,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    cublasHandle_t cublas,
    cudaStream_t stream
) {
    // GQA: 每个 KV head 服务 (num_heads / num_kv_heads) 个 Q heads
    const int heads_per_kv = num_heads / num_kv_heads;
    
    // 设置 stream
    cublasSetStream(cublas, stream);
    
    // 使用 FP32 累加避免数值溢出
    const float alpha_f = scale;
    const float beta_zero = 0.0f;
    const float alpha_one = 1.0f;

    // Decode fast path: seq_q=1, batch GEMMs across heads-per-kv to cut launch overhead.
    if (seq_q == 1) {
        for (int b = 0; b < batch_size; ++b) {
            for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
                const int q_head_start = kv_head * heads_per_kv;
                const int group_heads = heads_per_kv;

                const half* k_ptr = k_cache + (b * num_kv_heads + kv_head) * max_seq * head_dim;
                const half* v_ptr = v_cache + (b * num_kv_heads + kv_head) * max_seq * head_dim;
                const half* q_group = q + (b * num_heads + q_head_start) * head_dim;
                float* scores_group = attn_workspace + (b * num_heads + q_head_start) * seq_k;
                half* probs_group = attn_probs + (b * num_heads + q_head_start) * seq_k;
                half* out_group = output + (b * num_heads + q_head_start) * head_dim;

                cublasGemmStridedBatchedEx(
                    cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    seq_k, 1, head_dim,
                    &alpha_f,
                    k_ptr, CUDA_R_16F, head_dim, 0,
                    q_group, CUDA_R_16F, head_dim, head_dim,
                    &beta_zero,
                    scores_group, CUDA_R_32F, seq_k, seq_k,
                    group_heads,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );

                softmax_f32(scores_group, scores_group, 1, group_heads, 1, seq_k, 1.0f, stream);
                convert_f32_to_f16(
                    probs_group,
                    scores_group,
                    static_cast<int64_t>(group_heads) * seq_k,
                    stream
                );

                cublasGemmStridedBatchedEx(
                    cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    head_dim, 1, seq_k,
                    &alpha_one,
                    v_ptr, CUDA_R_16F, head_dim, 0,
                    probs_group, CUDA_R_16F, seq_k, seq_k,
                    &beta_zero,
                    out_group, CUDA_R_16F, head_dim, head_dim,
                    group_heads,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
            }
        }
        return;
    }
    
    // 对每个 batch 和 KV head group 处理
    for (int b = 0; b < batch_size; ++b) {
        for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
            // 这个 KV head 服务的 Q heads 范围
            int q_head_start = kv_head * heads_per_kv;
            int q_head_end = q_head_start + heads_per_kv;
            
            // K 指针: [batch, num_kv_heads, max_seq, head_dim] 中的一个 head
            // 注意：stride 是 max_seq，不是 seq_k！
            const half* k_ptr = k_cache + (b * num_kv_heads + kv_head) * max_seq * head_dim;
            
            // V 指针
            const half* v_ptr = v_cache + (b * num_kv_heads + kv_head) * max_seq * head_dim;
            
            for (int qh = q_head_start; qh < q_head_end; ++qh) {
                // Workspace 位置
                float* scores = attn_workspace + (b * num_heads + qh) * seq_q * seq_k;
                half* probs = attn_probs + (b * num_heads + qh) * seq_q * seq_k;

                // Batched over seq_q:
                // scores: [seq_q, seq_k] row-major in memory (written as column-major [seq_k, seq_q]).
                const half* q_mat = q + (b * seq_q * num_heads + qh) * head_dim;
                cublasGemmEx(
                    cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,  // K^T, Q
                    seq_k, seq_q, head_dim,    // m, n, k
                    &alpha_f,
                    k_ptr, CUDA_R_16F, head_dim,
                    q_mat, CUDA_R_16F, num_heads * head_dim,
                    &beta_zero,
                    scores, CUDA_R_32F, seq_k,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
                
                // Causal mask (prefill only, safe for decode)
                if (seq_q > 1) {
                    int total = seq_q * seq_k;
                    int block = 256;
                    int grid = (total + block - 1) / block;
                    causal_mask_kernel_f32<<<grid, block, 0, stream>>>(scores, seq_q, seq_k, start_pos);
                }
                
                // Softmax (in-place on scores)
                softmax_f32(scores, scores, 1, 1, seq_q, seq_k, 1.0f, stream);
                convert_f32_to_f16(probs, scores, static_cast<int64_t>(seq_q) * seq_k, stream);

                // Output = probs @ V, packed by token stride (num_heads * head_dim).
                half* out_mat = output + (b * seq_q * num_heads + qh) * head_dim;
                cublasGemmEx(
                    cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    head_dim, seq_q, seq_k,
                    &alpha_one,
                    v_ptr, CUDA_R_16F, head_dim,
                    probs, CUDA_R_16F, seq_k,
                    &beta_zero,
                    out_mat, CUDA_R_16F, num_heads * head_dim,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
            }
        }
    }
}

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
    cudaStream_t stream
) {
    const int heads_per_kv = num_heads / num_kv_heads;
    
    cublasSetStream(cublas, stream);
    
    const float alpha_f = scale;
    const float beta_zero = 0.0f;
    const float alpha_one = 1.0f;

    if (seq_q == 1) {
        for (int b = 0; b < batch_size; ++b) {
            for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
                const int q_head_start = kv_head * heads_per_kv;
                const int group_heads = heads_per_kv;

                const __nv_bfloat16* k_ptr = k_cache + (b * num_kv_heads + kv_head) * max_seq * head_dim;
                const __nv_bfloat16* v_ptr = v_cache + (b * num_kv_heads + kv_head) * max_seq * head_dim;
                const __nv_bfloat16* q_group = q + (b * num_heads + q_head_start) * head_dim;
                float* scores_group = attn_workspace + (b * num_heads + q_head_start) * seq_k;
                __nv_bfloat16* probs_group = attn_probs + (b * num_heads + q_head_start) * seq_k;
                __nv_bfloat16* out_group = output + (b * num_heads + q_head_start) * head_dim;

                cublasGemmStridedBatchedEx(
                    cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    seq_k, 1, head_dim,
                    &alpha_f,
                    k_ptr, CUDA_R_16BF, head_dim, 0,
                    q_group, CUDA_R_16BF, head_dim, head_dim,
                    &beta_zero,
                    scores_group, CUDA_R_32F, seq_k, seq_k,
                    group_heads,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );

                softmax_f32(scores_group, scores_group, 1, group_heads, 1, seq_k, 1.0f, stream);
                convert_f32_to_bf16(
                    probs_group,
                    scores_group,
                    static_cast<int64_t>(group_heads) * seq_k,
                    stream
                );

                cublasGemmStridedBatchedEx(
                    cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    head_dim, 1, seq_k,
                    &alpha_one,
                    v_ptr, CUDA_R_16BF, head_dim, 0,
                    probs_group, CUDA_R_16BF, seq_k, seq_k,
                    &beta_zero,
                    out_group, CUDA_R_16BF, head_dim, head_dim,
                    group_heads,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
            }
        }
        return;
    }
    
    for (int b = 0; b < batch_size; ++b) {
        for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
            int q_head_start = kv_head * heads_per_kv;
            int q_head_end = q_head_start + heads_per_kv;
            
            const __nv_bfloat16* k_ptr = k_cache + (b * num_kv_heads + kv_head) * max_seq * head_dim;
            const __nv_bfloat16* v_ptr = v_cache + (b * num_kv_heads + kv_head) * max_seq * head_dim;
            
            for (int qh = q_head_start; qh < q_head_end; ++qh) {
                float* scores = attn_workspace + (b * num_heads + qh) * seq_q * seq_k;
                __nv_bfloat16* probs = attn_probs + (b * num_heads + qh) * seq_q * seq_k;

                const __nv_bfloat16* q_mat = q + (b * seq_q * num_heads + qh) * head_dim;
                cublasGemmEx(
                    cublas,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    seq_k, seq_q, head_dim,
                    &alpha_f,
                    k_ptr, CUDA_R_16BF, head_dim,
                    q_mat, CUDA_R_16BF, num_heads * head_dim,
                    &beta_zero,
                    scores, CUDA_R_32F, seq_k,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
                
                if (seq_q > 1) {
                    int total = seq_q * seq_k;
                    int block = 256;
                    int grid = (total + block - 1) / block;
                    causal_mask_kernel_f32<<<grid, block, 0, stream>>>(scores, seq_q, seq_k, start_pos);
                }
                
                softmax_f32(scores, scores, 1, 1, seq_q, seq_k, 1.0f, stream);
                convert_f32_to_bf16(probs, scores, static_cast<int64_t>(seq_q) * seq_k, stream);

                __nv_bfloat16* out_mat = output + (b * seq_q * num_heads + qh) * head_dim;
                cublasGemmEx(
                    cublas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    head_dim, seq_q, seq_k,
                    &alpha_one,
                    v_ptr, CUDA_R_16BF, head_dim,
                    probs, CUDA_R_16BF, seq_k,
                    &beta_zero,
                    out_mat, CUDA_R_16BF, num_heads * head_dim,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP
                );
            }
        }
    }
}

}  // namespace kernels
}  // namespace cuda
}  // namespace ember
