# Tutorial #5 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 5 篇。

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
请写第 5 篇：Attention + Softmax — 从 Q·K^T 到加权求和。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 手算一个 2-token 的 attention 例子
- 解释为什么要除以 sqrt(d_k)
- 指出数值稳定性（max-trick）在 softmax 中的作用
```

---

## 2) 代码上下文（完整/相关段落）

### File: backends/cuda/kernels/attention.cu

````cpp
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

````

### File: backends/cuda/kernels/softmax.cu

````cpp
#include "kernels.h"
#include <cfloat>

namespace ember {
namespace cuda {
namespace kernels {

// =============================================================================
// Warp-level 归约
// =============================================================================

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// =============================================================================
// Softmax Kernel (F16)
// =============================================================================

template<int BLOCK_SIZE>
__global__ void softmax_kernel_f16(
    half* output,
    const half* input,
    int seq_k,
    float scale
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int num_warps = BLOCK_SIZE / 32;
    
    const half* row_input = input + row * seq_k;
    half* row_output = output + row * seq_k;
    
    // Step 1: Find max
    float max_val = -FLT_MAX;
    for (int i = tid; i < seq_k; i += BLOCK_SIZE) {
        float val = __half2float(row_input[i]) * scale;
        max_val = fmaxf(max_val, val);
    }
    
    // Warp-level max
    max_val = warp_reduce_max(max_val);
    
    // Cross-warp max
    __shared__ float s_warp_max[32];
    if (lane_id == 0) {
        s_warp_max[warp_id] = max_val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? s_warp_max[lane_id] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane_id == 0) {
            s_warp_max[0] = v;
        }
    }
    __syncthreads();
    
    float row_max = s_warp_max[0];
    
    // Step 2: Compute exp sum
    float sum = 0.0f;
    for (int i = tid; i < seq_k; i += BLOCK_SIZE) {
        float val = __half2float(row_input[i]) * scale;
        sum += expf(val - row_max);
    }
    
    // Warp-level sum
    sum = warp_reduce_sum(sum);
    
    // Cross-warp sum
    __shared__ float s_warp_sum[32];
    if (lane_id == 0) {
        s_warp_sum[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? s_warp_sum[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) {
            s_warp_sum[0] = v;
        }
    }
    __syncthreads();
    
    float row_sum_inv = 1.0f / (s_warp_sum[0] + 1e-6f);
    
    // Step 3: Write normalized output
    for (int i = tid; i < seq_k; i += BLOCK_SIZE) {
        float val = __half2float(row_input[i]) * scale;
        float prob = expf(val - row_max) * row_sum_inv;
        row_output[i] = __float2half(prob);
    }
}

// =============================================================================
// Softmax Kernel (BF16)
// =============================================================================

template<int BLOCK_SIZE>
__global__ void softmax_kernel_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int seq_k,
    float scale
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int num_warps = BLOCK_SIZE / 32;
    
    const __nv_bfloat16* row_input = input + row * seq_k;
    __nv_bfloat16* row_output = output + row * seq_k;
    
    float max_val = -FLT_MAX;
    for (int i = tid; i < seq_k; i += BLOCK_SIZE) {
        float val = __bfloat162float(row_input[i]) * scale;
        max_val = fmaxf(max_val, val);
    }
    
    max_val = warp_reduce_max(max_val);
    
    __shared__ float s_warp_max[32];
    if (lane_id == 0) {
        s_warp_max[warp_id] = max_val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? s_warp_max[lane_id] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane_id == 0) {
            s_warp_max[0] = v;
        }
    }
    __syncthreads();
    
    float row_max = s_warp_max[0];
    
    float sum = 0.0f;
    for (int i = tid; i < seq_k; i += BLOCK_SIZE) {
        float val = __bfloat162float(row_input[i]) * scale;
        sum += expf(val - row_max);
    }
    
    sum = warp_reduce_sum(sum);
    
    __shared__ float s_warp_sum[32];
    if (lane_id == 0) {
        s_warp_sum[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? s_warp_sum[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) {
            s_warp_sum[0] = v;
        }
    }
    __syncthreads();
    
    float row_sum_inv = 1.0f / (s_warp_sum[0] + 1e-6f);
    
    for (int i = tid; i < seq_k; i += BLOCK_SIZE) {
        float val = __bfloat162float(row_input[i]) * scale;
        float prob = expf(val - row_max) * row_sum_inv;
        row_output[i] = __float2bfloat16_rn(prob);
    }
}

// =============================================================================
// Softmax Kernel (F32)
// =============================================================================

template<int BLOCK_SIZE>
__global__ void softmax_kernel_f32(
    float* output,
    const float* input,
    int seq_k,
    float scale
) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    constexpr int num_warps = BLOCK_SIZE / 32;
    
    const float* row_input = input + row * seq_k;
    float* row_output = output + row * seq_k;
    
    // Step 1: Find max
    float max_val = -FLT_MAX;
    for (int i = tid; i < seq_k; i += BLOCK_SIZE) {
        float val = row_input[i] * scale;
        max_val = fmaxf(max_val, val);
    }
    
    max_val = warp_reduce_max(max_val);
    
    __shared__ float s_warp_max[32];
    if (lane_id == 0) {
        s_warp_max[warp_id] = max_val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? s_warp_max[lane_id] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (lane_id == 0) {
            s_warp_max[0] = v;
        }
    }
    __syncthreads();
    
    float row_max = s_warp_max[0];
    
    // Step 2: Compute exp sum
    float sum = 0.0f;
    for (int i = tid; i < seq_k; i += BLOCK_SIZE) {
        float val = row_input[i] * scale;
        sum += expf(val - row_max);
    }
    
    sum = warp_reduce_sum(sum);
    
    __shared__ float s_warp_sum[32];
    if (lane_id == 0) {
        s_warp_sum[warp_id] = sum;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float v = (lane_id < num_warps) ? s_warp_sum[lane_id] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane_id == 0) {
            s_warp_sum[0] = v;
        }
    }
    __syncthreads();
    
    float row_sum_inv = 1.0f / (s_warp_sum[0] + 1e-6f);
    
    // Step 3: Write output
    for (int i = tid; i < seq_k; i += BLOCK_SIZE) {
        float val = row_input[i] * scale;
        row_output[i] = expf(val - row_max) * row_sum_inv;
    }
}

// =============================================================================
// Host functions
// =============================================================================

void softmax_f16(
    half* output,
    const half* input,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    float scale,
    cudaStream_t stream
) {
    const int num_rows = batch_size * num_heads * seq_q;
    
    if (seq_k <= 256) {
        softmax_kernel_f16<128><<<num_rows, 128, 0, stream>>>(output, input, seq_k, scale);
    } else if (seq_k <= 1024) {
        softmax_kernel_f16<256><<<num_rows, 256, 0, stream>>>(output, input, seq_k, scale);
    } else if (seq_k <= 4096) {
        softmax_kernel_f16<512><<<num_rows, 512, 0, stream>>>(output, input, seq_k, scale);
    } else {
        softmax_kernel_f16<1024><<<num_rows, 1024, 0, stream>>>(output, input, seq_k, scale);
    }
}

void softmax_f32(
    float* output,
    const float* input,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    float scale,
    cudaStream_t stream
) {
    const int num_rows = batch_size * num_heads * seq_q;
    
    if (seq_k <= 256) {
        softmax_kernel_f32<128><<<num_rows, 128, 0, stream>>>(output, input, seq_k, scale);
    } else if (seq_k <= 1024) {
        softmax_kernel_f32<256><<<num_rows, 256, 0, stream>>>(output, input, seq_k, scale);
    } else if (seq_k <= 4096) {
        softmax_kernel_f32<512><<<num_rows, 512, 0, stream>>>(output, input, seq_k, scale);
    } else {
        softmax_kernel_f32<1024><<<num_rows, 1024, 0, stream>>>(output, input, seq_k, scale);
    }
}

void softmax_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int batch_size,
    int num_heads,
    int seq_q,
    int seq_k,
    float scale,
    cudaStream_t stream
) {
    const int num_rows = batch_size * num_heads * seq_q;
    
    if (seq_k <= 256) {
        softmax_kernel_bf16<128><<<num_rows, 128, 0, stream>>>(output, input, seq_k, scale);
    } else if (seq_k <= 1024) {
        softmax_kernel_bf16<256><<<num_rows, 256, 0, stream>>>(output, input, seq_k, scale);
    } else if (seq_k <= 4096) {
        softmax_kernel_bf16<512><<<num_rows, 512, 0, stream>>>(output, input, seq_k, scale);
    } else {
        softmax_kernel_bf16<1024><<<num_rows, 1024, 0, stream>>>(output, input, seq_k, scale);
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

### File: backends/cuda/cuda_runtime.cpp (attention (bf16) call block)

````cpp
                    head_dim,
                    sp,
                    config_.rope_theta,
                    stream
                );
                kernels::update_kv_cache_f16(
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
            }
        }
    }
    
    // =====================================================================
    // Attention: Q @ K^T / sqrt(d) -> Softmax -> @ V
    // =====================================================================
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
        }
    }
    
    // =====================================================================
    // O Projection: attn_out @ W_o -> hidden_states
    // attn_out: [M, num_heads*head_dim]
    // W_o: [hidden_size, num_heads*head_dim]
    // output: [M, hidden_size]
    // =====================================================================
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_size, M, num_heads * head_dim,
        &alpha_one,
        layer.o_proj_weight, cuda_dtype, num_heads * head_dim,
        act.attn_out, cuda_dtype, num_heads * head_dim,
        &beta_zero,
        act.mlp_down, cuda_dtype, hidden_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,

````

### File: backends/cuda/cuda_runtime.cpp (attention (f16) call block)

````cpp
                    num_kv_heads,
                    head_dim,
                    sp,
                    max_seq,
                    stream
                );
            }
        }
    }
    
    // =====================================================================
    // Attention: Q @ K^T / sqrt(d) -> Softmax -> @ V
    // =====================================================================
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
        }
    }
    
    // =====================================================================
    // O Projection: attn_out @ W_o -> hidden_states
    // attn_out: [M, num_heads*head_dim]
    // W_o: [hidden_size, num_heads*head_dim]
    // output: [M, hidden_size]
    // =====================================================================
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_size, M, num_heads * head_dim,
        &alpha_one,
        layer.o_proj_weight, cuda_dtype, num_heads * head_dim,
        act.attn_out, cuda_dtype, num_heads * head_dim,
        &beta_zero,
        act.mlp_down, cuda_dtype, hidden_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                     "layer_" + std::to_string(layer_idx) + "_attn_out",
                                     device_id, act.mlp_down,
                                     seq_len, hidden_size, compute_dtype, stream);
        if (err) return err;
    }
    
    // =====================================================================
    // Residual Connection (Attention)
    // =====================================================================
    if (compute_dtype == DType::BF16) {
        kernels::elementwise_add_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),

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

### Report: reports/stage1_split_profile_4b_20260225_mainline/stage12_vs_llama.md

````md
# Stage 1.2 Ember vs llama.cpp

- GGUF: `/home/dong/workspace/ember/reports/gguf/Qwen3-4B-BF16.gguf`
- llama split_mode: `layer`
- llama tensor_split: `auto/none`

| engine | config | prefill_ms | decode_tok_s | decode_per_token_ms | rollout_total_ms | rollout_tok_s | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ember | split=9+27 mode=overlap | 430.064 | 55.048 | 18.166 | 2755.312 | 46.456 | best split from stage12 sweep |
| llama.cpp | CUDA0 single | 542.871 | 73.898 | 13.532 | 2274.996 | 56.264 | llama-bench |
| llama.cpp | CUDA0/CUDA1 dual | 325.628 | 82.248 | 12.158 | 1881.900 | 68.016 | llama-bench split_mode=layer |

- Note: llama-bench and ember_stage_breakdown are different harnesses; compare as practical baseline, not strict apples-to-apples kernel microbenchmark.

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`
