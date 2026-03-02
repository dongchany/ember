# Tutorial #6 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 6 篇。

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
请写第 6 篇：SwiGLU MLP — 为什么不是简单 ReLU。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 讲清 gate/up/down 三条支路
- 用小向量演示 SiLU(x)*gate 的非线性效果
```

---

## 2) 代码上下文（完整/相关段落）

### File: backends/cuda/kernels/ops.cu

````cpp
#include "kernels.h"

namespace ember {
namespace cuda {
namespace kernels {

namespace {

static inline bool is_aligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

}  // namespace

// =============================================================================
// Greedy Sampling (Argmax)
// =============================================================================

__global__ void argmax_f32_kernel(int* output_ids, const float* logits, int vocab_size) {
    const int b = blockIdx.x;
    const int tid = threadIdx.x;

    float best_val = -1e20f;
    int best_idx = 0;

    const float* row = logits + static_cast<size_t>(b) * static_cast<size_t>(vocab_size);
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float v = row[i];
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    extern __shared__ unsigned char smem[];
    float* smax = reinterpret_cast<float*>(smem);
    int* sidx = reinterpret_cast<int*>(smem + blockDim.x * sizeof(float));

    smax[tid] = best_val;
    sidx[tid] = best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_val = smax[tid + stride];
            const int other_idx = sidx[tid + stride];
            if (other_val > smax[tid]) {
                smax[tid] = other_val;
                sidx[tid] = other_idx;
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        output_ids[b] = sidx[0];
    }
}

void argmax_f32(int* output_ids, const float* logits, int batch_size, int vocab_size, cudaStream_t stream) {
    const int block = 256;
    const int grid = batch_size;
    const size_t shmem = static_cast<size_t>(block) * (sizeof(float) + sizeof(int));
    argmax_f32_kernel<<<grid, block, shmem, stream>>>(output_ids, logits, vocab_size);
}

// =============================================================================
// SiLU (Swish) Activation
// =============================================================================

__global__ void silu_kernel_f16_scalar(half* output, const half* input, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __half2float(input[idx]);
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = __float2half(x * sigmoid);
    }
}

__global__ void silu_kernel_f16_half2(__half2* output, const __half2* input, int64_t size2) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size2) {
        const float2 x = __half22float2(input[idx]);
        const float2 sig = make_float2(
            1.0f / (1.0f + expf(-x.x)),
            1.0f / (1.0f + expf(-x.y))
        );
        output[idx] = __floats2half2_rn(x.x * sig.x, x.y * sig.y);
    }
}

void silu_f16(half* output, const half* input, int64_t size, cudaStream_t stream) {
    const int block_size = 256;
    if (size > 0 && (size % 2 == 0) && is_aligned(output, alignof(__half2)) &&
        is_aligned(input, alignof(__half2))) {
        const int64_t size2 = size / 2;
        const int num_blocks = (size2 + block_size - 1) / block_size;
        silu_kernel_f16_half2<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<__half2*>(output),
            reinterpret_cast<const __half2*>(input),
            size2
        );
        return;
    }
    const int num_blocks = (size + block_size - 1) / block_size;
    silu_kernel_f16_scalar<<<num_blocks, block_size, 0, stream>>>(output, input, size);
}

__global__ void silu_kernel_bf16_scalar(__nv_bfloat16* output, const __nv_bfloat16* input, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __bfloat162float(input[idx]);
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = __float2bfloat16_rn(x * sigmoid);
    }
}

__global__ void silu_kernel_bf16_bf162(__nv_bfloat162* output, const __nv_bfloat162* input, int64_t size2) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size2) {
        const float2 x = __bfloat1622float2(input[idx]);
        const float2 sig = make_float2(
            1.0f / (1.0f + expf(-x.x)),
            1.0f / (1.0f + expf(-x.y))
        );
        output[idx] = __floats2bfloat162_rn(x.x * sig.x, x.y * sig.y);
    }
}

void silu_bf16(__nv_bfloat16* output, const __nv_bfloat16* input, int64_t size, cudaStream_t stream) {
    const int block_size = 256;
    if (size > 0 && (size % 2 == 0) && is_aligned(output, alignof(__nv_bfloat162)) &&
        is_aligned(input, alignof(__nv_bfloat162))) {
        const int64_t size2 = size / 2;
        const int num_blocks = (size2 + block_size - 1) / block_size;
        silu_kernel_bf16_bf162<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(output),
            reinterpret_cast<const __nv_bfloat162*>(input),
            size2
        );
        return;
    }
    const int num_blocks = (size + block_size - 1) / block_size;
    silu_kernel_bf16_scalar<<<num_blocks, block_size, 0, stream>>>(output, input, size);
}

// =============================================================================
// Fused SiLU * Mul (SwiGLU helper)
// =============================================================================

__global__ void silu_mul_fused_kernel_f16_scalar(half* gate_inout, const half* up, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float x = __half2float(gate_inout[idx]);
        const float u = __half2float(up[idx]);
        const float sig = 1.0f / (1.0f + expf(-x));
        const float silu_f = x * sig;
        const __half silu_h = __float2half(silu_f);          // match non-fused path rounding
        const float silu_q = __half2float(silu_h);
        gate_inout[idx] = __float2half(silu_q * u);
    }
}

__global__ void silu_mul_fused_kernel_f16_half2(__half2* gate_inout, const __half2* up, int64_t size2) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size2) {
        const float2 x = __half22float2(gate_inout[idx]);
        const float2 u = __half22float2(up[idx]);
        const float2 sig = make_float2(
            1.0f / (1.0f + expf(-x.x)),
            1.0f / (1.0f + expf(-x.y))
        );
        const float2 silu_f = make_float2(x.x * sig.x, x.y * sig.y);
        const __half2 silu_h2 = __floats2half2_rn(silu_f.x, silu_f.y);  // match non-fused path rounding
        const float2 silu_q = __half22float2(silu_h2);
        gate_inout[idx] = __floats2half2_rn(silu_q.x * u.x, silu_q.y * u.y);
    }
}

void silu_mul_fused_f16(half* gate_inout, const half* up, int64_t size, cudaStream_t stream) {
    const int block_size = 256;
    if (size > 0 && (size % 2 == 0) && is_aligned(gate_inout, alignof(__half2)) &&
        is_aligned(up, alignof(__half2))) {
        const int64_t size2 = size / 2;
        const int num_blocks = (size2 + block_size - 1) / block_size;
        silu_mul_fused_kernel_f16_half2<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<__half2*>(gate_inout),
            reinterpret_cast<const __half2*>(up),
            size2
        );
        return;
    }
    const int num_blocks = (size + block_size - 1) / block_size;
    silu_mul_fused_kernel_f16_scalar<<<num_blocks, block_size, 0, stream>>>(gate_inout, up, size);
}

__global__ void silu_mul_fused_kernel_bf16_scalar(__nv_bfloat16* gate_inout, const __nv_bfloat16* up, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        const float x = __bfloat162float(gate_inout[idx]);
        const float u = __bfloat162float(up[idx]);
        const float sig = 1.0f / (1.0f + expf(-x));
        const float silu_f = x * sig;
        const __nv_bfloat16 silu_h = __float2bfloat16_rn(silu_f);  // match non-fused path rounding
        const float silu_q = __bfloat162float(silu_h);
        gate_inout[idx] = __float2bfloat16_rn(silu_q * u);
    }
}

__global__ void silu_mul_fused_kernel_bf16_bf162(
    __nv_bfloat162* gate_inout, const __nv_bfloat162* up, int64_t size2
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size2) {
        const float2 x = __bfloat1622float2(gate_inout[idx]);
        const float2 u = __bfloat1622float2(up[idx]);
        const float2 sig = make_float2(
            1.0f / (1.0f + expf(-x.x)),
            1.0f / (1.0f + expf(-x.y))
        );
        const float2 silu_f = make_float2(x.x * sig.x, x.y * sig.y);
        const __nv_bfloat162 silu_h2 = __floats2bfloat162_rn(silu_f.x, silu_f.y);  // match rounding
        const float2 silu_q = __bfloat1622float2(silu_h2);
        gate_inout[idx] = __floats2bfloat162_rn(silu_q.x * u.x, silu_q.y * u.y);
    }
}

void silu_mul_fused_bf16(__nv_bfloat16* gate_inout, const __nv_bfloat16* up, int64_t size, cudaStream_t stream) {
    const int block_size = 256;
    if (size > 0 && (size % 2 == 0) && is_aligned(gate_inout, alignof(__nv_bfloat162)) &&
        is_aligned(up, alignof(__nv_bfloat162))) {
        const int64_t size2 = size / 2;
        const int num_blocks = (size2 + block_size - 1) / block_size;
        silu_mul_fused_kernel_bf16_bf162<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(gate_inout),
            reinterpret_cast<const __nv_bfloat162*>(up),
            size2
        );
        return;
    }
    const int num_blocks = (size + block_size - 1) / block_size;
    silu_mul_fused_kernel_bf16_scalar<<<num_blocks, block_size, 0, stream>>>(gate_inout, up, size);
}

// =============================================================================
// Element-wise Multiply
// =============================================================================

__global__ void elementwise_mul_kernel_f16_scalar(
    half* output, const half* a, const half* b, int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(va * vb);
    }
}

__global__ void elementwise_mul_kernel_f16_half2(
    __half2* output, const __half2* a, const __half2* b, int64_t size2
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size2) {
        const float2 va = __half22float2(a[idx]);
        const float2 vb = __half22float2(b[idx]);
        output[idx] = __floats2half2_rn(va.x * vb.x, va.y * vb.y);
    }
}

void elementwise_mul_f16(
    half* output, const half* a, const half* b, int64_t size, cudaStream_t stream
) {
    const int block_size = 256;
    if (size > 0 && (size % 2 == 0) && is_aligned(output, alignof(__half2)) && is_aligned(a, alignof(__half2)) &&
        is_aligned(b, alignof(__half2))) {
        const int64_t size2 = size / 2;
        const int num_blocks = (size2 + block_size - 1) / block_size;
        elementwise_mul_kernel_f16_half2<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<__half2*>(output),
            reinterpret_cast<const __half2*>(a),
            reinterpret_cast<const __half2*>(b),
            size2
        );
        return;
    }
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_mul_kernel_f16_scalar<<<num_blocks, block_size, 0, stream>>>(output, a, b, size);
}

__global__ void elementwise_mul_kernel_bf16_scalar(
    __nv_bfloat16* output, const __nv_bfloat16* a, const __nv_bfloat16* b, int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        output[idx] = __float2bfloat16_rn(va * vb);
    }
}

__global__ void elementwise_mul_kernel_bf16_bf162(
    __nv_bfloat162* output, const __nv_bfloat162* a, const __nv_bfloat162* b, int64_t size2
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size2) {
        const float2 va = __bfloat1622float2(a[idx]);
        const float2 vb = __bfloat1622float2(b[idx]);
        output[idx] = __floats2bfloat162_rn(va.x * vb.x, va.y * vb.y);
    }
}

void elementwise_mul_bf16(
    __nv_bfloat16* output, const __nv_bfloat16* a, const __nv_bfloat16* b, int64_t size, cudaStream_t stream
) {
    const int block_size = 256;
    if (size > 0 && (size % 2 == 0) && is_aligned(output, alignof(__nv_bfloat162)) &&
        is_aligned(a, alignof(__nv_bfloat162)) && is_aligned(b, alignof(__nv_bfloat162))) {
        const int64_t size2 = size / 2;
        const int num_blocks = (size2 + block_size - 1) / block_size;
        elementwise_mul_kernel_bf16_bf162<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(output),
            reinterpret_cast<const __nv_bfloat162*>(a),
            reinterpret_cast<const __nv_bfloat162*>(b),
            size2
        );
        return;
    }
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_mul_kernel_bf16_scalar<<<num_blocks, block_size, 0, stream>>>(output, a, b, size);
}

// =============================================================================
// Element-wise Add
// =============================================================================

__global__ void elementwise_add_kernel_f16_scalar(
    half* output, const half* a, const half* b, int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(va + vb);
    }
}

__global__ void elementwise_add_kernel_f16_half2(
    __half2* output, const __half2* a, const __half2* b, int64_t size2
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size2) {
        const float2 va = __half22float2(a[idx]);
        const float2 vb = __half22float2(b[idx]);
        output[idx] = __floats2half2_rn(va.x + vb.x, va.y + vb.y);
    }
}

void elementwise_add_f16(
    half* output, const half* a, const half* b, int64_t size, cudaStream_t stream
) {
    const int block_size = 256;
    if (size > 0 && (size % 2 == 0) && is_aligned(output, alignof(__half2)) && is_aligned(a, alignof(__half2)) &&
        is_aligned(b, alignof(__half2))) {
        const int64_t size2 = size / 2;
        const int num_blocks = (size2 + block_size - 1) / block_size;
        elementwise_add_kernel_f16_half2<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<__half2*>(output),
            reinterpret_cast<const __half2*>(a),
            reinterpret_cast<const __half2*>(b),
            size2
        );
        return;
    }
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel_f16_scalar<<<num_blocks, block_size, 0, stream>>>(output, a, b, size);
}

__global__ void elementwise_add_kernel_bf16_scalar(
    __nv_bfloat16* output, const __nv_bfloat16* a, const __nv_bfloat16* b, int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        output[idx] = __float2bfloat16_rn(va + vb);
    }
}

__global__ void elementwise_add_kernel_bf16_bf162(
    __nv_bfloat162* output, const __nv_bfloat162* a, const __nv_bfloat162* b, int64_t size2
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size2) {
        const float2 va = __bfloat1622float2(a[idx]);
        const float2 vb = __bfloat1622float2(b[idx]);
        output[idx] = __floats2bfloat162_rn(va.x + vb.x, va.y + vb.y);
    }
}

void elementwise_add_bf16(
    __nv_bfloat16* output, const __nv_bfloat16* a, const __nv_bfloat16* b, int64_t size, cudaStream_t stream
) {
    const int block_size = 256;
    if (size > 0 && (size % 2 == 0) && is_aligned(output, alignof(__nv_bfloat162)) &&
        is_aligned(a, alignof(__nv_bfloat162)) && is_aligned(b, alignof(__nv_bfloat162))) {
        const int64_t size2 = size / 2;
        const int num_blocks = (size2 + block_size - 1) / block_size;
        elementwise_add_kernel_bf16_bf162<<<num_blocks, block_size, 0, stream>>>(
            reinterpret_cast<__nv_bfloat162*>(output),
            reinterpret_cast<const __nv_bfloat162*>(a),
            reinterpret_cast<const __nv_bfloat162*>(b),
            size2
        );
        return;
    }
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel_bf16_scalar<<<num_blocks, block_size, 0, stream>>>(output, a, b, size);
}

// =============================================================================
// Embedding Lookup
// =============================================================================

__global__ void embedding_lookup_kernel_f16(
    half* output,
    const half* embedding,
    const int* input_ids,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    // 每个 block 处理一个 token
    const int batch = blockIdx.y;
    const int seq = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int token_id = input_ids[batch * seq_len + seq];
    const half* emb_row = embedding + token_id * hidden_size;
    half* out_row = output + (batch * seq_len + seq) * hidden_size;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out_row[i] = emb_row[i];
    }
}

void embedding_lookup_f16(
    half* output,
    const half* embedding,
    const int* input_ids,
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream
) {
    dim3 grid(seq_len, batch_size);
    dim3 block(min(hidden_size, 256));
    
    embedding_lookup_kernel_f16<<<grid, block, 0, stream>>>(
        output, embedding, input_ids, batch_size, seq_len, hidden_size);
}

__global__ void embedding_lookup_kernel_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* embedding,
    const int* input_ids,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    const int batch = blockIdx.y;
    const int seq = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int token_id = input_ids[batch * seq_len + seq];
    const __nv_bfloat16* emb_row = embedding + token_id * hidden_size;
    __nv_bfloat16* out_row = output + (batch * seq_len + seq) * hidden_size;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out_row[i] = emb_row[i];
    }
}

void embedding_lookup_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* embedding,
    const int* input_ids,
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream
) {
    dim3 grid(seq_len, batch_size);
    dim3 block(min(hidden_size, 256));
    
    embedding_lookup_kernel_bf16<<<grid, block, 0, stream>>>(
        output, embedding, input_ids, batch_size, seq_len, hidden_size);
}

// =============================================================================
// Copy Last Hidden State
// =============================================================================

__global__ void copy_last_hidden_kernel_f16(
    half* output,
    const half* input,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    const int batch = blockIdx.x;
    const int tid = threadIdx.x;
    
    const half* last_hidden = input + (batch * seq_len + seq_len - 1) * hidden_size;
    half* out_row = output + batch * hidden_size;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out_row[i] = last_hidden[i];
    }
}

void copy_last_hidden_f16(
    half* output,
    const half* input,
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(min(hidden_size, 256));
    
    copy_last_hidden_kernel_f16<<<grid, block, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_size);
}

__global__ void copy_last_hidden_kernel_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int batch_size,
    int seq_len,
    int hidden_size
) {
    const int batch = blockIdx.x;
    const int tid = threadIdx.x;
    
    const __nv_bfloat16* last_hidden = input + (batch * seq_len + seq_len - 1) * hidden_size;
    __nv_bfloat16* out_row = output + batch * hidden_size;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        out_row[i] = last_hidden[i];
    }
}

void copy_last_hidden_bf16(
    __nv_bfloat16* output,
    const __nv_bfloat16* input,
    int batch_size,
    int seq_len,
    int hidden_size,
    cudaStream_t stream
) {
    dim3 grid(batch_size);
    dim3 block(min(hidden_size, 256));
    
    copy_last_hidden_kernel_bf16<<<grid, block, 0, stream>>>(
        output, input, batch_size, seq_len, hidden_size);
}

// =============================================================================
// Type Conversion
// =============================================================================

__global__ void convert_f32_to_f16_kernel(half* output, const float* input, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

__global__ void convert_f16_to_f32_kernel(float* output, const half* input, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(input[idx]);
    }
}

void convert_f32_to_f16(half* output, const float* input, int64_t size, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    convert_f32_to_f16_kernel<<<num_blocks, block_size, 0, stream>>>(output, input, size);
}

void convert_f16_to_f32(float* output, const half* input, int64_t size, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    convert_f16_to_f32_kernel<<<num_blocks, block_size, 0, stream>>>(output, input, size);
}

__global__ void convert_f32_to_bf16_kernel(__nv_bfloat16* output, const float* input, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2bfloat16_rn(input[idx]);
    }
}

__global__ void convert_bf16_to_f32_kernel(float* output, const __nv_bfloat16* input, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __bfloat162float(input[idx]);
    }
}

void convert_f32_to_bf16(__nv_bfloat16* output, const float* input, int64_t size, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    convert_f32_to_bf16_kernel<<<num_blocks, block_size, 0, stream>>>(output, input, size);
}

void convert_bf16_to_f32(float* output, const __nv_bfloat16* input, int64_t size, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    convert_bf16_to_f32_kernel<<<num_blocks, block_size, 0, stream>>>(output, input, size);
}

}  // namespace kernels
}  // namespace cuda
}  // namespace ember

````

### File: backends/cuda/cuda_runtime.cpp (MLP compute block (forward_layer))

````cpp
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
            kernels::elementwise_mul_bf16(
                static_cast<__nv_bfloat16*>(act.mlp_gate),
                static_cast<const __nv_bfloat16*>(act.mlp_gate),
                static_cast<const __nv_bfloat16*>(act.mlp_up),
                batch_size * seq_len * intermediate_size,
                stream
            );
        } else {
            kernels::elementwise_mul_f16(
                static_cast<half*>(act.mlp_gate),
                static_cast<const half*>(act.mlp_gate),
                static_cast<const half*>(act.mlp_up),
                batch_size * seq_len * intermediate_size,
                stream
            );
        }

        err = dump_last_row(session.runtime_config().dump_dir,
                            "layer_" + std::to_string(layer_idx) + "_mlp_mul",
                            device_id, act.mlp_gate,
                            seq_len, intermediate_size, compute_dtype, stream);
        if (err) return err;
    } else {
        // Fast path: fuse SiLU + mul (SwiGLU) to cut one launch and one global write+read.
        if (compute_dtype == DType::BF16) {
            kernels::silu_mul_fused_bf16(
                static_cast<__nv_bfloat16*>(act.mlp_gate),
                static_cast<const __nv_bfloat16*>(act.mlp_up),
                batch_size * seq_len * intermediate_size,
                stream
            );
        } else {
            kernels::silu_mul_fused_f16(
                static_cast<half*>(act.mlp_gate),
                static_cast<const half*>(act.mlp_up),
                batch_size * seq_len * intermediate_size,
                stream
            );
        }
    }
    
    // Down projection: [M, intermediate] @ [hidden, intermediate]^T = [M, hidden]
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        hidden_size, M, intermediate_size,
        &alpha_one,
        layer.down_proj_weight, cuda_dtype, intermediate_size,
        act.mlp_gate, cuda_dtype, intermediate_size,
        &beta_zero,
        act.mlp_down, cuda_dtype, hidden_size,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    if (session.runtime_config().check_correctness &&
        session.runtime_config().dump_layer == layer_idx) {
        Error err = dump_last_row(session.runtime_config().dump_dir,
                                     "layer_" + std::to_string(layer_idx) + "_mlp_out",
                                     device_id, act.mlp_down,
                                     seq_len, hidden_size, compute_dtype, stream);
        if (err) return err;
    }
    
    // =====================================================================
    // Residual Connection (MLP)
    // =====================================================================
    if (compute_dtype == DType::BF16) {
        kernels::elementwise_add_bf16(
            static_cast<__nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(act.mlp_down),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            batch_size * seq_len * hidden_size,
            stream
        );
    } else {
        kernels::elementwise_add_f16(
            static_cast<half*>(act.hidden_states),
            static_cast<const half*>(act.mlp_down),
            static_cast<const half*>(act.hidden_states),
            batch_size * seq_len * hidden_size,
            stream
        );
    }

    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.ffn_end, stream));
    }
    
    // Check mode: dump last token hidden state after layer
    if (session.runtime_config().check_correctness) {
        int target = session.runtime_config().dump_layer;
        if (target < 0 || target == layer_idx) {
            std::string name = "layer_" + std::to_string(layer_idx) + "_last_hidden";
            Error err = dump_last_row(session.runtime_config().dump_dir, name, device_id,
                                         act.hidden_states, seq_len, hidden_size, compute_dtype, stream);
            if (err) return err;
        }
    }

    if (profile_layers_ && static_cast<size_t>(layer_idx) < last_layer_profile_ms_.size()) {
        auto& ev = profile_events_[device_id];
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

````

---

## 3) 报告上下文（完整）

### Report: reports/stage31_base_operator_spotcheck_4b_20260225_mainline/stage31_base_operator_spotcheck.csv

````csv
layer,layer_max_abs_diff,layer_mean_abs_diff,attn_out_max_abs_diff,attn_residual_max_abs_diff,post_attn_norm_max_abs_diff,mlp_out_max_abs_diff,gate_proj_max_abs_diff,up_proj_max_abs_diff,gate_proj_ember_norm_max_abs_diff,up_proj_ember_norm_max_abs_diff
0,0.205282,0.017445,0.147710,0.149907,0.113919,0.370814,0.957237,0.715941,0.003892,0.001909
1,0.427129,0.031933,0.180284,0.364644,4.198868,0.287741,6.213398,5.650882,0.031280,0.030877

````

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

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`
