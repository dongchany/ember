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
