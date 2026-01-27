#include "kernels.h"

namespace ember {
namespace cuda {
namespace kernels {

// =============================================================================
// SiLU (Swish) Activation
// =============================================================================

__global__ void silu_kernel_f16(half* output, const half* input, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __half2float(input[idx]);
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = __float2half(x * sigmoid);
    }
}

void silu_f16(half* output, const half* input, int64_t size, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    silu_kernel_f16<<<num_blocks, block_size, 0, stream>>>(output, input, size);
}

__global__ void silu_kernel_bf16(__nv_bfloat16* output, const __nv_bfloat16* input, int64_t size) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __bfloat162float(input[idx]);
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[idx] = __float2bfloat16_rn(x * sigmoid);
    }
}

void silu_bf16(__nv_bfloat16* output, const __nv_bfloat16* input, int64_t size, cudaStream_t stream) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    silu_kernel_bf16<<<num_blocks, block_size, 0, stream>>>(output, input, size);
}

// =============================================================================
// Element-wise Multiply
// =============================================================================

__global__ void elementwise_mul_kernel_f16(
    half* output, const half* a, const half* b, int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(va * vb);
    }
}

void elementwise_mul_f16(
    half* output, const half* a, const half* b, int64_t size, cudaStream_t stream
) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_mul_kernel_f16<<<num_blocks, block_size, 0, stream>>>(output, a, b, size);
}

__global__ void elementwise_mul_kernel_bf16(
    __nv_bfloat16* output, const __nv_bfloat16* a, const __nv_bfloat16* b, int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        output[idx] = __float2bfloat16_rn(va * vb);
    }
}

void elementwise_mul_bf16(
    __nv_bfloat16* output, const __nv_bfloat16* a, const __nv_bfloat16* b, int64_t size, cudaStream_t stream
) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_mul_kernel_bf16<<<num_blocks, block_size, 0, stream>>>(output, a, b, size);
}

// =============================================================================
// Element-wise Add
// =============================================================================

__global__ void elementwise_add_kernel_f16(
    half* output, const half* a, const half* b, int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(va + vb);
    }
}

void elementwise_add_f16(
    half* output, const half* a, const half* b, int64_t size, cudaStream_t stream
) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel_f16<<<num_blocks, block_size, 0, stream>>>(output, a, b, size);
}

__global__ void elementwise_add_kernel_bf16(
    __nv_bfloat16* output, const __nv_bfloat16* a, const __nv_bfloat16* b, int64_t size
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        output[idx] = __float2bfloat16_rn(va + vb);
    }
}

void elementwise_add_bf16(
    __nv_bfloat16* output, const __nv_bfloat16* a, const __nv_bfloat16* b, int64_t size, cudaStream_t stream
) {
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    elementwise_add_kernel_bf16<<<num_blocks, block_size, 0, stream>>>(output, a, b, size);
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
