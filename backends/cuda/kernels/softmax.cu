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
