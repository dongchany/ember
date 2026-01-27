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
