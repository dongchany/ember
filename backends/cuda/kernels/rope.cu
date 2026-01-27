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
