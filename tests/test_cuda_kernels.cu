#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "backends/cuda/kernels/kernels.h"

namespace {

bool cuda_check(cudaError_t err, const char* msg) {
    if (err == cudaSuccess) {
        return true;
    }
    std::cerr << "[FAIL] " << msg << ": " << cudaGetErrorString(err) << "\n";
    return false;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a[i] - b[i]));
    }
    return max_diff;
}

void compute_attention_expected(std::vector<float>& out,
                                const std::vector<float>& q,
                                const std::vector<float>& k,
                                const std::vector<float>& v,
                                int batch,
                                int seq_q,
                                int seq_k,
                                int max_seq,
                                int num_heads,
                                int num_kv_heads,
                                int head_dim,
                                int start_pos,
                                float scale) {
    const int heads_per_kv = num_heads / num_kv_heads;
    out.assign(static_cast<size_t>(batch * seq_q * num_heads * head_dim), 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int qh = 0; qh < num_heads; ++qh) {
            int kv_head = qh / heads_per_kv;
            const float* k_ptr = k.data() + (b * num_kv_heads + kv_head) * max_seq * head_dim;
            const float* v_ptr = v.data() + (b * num_kv_heads + kv_head) * max_seq * head_dim;
            for (int sq = 0; sq < seq_q; ++sq) {
                std::vector<float> scores(seq_k, 0.0f);
                float max_val = -1e20f;
                for (int kk = 0; kk < seq_k; ++kk) {
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        float qv = q[((b * seq_q + sq) * num_heads + qh) * head_dim + d];
                        float kv = k_ptr[kk * head_dim + d];
                        acc += qv * kv;
                    }
                    float score = acc * scale;
                    if (seq_q > 1 || start_pos > 0) {
                        int max_k = start_pos + sq;
                        if (kk > max_k) score = -1e4f;
                    }
                    scores[kk] = score;
                    max_val = std::max(max_val, score);
                }

                float sum = 0.0f;
                for (int kk = 0; kk < seq_k; ++kk) {
                    sum += std::exp(scores[kk] - max_val);
                }

                std::vector<float> probs(seq_k, 0.0f);
                for (int kk = 0; kk < seq_k; ++kk) {
                    float prob = std::exp(scores[kk] - max_val) / (sum + 1e-6f);
                    prob = __half2float(__float2half(prob));
                    probs[kk] = prob;
                }

                for (int d = 0; d < head_dim; ++d) {
                    float acc = 0.0f;
                    for (int kk = 0; kk < seq_k; ++kk) {
                        float vv = v_ptr[kk * head_dim + d];
                        acc += probs[kk] * vv;
                    }
                    out[((b * seq_q + sq) * num_heads + qh) * head_dim + d] = acc;
                }
            }
        }
    }
}

bool test_update_kv_cache() {
    const int batch = 1;
    const int seq_len = 2;
    const int num_kv_heads = 1;
    const int head_dim = 4;
    const int start_pos = 1;
    const int max_seq = 4;
    const int cache_size = batch * num_kv_heads * max_seq * head_dim;
    const int new_size = batch * seq_len * num_kv_heads * head_dim;

    std::vector<float> k_new_f(new_size);
    std::vector<float> v_new_f(new_size);
    for (int i = 0; i < new_size; ++i) {
        k_new_f[i] = 0.1f * static_cast<float>(i + 1);
        v_new_f[i] = -0.2f * static_cast<float>(i + 1);
    }

    std::vector<float> expected_k(cache_size, -1.0f);
    std::vector<float> expected_v(cache_size, -1.0f);
    for (int s = 0; s < seq_len; ++s) {
        int cache_pos = start_pos + s;
        for (int d = 0; d < head_dim; ++d) {
            int dst_idx = cache_pos * head_dim + d;
            int src_idx = s * head_dim + d;
            expected_k[dst_idx] = k_new_f[src_idx];
            expected_v[dst_idx] = v_new_f[src_idx];
        }
    }

    std::vector<half> k_cache_h(cache_size, __float2half(-1.0f));
    std::vector<half> v_cache_h(cache_size, __float2half(-1.0f));
    std::vector<half> k_new_h(new_size);
    std::vector<half> v_new_h(new_size);
    for (int i = 0; i < new_size; ++i) {
        k_new_h[i] = __float2half(k_new_f[i]);
        v_new_h[i] = __float2half(v_new_f[i]);
    }

    half* d_k_cache = nullptr;
    half* d_v_cache = nullptr;
    half* d_k_new = nullptr;
    half* d_v_new = nullptr;
    if (!cuda_check(cudaMalloc(&d_k_cache, cache_size * sizeof(half)), "cudaMalloc k_cache")) return false;
    if (!cuda_check(cudaMalloc(&d_v_cache, cache_size * sizeof(half)), "cudaMalloc v_cache")) return false;
    if (!cuda_check(cudaMalloc(&d_k_new, new_size * sizeof(half)), "cudaMalloc k_new")) return false;
    if (!cuda_check(cudaMalloc(&d_v_new, new_size * sizeof(half)), "cudaMalloc v_new")) return false;
    cuda_check(cudaMemcpy(d_k_cache, k_cache_h.data(), cache_size * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy k_cache");
    cuda_check(cudaMemcpy(d_v_cache, v_cache_h.data(), cache_size * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy v_cache");
    cuda_check(cudaMemcpy(d_k_new, k_new_h.data(), new_size * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy k_new");
    cuda_check(cudaMemcpy(d_v_new, v_new_h.data(), new_size * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy v_new");

    ember::cuda::kernels::update_kv_cache_f16(
        d_k_cache, d_v_cache, d_k_new, d_v_new,
        batch, seq_len, num_kv_heads, head_dim, start_pos, max_seq, nullptr);
    if (!cuda_check(cudaGetLastError(), "update_kv_cache_f16 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "update_kv_cache_f16 sync")) return false;

    cuda_check(cudaMemcpy(k_cache_h.data(), d_k_cache, cache_size * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy k_cache out");
    cuda_check(cudaMemcpy(v_cache_h.data(), d_v_cache, cache_size * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy v_cache out");

    cudaFree(d_k_cache);
    cudaFree(d_v_cache);
    cudaFree(d_k_new);
    cudaFree(d_v_new);

    std::vector<float> k_out(cache_size);
    std::vector<float> v_out(cache_size);
    for (int i = 0; i < cache_size; ++i) {
        k_out[i] = __half2float(k_cache_h[i]);
        v_out[i] = __half2float(v_cache_h[i]);
    }

    float diff_k = max_abs_diff(k_out, expected_k);
    float diff_v = max_abs_diff(v_out, expected_v);
    float diff = std::max(diff_k, diff_v);
    if (diff > 1e-2f) {
        std::cerr << "[FAIL] update_kv_cache max_abs_diff=" << diff << "\n";
        return false;
    }
    std::cout << "[PASS] update_kv_cache\n";
    return true;
}

bool test_attention() {
    const int batch = 1;
    const int seq_q = 2;
    const int seq_k1 = 3;
    const int seq_k2 = 4;
    const int max_seq = 5;
    const int num_heads = 2;
    const int num_kv_heads = 1;
    const int head_dim = 4;
    const int heads_per_kv = num_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const int q_size = batch * seq_q * num_heads * head_dim;
    const int kv_size = batch * num_kv_heads * max_seq * head_dim;
    std::vector<float> q_f(q_size);
    std::vector<float> k_f(kv_size, 0.0f);
    std::vector<float> v_f(kv_size, 0.0f);

    for (int i = 0; i < q_size; ++i) {
        q_f[i] = 0.02f * static_cast<float>(i + 1);
    }
    for (int i = 0; i < seq_k2 * head_dim; ++i) {
        k_f[i] = 0.03f * static_cast<float>(i + 1);
        v_f[i] = -0.04f * static_cast<float>(i + 1);
    }

    std::vector<float> expected1;
    std::vector<float> expected2;
    compute_attention_expected(expected1, q_f, k_f, v_f, batch, seq_q, seq_k1, max_seq,
                               num_heads, num_kv_heads, head_dim, 0, scale);
    compute_attention_expected(expected2, q_f, k_f, v_f, batch, seq_q, seq_k2, max_seq,
                               num_heads, num_kv_heads, head_dim, 2, scale);

    std::vector<half> q_h(q_size);
    std::vector<half> k_h(kv_size, __float2half(0.0f));
    std::vector<half> v_h(kv_size, __float2half(0.0f));
    for (int i = 0; i < q_size; ++i) {
        q_h[i] = __float2half(q_f[i]);
    }
    for (int i = 0; i < seq_k2 * head_dim; ++i) {
        k_h[i] = __float2half(k_f[i]);
        v_h[i] = __float2half(v_f[i]);
    }

    half* d_q = nullptr;
    half* d_k = nullptr;
    half* d_v = nullptr;
    half* d_out = nullptr;
    float* d_workspace = nullptr;
    half* d_probs = nullptr;
    if (!cuda_check(cudaMalloc(&d_q, q_size * sizeof(half)), "cudaMalloc q")) return false;
    if (!cuda_check(cudaMalloc(&d_k, kv_size * sizeof(half)), "cudaMalloc k")) return false;
    if (!cuda_check(cudaMalloc(&d_v, kv_size * sizeof(half)), "cudaMalloc v")) return false;
    if (!cuda_check(cudaMalloc(&d_out, q_size * sizeof(half)), "cudaMalloc out")) return false;
    if (!cuda_check(cudaMalloc(&d_workspace, batch * num_heads * seq_q * seq_k2 * sizeof(float)),
                    "cudaMalloc workspace")) return false;
    if (!cuda_check(cudaMalloc(&d_probs, batch * num_heads * seq_q * seq_k2 * sizeof(half)),
                    "cudaMalloc probs")) return false;

    cuda_check(cudaMemcpy(d_q, q_h.data(), q_size * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy q");
    cuda_check(cudaMemcpy(d_k, k_h.data(), kv_size * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy k");
    cuda_check(cudaMemcpy(d_v, v_h.data(), kv_size * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy v");

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "[FAIL] cublasCreate failed\n";
        return false;
    }

    ember::cuda::kernels::attention_f16(
        d_out, d_q, d_k, d_v, d_workspace, d_probs,
        batch, seq_q, seq_k1, max_seq, 0,
        num_heads, num_kv_heads, head_dim, scale, handle, nullptr);
    bool ok = cuda_check(cudaGetLastError(), "attention_f16 launch (case1)") &&
              cuda_check(cudaDeviceSynchronize(), "attention_f16 sync (case1)");

    std::vector<half> out_h(q_size);
    cuda_check(cudaMemcpy(out_h.data(), d_out, q_size * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy out");
    std::vector<float> out_f(q_size);
    for (int i = 0; i < q_size; ++i) {
        out_f[i] = __half2float(out_h[i]);
    }

    float diff1 = max_abs_diff(out_f, expected1);
    if (diff1 > 1e-2f) {
        std::cerr << "[FAIL] attention(case1) max_abs_diff=" << diff1 << "\n";
        return false;
    }

    ember::cuda::kernels::attention_f16(
        d_out, d_q, d_k, d_v, d_workspace, d_probs,
        batch, seq_q, seq_k2, max_seq, 2,
        num_heads, num_kv_heads, head_dim, scale, handle, nullptr);
    ok = cuda_check(cudaGetLastError(), "attention_f16 launch (case2)") &&
         cuda_check(cudaDeviceSynchronize(), "attention_f16 sync (case2)");

    cublasDestroy(handle);
    if (!ok) return false;

    cuda_check(cudaMemcpy(out_h.data(), d_out, q_size * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy out case2");
    for (int i = 0; i < q_size; ++i) {
        out_f[i] = __half2float(out_h[i]);
    }

    float diff2 = max_abs_diff(out_f, expected2);
    if (diff2 > 1e-2f) {
        std::cerr << "[FAIL] attention(case2) max_abs_diff=" << diff2 << "\n";
        return false;
    }

    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_out);
    cudaFree(d_workspace);
    cudaFree(d_probs);

    std::cout << "[PASS] attention\n";
    return true;
}

bool test_silu() {
    const int size = 32;
    std::vector<float> input_f(size);
    for (int i = 0; i < size; ++i) {
        input_f[i] = 0.2f * static_cast<float>(i - 10);
    }
    std::vector<float> expected(size);
    for (int i = 0; i < size; ++i) {
        float x = input_f[i];
        float sig = 1.0f / (1.0f + std::exp(-x));
        expected[i] = x * sig;
    }

    std::vector<half> input_h(size);
    for (int i = 0; i < size; ++i) {
        input_h[i] = __float2half(input_f[i]);
    }

    half* d_input = nullptr;
    half* d_output = nullptr;
    if (!cuda_check(cudaMalloc(&d_input, input_h.size() * sizeof(half)), "cudaMalloc input")) return false;
    if (!cuda_check(cudaMalloc(&d_output, input_h.size() * sizeof(half)), "cudaMalloc output")) return false;
    cuda_check(cudaMemcpy(d_input, input_h.data(), input_h.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy input");

    ember::cuda::kernels::silu_f16(d_output, d_input, size, nullptr);
    if (!cuda_check(cudaGetLastError(), "silu_f16 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "silu_f16 sync")) return false;

    std::vector<half> output_h(size);
    cuda_check(cudaMemcpy(output_h.data(), d_output, output_h.size() * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy output");

    std::vector<float> output_f(size);
    for (int i = 0; i < size; ++i) {
        output_f[i] = __half2float(output_h[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    float diff = max_abs_diff(output_f, expected);
    if (diff > 1e-2f) {
        std::cerr << "[FAIL] silu max_abs_diff=" << diff << "\n";
        return false;
    }
    std::cout << "[PASS] silu\n";
    return true;
}

bool test_elementwise() {
    const int size = 64;
    std::vector<float> a_f(size);
    std::vector<float> b_f(size);
    for (int i = 0; i < size; ++i) {
        a_f[i] = 0.1f * static_cast<float>(i - 20);
        b_f[i] = -0.05f * static_cast<float>(i - 15);
    }

    std::vector<float> expected_mul(size);
    std::vector<float> expected_add(size);
    for (int i = 0; i < size; ++i) {
        expected_mul[i] = a_f[i] * b_f[i];
        expected_add[i] = a_f[i] + b_f[i];
    }

    std::vector<half> a_h(size);
    std::vector<half> b_h(size);
    for (int i = 0; i < size; ++i) {
        a_h[i] = __float2half(a_f[i]);
        b_h[i] = __float2half(b_f[i]);
    }

    half* d_a = nullptr;
    half* d_b = nullptr;
    half* d_out = nullptr;
    if (!cuda_check(cudaMalloc(&d_a, size * sizeof(half)), "cudaMalloc a")) return false;
    if (!cuda_check(cudaMalloc(&d_b, size * sizeof(half)), "cudaMalloc b")) return false;
    if (!cuda_check(cudaMalloc(&d_out, size * sizeof(half)), "cudaMalloc out")) return false;
    cuda_check(cudaMemcpy(d_a, a_h.data(), size * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy a");
    cuda_check(cudaMemcpy(d_b, b_h.data(), size * sizeof(half), cudaMemcpyHostToDevice), "cudaMemcpy b");

    ember::cuda::kernels::elementwise_mul_f16(d_out, d_a, d_b, size, nullptr);
    if (!cuda_check(cudaGetLastError(), "elementwise_mul_f16 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "elementwise_mul_f16 sync")) return false;

    std::vector<half> out_mul_h(size);
    cuda_check(cudaMemcpy(out_mul_h.data(), d_out, size * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy out mul");

    ember::cuda::kernels::elementwise_add_f16(d_out, d_a, d_b, size, nullptr);
    if (!cuda_check(cudaGetLastError(), "elementwise_add_f16 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "elementwise_add_f16 sync")) return false;

    std::vector<half> out_add_h(size);
    cuda_check(cudaMemcpy(out_add_h.data(), d_out, size * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy out add");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    std::vector<float> out_mul_f(size);
    std::vector<float> out_add_f(size);
    for (int i = 0; i < size; ++i) {
        out_mul_f[i] = __half2float(out_mul_h[i]);
        out_add_f[i] = __half2float(out_add_h[i]);
    }

    float diff_mul = max_abs_diff(out_mul_f, expected_mul);
    float diff_add = max_abs_diff(out_add_f, expected_add);
    float diff = std::max(diff_mul, diff_add);
    if (diff > 1e-2f) {
        std::cerr << "[FAIL] elementwise max_abs_diff=" << diff << "\n";
        return false;
    }
    std::cout << "[PASS] elementwise\n";
    return true;
}

bool test_embedding_lookup() {
    const int vocab = 4;
    const int hidden = 8;
    const int batch = 1;
    const int seq = 3;
    std::vector<float> embedding_f(vocab * hidden);
    for (int v = 0; v < vocab; ++v) {
        for (int h = 0; h < hidden; ++h) {
            embedding_f[v * hidden + h] = 0.01f * static_cast<float>(v * 10 + h);
        }
    }
    std::vector<int> input_ids = {1, 3, 0};

    std::vector<float> expected(batch * seq * hidden);
    for (int s = 0; s < seq; ++s) {
        int token = input_ids[s];
        for (int h = 0; h < hidden; ++h) {
            expected[s * hidden + h] = embedding_f[token * hidden + h];
        }
    }

    std::vector<half> embedding_h(embedding_f.size());
    for (size_t i = 0; i < embedding_f.size(); ++i) {
        embedding_h[i] = __float2half(embedding_f[i]);
    }

    half* d_embedding = nullptr;
    int* d_input = nullptr;
    half* d_output = nullptr;
    if (!cuda_check(cudaMalloc(&d_embedding, embedding_h.size() * sizeof(half)), "cudaMalloc embedding")) return false;
    if (!cuda_check(cudaMalloc(&d_input, input_ids.size() * sizeof(int)), "cudaMalloc input_ids")) return false;
    if (!cuda_check(cudaMalloc(&d_output, expected.size() * sizeof(half)), "cudaMalloc output")) return false;
    cuda_check(cudaMemcpy(d_embedding, embedding_h.data(), embedding_h.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy embedding");
    cuda_check(cudaMemcpy(d_input, input_ids.data(), input_ids.size() * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy input_ids");

    ember::cuda::kernels::embedding_lookup_f16(d_output, d_embedding, d_input, batch, seq, hidden, nullptr);
    if (!cuda_check(cudaGetLastError(), "embedding_lookup_f16 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "embedding_lookup_f16 sync")) return false;

    std::vector<half> output_h(expected.size());
    cuda_check(cudaMemcpy(output_h.data(), d_output, output_h.size() * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy output");

    cudaFree(d_embedding);
    cudaFree(d_input);
    cudaFree(d_output);

    std::vector<float> output_f(output_h.size());
    for (size_t i = 0; i < output_h.size(); ++i) {
        output_f[i] = __half2float(output_h[i]);
    }

    float diff = max_abs_diff(output_f, expected);
    if (diff > 1e-2f) {
        std::cerr << "[FAIL] embedding_lookup max_abs_diff=" << diff << "\n";
        return false;
    }
    std::cout << "[PASS] embedding_lookup\n";
    return true;
}

bool test_copy_last_hidden() {
    const int batch = 2;
    const int seq = 3;
    const int hidden = 4;
    std::vector<float> input_f(batch * seq * hidden);
    for (size_t i = 0; i < input_f.size(); ++i) {
        input_f[i] = 0.01f * static_cast<float>(i);
    }

    std::vector<float> expected(batch * hidden);
    for (int b = 0; b < batch; ++b) {
        int src_offset = (b * seq + (seq - 1)) * hidden;
        for (int h = 0; h < hidden; ++h) {
            expected[b * hidden + h] = input_f[src_offset + h];
        }
    }

    std::vector<half> input_h(input_f.size());
    for (size_t i = 0; i < input_f.size(); ++i) {
        input_h[i] = __float2half(input_f[i]);
    }

    half* d_input = nullptr;
    half* d_output = nullptr;
    if (!cuda_check(cudaMalloc(&d_input, input_h.size() * sizeof(half)), "cudaMalloc input")) return false;
    if (!cuda_check(cudaMalloc(&d_output, expected.size() * sizeof(half)), "cudaMalloc output")) return false;
    cuda_check(cudaMemcpy(d_input, input_h.data(), input_h.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy input");

    ember::cuda::kernels::copy_last_hidden_f16(d_output, d_input, batch, seq, hidden, nullptr);
    if (!cuda_check(cudaGetLastError(), "copy_last_hidden_f16 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "copy_last_hidden_f16 sync")) return false;

    std::vector<half> output_h(expected.size());
    cuda_check(cudaMemcpy(output_h.data(), d_output, output_h.size() * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy output");

    cudaFree(d_input);
    cudaFree(d_output);

    std::vector<float> output_f(output_h.size());
    for (size_t i = 0; i < output_h.size(); ++i) {
        output_f[i] = __half2float(output_h[i]);
    }

    float diff = max_abs_diff(output_f, expected);
    if (diff > 1e-2f) {
        std::cerr << "[FAIL] copy_last_hidden max_abs_diff=" << diff << "\n";
        return false;
    }
    std::cout << "[PASS] copy_last_hidden\n";
    return true;
}

bool test_convert_f16() {
    const int size = 64;
    std::vector<float> input_f(size);
    for (int i = 0; i < size; ++i) {
        input_f[i] = 0.03f * static_cast<float>(i - 20);
    }

    float* d_f32 = nullptr;
    half* d_f16 = nullptr;
    float* d_out = nullptr;
    if (!cuda_check(cudaMalloc(&d_f32, size * sizeof(float)), "cudaMalloc f32")) return false;
    if (!cuda_check(cudaMalloc(&d_f16, size * sizeof(half)), "cudaMalloc f16")) return false;
    if (!cuda_check(cudaMalloc(&d_out, size * sizeof(float)), "cudaMalloc out")) return false;
    cuda_check(cudaMemcpy(d_f32, input_f.data(), size * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy f32");

    ember::cuda::kernels::convert_f32_to_f16(d_f16, d_f32, size, nullptr);
    if (!cuda_check(cudaGetLastError(), "convert_f32_to_f16 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "convert_f32_to_f16 sync")) return false;

    ember::cuda::kernels::convert_f16_to_f32(d_out, d_f16, size, nullptr);
    if (!cuda_check(cudaGetLastError(), "convert_f16_to_f32 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "convert_f16_to_f32 sync")) return false;

    std::vector<float> output_f(size);
    cuda_check(cudaMemcpy(output_f.data(), d_out, size * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy out");

    cudaFree(d_f32);
    cudaFree(d_f16);
    cudaFree(d_out);

    float diff = max_abs_diff(output_f, input_f);
    if (diff > 1e-2f) {
        std::cerr << "[FAIL] convert_f16 max_abs_diff=" << diff << "\n";
        return false;
    }
    std::cout << "[PASS] convert_f16\n";
    return true;
}

bool test_rmsnorm() {
    const int batch = 1;
    const int seq = 2;
    const int hidden = 8;
    const int rows = batch * seq;
    const float eps = 1e-6f;

    std::vector<float> input_f(rows * hidden);
    std::vector<float> weight_f(hidden);
    for (int i = 0; i < rows * hidden; ++i) {
        input_f[i] = 0.05f * static_cast<float>(i - 7);
    }
    for (int i = 0; i < hidden; ++i) {
        weight_f[i] = 1.0f + 0.01f * static_cast<float>(i);
    }

    std::vector<float> expected(input_f.size());
    for (int r = 0; r < rows; ++r) {
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; ++i) {
            float v = input_f[r * hidden + i];
            sum_sq += v * v;
        }
        float rms_inv = 1.0f / std::sqrt(sum_sq / hidden + eps);
        for (int i = 0; i < hidden; ++i) {
            expected[r * hidden + i] = input_f[r * hidden + i] * rms_inv * weight_f[i];
        }
    }

    std::vector<half> input_h(input_f.size());
    std::vector<half> weight_h(weight_f.size());
    for (size_t i = 0; i < input_f.size(); ++i) {
        input_h[i] = __float2half(input_f[i]);
    }
    for (size_t i = 0; i < weight_f.size(); ++i) {
        weight_h[i] = __float2half(weight_f[i]);
    }

    half* d_input = nullptr;
    half* d_weight = nullptr;
    half* d_output = nullptr;
    if (!cuda_check(cudaMalloc(&d_input, input_h.size() * sizeof(half)), "cudaMalloc input")) return false;
    if (!cuda_check(cudaMalloc(&d_weight, weight_h.size() * sizeof(half)), "cudaMalloc weight")) return false;
    if (!cuda_check(cudaMalloc(&d_output, input_h.size() * sizeof(half)), "cudaMalloc output")) return false;

    cuda_check(cudaMemcpy(d_input, input_h.data(), input_h.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy input");
    cuda_check(cudaMemcpy(d_weight, weight_h.data(), weight_h.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy weight");

    ember::cuda::kernels::rms_norm_f16(d_output, d_input, d_weight, batch, seq, hidden, eps, nullptr);
    if (!cuda_check(cudaGetLastError(), "rms_norm_f16 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "rms_norm_f16 sync")) return false;

    std::vector<half> output_h(input_h.size());
    cuda_check(cudaMemcpy(output_h.data(), d_output, output_h.size() * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy output");

    std::vector<float> output_f(output_h.size());
    for (size_t i = 0; i < output_h.size(); ++i) {
        output_f[i] = __half2float(output_h[i]);
    }

    cudaFree(d_input);
    cudaFree(d_weight);
    cudaFree(d_output);

    float diff = max_abs_diff(output_f, expected);
    if (diff > 1e-2f) {
        std::cerr << "[FAIL] rmsnorm max_abs_diff=" << diff << "\n";
        return false;
    }
    std::cout << "[PASS] rmsnorm\n";
    return true;
}

bool test_softmax() {
    const int batch = 1;
    const int heads = 1;
    const int seq_q = 2;
    const int seq_k = 4;
    const float scale = 0.5f;
    const int rows = batch * heads * seq_q;

    std::vector<float> input_f = {
        0.0f, 1.0f, 2.0f, -1.0f,
        1.0f, 3.0f, -2.0f, 0.5f,
    };

    std::vector<float> expected(input_f.size());
    for (int r = 0; r < rows; ++r) {
        float max_val = -1e20f;
        for (int i = 0; i < seq_k; ++i) {
            float v = input_f[r * seq_k + i] * scale;
            max_val = std::max(max_val, v);
        }
        float sum = 0.0f;
        for (int i = 0; i < seq_k; ++i) {
            float v = input_f[r * seq_k + i] * scale;
            sum += std::exp(v - max_val);
        }
        float inv = 1.0f / (sum + 1e-6f);
        for (int i = 0; i < seq_k; ++i) {
            float v = input_f[r * seq_k + i] * scale;
            expected[r * seq_k + i] = std::exp(v - max_val) * inv;
        }
    }

    std::vector<half> input_h(input_f.size());
    for (size_t i = 0; i < input_f.size(); ++i) {
        input_h[i] = __float2half(input_f[i]);
    }
    half* d_input = nullptr;
    half* d_output = nullptr;
    if (!cuda_check(cudaMalloc(&d_input, input_h.size() * sizeof(half)), "cudaMalloc input")) return false;
    if (!cuda_check(cudaMalloc(&d_output, input_h.size() * sizeof(half)), "cudaMalloc output")) return false;
    cuda_check(cudaMemcpy(d_input, input_h.data(), input_h.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy input");

    ember::cuda::kernels::softmax_f16(d_output, d_input, batch, heads, seq_q, seq_k, scale, nullptr);
    if (!cuda_check(cudaGetLastError(), "softmax_f16 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "softmax_f16 sync")) return false;

    std::vector<half> output_h(input_h.size());
    cuda_check(cudaMemcpy(output_h.data(), d_output, output_h.size() * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy output");

    std::vector<float> output_f(output_h.size());
    for (size_t i = 0; i < output_h.size(); ++i) {
        output_f[i] = __half2float(output_h[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    float diff = max_abs_diff(output_f, expected);
    if (diff > 1e-2f) {
        std::cerr << "[FAIL] softmax max_abs_diff=" << diff << "\n";
        return false;
    }
    std::cout << "[PASS] softmax\n";
    return true;
}

void apply_rope_ref(std::vector<float>& q, std::vector<float>& k,
                    int batch, int seq_len, int num_heads, int num_kv_heads,
                    int head_dim, int start_pos, float theta) {
    const int half_head = head_dim / 2;
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int pos = start_pos + s;
            for (int h = 0; h < num_heads; ++h) {
                for (int d = 0; d < half_head; ++d) {
                    float freq = 1.0f / std::pow(theta, float(d * 2) / float(head_dim));
                    float angle = pos * freq;
                    float c = std::cos(angle);
                    float si = std::sin(angle);
                    int offset = ((b * seq_len + s) * num_heads + h) * head_dim;
                    float q0 = q[offset + d * 2];
                    float q1 = q[offset + d * 2 + 1];
                    q[offset + d * 2] = q0 * c - q1 * si;
                    q[offset + d * 2 + 1] = q0 * si + q1 * c;
                }
            }
            for (int h = 0; h < num_kv_heads; ++h) {
                for (int d = 0; d < half_head; ++d) {
                    float freq = 1.0f / std::pow(theta, float(d * 2) / float(head_dim));
                    float angle = pos * freq;
                    float c = std::cos(angle);
                    float si = std::sin(angle);
                    int offset = ((b * seq_len + s) * num_kv_heads + h) * head_dim;
                    float k0 = k[offset + d * 2];
                    float k1 = k[offset + d * 2 + 1];
                    k[offset + d * 2] = k0 * c - k1 * si;
                    k[offset + d * 2 + 1] = k0 * si + k1 * c;
                }
            }
        }
    }
}

bool test_rope() {
    const int batch = 1;
    const int seq_len = 2;
    const int num_heads = 2;
    const int num_kv_heads = 1;
    const int head_dim = 4;
    const int start_pos = 0;
    const float theta = 10000.0f;

    const int q_size = batch * seq_len * num_heads * head_dim;
    const int k_size = batch * seq_len * num_kv_heads * head_dim;

    std::vector<float> q_f(q_size);
    std::vector<float> k_f(k_size);
    for (int i = 0; i < q_size; ++i) {
        q_f[i] = 0.01f * static_cast<float>(i + 1);
    }
    for (int i = 0; i < k_size; ++i) {
        k_f[i] = 0.02f * static_cast<float>(i + 1);
    }

    std::vector<float> q_expected = q_f;
    std::vector<float> k_expected = k_f;
    apply_rope_ref(q_expected, k_expected, batch, seq_len, num_heads, num_kv_heads,
                   head_dim, start_pos, theta);

    std::vector<half> q_h(q_size);
    std::vector<half> k_h(k_size);
    for (int i = 0; i < q_size; ++i) {
        q_h[i] = __float2half(q_f[i]);
    }
    for (int i = 0; i < k_size; ++i) {
        k_h[i] = __float2half(k_f[i]);
    }

    half* d_q = nullptr;
    half* d_k = nullptr;
    if (!cuda_check(cudaMalloc(&d_q, q_h.size() * sizeof(half)), "cudaMalloc q")) return false;
    if (!cuda_check(cudaMalloc(&d_k, k_h.size() * sizeof(half)), "cudaMalloc k")) return false;
    cuda_check(cudaMemcpy(d_q, q_h.data(), q_h.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy q");
    cuda_check(cudaMemcpy(d_k, k_h.data(), k_h.size() * sizeof(half), cudaMemcpyHostToDevice),
               "cudaMemcpy k");

    ember::cuda::kernels::apply_rope_f16(d_q, d_k, batch, seq_len, num_heads, num_kv_heads,
                                         head_dim, start_pos, theta, nullptr);
    if (!cuda_check(cudaGetLastError(), "apply_rope_f16 launch")) return false;
    if (!cuda_check(cudaDeviceSynchronize(), "apply_rope_f16 sync")) return false;

    cuda_check(cudaMemcpy(q_h.data(), d_q, q_h.size() * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy q out");
    cuda_check(cudaMemcpy(k_h.data(), d_k, k_h.size() * sizeof(half), cudaMemcpyDeviceToHost),
               "cudaMemcpy k out");

    std::vector<float> q_out(q_size);
    std::vector<float> k_out(k_size);
    for (int i = 0; i < q_size; ++i) {
        q_out[i] = __half2float(q_h[i]);
    }
    for (int i = 0; i < k_size; ++i) {
        k_out[i] = __half2float(k_h[i]);
    }

    cudaFree(d_q);
    cudaFree(d_k);

    float diff_q = max_abs_diff(q_out, q_expected);
    float diff_k = max_abs_diff(k_out, k_expected);
    float diff = std::max(diff_q, diff_k);
    if (diff > 1e-2f) {
        std::cerr << "[FAIL] rope max_abs_diff=" << diff << "\n";
        return false;
    }
    std::cout << "[PASS] rope\n";
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "[skip] No CUDA device available\n";
        return 0;
    }

    bool ok = true;
    ok &= test_update_kv_cache();
    ok &= test_attention();
    ok &= test_silu();
    ok &= test_elementwise();
    ok &= test_embedding_lookup();
    ok &= test_copy_last_hidden();
    ok &= test_convert_f16();
    ok &= test_rmsnorm();
    ok &= test_softmax();
    ok &= test_rope();

    if (ok) {
        std::cout << "[PASS] cuda kernels smoke test\n";
        return 0;
    }
    return 1;
}
