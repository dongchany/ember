#pragma once

#include "../../runtime/iruntime.h"
#include "../../runtime/batch_runtime.h"
#include "../../formats/safetensors.h"
#include "cuda_utils.h"
#include "kernels/kernels.h"
#include <unordered_map>
#include <memory>

namespace ember {
namespace cuda {

// Qwen3 层的权重
struct Qwen3LayerWeights {
    // Self Attention
    void* q_proj_weight = nullptr;   // [num_heads * head_dim, hidden_size]
    void* k_proj_weight = nullptr;   // [num_kv_heads * head_dim, hidden_size]
    void* v_proj_weight = nullptr;   // [num_kv_heads * head_dim, hidden_size]
    void* o_proj_weight = nullptr;   // [hidden_size, num_heads * head_dim]
    void* q_norm_weight = nullptr;   // [head_dim]
    void* k_norm_weight = nullptr;   // [head_dim]
    
    // MLP
    void* gate_proj_weight = nullptr;  // [intermediate_size, hidden_size]
    void* up_proj_weight = nullptr;    // [intermediate_size, hidden_size]
    void* down_proj_weight = nullptr;  // [hidden_size, intermediate_size]
    
    // LayerNorm
    void* input_layernorm_weight = nullptr;          // [hidden_size]
    void* post_attention_layernorm_weight = nullptr; // [hidden_size]
    
    int device_id = 0;
    bool allocated = false;
    
    size_t memory_usage(const ModelConfig& config, DType dtype) const;
};

// Qwen3 完整模型权重
struct Qwen3Weights {
    void* embed_tokens = nullptr;   // [vocab_size, hidden_size]
    void* lm_head = nullptr;        // [vocab_size, hidden_size] (可能与 embed_tokens 共享)
    void* final_norm = nullptr;     // [hidden_size]
    
    std::vector<Qwen3LayerWeights> layers;
    
    DType dtype = DType::F16;
    bool lm_head_owns_allocation = false;  // multi-GPU + tied embedding may need a copy
    int embed_device_id = 0;
    int lm_head_device_id = 0;
    int final_norm_device_id = 0;
};

// 中间激活值缓冲区
struct ActivationBuffers {
    void* hidden_states = nullptr;     // [batch, seq, hidden_size]
    void* residual = nullptr;          // [batch, seq, hidden_size]
    void* norm_out = nullptr;          // [batch, seq, hidden_size]
    void* last_hidden = nullptr;       // [batch, hidden_size] (packed last token per batch)
    void* q_proj_out = nullptr;        // [batch, seq, num_heads * head_dim]
    void* k_proj_out = nullptr;        // [batch, seq, num_kv_heads * head_dim]
    void* v_proj_out = nullptr;        // [batch, seq, num_kv_heads * head_dim]
    void* attn_out = nullptr;          // [batch, seq, num_heads * head_dim]
    void* attn_scores = nullptr;       // [batch, num_heads, seq_q, seq_k] (FP32)
    void* attn_probs = nullptr;        // [batch, num_heads, seq_q, seq_k] (compute dtype)
    void* mlp_gate = nullptr;          // [batch, seq, intermediate_size]
    void* mlp_up = nullptr;            // [batch, seq, intermediate_size]
    void* mlp_down = nullptr;          // [batch, seq, hidden_size]
    void* logits = nullptr;            // [batch, vocab_size] (仅最后一个 token)
    
    int device_id = 0;
    size_t max_seq_len = 0;
    size_t batch_size = 0;

    // Attention workspace capacities.
    size_t attn_q_max = 0;
    size_t attn_k_max = 0;
    
    bool allocated = false;
};

// CUDA Runtime 实现
class CudaRuntime : public IRuntime, public IBatchRuntime {
public:
    struct StageProfileMs {
        float embedding_ms = 0.0f;
        float rmsnorm_ms = 0.0f;
        float attention_ms = 0.0f;
        float ffn_ms = 0.0f;
        float p2p_ms = 0.0f;
        float memcpy_h2d_ms = 0.0f;
        float memcpy_d2h_ms = 0.0f;
        float lm_head_ms = 0.0f;
        float total_ms = 0.0f;
    };

    CudaRuntime();
    ~CudaRuntime() override;
    
    std::string name() const override { return "CUDA"; }
    bool available() const override;
    
    Error load(const std::string& model_path, 
               const ModelConfig& config, 
               const DeviceMap& device_map) override;
    
    Error prefill(const std::vector<int>& tokens, Session& session) override;
    Error prefill_with_logits(const std::vector<int>& tokens, Session& session, std::vector<float>& logits) override;
    Error decode(int last_token, Session& session, std::vector<float>& logits) override;
    
    MemoryEstimate estimate_memory(const ModelConfig& config, 
                                   int ctx_len, int batch_size) override;
    
    Error allocate_kv_cache(Session& session) override;
    void free_kv_cache(Session& session) override;
    
    void unload() override;
    bool loaded() const override { return loaded_; }
    
    const DeviceMap& device_map() const override { return device_map_; }

    // Benchmarking / profiling helpers (opt-in).
    void set_layer_profiling(bool enabled);
    std::vector<float> take_last_layer_profile_ms();
    void set_stage_profiling(bool enabled);
    StageProfileMs take_last_stage_profile_ms();

    // Chunked prefill baseline for 2-GPU pipeline experiments.
    // - When overlap=false: still uses chunked execution but leaves communication implicit (synchronous boundary copy).
    // - When overlap=true: overlaps GPU0 compute with GPU0->GPU1 activation transfers (2-GPU, contiguous split).
    Error prefill_chunked_pipeline(const std::vector<int>& tokens,
                                   Session& session,
                                   int chunk_len,
                                   bool overlap,
                                   std::vector<float>* out_logits) override;

    // Batch helpers for decode throughput experiments (keeps logits on GPU; no host copy).
    // These require session.runtime_config().batch_size to match the provided inputs.
    Error prefill_batch_flat(const std::vector<int>& input_ids_flat,
                             int seq_len,
                             Session& session);
    // Prefill a single request into a specific batch slot (KV cache slice), keeping other slots intact.
    // - Uses batch_size=1 execution with KV cache pointers temporarily offset to `slot`.
    // - Optionally returns logits for the last prompt token (vocab-sized float vector).
    Error prefill_into_slot(const std::vector<int>& tokens,
                            int slot,
                            Session& session,
                            std::vector<float>* out_logits) override;
    Error prefill_into_slot_pipeline(const std::vector<int>& tokens,
                                     int slot,
                                     Session& session,
                                     int chunk_len,
                                     bool overlap,
                                     std::vector<float>* out_logits) override;
    Error decode_to_device(int last_token, Session& session);
    Error decode_batch_to_device(const std::vector<int>& last_tokens,
                                 Session& session);
    Error decode_batch(const std::vector<int>& last_tokens,
                       Session& session,
                       std::vector<float>& logits_flat) override;
    // Decode one step and return greedy next tokens (argmax over logits) for each batch slot.
    Error decode_batch_greedy(const std::vector<int>& last_tokens,
                              Session& session,
                              std::vector<int>& next_tokens) override;

private:
    struct LayerStageEvents;
    struct SimpleStageEvents;

    void destroy_stage_profiling_events_();
    void ensure_stage_profiling_events_();
    void ensure_simple_stage_events_(SimpleStageEvents& ev, int device_id);
    void ensure_layer_stage_events_(LayerStageEvents& ev, int device_id);
    Error begin_stage_profile_();
    Error finalize_stage_profile_(bool sync_all_devices,
                                  bool include_final_norm,
                                  bool include_lm_head);
    void add_stage_profile_h2d_ms_(double ms);
    void add_stage_profile_d2h_ms_(double ms);

    // 加载模型权重
    Error load_weights(const std::string& model_path);
    Error load_layer_weights(int layer_idx, ModelWeightLoader& loader, int device_id);
    
    // 分配激活缓冲区
    // - max_seq_len: token buffers capacity (hidden/qkv/mlp) along sequence dimension.
    // - attn_q_max/attn_k_max: attention workspace capacity (scores/probs) for [seq_q, seq_k].
    Error allocate_activation_buffers(int max_seq_len, int batch_size, int attn_q_max, int attn_k_max);
    void free_activation_buffers();
    
    // 前向计算
    Error forward_embedding(const int* input_ids, int batch_size, int seq_len);
    Error forward_layer(int layer_idx, int batch_size, int seq_len, int start_pos, Session& session,
                        bool skip_input_copy,
                        const int* start_pos_by_batch = nullptr);
    Error forward_final_norm(int batch_size, int seq_len, Session& session);
    Error forward_lm_head(int batch_size, int seq_len);
    
    // Attention 计算
    Error compute_qkv(int layer_idx, int batch_size, int seq_len);
    Error compute_attention(int layer_idx, int batch_size, int seq_len, int start_pos, Session& session);
    Error compute_mlp(int layer_idx, int batch_size, int seq_len);
    
    // GEMM helper
    Error gemm_f16(half* C, const half* A, const half* B, 
                   int M, int N, int K, bool transA, bool transB,
                   int device_id);
    
    // 模型配置
    ModelConfig config_;
    DeviceMap device_map_;
    bool loaded_ = false;
    
    // 权重
    Qwen3Weights weights_;
    
    // 激活缓冲区（每个设备一份）
    std::vector<ActivationBuffers> activations_;
    
    // cuBLAS 句柄（每个设备一个）
    std::vector<CublasHandle> cublas_handles_;
    
    // CUDA streams（每个设备一个）
    std::vector<cudaStream_t> streams_;
    std::vector<cudaStream_t> transfer_streams_;

    // Per-layer timing profile (ms) for the most recent prefill/decode call.
    bool profile_layers_ = false;
    std::vector<float> last_layer_profile_ms_;
    struct ProfileEvents {
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
    };
    std::vector<ProfileEvents> profile_events_;

    // Per-stage timing profile (ms) for the most recent prefill/decode call.
    bool profile_stages_ = false;
    StageProfileMs last_stage_profile_ms_;
    struct LayerStageEvents {
        int device_id = -1;
        cudaEvent_t in_norm_start = nullptr;
        cudaEvent_t in_norm_end = nullptr;
        cudaEvent_t attn_start = nullptr;
        cudaEvent_t attn_end = nullptr;
        cudaEvent_t post_norm_start = nullptr;
        cudaEvent_t post_norm_end = nullptr;
        cudaEvent_t ffn_start = nullptr;
        cudaEvent_t ffn_end = nullptr;
    };
    std::vector<LayerStageEvents> layer_stage_events_;
    struct SimpleStageEvents {
        int device_id = -1;
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
    };
    SimpleStageEvents embedding_events_;
    SimpleStageEvents final_norm_events_;
    SimpleStageEvents lm_head_events_;
    SimpleStageEvents total_events_;

    // KV cache 内存（由 Session 管理，这里只保存指针）
    std::vector<void*> kv_cache_buffers_;
    
    // 临时 CPU 缓冲区（用于与 GPU 通信）
    std::vector<int> input_ids_cpu_;
    std::vector<float> logits_cpu_;

    // Small persistent device buffer for greedy sampling (allocated on lm_head device).
    int* next_tokens_dev_ = nullptr;
    int next_tokens_cap_ = 0;
};

}  // namespace cuda
}  // namespace ember
