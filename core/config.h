#pragma once

#include "types.h"
#include <string>
#include <vector>
#include <cstdint>

namespace ember {

// 模型配置（从模型目录的 config.json 读取或由预设构造）
struct ModelConfig {
    // 基本信息
    std::string model_type;
    std::string model_path;
    
    // 模型架构参数
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;      // GQA
    int64_t head_dim = 0;          // hidden_size / num_heads
    int64_t max_position_embeddings = 0;
    
    // RoPE 参数
    double rope_theta = 0.0;
    std::string rope_scaling_type;  // none, linear, dynamic
    double rope_scaling_factor = 0.0;
    
    // Norm 参数
    double rms_norm_eps = 0.0;
    
    // 其他
    bool tie_word_embeddings = true;
    std::string torch_dtype;
    
    // 计算每层的 KV cache 大小（字节）
    size_t kv_cache_size_per_layer(int ctx_len, int batch_size = 1, DType dtype = DType::F16) const {
        // K 和 V 各需要 [batch, num_kv_heads, ctx_len, head_dim]
        size_t kv_elements = batch_size * num_kv_heads * ctx_len * head_dim * 2;
        return kv_elements * dtype_size(dtype);
    }
    
    // 估算模型权重大小（字节）
    size_t estimate_weights_size(DType dtype = DType::F16) const {
        size_t elem_size = dtype_size(dtype);
        size_t total = 0;
        
        // Embedding
        total += vocab_size * hidden_size * elem_size;
        
        // 每层的参数
        size_t per_layer = 0;
        // self_attn: q_proj, k_proj, v_proj, o_proj
        per_layer += hidden_size * (num_heads * head_dim) * elem_size;       // q_proj
        per_layer += hidden_size * (num_kv_heads * head_dim) * elem_size;    // k_proj
        per_layer += hidden_size * (num_kv_heads * head_dim) * elem_size;    // v_proj
        per_layer += (num_heads * head_dim) * hidden_size * elem_size;       // o_proj
        // q/k per-head norm
        per_layer += head_dim * elem_size * 2;
        // mlp: gate_proj, up_proj, down_proj
        per_layer += hidden_size * intermediate_size * elem_size;  // gate_proj
        per_layer += hidden_size * intermediate_size * elem_size;  // up_proj
        per_layer += intermediate_size * hidden_size * elem_size;  // down_proj
        // layernorms (相对小，忽略)
        per_layer += hidden_size * elem_size * 2;
        
        total += per_layer * num_layers;
        
        // Final norm
        total += hidden_size * elem_size;
        
        // lm_head (如果不 tie)
        if (!tie_word_embeddings) {
            total += hidden_size * vocab_size * elem_size;
        }
        
        return total;
    }
    
    // 打印配置
    void print() const;
};

// 运行时配置
struct RuntimeConfig {
    // 上下文参数
    int max_ctx_len = 2048;
    int batch_size = 1;
    
    // 采样参数
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    int no_repeat_ngram_size = 0;
    
    // 生成参数
    int max_new_tokens = 128;
    bool stream = true;
    
    // 设备参数
    std::vector<int> device_ids = {0};  // 使用的 GPU 列表
    float memory_fraction = 0.9f;        // 每张卡最多使用的显存比例
    
    // 精度
    DType kv_cache_dtype = DType::F16;
    
    // 调试
    bool verbose = false;
    bool check_correctness = false;
    int dump_layer = -1;            // <0 = all (when check_correctness), >=0 = specific layer
    std::string dump_dir = "debug";
};

// 预设的模型配置
namespace ModelPresets {

inline ModelConfig qwen3_base() {
    ModelConfig c;
    c.model_type = "qwen3";
    c.rope_theta = 1000000.0;
    c.rope_scaling_type = "none";
    c.rope_scaling_factor = 1.0;
    c.rms_norm_eps = 1e-6;
    c.tie_word_embeddings = true;
    c.torch_dtype = "float16";
    c.max_position_embeddings = 32768;
    return c;
}

inline ModelConfig qwen3_0_6b() {
    ModelConfig c = qwen3_base();
    c.vocab_size = 151936;
    c.hidden_size = 1024;
    c.intermediate_size = 3072;
    c.num_layers = 28;
    c.num_heads = 16;
    c.num_kv_heads = 8;
    c.head_dim = 64;
    return c;
}

inline ModelConfig qwen3_1_7b() {
    ModelConfig c = qwen3_base();
    c.vocab_size = 151936;
    c.hidden_size = 2048;
    c.intermediate_size = 6144;
    c.num_layers = 28;
    c.num_heads = 16;
    c.num_kv_heads = 8;
    c.head_dim = 128;
    return c;
}

inline ModelConfig qwen3_4b() {
    ModelConfig c = qwen3_base();
    c.vocab_size = 151936;
    c.hidden_size = 2560;
    c.intermediate_size = 9728;
    c.num_layers = 36;
    c.num_heads = 32;
    c.num_kv_heads = 8;
    c.head_dim = 80;
    return c;
}

inline ModelConfig qwen3_8b() {
    ModelConfig c = qwen3_base();
    c.vocab_size = 152064;
    c.hidden_size = 4096;
    c.intermediate_size = 12288;
    c.num_layers = 36;
    c.num_heads = 32;
    c.num_kv_heads = 8;
    c.head_dim = 128;
    return c;
}

inline ModelConfig qwen3_14b() {
    ModelConfig c = qwen3_base();
    c.vocab_size = 152064;
    c.hidden_size = 5120;
    c.intermediate_size = 17408;
    c.num_layers = 40;
    c.num_heads = 40;
    c.num_kv_heads = 8;
    c.head_dim = 128;
    return c;
}

}  // namespace ModelPresets

}  // namespace ember
