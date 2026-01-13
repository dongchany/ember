#pragma once

namespace ember {
// Model Configuration (read from config.json in the model directory or
// constructed from presets)
struct ModelConfig {
    std::string model_type;
    std::string model_path;

    // Model arch params
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_layers = 0;
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;      // GQA
    int64_t head_dim = 0;          // hidden_size / num_heads
    int64_t max_position_embeddings = 0;
    
    // RoPE 
    double rope_theta = 0.0;
    std::string rope_scaling_type;  // none, linear, dynamic
    double rope_scaling_factor = 0.0;
    
    // Norm 
    double rms_norm_eps = 0.0;
    
    
    bool tie_word_embeddings = true;
    std::string torch_dtype;
    
};

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
    DType compute_dtype = DType::F16;
    DType kv_cache_dtype = DType::F16;
    
    // 调试
    bool verbose = false;
    bool check_correctness = false;
    int dump_layer = -1;            // <0 = all (when check_correctness), >=0 = specific layer
    std::string dump_dir = "debug";
};
}  // namespace ember