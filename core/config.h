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
}  // namespace ember