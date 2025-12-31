#pragma once

#include "model.h"
#include "cli.h"
#include <vector>
#include <string>

namespace ember {

// KV Cache
struct KVCache {
    std::vector<void*> k_cache;  // 每层的 key cache
    std::vector<void*> v_cache;  // 每层的 value cache
    int32_t n_ctx;               // 上下文长度
    int32_t n_used;              // 已使用的位置
    
    Error init(const Qwen3Model& model, int32_t ctx_size);
    void clear();
    ~KVCache();
};

// 推理上下文
struct InferenceContext {
    Qwen3Model* model;
    KVCache kv_cache;
    
    // 采样参数
    float temperature;
    float top_p;
    int32_t top_k;
    
    // 状态
    std::vector<int32_t> tokens;  // 当前 token 序列
    int32_t n_past;               // 已处理的 token 数
    
    Error init(Qwen3Model* model, const CliArgs& args);
    void reset();
};

// Tokenizer（简化版，实际应使用完整实现）
class Tokenizer {
public:
    Error load(const Qwen3Model& model);
    
    // 编码文本
    std::vector<int32_t> encode(const std::string& text) const;
    
    // 解码 token
    std::string decode(const std::vector<int32_t>& tokens) const;
    std::string decode(int32_t token) const;
    
    // 特殊 token
    int32_t bos_token() const { return bos_id_; }
    int32_t eos_token() const { return eos_id_; }
    int32_t pad_token() const { return pad_id_; }
    
    bool is_loaded() const { return loaded_; }
    
private:
    bool loaded_ = false;
    int32_t bos_id_ = 1;
    int32_t eos_id_ = 2;
    int32_t pad_id_ = 0;
    
    std::vector<std::string> vocab_;
    // TODO: 实际需要完整的 tokenizer 实现（BPE/SentencePiece）
};

// 前向推理
Error forward(InferenceContext& ctx, const std::vector<int32_t>& input_tokens, 
              std::vector<float>& logits);

// 采样下一个 token
int32_t sample(const std::vector<float>& logits, const InferenceContext& ctx);

// 生成文本
struct GenerateResult {
    std::vector<int32_t> tokens;
    std::string text;
    int32_t n_tokens;
    double time_ms;
    double tokens_per_second;
};

Error generate(InferenceContext& ctx, 
               const std::string& prompt,
               int32_t n_predict,
               GenerateResult& result);

// 交互式推理
Error interactive_loop(InferenceContext& ctx);

} // namespace ember
