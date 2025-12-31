#include "inference.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

namespace ember {

// ============ KV Cache ============

Error KVCache::init(const Qwen3Model& model, int32_t ctx_size) {
    const auto& hp = model.hparams();
    n_ctx = ctx_size;
    n_used = 0;
    
    // 计算每层 KV cache 大小
    // K/V shape: [n_ctx, n_head_kv, head_dim]
    size_t head_dim = hp.n_embd / hp.n_head;
    size_t kv_size = ctx_size * hp.n_head_kv * head_dim * sizeof(float);  // FP32
    
    k_cache.resize(hp.n_layer);
    v_cache.resize(hp.n_layer);
    
    // 为每层分配 KV cache（在对应的 GPU 上）
    for (uint32_t layer = 0; layer < hp.n_layer; ++layer) {
        // TODO: 根据 layer allocation 选择正确的 GPU
        cudaError_t err;
        err = cudaMalloc(&k_cache[layer], kv_size);
        if (err != cudaSuccess) {
            LOG_ERROR("分配 K cache 失败 (layer %d): %s", layer, cudaGetErrorString(err));
            return Error::OUT_OF_MEMORY;
        }
        
        err = cudaMalloc(&v_cache[layer], kv_size);
        if (err != cudaSuccess) {
            LOG_ERROR("分配 V cache 失败 (layer %d): %s", layer, cudaGetErrorString(err));
            return Error::OUT_OF_MEMORY;
        }
    }
    
    LOG_INFO("KV Cache 初始化: %u 层, 每层 %s", hp.n_layer, format_bytes(kv_size * 2).c_str());
    return Error::OK;
}

void KVCache::clear() {
    n_used = 0;
    // 不需要清零内存，重新写入会覆盖
}

KVCache::~KVCache() {
    for (void* p : k_cache) if (p) cudaFree(p);
    for (void* p : v_cache) if (p) cudaFree(p);
}

// ============ InferenceContext ============

Error InferenceContext::init(Qwen3Model* m, const CliArgs& args) {
    model = m;
    temperature = args.temperature;
    top_p = args.top_p;
    top_k = args.top_k;
    n_past = 0;
    tokens.clear();
    
    return kv_cache.init(*model, args.n_ctx);
}

void InferenceContext::reset() {
    kv_cache.clear();
    tokens.clear();
    n_past = 0;
}

// ============ Tokenizer（简化实现） ============

Error Tokenizer::load(const Qwen3Model& model) {
    // TODO: 从 GGUF 元数据加载词表
    // 目前使用占位实现
    
    const auto& hp = model.hparams();
    vocab_.resize(hp.n_vocab);
    
    // 设置特殊 token（Qwen3 默认值）
    bos_id_ = 151643;  // <|im_start|>
    eos_id_ = 151645;  // <|im_end|>
    pad_id_ = 151643;
    
    loaded_ = true;
    LOG_INFO("Tokenizer 加载完成 (词表: %u)", hp.n_vocab);
    return Error::OK;
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    // TODO: 实现完整的 BPE/SentencePiece 编码
    // 目前返回占位结果
    
    LOG_WARN("Tokenizer::encode 尚未完全实现");
    std::vector<int32_t> tokens;
    tokens.push_back(bos_id_);  // BOS
    
    // 简单的字符级编码（仅用于测试）
    for (char c : text) {
        tokens.push_back(static_cast<int32_t>(static_cast<uint8_t>(c)));
    }
    
    return tokens;
}

std::string Tokenizer::decode(const std::vector<int32_t>& tokens) const {
    std::string result;
    for (int32_t t : tokens) {
        result += decode(t);
    }
    return result;
}

std::string Tokenizer::decode(int32_t token) const {
    // TODO: 实现完整解码
    if (token == bos_id_ || token == eos_id_ || token == pad_id_) {
        return "";
    }
    if (token < 256) {
        return std::string(1, static_cast<char>(token));
    }
    return "[?]";
}

// ============ 前向推理 ============

Error forward(InferenceContext& ctx, const std::vector<int32_t>& input_tokens, 
              std::vector<float>& logits) {
    // TODO: 实现完整的 Qwen3 前向推理
    // 这里是核心计算，需要调用 CUDA kernel
    
    /*
    前向计算流程：
    1. Embedding lookup
    2. 对每一层:
       a. RMSNorm (pre-attention)
       b. Self-Attention (Q/K/V projection, RoPE, Attention, Output projection)
       c. 残差连接
       d. RMSNorm (pre-FFN)
       e. FFN (gate/up projection, SiLU, down projection)
       f. 残差连接
    3. Final RMSNorm
    4. LM Head (output projection)
    */
    
    const auto& hp = ctx.model->hparams();
    size_t n_tokens = input_tokens.size();
    
    LOG_DEBUG("Forward: %zu tokens, n_past=%d", n_tokens, ctx.n_past);
    
    // 输出 logits 大小
    logits.resize(hp.n_vocab);
    
    // 占位实现：返回随机 logits
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (size_t i = 0; i < hp.n_vocab; ++i) {
        logits[i] = dist(gen);
    }
    
    // 更新状态
    ctx.n_past += n_tokens;
    ctx.kv_cache.n_used = ctx.n_past;
    
    LOG_WARN("forward() 使用占位实现，需要完成 CUDA kernel");
    
    return Error::OK;
}

// ============ 采样 ============

int32_t sample(const std::vector<float>& logits, const InferenceContext& ctx) {
    // Temperature 缩放
    std::vector<float> scaled_logits = logits;
    
    if (ctx.temperature > 0) {
        for (auto& l : scaled_logits) {
            l /= ctx.temperature;
        }
    }
    
    // Softmax
    float max_logit = *std::max_element(scaled_logits.begin(), scaled_logits.end());
    std::vector<float> probs(scaled_logits.size());
    float sum = 0.0f;
    
    for (size_t i = 0; i < scaled_logits.size(); ++i) {
        probs[i] = std::exp(scaled_logits[i] - max_logit);
        sum += probs[i];
    }
    for (auto& p : probs) p /= sum;
    
    // Top-K 筛选
    std::vector<std::pair<float, int32_t>> prob_idx;
    prob_idx.reserve(probs.size());
    for (size_t i = 0; i < probs.size(); ++i) {
        prob_idx.emplace_back(probs[i], static_cast<int32_t>(i));
    }
    
    std::partial_sort(prob_idx.begin(), 
                      prob_idx.begin() + std::min(ctx.top_k, (int32_t)prob_idx.size()),
                      prob_idx.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    prob_idx.resize(std::min(ctx.top_k, (int32_t)prob_idx.size()));
    
    // Top-P 筛选
    float cumsum = 0.0f;
    size_t cutoff = prob_idx.size();
    for (size_t i = 0; i < prob_idx.size(); ++i) {
        cumsum += prob_idx[i].first;
        if (cumsum >= ctx.top_p) {
            cutoff = i + 1;
            break;
        }
    }
    prob_idx.resize(cutoff);
    
    // 重新归一化
    sum = 0.0f;
    for (const auto& pi : prob_idx) sum += pi.first;
    
    // 采样
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, sum);
    
    float r = dist(gen);
    cumsum = 0.0f;
    for (const auto& pi : prob_idx) {
        cumsum += pi.first;
        if (r <= cumsum) {
            return pi.second;
        }
    }
    
    return prob_idx.back().second;
}

// ============ 文本生成 ============

Error generate(InferenceContext& ctx, 
               const std::string& prompt,
               int32_t n_predict,
               GenerateResult& result) {
    Timer timer;
    
    // TODO: 使用真实的 tokenizer
    Tokenizer tokenizer;
    tokenizer.load(*ctx.model);
    
    // 编码 prompt
    std::vector<int32_t> prompt_tokens = tokenizer.encode(prompt);
    LOG_INFO("Prompt: %zu tokens", prompt_tokens.size());
    
    // 初始化结果
    result.tokens = prompt_tokens;
    result.n_tokens = 0;
    
    // Prefill: 处理整个 prompt
    std::vector<float> logits;
    Error err = forward(ctx, prompt_tokens, logits);
    if (err != Error::OK) return err;
    
    // 逐 token 生成
    for (int32_t i = 0; i < n_predict; ++i) {
        // 采样
        int32_t next_token = sample(logits, ctx);
        result.tokens.push_back(next_token);
        result.n_tokens++;
        
        // 检查 EOS
        if (next_token == tokenizer.eos_token()) {
            LOG_INFO("生成结束 (EOS)");
            break;
        }
        
        // 流式输出
        std::string token_str = tokenizer.decode(next_token);
        std::cout << token_str << std::flush;
        
        // Decode: 处理单个 token
        err = forward(ctx, {next_token}, logits);
        if (err != Error::OK) return err;
    }
    
    std::cout << std::endl;
    
    // 解码结果
    result.text = tokenizer.decode(result.tokens);
    result.time_ms = timer.elapsed_ms();
    result.tokens_per_second = result.n_tokens / (result.time_ms / 1000.0);
    
    LOG_INFO("生成完成: %d tokens, %.2f ms, %.2f tokens/s",
             result.n_tokens, result.time_ms, result.tokens_per_second);
    
    return Error::OK;
}

// ============ 交互模式 ============

Error interactive_loop(InferenceContext& ctx) {
    LOG_INFO("进入交互模式 (输入 'quit' 退出)");
    
    std::string line;
    while (true) {
        std::cout << "\n> " << std::flush;
        if (!std::getline(std::cin, line)) break;
        
        if (line == "quit" || line == "exit") {
            break;
        }
        
        if (line == "clear" || line == "reset") {
            ctx.reset();
            LOG_INFO("上下文已重置");
            continue;
        }
        
        if (line.empty()) continue;
        
        GenerateResult result;
        Error err = generate(ctx, line, 128, result);
        if (err != Error::OK) {
            LOG_ERROR("生成失败: %s", error_to_string(err));
        }
    }
    
    return Error::OK;
}

} // namespace ember
