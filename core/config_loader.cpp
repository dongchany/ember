#include "core/config_loader.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ember {

namespace {

std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

std::string trim_copy(std::string s) {
    size_t begin = 0;
    while (begin < s.size() && std::isspace(static_cast<unsigned char>(s[begin]))) {
        ++begin;
    }
    size_t end = s.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
        --end;
    }
    return s.substr(begin, end - begin);
}

bool is_qwen35_model_type(const std::string& model_type) {
    const std::string lower = to_lower_copy(model_type);
    return lower.find("qwen3.5") != std::string::npos ||
           lower.find("qwen3_5") != std::string::npos ||
           lower.find("qwen3-5") != std::string::npos;
}

bool parse_layer_type_token(const std::string& token_raw, HybridLayerType& out) {
    std::string token = to_lower_copy(trim_copy(token_raw));
    if (token.empty()) {
        return false;
    }

    if (token == "0" || token == "attn" || token == "attention" ||
        token == "gated_attn" || token == "gated_attention" || token == "a") {
        out = HybridLayerType::GATED_ATTENTION;
        return true;
    }
    if (token == "1" || token == "deltanet" || token == "delta" ||
        token == "ssm" || token == "d") {
        out = HybridLayerType::DELTANET;
        return true;
    }

    // 宽松匹配，兼容不同命名风格。
    if (token.find("delta") != std::string::npos || token.find("ssm") != std::string::npos) {
        out = HybridLayerType::DELTANET;
        return true;
    }
    if (token.find("attn") != std::string::npos || token.find("attention") != std::string::npos) {
        out = HybridLayerType::GATED_ATTENTION;
        return true;
    }
    return false;
}

bool expand_layer_pattern(const std::vector<HybridLayerType>& base,
                          int64_t num_layers,
                          std::vector<HybridLayerType>& out) {
    if (num_layers <= 0 || base.empty()) {
        return false;
    }
    const size_t n_layers = static_cast<size_t>(num_layers);
    if (base.size() > n_layers) {
        return false;
    }
    out.resize(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        out[i] = base[i % base.size()];
    }
    return true;
}

bool parse_pattern_string(const std::string& pattern,
                          int64_t num_layers,
                          std::vector<HybridLayerType>& out) {
    std::string s = trim_copy(pattern);
    if (s.empty()) {
        return false;
    }
    std::string lower = to_lower_copy(s);
    if (lower == "3:1" || lower == "3/1" || lower == "3-1") {
        const std::vector<HybridLayerType> base = {
            HybridLayerType::DELTANET,
            HybridLayerType::DELTANET,
            HybridLayerType::DELTANET,
            HybridLayerType::GATED_ATTENTION,
        };
        return expand_layer_pattern(base, num_layers, out);
    }

    std::vector<HybridLayerType> base;
    bool has_delim = false;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c)) || c == ',' || c == ';' ||
            c == '|' || c == '/') {
            has_delim = true;
            break;
        }
    }

    if (!has_delim) {
        for (char c : s) {
            if (c == '0' || c == 'a' || c == 'A' || c == 'g' || c == 'G') {
                base.push_back(HybridLayerType::GATED_ATTENTION);
            } else if (c == '1' || c == 'd' || c == 'D') {
                base.push_back(HybridLayerType::DELTANET);
            } else {
                return false;
            }
        }
        return expand_layer_pattern(base, num_layers, out);
    }

    std::string token;
    for (size_t i = 0; i <= s.size(); ++i) {
        const bool at_end = (i == s.size());
        const char c = at_end ? ',' : s[i];
        if (at_end || std::isspace(static_cast<unsigned char>(c)) || c == ',' ||
            c == ';' || c == '|' || c == '/') {
            if (!token.empty()) {
                HybridLayerType t = HybridLayerType::GATED_ATTENTION;
                if (!parse_layer_type_token(token, t)) {
                    return false;
                }
                base.push_back(t);
                token.clear();
            }
        } else {
            token.push_back(c);
        }
    }
    return expand_layer_pattern(base, num_layers, out);
}

bool find_array_raw(const std::string& content, const std::string& key, std::string& out) {
    size_t pos = content.find("\"" + key + "\"");
    if (pos == std::string::npos) return false;
    pos = content.find(":", pos);
    if (pos == std::string::npos) return false;
    pos = content.find("[", pos);
    if (pos == std::string::npos) return false;

    size_t i = pos;
    int depth = 0;
    bool in_string = false;
    bool escape = false;
    for (; i < content.size(); ++i) {
        char c = content[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (c == '\\') {
                escape = true;
            } else if (c == '"') {
                in_string = false;
            }
            continue;
        }
        if (c == '"') {
            in_string = true;
            continue;
        }
        if (c == '[') {
            ++depth;
        } else if (c == ']') {
            --depth;
            if (depth == 0) {
                out = content.substr(pos + 1, i - pos - 1);
                return true;
            }
        }
    }
    return false;
}

bool parse_layer_type_array_raw(const std::string& raw,
                                int64_t num_layers,
                                std::vector<HybridLayerType>& out) {
    std::vector<HybridLayerType> base;
    size_t i = 0;
    while (i < raw.size()) {
        while (i < raw.size() && (std::isspace(static_cast<unsigned char>(raw[i])) || raw[i] == ',')) {
            ++i;
        }
        if (i >= raw.size()) break;

        std::string token;
        if (raw[i] == '"') {
            ++i;
            while (i < raw.size()) {
                char c = raw[i++];
                if (c == '\\' && i < raw.size()) {
                    token.push_back(raw[i++]);
                    continue;
                }
                if (c == '"') {
                    break;
                }
                token.push_back(c);
            }
        } else {
            size_t start = i;
            while (i < raw.size() && raw[i] != ',') {
                ++i;
            }
            token = raw.substr(start, i - start);
        }
        token = trim_copy(token);
        if (token.empty()) {
            continue;
        }

        HybridLayerType t = HybridLayerType::GATED_ATTENTION;
        if (!parse_layer_type_token(token, t)) {
            return false;
        }
        base.push_back(t);
    }

    return expand_layer_pattern(base, num_layers, out);
}

void apply_default_qwen35_pattern_if_needed(const ModelConfig& config, std::vector<HybridLayerType>& out) {
    if (config.num_layers <= 0 || !is_qwen35_model_type(config.model_type)) {
        return;
    }
    out.resize(static_cast<size_t>(config.num_layers));
    for (int64_t i = 0; i < config.num_layers; ++i) {
        // 默认 3:1（DeltaNet : Attention）交错。
        out[static_cast<size_t>(i)] = ((i + 1) % 4 == 0)
            ? HybridLayerType::GATED_ATTENTION
            : HybridLayerType::DELTANET;
    }
}

}  // namespace

ModelConfig parse_model_config(const std::string& config_path) {
    ModelConfig config;

    std::ifstream file(config_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config file: " + config_path);
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

    auto find_int = [&content](const std::string& key, int64_t& out) -> bool {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        pos = content.find(":", pos);
        if (pos == std::string::npos) return false;
        pos++;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
                                        content[pos] == '\n' || content[pos] == '\r')) {
            pos++;
        }
        size_t end = pos;
        while (end < content.size() &&
               (isdigit(static_cast<unsigned char>(content[end])) || content[end] == '-' || content[end] == '+')) {
            end++;
        }
        if (end == pos) return false;
        try {
            out = std::stoll(content.substr(pos, end - pos));
        } catch (...) {
            return false;
        }
        return true;
    };

    auto find_double = [&content](const std::string& key, double& out) -> bool {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        pos = content.find(":", pos);
        if (pos == std::string::npos) return false;
        pos++;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
                                        content[pos] == '\n' || content[pos] == '\r')) {
            pos++;
        }
        size_t end = pos;
        while (end < content.size()) {
            char c = content[end];
            if (!(isdigit(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')) {
                break;
            }
            end++;
        }
        if (end == pos) return false;
        try {
            out = std::stod(content.substr(pos, end - pos));
        } catch (...) {
            return false;
        }
        return true;
    };

    auto find_string = [&content](const std::string& key, std::string& out) -> bool {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        pos = content.find(":", pos);
        if (pos == std::string::npos) return false;
        pos = content.find("\"", pos);
        if (pos == std::string::npos) return false;
        size_t start = pos + 1;
        size_t end = content.find("\"", start);
        if (end == std::string::npos) return false;
        out = content.substr(start, end - start);
        return true;
    };

    auto find_bool = [&content](const std::string& key, bool& out) -> bool {
        size_t pos = content.find("\"" + key + "\"");
        if (pos == std::string::npos) return false;
        pos = content.find(":", pos);
        if (pos == std::string::npos) return false;
        pos++;
        while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t' ||
                                        content[pos] == '\n' || content[pos] == '\r')) {
            pos++;
        }
        if (content.compare(pos, 4, "true") == 0) {
            out = true;
            return true;
        }
        if (content.compare(pos, 5, "false") == 0) {
            out = false;
            return true;
        }
        return false;
    };

    std::vector<std::string> missing;

    if (!find_string("model_type", config.model_type) || config.model_type.empty()) {
        missing.push_back("model_type");
    }

    if (!find_int("vocab_size", config.vocab_size)) missing.push_back("vocab_size");
    if (!find_int("hidden_size", config.hidden_size)) missing.push_back("hidden_size");
    if (!find_int("num_hidden_layers", config.num_layers)) missing.push_back("num_hidden_layers");
    if (!find_int("num_attention_heads", config.num_heads)) missing.push_back("num_attention_heads");
    if (!find_int("intermediate_size", config.intermediate_size)) missing.push_back("intermediate_size");
    if (!find_bool("tie_word_embeddings", config.tie_word_embeddings)) {
        missing.push_back("tie_word_embeddings");
    }

    if (!find_string("torch_dtype", config.torch_dtype)) {
        config.torch_dtype.clear();
    }

    bool has_kv_heads = find_int("num_key_value_heads", config.num_kv_heads);
    if (!has_kv_heads && config.num_heads > 0) {
        config.num_kv_heads = config.num_heads;
    }

    bool has_head_dim = find_int("head_dim", config.head_dim);
    if (!has_head_dim) {
        if (config.hidden_size > 0 && config.num_heads > 0) {
            if (config.hidden_size % config.num_heads != 0) {
                throw std::runtime_error("hidden_size is not divisible by num_attention_heads");
            }
            config.head_dim = config.hidden_size / config.num_heads;
        } else {
            missing.push_back("head_dim");
        }
    }

    if (!find_double("rope_theta", config.rope_theta)) missing.push_back("rope_theta");
    if (!find_double("rms_norm_eps", config.rms_norm_eps)) missing.push_back("rms_norm_eps");

    if (!find_int("max_position_embeddings", config.max_position_embeddings)) {
        config.max_position_embeddings = 0;
    }

    // 可选 MoE 字段（不同实现命名不同，做兼容解析）
    int64_t tmp = 0;
    if (find_int("num_experts", tmp) || find_int("n_routed_experts", tmp) || find_int("num_local_experts", tmp)) {
        config.num_experts = tmp;
    }
    tmp = 0;
    if (find_int("num_activated_experts", tmp) || find_int("num_experts_per_tok", tmp) || find_int("moe_top_k", tmp)) {
        config.num_activated_experts = tmp;
    }

    // 可选 Hybrid 层布局：优先数组，再字符串，再默认规则。
    std::vector<HybridLayerType> parsed_layer_types;
    std::string layer_array_raw;
    if (find_array_raw(content, "hybrid_layer_pattern", layer_array_raw) ||
        find_array_raw(content, "hybrid_layer_layout", layer_array_raw) ||
        find_array_raw(content, "layer_types", layer_array_raw)) {
        if (!parse_layer_type_array_raw(layer_array_raw, config.num_layers, parsed_layer_types)) {
            throw std::runtime_error("Invalid hybrid layer pattern array");
        }
    } else {
        std::string pattern;
        if (find_string("hybrid_layer_pattern", pattern) || find_string("hybrid_layer_layout", pattern)) {
            if (!parse_pattern_string(pattern, config.num_layers, parsed_layer_types)) {
                throw std::runtime_error("Invalid hybrid layer pattern string: " + pattern);
            }
        }
    }
    if (parsed_layer_types.empty()) {
        apply_default_qwen35_pattern_if_needed(config, parsed_layer_types);
    }
    config.layer_types = std::move(parsed_layer_types);

    if (!missing.empty()) {
        std::ostringstream ss;
        ss << "Missing required model config fields: ";
        for (size_t i = 0; i < missing.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << missing[i];
        }
        throw std::runtime_error(ss.str());
    }

    auto require_positive = [](const char* name, int64_t value) {
        if (value <= 0) {
            throw std::runtime_error(std::string("Invalid value for ") + name + ": " + std::to_string(value));
        }
    };

    require_positive("vocab_size", config.vocab_size);
    require_positive("hidden_size", config.hidden_size);
    require_positive("num_hidden_layers", config.num_layers);
    require_positive("num_attention_heads", config.num_heads);
    require_positive("intermediate_size", config.intermediate_size);
    require_positive("num_key_value_heads", config.num_kv_heads);
    require_positive("head_dim", config.head_dim);

    if (config.num_kv_heads > config.num_heads) {
        throw std::runtime_error("num_key_value_heads cannot exceed num_attention_heads");
    }
    if (config.rope_theta <= 0.0) {
        throw std::runtime_error("rope_theta must be > 0");
    }
    if (config.rms_norm_eps <= 0.0) {
        throw std::runtime_error("rms_norm_eps must be > 0");
    }
    if (config.num_experts < 0) {
        throw std::runtime_error("num_experts must be >= 0");
    }
    if (config.num_activated_experts < 0) {
        throw std::runtime_error("num_activated_experts must be >= 0");
    }
    if (config.num_experts > 0 && config.num_activated_experts > config.num_experts) {
        throw std::runtime_error("num_activated_experts cannot exceed num_experts");
    }
    if (!config.layer_types.empty() && static_cast<int64_t>(config.layer_types.size()) != config.num_layers) {
        throw std::runtime_error("hybrid layer pattern size does not match num_hidden_layers");
    }

    return config;
}

}  // namespace ember
