#include "core/config_loader.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ember {

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

    return config;
}

}  // namespace ember
