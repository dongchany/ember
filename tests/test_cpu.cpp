#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config_loader.h"
#include "core/sampler.h"
#include "formats/safetensors.h"

namespace {

class TestFailure : public std::runtime_error {
public:
    explicit TestFailure(const std::string& msg) : std::runtime_error(msg) {}
};

void expect_true(bool cond, const char* expr, const char* file, int line) {
    if (!cond) {
        std::ostringstream oss;
        oss << file << ":" << line << " EXPECT_TRUE(" << expr << ") failed";
        throw TestFailure(oss.str());
    }
}

template <typename A, typename B>
void expect_eq(const A& a, const B& b, const char* expr_a, const char* expr_b,
               const char* file, int line) {
    if (!(a == b)) {
        std::ostringstream oss;
        oss << file << ":" << line << " EXPECT_EQ(" << expr_a << ", " << expr_b
            << ") failed: " << a << " vs " << b;
        throw TestFailure(oss.str());
    }
}

#define EXPECT_TRUE(cond) expect_true((cond), #cond, __FILE__, __LINE__)
#define EXPECT_EQ(a, b) expect_eq((a), (b), #a, #b, __FILE__, __LINE__)

struct TempFile {
    std::filesystem::path path;

    explicit TempFile(std::filesystem::path p) : path(std::move(p)) {}
    TempFile(const TempFile&) = delete;
    TempFile& operator=(const TempFile&) = delete;

    TempFile(TempFile&& other) noexcept : path(std::move(other.path)) {
        other.path.clear();
    }
    TempFile& operator=(TempFile&& other) noexcept {
        if (this != &other) {
            cleanup();
            path = std::move(other.path);
            other.path.clear();
        }
        return *this;
    }

    ~TempFile() { cleanup(); }

    void cleanup() {
        if (!path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
};

TempFile write_temp_json(const std::string& content) {
    auto dir = std::filesystem::temp_directory_path();
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto path = dir / ("ember_test_config_" + std::to_string(stamp) + ".json");
    std::ofstream out(path);
    if (!out.is_open()) {
        throw TestFailure("Failed to create temp config file");
    }
    out << content;
    out.close();
    return TempFile(path);
}

void test_config_loader_defaults() {
    const char* json = R"({
  "model_type": "qwen3",
  "vocab_size": 100,
  "hidden_size": 64,
  "num_hidden_layers": 2,
  "num_attention_heads": 8,
  "intermediate_size": 256,
  "tie_word_embeddings": true,
  "rope_theta": 10000.0,
  "rms_norm_eps": 1e-6
})";
    TempFile tmp = write_temp_json(json);
    ember::ModelConfig config = ember::parse_model_config(tmp.path.string());
    EXPECT_EQ(config.model_type, std::string("qwen3"));
    EXPECT_EQ(config.num_kv_heads, config.num_heads);
    EXPECT_EQ(config.head_dim, config.hidden_size / config.num_heads);
}

void test_config_loader_missing_field() {
    const char* json = R"({
  "model_type": "qwen3",
  "hidden_size": 64,
  "num_hidden_layers": 2,
  "num_attention_heads": 8,
  "intermediate_size": 256,
  "tie_word_embeddings": true,
  "rope_theta": 10000.0,
  "rms_norm_eps": 1e-6
})";
    TempFile tmp = write_temp_json(json);
    bool threw = false;
    try {
        ember::parse_model_config(tmp.path.string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

void test_config_loader_invalid_head_dim() {
    const char* json = R"({
  "model_type": "qwen3",
  "vocab_size": 100,
  "hidden_size": 63,
  "num_hidden_layers": 2,
  "num_attention_heads": 8,
  "intermediate_size": 256,
  "tie_word_embeddings": true,
  "rope_theta": 10000.0,
  "rms_norm_eps": 1e-6
})";
    TempFile tmp = write_temp_json(json);
    bool threw = false;
    try {
        ember::parse_model_config(tmp.path.string());
    } catch (const std::runtime_error&) {
        threw = true;
    }
    EXPECT_TRUE(threw);
}

void test_sampler_greedy() {
    ember::Sampler sampler;
    ember::RuntimeConfig cfg;
    cfg.temperature = 0.0f;
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    std::vector<float> logits = {1.0f, 5.0f, 2.0f, 3.0f};
    int token = sampler.sample(logits, cfg);
    EXPECT_EQ(token, 1);
}

void test_sampler_top_k() {
    ember::Sampler sampler;
    sampler.set_seed(123);
    ember::RuntimeConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 2;
    cfg.top_p = 1.0f;
    std::vector<float> logits = {1.0f, 5.0f, 2.0f, 3.0f};
    for (int i = 0; i < 100; ++i) {
        int token = sampler.sample(logits, cfg);
        EXPECT_TRUE(token == 1 || token == 3);
    }
}

void test_sampler_no_repeat_ngram() {
    ember::Sampler sampler;
    sampler.set_seed(7);
    ember::RuntimeConfig cfg;
    cfg.temperature = 0.0f;
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.no_repeat_ngram_size = 3;
    std::vector<float> logits(10, 0.0f);
    logits[6] = 5.0f;
    logits[2] = 4.0f;
    std::vector<int> history = {5, 6, 7, 6, 7};
    int token = sampler.sample(logits, cfg, history);
    EXPECT_EQ(token, 2);
}

void test_safetensors_dtype_map() {
    EXPECT_EQ(static_cast<int>(ember::safetensors_dtype_to_ember("F32")),
              static_cast<int>(ember::DType::F32));
    EXPECT_EQ(static_cast<int>(ember::safetensors_dtype_to_ember("F16")),
              static_cast<int>(ember::DType::F16));
    EXPECT_EQ(static_cast<int>(ember::safetensors_dtype_to_ember("BF16")),
              static_cast<int>(ember::DType::BF16));
    EXPECT_EQ(static_cast<int>(ember::safetensors_dtype_to_ember("I8")),
              static_cast<int>(ember::DType::INT8));
    EXPECT_EQ(static_cast<int>(ember::safetensors_dtype_to_ember("UNKNOWN")),
              static_cast<int>(ember::DType::UNKNOWN));
}

void test_safetensors_missing_file() {
    auto dir = std::filesystem::temp_directory_path();
    auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    auto path = dir / ("ember_missing_" + std::to_string(stamp) + ".safetensors");
    ember::SafetensorsReader reader;
    ember::Error err = reader.open(path.string());
    EXPECT_TRUE(!err.ok());
    EXPECT_EQ(static_cast<int>(err.code()), static_cast<int>(ember::ErrorCode::FILE_NOT_FOUND));
}

struct TestCase {
    const char* name;
    void (*fn)();
};

}  // namespace

int main() {
    const TestCase tests[] = {
        {"config_loader_defaults", test_config_loader_defaults},
        {"config_loader_missing_field", test_config_loader_missing_field},
        {"config_loader_invalid_head_dim", test_config_loader_invalid_head_dim},
        {"sampler_greedy", test_sampler_greedy},
        {"sampler_top_k", test_sampler_top_k},
        {"sampler_no_repeat_ngram", test_sampler_no_repeat_ngram},
        {"safetensors_dtype_map", test_safetensors_dtype_map},
        {"safetensors_missing_file", test_safetensors_missing_file},
    };

    int passed = 0;
    int failed = 0;
    for (const auto& test : tests) {
        try {
            test.fn();
            std::cout << "[PASS] " << test.name << "\n";
            ++passed;
        } catch (const std::exception& ex) {
            std::cout << "[FAIL] " << test.name << ": " << ex.what() << "\n";
            ++failed;
        }
    }

    std::cout << "Passed: " << passed << ", Failed: " << failed << "\n";
    return failed == 0 ? 0 : 1;
}
