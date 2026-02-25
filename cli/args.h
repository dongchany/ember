#pragma once

#include <string>
#include <vector>

namespace ember {
namespace cli {

struct Args {
    std::string model_path;
    std::string adapter_path;
    std::string prompt;
    std::vector<int> devices = {0};
    int ctx_size = 2048;
    int n_predict = 128;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repeat_penalty = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    int no_repeat_ngram = 0;
    float lora_scale = 1.0f;
    float memory_fraction = 0.9f;
    bool verbose = false;
    bool interactive = false;
    bool check_mode = false;
    bool phase_aware = false;
    int prefill_chunk_len = 128;
    bool prefill_overlap = true;
    bool temperature_set = false;
    bool top_p_set = false;
    bool top_k_set = false;
    bool repeat_penalty_set = false;
    bool presence_penalty_set = false;
    bool frequency_penalty_set = false;
    bool no_repeat_ngram_set = false;
    int dump_layer = -1;
    std::string dump_dir = "debug";
};

void print_usage(const char* prog);
bool parse_args(int argc, char** argv, Args& args);

}  // namespace cli
}  // namespace ember
