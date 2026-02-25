#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "backends/cuda/cuda_runtime.h"
#include "core/config_loader.h"
#include "core/sampler.h"
#include "core/tokenizer.h"
#include "runtime/runtime_setup.h"

namespace {

[[noreturn]] void die(const std::string& msg) {
    std::cerr << "error: " << msg << "\n";
    std::exit(1);
}

std::vector<int> split_ints(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (!tok.empty()) out.push_back(std::stoi(tok));
    }
    return out;
}

std::string join_with_plus(const std::vector<int>& v) {
    std::ostringstream oss;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) oss << "+";
        oss << v[i];
    }
    return oss.str();
}

double ms_since(std::chrono::high_resolution_clock::time_point t0,
                std::chrono::high_resolution_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

float token_logprob_from_logits(const std::vector<float>& logits, int token_id) {
    if (token_id < 0 || token_id >= static_cast<int>(logits.size())) {
        return -std::numeric_limits<float>::infinity();
    }
    const float max_logit = *std::max_element(logits.begin(), logits.end());
    double sum_exp = 0.0;
    for (float x : logits) {
        sum_exp += std::exp(static_cast<double>(x - max_logit));
    }
    const double log_denom = std::log(sum_exp);
    return static_cast<float>(static_cast<double>(logits[token_id] - max_logit) - log_denom);
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += c; break;
        }
    }
    return out;
}

bool write_candidates_jsonl(const std::filesystem::path& path,
                            const std::vector<std::vector<int>>& tokens,
                            const std::vector<std::vector<float>>& token_logprobs,
                            const std::vector<std::string>& texts,
                            const std::vector<std::string>& finish_reasons) {
    std::ofstream out(path);
    if (!out.is_open()) return false;

    for (size_t i = 0; i < tokens.size(); ++i) {
        const auto& t = tokens[i];
        const auto& lp = token_logprobs[i];
        double sum_lp = std::accumulate(lp.begin(), lp.end(), 0.0);
        double avg_lp = lp.empty() ? 0.0 : (sum_lp / static_cast<double>(lp.size()));

        out << "{";
        out << "\"candidate_id\":" << i << ",";
        out << "\"num_tokens\":" << t.size() << ",";
        out << "\"sum_logprob\":" << std::fixed << std::setprecision(6) << sum_lp << ",";
        out << "\"avg_logprob\":" << std::fixed << std::setprecision(6) << avg_lp << ",";
        out << "\"finish_reason\":\""
            << json_escape(i < finish_reasons.size() ? finish_reasons[i] : std::string("unknown"))
            << "\",";

        out << "\"tokens\":[";
        for (size_t j = 0; j < t.size(); ++j) {
            if (j) out << ",";
            out << t[j];
        }
        out << "],";

        out << "\"token_logprobs\":[";
        for (size_t j = 0; j < lp.size(); ++j) {
            if (j) out << ",";
            out << std::fixed << std::setprecision(6) << lp[j];
        }
        out << "],";

        out << "\"text\":\"" << json_escape(i < texts.size() ? texts[i] : std::string()) << "\"";
        out << "}\n";
    }
    return true;
}

std::vector<int> sample_random_prompt(int prompt_len, int vocab_size, int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, std::max(1, vocab_size - 1));
    std::vector<int> tokens(static_cast<size_t>(prompt_len));
    for (int& t : tokens) t = dist(rng);
    return tokens;
}

bool has_suffix(const std::vector<int>& seq, const std::vector<int>& pattern) {
    if (pattern.empty() || seq.size() < pattern.size()) return false;
    const size_t off = seq.size() - pattern.size();
    for (size_t i = 0; i < pattern.size(); ++i) {
        if (seq[off + i] != pattern[i]) return false;
    }
    return true;
}

int find_matching_stop_seq(const std::vector<int>& generated,
                           const std::vector<std::vector<int>>& stop_seqs) {
    for (size_t i = 0; i < stop_seqs.size(); ++i) {
        if (has_suffix(generated, stop_seqs[i])) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir;
    std::string prompt_text;
    int prompt_len = 256;
    int gen_len = 128;
    int num_candidates = 8;
    std::vector<int> gpus = {0, 1};
    std::vector<int> split = {};
    bool overlap = true;
    int chunk_len = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int top_k = 40;
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0f;
    float frequency_penalty = 0.0f;
    int no_repeat_ngram_size = 0;
    std::vector<std::string> stop_seq_texts;
    bool strip_stop = true;
    int seed = 1234;
    bool decode_text = true;
    std::string csv_path;
    std::string candidates_jsonl_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + name);
            return argv[++i];
        };
        if (arg == "-h" || arg == "--help") {
            std::cout
                << "Ember Multi-Candidate Rollout Benchmark\n\n"
                << "Usage:\n"
                << "  " << argv[0] << " --model <dir> [options]\n\n"
                << "Options:\n"
                << "  --model <dir>         model directory\n"
                << "  --prompt TEXT         prompt text (if empty, use random token prompt)\n"
                << "  --prompt-len N        random prompt length (default: 256)\n"
                << "  --gen-len N           generation length per candidate (default: 128)\n"
                << "  --num-candidates N    number of candidates (default: 8)\n"
                << "  --gpus LIST           e.g. 0 or 0,1 (default: 0,1)\n"
                << "  --split A,B           layer split for 2 GPUs (default: even)\n"
                << "  --chunk-len N         chunk length for 2-GPU slot prefill (default: 512)\n"
                << "  --overlap             enable overlap for 2-GPU slot prefill (default: on)\n"
                << "  --no-overlap          disable overlap\n"
                << "  --temperature F       sampling temperature (default: 0.7)\n"
                << "  --top-p F             top-p (default: 0.9)\n"
                << "  --top-k N             top-k (default: 40)\n"
                << "  --repetition-penalty F (default: 1.0)\n"
                << "  --presence-penalty F  (default: 0.0)\n"
                << "  --frequency-penalty F (default: 0.0)\n"
                << "  --no-repeat-ngram-size N (default: 0)\n"
                << "  --stop-seq TEXT       stop sequence string (repeatable)\n"
                << "  --no-strip-stop       keep stop sequence tokens in output\n"
                << "  --seed N              RNG seed (default: 1234)\n"
                << "  --no-decode-text      skip candidate text decode\n"
                << "  --csv PATH            write summary CSV row\n"
                << "  --candidates-jsonl PATH write candidate details\n";
            return 0;
        } else if (arg == "--model") {
            model_dir = need("--model");
        } else if (arg == "--prompt") {
            prompt_text = need("--prompt");
        } else if (arg == "--prompt-len") {
            prompt_len = std::stoi(need("--prompt-len"));
        } else if (arg == "--gen-len") {
            gen_len = std::stoi(need("--gen-len"));
        } else if (arg == "--num-candidates") {
            num_candidates = std::stoi(need("--num-candidates"));
        } else if (arg == "--gpus") {
            gpus = split_ints(need("--gpus"));
        } else if (arg == "--split") {
            split = split_ints(need("--split"));
        } else if (arg == "--chunk-len") {
            chunk_len = std::stoi(need("--chunk-len"));
        } else if (arg == "--overlap") {
            overlap = true;
        } else if (arg == "--no-overlap") {
            overlap = false;
        } else if (arg == "--temperature") {
            temperature = std::stof(need("--temperature"));
        } else if (arg == "--top-p") {
            top_p = std::stof(need("--top-p"));
        } else if (arg == "--top-k") {
            top_k = std::stoi(need("--top-k"));
        } else if (arg == "--repetition-penalty") {
            repetition_penalty = std::stof(need("--repetition-penalty"));
        } else if (arg == "--presence-penalty") {
            presence_penalty = std::stof(need("--presence-penalty"));
        } else if (arg == "--frequency-penalty") {
            frequency_penalty = std::stof(need("--frequency-penalty"));
        } else if (arg == "--no-repeat-ngram-size") {
            no_repeat_ngram_size = std::stoi(need("--no-repeat-ngram-size"));
        } else if (arg == "--stop-seq") {
            stop_seq_texts.push_back(need("--stop-seq"));
        } else if (arg == "--no-strip-stop") {
            strip_stop = false;
        } else if (arg == "--seed") {
            seed = std::stoi(need("--seed"));
        } else if (arg == "--no-decode-text") {
            decode_text = false;
        } else if (arg == "--csv") {
            csv_path = need("--csv");
        } else if (arg == "--candidates-jsonl") {
            candidates_jsonl_path = need("--candidates-jsonl");
        } else {
            die("unknown argument: " + arg);
        }
    }

    if (model_dir.empty()) die("--model is required");
    if (prompt_len <= 0) die("--prompt-len must be > 0");
    if (gen_len <= 0) die("--gen-len must be > 0");
    if (num_candidates <= 0) die("--num-candidates must be > 0");
    if (gpus.empty()) die("--gpus is empty");
    if (!split.empty() && split.size() != 2) die("--split expects A,B");
    if (gpus.size() > 2) die("benchmark supports only 1 or 2 GPUs");

    ember::ModelConfig config;
    try {
        config = ember::parse_model_config(model_dir + "/config.json");
    } catch (const std::exception& ex) {
        die(std::string("parse_model_config failed: ") + ex.what());
    }

    ember::HFTokenizer tokenizer;
    bool tokenizer_ok = !tokenizer.load(model_dir);

    std::vector<int> prompt_tokens;
    if (!prompt_text.empty()) {
        if (!tokenizer_ok) {
            die("tokenizer load failed but --prompt text was provided");
        }
        prompt_tokens = tokenizer.encode(prompt_text, /*add_special_tokens=*/true);
    } else {
        prompt_tokens = sample_random_prompt(prompt_len, config.vocab_size, seed);
    }
    if (prompt_tokens.empty()) die("empty prompt tokens");
    prompt_len = static_cast<int>(prompt_tokens.size());

    std::vector<std::vector<int>> stop_seq_tokens;
    if (!stop_seq_texts.empty()) {
        if (!tokenizer_ok) {
            die("tokenizer load failed; --stop-seq requires tokenizer");
        }
        for (const std::string& s : stop_seq_texts) {
            if (s.empty()) continue;
            std::vector<int> ids = tokenizer.encode(s, /*add_special_tokens=*/false);
            if (!ids.empty()) {
                stop_seq_tokens.push_back(std::move(ids));
            }
        }
    }

    auto runtime = ember::RuntimeFactory::create_cuda();
    if (!runtime || !runtime->available()) die("CUDA runtime not available");
    auto* cuda_rt = dynamic_cast<ember::cuda::CudaRuntime*>(runtime.get());
    if (!cuda_rt) die("expected CUDA runtime");

    ember::DeviceMap device_map;
    if (gpus.size() == 1) {
        device_map = ember::DeviceMap::single_device(config.num_layers, gpus[0]);
    } else {
        int a = split.empty() ? (static_cast<int>(config.num_layers) / 2) : split[0];
        int b = split.empty() ? (static_cast<int>(config.num_layers) - a) : split[1];
        if (a <= 0 || b <= 0 || a + b != config.num_layers) die("invalid --split");
        device_map.num_devices = 2;
        device_map.embedding_device = gpus[0];
        device_map.lm_head_device = gpus[1];
        device_map.layer_to_device.resize(static_cast<size_t>(config.num_layers));
        for (int i = 0; i < config.num_layers; ++i) {
            device_map.layer_to_device[static_cast<size_t>(i)] = (i < a) ? gpus[0] : gpus[1];
        }
    }

    ember::RuntimeConfig runtime_config;
    runtime_config.max_ctx_len = prompt_len + gen_len + 8;
    runtime_config.batch_size = num_candidates;
    runtime_config.device_ids = gpus;
    runtime_config.kv_cache_dtype = ember::dtype_from_string(config.torch_dtype);
    if (runtime_config.kv_cache_dtype == ember::DType::UNKNOWN) runtime_config.kv_cache_dtype = ember::DType::F16;
    runtime_config.temperature = temperature;
    runtime_config.top_p = top_p;
    runtime_config.top_k = top_k;
    runtime_config.repetition_penalty = repetition_penalty;
    runtime_config.presence_penalty = presence_penalty;
    runtime_config.frequency_penalty = frequency_penalty;
    runtime_config.no_repeat_ngram_size = no_repeat_ngram_size;

    ember::RuntimeSetup setup;
    ember::Error err = ember::load_runtime(*runtime, model_dir, config, device_map, setup);
    if (err) die("load_runtime failed: " + err.to_string());
    err = ember::init_session_and_kv(*runtime, config, runtime_config, setup);
    if (err) die("init_session_and_kv failed: " + err.to_string());

    ember::Sampler sampler(temperature, top_k, top_p);
    sampler.set_seed(static_cast<uint64_t>(seed));

    std::vector<std::vector<int>> generated(static_cast<size_t>(num_candidates));
    std::vector<std::vector<float>> token_logprobs(static_cast<size_t>(num_candidates));
    std::vector<std::vector<int>> histories(static_cast<size_t>(num_candidates), prompt_tokens);
    std::vector<int> last_tokens(static_cast<size_t>(num_candidates), 0);
    std::vector<bool> finished(static_cast<size_t>(num_candidates), false);
    std::vector<std::string> finish_reasons(static_cast<size_t>(num_candidates), "max_len");

    const int eos_id = tokenizer_ok ? tokenizer.eos_token_id() : -1;

    // Prefill each slot and sample first token from prefill logits.
    auto t_prefill0 = std::chrono::high_resolution_clock::now();
    for (int slot = 0; slot < num_candidates; ++slot) {
        std::vector<float> logits;
        if (gpus.size() == 2) {
            err = cuda_rt->prefill_into_slot_pipeline(prompt_tokens, slot, setup.session, chunk_len, overlap, &logits);
        } else {
            err = cuda_rt->prefill_into_slot(prompt_tokens, slot, setup.session, &logits);
        }
        if (err) {
            die("prefill_into_slot failed at slot " + std::to_string(slot) + ": " + err.to_string());
        }

        const int tok = sampler.sample(logits, runtime_config, histories[static_cast<size_t>(slot)]);
        const float lp = token_logprob_from_logits(logits, tok);
        generated[static_cast<size_t>(slot)].push_back(tok);
        token_logprobs[static_cast<size_t>(slot)].push_back(lp);
        histories[static_cast<size_t>(slot)].push_back(tok);
        last_tokens[static_cast<size_t>(slot)] = tok;
        if (eos_id >= 0 && tok == eos_id) {
            finished[static_cast<size_t>(slot)] = true;
            finish_reasons[static_cast<size_t>(slot)] = "eos";
            setup.session.set_inactive(slot);
            continue;
        }
        const int stop_idx = find_matching_stop_seq(generated[static_cast<size_t>(slot)], stop_seq_tokens);
        if (stop_idx >= 0) {
            if (strip_stop) {
                const size_t n = stop_seq_tokens[static_cast<size_t>(stop_idx)].size();
                if (n > 0 && n <= generated[static_cast<size_t>(slot)].size()) {
                    generated[static_cast<size_t>(slot)].resize(generated[static_cast<size_t>(slot)].size() - n);
                    token_logprobs[static_cast<size_t>(slot)].resize(token_logprobs[static_cast<size_t>(slot)].size() - n);
                }
            }
            finished[static_cast<size_t>(slot)] = true;
            finish_reasons[static_cast<size_t>(slot)] = "stop_seq";
            setup.session.set_inactive(slot);
        }
    }
    auto t_prefill1 = std::chrono::high_resolution_clock::now();
    const double prefill_ms = ms_since(t_prefill0, t_prefill1);

    // Decode batched steps for remaining tokens.
    auto t_decode0 = std::chrono::high_resolution_clock::now();
    for (int step = 1; step < gen_len; ++step) {
        bool any_active = false;
        for (int slot = 0; slot < num_candidates; ++slot) {
            if (!finished[static_cast<size_t>(slot)]) {
                any_active = true;
                break;
            }
        }
        if (!any_active) break;

        std::vector<float> logits_flat;
        err = cuda_rt->decode_batch(last_tokens, setup.session, logits_flat);
        if (err) die("decode_batch failed: " + err.to_string());

        const size_t vocab = static_cast<size_t>(config.vocab_size);
        if (logits_flat.size() != static_cast<size_t>(num_candidates) * vocab) {
            die("unexpected logits_flat shape");
        }

        for (int slot = 0; slot < num_candidates; ++slot) {
            if (finished[static_cast<size_t>(slot)]) continue;
            const size_t off = static_cast<size_t>(slot) * vocab;
            std::vector<float> row(logits_flat.begin() + static_cast<std::ptrdiff_t>(off),
                                   logits_flat.begin() + static_cast<std::ptrdiff_t>(off + vocab));
            const int tok = sampler.sample(row, runtime_config, histories[static_cast<size_t>(slot)]);
            const float lp = token_logprob_from_logits(row, tok);
            generated[static_cast<size_t>(slot)].push_back(tok);
            token_logprobs[static_cast<size_t>(slot)].push_back(lp);
            histories[static_cast<size_t>(slot)].push_back(tok);
            last_tokens[static_cast<size_t>(slot)] = tok;
            if (eos_id >= 0 && tok == eos_id) {
                finished[static_cast<size_t>(slot)] = true;
                finish_reasons[static_cast<size_t>(slot)] = "eos";
                setup.session.set_inactive(slot);
                continue;
            }
            const int stop_idx = find_matching_stop_seq(generated[static_cast<size_t>(slot)], stop_seq_tokens);
            if (stop_idx >= 0) {
                if (strip_stop) {
                    const size_t n = stop_seq_tokens[static_cast<size_t>(stop_idx)].size();
                    if (n > 0 && n <= generated[static_cast<size_t>(slot)].size()) {
                        generated[static_cast<size_t>(slot)].resize(generated[static_cast<size_t>(slot)].size() - n);
                        token_logprobs[static_cast<size_t>(slot)].resize(token_logprobs[static_cast<size_t>(slot)].size() - n);
                    }
                }
                finished[static_cast<size_t>(slot)] = true;
                finish_reasons[static_cast<size_t>(slot)] = "stop_seq";
                setup.session.set_inactive(slot);
            }
        }
    }
    auto t_decode1 = std::chrono::high_resolution_clock::now();
    const double decode_ms = ms_since(t_decode0, t_decode1);

    const double total_ms = prefill_ms + decode_ms;
    size_t total_gen_tokens = 0;
    for (const auto& v : generated) total_gen_tokens += v.size();
    const double gen_tok_s = total_ms > 0.0 ? (static_cast<double>(total_gen_tokens) * 1000.0 / total_ms) : 0.0;

    std::vector<std::string> decoded_texts(static_cast<size_t>(num_candidates));
    if (decode_text && tokenizer_ok) {
        for (int slot = 0; slot < num_candidates; ++slot) {
            decoded_texts[static_cast<size_t>(slot)] = tokenizer.decode(generated[static_cast<size_t>(slot)], true);
        }
    }

    if (!candidates_jsonl_path.empty()) {
        if (!write_candidates_jsonl(candidates_jsonl_path, generated, token_logprobs, decoded_texts, finish_reasons)) {
            die("failed to write candidates jsonl: " + candidates_jsonl_path);
        }
    }

    std::ostringstream row;
    row << std::fixed << std::setprecision(3)
        << "multi_candidate_rollout" << ","
        << prompt_len << ","
        << gen_len << ","
        << num_candidates << ","
        << join_with_plus(gpus) << ","
        << join_with_plus(split) << ","
        << prefill_ms << ","
        << decode_ms << ","
        << total_ms << ","
        << total_gen_tokens << ","
        << gen_tok_s << ","
        << temperature << ","
        << top_p << ","
        << top_k << ","
        << stop_seq_tokens.size();

    const std::string header =
        "mode,prompt_len,gen_len,num_candidates,gpus,split,prefill_ms,decode_ms,total_ms,total_gen_tokens,gen_tok_s,"
        "temperature,top_p,top_k,num_stop_sequences";

    if (!csv_path.empty()) {
        std::ofstream out(csv_path);
        if (!out.is_open()) die("failed to open csv path: " + csv_path);
        out << header << "\n" << row.str() << "\n";
    } else {
        std::cout << header << "\n" << row.str() << "\n";
    }

    return 0;
}
