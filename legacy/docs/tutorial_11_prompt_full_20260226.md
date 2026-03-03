# Tutorial #11 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 11 篇。

## 项目简介
Ember (https://github.com/dongchany/ember) 是一个从零手写的 Qwen3 CUDA 推理引擎，
纯 C++ + CUDA，不依赖 ggml/llama.cpp。支持消费级多 GPU Pipeline Parallel（如双卡 RTX 3080Ti）。
除了推理，Ember 还支持完整的 RL 训练闭环：多候选 Rollout → Verifier/Reward → LoRA 热更新 →
Cache 策略复用，实现了统一后端（推理和训练共享同份权重），相比双栈方案节省 50% 显存。

## 项目 5 层结构
Layer 1: 推理引擎（CUDA kernels, Transformer forward, Pipeline Parallel）
Layer 2: Rollout 能力（多候选、logprobs、stop sequences）
Layer 3: LoRA 热更新 + Cache 策略（UpdateLocality / Prefix / Periodic / Hybrid）
Layer 4: 验证器 + Reward（Extraction / SQL verifier，字段级打分）
Layer 5: 训练闭环（SFT → Best-of-N → DPO → GRPO 可选）+ 统一后端 vs 双栈

## 写作硬性要求
1. 目标读者：想了解 LLM 内部原理的开发者，数学基础较弱也能看懂
2. 数学四步法：直觉 → 小例子手算 → 公式 → 对应 CUDA/训练代码
3. 语言：中文为主，术语和代码注释保留英文
4. 必须引用我提供的真实源码与真实报告，不得编造实验数字
5. 每篇开头必须写：源文件路径、前置知识、下一篇链接
6. 每篇结尾自然放 GitHub 链接：https://github.com/dongchany/ember
7. 风格：友好、像学长讲解，不要居高临下
8. 不要只列 bullet；以叙述为主

## 输出质量要求（必须遵守）
- 你只能使用我提供的“完整代码片段”和“完整报告片段”作为事实来源
- 所有结论都要标注来自哪个文件
- 任何数字都要能在报告中定位到
- 如果某结论缺证据，明确写“当前资料不足”

## 数学深度加严（额外要求）
- 在不影响可读性的前提下，尽量给出更详细的数学推导
- 对每个关键公式都解释“它在数值稳定性/并行实现上的意义”
- 允许在附录给出更完整推导（正文保持循序渐进）
```

---

## 1) 写作任务

```text
请写第 11 篇：多候选 Rollout — 为 RL 生成 N 条候选。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 强调这篇是 Best-of-N / DPO 的前置
- 解释 stop sequences、logprobs 导出和 same-seed 一致性验证
```

---

## 2) 代码上下文（完整/相关段落）

### File: benchmarks/multi_candidate_rollout.cpp

````cpp
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

````

### File: runtime/batch_runtime.h

````h
#pragma once

#include "../core/error.h"
#include "../core/session.h"

#include <vector>

namespace ember {

// Optional runtime extension for batching / phase-aware scheduling.
// Backends may implement this interface to expose extra capabilities without
// forcing them into the base IRuntime API.
class IBatchRuntime {
public:
    virtual ~IBatchRuntime() = default;

    // Chunked 2-GPU prefill pipeline (batch_size=1 execution).
    virtual Error prefill_chunked_pipeline(const std::vector<int>& tokens,
                                           Session& session,
                                           int chunk_len,
                                           bool overlap,
                                           std::vector<float>* out_logits) = 0;

    // Decode one step for batch_size>1, returning logits to host.
    virtual Error decode_batch(const std::vector<int>& last_tokens,
                               Session& session,
                               std::vector<float>& logits_flat) = 0;

    // Prefill a single request into a specific batch slot (KV cache slice).
    virtual Error prefill_into_slot(const std::vector<int>& tokens,
                                    int slot,
                                    Session& session,
                                    std::vector<float>* out_logits) = 0;

    // Prefill into a specific slot using chunked pipeline (when multi-GPU).
    virtual Error prefill_into_slot_pipeline(const std::vector<int>& tokens,
                                             int slot,
                                             Session& session,
                                             int chunk_len,
                                             bool overlap,
                                             std::vector<float>* out_logits) = 0;

    // Decode one step and return greedy next tokens (argmax over logits) for each slot.
    virtual Error decode_batch_greedy(const std::vector<int>& last_tokens,
                                      Session& session,
                                      std::vector<int>& next_tokens) = 0;
};

}  // namespace ember


````

### File: scripts/report/run_stage2_multi_candidate.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from common_report import die, read_csv, run_cmd, safe_float


def hf_hub_root() -> Path:
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "hub")
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def resolve_snapshot_dir(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if (path / "config.json").exists() and list(path.glob("*.safetensors")):
        return path
    snap_root = path / "snapshots"
    if not snap_root.exists():
        return None
    candidates = [
        p
        for p in snap_root.iterdir()
        if p.is_dir() and (p / "config.json").exists() and list(p.glob("*.safetensors"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_model_dir(model_arg: str) -> Path:
    raw = model_arg.strip()
    if not raw:
        die("--model is empty")
    p = Path(raw).expanduser().resolve()
    resolved = resolve_snapshot_dir(p)
    if resolved is not None:
        return resolved
    hub_root = hf_hub_root()
    model_cache_dir = hub_root / ("models--" + raw.replace("/", "--"))
    resolved = resolve_snapshot_dir(model_cache_dir)
    if resolved is not None:
        return resolved
    die(
        "failed to resolve model from local cache: "
        f"{raw}. Checked path='{p}' and HF cache='{model_cache_dir}'."
    )
    raise AssertionError("unreachable")


def load_candidates(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def write_summary_md(path: Path, model_dir: Path, row: Dict[str, str], cand: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# Stage 2.1 Multi-Candidate Rollout")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("| --- | --- |")
    for key in [
        "prompt_len",
        "gen_len",
        "num_candidates",
        "num_stop_sequences",
        "gpus",
        "split",
        "prefill_ms",
        "decode_ms",
        "total_ms",
        "total_gen_tokens",
        "gen_tok_s",
        "temperature",
        "top_p",
        "top_k",
    ]:
        lines.append(f"| {key} | `{row.get(key, '')}` |")

    if cand:
        avg_lp = sum(float(x.get("avg_logprob", 0.0)) for x in cand) / float(len(cand))
        best = max(cand, key=lambda x: float(x.get("sum_logprob", -1e9)))
        lines.append("")
        lines.append("## Candidate Stats")
        lines.append(f"- mean(avg_logprob): `{avg_lp:.6f}`")
        lines.append(f"- best candidate id: `{best.get('candidate_id')}`")
        lines.append(f"- best sum_logprob: `{float(best.get('sum_logprob', 0.0)):.6f}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_p1_input(path: Path, row: Dict[str, str], cand: List[Dict[str, object]]) -> None:
    lines: List[str] = []
    lines.append("# P1/P4 Input — Multi-Candidate Rollout")
    lines.append("")
    lines.append(
        f"- Rollout throughput: `{row.get('gen_tok_s','')}` tok/s "
        f"(total_gen_tokens={row.get('total_gen_tokens','')}, total_ms={row.get('total_ms','')})."
    )
    lines.append(
        f"- Multi-candidate setting: num_candidates=`{row.get('num_candidates','')}`, "
        f"gen_len=`{row.get('gen_len','')}`."
    )
    if cand:
        avg_lp = sum(float(x.get("avg_logprob", 0.0)) for x in cand) / float(len(cand))
        lines.append(f"- Mean per-candidate avg_logprob: `{avg_lp:.6f}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Stage 2.1 multi-candidate rollout benchmark.")
    ap.add_argument("--model", type=str, required=True, help="model path or HF model id")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--gen-len", type=int, default=128)
    ap.add_argument("--num-candidates", type=int, default=8)
    ap.add_argument("--gpus", type=str, default="0,1")
    ap.add_argument("--split", type=str, default="9,27")
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--overlap", action="store_true", default=True)
    ap.add_argument("--no-overlap", dest="overlap", action="store_false")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--stop-seqs", type=str, default="", help="stop sequences joined by ||, e.g. '<|im_end|>||###'")
    ap.add_argument("--strip-stop", action="store_true", default=True)
    ap.add_argument("--no-strip-stop", dest="strip_stop", action="store_false")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--decode-text", action="store_true", default=True)
    ap.add_argument("--no-decode-text", dest="decode_text", action="store_false")
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.gen_len <= 0 or args.num_candidates <= 0:
        die("prompt/gen/candidates must be > 0")

    model_dir = resolve_model_dir(args.model)
    repo = Path.cwd()
    bench_bin = (repo / args.build_dir / "ember_multi_candidate_rollout").resolve()
    if not bench_bin.exists():
        die(f"missing binary: {bench_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage21_multi_candidate_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "stage21_multi_candidate.csv"
    cand_path = out_dir / "stage21_candidates.jsonl"

    cmd = [
        str(bench_bin),
        "--model",
        str(model_dir),
        "--prompt-len",
        str(args.prompt_len),
        "--gen-len",
        str(args.gen_len),
        "--num-candidates",
        str(args.num_candidates),
        "--gpus",
        args.gpus,
        "--split",
        args.split,
        "--chunk-len",
        str(args.chunk_len),
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--top-k",
        str(args.top_k),
        "--seed",
        str(args.seed),
        "--csv",
        str(csv_path),
        "--candidates-jsonl",
        str(cand_path),
    ]
    if args.stop_seqs.strip():
        stop_list = [x for x in args.stop_seqs.split("||") if x]
        for s in stop_list:
            cmd += ["--stop-seq", s]
    if not args.strip_stop:
        cmd += ["--no-strip-stop"]
    if args.prompt:
        cmd += ["--prompt", args.prompt]
    if args.overlap:
        cmd += ["--overlap"]
    else:
        cmd += ["--no-overlap"]
    if not args.decode_text:
        cmd += ["--no-decode-text"]

    p = run_cmd(cmd, cwd=repo, log_path=logs_dir / "run.log", check=False)
    if p.returncode != 0:
        die(f"benchmark failed rc={p.returncode}; see {logs_dir / 'run.log'}")

    rows = read_csv(csv_path)
    if not rows:
        die(f"empty csv: {csv_path}")
    row = rows[0]
    cand = load_candidates(cand_path)

    write_summary_md(out_dir / "stage21_summary.md", model_dir, row, cand)
    write_p1_input(out_dir / "stage21_p1_input.md", row, cand)

    print(f"[done] out_dir={out_dir}")
    print(
        f"[result] candidates={row.get('num_candidates','')} "
        f"gen_tok_s={safe_float(row.get('gen_tok_s','0')):.3f} "
        f"total_ms={safe_float(row.get('total_ms','0')):.3f}"
    )


if __name__ == "__main__":
    main()

````

### File: scripts/report/run_stage2_numeric_consistency.py

````py
#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from common_report import die, run_cmd, write_csv


def hf_hub_root() -> Path:
    hub_cache = os.environ.get("HUGGINGFACE_HUB_CACHE", "").strip()
    if hub_cache:
        return Path(hub_cache).expanduser().resolve()
    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return (Path(hf_home).expanduser().resolve() / "hub")
    return (Path.home() / ".cache" / "huggingface" / "hub").resolve()


def resolve_snapshot_dir(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    if (path / "config.json").exists() and list(path.glob("*.safetensors")):
        return path
    snap_root = path / "snapshots"
    if not snap_root.exists():
        return None
    candidates = [
        p
        for p in snap_root.iterdir()
        if p.is_dir() and (p / "config.json").exists() and list(p.glob("*.safetensors"))
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def resolve_model_dir(model_arg: str) -> Path:
    raw = model_arg.strip()
    if not raw:
        die("--model is empty")
    p = Path(raw).expanduser().resolve()
    resolved = resolve_snapshot_dir(p)
    if resolved is not None:
        return resolved
    hub_root = hf_hub_root()
    model_cache_dir = hub_root / ("models--" + raw.replace("/", "--"))
    resolved = resolve_snapshot_dir(model_cache_dir)
    if resolved is not None:
        return resolved
    die(
        "failed to resolve model from local cache: "
        f"{raw}. Checked path='{p}' and HF cache='{model_cache_dir}'."
    )
    raise AssertionError("unreachable")


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            out.append(json.loads(ln))
    return out


def to_int_list(v: object) -> List[int]:
    if not isinstance(v, list):
        return []
    out: List[int] = []
    for x in v:
        out.append(int(x))
    return out


def to_float_list(v: object) -> List[float]:
    if not isinstance(v, list):
        return []
    out: List[float] = []
    for x in v:
        out.append(float(x))
    return out


def compare_candidates(
    a: List[Dict[str, object]],
    b: List[Dict[str, object]],
    tol: float,
) -> Tuple[bool, Dict[str, str]]:
    if len(a) != len(b):
        return False, {
            "num_candidates_a": str(len(a)),
            "num_candidates_b": str(len(b)),
            "token_mismatch_candidates": "0",
            "finish_reason_mismatch_candidates": "0",
            "max_abs_logprob_diff": "nan",
            "mean_abs_logprob_diff": "nan",
        }

    token_mismatch = 0
    finish_mismatch = 0
    total_abs = 0.0
    count_abs = 0
    max_abs = 0.0

    for ca, cb in zip(a, b):
        ta = to_int_list(ca.get("tokens", []))
        tb = to_int_list(cb.get("tokens", []))
        if ta != tb:
            token_mismatch += 1

        fa = str(ca.get("finish_reason", ""))
        fb = str(cb.get("finish_reason", ""))
        if fa != fb:
            finish_mismatch += 1

        la = to_float_list(ca.get("token_logprobs", []))
        lb = to_float_list(cb.get("token_logprobs", []))
        n = min(len(la), len(lb))
        for i in range(n):
            d = abs(la[i] - lb[i])
            total_abs += d
            count_abs += 1
            if d > max_abs:
                max_abs = d
        if len(la) != len(lb):
            token_mismatch += 1

    mean_abs = (total_abs / float(count_abs)) if count_abs > 0 else 0.0
    ok = (token_mismatch == 0 and finish_mismatch == 0 and max_abs <= tol)
    return ok, {
        "num_candidates_a": str(len(a)),
        "num_candidates_b": str(len(b)),
        "token_mismatch_candidates": str(token_mismatch),
        "finish_reason_mismatch_candidates": str(finish_mismatch),
        "max_abs_logprob_diff": f"{max_abs:.8f}",
        "mean_abs_logprob_diff": f"{mean_abs:.8f}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2.2 deterministic numeric consistency check.")
    ap.add_argument("--model", type=str, required=True, help="model path or HF model id")
    ap.add_argument("--prompt", type=str, default="")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-len", type=int, default=64)
    ap.add_argument("--num-candidates", type=int, default=4)
    ap.add_argument("--gpus", type=str, default="0,1")
    ap.add_argument("--split", type=str, default="9,27")
    ap.add_argument("--chunk-len", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--build-dir", type=str, default="build")
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    if args.prompt_len <= 0 or args.gen_len <= 0 or args.num_candidates <= 0:
        die("prompt/gen/candidates must be > 0")
    if args.tol < 0.0:
        die("--tol must be >= 0")

    repo = Path.cwd()
    model_dir = resolve_model_dir(args.model)
    bench_bin = (repo / args.build_dir / "ember_multi_candidate_rollout").resolve()
    if not bench_bin.exists():
        die(f"missing binary: {bench_bin}")

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (repo / "reports" / f"stage22_numeric_consistency_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_paths = {
        "a": out_dir / "run_a_candidates.jsonl",
        "b": out_dir / "run_b_candidates.jsonl",
        "c": out_dir / "run_c_seed_plus_1_candidates.jsonl",
    }

    def run_once(tag: str, seed: int) -> None:
        cmd = [
            str(bench_bin),
            "--model",
            str(model_dir),
            "--prompt-len",
            str(args.prompt_len),
            "--gen-len",
            str(args.gen_len),
            "--num-candidates",
            str(args.num_candidates),
            "--gpus",
            args.gpus,
            "--split",
            args.split,
            "--chunk-len",
            str(args.chunk_len),
            "--temperature",
            str(args.temperature),
            "--top-p",
            str(args.top_p),
            "--top-k",
            str(args.top_k),
            "--seed",
            str(seed),
            "--candidates-jsonl",
            str(run_paths[tag]),
            "--no-decode-text",
        ]
        if args.prompt:
            cmd += ["--prompt", args.prompt]
        p = run_cmd(cmd, cwd=repo, log_path=logs_dir / f"run_{tag}.log", check=False)
        if p.returncode != 0:
            die(f"benchmark run_{tag} failed rc={p.returncode}; see {logs_dir / ('run_' + tag + '.log')}")

    run_once("a", args.seed)
    run_once("b", args.seed)
    run_once("c", args.seed + 1)

    a = load_jsonl(run_paths["a"])
    b = load_jsonl(run_paths["b"])
    c = load_jsonl(run_paths["c"])

    same_ok, same_stats = compare_candidates(a, b, tol=args.tol)
    diff_ok, diff_stats = compare_candidates(a, c, tol=args.tol)
    # For seed sensitivity we expect "not same".
    seed_sensitive = not diff_ok

    summary_rows = [
        {
            "check": "same_seed_repro",
            "ok": "1" if same_ok else "0",
            **same_stats,
            "tol": f"{args.tol:.8f}",
            "seed_a": str(args.seed),
            "seed_b": str(args.seed),
        },
        {
            "check": "seed_plus_1_diff",
            "ok": "1" if seed_sensitive else "0",
            **diff_stats,
            "tol": f"{args.tol:.8f}",
            "seed_a": str(args.seed),
            "seed_b": str(args.seed + 1),
        },
    ]

    summary_csv = out_dir / "stage22_numeric_consistency.csv"
    summary_md = out_dir / "stage22_summary.md"
    write_csv(summary_csv, summary_rows)

    lines: List[str] = []
    lines.append("# Stage 2.2 Numeric Consistency")
    lines.append("")
    lines.append(f"- Model: `{model_dir}`")
    lines.append(
        f"- Setting: prompt_len={args.prompt_len}, gen_len={args.gen_len}, "
        f"num_candidates={args.num_candidates}, gpus={args.gpus}, split={args.split}"
    )
    lines.append(f"- Generated at: `{dt.datetime.now().isoformat(timespec='seconds')}`")
    lines.append("")
    lines.append("| check | ok | token_mismatch_candidates | finish_reason_mismatch_candidates | max_abs_logprob_diff | mean_abs_logprob_diff |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in summary_rows:
        lines.append(
            f"| {r['check']} | {r['ok']} | {r['token_mismatch_candidates']} | "
            f"{r['finish_reason_mismatch_candidates']} | {r['max_abs_logprob_diff']} | {r['mean_abs_logprob_diff']} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[done] out_dir={out_dir}")
    print(f"- csv: {summary_csv}")
    print(f"- md: {summary_md}")
    print(f"- same_seed_ok={int(same_ok)} seed_plus_1_diff_ok={int(seed_sensitive)}")


if __name__ == "__main__":
    main()

````

---

## 3) 报告上下文（完整）

### Report: reports/stage21_multi_candidate_4b_20260225_mainline/stage21_summary.md

````md
# Stage 2.1 Multi-Candidate Rollout

- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Generated at: `2026-02-25T12:56:38`

| metric | value |
| --- | --- |
| prompt_len | `2048` |
| gen_len | `128` |
| num_candidates | `8` |
| gpus | `0+1` |
| split | `9+27` |
| prefill_ms | `3281.765` |
| decode_ms | `10746.881` |
| total_ms | `14028.646` |
| total_gen_tokens | `1024` |
| gen_tok_s | `72.994` |
| temperature | `0.700` |
| top_p | `0.900` |
| top_k | `40` |

## Candidate Stats
- mean(avg_logprob): `-2.158189`
- best candidate id: `6`
- best sum_logprob: `-266.118321`

````

### Report: reports/stage21_multi_candidate_4b_20260225_mainline/stage21_multi_candidate.csv

````csv
mode,prompt_len,gen_len,num_candidates,gpus,split,prefill_ms,decode_ms,total_ms,total_gen_tokens,gen_tok_s,temperature,top_p,top_k
multi_candidate_rollout,2048,128,8,0+1,9+27,3281.765,10746.881,14028.646,1024,72.994,0.700,0.900,40

````

### Report: reports/stage22_numeric_consistency_4b_20260225_mainline/stage22_numeric_consistency.csv

````csv
check,ok,num_candidates_a,num_candidates_b,token_mismatch_candidates,finish_reason_mismatch_candidates,max_abs_logprob_diff,mean_abs_logprob_diff,tol,seed_a,seed_b
same_seed_repro,1,4,4,0,0,0.00000000,0.00000000,0.00000100,1234,1234
seed_plus_1_diff,1,4,4,4,0,4.75196700,1.18575816,0.00000100,1234,1235

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`
