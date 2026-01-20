#pragma once

#include <algorithm>
#include <random>
#include <unordered_map>

namespace ember {

class Sampler {
   public:
    void set_seed(uint64_t seed) { rng_.seed(seed); }

    // Sample the next token from the logits (using RuntimeConfig)
    int sample(const std::vector<float>& logits, const RuntimeConfig& config) {
        static const std::vector<int> empty_history;
        return sample(logits, config, empty_history);
    }

    // Sample the next token from the logits (using RuntimeConfig)
    int sample(const std::vector<float>& logits, const RuntimeConfig& config,
               const std::vector<int>& history) {
        std::vector<float> probs = logits;

        if (!history.empty()) {
            std::unordered_map<int, int> counts;
            counts.reserve(history.size());
            for (int token : history) {
                if (token >= 0 && token < static_cast<int>(probs.size())) {
                    ++counts[token];
                }
            }

            // Repetition penalty (pre-softmax)
            if (config.repetition_penalty > 1.0f) {
                for (const auto& item : counts) {
                    float& logit = probs[item.first];
                    if (logit < 0.0f) {
                        logit *= config.repetition_penalty;
                    } else {
                        logit /= config.repetition_penalty;
                    }
                }
            }

            // Presence/frequency penalties (pre-softmax)
            if (config.presence_penalty != 0.0f ||
                config.frequency_penalty != 0.0f) {
                for (const auto& item : counts) {
                    float& logit = probs[item.first];
                    if (config.presence_penalty != 0.0f) {
                        logit -= config.presence_penalty;
                    }
                    if (config.frequency_penalty != 0.0f) {
                        logit -= config.frequency_penalty *
                                 static_cast<float>(item.second);
                    }
                }
            }

            // No-repeat ngram: ban tokens that would repeat the last n-gram
            int ngram = config.no_repeat_ngram_size;
            if (ngram > 1 && history.size() >= static_cast<size_t>(ngram)) {
                size_t prefix_start =
                    history.size() - static_cast<size_t>(ngram - 1);
                for (size_t i = 0;
                     i + static_cast<size_t>(ngram) <= history.size(); ++i) {
                    bool match = true;
                    for (int j = 0; j < ngram - 1; ++j) {
                        if (history[i + static_cast<size_t>(j)] !=
                            history[prefix_start + static_cast<size_t>(j)]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        int banned =
                            history[i + static_cast<size_t>(ngram - 1)];
                        if (banned >= 0 &&
                            banned < static_cast<int>(probs.size())) {
                            probs[banned] = -1e9f;
                        }
                    }
                }
            }
        }

        // 应用 temperature
        if (config.temperature > 0 && config.temperature != 1.0f) {
            for (float& p : probs) {
                p /= config.temperature;
            }
        }

        // Softmax
        softmax(probs);

        // 应用 top-k
        if (config.top_k > 0 && config.top_k < static_cast<int>(probs.size())) {
            top_k_filter(probs, config.top_k);
        }

        // 应用 top-p (nucleus sampling)
        if (config.top_p > 0 && config.top_p < 1.0f) {
            top_p_filter(probs, config.top_p);
        }

        // 根据 temperature 决定采样方式
        if (config.temperature <= 0) {
            // Greedy: 返回最大概率的 token
            return static_cast<int>(std::distance(
                probs.begin(), std::max_element(probs.begin(), probs.end())));
        } else {
            // 随机采样
            return categorical_sample(probs);
        }
    }

    static void softmax(std::vector<float>& x) {
        float max_val = *std::max_element(x.begin(), x.end());
        float sum = 0;
        for (float& v : x) {
            v = std::exp(v - max_val);
            sum += v;
        }
        for (float& v : x) {
            v /= sum;
        }
    }

    static void top_k_filter(std::vector<float>& probs, int k) {
        if (k >= static_cast<int>(probs.size())) return;
        if (k <= 0) return;

        // 找到第 k 大的值
        std::vector<float> sorted = probs;
        std::nth_element(sorted.begin(), sorted.begin() + (k - 1), sorted.end(),
                         std::greater<float>());
        float threshold = sorted[k - 1];

        // 过滤掉小于阈值的
        for (float& p : probs) {
            if (p < threshold) p = 0;
        }

        // 重新归一化
        float sum = std::accumulate(probs.begin(), probs.end(), 0.0f);
        if (sum > 0) {
            for (float& p : probs) p /= sum;
        }
    }

    // Top-P (Nucleus) 过滤
    static void top_p_filter(std::vector<float>& probs, float p) {
        // 创建索引-概率对并排序
        std::vector<std::pair<int, float>> indexed(probs.size());
        for (size_t i = 0; i < probs.size(); ++i) {
            indexed[i] = {static_cast<int>(i), probs[i]};
        }
        std::sort(
            indexed.begin(), indexed.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

        // 累积概率，找到截断点
        float cumsum = 0;
        size_t cutoff = indexed.size();
        for (size_t i = 0; i < indexed.size(); ++i) {
            cumsum += indexed[i].second;
            if (cumsum >= p) {
                cutoff = i + 1;
                break;
            }
        }

        // 过滤
        std::vector<float> new_probs(probs.size(), 0);
        for (size_t i = 0; i < cutoff; ++i) {
            new_probs[indexed[i].first] = indexed[i].second;
        }

        // 重新归一化
        float sum = std::accumulate(new_probs.begin(), new_probs.end(), 0.0f);
        if (sum > 0) {
            for (float& p : new_probs) p /= sum;
        }

        probs = std::move(new_probs);
    }
    int categorical_sample(const std::vector<float>& probs) {
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng_);

        float cumsum = 0;
        for (size_t i = 0; i < probs.size(); ++i) {
            cumsum += probs[i];
            if (r < cumsum) {
                return static_cast<int>(i);
            }
        }

        // 回退到最后一个有效 token
        for (int i = static_cast<int>(probs.size()) - 1; i >= 0; --i) {
            if (probs[i] > 0) return i;
        }
        return 0;
    }

   private:
    std::mt19937_64 rng_;
};
}  // namespace ember