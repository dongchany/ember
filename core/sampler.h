#pragma once

#include <random>

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
               const std::vector<int>& history) {}

   private:
    std::mt19937_64 rng_;
};
}  // namespace ember