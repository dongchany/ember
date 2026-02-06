#pragma once

#include "iruntime.h"

#include <cstdint>
#include <vector>

namespace ember {

struct PhaseAwareSchedulerConfig {
    // Prefill: use chunked pipeline when multi-GPU and prompt is long.
    int prefill_chunk_len = 128;
    bool prefill_overlap = true;

    // Decode: placeholder knobs (real request-level batching needs variable-length support).
    int decode_batch_size = 1;
};

class PhaseAwareScheduler {
public:
    explicit PhaseAwareScheduler(IRuntime& runtime, PhaseAwareSchedulerConfig cfg = {});

    Error prefill(const std::vector<int>& tokens, Session& session);
    Error prefill_with_logits(const std::vector<int>& tokens, Session& session, std::vector<float>& logits);

    // Decode one step for batch_size=1.
    Error decode(int last_token, Session& session, std::vector<float>& logits);

    // Decode one step for batch_size>1 (logits_flat is [batch, vocab] contiguous).
    Error decode_batch(const std::vector<int>& last_tokens, Session& session, std::vector<float>& logits_flat);

private:
    IRuntime* runtime_ = nullptr;
    PhaseAwareSchedulerConfig cfg_;
};

// A simple continuous-batching scheduler for decode:
// - Prefill each request into a fixed batch slot (KV cache slice).
// - Decode step runs as a batch with per-slot start_pos support.
// - Uses greedy sampling (argmax) on GPU.
struct PhaseAwareBatchSchedulerConfig {
    // When a slot becomes free, immediately admit the next request (if any).
    bool refill_on_step = true;

    // Prefill: optionally use 2-GPU chunked pipeline per slot when prompt is long.
    int prefill_chunk_len = 128;
    bool prefill_overlap = true;
};

class PhaseAwareBatchScheduler {
public:
    PhaseAwareBatchScheduler(IRuntime& runtime, Session& session, PhaseAwareBatchSchedulerConfig cfg = {});

    // Returns the assigned slot index.
    Result<int> submit(const std::vector<int>& prompt_tokens, int max_new_tokens);

    // Runs one batched decode step across active slots.
    // Returns the number of tokens generated in this step (== active slots before the step).
    Result<int> step();

    int batch_size() const { return batch_size_; }
    int active_slots() const;
    bool has_active() const { return active_slots() > 0; }

private:
    IRuntime* runtime_ = nullptr;
    Session* session_ = nullptr;
    PhaseAwareBatchSchedulerConfig cfg_;

    int batch_size_ = 0;
    std::vector<uint8_t> active_;
    std::vector<int> remaining_;
    std::vector<int> last_tokens_;
};

}  // namespace ember
