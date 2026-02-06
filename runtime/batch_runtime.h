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

