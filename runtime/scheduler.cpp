#include "scheduler.h"

#include "batch_runtime.h"

namespace ember {

PhaseAwareScheduler::PhaseAwareScheduler(IRuntime& runtime, PhaseAwareSchedulerConfig cfg)
    : runtime_(&runtime), cfg_(cfg) {}

Error PhaseAwareScheduler::prefill(const std::vector<int>& tokens, Session& session) {
    const auto& dm = runtime_->device_map();
    const bool use_chunked = (dm.num_devices == 2) &&
                             (cfg_.prefill_chunk_len > 0) &&
                             (static_cast<int>(tokens.size()) > cfg_.prefill_chunk_len);
    if (!use_chunked) {
        return runtime_->prefill(tokens, session);
    }

    auto* ext = dynamic_cast<IBatchRuntime*>(runtime_);
    if (!ext) {
        return runtime_->prefill(tokens, session);
    }
    return ext->prefill_chunked_pipeline(tokens, session, cfg_.prefill_chunk_len, cfg_.prefill_overlap, nullptr);
}

Error PhaseAwareScheduler::prefill_with_logits(const std::vector<int>& tokens, Session& session, std::vector<float>& logits) {
    const auto& dm = runtime_->device_map();
    const bool use_chunked = (dm.num_devices == 2) &&
                             (cfg_.prefill_chunk_len > 0) &&
                             (static_cast<int>(tokens.size()) > cfg_.prefill_chunk_len);
    if (!use_chunked) {
        return runtime_->prefill_with_logits(tokens, session, logits);
    }

    auto* ext = dynamic_cast<IBatchRuntime*>(runtime_);
    if (!ext) {
        return runtime_->prefill_with_logits(tokens, session, logits);
    }
    return ext->prefill_chunked_pipeline(tokens, session, cfg_.prefill_chunk_len, cfg_.prefill_overlap, &logits);
}

Error PhaseAwareScheduler::decode(int last_token, Session& session, std::vector<float>& logits) {
    return runtime_->decode(last_token, session, logits);
}

Error PhaseAwareScheduler::decode_batch(const std::vector<int>& last_tokens, Session& session, std::vector<float>& logits_flat) {
    auto* ext = dynamic_cast<IBatchRuntime*>(runtime_);
    if (!ext) {
        return Error::not_implemented("decode_batch requires batch runtime extension");
    }
    return ext->decode_batch(last_tokens, session, logits_flat);
}

PhaseAwareBatchScheduler::PhaseAwareBatchScheduler(IRuntime& runtime,
                                                   Session& session,
                                                   PhaseAwareBatchSchedulerConfig cfg)
    : runtime_(&runtime), session_(&session), cfg_(cfg) {
    batch_size_ = session.runtime_config().batch_size;
    if (batch_size_ < 0) batch_size_ = 0;
    active_.assign(static_cast<size_t>(batch_size_), 0);
    remaining_.assign(static_cast<size_t>(batch_size_), 0);
    last_tokens_.assign(static_cast<size_t>(batch_size_), 0);

    for (int i = 0; i < batch_size_; ++i) {
        session_->set_inactive(i);
    }
}

Result<int> PhaseAwareBatchScheduler::submit(const std::vector<int>& prompt_tokens, int max_new_tokens) {
    if (!runtime_ || !session_) return Error::invalid_argument("scheduler not initialized");
    if (prompt_tokens.empty()) return Error::invalid_argument("prompt_tokens is empty");
    if (max_new_tokens < 0) return Error::invalid_argument("max_new_tokens must be >= 0");

    auto* ext = dynamic_cast<IBatchRuntime*>(runtime_);
    if (!ext) {
        return Error::not_implemented("PhaseAwareBatchScheduler requires batch runtime extension");
    }

    int slot = -1;
    for (int i = 0; i < batch_size_; ++i) {
        if (!active_[static_cast<size_t>(i)]) {
            slot = i;
            break;
        }
    }
    if (slot < 0) return Error::out_of_memory("no free slot");

    // Prefill prompt into this slot.
    session_->set_cur_pos(slot, 0);
    Error err = Error::success();
    if (cfg_.prefill_chunk_len > 0 &&
        runtime_->device_map().num_devices == 2 &&
        static_cast<int>(prompt_tokens.size()) > cfg_.prefill_chunk_len) {
        err = ext->prefill_into_slot_pipeline(prompt_tokens, slot, *session_, cfg_.prefill_chunk_len, cfg_.prefill_overlap, nullptr);
    } else {
        err = ext->prefill_into_slot(prompt_tokens, slot, *session_, nullptr);
    }
    if (err) return err;

    active_[static_cast<size_t>(slot)] = 1;
    remaining_[static_cast<size_t>(slot)] = max_new_tokens;
    last_tokens_[static_cast<size_t>(slot)] = prompt_tokens.back();

    if (max_new_tokens == 0) {
        active_[static_cast<size_t>(slot)] = 0;
        session_->set_inactive(slot);
    }

    return slot;
}

Result<int> PhaseAwareBatchScheduler::step() {
    if (!runtime_ || !session_) return Error::invalid_argument("scheduler not initialized");
    auto* ext = dynamic_cast<IBatchRuntime*>(runtime_);
    if (!ext) {
        return Error::not_implemented("PhaseAwareBatchScheduler requires batch runtime extension");
    }

    int active_before = 0;
    std::vector<int> inputs(static_cast<size_t>(batch_size_), 0);
    for (int i = 0; i < batch_size_; ++i) {
        if (active_[static_cast<size_t>(i)] && remaining_[static_cast<size_t>(i)] > 0) {
            ++active_before;
            inputs[static_cast<size_t>(i)] = last_tokens_[static_cast<size_t>(i)];
        } else {
            inputs[static_cast<size_t>(i)] = 0;
        }
    }
    if (active_before == 0) return Error::invalid_argument("no active slots");

    std::vector<int> next;
    Error err = ext->decode_batch_greedy(inputs, *session_, next);
    if (err) return err;
    if (static_cast<int>(next.size()) != batch_size_) {
        return Error::invalid_argument("decode_batch_greedy returned unexpected batch size");
    }

    for (int i = 0; i < batch_size_; ++i) {
        if (!active_[static_cast<size_t>(i)] || remaining_[static_cast<size_t>(i)] <= 0) continue;
        last_tokens_[static_cast<size_t>(i)] = next[static_cast<size_t>(i)];
        remaining_[static_cast<size_t>(i)] -= 1;
        if (remaining_[static_cast<size_t>(i)] <= 0) {
            active_[static_cast<size_t>(i)] = 0;
            session_->set_inactive(i);
        }
    }

    return active_before;
}

int PhaseAwareBatchScheduler::active_slots() const {
    int n = 0;
    for (int i = 0; i < batch_size_; ++i) {
        if (active_[static_cast<size_t>(i)] && remaining_[static_cast<size_t>(i)] > 0) ++n;
    }
    return n;
}

}  // namespace ember
