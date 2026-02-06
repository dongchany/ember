#pragma once

#include "../core/error.h"
#include "../core/session.h"

#include <cstddef>
#include <vector>

namespace ember {

// RAII guard that temporarily remaps Session's KV cache pointers to a single batch slot slice.
// Useful for implementing "multi-request continuous batching" by prefill-ing one request at a time
// into a fixed slot, without touching other slots' KV regions.
class KVCacheSlotGuard {
public:
    KVCacheSlotGuard() = default;
    KVCacheSlotGuard(const KVCacheSlotGuard&) = delete;
    KVCacheSlotGuard& operator=(const KVCacheSlotGuard&) = delete;
    KVCacheSlotGuard(KVCacheSlotGuard&& other) noexcept { *this = std::move(other); }
    KVCacheSlotGuard& operator=(KVCacheSlotGuard&& other) noexcept {
        if (this == &other) return *this;
        restore();
        session_ = other.session_;
        slot_ = other.slot_;
        saved_ = std::move(other.saved_);
        active_ = other.active_;
        other.session_ = nullptr;
        other.active_ = false;
        return *this;
    }

    ~KVCacheSlotGuard() { restore(); }

    static Result<KVCacheSlotGuard> create(Session& session, int slot) {
        KVCacheSlotGuard g;
        g.session_ = &session;
        g.slot_ = slot;

        const int batch_size = session.runtime_config().batch_size;
        if (batch_size <= 0) return Error::invalid_argument("runtime_config.batch_size must be > 0");
        if (slot < 0 || slot >= batch_size) return Error::invalid_argument("slot out of range for runtime_config.batch_size");

        const auto& mc = session.model_config();
        const int max_ctx = session.runtime_config().max_ctx_len;
        if (max_ctx <= 0) return Error::invalid_argument("runtime_config.max_ctx_len must be > 0");

        const size_t elem_size = dtype_size(session.runtime_config().kv_cache_dtype);
        if (elem_size == 0) return Error::invalid_argument("unsupported kv_cache_dtype");

        const size_t cache_elems_per_slot =
            static_cast<size_t>(mc.num_kv_heads) * static_cast<size_t>(max_ctx) * static_cast<size_t>(mc.head_dim);
        const size_t cache_bytes_per_slot = cache_elems_per_slot * elem_size;

        g.saved_.resize(static_cast<size_t>(mc.num_layers));
        auto& kv = session.kv_cache();
        for (int l = 0; l < mc.num_layers; ++l) {
            auto& layer_kv = kv.layer(l);
            if (!layer_kv.key_cache.data || !layer_kv.value_cache.data) {
                g.restore();
                return Error(ErrorCode::OUT_OF_MEMORY, "KV cache not allocated");
            }
            g.saved_[static_cast<size_t>(l)] = SavedKV{layer_kv.key_cache.data, layer_kv.value_cache.data, layer_kv.device_id};
            void* k_ptr =
                static_cast<void*>(static_cast<char*>(layer_kv.key_cache.data) + static_cast<size_t>(slot) * cache_bytes_per_slot);
            void* v_ptr =
                static_cast<void*>(static_cast<char*>(layer_kv.value_cache.data) + static_cast<size_t>(slot) * cache_bytes_per_slot);
            kv.set_layer_data(l, k_ptr, v_ptr, layer_kv.device_id);
        }

        g.active_ = true;
        return g;
    }

    void restore() {
        if (!active_ || !session_) return;
        auto& kv = session_->kv_cache();
        for (size_t l = 0; l < saved_.size(); ++l) {
            const auto& s = saved_[l];
            if (s.k && s.v) {
                kv.set_layer_data(static_cast<int>(l), s.k, s.v, s.dev);
            }
        }
        active_ = false;
    }

private:
    struct SavedKV {
        void* k = nullptr;
        void* v = nullptr;
        int dev = 0;
    };

    Session* session_ = nullptr;
    int slot_ = 0;
    std::vector<SavedKV> saved_;
    bool active_ = false;
};

}  // namespace ember

