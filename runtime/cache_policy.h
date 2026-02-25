#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace ember {

enum class CachePolicyType {
    NAIVE = 0,          // Recompute all layers every update.
    UPDATE_LOCALITY,    // Reuse prefix layers; recompute suffix layers.
    PERIODIC_REFRESH,   // UpdateLocality + periodic full refresh.
};

struct CachePolicyConfig {
    CachePolicyType type = CachePolicyType::NAIVE;
    int total_layers = 0;
    // Number of prefix layers to reuse across updates.
    // recompute starts from this layer (inclusive).
    int freeze_prefix_layers = 0;
    // Every k updates do full refresh. 0 disables periodic full refresh.
    int periodic_refresh_k = 0;
};

struct CachePolicyDecision {
    int update_step = 0;           // 1-based update index
    bool full_refresh = false;     // true => recompute all layers
    int recompute_layers = 0;
    int reused_layers = 0;
    float recompute_ratio = 1.0f;  // recompute_layers / total_layers
    // 1 => recompute this layer, 0 => reuse this layer
    std::vector<uint8_t> recompute_mask;
};

struct CachePolicyStats {
    int updates = 0;
    int full_refreshes = 0;
    int total_recompute_layers = 0;
    int total_reused_layers = 0;
    float avg_recompute_ratio = 1.0f;
};

class CachePolicyEngine {
public:
    CachePolicyEngine() = default;
    explicit CachePolicyEngine(const CachePolicyConfig& cfg) { configure(cfg); }

    void configure(const CachePolicyConfig& cfg) {
        config_ = cfg;
        if (config_.total_layers < 0) config_.total_layers = 0;
        if (config_.freeze_prefix_layers < 0) config_.freeze_prefix_layers = 0;
        if (config_.freeze_prefix_layers > config_.total_layers) {
            config_.freeze_prefix_layers = config_.total_layers;
        }
        if (config_.periodic_refresh_k < 0) config_.periodic_refresh_k = 0;
        reset();
    }

    void reset() {
        step_ = 0;
        stats_ = CachePolicyStats{};
        stats_.avg_recompute_ratio = (config_.total_layers > 0) ? 1.0f : 0.0f;
    }

    CachePolicyDecision on_policy_update() {
        CachePolicyDecision d;
        d.update_step = ++step_;
        d.recompute_mask.assign(static_cast<size_t>(config_.total_layers), static_cast<uint8_t>(1));

        const int total = config_.total_layers;
        const int freeze_prefix = std::clamp(config_.freeze_prefix_layers, 0, total);

        bool full_refresh = true;
        switch (config_.type) {
            case CachePolicyType::NAIVE:
                full_refresh = true;
                break;
            case CachePolicyType::UPDATE_LOCALITY:
                // First step full recompute; afterwards reuse prefix.
                full_refresh = (step_ <= 1);
                break;
            case CachePolicyType::PERIODIC_REFRESH:
                if (step_ <= 1) {
                    full_refresh = true;
                } else if (config_.periodic_refresh_k > 0 && (step_ % config_.periodic_refresh_k == 0)) {
                    full_refresh = true;
                } else {
                    full_refresh = false;
                }
                break;
            default:
                full_refresh = true;
                break;
        }

        if (!full_refresh) {
            for (int l = 0; l < freeze_prefix; ++l) {
                d.recompute_mask[static_cast<size_t>(l)] = static_cast<uint8_t>(0);
            }
        }

        d.full_refresh = full_refresh;
        int recompute = 0;
        for (uint8_t x : d.recompute_mask) recompute += (x ? 1 : 0);
        d.recompute_layers = recompute;
        d.reused_layers = total - recompute;
        d.recompute_ratio = (total > 0) ? static_cast<float>(recompute) / static_cast<float>(total) : 0.0f;

        stats_.updates += 1;
        stats_.full_refreshes += (full_refresh ? 1 : 0);
        stats_.total_recompute_layers += d.recompute_layers;
        stats_.total_reused_layers += d.reused_layers;
        if (stats_.updates > 0 && total > 0) {
            stats_.avg_recompute_ratio =
                static_cast<float>(stats_.total_recompute_layers) /
                static_cast<float>(stats_.updates * total);
        } else {
            stats_.avg_recompute_ratio = 0.0f;
        }
        return d;
    }

    const CachePolicyConfig& config() const { return config_; }
    const CachePolicyStats& stats() const { return stats_; }

    static std::string policy_name(CachePolicyType t) {
        switch (t) {
            case CachePolicyType::NAIVE: return "naive";
            case CachePolicyType::UPDATE_LOCALITY: return "update_locality";
            case CachePolicyType::PERIODIC_REFRESH: return "periodic_refresh";
            default: return "unknown";
        }
    }

private:
    CachePolicyConfig config_;
    CachePolicyStats stats_;
    int step_ = 0;
};

}  // namespace ember

