#pragma once

#include "types.h"
#include "tensor.h"
#include "config.h"
#include <vector>
#include <memory>

namespace ember {

// 单层的 KV Cache
struct LayerKVCache {
    Tensor key_cache;    // [batch, num_kv_heads, max_ctx, head_dim]
    Tensor value_cache;  // [batch, num_kv_heads, max_ctx, head_dim]
    int device_id = 0;
    bool enabled = true;
    
    bool allocated() const { return enabled && key_cache.data != nullptr; }
};

// KV Cache 管理器
class KVCache {
public:
    KVCache() = default;
    
    // 初始化缓存（不分配内存，只设置元数据）
    void init(int num_layers, int batch_size, int max_ctx_len,
              int num_kv_heads, int head_dim, DType dtype,
              const std::vector<HybridLayerType>& layer_types = {}) {
        num_layers_ = num_layers;
        batch_size_ = batch_size;
        max_ctx_len_ = max_ctx_len;
        num_kv_heads_ = num_kv_heads;
        head_dim_ = head_dim;
        dtype_ = dtype;
        num_enabled_layers_ = 0;
        
        layer_caches_.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            const bool has_layer_types = layer_types.size() == static_cast<size_t>(num_layers);
            const bool enabled = !has_layer_types ||
                layer_types[static_cast<size_t>(i)] == HybridLayerType::GATED_ATTENTION;
            auto& layer = layer_caches_[i];
            layer.enabled = enabled;
            layer.key_cache.dtype = dtype;
            layer.value_cache.dtype = dtype;
            if (enabled) {
                layer.key_cache.shape = {batch_size, num_kv_heads, max_ctx_len, head_dim};
                layer.value_cache.shape = {batch_size, num_kv_heads, max_ctx_len, head_dim};
                ++num_enabled_layers_;
            } else {
                layer.key_cache.shape.clear();
                layer.value_cache.shape.clear();
            }
        }
    }
    
    // 获取指定层的 KV cache
    LayerKVCache& layer(int i) { return layer_caches_[i]; }
    const LayerKVCache& layer(int i) const { return layer_caches_[i]; }
    
    // 设置层的缓存指针
    void set_layer_data(int layer_idx, void* key_data, void* value_data, int device_id) {
        if (!layer_caches_[layer_idx].enabled) {
            return;
        }
        layer_caches_[layer_idx].key_cache.data = key_data;
        layer_caches_[layer_idx].key_cache.device_id = device_id;
        layer_caches_[layer_idx].value_cache.data = value_data;
        layer_caches_[layer_idx].value_cache.device_id = device_id;
        layer_caches_[layer_idx].device_id = device_id;
    }

    bool layer_enabled(int layer_idx) const {
        return layer_caches_[layer_idx].enabled;
    }
    
    // 计算单层缓存大小
    size_t layer_size_bytes() const {
        return static_cast<size_t>(batch_size_) * num_kv_heads_ * max_ctx_len_ * head_dim_ 
               * dtype_size(dtype_) * 2;  // K 和 V
    }
    
    // 计算总缓存大小
    size_t total_size_bytes() const {
        return layer_size_bytes() * static_cast<size_t>(num_enabled_layers_);
    }
    
    int num_layers() const { return num_layers_; }
    int num_enabled_layers() const { return num_enabled_layers_; }
    int max_ctx_len() const { return max_ctx_len_; }
    DType dtype() const { return dtype_; }

private:
    int num_layers_ = 0;
    int batch_size_ = 1;
    int max_ctx_len_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    int num_enabled_layers_ = 0;
    DType dtype_ = DType::F16;
    
    std::vector<LayerKVCache> layer_caches_;
};

// 单层 DeltaNet recurrent state
struct LayerRecurrentState {
    Tensor state;  // [batch, num_heads, head_dim, head_dim]
    int device_id = 0;
    bool enabled = false;

    bool allocated() const { return enabled && state.data != nullptr; }
};

// Recurrent state 管理器（DeltaNet 层使用）
class RecurrentStateCache {
public:
    RecurrentStateCache() = default;

    void init(int num_layers, int batch_size, int num_heads, int head_dim, DType dtype,
              const std::vector<HybridLayerType>& layer_types = {}) {
        num_layers_ = num_layers;
        batch_size_ = batch_size;
        num_heads_ = num_heads;
        head_dim_ = head_dim;
        dtype_ = dtype;
        num_enabled_layers_ = 0;

        states_.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            const bool has_layer_types = layer_types.size() == static_cast<size_t>(num_layers);
            const bool enabled = has_layer_types &&
                layer_types[static_cast<size_t>(i)] == HybridLayerType::DELTANET;
            auto& layer = states_[i];
            layer.enabled = enabled;
            layer.state.dtype = dtype;
            if (enabled) {
                layer.state.shape = {batch_size, num_heads, head_dim, head_dim};
                ++num_enabled_layers_;
            } else {
                layer.state.shape.clear();
            }
        }
    }

    LayerRecurrentState& layer(int i) { return states_[i]; }
    const LayerRecurrentState& layer(int i) const { return states_[i]; }

    void set_layer_data(int layer_idx, void* state_data, int device_id) {
        auto& layer = states_[layer_idx];
        if (!layer.enabled) {
            return;
        }
        layer.state.data = state_data;
        layer.state.device_id = device_id;
        layer.device_id = device_id;
    }

    bool layer_enabled(int layer_idx) const {
        return states_[layer_idx].enabled;
    }

    size_t layer_size_bytes() const {
        return static_cast<size_t>(batch_size_) * static_cast<size_t>(num_heads_) *
               static_cast<size_t>(head_dim_) * static_cast<size_t>(head_dim_) *
               dtype_size(dtype_);
    }

    size_t total_size_bytes() const {
        return layer_size_bytes() * static_cast<size_t>(num_enabled_layers_);
    }

    int num_layers() const { return num_layers_; }
    int num_enabled_layers() const { return num_enabled_layers_; }

private:
    int num_layers_ = 0;
    int batch_size_ = 1;
    int num_heads_ = 0;
    int head_dim_ = 0;
    int num_enabled_layers_ = 0;
    DType dtype_ = DType::F16;
    std::vector<LayerRecurrentState> states_;
};

// 推理会话状态
class Session {
public:
    Session() = default;
    
    // 初始化会话
    void init(const ModelConfig& model_config, const RuntimeConfig& runtime_config) {
        model_config_ = model_config;
        runtime_config_ = runtime_config;
        
        kv_cache_.init(
            model_config.num_layers,
            runtime_config.batch_size,
            runtime_config.max_ctx_len,
            model_config.num_kv_heads,
            model_config.head_dim,
            runtime_config.kv_cache_dtype,
            model_config.layer_types
        );

        recurrent_states_.init(
            model_config.num_layers,
            runtime_config.batch_size,
            model_config.num_heads,
            model_config.head_dim,
            runtime_config.kv_cache_dtype,
            model_config.layer_types
        );
        
        cur_pos_by_batch_.assign(static_cast<size_t>(runtime_config.batch_size), 0);
    }
    
    // 当前位置（已处理的 token 数）
    int cur_pos() const { return cur_pos_by_batch_.empty() ? 0 : cur_pos_by_batch_[0]; }
    int cur_pos(int slot) const { return cur_pos_by_batch_.at(static_cast<size_t>(slot)); }

    // Backward-compatible: for uniform batches, set/advance all slots.
    void set_cur_pos(int pos) {
        for (int& p : cur_pos_by_batch_) p = pos;
    }
    void set_cur_pos(int slot, int pos) { cur_pos_by_batch_.at(static_cast<size_t>(slot)) = pos; }

    void advance(int n = 1) {
        for (int& p : cur_pos_by_batch_) {
            if (p >= 0) p += n;
        }
    }
    void advance_slot(int slot, int n = 1) { cur_pos_by_batch_.at(static_cast<size_t>(slot)) += n; }

    void set_inactive(int slot) { cur_pos_by_batch_.at(static_cast<size_t>(slot)) = -1; }
    bool active(int slot) const { return cur_pos(slot) >= 0; }
    
    // 剩余可用上下文
    int remaining_ctx() const { return runtime_config_.max_ctx_len - cur_pos(); }
    int remaining_ctx(int slot) const { return runtime_config_.max_ctx_len - cur_pos(slot); }
    
    // 是否还能继续生成
    bool can_continue() const { return cur_pos() < runtime_config_.max_ctx_len; }
    bool can_continue(int slot) const { return cur_pos(slot) < runtime_config_.max_ctx_len; }
    
    // 重置会话（清除 KV cache 内容，重置位置）
    void reset() {
        for (int& p : cur_pos_by_batch_) p = 0;
        // 注意：不释放内存，只重置位置
    }
    
    // 获取 KV cache
    KVCache& kv_cache() { return kv_cache_; }
    const KVCache& kv_cache() const { return kv_cache_; }

    RecurrentStateCache& recurrent_states() { return recurrent_states_; }
    const RecurrentStateCache& recurrent_states() const { return recurrent_states_; }
    
    // 获取配置
    const ModelConfig& model_config() const { return model_config_; }
    const RuntimeConfig& runtime_config() const { return runtime_config_; }
    
    // 生成的 token 序列
    std::vector<int>& generated_tokens() { return generated_tokens_; }
    const std::vector<int>& generated_tokens() const { return generated_tokens_; }

private:
    ModelConfig model_config_;
    RuntimeConfig runtime_config_;
    KVCache kv_cache_;
    RecurrentStateCache recurrent_states_;
    std::vector<int> cur_pos_by_batch_;
    std::vector<int> generated_tokens_;
};

}  // namespace ember
