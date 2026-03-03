# Tutorial #8 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 8 篇。

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
请写第 8 篇：KV Cache — 为什么 decode 每次只算一个 token。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 给出 KV cache 显存占用估算公式
- 解释为什么 RL 多轮场景会放大 prefill 成本
```

---

## 2) 代码上下文（完整/相关段落）

### File: core/session.h

````h
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
    
    bool allocated() const { return key_cache.data != nullptr; }
};

// KV Cache 管理器
class KVCache {
public:
    KVCache() = default;
    
    // 初始化缓存（不分配内存，只设置元数据）
    void init(int num_layers, int batch_size, int max_ctx_len, 
              int num_kv_heads, int head_dim, DType dtype) {
        num_layers_ = num_layers;
        batch_size_ = batch_size;
        max_ctx_len_ = max_ctx_len;
        num_kv_heads_ = num_kv_heads;
        head_dim_ = head_dim;
        dtype_ = dtype;
        
        layer_caches_.resize(num_layers);
        for (int i = 0; i < num_layers; ++i) {
            layer_caches_[i].key_cache.shape = {batch_size, num_kv_heads, max_ctx_len, head_dim};
            layer_caches_[i].key_cache.dtype = dtype;
            layer_caches_[i].value_cache.shape = {batch_size, num_kv_heads, max_ctx_len, head_dim};
            layer_caches_[i].value_cache.dtype = dtype;
        }
    }
    
    // 获取指定层的 KV cache
    LayerKVCache& layer(int i) { return layer_caches_[i]; }
    const LayerKVCache& layer(int i) const { return layer_caches_[i]; }
    
    // 设置层的缓存指针
    void set_layer_data(int layer_idx, void* key_data, void* value_data, int device_id) {
        layer_caches_[layer_idx].key_cache.data = key_data;
        layer_caches_[layer_idx].key_cache.device_id = device_id;
        layer_caches_[layer_idx].value_cache.data = value_data;
        layer_caches_[layer_idx].value_cache.device_id = device_id;
        layer_caches_[layer_idx].device_id = device_id;
    }
    
    // 计算单层缓存大小
    size_t layer_size_bytes() const {
        return static_cast<size_t>(batch_size_) * num_kv_heads_ * max_ctx_len_ * head_dim_ 
               * dtype_size(dtype_) * 2;  // K 和 V
    }
    
    // 计算总缓存大小
    size_t total_size_bytes() const {
        return layer_size_bytes() * num_layers_;
    }
    
    int num_layers() const { return num_layers_; }
    int max_ctx_len() const { return max_ctx_len_; }
    DType dtype() const { return dtype_; }

private:
    int num_layers_ = 0;
    int batch_size_ = 1;
    int max_ctx_len_ = 0;
    int num_kv_heads_ = 0;
    int head_dim_ = 0;
    DType dtype_ = DType::F16;
    
    std::vector<LayerKVCache> layer_caches_;
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
            runtime_config.kv_cache_dtype
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
    std::vector<int> cur_pos_by_batch_;
    std::vector<int> generated_tokens_;
};

}  // namespace ember

````

### File: backends/cuda/cuda_runtime.cpp (allocate_kv_cache)

````cpp
            cuda_free(act.v_proj_out);
            cuda_free(act.attn_out);
            cuda_free(act.attn_scores);
            cuda_free(act.attn_probs);
            if (act.mlp_gate_up_packed) {
                cuda_free(act.mlp_gate);
                act.mlp_up = nullptr;
                act.mlp_gate_up_packed = false;
            } else {
                cuda_free(act.mlp_gate);
                cuda_free(act.mlp_up);
            }
            cuda_free(act.mlp_down);
            cuda_free(act.logits);
            act.allocated = false;
        }
    }
    activations_.clear();
}

Error CudaRuntime::allocate_kv_cache(Session& session) {
    auto& kv = session.kv_cache();
    
    for (int layer_idx = 0; layer_idx < kv.num_layers(); ++layer_idx) {
        int device_id = device_map_.layer_to_device[layer_idx];
        size_t layer_size = kv.layer_size_bytes() / 2;  // K 和 V 各一半
        
        void* k_data = nullptr;
        void* v_data = nullptr;
        
        Error err = cuda_malloc(&k_data, layer_size, device_id);
        if (err) return err;
        
        err = cuda_malloc(&v_data, layer_size, device_id);
        if (err) return err;
        
        // 清零
        err = cuda_memset(k_data, 0, layer_size, device_id);
        if (err) return err;
        err = cuda_memset(v_data, 0, layer_size, device_id);
        if (err) return err;
        
        kv.set_layer_data(layer_idx, k_data, v_data, device_id);
    }
    
    std::cout << "[CudaRuntime] Allocated KV cache: " 
              << format_bytes(kv.total_size_bytes()) << std::endl;
    
    return Error::success();
}

void CudaRuntime::free_kv_cache(Session& session) {
    auto& kv = session.kv_cache();
    
    for (int i = 0; i < kv.num_layers(); ++i) {
        auto& layer = kv.layer(i);
        if (layer.key_cache.data) {
            cudaSetDevice(layer.device_id);
            cuda_free(layer.key_cache.data);
            cuda_free(layer.value_cache.data);
            layer.key_cache.data = nullptr;
            layer.value_cache.data = nullptr;
        }
    }
}

void CudaRuntime::unload() {
    if (next_tokens_dev_) {
        int dev = 0;
        if (device_map_.num_devices > 0) dev = device_map_.lm_head_device;
        cudaSetDevice(dev);
        cuda_free(next_tokens_dev_);
        next_tokens_dev_ = nullptr;
        next_tokens_cap_ = 0;
    }

    // 释放激活缓冲区
    free_activation_buffers();
    
    // 释放权重
    if (weights_.embed_tokens) {
        cudaSetDevice(weights_.embed_device_id);
        cuda_free(weights_.embed_tokens);
    }
    if (weights_.lm_head && weights_.lm_head_owns_allocation) {
        cudaSetDevice(weights_.lm_head_device_id);
        cuda_free(weights_.lm_head);
    }
    if (weights_.final_norm) {
        cudaSetDevice(weights_.final_norm_device_id);
        cuda_free(weights_.final_norm);
    }
    
    weights_.embed_tokens = nullptr;
    weights_.lm_head = nullptr;
    weights_.final_norm = nullptr;
    weights_.lm_head_owns_allocation = false;
    
    for (auto& layer : weights_.layers) {
        if (layer.allocated) {
            cudaSetDevice(layer.device_id);
            cuda_free(layer.q_proj_weight);
            cuda_free(layer.k_proj_weight);
            cuda_free(layer.v_proj_weight);
            cuda_free(layer.o_proj_weight);
            cuda_free(layer.q_norm_weight);
            cuda_free(layer.k_norm_weight);
            if (layer.gate_up_proj_packed) {
                cuda_free(layer.gate_up_proj_weight);
            } else {
                cuda_free(layer.gate_proj_weight);
                cuda_free(layer.up_proj_weight);
            }
            cuda_free(layer.down_proj_weight);
            cuda_free(layer.input_layernorm_weight);
            cuda_free(layer.post_attention_layernorm_weight);
            layer.gate_proj_weight = nullptr;
            layer.up_proj_weight = nullptr;
            layer.gate_up_proj_weight = nullptr;
            layer.gate_up_proj_packed = false;
            layer.allocated = false;
        }
    }
    weights_.layers.clear();
    has_active_lora_adapter_ = false;
    active_lora_adapter_dir_.clear();
    active_lora_scale_ = 0.0f;
    
    // 销毁 streams 和 cuBLAS 句柄
    for (auto& stream : streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    streams_.clear();
    for (auto& stream : transfer_streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    transfer_streams_.clear();

    for (auto& ev : profile_events_) {
        if (ev.start) cudaEventDestroy(ev.start);
        if (ev.end) cudaEventDestroy(ev.end);
        ev.start = nullptr;
        ev.end = nullptr;
    }
    profile_events_.clear();
    destroy_stage_profiling_events_();
    cublas_handles_.clear();
    
    loaded_ = false;
}

void CudaRuntime::set_layer_profiling(bool enabled) {
    profile_layers_ = enabled;
}

std::vector<float> CudaRuntime::take_last_layer_profile_ms() {
    std::vector<float> out = last_layer_profile_ms_;
    std::fill(last_layer_profile_ms_.begin(), last_layer_profile_ms_.end(), 0.0f);
    return out;
}

void CudaRuntime::set_stage_profiling(bool enabled) {
    profile_stages_ = enabled;
}

CudaRuntime::StageProfileMs CudaRuntime::take_last_stage_profile_ms() {
    StageProfileMs out = last_stage_profile_ms_;
    last_stage_profile_ms_ = {};
    return out;
}

Error CudaRuntime::begin_stage_profile_() {
    if (!profile_stages_) {
        return Error::success();
    }
    ensure_stage_profiling_events_();
    last_stage_profile_ms_ = {};

````

### File: backends/cuda/cuda_runtime.cpp (update_kv_cache call sites)

````cpp
                static_cast<half*>(act.q_proj_out),
                static_cast<half*>(act.k_proj_out),
                batch_size, seq_len, num_heads, num_kv_heads, head_dim,
                start_pos, config_.rope_theta,
                stream
            );
        }
    }
    
    // =====================================================================
    // Update KV Cache
    // =====================================================================
    auto& kv_cache = session.kv_cache();
    auto& layer_kv = kv_cache.layer(layer_idx);
    
    const int max_seq = session.runtime_config().max_ctx_len;
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::update_kv_cache_bf16(
                static_cast<__nv_bfloat16*>(layer_kv.key_cache.data),
                static_cast<__nv_bfloat16*>(layer_kv.value_cache.data),
                static_cast<const __nv_bfloat16*>(act.k_proj_out),
                static_cast<const __nv_bfloat16*>(act.v_proj_out),
                batch_size, seq_len, num_kv_heads, head_dim,
                start_pos, max_seq,
                stream
            );
        } else {
            kernels::update_kv_cache_f16(
                static_cast<half*>(layer_kv.key_cache.data),
                static_cast<half*>(layer_kv.value_cache.data),
                static_cast<const half*>(act.k_proj_out),
                static_cast<const half*>(act.v_proj_out),
                batch_size, seq_len, num_kv_heads, head_dim,
                start_pos, max_seq,
                stream
            );
        }
    } else {
        for (int b = 0; b < batch_size; ++b) {
            const int sp = start_pos_by_batch[b];
            if (sp < 0) continue;
            if (sp + seq_len > max_seq) {
                return Error(ErrorCode::CONTEXT_TOO_LONG, "Context full (varpos batch)");
            }

            const size_t q_off = static_cast<size_t>(b) * static_cast<size_t>(seq_len) *
                                 static_cast<size_t>(num_heads) * static_cast<size_t>(head_dim);
            const size_t kv_off = static_cast<size_t>(b) * static_cast<size_t>(seq_len) *
                                  static_cast<size_t>(num_kv_heads) * static_cast<size_t>(head_dim);
            const size_t cache_off = static_cast<size_t>(b) * static_cast<size_t>(num_kv_heads) *
                                     static_cast<size_t>(max_seq) * static_cast<size_t>(head_dim);

            if (compute_dtype == DType::BF16) {
                auto* q_ptr = static_cast<__nv_bfloat16*>(act.q_proj_out) + q_off;
                auto* k_ptr = static_cast<__nv_bfloat16*>(act.k_proj_out) + kv_off;
                auto* v_ptr = static_cast<__nv_bfloat16*>(act.v_proj_out) + kv_off;
                auto* k_cache_ptr = static_cast<__nv_bfloat16*>(layer_kv.key_cache.data) + cache_off;
                auto* v_cache_ptr = static_cast<__nv_bfloat16*>(layer_kv.value_cache.data) + cache_off;

                kernels::apply_rope_bf16(
                    q_ptr,
                    k_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    sp,
                    config_.rope_theta,
                    stream
                );
                kernels::update_kv_cache_bf16(
                    k_cache_ptr,
                    v_cache_ptr,
                    k_ptr,
                    v_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_kv_heads,
                    head_dim,
                    sp,
                    max_seq,
                    stream
                );
            } else {
                auto* q_ptr = static_cast<half*>(act.q_proj_out) + q_off;
                auto* k_ptr = static_cast<half*>(act.k_proj_out) + kv_off;
                auto* v_ptr = static_cast<half*>(act.v_proj_out) + kv_off;
                auto* k_cache_ptr = static_cast<half*>(layer_kv.key_cache.data) + cache_off;
                auto* v_cache_ptr = static_cast<half*>(layer_kv.value_cache.data) + cache_off;

                kernels::apply_rope_f16(
                    q_ptr,
                    k_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    sp,
                    config_.rope_theta,
                    stream
                );
                kernels::update_kv_cache_f16(
                    k_cache_ptr,
                    v_cache_ptr,
                    k_ptr,
                    v_ptr,
                    /*batch_size=*/1,
                    seq_len,
                    num_kv_heads,
                    head_dim,
                    sp,
                    max_seq,
                    stream
                );
            }
        }
    }
    
    // =====================================================================
    // Attention: Q @ K^T / sqrt(d) -> Softmax -> @ V
    // =====================================================================
    int seq_k = start_pos + seq_len;  // KV cache 中的有效长度
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::attention_bf16(
                static_cast<__nv_bfloat16*>(act.attn_out),
                static_cast<const __nv_bfloat16*>(act.q_proj_out),
                static_cast<const __nv_bfloat16*>(layer_kv.key_cache.data),
                static_cast<const __nv_bfloat16*>(layer_kv.value_cache.data),
                static_cast<float*>(act.attn_scores),
                static_cast<__nv_bfloat16*>(act.attn_probs),
                batch_size, seq_len, seq_k, max_seq, start_pos,
                num_heads, num_kv_heads, head_dim,
                scale, cublas.get(), stream
            );
        } else {
            kernels::attention_f16(
                static_cast<half*>(act.attn_out),
                static_cast<const half*>(act.q_proj_out),
                static_cast<const half*>(layer_kv.key_cache.data),
                static_cast<const half*>(layer_kv.value_cache.data),
                static_cast<float*>(act.attn_scores),
                static_cast<half*>(act.attn_probs),
                batch_size, seq_len, seq_k, max_seq, start_pos,
                num_heads, num_kv_heads, head_dim,
                scale, cublas.get(), stream

````

---

## 3) 报告上下文（完整）

### Report: reports/stage1_prefix_cache_4b_20260225_mainline/stage13_prefix_cache_summary.md

````md
# Stage 1.3 Prefix Cache Sweep

- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Prompt length: `2048`
- Docs per run: `100`
- Generated at: `2026-02-25T09:34:47`

| prefix_len | suffix_len | no_cache_total_ms | with_cache_total_ms | speedup_x | savings_% | theoretical_savings_% |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2048 | 35957.866 | 35928.545 | 1.001 | 0.082 | 0.000 |
| 256 | 1792 | 36479.788 | 32984.744 | 1.106 | 9.581 | 12.375 |
| 512 | 1536 | 36839.712 | 29553.679 | 1.247 | 19.778 | 24.750 |
| 768 | 1280 | 37148.629 | 26100.931 | 1.423 | 29.739 | 37.125 |
| 1024 | 1024 | 37385.412 | 22677.374 | 1.649 | 39.342 | 49.500 |
| 1280 | 768 | 37400.463 | 18012.815 | 2.076 | 51.838 | 61.875 |
| 1536 | 512 | 37583.473 | 13082.734 | 2.873 | 65.190 | 74.250 |

## Key Point
- Best measured savings: prefix_len=1536 -> `65.190%` (`2.873x`).
- Shared-prefix ~1k tokens result: savings `39.342%`, speedup `1.649x`.

````

### Report: reports/stage1_prefix_cache_4b_20260225_mainline/stage13_prefix_cache_sweep.csv

````csv
gpus,split,mode,prompt_len,prefix_len,suffix_len,num_docs,iters,warmup,no_cache_total_ms,with_cache_total_ms,cache_prefix_once_ms,cache_suffix_total_ms,no_cache_per_doc_ms,with_cache_per_doc_ms,speedup_x,savings_pct,theoretical_savings_pct
0+1,9+27,overlap,2048,0,2048,100,1,0,35957.866,35928.545,0.000,35928.545,359.579,359.285,1.001,0.082,0.000
0+1,9+27,overlap,2048,256,1792,100,1,0,36479.788,32984.744,58.523,32926.221,364.798,329.847,1.106,9.581,12.375
0+1,9+27,overlap,2048,512,1536,100,1,0,36839.712,29553.679,88.241,29465.438,368.397,295.537,1.247,19.778,24.750
0+1,9+27,overlap,2048,768,1280,100,1,0,37148.629,26100.931,206.862,25894.069,371.486,261.009,1.423,29.739,37.125
0+1,9+27,overlap,2048,1024,1024,100,1,0,37385.412,22677.374,197.830,22479.544,373.854,226.774,1.649,39.342,49.500
0+1,9+27,overlap,2048,1280,768,100,1,0,37400.463,18012.815,225.344,17787.471,374.005,180.128,2.076,51.838,61.875
0+1,9+27,overlap,2048,1536,512,100,1,0,37583.473,13082.734,268.452,12814.282,375.835,130.827,2.873,65.190,74.250

````

### Report: reports/stage14_cumulative_profile_4b_20260225_mainline/stage14_summary.md

````md
# Stage 1.4 Cumulative Cost Simulation

- Generated at: `2026-02-25T10:22:54`
- Base row: split=9+27, mode=overlap, prompt_len=2048, decode_steps=128
- Requests per round: num_prompts=100, num_candidates=8, total=800
- Locality assumptions: recompute_ratio=0.5, periodic_refresh_k=10
- Prefix assumptions: prefix reuse constants from stage13_prefix_cache_sweep.csv: prefix_once=197.830ms, suffix_per_req=224.795ms

| strategy | cumulative_ms | cumulative_wall_hours | cumulative_gpu_hours | reduction_vs_naive_% |
| --- | --- | --- | --- | --- |
| naive | 66127488.000 | 18.368747 | 36.737493 | 0.000 |
| prefix_only | 61206977.460 | 17.001938 | 34.003876 | 7.441 |
| update_locality | 61654822.400 | 17.126340 | 34.252679 | 6.764 |

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`
