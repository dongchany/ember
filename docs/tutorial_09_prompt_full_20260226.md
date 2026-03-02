# Tutorial #9 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 9 篇。

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
请写第 9 篇：Pipeline Parallel — 两张消费卡怎么协作。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 讲清 9+27 / 18+18 这类 split 的含义
- 解释 overlap/no-overlap 对吞吐的影响
```

---

## 2) 代码上下文（完整/相关段落）

### File: runtime/iruntime.h

````h
#pragma once

#include "../core/error.h"
#include "../core/tensor.h"
#include "../core/config.h"
#include "../core/session.h"
#include <vector>
#include <memory>
#include <string>

namespace ember {

// 显存估算结果
struct MemoryEstimate {
    size_t weights_bytes = 0;        // 模型权重
    size_t kv_cache_bytes = 0;       // KV Cache
    size_t activation_bytes = 0;     // 激活值（中间结果）
    size_t workspace_bytes = 0;      // cuBLAS workspace 等
    size_t total_bytes = 0;          // 总计
    
    void compute_total() {
        total_bytes = weights_bytes + kv_cache_bytes + activation_bytes + workspace_bytes;
    }
    
    // 人类可读的大小
    std::string to_string() const;
};

// 层分配到设备的映射
struct DeviceMap {
    std::vector<int> layer_to_device;  // layer_to_device[i] = GPU id
    int embedding_device = 0;
    int lm_head_device = 0;
    int num_devices = 1;
    
    // 创建单卡映射
    static DeviceMap single_device(int num_layers, int device_id = 0) {
        DeviceMap dm;
        dm.layer_to_device.assign(num_layers, device_id);
        dm.embedding_device = device_id;
        dm.lm_head_device = device_id;
        dm.num_devices = 1;
        return dm;
    }
    
    // 自动生成：根据 GPU 显存和模型大小
    static DeviceMap auto_map(const ModelConfig& config, 
                              const std::vector<size_t>& gpu_free_memory,
                              int ctx_len,
                              int batch_size = 1);
    
    // 获取某设备上的层范围 [start, end)
    std::pair<int, int> device_layer_range(int device_id) const {
        int start = -1, end = -1;
        for (size_t i = 0; i < layer_to_device.size(); ++i) {
            if (layer_to_device[i] == device_id) {
                if (start < 0) start = static_cast<int>(i);
                end = static_cast<int>(i) + 1;
            }
        }
        return {start, end};
    }
    
    // 打印映射
    void print() const;
};

// Runtime 后端接口
class IRuntime {
public:
    virtual ~IRuntime() = default;
    
    // 获取后端名称
    virtual std::string name() const = 0;
    
    // 检查后端是否可用
    virtual bool available() const = 0;
    
    // 加载模型到指定设备
    virtual Error load(const std::string& model_path, 
                       const ModelConfig& config, 
                       const DeviceMap& device_map) = 0;
    
    // Prefill: 处理 prompt，填充 KV cache
    // tokens: 输入 token IDs
    // session: 会话状态（包含 KV cache）
    virtual Error prefill(const std::vector<int>& tokens, Session& session) = 0;
    
    // Prefill 并返回最后位置的 logits (用于采样第一个生成 token)
    virtual Error prefill_with_logits(const std::vector<int>& tokens, Session& session, std::vector<float>& logits) = 0;
    
    // Decode: 生成下一个 token 的 logits
    // last_token: 上一个生成的 token
    // session: 会话状态
    // logits: 输出的 logits [vocab_size]
    virtual Error decode(int last_token, Session& session, std::vector<float>& logits) = 0;
    
    // 显存估算（用于自动切分）
    virtual MemoryEstimate estimate_memory(const ModelConfig& config, 
                                           int ctx_len, 
                                           int batch_size = 1) = 0;
    
    // 分配 KV cache
    virtual Error allocate_kv_cache(Session& session) = 0;
    
    // 释放 KV cache
    virtual void free_kv_cache(Session& session) = 0;
    
    // 卸载模型
    virtual void unload() = 0;
    
    // 模型是否已加载
    virtual bool loaded() const = 0;
    
    // 获取当前 device map
    virtual const DeviceMap& device_map() const = 0;
};

// Runtime 工厂
class RuntimeFactory {
public:
    // 创建 CUDA Runtime
    static std::unique_ptr<IRuntime> create_cuda();
    
    // 创建 CPU Runtime（用于正确性参考）
    static std::unique_ptr<IRuntime> create_cpu();
    
    // 自动选择最佳 Runtime
    static std::unique_ptr<IRuntime> create_auto();
};

}  // namespace ember

````

### File: runtime/runtime_setup.h

````h
#pragma once

#include "iruntime.h"

namespace ember {

struct RuntimeSetup {
    Session session;
    bool loaded = false;
    bool kv_allocated = false;
};

inline Error load_runtime(IRuntime& runtime,
                          const std::string& model_path,
                          const ModelConfig& model_config,
                          const DeviceMap& device_map,
                          RuntimeSetup& setup) {
    Error err = runtime.load(model_path, model_config, device_map);
    if (err) return err;
    setup.loaded = true;
    return Error::success();
}

inline Error init_session_and_kv(IRuntime& runtime,
                                 const ModelConfig& model_config,
                                 const RuntimeConfig& runtime_config,
                                 RuntimeSetup& setup) {
    setup.session.init(model_config, runtime_config);
    Error err = runtime.allocate_kv_cache(setup.session);
    if (err) return err;
    setup.kv_allocated = true;
    return Error::success();
}

inline void shutdown_runtime(IRuntime& runtime, RuntimeSetup& setup) {
    if (setup.kv_allocated) {
        runtime.free_kv_cache(setup.session);
        setup.kv_allocated = false;
    }
    if (setup.loaded) {
        runtime.unload();
        setup.loaded = false;
    }
}

}  // namespace ember

````

### File: backends/cuda/cuda_runtime.cpp (prefill_with_logits (entry))

````cpp
        err = forward_layer(layer_idx, batch_size, seq_len, start_pos, session, /*skip_input_copy=*/false);
        if (err) return err;
    }
    
    // 同步所有设备
    for (int dev = 0; dev < device_map_.num_devices; ++dev) {
        cuda_sync(dev);
    }

    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/false,
                                                  /*include_final_norm=*/false,
                                                  /*include_lm_head=*/false));
    
    // 更新位置
    session.set_cur_pos(start_pos + seq_len);
    
    return Error::success();
}

// 带 logits 返回的 prefill (用于立即采样)
Error CudaRuntime::prefill_with_logits(const std::vector<int>& tokens, Session& session, std::vector<float>& logits) {
    Error err = prefill(tokens, session);
    if (err) return err;
    
    // Final norm 和 LM head 来计算最后一个位置的 logits
    int batch_size = 1;
    int seq_len = static_cast<int>(tokens.size());
    
    err = forward_final_norm(batch_size, seq_len, session);
    if (err) return err;
    
    err = forward_lm_head(batch_size, seq_len);
    if (err) return err;
    
    // 拷贝 logits
    int lm_device = device_map_.lm_head_device;
    cuda_sync(lm_device);
    
    logits.resize(config_.vocab_size);
    auto d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(logits.data(), activations_[lm_device].logits, 
                          config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    auto d2h_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_d2h_ms_(std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count());
    
    return Error::success();
}

Error CudaRuntime::decode(int last_token, Session& session, std::vector<float>& logits) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }

    if (profile_layers_) {
        last_layer_profile_ms_.assign(static_cast<size_t>(config_.num_layers), 0.0f);
    }
    EMBER_RETURN_IF_ERROR(decode_single_forward_to_lm_head_(last_token, session));
    
    // 同步并拷贝 logits 回 CPU
    int lm_device = device_map_.lm_head_device;
    if (profile_stages_) {
        for (int dev = 0; dev < device_map_.num_devices; ++dev) {
            cuda_sync(dev);
        }
    } else {
    cuda_sync(lm_device);
    }
    
    logits.resize(config_.vocab_size);
    auto d2h_start = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(logits.data(), activations_[lm_device].logits, 
                          config_.vocab_size * sizeof(float), cudaMemcpyDeviceToHost));
    auto d2h_end = std::chrono::high_resolution_clock::now();
    add_stage_profile_d2h_ms_(std::chrono::duration<double, std::milli>(d2h_end - d2h_start).count());

    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/false,
                                                  /*include_final_norm=*/true,
                                                  /*include_lm_head=*/true));
    
    // 更新位置
    session.advance(1);
    
    return Error::success();
}

Error CudaRuntime::forward_layer(int layer_idx,
                                 int batch_size,
                                 int seq_len,
                                 int start_pos,
                                 Session& session,
                                 bool skip_input_copy,
                                 const int* start_pos_by_batch) {
    int device_id = device_map_.layer_to_device[layer_idx];
    auto& act = activations_[device_id];
    auto& layer = weights_.layers[layer_idx];
    auto stream = streams_[device_id];
    auto& cublas = cublas_handles_[device_id];
    
    CUDA_CHECK(cudaSetDevice(device_id));
    cublasSetStream(cublas.get(), stream);
    
    DType compute_dtype = weights_.dtype;
    cudaDataType_t cuda_dtype = to_cuda_dtype(compute_dtype);
    size_t hidden_size = config_.hidden_size;
    size_t num_heads = config_.num_heads;
    size_t num_kv_heads = config_.num_kv_heads;
    size_t head_dim = config_.head_dim;
    size_t intermediate_size = config_.intermediate_size;
    size_t elem_size = dtype_size(compute_dtype);
    
    // 如果上一层在不同设备，需要拷贝 hidden_states
    if (layer_idx > 0) {
        int prev_device = device_map_.layer_to_device[layer_idx - 1];
        if (prev_device != device_id) {
            if (!skip_input_copy) {
            size_t size = batch_size * seq_len * hidden_size * elem_size;
            auto t0 = std::chrono::high_resolution_clock::now();
            Error err = copy_bytes_peer_or_staged(act.hidden_states, device_id,
                                                  activations_[prev_device].hidden_states, prev_device,
                                                  size);
            if (err) return err;
            if (profile_stages_) {
                auto t1 = std::chrono::high_resolution_clock::now();
                last_stage_profile_ms_.p2p_ms += static_cast<float>(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            }
        }
    }

    if (profile_layers_ && static_cast<size_t>(layer_idx) < last_layer_profile_ms_.size()) {
        auto& ev = profile_events_[device_id];
        CUDA_CHECK(cudaEventRecord(ev.start, stream));
    }

    if (session.runtime_config().check_correctness) {
        int target = session.runtime_config().dump_layer;
        if (target < 0 || target == layer_idx) {
            Error err = dump_last_row(session.runtime_config().dump_dir,
                                      "layer_" + std::to_string(layer_idx) + "_layer_input",
                                      device_id, act.hidden_states,
                                      seq_len, hidden_size, compute_dtype, stream);
            if (err) return err;
        }
    }
    
    // =====================================================================
    // Input LayerNorm
    // =====================================================================
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_start, stream));
    }
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.norm_out),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.norm_out),
            static_cast<const half*>(act.hidden_states),
            static_cast<const half*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    }
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_end, stream));
        CUDA_CHECK(cudaEventRecord(sev.attn_start, stream));
    }
    
    // =====================================================================
    // QKV Projection: norm_out @ W_q/k/v -> q/k/v_proj_out
    // 权重布局: [out_features, in_features] (row-major, Qwen3 safetensors格式)
    // 输入: [batch*seq, hidden_size]
    // 输出: [batch*seq, out_features]
    // =====================================================================
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    
    int M = batch_size * seq_len;  // 批次*序列长度
    
    // Q projection: [M, hidden] @ [hidden, num_heads*head_dim]^T = [M, num_heads*head_dim]
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,  // W^T @ X
        num_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.q_proj_weight, cuda_dtype, hidden_size,  // W: [num_heads*head_dim, hidden] -> W^T
        act.norm_out, cuda_dtype, hidden_size,         // X: [M, hidden]
        &beta_zero,
        act.q_proj_out, cuda_dtype, num_heads * head_dim,  // Y: [M, num_heads*head_dim]
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // K projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.k_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.k_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // V projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.v_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.v_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // =====================================================================
    // Q/K per-head RMSNorm (Qwen3)
    // =====================================================================
    int q_rows = batch_size * seq_len * static_cast<int>(num_heads);
    int k_rows = batch_size * seq_len * static_cast<int>(num_kv_heads);
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.q_proj_out),
            static_cast<const half*>(act.q_proj_out),
            static_cast<const half*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_f16(
            static_cast<half*>(act.k_proj_out),
            static_cast<const half*>(act.k_proj_out),
            static_cast<const half*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    }
    
    // =====================================================================
    // Apply RoPE to Q and K
    // =====================================================================
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::apply_rope_bf16(
                static_cast<__nv_bfloat16*>(act.q_proj_out),
                static_cast<__nv_bfloat16*>(act.k_proj_out),
                batch_size, seq_len, num_heads, num_kv_heads, head_dim,

````

### File: backends/cuda/cuda_runtime.cpp (forward_layer (entry))

````cpp
    EMBER_RETURN_IF_ERROR(finalize_stage_profile_(/*sync_all_devices=*/false,
                                                  /*include_final_norm=*/true,
                                                  /*include_lm_head=*/true));
    
    // 更新位置
    session.advance(1);
    
    return Error::success();
}

Error CudaRuntime::forward_layer(int layer_idx,
                                 int batch_size,
                                 int seq_len,
                                 int start_pos,
                                 Session& session,
                                 bool skip_input_copy,
                                 const int* start_pos_by_batch) {
    int device_id = device_map_.layer_to_device[layer_idx];
    auto& act = activations_[device_id];
    auto& layer = weights_.layers[layer_idx];
    auto stream = streams_[device_id];
    auto& cublas = cublas_handles_[device_id];
    
    CUDA_CHECK(cudaSetDevice(device_id));
    cublasSetStream(cublas.get(), stream);
    
    DType compute_dtype = weights_.dtype;
    cudaDataType_t cuda_dtype = to_cuda_dtype(compute_dtype);
    size_t hidden_size = config_.hidden_size;
    size_t num_heads = config_.num_heads;
    size_t num_kv_heads = config_.num_kv_heads;
    size_t head_dim = config_.head_dim;
    size_t intermediate_size = config_.intermediate_size;
    size_t elem_size = dtype_size(compute_dtype);
    
    // 如果上一层在不同设备，需要拷贝 hidden_states
    if (layer_idx > 0) {
        int prev_device = device_map_.layer_to_device[layer_idx - 1];
        if (prev_device != device_id) {
            if (!skip_input_copy) {
            size_t size = batch_size * seq_len * hidden_size * elem_size;
            auto t0 = std::chrono::high_resolution_clock::now();
            Error err = copy_bytes_peer_or_staged(act.hidden_states, device_id,
                                                  activations_[prev_device].hidden_states, prev_device,
                                                  size);
            if (err) return err;
            if (profile_stages_) {
                auto t1 = std::chrono::high_resolution_clock::now();
                last_stage_profile_ms_.p2p_ms += static_cast<float>(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            }
        }
    }

    if (profile_layers_ && static_cast<size_t>(layer_idx) < last_layer_profile_ms_.size()) {
        auto& ev = profile_events_[device_id];
        CUDA_CHECK(cudaEventRecord(ev.start, stream));
    }

    if (session.runtime_config().check_correctness) {
        int target = session.runtime_config().dump_layer;
        if (target < 0 || target == layer_idx) {
            Error err = dump_last_row(session.runtime_config().dump_dir,
                                      "layer_" + std::to_string(layer_idx) + "_layer_input",
                                      device_id, act.hidden_states,
                                      seq_len, hidden_size, compute_dtype, stream);
            if (err) return err;
        }
    }
    
    // =====================================================================
    // Input LayerNorm
    // =====================================================================
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_start, stream));
    }
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.norm_out),
            static_cast<const __nv_bfloat16*>(act.hidden_states),
            static_cast<const __nv_bfloat16*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.norm_out),
            static_cast<const half*>(act.hidden_states),
            static_cast<const half*>(layer.input_layernorm_weight),
            batch_size, seq_len, hidden_size,
            config_.rms_norm_eps,
            stream
        );
    }
    if (profile_stages_) {
        auto& sev = layer_stage_events_[static_cast<size_t>(layer_idx)];
        CUDA_CHECK(cudaEventRecord(sev.in_norm_end, stream));
        CUDA_CHECK(cudaEventRecord(sev.attn_start, stream));
    }
    
    // =====================================================================
    // QKV Projection: norm_out @ W_q/k/v -> q/k/v_proj_out
    // 权重布局: [out_features, in_features] (row-major, Qwen3 safetensors格式)
    // 输入: [batch*seq, hidden_size]
    // 输出: [batch*seq, out_features]
    // =====================================================================
    const float alpha_one = 1.0f;
    const float beta_zero = 0.0f;
    
    int M = batch_size * seq_len;  // 批次*序列长度
    
    // Q projection: [M, hidden] @ [hidden, num_heads*head_dim]^T = [M, num_heads*head_dim]
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,  // W^T @ X
        num_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.q_proj_weight, cuda_dtype, hidden_size,  // W: [num_heads*head_dim, hidden] -> W^T
        act.norm_out, cuda_dtype, hidden_size,         // X: [M, hidden]
        &beta_zero,
        act.q_proj_out, cuda_dtype, num_heads * head_dim,  // Y: [M, num_heads*head_dim]
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // K projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.k_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.k_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
    
    // V projection
    CUBLAS_CHECK(cublasGemmEx(
        cublas.get(),
        CUBLAS_OP_T, CUBLAS_OP_N,
        num_kv_heads * head_dim, M, hidden_size,
        &alpha_one,
        layer.v_proj_weight, cuda_dtype, hidden_size,
        act.norm_out, cuda_dtype, hidden_size,
        &beta_zero,
        act.v_proj_out, cuda_dtype, num_kv_heads * head_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // =====================================================================
    // Q/K per-head RMSNorm (Qwen3)
    // =====================================================================
    int q_rows = batch_size * seq_len * static_cast<int>(num_heads);
    int k_rows = batch_size * seq_len * static_cast<int>(num_kv_heads);
    if (compute_dtype == DType::BF16) {
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(act.q_proj_out),
            static_cast<const __nv_bfloat16*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_bf16(
            static_cast<__nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(act.k_proj_out),
            static_cast<const __nv_bfloat16*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    } else {
        kernels::rms_norm_f16(
            static_cast<half*>(act.q_proj_out),
            static_cast<const half*>(act.q_proj_out),
            static_cast<const half*>(layer.q_norm_weight),
            q_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
        kernels::rms_norm_f16(
            static_cast<half*>(act.k_proj_out),
            static_cast<const half*>(act.k_proj_out),
            static_cast<const half*>(layer.k_norm_weight),
            k_rows, 1, static_cast<int>(head_dim),
            config_.rms_norm_eps,
            stream
        );
    }
    
    // =====================================================================
    // Apply RoPE to Q and K
    // =====================================================================
    if (!start_pos_by_batch) {
        if (compute_dtype == DType::BF16) {
            kernels::apply_rope_bf16(
                static_cast<__nv_bfloat16*>(act.q_proj_out),
                static_cast<__nv_bfloat16*>(act.k_proj_out),
                batch_size, seq_len, num_heads, num_kv_heads, head_dim,
                start_pos, config_.rope_theta,
                stream
            );
        } else {
            kernels::apply_rope_f16(
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


````

### File: backends/cuda/cuda_runtime.cpp (pipelined prefill stage comments)

````cpp
        CUDA_CHECK(cudaSetDevice(gpu0));
        CUDA_CHECK(cudaEventSynchronize(xfer_done[slot]));

        CUDA_CHECK(cudaSetDevice(gpu1));
        void* saved_hidden = activations_[gpu1].hidden_states;
        activations_[gpu1].hidden_states = gpu1_io[slot];

        const int chunk_start_pos = start_pos0 + off;
        for (int layer = boundary; layer < config_.num_layers; ++layer) {
            const bool skip = (layer == boundary);
            Error e = forward_layer(layer, batch_size, clen, chunk_start_pos, session, /*skip_input_copy=*/skip);
            if (e) {
                activations_[gpu1].hidden_states = saved_hidden;
                return e;
            }
            Error pe = accumulate_layer_stage_ms(layer);
            if (pe) {
                activations_[gpu1].hidden_states = saved_hidden;
                return pe;
            }
        }

        // Mark this slot as free to reuse (recorded on gpu1 compute stream).
        CUDA_CHECK(cudaEventRecord(stage1_done[slot], streams_[gpu1]));
        activations_[gpu1].hidden_states = saved_hidden;
        return Error::success();
    };

    for (int i = 0; i < num_chunks; ++i) {
        const int off = i * chunk_len;
        const int clen = std::min(chunk_len, total_len - off);
        const int slot = i & 1;

        // Slot reuse guard: wait until stage1 has finished using this slot.
        if (i >= 2) {
            CUDA_CHECK(cudaSetDevice(gpu1));
            CUDA_CHECK(cudaEventSynchronize(stage1_done[slot]));
        }

        // --------------------
        // Stage 0 (GPU0): embedding + layers [0, boundary)
        // --------------------
        CUDA_CHECK(cudaSetDevice(gpu0));
        auto compute_stream0 = streams_[gpu0];
        auto transfer_stream0 = transfer_streams_[gpu0];

        CUDA_CHECK(cudaMemcpyAsync(d_input_ids, tokens.data() + off,
                                   static_cast<size_t>(clen) * sizeof(int),
                                   cudaMemcpyHostToDevice, compute_stream0));

        auto& act0 = activations_[gpu0];
        if (profile_stages_) {
            TimedEvents ee;
            ee.device_id = gpu0;
            CUDA_CHECK(cudaSetDevice(gpu0));
            CUDA_CHECK(cudaEventCreate(&ee.start));
            CUDA_CHECK(cudaEventCreate(&ee.end));
            CUDA_CHECK(cudaEventRecord(ee.start, compute_stream0));
            embed_events.push_back(ee);
        }
        if (weights_.dtype == DType::BF16) {
            kernels::embedding_lookup_bf16(
                static_cast<__nv_bfloat16*>(act0.hidden_states),
                static_cast<const __nv_bfloat16*>(weights_.embed_tokens),
                d_input_ids,
                batch_size, clen, config_.hidden_size,
                compute_stream0
            );
        } else {
            kernels::embedding_lookup_f16(
                static_cast<half*>(act0.hidden_states),
                static_cast<const half*>(weights_.embed_tokens),
                d_input_ids,
                batch_size, clen, config_.hidden_size,
                compute_stream0
            );
        }
        if (profile_stages_) {
            auto& ee = embed_events.back();
            CUDA_CHECK(cudaEventRecord(ee.end, compute_stream0));
        }

        const int chunk_start_pos = start_pos0 + off;
        for (int layer = 0; layer < boundary; ++layer) {
            err = forward_layer(layer, batch_size, clen, chunk_start_pos, session, /*skip_input_copy=*/false);
            if (err) {
                destroy_copy_events();
                cleanup();
                return err;
            }
            Error pe = accumulate_layer_stage_ms(layer);
            if (pe) {
                destroy_copy_events();
                cleanup();
                return pe;
            }
        }

        // Snapshot stage0 output so GPU0 can continue computing next chunk while transfer reads this buffer.
        const size_t bytes = static_cast<size_t>(clen) * hidden * elem;
        CUDA_CHECK(cudaMemcpyAsync(stage0_out[slot], act0.hidden_states, bytes,
                                   cudaMemcpyDeviceToDevice, compute_stream0));
        CUDA_CHECK(cudaEventRecord(stage0_ready[slot], compute_stream0));

        // --------------------
        // Transfer (GPU0): stage0_out -> gpu1_io
        // --------------------
        if (overlap) {
            CUDA_CHECK(cudaStreamWaitEvent(transfer_stream0, stage0_ready[slot], 0));
            if (profile_stages_) {
                TimedEvents ce;
                ce.device_id = gpu0;
                CUDA_CHECK(cudaSetDevice(gpu0));
                CUDA_CHECK(cudaEventCreate(&ce.start));
                CUDA_CHECK(cudaEventCreate(&ce.end));
                CUDA_CHECK(cudaEventRecord(ce.start, transfer_stream0));
                copy_events.push_back(ce);
            }
            cudaError_t cperr = cudaMemcpyPeerAsync(gpu1_io[slot], gpu1, stage0_out[slot], gpu0, bytes, transfer_stream0);
            if (cperr != cudaSuccess) {
                // Fall back to staging (synchronous) on failure.
                cudaGetLastError();
                CUDA_CHECK(cudaStreamSynchronize(transfer_stream0));
                Error se = copy_bytes_peer_or_staged(gpu1_io[slot], gpu1, stage0_out[slot], gpu0, bytes);
                if (se) {
                    destroy_copy_events();
                    cleanup();
                    return se;
                }
            }
            if (profile_stages_) {
                auto& ce = copy_events.back();
                CUDA_CHECK(cudaEventRecord(ce.end, transfer_stream0));
            }
            CUDA_CHECK(cudaEventRecord(xfer_done[slot], transfer_stream0));
        } else {
            // Baseline: block on stage0 completion, then do synchronous peer copy.
            CUDA_CHECK(cudaEventSynchronize(stage0_ready[slot]));
            auto t0 = std::chrono::high_resolution_clock::now();
            err = copy_bytes_peer_or_staged(gpu1_io[slot], gpu1, stage0_out[slot], gpu0, bytes);
            if (err) {
                destroy_copy_events();
                cleanup();
                return err;
            }
            if (profile_stages_) {
                auto t1 = std::chrono::high_resolution_clock::now();
                last_stage_profile_ms_.p2p_ms += static_cast<float>(
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            CUDA_CHECK(cudaEventRecord(xfer_done[slot], compute_stream0));
        }

        // --------------------
        // Stage 1 (GPU1): consume previous chunk if available.
        // --------------------
        if (i - 1 >= 0) {
            Error e = run_stage1(i - 1);
            if (e) {
                destroy_copy_events();
                cleanup();
                return e;
            }
        }
    }

    // Consume the last chunk on stage 1.
    if (num_chunks > 0) {
        err = run_stage1(num_chunks - 1);
        if (err) {
            destroy_copy_events();
            cleanup();
            return err;
        }
    }

    if (out_logits) {
        // Compute logits for the last prompt token from the last chunk output on GPU1.
        const int last_chunk_index = num_chunks - 1;
        const int last_off = last_chunk_index * chunk_len;
        const int last_clen = std::min(chunk_len, total_len - last_off);
        const int slot = last_chunk_index & 1;

        const size_t bytes_per_token = hidden * elem;
        const char* base = static_cast<const char*>(gpu1_io[slot]);
        const void* last_hidden_ptr = static_cast<const void*>(base + static_cast<size_t>(last_clen - 1) * bytes_per_token);

        CUDA_CHECK(cudaSetDevice(gpu1));
        auto& act1 = activations_[gpu1];
        auto stream1 = streams_[gpu1];

        if (compute_dtype == DType::BF16) {
            kernels::rms_norm_bf16(
                static_cast<__nv_bfloat16*>(act1.norm_out),
                static_cast<const __nv_bfloat16*>(last_hidden_ptr),
                static_cast<const __nv_bfloat16*>(weights_.final_norm),
                /*batch_size=*/1, /*seq_len=*/1, static_cast<int>(hidden),
                config_.rms_norm_eps,
                stream1
            );
            kernels::copy_last_hidden_bf16(
                static_cast<__nv_bfloat16*>(act1.last_hidden),
                static_cast<const __nv_bfloat16*>(act1.norm_out),
                /*batch_size=*/1, /*seq_len=*/1, static_cast<int>(hidden),
                stream1
            );
        } else {
            kernels::rms_norm_f16(
                static_cast<half*>(act1.norm_out),
                static_cast<const half*>(last_hidden_ptr),
                static_cast<const half*>(weights_.final_norm),
                /*batch_size=*/1, /*seq_len=*/1, static_cast<int>(hidden),
                config_.rms_norm_eps,
                stream1
            );
            kernels::copy_last_hidden_f16(
                static_cast<half*>(act1.last_hidden),
                static_cast<const half*>(act1.norm_out),
                /*batch_size=*/1, /*seq_len=*/1, static_cast<int>(hidden),
                stream1
            );

````

---

## 3) 报告上下文（完整）

### Report: reports/stage1_split_profile_4b_20260225_mainline/stage12_summary.md

````md
# Stage 1.2 Pipeline Split Profiling

- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Generated at: `2026-02-25T01:32:46`

## Throughput vs Split
| split | mode | total_ms | tok/s_est | prefill_share_% |
| --- | --- | --- | --- | --- |
| 12+24 | no_overlap | 2770.230 | 46.206 | 15.73 |
| 12+24 | overlap | 2767.109 | 46.258 | 15.61 |
| 18+18 | no_overlap | 2830.028 | 45.229 | 16.08 |
| 18+18 | overlap | 2817.737 | 45.427 | 16.07 |
| 24+12 | no_overlap | 2957.505 | 43.280 | 16.86 |
| 24+12 | overlap | 2915.003 | 43.911 | 16.63 |
| 27+9 | no_overlap | 3014.120 | 42.467 | 16.69 |
| 27+9 | overlap | 2987.478 | 42.846 | 16.74 |
| 9+27 | no_overlap | 2769.227 | 46.222 | 15.83 |
| 9+27 | overlap | 2755.312 | 46.456 | 15.61 |

## Key Point
- Best rollout tok/s split: `9+27` in `overlap` mode (`46.456 tok/s`, total `2755.312 ms`).

## Bubble vs Split
| split | no_overlap_ms | overlap_ms | speedup_x | hidden_% |
| --- | --- | --- | --- | --- |
| 12+24 | 2770.230 | 2767.109 | 1.0011 | 0.11 |
| 18+18 | 2830.028 | 2817.737 | 1.0044 | 0.43 |
| 24+12 | 2957.505 | 2915.003 | 1.0146 | 1.44 |
| 27+9 | 3014.120 | 2987.478 | 1.0089 | 0.88 |
| 9+27 | 2769.227 | 2755.312 | 1.0051 | 0.50 |

````

### Report: reports/stage1_split_profile_4b_20260225_mainline/stage12_vs_llama.md

````md
# Stage 1.2 Ember vs llama.cpp

- GGUF: `/home/dong/workspace/ember/reports/gguf/Qwen3-4B-BF16.gguf`
- llama split_mode: `layer`
- llama tensor_split: `auto/none`

| engine | config | prefill_ms | decode_tok_s | decode_per_token_ms | rollout_total_ms | rollout_tok_s | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ember | split=9+27 mode=overlap | 430.064 | 55.048 | 18.166 | 2755.312 | 46.456 | best split from stage12 sweep |
| llama.cpp | CUDA0 single | 542.871 | 73.898 | 13.532 | 2274.996 | 56.264 | llama-bench |
| llama.cpp | CUDA0/CUDA1 dual | 325.628 | 82.248 | 12.158 | 1881.900 | 68.016 | llama-bench split_mode=layer |

- Note: llama-bench and ember_stage_breakdown are different harnesses; compare as practical baseline, not strict apples-to-apples kernel microbenchmark.

````

### Report: reports/stage1_split_profile_4b_20260225_mainline/stage12_bubble_vs_split.csv

````csv
split,no_overlap_total_ms,overlap_total_ms,overlap_speedup_x,bubble_hidden_pct_est
12+24,2770.230,2767.109,1.0011,0.11
18+18,2830.028,2817.737,1.0044,0.43
24+12,2957.505,2915.003,1.0146,1.44
27+9,3014.120,2987.478,1.0089,0.88
9+27,2769.227,2755.312,1.0051,0.50

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`
