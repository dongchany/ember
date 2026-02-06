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
