#pragma once

#include "cli.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace ember {

// Qwen3 模型超参数
struct Qwen3HParams {
    uint32_t n_vocab;       // 词表大小
    uint32_t n_ctx_train;   // 训练时上下文长度
    uint32_t n_embd;        // 嵌入维度
    uint32_t n_head;        // 注意力头数
    uint32_t n_head_kv;     // KV 头数 (GQA)
    uint32_t n_layer;       // 层数
    uint32_t n_rot;         // RoPE 维度
    uint32_t n_ff;          // FFN 隐藏层维度
    float rope_freq_base;   // RoPE 基础频率
    float rope_freq_scale;  // RoPE 频率缩放
    float rms_norm_eps;     // RMSNorm epsilon
};

// Tensor 信息
struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;
    size_t size_bytes;
    int ggml_type;          // 量化类型
    int gpu_id;             // 分配到的 GPU (-1 = CPU)
    void* data;             // 数据指针
};

// 层分配信息
struct LayerAllocation {
    int layer_id;
    int gpu_id;
    size_t memory_usage;
};

// 模型类
class Qwen3Model {
public:
    Qwen3Model() = default;
    ~Qwen3Model();
    
    // 禁止拷贝
    Qwen3Model(const Qwen3Model&) = delete;
    Qwen3Model& operator=(const Qwen3Model&) = delete;
    
    // 加载模型
    Error load(const std::string& path, const CliArgs& args);
    
    // 获取模型信息
    const Qwen3HParams& hparams() const { return hparams_; }
    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const std::vector<LayerAllocation>& layer_allocation() const { return layer_alloc_; }
    
    // 获取特定 tensor
    const TensorInfo* get_tensor(const std::string& name) const;
    
    // 模型是否已加载
    bool is_loaded() const { return loaded_; }
    
    // 打印模型信息
    void print_info() const;
    
private:
    // 解析 GGUF 文件头
    Error parse_gguf_header(const std::string& path);
    
    // 验证架构
    Error validate_architecture();
    
    // 加载 tensor
    Error load_tensors(const CliArgs& args);
    
    // 计算层分配
    Error compute_layer_allocation(const CliArgs& args);
    
    // 分配 GPU 内存
    Error allocate_gpu_memory();
    
private:
    bool loaded_ = false;
    std::string model_path_;
    std::string arch_name_;
    
    Qwen3HParams hparams_ = {};
    std::vector<TensorInfo> tensors_;
    std::unordered_map<std::string, size_t> tensor_map_;  // name -> index
    std::vector<LayerAllocation> layer_alloc_;
    
    // 文件映射
    void* mapped_data_ = nullptr;
    size_t mapped_size_ = 0;
    
    // GPU 资源
    std::vector<void*> gpu_buffers_;
};

// 全局模型实例（简化管理）
Qwen3Model& get_model();

} // namespace ember
