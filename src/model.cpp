#include "model.h"
#include <fstream>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace ember {

// GGUF 魔数和版本
static constexpr uint32_t GGUF_MAGIC = 0x46554747;  // "GGUF"
static constexpr uint32_t GGUF_VERSION = 3;

// GGUF 数据类型
enum GGUFType {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// GGML 量化类型
enum GGMLType {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_BF16 = 30,
};

// 获取类型大小（字节/元素）
static size_t ggml_type_size(int type) {
    switch (type) {
        case GGML_TYPE_F32:  return 4;
        case GGML_TYPE_F16:  return 2;
        case GGML_TYPE_BF16: return 2;
        case GGML_TYPE_Q4_0: return 18;   // block size
        case GGML_TYPE_Q4_1: return 20;
        case GGML_TYPE_Q5_0: return 22;
        case GGML_TYPE_Q5_1: return 24;
        case GGML_TYPE_Q8_0: return 34;
        case GGML_TYPE_Q8_1: return 36;
        case GGML_TYPE_Q4_K: return 144;
        case GGML_TYPE_Q5_K: return 176;
        case GGML_TYPE_Q6_K: return 210;
        default: return 0;
    }
}

// 获取 block size
static size_t ggml_blck_size(int type) {
    switch (type) {
        case GGML_TYPE_F32:  return 1;
        case GGML_TYPE_F16:  return 1;
        case GGML_TYPE_BF16: return 1;
        case GGML_TYPE_Q4_0: return 32;
        case GGML_TYPE_Q4_1: return 32;
        case GGML_TYPE_Q5_0: return 32;
        case GGML_TYPE_Q5_1: return 32;
        case GGML_TYPE_Q8_0: return 32;
        case GGML_TYPE_Q8_1: return 32;
        case GGML_TYPE_Q4_K: return 256;
        case GGML_TYPE_Q5_K: return 256;
        case GGML_TYPE_Q6_K: return 256;
        default: return 1;
    }
}

// 全局模型实例
static Qwen3Model g_model;
Qwen3Model& get_model() { return g_model; }

Qwen3Model::~Qwen3Model() {
    // 释放 GPU 内存
    for (void* buf : gpu_buffers_) {
        if (buf) cudaFree(buf);
    }
    
    // 取消文件映射
    if (mapped_data_ && mapped_size_ > 0) {
        munmap(mapped_data_, mapped_size_);
    }
}

Error Qwen3Model::load(const std::string& path, const CliArgs& args) {
    Timer timer;
    LOG_INFO("正在加载模型: %s", path.c_str());
    
    model_path_ = path;
    
    // 1. 检查文件
    if (!file_exists(path)) {
        LOG_ERROR("模型文件不存在: %s", path.c_str());
        return Error::FILE_NOT_FOUND;
    }
    
    // 2. 解析 GGUF 头
    Error err = parse_gguf_header(path);
    if (err != Error::OK) return err;
    
    // 3. 验证架构
    err = validate_architecture();
    if (err != Error::OK) return err;
    
    // 4. 计算层分配
    err = compute_layer_allocation(args);
    if (err != Error::OK) return err;
    
    // 5. 加载 tensor 到 GPU
    err = load_tensors(args);
    if (err != Error::OK) return err;
    
    loaded_ = true;
    LOG_INFO("模型加载完成，耗时: %.2f 秒", timer.elapsed_s());
    
    return Error::OK;
}

Error Qwen3Model::parse_gguf_header(const std::string& path) {
    // 打开文件
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        LOG_ERROR("无法打开文件: %s", path.c_str());
        return Error::FILE_NOT_FOUND;
    }
    
    // 获取文件大小
    struct stat st;
    fstat(fd, &st);
    mapped_size_ = st.st_size;
    
    // 内存映射
    mapped_data_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    
    if (mapped_data_ == MAP_FAILED) {
        LOG_ERROR("内存映射失败");
        mapped_data_ = nullptr;
        return Error::FILE_NOT_FOUND;
    }
    
    const uint8_t* ptr = static_cast<const uint8_t*>(mapped_data_);
    const uint8_t* end = ptr + mapped_size_;
    
    // 读取魔数
    uint32_t magic;
    memcpy(&magic, ptr, sizeof(magic));
    ptr += sizeof(magic);
    
    if (magic != GGUF_MAGIC) {
        LOG_ERROR("无效的 GGUF 文件（魔数不匹配）");
        return Error::INVALID_MODEL;
    }
    
    // 读取版本
    uint32_t version;
    memcpy(&version, ptr, sizeof(version));
    ptr += sizeof(version);
    
    if (version < 2 || version > GGUF_VERSION) {
        LOG_ERROR("不支持的 GGUF 版本: %u", version);
        return Error::INVALID_MODEL;
    }
    
    // 读取 tensor 数量和元数据 KV 数量
    uint64_t n_tensors, n_kv;
    memcpy(&n_tensors, ptr, sizeof(n_tensors));
    ptr += sizeof(n_tensors);
    memcpy(&n_kv, ptr, sizeof(n_kv));
    ptr += sizeof(n_kv);
    
    LOG_INFO("GGUF v%u: %lu tensors, %lu metadata", version, n_tensors, n_kv);
    
    // 读取元数据
    auto read_string = [&ptr, end]() -> std::string {
        if (ptr + 8 > end) return "";
        uint64_t len;
        memcpy(&len, ptr, sizeof(len));
        ptr += sizeof(len);
        if (ptr + len > end) return "";
        std::string s(reinterpret_cast<const char*>(ptr), len);
        ptr += len;
        return s;
    };
    
    auto skip_value = [&ptr, end](uint32_t type) {
        switch (type) {
            case GGUF_TYPE_UINT8:   ptr += 1; break;
            case GGUF_TYPE_INT8:    ptr += 1; break;
            case GGUF_TYPE_UINT16:  ptr += 2; break;
            case GGUF_TYPE_INT16:   ptr += 2; break;
            case GGUF_TYPE_UINT32:  ptr += 4; break;
            case GGUF_TYPE_INT32:   ptr += 4; break;
            case GGUF_TYPE_FLOAT32: ptr += 4; break;
            case GGUF_TYPE_BOOL:    ptr += 1; break;
            case GGUF_TYPE_UINT64:  ptr += 8; break;
            case GGUF_TYPE_INT64:   ptr += 8; break;
            case GGUF_TYPE_FLOAT64: ptr += 8; break;
            case GGUF_TYPE_STRING: {
                uint64_t len;
                memcpy(&len, ptr, sizeof(len));
                ptr += sizeof(len) + len;
                break;
            }
            case GGUF_TYPE_ARRAY: {
                uint32_t arr_type;
                uint64_t arr_len;
                memcpy(&arr_type, ptr, sizeof(arr_type));
                ptr += sizeof(arr_type);
                memcpy(&arr_len, ptr, sizeof(arr_len));
                ptr += sizeof(arr_len);
                // 递归跳过数组元素
                for (uint64_t i = 0; i < arr_len; ++i) {
                    // 简化处理，假设数组元素不是嵌套数组
                    switch (arr_type) {
                        case GGUF_TYPE_UINT32: ptr += 4; break;
                        case GGUF_TYPE_INT32:  ptr += 4; break;
                        case GGUF_TYPE_FLOAT32: ptr += 4; break;
                        case GGUF_TYPE_STRING: {
                            uint64_t slen;
                            memcpy(&slen, ptr, sizeof(slen));
                            ptr += sizeof(slen) + slen;
                            break;
                        }
                        default: ptr += 4; break;
                    }
                }
                break;
            }
            default: ptr += 4; break;
        }
    };
    
    // 解析元数据
    for (uint64_t i = 0; i < n_kv && ptr < end; ++i) {
        std::string key = read_string();
        
        uint32_t val_type;
        memcpy(&val_type, ptr, sizeof(val_type));
        ptr += sizeof(val_type);
        
        // 提取关键信息
        if (key == "general.architecture") {
            arch_name_ = read_string();
            LOG_INFO("模型架构: %s", arch_name_.c_str());
        }
        else if (key == "qwen3.embedding_length" || key == "qwen2.embedding_length") {
            memcpy(&hparams_.n_embd, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);
        }
        else if (key == "qwen3.block_count" || key == "qwen2.block_count") {
            memcpy(&hparams_.n_layer, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);
        }
        else if (key == "qwen3.attention.head_count" || key == "qwen2.attention.head_count") {
            memcpy(&hparams_.n_head, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);
        }
        else if (key == "qwen3.attention.head_count_kv" || key == "qwen2.attention.head_count_kv") {
            memcpy(&hparams_.n_head_kv, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);
        }
        else if (key == "qwen3.feed_forward_length" || key == "qwen2.feed_forward_length") {
            memcpy(&hparams_.n_ff, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);
        }
        else if (key == "qwen3.context_length" || key == "qwen2.context_length") {
            memcpy(&hparams_.n_ctx_train, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);
        }
        else if (key == "qwen3.rope.freq_base" || key == "qwen2.rope.freq_base") {
            memcpy(&hparams_.rope_freq_base, ptr, sizeof(float));
            ptr += sizeof(float);
        }
        else if (key == "qwen3.attention.layer_norm_rms_epsilon" || 
                 key == "qwen2.attention.layer_norm_rms_epsilon") {
            memcpy(&hparams_.rms_norm_eps, ptr, sizeof(float));
            ptr += sizeof(float);
        }
        else if (key.find(".vocab_size") != std::string::npos) {
            memcpy(&hparams_.n_vocab, ptr, sizeof(uint32_t));
            ptr += sizeof(uint32_t);
        }
        else {
            skip_value(val_type);
        }
    }
    
    // 计算 RoPE 维度
    if (hparams_.n_head > 0) {
        hparams_.n_rot = hparams_.n_embd / hparams_.n_head;
    }
    
    // 设置默认值
    if (hparams_.rope_freq_base == 0) hparams_.rope_freq_base = 1000000.0f;
    if (hparams_.rms_norm_eps == 0) hparams_.rms_norm_eps = 1e-6f;
    
    // 读取 tensor 信息
    tensors_.reserve(n_tensors);
    
    size_t data_offset = 0;  // tensor 数据起始位置（对齐后）
    
    for (uint64_t i = 0; i < n_tensors && ptr < end; ++i) {
        TensorInfo info;
        info.name = read_string();
        
        uint32_t n_dims;
        memcpy(&n_dims, ptr, sizeof(n_dims));
        ptr += sizeof(n_dims);
        
        info.shape.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim;
            memcpy(&dim, ptr, sizeof(dim));
            ptr += sizeof(dim);
            info.shape[d] = dim;
        }
        
        uint32_t type;
        memcpy(&type, ptr, sizeof(type));
        ptr += sizeof(type);
        info.ggml_type = type;
        
        uint64_t offset;
        memcpy(&offset, ptr, sizeof(offset));
        ptr += sizeof(offset);
        
        // 计算 tensor 大小
        int64_t n_elements = 1;
        for (auto dim : info.shape) n_elements *= dim;
        info.size_bytes = (n_elements / ggml_blck_size(type)) * ggml_type_size(type);
        
        info.gpu_id = -1;  // 默认在 CPU
        info.data = nullptr;
        
        tensor_map_[info.name] = tensors_.size();
        tensors_.push_back(info);
    }
    
    LOG_INFO("解析完成: %zu tensors", tensors_.size());
    
    return Error::OK;
}

Error Qwen3Model::validate_architecture() {
    // 只支持 qwen3（dense）和 qwen2（dense）
    if (arch_name_ != "qwen3" && arch_name_ != "qwen2") {
        LOG_ERROR("不支持的架构: %s（仅支持 qwen3/qwen2 dense）", arch_name_.c_str());
        return Error::UNSUPPORTED_ARCH;
    }
    
    // 检查是否为 MoE（通过检查 expert 相关 tensor）
    for (const auto& t : tensors_) {
        if (t.name.find("expert") != std::string::npos) {
            LOG_ERROR("检测到 MoE 模型，当前仅支持 dense 模型");
            return Error::UNSUPPORTED_ARCH;
        }
    }
    
    LOG_INFO("架构验证通过: %s (dense)", arch_name_.c_str());
    return Error::OK;
}

Error Qwen3Model::compute_layer_allocation(const CliArgs& args) {
    int gpu_count = get_gpu_count();
    if (gpu_count == 0) {
        LOG_ERROR("未检测到 CUDA GPU");
        return Error::CUDA_ERROR;
    }
    
    // 验证 main_gpu
    if (args.main_gpu >= gpu_count) {
        LOG_ERROR("main_gpu (%d) 超出可用 GPU 数量 (%d)", args.main_gpu, gpu_count);
        return Error::INVALID_ARGUMENT;
    }
    
    // 验证 tensor_split
    if (!args.tensor_split.empty() && (int)args.tensor_split.size() > gpu_count) {
        LOG_ERROR("tensor_split 维度 (%zu) 超出 GPU 数量 (%d)", 
                  args.tensor_split.size(), gpu_count);
        return Error::INVALID_ARGUMENT;
    }
    
    // 计算实际使用的层数
    int n_gpu_layers = args.n_gpu_layers;
    if (n_gpu_layers < 0 || n_gpu_layers > (int)hparams_.n_layer) {
        n_gpu_layers = hparams_.n_layer;
    }
    
    LOG_INFO("GPU 层分配: %d / %u 层", n_gpu_layers, hparams_.n_layer);
    
    // 获取各 GPU 可用显存
    std::vector<size_t> gpu_free(gpu_count);
    for (int i = 0; i < gpu_count; ++i) {
        GpuInfo info = get_gpu_info(i);
        gpu_free[i] = info.free_memory;
    }
    
    // 计算分配比例
    std::vector<float> split_ratios;
    if (!args.tensor_split.empty()) {
        split_ratios = args.tensor_split;
        // 填充到 GPU 数量
        while ((int)split_ratios.size() < gpu_count) {
            split_ratios.push_back(0.0f);
        }
    } else {
        // 按显存比例自动分配
        float total = std::accumulate(gpu_free.begin(), gpu_free.end(), 0.0f);
        for (int i = 0; i < gpu_count; ++i) {
            split_ratios.push_back(gpu_free[i] / total);
        }
    }
    
    // 归一化
    float sum = std::accumulate(split_ratios.begin(), split_ratios.end(), 0.0f);
    for (auto& r : split_ratios) r /= sum;
    
    // 分配层
    layer_alloc_.clear();
    
    if (args.split_mode == SplitMode::NONE) {
        // 单 GPU 模式
        for (int i = 0; i < n_gpu_layers; ++i) {
            layer_alloc_.push_back({i, args.main_gpu, 0});
        }
    } else {
        // 多 GPU 模式：按比例分配层
        int layer_idx = 0;
        for (int gpu = 0; gpu < gpu_count && layer_idx < n_gpu_layers; ++gpu) {
            int layers_for_gpu = static_cast<int>(n_gpu_layers * split_ratios[gpu] + 0.5f);
            if (gpu == gpu_count - 1) {
                // 最后一个 GPU 拿剩余所有层
                layers_for_gpu = n_gpu_layers - layer_idx;
            }
            
            for (int j = 0; j < layers_for_gpu && layer_idx < n_gpu_layers; ++j, ++layer_idx) {
                layer_alloc_.push_back({layer_idx, gpu, 0});
            }
        }
    }
    
    // 打印分配结果
    std::vector<int> layers_per_gpu(gpu_count, 0);
    for (const auto& alloc : layer_alloc_) {
        layers_per_gpu[alloc.gpu_id]++;
    }
    
    LOG_INFO("层分配结果 (split_mode=%s):", split_mode_to_string(args.split_mode));
    for (int i = 0; i < gpu_count; ++i) {
        if (layers_per_gpu[i] > 0) {
            LOG_INFO("  GPU %d: %d 层 (比例: %.1f%%)", 
                     i, layers_per_gpu[i], split_ratios[i] * 100);
        }
    }
    
    return Error::OK;
}

Error Qwen3Model::load_tensors(const CliArgs& args) {
    // 这里是简化实现，实际需要：
    // 1. 根据 layer_alloc_ 确定每个 tensor 应该在哪个 GPU
    // 2. 分配 GPU 内存
    // 3. 从 mmap 数据拷贝到 GPU
    
    // 构建层->GPU 映射
    std::unordered_map<int, int> layer_gpu_map;
    for (const auto& alloc : layer_alloc_) {
        layer_gpu_map[alloc.layer_id] = alloc.gpu_id;
    }
    
    // 统计各 GPU 需要的内存
    int gpu_count = get_gpu_count();
    std::vector<size_t> gpu_memory_needed(gpu_count, 0);
    
    for (auto& tensor : tensors_) {
        int gpu_id = args.main_gpu;  // 默认 GPU
        
        // 解析 tensor 名，确定属于哪一层
        // 格式: blk.{N}.xxx
        if (tensor.name.find("blk.") != std::string::npos) {
            size_t pos = tensor.name.find("blk.") + 4;
            size_t end_pos = tensor.name.find(".", pos);
            int layer = std::stoi(tensor.name.substr(pos, end_pos - pos));
            
            auto it = layer_gpu_map.find(layer);
            if (it != layer_gpu_map.end()) {
                gpu_id = it->second;
            }
        }
        
        tensor.gpu_id = gpu_id;
        gpu_memory_needed[gpu_id] += tensor.size_bytes;
    }
    
    // 打印内存需求
    LOG_INFO("GPU 内存需求:");
    for (int i = 0; i < gpu_count; ++i) {
        if (gpu_memory_needed[i] > 0) {
            GpuInfo info = get_gpu_info(i);
            LOG_INFO("  GPU %d: 需要 %s / 可用 %s",
                     i, format_bytes(gpu_memory_needed[i]).c_str(),
                     format_bytes(info.free_memory).c_str());
            
            if (gpu_memory_needed[i] > info.free_memory) {
                LOG_WARN("  警告: GPU %d 显存可能不足", i);
            }
        }
    }
    
    // TODO: 实际的内存分配和数据拷贝
    // 这部分需要配合 ggml 库完成
    
    return Error::OK;
}

const TensorInfo* Qwen3Model::get_tensor(const std::string& name) const {
    auto it = tensor_map_.find(name);
    if (it == tensor_map_.end()) return nullptr;
    return &tensors_[it->second];
}

void Qwen3Model::print_info() const {
    LOG_INFO("========== Qwen3 模型信息 ==========");
    LOG_INFO("架构: %s", arch_name_.c_str());
    LOG_INFO("词表大小: %u", hparams_.n_vocab);
    LOG_INFO("嵌入维度: %u", hparams_.n_embd);
    LOG_INFO("层数: %u", hparams_.n_layer);
    LOG_INFO("注意力头: %u (KV: %u)", hparams_.n_head, hparams_.n_head_kv);
    LOG_INFO("FFN 维度: %u", hparams_.n_ff);
    LOG_INFO("RoPE 维度: %u", hparams_.n_rot);
    LOG_INFO("RoPE 基频: %.0f", hparams_.rope_freq_base);
    LOG_INFO("RMSNorm eps: %.2e", hparams_.rms_norm_eps);
    LOG_INFO("训练上下文: %u", hparams_.n_ctx_train);
    LOG_INFO("Tensor 数量: %zu", tensors_.size());
    LOG_INFO("=====================================");
}

} // namespace ember
