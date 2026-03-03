# Tutorial #7 Prompt (Full, With Code + Reports)

> 生成日期：2026-02-26
> 用途：整份复制到新 chat 作为写作输入
> 标注：大文件（如 cuda_runtime.cpp）按“该篇相关段落”摘取，避免上下文爆炸

---

## 0) 系统 Prompt（先贴这段）

```text
我正在为我的开源项目 Ember 写一个系列教程。请帮我写第 7 篇。

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
请写第 7 篇：Safetensors 加载与权重映射。

## 本篇必须引用的代码与报告都在本消息里（不要再让我补充路径）

- 重点讲“磁盘 tensor 名称 -> 运行时权重指针”的映射链路
- 提醒常见 shape mismatch / dtype mismatch
```

---

## 2) 代码上下文（完整/相关段落）

### File: formats/safetensors.h

````h
#pragma once

#include "../core/error.h"
#include "../core/tensor.h"
#include "../core/types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstdint>
#include <memory>

namespace ember {

// Safetensors 文件中的张量元数据
struct SafetensorsMeta {
    std::string name;
    DType dtype;
    std::vector<int64_t> shape;
    size_t data_offset;  // 在文件中的偏移（相对于数据区起始）
    size_t data_size;    // 字节大小
};

// Safetensors 文件读取器
class SafetensorsReader {
public:
    SafetensorsReader() = default;
    ~SafetensorsReader();
    
    // 打开文件并解析头部
    Error open(const std::string& path);
    
    // 关闭文件
    void close();
    
    // 获取所有张量的元数据
    const std::unordered_map<std::string, SafetensorsMeta>& tensors() const { 
        return tensors_; 
    }
    
    // 检查张量是否存在
    bool has_tensor(const std::string& name) const {
        return tensors_.count(name) > 0;
    }
    
    // 获取张量元数据
    const SafetensorsMeta* get_meta(const std::string& name) const {
        auto it = tensors_.find(name);
        return (it != tensors_.end()) ? &it->second : nullptr;
    }
    
    // 读取张量数据到 CPU 内存
    Error read_tensor(const std::string& name, void* dst, size_t dst_size);
    
    // 读取张量数据并返回 Tensor（会分配内存）
    Error read_tensor(const std::string& name, Tensor& out);
    
    // 获取原始数据指针（如果使用 mmap）
    const void* get_data_ptr(const std::string& name) const;
    
    // 获取文件总大小
    size_t file_size() const { return file_size_; }
    
    // 获取数据区大小
    size_t data_size() const { return data_size_; }
    
    // 列出所有张量名称
    std::vector<std::string> tensor_names() const;

private:
    std::string path_;
    std::ifstream file_;
    size_t file_size_ = 0;
    size_t header_size_ = 0;
    size_t data_offset_ = 0;
    size_t data_size_ = 0;
    
    std::unordered_map<std::string, SafetensorsMeta> tensors_;
    
    // mmap 支持（可选）
    void* mmap_ptr_ = nullptr;
    size_t mmap_size_ = 0;
    
    Error parse_header();
};

// 模型权重加载器（处理多文件 sharding）
class ModelWeightLoader {
public:
    // 打开模型目录
    Error open(const std::string& model_dir);
    
    // 获取所有张量名称
    std::vector<std::string> tensor_names() const;
    
    // 检查张量是否存在
    bool has_tensor(const std::string& name) const;
    
    // 获取张量元数据
    const SafetensorsMeta* get_meta(const std::string& name) const;
    
    // 读取张量数据
    Error read_tensor(const std::string& name, void* dst, size_t dst_size);
    Error read_tensor(const std::string& name, Tensor& out);
    
    // 关闭
    void close();

private:
    std::string model_dir_;
    std::vector<std::unique_ptr<SafetensorsReader>> readers_;
    std::unordered_map<std::string, int> tensor_to_file_;  // 张量名 -> 文件索引
};

// Safetensors 数据类型转换
inline DType safetensors_dtype_to_ember(const std::string& dtype_str) {
    if (dtype_str == "F32") return DType::F32;
    if (dtype_str == "F16") return DType::F16;
    if (dtype_str == "BF16") return DType::BF16;
    if (dtype_str == "I8") return DType::INT8;
    return DType::UNKNOWN;
}

}  // namespace ember

````

### File: formats/safetensors.cpp

````cpp
#include "safetensors.h"
#include <fstream>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <iostream>

// 简单的 JSON 解析（只支持 safetensors header 格式）
// 注意：这是一个简化实现，生产环境建议使用 nlohmann/json 等库

namespace ember {

namespace {

// 跳过空白字符
void skip_whitespace(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
        ++p;
    }
}

// 解析字符串（带引号）
bool parse_string(const char*& p, const char* end, std::string& out) {
    skip_whitespace(p, end);
    if (p >= end || *p != '"') return false;
    ++p;  // 跳过开始引号
    
    out.clear();
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) {
            ++p;
            switch (*p) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '\\': out += '\\'; break;
                case '"': out += '"'; break;
                default: out += *p; break;
            }
        } else {
            out += *p;
        }
        ++p;
    }
    
    if (p >= end) return false;
    ++p;  // 跳过结束引号
    return true;
}

// 解析数字
bool parse_number(const char*& p, const char* end, int64_t& out) {
    skip_whitespace(p, end);
    if (p >= end) return false;
    
    bool negative = false;
    if (*p == '-') {
        negative = true;
        ++p;
    }
    
    if (p >= end || !std::isdigit(*p)) return false;
    
    out = 0;
    while (p < end && std::isdigit(*p)) {
        out = out * 10 + (*p - '0');
        ++p;
    }
    
    if (negative) out = -out;
    return true;
}

// 跳过到指定字符
bool skip_to(const char*& p, const char* end, char c) {
    skip_whitespace(p, end);
    if (p >= end || *p != c) return false;
    ++p;
    return true;
}

}  // namespace

SafetensorsReader::~SafetensorsReader() {
    close();
}

void SafetensorsReader::close() {
    if (file_.is_open()) {
        file_.close();
    }
    tensors_.clear();
    
    // TODO: 如果使用了 mmap，需要 munmap
}

Error SafetensorsReader::open(const std::string& path) {
    path_ = path;
    
    // 打开文件
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) {
        return Error::file_not_found(path);
    }
    
    // 获取文件大小
    file_.seekg(0, std::ios::end);
    file_size_ = file_.tellg();
    file_.seekg(0, std::ios::beg);
    
    if (file_size_ < 8) {
        return Error(ErrorCode::INVALID_FORMAT, "File too small");
    }
    
    // 解析头部
    return parse_header();
}

Error SafetensorsReader::parse_header() {
    // 读取头部大小（8 字节小端序 uint64）
    uint64_t header_len;
    file_.read(reinterpret_cast<char*>(&header_len), 8);
    if (!file_) {
        return Error(ErrorCode::FILE_READ_ERROR, "Failed to read header size");
    }
    
    header_size_ = header_len;
    data_offset_ = 8 + header_len;
    data_size_ = file_size_ - data_offset_;
    
    if (header_len > 100 * 1024 * 1024) {  // 100MB 限制
        return Error(ErrorCode::INVALID_FORMAT, "Header too large");
    }
    
    // 读取 JSON 头部
    std::string header(header_len, '\0');
    file_.read(&header[0], header_len);
    if (!file_) {
        return Error(ErrorCode::FILE_READ_ERROR, "Failed to read header");
    }
    
    // 解析 JSON
    const char* p = header.c_str();
    const char* end = p + header.size();
    
    skip_whitespace(p, end);
    if (p >= end || *p != '{') {
        return Error(ErrorCode::INVALID_FORMAT, "Expected '{' at start of header");
    }
    p++;  // 消耗 {
    
    // 解析每个张量条目
    while (p < end) {
        skip_whitespace(p, end);
        if (*p == '}') break;
        
        // 解析张量名
        std::string name;
        if (!parse_string(p, end, name)) {
            return Error(ErrorCode::INVALID_FORMAT, "Expected tensor name");
        }
        
        // 跳过 ':'
        if (!skip_to(p, end, ':')) {
            return Error(ErrorCode::INVALID_FORMAT, "Expected ':'");
        }
        
        // 跳过 '__metadata__' 条目
        if (name == "__metadata__") {
            // 跳过整个对象
            skip_whitespace(p, end);
            if (p < end && *p == '{') {
                int depth = 1;
                p++;  // 消耗开始的 {
                bool in_string = false;
                while (p < end && depth > 0) {
                    if (!in_string) {
                        if (*p == '{') depth++;
                        else if (*p == '}') depth--;
                        else if (*p == '"') in_string = true;
                    } else {
                        if (*p == '\\' && p + 1 < end) p++;
                        else if (*p == '"') in_string = false;
                    }
                    p++;
                }
            }
            skip_whitespace(p, end);
            if (p < end && *p == ',') p++;
            continue;
        }
        
        SafetensorsMeta meta;
        meta.name = name;
        
        // 解析张量元数据对象
        if (!skip_to(p, end, '{')) {
            return Error(ErrorCode::INVALID_FORMAT, "Expected tensor metadata object");
        }
        
        while (p < end && *p != '}') {
            skip_whitespace(p, end);
            
            std::string key;
            if (!parse_string(p, end, key)) break;
            
            if (!skip_to(p, end, ':')) {
                return Error(ErrorCode::INVALID_FORMAT, "Expected ':'");
            }
            
            if (key == "dtype") {
                std::string dtype_str;
                if (!parse_string(p, end, dtype_str)) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected dtype string");
                }
                meta.dtype = safetensors_dtype_to_ember(dtype_str);
            } else if (key == "shape") {
                if (!skip_to(p, end, '[')) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected shape array");
                }
                
                while (p < end && *p != ']') {
                    skip_whitespace(p, end);
                    if (*p == ']') break;
                    
                    int64_t dim;
                    if (!parse_number(p, end, dim)) break;
                    meta.shape.push_back(dim);
                    
                    skip_whitespace(p, end);
                    if (*p == ',') p++;
                }
                
                if (!skip_to(p, end, ']')) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected ']'");
                }
            } else if (key == "data_offsets") {
                if (!skip_to(p, end, '[')) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected data_offsets array");
                }
                
                int64_t start_offset, end_offset;
                if (!parse_number(p, end, start_offset)) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected start offset");
                }
                
                skip_whitespace(p, end);
                if (!skip_to(p, end, ',')) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected ','");
                }
                
                if (!parse_number(p, end, end_offset)) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected end offset");
                }
                
                meta.data_offset = start_offset;
                meta.data_size = end_offset - start_offset;
                
                if (!skip_to(p, end, ']')) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected ']'");
                }
            }
            
            skip_whitespace(p, end);
            if (*p == ',') p++;
        }
        
        if (!skip_to(p, end, '}')) {
            return Error(ErrorCode::INVALID_FORMAT, "Expected '}'");
        }
        
        tensors_[name] = std::move(meta);
        
        skip_whitespace(p, end);
        if (*p == ',') p++;
    }
    
    return Error::success();
}

Error SafetensorsReader::read_tensor(const std::string& name, void* dst, size_t dst_size) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Tensor not found: " + name);
    }
    
    const auto& meta = it->second;
    if (dst_size < meta.data_size) {
        return Error(ErrorCode::INVALID_ARGUMENT, "Buffer too small");
    }
    
    file_.seekg(data_offset_ + meta.data_offset);
    file_.read(static_cast<char*>(dst), meta.data_size);
    
    if (!file_) {
        return Error(ErrorCode::FILE_READ_ERROR, "Failed to read tensor data");
    }
    
    return Error::success();
}

Error SafetensorsReader::read_tensor(const std::string& name, Tensor& out) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Tensor not found: " + name);
    }
    
    const auto& meta = it->second;
    
    // 分配内存
    void* data = malloc(meta.data_size);
    if (!data) {
        return Error::out_of_memory("Failed to allocate tensor memory");
    }
    
    // 读取数据
    Error err = read_tensor(name, data, meta.data_size);
    if (err) {
        free(data);
        return err;
    }
    
    out.shape = meta.shape;
    out.dtype = meta.dtype;
    out.data = data;
    out.device_id = DEVICE_CPU;
    
    return Error::success();
}

const void* SafetensorsReader::get_data_ptr(const std::string& name) const {
    // 如果使用 mmap，返回映射地址
    if (mmap_ptr_) {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            return static_cast<const char*>(mmap_ptr_) + data_offset_ + it->second.data_offset;
        }
    }
    return nullptr;
}

std::vector<std::string> SafetensorsReader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        names.push_back(name);
    }
    return names;
}

// ModelWeightLoader 实现

Error ModelWeightLoader::open(const std::string& model_dir) {
    model_dir_ = model_dir;
    namespace fs = std::filesystem;
    
    if (!fs::exists(model_dir)) {
        return Error::file_not_found(model_dir);
    }
    
    // 查找所有 .safetensors 文件
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".safetensors") {
            files.push_back(entry.path().string());
        }
    }
    
    if (files.empty()) {
        return Error(ErrorCode::FILE_NOT_FOUND, "No safetensors files found in " + model_dir);
    }
    
    // 排序以保证顺序一致
    std::sort(files.begin(), files.end());
    
    // 打开每个文件
    for (size_t i = 0; i < files.size(); ++i) {
        auto reader = std::make_unique<SafetensorsReader>();
        Error err = reader->open(files[i]);
        if (err) {
            return err;
        }
        
        // 记录每个张量在哪个文件
        for (const auto& name : reader->tensor_names()) {
            tensor_to_file_[name] = static_cast<int>(i);
        }
        
        readers_.push_back(std::move(reader));
    }
    
    return Error::success();
}

void ModelWeightLoader::close() {
    readers_.clear();
    tensor_to_file_.clear();
}

std::vector<std::string> ModelWeightLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensor_to_file_.size());
    for (const auto& [name, _] : tensor_to_file_) {
        names.push_back(name);
    }
    return names;
}

bool ModelWeightLoader::has_tensor(const std::string& name) const {
    return tensor_to_file_.count(name) > 0;
}

const SafetensorsMeta* ModelWeightLoader::get_meta(const std::string& name) const {
    auto it = tensor_to_file_.find(name);
    if (it == tensor_to_file_.end()) return nullptr;
    return readers_[it->second]->get_meta(name);
}

Error ModelWeightLoader::read_tensor(const std::string& name, void* dst, size_t dst_size) {
    auto it = tensor_to_file_.find(name);
    if (it == tensor_to_file_.end()) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Tensor not found: " + name);
    }
    return readers_[it->second]->read_tensor(name, dst, dst_size);
}

Error ModelWeightLoader::read_tensor(const std::string& name, Tensor& out) {
    auto it = tensor_to_file_.find(name);
    if (it == tensor_to_file_.end()) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Tensor not found: " + name);
    }
    return readers_[it->second]->read_tensor(name, out);
}

}  // namespace ember

````

### File: backends/cuda/cuda_runtime.cpp (load_weights (top))

````cpp
    for (int i = 0; i < num_devices; ++i) {
        CUDA_CHECK(cudaSetDevice(i));
        CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        CUDA_CHECK(cudaStreamCreate(&transfer_streams_[i]));
        Error err = cublas_handles_[i].create(i);
        if (err) return err;
        CUDA_CHECK(cudaEventCreate(&profile_events_[i].start));
        CUDA_CHECK(cudaEventCreate(&profile_events_[i].end));
    }

    try_enable_peer_access_all_pairs(num_devices);
    
    // 加载权重
    Error err = load_weights(model_path);
    if (err) {
        unload();
        return err;
    }
    
    loaded_ = true;
    std::cout << "[CudaRuntime] Model loaded successfully" << std::endl;
    
    // 打印显存使用
    for (int i = 0; i < num_devices; ++i) {
        print_memory_usage(i);
    }
    
    return Error::success();
}

Error CudaRuntime::load_weights(const std::string& model_path) {
    ModelWeightLoader loader;
    Error err = loader.open(model_path);
    if (err) return err;
    
    // 检查是否有必要的权重
    if (!loader.has_tensor("model.embed_tokens.weight")) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Missing embedding weights");
    }
    auto embed_meta = loader.get_meta("model.embed_tokens.weight");
    if (!embed_meta) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Missing embedding weights");
    }
    
    if (embed_meta->dtype == DType::BF16 || embed_meta->dtype == DType::F16) {
        weights_.dtype = embed_meta->dtype;
    } else {
        weights_.dtype = DType::F16;
    }
    
    int device_id = device_map_.embedding_device;
    weights_.embed_device_id = device_id;
    
    // 加载 embedding
    {
        const std::string name = "model.embed_tokens.weight";
        auto meta = loader.get_meta(name);
        size_t size = meta ? meta->data_size : 0;
        err = load_tensor_to_device(loader, name, device_id, weights_.dtype, &weights_.embed_tokens);
        if (err) return err;
        
        std::cout << "[CudaRuntime] Loaded embed_tokens: " << format_bytes(size) 
                  << " -> GPU " << device_id << std::endl;
    }
    
    // 加载每层权重
    weights_.layers.resize(config_.num_layers);
    
    for (int layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        int layer_device = device_map_.layer_to_device[layer_idx];
        err = load_layer_weights(layer_idx, loader, layer_device);
        if (err) return err;
        
        if ((layer_idx + 1) % 10 == 0 || layer_idx == config_.num_layers - 1) {
            std::cout << "[CudaRuntime] Loaded layers 0-" << layer_idx << std::endl;
        }
    }
    
    // 加载 final norm
    {
        const std::string name = "model.norm.weight";
        int lm_device = device_map_.lm_head_device;
        err = load_tensor_to_device(loader, name, lm_device, weights_.dtype, &weights_.final_norm);
        if (err) return err;
        weights_.final_norm_device_id = lm_device;
    }
    
    // 加载 lm_head（如果不与 embedding 共享）
    if (loader.has_tensor("lm_head.weight")) {
        const std::string name = "lm_head.weight";
        int lm_device = device_map_.lm_head_device;
        err = load_tensor_to_device(loader, name, lm_device, weights_.dtype, &weights_.lm_head);
        if (err) return err;
        weights_.lm_head_device_id = lm_device;
        weights_.lm_head_owns_allocation = true;
        std::cout << "[CudaRuntime] Loaded separate lm_head" << std::endl;
    } else {
        // 共享 embedding（若 lm_head 在不同 GPU，需要复制一份 embedding 权重到 lm_head GPU）
        int lm_device = device_map_.lm_head_device;
        weights_.lm_head_device_id = device_id;
        weights_.lm_head_owns_allocation = false;
        weights_.lm_head = weights_.embed_tokens;

        if (lm_device != device_id) {
            const size_t bytes = static_cast<size_t>(config_.vocab_size) *
                                 static_cast<size_t>(config_.hidden_size) *
                                 dtype_size(weights_.dtype);
            void* lm_copy = nullptr;
            err = cuda::cuda_malloc(&lm_copy, bytes, lm_device);
            if (err) return err;
            err = copy_bytes_peer_or_staged(lm_copy, lm_device, weights_.embed_tokens, device_id, bytes);
            if (err) return err;
            weights_.lm_head = lm_copy;
            weights_.lm_head_device_id = lm_device;
            weights_.lm_head_owns_allocation = true;
            std::cout << "[CudaRuntime] Copied tied lm_head to GPU " << lm_device << std::endl;
        } else {
            std::cout << "[CudaRuntime] Using tied embeddings for lm_head" << std::endl;
        }
    }
    
    return Error::success();
}

Error CudaRuntime::load_layer_weights(int layer_idx, ModelWeightLoader& loader, int device_id) {
    auto& layer = weights_.layers[layer_idx];
    layer.device_id = device_id;
    
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    auto load_weight = [&](const std::string& name, void** ptr) -> Error {
        return load_tensor_to_device(loader, prefix + name, device_id, weights_.dtype, ptr);
    };
    
    // Self Attention
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.q_proj.weight", &layer.q_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.k_proj.weight", &layer.k_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.v_proj.weight", &layer.v_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.o_proj.weight", &layer.o_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.q_norm.weight", &layer.q_norm_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.k_norm.weight", &layer.k_norm_weight));
    
    // MLP gate/up packed contiguously: [2 * intermediate_size, hidden_size]
    // This enables decode fast path with one strided-batched GEMM for gate+up.
    const std::string gate_name = prefix + "mlp.gate_proj.weight";
    const std::string up_name = prefix + "mlp.up_proj.weight";
    const SafetensorsMeta* gate_meta = loader.get_meta(gate_name);
    const SafetensorsMeta* up_meta = loader.get_meta(up_name);
    if (!gate_meta || !up_meta) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND,
                     "Missing MLP gate/up weight at layer " + std::to_string(layer_idx));
    }
    if (gate_meta->shape != up_meta->shape) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "MLP gate/up shape mismatch at layer " + std::to_string(layer_idx));
    }

    const size_t gate_elem_size = dtype_size(gate_meta->dtype);
    const size_t up_elem_size = dtype_size(up_meta->dtype);
    if (gate_elem_size == 0 || up_elem_size == 0) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "Invalid MLP gate/up dtype at layer " + std::to_string(layer_idx));
    }
    if (gate_meta->data_size % gate_elem_size != 0 || up_meta->data_size % up_elem_size != 0) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "Invalid MLP gate/up tensor size at layer " + std::to_string(layer_idx));
    }

    const size_t gate_count = gate_meta->data_size / gate_elem_size;
    const size_t up_count = up_meta->data_size / up_elem_size;
    if (gate_count != up_count) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "MLP gate/up element count mismatch at layer " + std::to_string(layer_idx));
    }

    const size_t target_elem_size = dtype_size(weights_.dtype);
    const size_t gate_bytes = gate_count * target_elem_size;
    const size_t up_bytes = up_count * target_elem_size;
    EMBER_RETURN_IF_ERROR(cuda::cuda_malloc(&layer.gate_up_proj_weight, gate_bytes + up_bytes, device_id));

    void* gate_tmp = nullptr;
    Error err = load_tensor_to_device(loader, gate_name, device_id, weights_.dtype, &gate_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }
    err = cuda::cuda_memcpy_d2d(layer.gate_up_proj_weight, gate_tmp, gate_bytes, device_id);
    cuda_free(gate_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }

    void* up_tmp = nullptr;
    err = load_tensor_to_device(loader, up_name, device_id, weights_.dtype, &up_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }
    void* up_dst = static_cast<void*>(static_cast<char*>(layer.gate_up_proj_weight) + gate_bytes);
    err = cuda::cuda_memcpy_d2d(up_dst, up_tmp, up_bytes, device_id);
    cuda_free(up_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }

    layer.gate_proj_weight = layer.gate_up_proj_weight;
    layer.up_proj_weight = up_dst;
    layer.gate_up_proj_packed = true;
    EMBER_RETURN_IF_ERROR(load_weight("mlp.down_proj.weight", &layer.down_proj_weight));
    
    // LayerNorms
    EMBER_RETURN_IF_ERROR(load_weight("input_layernorm.weight", &layer.input_layernorm_weight));
    EMBER_RETURN_IF_ERROR(load_weight("post_attention_layernorm.weight", &layer.post_attention_layernorm_weight));
    
    layer.allocated = true;
    return Error::success();
}

Error CudaRuntime::apply_lora_adapter(const std::string& adapter_dir,
                                      float scale,
                                      bool replace_existing,
                                      LoraApplyStats* stats) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    if (adapter_dir.empty()) {
        return Error::invalid_argument("adapter_dir is empty");
    }
    if (weights_.dtype != DType::F16 && weights_.dtype != DType::BF16) {
        return Error(ErrorCode::INVALID_FORMAT, "LoRA apply supports F16/BF16 weights only");
    }

    auto apply_one = [&](const std::string& one_adapter_dir,
                         float one_scale,
                         bool print_log,
                         LoraApplyStats* out_stats) -> Error {
        LoraApplyStats local_stats{};
        auto t0 = std::chrono::high_resolution_clock::now();

        ModelWeightLoader loader;
        Error err = loader.open(one_adapter_dir);
        if (err) return err;

        struct ABPair {
            std::string a_name;

````

### File: backends/cuda/cuda_runtime.cpp (load_layer_weights)

````cpp
            weights_.lm_head_owns_allocation = true;
            std::cout << "[CudaRuntime] Copied tied lm_head to GPU " << lm_device << std::endl;
        } else {
            std::cout << "[CudaRuntime] Using tied embeddings for lm_head" << std::endl;
        }
    }
    
    return Error::success();
}

Error CudaRuntime::load_layer_weights(int layer_idx, ModelWeightLoader& loader, int device_id) {
    auto& layer = weights_.layers[layer_idx];
    layer.device_id = device_id;
    
    std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
    
    auto load_weight = [&](const std::string& name, void** ptr) -> Error {
        return load_tensor_to_device(loader, prefix + name, device_id, weights_.dtype, ptr);
    };
    
    // Self Attention
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.q_proj.weight", &layer.q_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.k_proj.weight", &layer.k_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.v_proj.weight", &layer.v_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.o_proj.weight", &layer.o_proj_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.q_norm.weight", &layer.q_norm_weight));
    EMBER_RETURN_IF_ERROR(load_weight("self_attn.k_norm.weight", &layer.k_norm_weight));
    
    // MLP gate/up packed contiguously: [2 * intermediate_size, hidden_size]
    // This enables decode fast path with one strided-batched GEMM for gate+up.
    const std::string gate_name = prefix + "mlp.gate_proj.weight";
    const std::string up_name = prefix + "mlp.up_proj.weight";
    const SafetensorsMeta* gate_meta = loader.get_meta(gate_name);
    const SafetensorsMeta* up_meta = loader.get_meta(up_name);
    if (!gate_meta || !up_meta) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND,
                     "Missing MLP gate/up weight at layer " + std::to_string(layer_idx));
    }
    if (gate_meta->shape != up_meta->shape) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "MLP gate/up shape mismatch at layer " + std::to_string(layer_idx));
    }

    const size_t gate_elem_size = dtype_size(gate_meta->dtype);
    const size_t up_elem_size = dtype_size(up_meta->dtype);
    if (gate_elem_size == 0 || up_elem_size == 0) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "Invalid MLP gate/up dtype at layer " + std::to_string(layer_idx));
    }
    if (gate_meta->data_size % gate_elem_size != 0 || up_meta->data_size % up_elem_size != 0) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "Invalid MLP gate/up tensor size at layer " + std::to_string(layer_idx));
    }

    const size_t gate_count = gate_meta->data_size / gate_elem_size;
    const size_t up_count = up_meta->data_size / up_elem_size;
    if (gate_count != up_count) {
        return Error(ErrorCode::INVALID_FORMAT,
                     "MLP gate/up element count mismatch at layer " + std::to_string(layer_idx));
    }

    const size_t target_elem_size = dtype_size(weights_.dtype);
    const size_t gate_bytes = gate_count * target_elem_size;
    const size_t up_bytes = up_count * target_elem_size;
    EMBER_RETURN_IF_ERROR(cuda::cuda_malloc(&layer.gate_up_proj_weight, gate_bytes + up_bytes, device_id));

    void* gate_tmp = nullptr;
    Error err = load_tensor_to_device(loader, gate_name, device_id, weights_.dtype, &gate_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }
    err = cuda::cuda_memcpy_d2d(layer.gate_up_proj_weight, gate_tmp, gate_bytes, device_id);
    cuda_free(gate_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }

    void* up_tmp = nullptr;
    err = load_tensor_to_device(loader, up_name, device_id, weights_.dtype, &up_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }
    void* up_dst = static_cast<void*>(static_cast<char*>(layer.gate_up_proj_weight) + gate_bytes);
    err = cuda::cuda_memcpy_d2d(up_dst, up_tmp, up_bytes, device_id);
    cuda_free(up_tmp);
    if (err) {
        cuda_free(layer.gate_up_proj_weight);
        layer.gate_up_proj_weight = nullptr;
        return err;
    }

    layer.gate_proj_weight = layer.gate_up_proj_weight;
    layer.up_proj_weight = up_dst;
    layer.gate_up_proj_packed = true;
    EMBER_RETURN_IF_ERROR(load_weight("mlp.down_proj.weight", &layer.down_proj_weight));
    
    // LayerNorms
    EMBER_RETURN_IF_ERROR(load_weight("input_layernorm.weight", &layer.input_layernorm_weight));
    EMBER_RETURN_IF_ERROR(load_weight("post_attention_layernorm.weight", &layer.post_attention_layernorm_weight));
    
    layer.allocated = true;
    return Error::success();
}

Error CudaRuntime::apply_lora_adapter(const std::string& adapter_dir,
                                      float scale,
                                      bool replace_existing,
                                      LoraApplyStats* stats) {
    if (!loaded_) {
        return Error(ErrorCode::MODEL_NOT_LOADED, "Model not loaded");
    }
    if (adapter_dir.empty()) {
        return Error::invalid_argument("adapter_dir is empty");
    }
    if (weights_.dtype != DType::F16 && weights_.dtype != DType::BF16) {
        return Error(ErrorCode::INVALID_FORMAT, "LoRA apply supports F16/BF16 weights only");
    }

    auto apply_one = [&](const std::string& one_adapter_dir,
                         float one_scale,
                         bool print_log,
                         LoraApplyStats* out_stats) -> Error {
        LoraApplyStats local_stats{};
        auto t0 = std::chrono::high_resolution_clock::now();

        ModelWeightLoader loader;
        Error err = loader.open(one_adapter_dir);
        if (err) return err;

        struct ABPair {
            std::string a_name;
            std::string b_name;
        };
        std::unordered_map<std::string, ABPair> pairs;
        const std::vector<std::string> names = loader.tensor_names();
        for (const std::string& name : names) {
            LoraTargetKey key;
            if (!parse_lora_target_key(name, key)) continue;
            const std::string pair_key = std::to_string(key.layer_idx) + ":" + key.proj;
            auto& p = pairs[pair_key];
            if (key.is_a) {
                p.a_name = name;
            } else {
                p.b_name = name;
            }
        }
        if (pairs.empty()) {
            return Error(ErrorCode::WEIGHT_NOT_FOUND,
                         "No supported LoRA tensors found under " + one_adapter_dir);
        }

        const float alpha_over_r = read_lora_alpha_over_r(one_adapter_dir);
        const float effective_scale = one_scale * alpha_over_r;
        local_stats.scale_used = effective_scale;

        const cudaDataType_t cuda_dtype = to_cuda_dtype(weights_.dtype);

        auto pick_target = [&](int layer_idx, const std::string& proj, void** weight_ptr,
                               int* out_dim, int* in_dim, int* device_id) -> Error {
            if (layer_idx < 0 || layer_idx >= config_.num_layers) {
                return Error(ErrorCode::INVALID_ARGUMENT,
                             "LoRA layer index out of range: " + std::to_string(layer_idx));
            }
            auto& layer = weights_.layers[static_cast<size_t>(layer_idx)];
            *device_id = layer.device_id;
            if (proj == "q_proj") {
                *weight_ptr = layer.q_proj_weight;
                *out_dim = config_.num_heads * config_.head_dim;
                *in_dim = config_.hidden_size;
            } else if (proj == "k_proj") {
                *weight_ptr = layer.k_proj_weight;
                *out_dim = config_.num_kv_heads * config_.head_dim;
                *in_dim = config_.hidden_size;
            } else if (proj == "v_proj") {
                *weight_ptr = layer.v_proj_weight;
                *out_dim = config_.num_kv_heads * config_.head_dim;
                *in_dim = config_.hidden_size;
            } else if (proj == "o_proj") {
                *weight_ptr = layer.o_proj_weight;
                *out_dim = config_.hidden_size;
                *in_dim = config_.num_heads * config_.head_dim;
            } else {
                return Error(ErrorCode::INVALID_ARGUMENT, "Unsupported LoRA target: " + proj);
            }
            if (*weight_ptr == nullptr) {
                return Error(ErrorCode::WEIGHT_NOT_FOUND,
                             "Target weight not allocated for layer " + std::to_string(layer_idx) +
                             " proj " + proj);
            }
            return Error::success();
        };

        for (const auto& kv : pairs) {
            const ABPair& p = kv.second;
            if (p.a_name.empty() || p.b_name.empty()) {
                local_stats.skipped_matrices++;
                continue;
            }

            LoraTargetKey key{};
            if (!parse_lora_target_key(p.a_name, key)) {
                local_stats.skipped_matrices++;
                continue;
            }


````

---

## 3) 报告上下文（完整）

### Report: reports/stage1_milestone_4b_20260225_mainline/stage1_mainline_ready.md

````md
# Stage 1.1 Mainline Readiness

- usable summary rows: 24
- failed combos: 0
- oom combos: 0

## Missing Combos
- none

## Recommendation
- Use current 8B results for Stage 1.1 mainline plots/tables.
- Full grid coverage achieved for current Stage 1.1 matrix.

````

### Report: reports/stage31_lora_hot_update_4b_20260225_mainline/stage31_summary.md

````md
# Stage 3.1 LoRA Hot Update

- Model: `/home/dong/xilinx/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554`
- Adapter: `/home/dong/workspace/ember/reports/synthetic_lora_qwen3_4b_r8`
- Generated at: `2026-02-25T13:19:30`

| metric | value |
| --- | --- |
| gpus | `0+1` |
| split | `9+27` |
| scale | `1.000` |
| replace_existing | `0` |
| effective_scale | `2.000` |
| updated_matrices | `144` |
| skipped_matrices | `0` |
| apply_ms_ext | `353.980` |
| apply_ms_inner | `353.876` |

## Key Point
- Attention q/k/v/o matrices can be merged in-place from PEFT LoRA adapter without reloading base model weights.

````

---

## 4) 风格参考

附件上传：`tutorial_01_life_of_a_token.md`
