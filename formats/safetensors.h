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
