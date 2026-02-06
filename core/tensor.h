#pragma once

#include "types.h"
#include <vector>
#include <cstdint>
#include <cassert>
#include <numeric>
#include <string>
#include <sstream>

namespace ember {

// 轻量级 Tensor 结构
// 不管理内存生命周期，只是一个视图
struct Tensor {
    std::vector<int64_t> shape;
    DType dtype = DType::F32;
    void* data = nullptr;
    int device_id = DEVICE_CPU;  // -1 = CPU, 0+ = GPU
    
    // 默认构造
    Tensor() = default;
    
    // 完整构造
    Tensor(std::vector<int64_t> shape_, DType dtype_, void* data_, int device_ = DEVICE_CPU)
        : shape(std::move(shape_)), dtype(dtype_), data(data_), device_id(device_) {}
    
    // 元素数量
    size_t numel() const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 
                               int64_t(1), std::multiplies<int64_t>());
    }
    
    // 字节大小
    size_t size_bytes() const {
        return numel() * dtype_size(dtype);
    }
    
    // 维度数
    size_t ndim() const { return shape.size(); }
    
    // 是否为空
    bool empty() const { return data == nullptr || numel() == 0; }
    
    // 是否在 CPU
    bool is_cpu() const { return device_id == DEVICE_CPU; }
    
    // 是否在 GPU
    bool is_cuda() const { return device_id >= 0; }
    
    // 获取指定维度大小
    int64_t dim(int i) const {
        if (i < 0) i += static_cast<int>(shape.size());
        assert(i >= 0 && i < static_cast<int>(shape.size()));
        return shape[i];
    }
    
    // 形状字符串
    std::string shape_str() const {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << shape[i];
        }
        oss << "]";
        return oss.str();
    }
    
    // 完整信息字符串
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Tensor(" << shape_str() << ", " << dtype_name(dtype);
        if (is_cuda()) {
            oss << ", cuda:" << device_id;
        } else {
            oss << ", cpu";
        }
        oss << ")";
        return oss.str();
    }
    
    // 类型转换访问（仅 CPU）
    template<typename T>
    T* data_ptr() {
        assert(is_cpu() && "data_ptr() only works for CPU tensors");
        return static_cast<T*>(data);
    }
    
    template<typename T>
    const T* data_ptr() const {
        assert(is_cpu() && "data_ptr() only works for CPU tensors");
        return static_cast<const T*>(data);
    }
};

// 计算 strides（row-major）
inline std::vector<int64_t> compute_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    if (shape.empty()) return strides;
    
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

// 检查两个形状是否匹配
inline bool shapes_match(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    return a == b;
}

// 广播形状计算（简化版，只支持末尾对齐）
inline bool shapes_broadcastable(const std::vector<int64_t>& a, 
                                  const std::vector<int64_t>& b) {
    size_t na = a.size(), nb = b.size();
    size_t max_dim = std::max(na, nb);
    
    for (size_t i = 0; i < max_dim; ++i) {
        int64_t da = (i < na) ? a[na - 1 - i] : 1;
        int64_t db = (i < nb) ? b[nb - 1 - i] : 1;
        if (da != db && da != 1 && db != 1) {
            return false;
        }
    }
    return true;
}

}  // namespace ember
