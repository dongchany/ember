#pragma once

#include <cstdint>
#include <cstddef>
#include <string>

namespace ember {

// 数据类型枚举
enum class DType : uint8_t {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    INT8 = 3,
    INT4 = 4,
    UNKNOWN = 255
};

// 获取数据类型的字节大小
inline size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::F32:  return 4;
        case DType::F16:  return 2;
        case DType::BF16: return 2;
        case DType::INT8: return 1;
        case DType::INT4: return 1;  // 实际是 0.5，但按 1 处理
        default: return 0;
    }
}

// 数据类型名称
inline const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::F32:  return "f32";
        case DType::F16:  return "f16";
        case DType::BF16: return "bf16";
        case DType::INT8: return "int8";
        case DType::INT4: return "int4";
        default: return "unknown";
    }
}

// 从字符串解析数据类型
inline DType dtype_from_string(const std::string& s) {
    if (s == "float32" || s == "f32" || s == "F32") return DType::F32;
    if (s == "float16" || s == "f16" || s == "F16") return DType::F16;
    if (s == "bfloat16" || s == "bf16" || s == "BF16") return DType::BF16;
    if (s == "int8" || s == "INT8") return DType::INT8;
    if (s == "int4" || s == "INT4") return DType::INT4;
    return DType::UNKNOWN;
}

// 设备常量
constexpr int DEVICE_CPU = -1;

}  // namespace ember
