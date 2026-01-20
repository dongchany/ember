#pragma once

#include <cstdint>
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
// 从字符串解析数据类型
inline DType dtype_from_string(const std::string& s) {
    if (s == "float32" || s == "f32" || s == "F32") return DType::F32;
    if (s == "float16" || s == "f16" || s == "F16") return DType::F16;
    if (s == "bfloat16" || s == "bf16" || s == "BF16") return DType::BF16;
    if (s == "int8" || s == "INT8") return DType::INT8;
    if (s == "int4" || s == "INT4") return DType::INT4;
    return DType::UNKNOWN;
}
constexpr int DEVICE_CPU = -1;

}  // namespace ember
