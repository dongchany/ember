#pragma once


#include "types.h"
#include <vector>

namespace ember {
struct Tensor {
    std::vector<int64_t> shape;
    DType dtype = DType::F32;
    void* data = nullptr;
    int device_id = DEVICE_CPU;  // -1 = CPU, 0+ = GPU

    Tensor() = default;
};

}  // namespace ember
