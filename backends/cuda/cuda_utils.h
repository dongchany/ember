#pragma once

#include <cuda_runtime.h>

namespace ember {
namespace cuda {

    inline int get_device_count() {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
    }

}
}  // namespace ember