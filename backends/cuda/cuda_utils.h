#pragma once

#include <cuda_runtime.h>

#include "../../core/error.h"

namespace ember {
namespace cuda {

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            return Error::cuda_error(std::string(#call) +                    \
                                     " failed: " + cudaGetErrorString(err)); \
        }                                                                    \
    } while (0)                                                              \
    

struct GPUInfo {
    int device_id = 0;
    std::string name;
    size_t total_memory = 0;
    size_t free_memory = 0;
    int compute_capability_major = 0;
    int compute_capability_minor = 0;
    int multi_processor_count = 0;
    int max_threads_per_block = 0;

    void print() const {
        std::cout << "GPU " << device_id << ": " << name << std::endl;
        std::cout << "  Memory: " << free_memory / (1024 * 1024) << " / "
                  << total_memory / (1024 * 1024) << " MB free" << std::endl;
        std::cout << "  Compute: SM " << compute_capability_major << "."
                  << compute_capability_minor << std::endl;
        std::cout << "  SMs: " << multi_processor_count << std::endl;
    }
};

inline int get_device_count() {
    int count = 0;
    cudaGetDeviceCount(&count);
    return count;
}

inline Error get_gpu_info(int device_id, GPUInfo& info) {
    info.device_id = device_id;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));

    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    info.multi_processor_count = prop.multiProcessorCount;
    info.max_threads_per_block = prop.maxThreadsPerBlock;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    info.free_memory = free_mem;

    return Error::success();
}

inline GPUInfo get_gpu_info(int device_id) {
    GPUInfo info;
    get_gpu_info(device_id, info);
    return info;
}

}  // namespace cuda
}  // namespace ember