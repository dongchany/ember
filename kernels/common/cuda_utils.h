#pragma once

#include "core/error.h"
#include "core/types.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <iostream>
#include <string>
#include <vector>

namespace ember {
namespace cuda {

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                    \
    do {                                                                     \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            return Error::cuda_error(std::string(#call) + " failed: " +     \
                                     cudaGetErrorString(err));               \
        }                                                                    \
    } while (0)

#define CUDA_CHECK_LAST()                                                   \
    do {                                                                     \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            return Error::cuda_error(std::string("CUDA error: ") +          \
                                     cudaGetErrorString(err));               \
        }                                                                    \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                     \
        cublasStatus_t status = (call);                                     \
        if (status != CUBLAS_STATUS_SUCCESS) {                              \
            return Error::cuda_error(std::string(#call) + " failed: " +     \
                                     std::to_string(static_cast<int>(status))); \
        }                                                                    \
    } while (0)

// GPU 信息
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

inline std::vector<GPUInfo> get_all_gpu_info() {
    std::vector<GPUInfo> infos;
    int count = get_device_count();

    for (int i = 0; i < count; ++i) {
        GPUInfo info;
        Error err = get_gpu_info(i, info);
        if (!err) infos.push_back(info);
    }

    return infos;
}

inline Error cuda_malloc(void** ptr, size_t size, int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(ptr, size));
    return Error::success();
}

inline void cuda_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

inline Error cuda_memcpy_h2d(void* dst, const void* src, size_t size, int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return Error::success();
}

inline Error cuda_memcpy_d2h(void* dst, const void* src, size_t size, int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return Error::success();
}

inline Error cuda_memcpy_d2d(void* dst, const void* src, size_t size, int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    return Error::success();
}

inline Error cuda_memcpy_peer(void* dst, int dst_device, const void* src, int src_device, size_t size) {
    CUDA_CHECK(cudaMemcpyPeer(dst, dst_device, src, src_device, size));
    return Error::success();
}

inline Error cuda_memset(void* ptr, int value, size_t size, int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMemset(ptr, value, size));
    return Error::success();
}

inline Error cuda_sync(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaDeviceSynchronize());
    return Error::success();
}

// cuBLAS 句柄管理
class CublasHandle {
public:
    CublasHandle() = default;
    ~CublasHandle() { destroy(); }

    Error create(int device_id) {
        device_id_ = device_id;
        CUDA_CHECK(cudaSetDevice(device_id));
        CUBLAS_CHECK(cublasCreate(&handle_));
        CUBLAS_CHECK(cublasSetMathMode(handle_, CUBLAS_TF32_TENSOR_OP_MATH));
        return Error::success();
    }

    void destroy() {
        if (handle_) {
            cublasDestroy(handle_);
            handle_ = nullptr;
        }
    }

    cublasHandle_t get() const { return handle_; }
    int device_id() const { return device_id_; }

private:
    cublasHandle_t handle_ = nullptr;
    int device_id_ = 0;
};

// 打印显存使用情况
inline void print_memory_usage(int device_id) {
    size_t free_mem, total_mem;
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_mem, &total_mem);

    size_t used = total_mem - free_mem;
    std::cout << "GPU " << device_id << " Memory: " << used / (1024 * 1024) << " / "
              << total_mem / (1024 * 1024) << " MB used" << std::endl;
}

}  // namespace cuda
}  // namespace ember

