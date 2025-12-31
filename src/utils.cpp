#include "utils.h"
#include <cuda_runtime.h>
#include <cstdarg>
#include <fstream>

namespace ember {

// ============ 日志实现 ============

static LogLevel g_log_level = LogLevel::INFO;

void set_log_level(LogLevel level) {
    g_log_level = level;
}

LogLevel get_log_level() {
    return g_log_level;
}

void log_message(LogLevel level, const char* fmt, ...) {
    if (level < g_log_level) return;
    
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

// ============ 错误处理实现 ============

const char* error_to_string(Error err) {
    switch (err) {
        case Error::OK:               return "成功";
        case Error::FILE_NOT_FOUND:   return "文件未找到";
        case Error::INVALID_MODEL:    return "无效的模型文件";
        case Error::UNSUPPORTED_ARCH: return "不支持的模型架构（仅支持 Qwen3 dense）";
        case Error::CUDA_ERROR:       return "CUDA 错误";
        case Error::OUT_OF_MEMORY:    return "显存不足";
        case Error::INVALID_ARGUMENT: return "参数无效";
        case Error::TOKENIZER_ERROR:  return "Tokenizer 错误";
    }
    return "未知错误";
}

// ============ 内存工具实现 ============

std::string format_bytes(size_t bytes) {
    char buf[64];
    if (bytes >= 1024ULL * 1024 * 1024) {
        snprintf(buf, sizeof(buf), "%.2f GiB", bytes / (1024.0 * 1024 * 1024));
    } else if (bytes >= 1024ULL * 1024) {
        snprintf(buf, sizeof(buf), "%.2f MiB", bytes / (1024.0 * 1024));
    } else if (bytes >= 1024) {
        snprintf(buf, sizeof(buf), "%.2f KiB", bytes / 1024.0);
    } else {
        snprintf(buf, sizeof(buf), "%zu B", bytes);
    }
    return buf;
}

int get_gpu_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        LOG_ERROR("无法获取 GPU 数量: %s", cudaGetErrorString(err));
        return 0;
    }
    return count;
}

GpuInfo get_gpu_info(int device_id) {
    GpuInfo info = {};
    info.device_id = device_id;
    
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        LOG_ERROR("无法获取 GPU %d 信息: %s", device_id, cudaGetErrorString(err));
        return info;
    }
    
    info.name = prop.name;
    info.total_memory = prop.totalGlobalMem;
    info.compute_capability_major = prop.major;
    info.compute_capability_minor = prop.minor;
    
    size_t free_mem, total_mem;
    cudaSetDevice(device_id);
    cudaMemGetInfo(&free_mem, &total_mem);
    info.free_memory = free_mem;
    
    return info;
}

void print_gpu_info() {
    int gpu_count = get_gpu_count();
    if (gpu_count == 0) {
        LOG_ERROR("未检测到 CUDA GPU");
        return;
    }
    
    LOG_INFO("检测到 %d 个 GPU:", gpu_count);
    for (int i = 0; i < gpu_count; ++i) {
        GpuInfo info = get_gpu_info(i);
        LOG_INFO("  [%d] %s - %s / %s (SM %d.%d)",
                 i, info.name.c_str(),
                 format_bytes(info.free_memory).c_str(),
                 format_bytes(info.total_memory).c_str(),
                 info.compute_capability_major,
                 info.compute_capability_minor);
    }
}

// ============ 文件工具实现 ============

bool file_exists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

size_t file_size(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return f.tellg();
}

} // namespace ember
