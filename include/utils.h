#pragma once

#include <cstdio>
#include <cstdint>
#include <string>
#include <chrono>

namespace ember {

// ============ 日志 ============

enum class LogLevel {
    DEBUG = 0,
    INFO  = 1,
    WARN  = 2,
    ERROR = 3
};

void set_log_level(LogLevel level);
LogLevel get_log_level();

// 日志宏
#define LOG_DEBUG(fmt, ...) \
    ember::log_message(ember::LogLevel::DEBUG, "[DEBUG] " fmt "\n", ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) \
    ember::log_message(ember::LogLevel::INFO, "[INFO]  " fmt "\n", ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) \
    ember::log_message(ember::LogLevel::WARN, "[WARN]  " fmt "\n", ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) \
    ember::log_message(ember::LogLevel::ERROR, "[ERROR] " fmt "\n", ##__VA_ARGS__)

void log_message(LogLevel level, const char* fmt, ...);

// ============ 错误处理 ============

// 错误码
enum class Error {
    OK = 0,
    FILE_NOT_FOUND,
    INVALID_MODEL,
    UNSUPPORTED_ARCH,
    CUDA_ERROR,
    OUT_OF_MEMORY,
    INVALID_ARGUMENT,
    TOKENIZER_ERROR
};

const char* error_to_string(Error err);

// CUDA 错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            LOG_ERROR("CUDA 错误 %s:%d: %s", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return Error::CUDA_ERROR; \
        } \
    } while(0)

// ============ 性能计时 ============

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
    double elapsed_s() const {
        return elapsed_ms() / 1000.0;
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============ 内存工具 ============

// 格式化字节数
std::string format_bytes(size_t bytes);

// 获取 GPU 信息
struct GpuInfo {
    int device_id;
    std::string name;
    size_t total_memory;
    size_t free_memory;
    int compute_capability_major;
    int compute_capability_minor;
};

int get_gpu_count();
GpuInfo get_gpu_info(int device_id);
void print_gpu_info();

// ============ 文件工具 ============

bool file_exists(const std::string& path);
size_t file_size(const std::string& path);

} // namespace ember
