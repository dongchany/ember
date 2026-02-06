#pragma once

#include <string>
#include <variant>
#include <utility>

namespace ember {

// 错误码
enum class ErrorCode {
    OK = 0,
    
    // 通用错误
    UNKNOWN = 1,
    INVALID_ARGUMENT = 2,
    OUT_OF_MEMORY = 3,
    NOT_IMPLEMENTED = 4,
    
    // 文件/IO 错误
    FILE_NOT_FOUND = 100,
    FILE_READ_ERROR = 101,
    FILE_WRITE_ERROR = 102,
    INVALID_FORMAT = 103,
    
    // 模型错误
    MODEL_LOAD_FAILED = 200,
    MODEL_NOT_LOADED = 201,
    UNSUPPORTED_ARCH = 202,
    WEIGHT_NOT_FOUND = 203,
    SHAPE_MISMATCH = 204,
    
    // CUDA 错误
    CUDA_ERROR = 300,
    CUDA_OUT_OF_MEMORY = 301,
    CUDA_DEVICE_NOT_FOUND = 302,
    
    // 推理错误
    CONTEXT_TOO_LONG = 400,
    INVALID_TOKEN = 401,
};

// 错误类
class Error {
public:
    Error() : code_(ErrorCode::OK) {}
    Error(ErrorCode code) : code_(code) {}
    Error(ErrorCode code, std::string message) 
        : code_(code), message_(std::move(message)) {}
    
    bool ok() const { return code_ == ErrorCode::OK; }
    explicit operator bool() const { return !ok(); }  // if (error) 表示有错误
    
    ErrorCode code() const { return code_; }
    const std::string& message() const { return message_; }
    
    std::string to_string() const {
        if (ok()) return "OK";
        std::string result = "Error[" + std::to_string(static_cast<int>(code_)) + "]";
        if (!message_.empty()) {
            result += ": " + message_;
        }
        return result;
    }
    
    // 静态工厂方法
    static Error success() { return Error(); }
    
    static Error invalid_argument(const std::string& msg) {
        return Error(ErrorCode::INVALID_ARGUMENT, msg);
    }
    
    static Error file_not_found(const std::string& path) {
        return Error(ErrorCode::FILE_NOT_FOUND, "File not found: " + path);
    }
    
    static Error out_of_memory(const std::string& msg = "") {
        return Error(ErrorCode::OUT_OF_MEMORY, msg);
    }
    
    static Error cuda_error(const std::string& msg) {
        return Error(ErrorCode::CUDA_ERROR, msg);
    }
    
    static Error not_implemented(const std::string& msg = "") {
        return Error(ErrorCode::NOT_IMPLEMENTED, msg);
    }

private:
    ErrorCode code_;
    std::string message_;
};

// Result 类型：要么是值，要么是错误
template<typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Error error) : data_(std::move(error)) {}
    
    bool ok() const { return std::holds_alternative<T>(data_); }
    explicit operator bool() const { return ok(); }
    
    T& value() { return std::get<T>(data_); }
    const T& value() const { return std::get<T>(data_); }
    
    Error& error() { return std::get<Error>(data_); }
    const Error& error() const { return std::get<Error>(data_); }
    
    T value_or(T default_value) const {
        return ok() ? value() : default_value;
    }

private:
    std::variant<T, Error> data_;
};

// 用于无返回值的操作
#define EMBER_RETURN_IF_ERROR(expr) \
    do { \
        auto _err = (expr); \
        if (_err) return _err; \
    } while (0)

#define EMBER_CHECK(cond, error) \
    do { \
        if (!(cond)) return (error); \
    } while (0)

}  // namespace ember
