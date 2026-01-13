#pragma once

namespace ember {
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

class Error {
   public:
    Error() : code_(ErrorCode::OK) {}
    Error(ErrorCode code, std::string message)
        : code_(code), message_(std::move(message)) {}

    const std::string& message() const {return message_;}

    static Error success() { return Error(); }

    static Error cuda_error(const std::string& msg) {
        return Error(ErrorCode::CUDA_ERROR, msg);
    }

   private:
    ErrorCode code_;
    std::string message_;
};

}  // namespace ember