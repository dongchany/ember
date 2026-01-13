#include "cuda_runtime.h"

namespace ember {

static std::string format_bytes(size_t bytes) {
    if (bytes >= 1024 * 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024 * 1024)) + "." +
               std::to_string((bytes / (1024 * 1024 * 10)) % 100) + " GB";
    } else if (bytes >= 1024 * 1024) {
        return std::to_string(bytes / (1024 * 1024)) + "." +
               std::to_string((bytes / (1024 * 10)) % 100) + " MB";
    } else {
        return std::to_string(bytes / 1024) + " KB";
    }
}

std::string MemoryEstimate::to_string() const {
    std::string s;
    s += "Memory Estimate:\n";
    s += "  Weights:     " + format_bytes(weights_bytes) + "\n";
    s += "  KV Cache:    " + format_bytes(kv_cache_bytes) + "\n";
    s += "  Activations: " + format_bytes(activation_bytes) + "\n";
    s += "  Workspace:   " + format_bytes(workspace_bytes) + "\n";
    s += "  Total:       " + format_bytes(total_bytes) + "\n";
    return s;
}

namespace cuda {
bool CudaRuntime::available() const { return get_device_count() > 0; }

MemoryEstimate CudaRuntime::estimate_memory(const ModelConfig& config,
                                            int ctx_len, int batch_size) {}

Error CudaRuntime::load(const std::string& model_path,
                        const ModelConfig& config,
                        const DeviceMap& device_map) {
    std::cout << "[CudaRuntime] Loading model from: " << model_path
              << std::endl;
    std::cout << "[CudaRuntime] Model: " << config.model_type << std::endl;
    std::cout << "[CudaRuntime] Layers: " << config.num_layers << std::endl;
    std::cout << "[CudaRuntime] Hidden size: " << config.hidden_size
              << std::endl;
}
}  // namespace cuda
std::unique_ptr<IRuntime> RuntimeFactory::create_cuda() {
    return std::make_unique<cuda::CudaRuntime>();
}
}  // namespace ember