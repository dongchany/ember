#include <vector>

namespace ember {

struct MemoryEstimate {
    size_t weights_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t activation_bytes = 0;
    size_t workspace_bytes = 0;
    size_t total_bytes = 0;

    void compute_total() {
        total_bytes =
            weights_bytes + kv_cache_bytes + activation_bytes + workspace_bytes;
    }

    std::string to_string() const;
};

struct DeviceMap {
    std::vector<int> layer_to_device;
    int embedding_device = 0;
    int lm_head_device = 0;
    int num_devices = 1;

    static DeviceMap single_device(int num_layers, int device_id = 0) {
        DeviceMap dm;
        dm.layer_to_device.assign(num_layers, device_id);
        dm.embedding_device = device_id;
        dm.lm_head_device = device_id;
        dm.num_devices = 1;
        return dm;
    }
};

// Runtime backend interface
class IRuntime {
   public:
    virtual const DeviceMap& device_map() const = 0;

    virtual bool available() const = 0;

    // Load model to specify device
    virtual Error load(const std::string& model_path, const ModelConfig& config,
                       const DeviceMap& device_map) = 0;

    virtual MemoryEstimate estimate_memory(const ModelConfig& config,
                                           int ctx_len, int batch_size = 1) = 0;
};

class RuntimeFactory {
   public:
    static std::unique_ptr<IRuntime> create_cuda();
};

}  // namespace ember