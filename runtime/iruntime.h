#include <vector>

namespace ember {

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
};

class RuntimeFactory {
   public:
    static std::unique_ptr<IRuntime> create_cuda();
};

}  // namespace ember