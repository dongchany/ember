#pragma once

#include "../../runtime/iruntime.h"

namespace ember {
namespace cuda {

class CudaRuntime : public IRuntime {
   public:
    CudaRuntime();
    ~CudaRuntime() override;

    bool available() const override;

    Error load(const std::string& model_path, const ModelConfig& config,
               const DeviceMap& device_map) override;

    MemoryEstimate estimate_memory(const ModelConfig& config, int ctx_len,
                                   int batch_size) override;
};
}  // namespace cuda

}  // namespace ember