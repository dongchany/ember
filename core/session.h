#include "config.h"

namespace ember {
class Session {
   public:
    Session() = default;

    void init(const ModelConfig& model_config,
              const RuntimeConfig& runtime_config) {
        model_config_ = model_config;
        runtime_config_ = runtime_config;
    }

   private:
    ModelConfig model_config_;
    RuntimeConfig runtime_config_;
};
}