#pragma once

#include "iruntime.h"

namespace ember {

struct RuntimeSetup {
    Session session;
    bool loaded = false;
    bool kv_allocated = false;
};

inline Error load_runtime(IRuntime& runtime,
                          const std::string& model_path,
                          const ModelConfig& model_config,
                          const DeviceMap& device_map,
                          RuntimeSetup& setup) {
    Error err = runtime.load(model_path, model_config, device_map);
    if (err) return err;
    setup.loaded = true;
    return Error::success();
}

inline Error init_session_and_kv(IRuntime& runtime,
                                 const ModelConfig& model_config,
                                 const RuntimeConfig& runtime_config,
                                 RuntimeSetup& setup) {
    setup.session.init(model_config, runtime_config);
    Error err = runtime.allocate_kv_cache(setup.session);
    if (err) return err;
    setup.kv_allocated = true;
    return Error::success();
}

inline void shutdown_runtime(IRuntime& runtime, RuntimeSetup& setup) {
    if (setup.kv_allocated) {
        runtime.free_kv_cache(setup.session);
        setup.kv_allocated = false;
    }
    if (setup.loaded) {
        runtime.unload();
        setup.loaded = false;
    }
}

}  // namespace ember
