#pragma once

#include <string>

#include "core/config.h"

namespace ember {

ModelConfig parse_model_config(const std::string& config_path);

}  // namespace ember
