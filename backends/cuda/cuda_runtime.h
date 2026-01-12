#pragma once

#include "../../runtime/iruntime.h"

namespace ember {
namespace cuda {

class CudaRuntime : public IRuntime {
   public:
    CudaRuntime();
    ~CudaRuntime() override;
};
}  // namespace cuda

}  // namespace ember