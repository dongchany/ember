#pragma once

#include "kernels/common/cuda_utils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

namespace ember {
namespace cuda {

// Row-major GEMM wrapper.
//
// Notes:
// - cuBLAS is column-major by default. This wrapper provides a row-major view by
//   swapping operands and transposes under the hood.
// - This is an intentionally minimal interface to stabilize call sites before we
//   consider cuBLASLt and mixed-precision tuning.
inline Error gemm_row_major_f32(cublasHandle_t handle,
                                bool transA,
                                bool transB,
                                int M,
                                int N,
                                int K,
                                const float* A,
                                int lda,
                                const float* B,
                                int ldb,
                                float* C,
                                int ldc,
                                float alpha = 1.0f,
                                float beta = 0.0f,
                                cudaStream_t stream = nullptr) {
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // RowMajor: C[M,N] = op(A)[M,K] * op(B)[K,N]
    // Map to ColMajor: C^T[N,M] = op(B)^T[N,K] * op(A)^T[K,M]
    const cublasOperation_t opA_col = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t opB_col = transA ? CUBLAS_OP_T : CUBLAS_OP_N;

    // In column-major, leading dimension is the number of rows.
    const int lda_col = ldb;
    const int ldb_col = lda;
    const int ldc_col = N;

    CUBLAS_CHECK(cublasSgemm(handle,
                             opA_col,
                             opB_col,
                             N,
                             M,
                             K,
                             &alpha,
                             B,
                             lda_col,
                             A,
                             ldb_col,
                             &beta,
                             C,
                             ldc_col));
    return Error::success();
}

}  // namespace cuda
}  // namespace ember

