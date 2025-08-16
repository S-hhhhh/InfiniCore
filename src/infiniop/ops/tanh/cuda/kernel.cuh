#ifndef TANH_CUDA_H
#define TANH_CUDA_H

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace op::tanh::cuda {
typedef struct TanhOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // For half2, process each half separately
            half x1 = __low2half(x);
            half x2 = __high2half(x);
            half tanh_x1 = htanh(x1);
            half tanh_x2 = htanh(x2);
            return __halves2half2(tanh_x1, tanh_x2);
        } else if constexpr (std::is_same_v<T, half>) {
            // Use CUDA intrinsic for half precision
            return htanh(x);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            // Convert to float for computation to maintain precision
            float x_f = __bfloat162float(x);
            float result = tanhf(x_f);
            return __float2bfloat16(result);
        } else if constexpr (std::is_same_v<T, float>) {
            // Use fast math functions for float
            return tanhf(x);
        } else if constexpr (std::is_same_v<T, double>) {
            return ::tanh(x);
        } else {
            // Fallback
            return tanhf(x);
        }
    }
} TanhOp;
} // namespace op::tanh::cuda

#endif // TANH_CUDA_H