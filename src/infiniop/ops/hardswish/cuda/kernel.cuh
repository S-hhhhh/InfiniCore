#ifndef HARDSWISH_CUDA_H
#define HARDSWISH_CUDA_H

#include <type_traits>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace op::hardswish::cuda {

typedef struct HardSwishOp {
public:
    static constexpr size_t num_inputs = 1;

    template <typename T>
    __device__ __forceinline__ T operator()(const T &x) const {
        if constexpr (std::is_same_v<T, half2>) {
            // 先 clamp 到 [0,6]，再乘以 1/6，最后与 x 相乘
            const half2 three      = __float2half2_rn(3.0f);
            const half2 six        = __float2half2_rn(6.0f);
            const half2 zero       = __float2half2_rn(0.0f);
            const half2 one_sixth  = __float2half2_rn(1.0f / 6.0f);

            half2 t = __hadd2(x, three);               // x + 3
            t = __hmax2(zero, __hmin2(t, six));        // clamp to [0, 6]
            t = __hmul2(t, one_sixth);                 // t /= 6
            return __hmul2(x, t);                      // x * (clamp(x+3,0,6)/6)

            // 若仍需更高精度，可改用 float2 路径（更稳，稍慢）
            // float2 xf = __half22float2(x);
            // float2 tf;
            // tf.x = fminf(6.f, fmaxf(0.f, xf.x + 3.f)) * (1.f/6.f);
            // tf.y = fminf(6.f, fmaxf(0.f, xf.y + 3.f)) * (1.f/6.f);
            // return __floats2half2_rn(xf.x * tf.x, xf.y * tf.y);
        } else if constexpr (std::is_same_v<T, half>) {
            float xf = __half2float(x);
            float t  = fminf(6.0f, fmaxf(0.0f, xf + 3.0f)) * (1.0f / 6.0f);
            return __float2half(xf * t);
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            float xf = __bfloat162float(x);
            float t  = fminf(6.0f, fmaxf(0.0f, xf + 3.0f)) * (1.0f / 6.0f);
            return __float2bfloat16(xf * t);
        } else if constexpr (std::is_same_v<T, float>) {
            float t = fminf(6.0f, fmaxf(0.0f, x + 3.0f)) * (1.0f / 6.0f);
            return x * t;
        } else if constexpr (std::is_same_v<T, double>) {
            double t = fmin(6.0, fmax(0.0, x + 3.0)) * (1.0 / 6.0);
            return x * t;
        } else {
            float xf = static_cast<float>(x);
            float t  = fminf(6.0f, fmaxf(0.0f, xf + 3.0f)) * (1.0f / 6.0f);
            return static_cast<T>(xf * t);
        }
    }
} HardSwishOp;

} // namespace op::hardswish::cuda

#endif // HARDSWISH_CUDA_H
