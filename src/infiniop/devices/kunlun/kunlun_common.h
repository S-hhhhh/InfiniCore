#ifndef __INFINIOP_KUNLUN_COMMON_H__
#define __INFINIOP_KUNLUN_COMMON_H__

// This header file will only be include by .xpu file
#include "xpu/kernel/xtdk.h"
#include "xpu/kernel/xtdk_math.h"
#include "xpu/kernel/xtdk_simd.h"
#include "xpu/runtime.h"

#include <utility>
#if !defined(__xpu__) || defined(__xpu_on_host__)
#include_next <assert.h>
#else
#define assert(x)
#endif

// Get mask for kunlun xpu 512bit register calculation
// if data is not enough to 512bit, padding zero and use
// mask to identify real data
// 0 - i bit 1, others 0
inline __device__ float lowerBitMask(int i) {
    return (1 << (i + 1)) - 1;
}

// Atomic add for reduce
inline __device__ void atomicAddF32(__shared_ptr__ float *ptr, float value) {
    int success = 1;
    while (success) {
        // SM2REG read 32bit data to register
        float a = SM2REG_atomic(ptr);
        a = a + value;
        success = REG2SM_atomic(ptr, a);
    }
}
namespace op::kunlun_common {

inline __device__ long long indexToReducedOffset(
    long long flat_index,
    unsigned long long ndim,
    _global_ptr_ long long *broadcasted_strides,
    _global_ptr_ long long *target_strides) {
    long long res = 0;

    __local__ long long a[8];
    __local__ long long b[8];

    for (unsigned long long i = 0; i < ndim; ++i) {
        GM2LM(broadcasted_strides + i, a + i, 1 * sizeof(long long));
        GM2LM(target_strides + i, b + i, 1 * sizeof(long long));
        res += flat_index / a[i] * b[i];
        flat_index %= a[i];
        mfence();
    }
    return res;
}

inline __device__ long long indexToOffset(
    long long flat_index,
    unsigned long long ndim,
    _global_ptr_ unsigned long long *shape,
    _global_ptr_ long long *strides) {
    long long res = 0;

    __local__ long long b[8];
    __local__ unsigned long long c[8];

    for (unsigned long long i = ndim; i-- > 0;) {
        GM2LM(shape + i, c + i, 1 * sizeof(unsigned long long));
        GM2LM(strides + i, b + i, 1 * sizeof(long long));

        res += (flat_index % c[i]) * b[i];
        flat_index /= c[i];
        mfence();
    }
    return res;
}

inline __device__ long long getPaddedSize(
    unsigned long long ndim,
    _global_ptr_ unsigned long long *shape,
    _global_ptr_ long long *pads) {
    long long total_size = 1;

    __local__ unsigned long long c[8];
    __local__ long long d[8];
    for (unsigned long long i = 0; i < ndim; ++i) {
        GM2LM(shape + i, c + i, 1 * sizeof(unsigned long long));
        GM2LM(pads + i, d + i, 1 * sizeof(long long));

        total_size *= c[i] + (i < 2 ? 0 : 2 * d[i - 2]);
        mfence();
    }
    return total_size;
}

} // namespace op::kunlun_common

// TODO: atomicAddF16
// TODO: atomicAddI8
#endif
