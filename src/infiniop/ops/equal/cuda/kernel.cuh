#ifndef __EQUAL_CUDA_H__
#define __EQUAL_CUDA_H__

namespace op::equal_op::cuda {
typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;
    
    template <typename T>
    __device__ __forceinline__ bool operator()(const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2>) {
            // For half2, both components must be equal
            half2 result = __heq2(a, b);
            // __heq2 returns 0xFFFF for equal, 0x0000 for not equal
            // We need to check if both components are equal (both are 0xFFFF)
            return (__low2half(result) == __float2half(-1.0f)) && 
                   (__high2half(result) == __float2half(-1.0f));
        } else if constexpr (std::is_same_v<T, half>) {
            // __heq returns __nv_bool, we can directly return it as bool
            return static_cast<bool>(__heq(a, b));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            return a == b;  // CUDA supports == for bfloat16
        } else {
            // For all other types (float, double, int, bool, etc.)
            return a == b;
        }
    }
} EqualOp;
} // namespace op::equal_op::cuda

#endif // __EQUAL_CUDA_H__