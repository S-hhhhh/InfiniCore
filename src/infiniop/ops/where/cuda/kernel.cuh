#ifndef __WHERE_CUDA_H__
#define __WHERE_CUDA_H__

namespace op::where::cuda {

typedef struct WhereOp {
public:
    static constexpr size_t num_inputs = 3;

    template <typename T>
    __device__ __forceinline__ T operator()(const bool &condition, const T &a, const T &b) const {
        if constexpr (std::is_same_v<T, half2> || std::is_same_v<T, cuda_bfloat162>) {
            // For vectorized half types, apply element-wise selection
            return condition ? a : b;
        } else {
            return condition ? a : b;
        }
    }
} WhereOp;

} // namespace op::where::cuda

#endif // __WHERE_CUDA_H__