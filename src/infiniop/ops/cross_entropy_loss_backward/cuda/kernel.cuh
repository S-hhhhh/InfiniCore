#ifndef CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H
#define CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H

namespace op::cross_entropy_loss_backward::cuda {
typedef struct CrossEntropyLossBackwardOp {
public:
    static constexpr size_t num_inputs = 2;
    // CrossEntropyLossBackwardOp(size_t batch_size) : batch_size_(batch_size) {}
    
    template <typename T>
    __device__ __forceinline__ T operator()(const T &probs, const T &target, const size_t batch_size) const {
        // grad_logits = (probs - target) / batch_size (reduction='mean')
        T diff;
        T scale = static_cast<T>(1.0) / static_cast<T>(batch_size);
        
        if constexpr (std::is_same_v<T, half2>) {
            diff = __hsub2(probs, target);
            return __hmul2(diff, __float2half2_rn(static_cast<float>(scale)));
        } else if constexpr (std::is_same_v<T, half>) {
            diff = __hsub(probs, target);
            return __hmul(diff, __float2half(static_cast<float>(scale)));
        } else if constexpr (std::is_same_v<T, cuda_bfloat16>) {
            diff = __hsub(probs, target);
            return __hmul(diff, __float2bfloat16(static_cast<float>(scale)));
        } else if constexpr (std::is_same_v<T, float>) {
            diff = __fsub_rd(probs, target);
            return __fmul_rd(diff, static_cast<float>(scale));
        } else {
            // fallback for other types (double, etc.)
            diff = probs - target;
            return diff * scale;
        }
    }
    
} CrossEntropyLossBackwardOp;
} // namespace op::cross_entropy_loss_backward::cuda

#endif // CROSS_ENTROPY_LOSS_BACKWARD_CUDA_H