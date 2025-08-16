#ifndef __EQUAL_CPU_H__
#define __EQUAL_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"

ELEMENTWISE_DESCRIPTOR(equal_op, cpu)

namespace op::equal_op::cpu {
typedef struct EqualOp {
public:
    static constexpr size_t num_inputs = 2;
    template <typename InputT>
    bool operator()(const InputT &a, const InputT &b) const {
        // Element-wise equality comparison
        if constexpr (std::is_floating_point_v<InputT>) {
            // For floating point types, use exact equality (same as torch.equal)
            return a == b;
        } else {
            return a == b;
        }
    }
} EqualOp;

} // namespace op::equal_op::cpu

#endif // __EQUAL_CPU_H__