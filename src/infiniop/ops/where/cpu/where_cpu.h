#ifndef __WHERE_CPU_H__
#define __WHERE_CPU_H__

#include "../../../elementwise/cpu/elementwise_cpu.h"
#include "infiniop/ops/where.h"

ELEMENTWISE_DESCRIPTOR(where, cpu)

namespace op::where::cpu {

typedef struct WhereOp {
public:
    static constexpr size_t num_inputs = 3;

    template <typename T, typename Tcond, typename Ta, typename Tb>
    T operator()(const Tcond &cond, const Ta &a, const Tb &b) const {
        return static_cast<T>(static_cast<bool>(cond) ? a : b); 
    }
} WhereOp;

} // namespace op::where::cpu

#endif // __WHERE_CPU_H__