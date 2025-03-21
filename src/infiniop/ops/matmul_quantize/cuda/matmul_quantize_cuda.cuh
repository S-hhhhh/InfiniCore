#ifndef __MATMUL_QUANTIZE_CUDA_CUH__
#define __MATMUL_QUANTIZE_CUDA_CUH__

#include "../../../operator.h"
#include "../../../tensor.h"

namespace op::matmul_quantize::cuda {
class Descriptor : public InfiniopDescriptor {
    int _m, _n, _k;
    size_t _workspace_size;
    infiniDtype_t _atype, _btype;
    int _num_bits;
    int _num_groups;
    int _group_size;

    Descriptor(int m, int n, int k,
               size_t workspace_size,
               infiniDtype_t atype, infiniDtype_t btype,
               int num_bits, int num_groups, int group_size,
               infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id},
          _m(m), _n(n), _k(k), _workspace_size(workspace_size),
          _atype(atype), _btype(btype),
          _num_bits(num_bits), _num_groups(num_groups), _group_size(group_size) {}

public:
    ~Descriptor();
    size_t minWorkspaceSize() const { return _workspace_size; }

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                 infiniopTensorDescriptor_t c_desc,
                                 infiniopTensorDescriptor_t a_desc,
                                 infiniopTensorDescriptor_t b_desc,
                                 infiniopTensorDescriptor_t b_scale_desc);
    infiniStatus_t calculate(
        void *workspace,
        size_t workspace_size,
        void *c,
        const void *a,
        const void *b,
        const void *b_scale,
        void *stream) const;
};
} // namespace op::matmul_quantize::cuda

#endif // __MATMUL_QUANTIZE_CUDA_CUH__
