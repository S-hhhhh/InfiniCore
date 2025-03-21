#include "../../../../utils.h"
#include "../../../handle.h"
#include "gptq_marlin.cuh"
#include "matmul_quantize_cuda.cuh"

namespace op::matmul_quantize::cuda {
Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(infiniopHandle_t handle, Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t c_desc,
                                  infiniopTensorDescriptor_t a_desc,
                                  infiniopTensorDescriptor_t b_desc,
                                  infiniopTensorDescriptor_t b_scale_desc) {
    auto atype = a_desc->dtype();
    auto btype = b_desc->dtype();
    if ((atype != INFINI_DTYPE_F16 && atype != INFINI_DTYPE_BF16) || (btype != INFINI_DTYPE_I4 && btype != INFINI_DTYPE_I8)) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    int num_bits = b_desc->dtype() == INFINI_DTYPE_I4 ? 4 : 8;
    int m = int(c_desc->dim(0));
    int n = int(c_desc->dim(1));
    int k = int(a_desc->dim(1));
    int num_groups = int(b_scale_desc->dim(0));
    int group_size = num_groups > 1 ? group_size = k / num_groups : -1;
    int max_par = gptq_marlin::max_par;
    size_t min_workspace_size = n / gptq_marlin::min_thread_n * max_par * sizeof(int) + m * k * infiniSizeOf(atype);

    *desc_ptr = new Descriptor(m, n, k, min_workspace_size, atype, btype, num_bits, num_groups, group_size, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *c,
    const void *a,
    const void *b,
    const void *b_scale,
    void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    if (_atype == INFINI_DTYPE_F16) {
        gptq_marlin::gptq_marlin_mm_fp16(c, a, b, b_scale,
                                         _m, _n, _k,
                                         workspace, _num_bits,
                                         _num_groups, _group_size,
                                         this->device_id, (cudaStream_t)stream);
    } else if (_atype == INFINI_DTYPE_BF16) {
        gptq_marlin::gptq_marlin_mm_bf16(c, a, b, b_scale,
                                         _m, _n, _k,
                                         workspace, _num_bits,
                                         _num_groups, _group_size,
                                         this->device_id, (cudaStream_t)stream);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::matmul_quantize::cuda
