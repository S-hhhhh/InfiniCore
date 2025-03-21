#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/matmul_quantize.h"

#ifdef ENABLE_CUDA_API
#include "cuda/matmul_quantize_cuda.cuh"
#endif

__C infiniStatus_t infiniopCreateMatmulQuantizeDescriptor(infiniopHandle_t handle,
                                                          infiniopMatmulQuantizeDescriptor_t *desc_ptr,
                                                          infiniopTensorDescriptor_t c_desc,
                                                          infiniopTensorDescriptor_t a_desc,
                                                          infiniopTensorDescriptor_t b_desc,
                                                          infiniopTensorDescriptor_t b_scale_desc) {
    switch (handle->device) {
#ifdef ENABLE_CUDA_API
    case INFINI_DEVICE_NVIDIA:
        return op::matmul_quantize::cuda::Descriptor::create(handle, reinterpret_cast<op::matmul_quantize::cuda::Descriptor **>(desc_ptr), c_desc, a_desc, b_desc, b_scale_desc);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopGetMatmulQuantizeWorkspaceSize(infiniopMatmulQuantizeDescriptor_t desc, size_t *size) {
    switch (desc->device_type) {
#ifdef ENABLE_CUDA_API
    case INFINI_DEVICE_NVIDIA:
        *size = reinterpret_cast<op::matmul_quantize::cuda::Descriptor *>(desc)->minWorkspaceSize();
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopMatmulQuantize(infiniopMatmulQuantizeDescriptor_t desc,
                                          void *workspace,
                                          size_t workspace_size,
                                          void *c,
                                          void const *a,
                                          void const *b,
                                          void const *b_scale,
                                          void *stream) {
    switch (desc->device_type) {
    case INFINI_DEVICE_NVIDIA:
#ifdef ENABLE_CUDA_API
        return reinterpret_cast<op::matmul_quantize::cuda::Descriptor *>(desc)->calculate(workspace, workspace_size, c, a, b, b_scale, stream);
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}

__C infiniStatus_t infiniopDestroyMatmulQuantizeDescriptor(infiniopMatmulQuantizeDescriptor_t desc) {
    switch (desc->device_type) {
    case INFINI_DEVICE_NVIDIA:
#ifdef ENABLE_CUDA_API
        delete reinterpret_cast<op::matmul_quantize::cuda::Descriptor *>(desc);
        return INFINI_STATUS_SUCCESS;
#endif
    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }
}
