#ifndef __INFINIOP_MATMUL_QUANTIZE_API_H__
#define __INFINIOP_MATMUL_QUANTIZE_API_H__

#include "../operator_descriptor.h"

typedef InfiniopDescriptor *infiniopMatmulQuantizeDescriptor_t;

__C __export infiniStatus_t infiniopCreateMatmulQuantizeDescriptor(
    infiniopHandle_t handle,
    infiniopMatmulQuantizeDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc,
    infiniopTensorDescriptor_t b_desc,
    infiniopTensorDescriptor_t b_scale_desc);

__C __export infiniStatus_t infiniopGetMatmulQuantizeWorkspaceSize(infiniopMatmulQuantizeDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopMatmulQuantize(infiniopMatmulQuantizeDescriptor_t desc,
                                                   void *workspace,
                                                   size_t workspace_size,
                                                   void *c,
                                                   void const *a,
                                                   void const *b,
                                                   void const *b_scale,
                                                   void *stream);

__C __export infiniStatus_t infiniopDestroyMatmulQuantizeDescriptor(infiniopMatmulQuantizeDescriptor_t desc);

#endif
