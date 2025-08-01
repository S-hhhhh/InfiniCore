#include "../../../devices/nvidia/nvidia_common.cuh"
#include "reduce_mean_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "kernel.cuh"

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL ReduceMean(
    Tdata *y_, const Tdata *x_,
    size_t batch, size_t height, size_t width,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_h,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_h, ptrdiff_t x_stride_w) {
    ReduceMeanKernel<BLOCK_SIZE, Tdata, Tcompute>(y_, x_, batch, height, width, y_stride_b, y_stride_h, x_stride_b, x_stride_h, x_stride_w);
}

namespace op::reduce_mean::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    size_t dim) {
    auto info = ReduceMeanInfo::create(y_desc, x_desc, dim);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE>
infiniStatus_t launchKernel(void *y, const void *x, infiniDtype_t dtype,
                            size_t batch_size, size_t height, size_t width,
                            ptrdiff_t y_stride_b, ptrdiff_t y_stride_h,
                            ptrdiff_t x_stride_b, ptrdiff_t x_stride_h, ptrdiff_t x_stride_w, 
                            cudaStream_t stream) {
    dim3 grid(uint32_t(batch_size), uint32_t(height), 1);
    if (dtype == INFINI_DTYPE_F16) {
        ReduceMean<BLOCK_SIZE, half, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((half *)y, (const half *)x,
                                              batch_size, height, width,
                                              y_stride_b, y_stride_h,
                                              x_stride_b, x_stride_h, x_stride_w);
    } else if (dtype == INFINI_DTYPE_BF16) {
        ReduceMean<BLOCK_SIZE, __nv_bfloat16, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((__nv_bfloat16 *)y, (const __nv_bfloat16 *)x,
                                              batch_size, height, width,
                                              y_stride_b, y_stride_h,
                                              x_stride_b, x_stride_h, x_stride_w);
    } else if (dtype == INFINI_DTYPE_F32) {
        ReduceMean<BLOCK_SIZE, float, float>
            <<<grid, BLOCK_SIZE, 0, stream>>>((float *)y, (const float *)x,
                                              batch_size, height, width,
                                              y_stride_b, y_stride_h,
                                              x_stride_b, x_stride_h, x_stride_w);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y,
                                     const void *x,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_1024>(
            y, x, _info.dtype, _info.shape[0], _info.shape[1], _info.shape[2],
            _info.y_strides[0], _info.y_strides[1], _info.x_strides[0], _info.x_strides[1], _info.x_strides[2], stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_512>(
            y, x, _info.dtype, _info.shape[0], _info.shape[1], _info.shape[2],
            _info.y_strides[0], _info.y_strides[1], _info.x_strides[0], _info.x_strides[1], _info.x_strides[2], stream));
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CHECK_STATUS(launchKernel<CUDA_BLOCK_SIZE_4096>(
            y, x, _info.dtype, _info.shape[0], _info.shape[1], _info.shape[2],
            _info.y_strides[0], _info.y_strides[1], _info.x_strides[0], _info.x_strides[1], _info.x_strides[2], stream));
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::reduce_mean::nvidia
