#include "reduce_max_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../reduce/cpu/reduce.h"

namespace op::reduce_max::cpu {

Descriptor::~Descriptor() {}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t dim) {
    auto result = ReduceMaxInfo::create(output_desc, input_desc, dim);
    CHECK_RESULT(result);
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
infiniStatus_t reduce_max(const ReduceMaxInfo *info, T *output, const T *input) {
    const size_t batch_size = info->shape[0];
    const size_t rows = info->shape[1];
    const size_t cols = info->shape[2];  // 最后一维（规约维度）

    const ptrdiff_t output_batch_stride = info->output_strides[0];
    const ptrdiff_t output_row_stride = info->output_strides[1];
    const ptrdiff_t input_batch_stride = info->input_strides[0];
    const ptrdiff_t input_row_stride = info->input_strides[1];
    const ptrdiff_t input_col_stride = info->input_strides[2];
    
    #pragma omp parallel for collapse(2)
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t row = 0; row < rows; ++row) {
            const T* input_start = input + batch * input_batch_stride + row * input_row_stride;
            T* output_ptr = output + batch * output_batch_stride + row * output_row_stride;
            
            // 手动计算最大值，处理步幅
            if (cols == 0) {
                continue; // 跳过空的reduction
            }
            
            // 使用utils::cast进行类型转换
            float max_val = utils::cast<float>(input_start[0]);
            for (size_t i = 1; i < cols; ++i) {
                float val = utils::cast<float>(input_start[i * input_col_stride]);
                if (val > max_val) {
                    max_val = val;
                }
            }
            
            // 转换回原始类型
            *output_ptr = utils::cast<T>(max_val);
        }
    }
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *output,
    const void *input,
    void *stream) const {

    if (_info.dtype == INFINI_DTYPE_F16) {
        CHECK_STATUS(reduce_max<fp16_t>(&_info, (fp16_t *)output, (const fp16_t *)input));
    } else if (_info.dtype == INFINI_DTYPE_BF16) {
        CHECK_STATUS(reduce_max<bf16_t>(&_info, (bf16_t *)output, (const bf16_t *)input));
    } else if (_info.dtype == INFINI_DTYPE_F32) {
        CHECK_STATUS(reduce_max<float>(&_info, (float *)output, (const float *)input));
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::reduce_max::cpu