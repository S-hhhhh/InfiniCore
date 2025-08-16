#ifndef __REDUCE_MAX_INFO_H__
#define __REDUCE_MAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::reduce_max {

class ReduceMaxInfo {
    ReduceMaxInfo() = default;

public:
    infiniDtype_t dtype;

    std::vector<size_t> shape;
    std::vector<ptrdiff_t> output_strides;
    std::vector<ptrdiff_t> input_strides;

    static utils::Result<ReduceMaxInfo> create(infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, size_t dim) {
        auto dtype = output_desc->dtype();
        if (dtype != input_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        size_t ndim = output_desc->ndim();
        if (input_desc->ndim() != ndim) {
            CHECK_STATUS(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }
        CHECK_REDUCE_SHAPE(input_desc->shape(), dim, output_desc->shape());
        
        if(ndim == 3){
            std::vector<size_t> shape = input_desc->shape();
            std::vector<ptrdiff_t> output_strides = output_desc->strides();
            std::vector<ptrdiff_t> input_strides = input_desc->strides();
            if (dim != 2){
                std::swap(shape[dim], shape[2]);
                std::swap(output_strides[dim], output_strides[2]);
                std::swap(input_strides[dim], input_strides[2]);
            }
            return utils::Result<ReduceMaxInfo>(ReduceMaxInfo{
                dtype, shape, output_strides, input_strides});
        }
        else if (ndim == 2){
            std::vector<size_t> shape = input_desc->shape();
            std::vector<ptrdiff_t> output_strides = output_desc->strides();
            std::vector<ptrdiff_t> input_strides = input_desc->strides();
            if (dim != 1){
                std::swap(shape[dim], shape[1]);
                std::swap(output_strides[dim], output_strides[1]);
                std::swap(input_strides[dim], input_strides[1]);
            }
            shape.insert(shape.begin(), 1);
            output_strides.insert(output_strides.begin(), 0);
            input_strides.insert(input_strides.begin(), 0);
            return utils::Result<ReduceMaxInfo>(ReduceMaxInfo{
                dtype, shape, output_strides, input_strides});
        }
        else if (ndim == 1){
            std::vector<size_t> shape = {1, 1, (input_desc->shape())[0]};
            std::vector<ptrdiff_t> output_strides = {0, 0, (output_desc->strides())[0]};
            std::vector<ptrdiff_t> input_strides = {0, 0, (input_desc->strides())[0]};
            return utils::Result<ReduceMaxInfo>(ReduceMaxInfo{
                dtype, shape, output_strides, input_strides});
        }
        else if (ndim == 0){
            std::vector<size_t> shape = {1, 1, 1};
            std::vector<ptrdiff_t> output_strides = {0, 0, 0};
            std::vector<ptrdiff_t> input_strides = {0, 0, 0};
            return utils::Result<ReduceMaxInfo>(ReduceMaxInfo{
                dtype, shape, output_strides, input_strides});
        }
        else{
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }
};

} // namespace op::reduce_max

#endif // __REDUCE_MAX_INFO_H__