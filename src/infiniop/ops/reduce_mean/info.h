﻿#ifndef __REDUCE_MEAN_INFO_H__
#define __REDUCE_MEAN_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::reduce_mean {

class ReduceMeanInfo {
    ReduceMeanInfo() = default;

public:
    infiniDtype_t dtype;

    std::vector<size_t> shape;
    std::vector<ptrdiff_t> y_strides;
    std::vector<ptrdiff_t> x_strides;

    static utils::Result<ReduceMeanInfo> create(infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc, size_t dim) {
        auto dtype = y_desc->dtype();
        if (dtype != x_desc->dtype()) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        size_t ndim = y_desc->ndim();
        if (x_desc->ndim() != ndim) {
            CHECK_STATUS(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }
        CHECK_REDUCE_SHAPE(x_desc->shape(), dim, y_desc->shape());
        if(ndim == 3){
            std::vector<size_t> shape = x_desc->shape();
            std::vector<ptrdiff_t> y_strides = y_desc->strides();
            std::vector<ptrdiff_t> x_strides = x_desc->strides();
            if (dim != 2){
                std::swap(shape[dim], shape[2]);
                std::swap(y_strides[dim], y_strides[2]);
                std::swap(x_strides[dim], x_strides[2]);
            }
            return utils::Result<ReduceMeanInfo>(ReduceMeanInfo{
                dtype, shape, y_strides, x_strides});
        }
        else if (ndim == 2){
            std::vector<size_t> shape = x_desc->shape();
            std::vector<ptrdiff_t> y_strides = y_desc->strides();
            std::vector<ptrdiff_t> x_strides = x_desc->strides();
            if (dim != 1){
                std::swap(shape[dim], shape[1]);
                std::swap(y_strides[dim], y_strides[1]);
                std::swap(x_strides[dim], x_strides[1]);
            }
            shape.insert(shape.begin(), 1);
            y_strides.insert(y_strides.begin(), 0);
            x_strides.insert(x_strides.begin(), 0);
            return utils::Result<ReduceMeanInfo>(ReduceMeanInfo{
                dtype, shape, y_strides, x_strides});
        }
        else if (ndim == 1){
            std::vector<size_t> shape = {1, 1, (x_desc->shape())[0]};
            std::vector<ptrdiff_t> y_strides = {0, 0, (y_desc->strides())[0]};
            std::vector<ptrdiff_t> x_strides = {0, 0, (x_desc->strides())[0]};
            return utils::Result<ReduceMeanInfo>(ReduceMeanInfo{
                dtype, shape, y_strides, x_strides});
        }
        else if (ndim == 0){
            std::vector<size_t> shape = {1, 1, 1};
            std::vector<ptrdiff_t> y_strides = {0, 0, 0};
            std::vector<ptrdiff_t> x_strides = {0, 0, 0};
            return utils::Result<ReduceMeanInfo>(ReduceMeanInfo{
                dtype, shape, y_strides, x_strides});
        }
        else{
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }
};

} // namespace op::reduce_mean

#endif // __REDUCE_MEAN_INFO_H__
