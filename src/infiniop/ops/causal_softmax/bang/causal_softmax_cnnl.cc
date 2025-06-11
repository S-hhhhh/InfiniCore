#include "causal_softmax_cnnl.h"
#include "../../../devices/bang/bang_handle.h"
#include "../../../devices/bang/common_bang.h"

namespace op::causal_softmax::cnnl {

struct Descriptor::Opaque {
    cnnlTensorDescriptor_t yDesc;
    cnnlTensorDescriptor_t maskDesc;
    std::shared_ptr<device::bang::Handle::Internal> internal;
    std::vector<int> dims;

    ~Opaque() {
        cnnlDestroyTensorDescriptor(yDesc);
        cnnlDestroyTensorDescriptor(maskDesc);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc) {
    auto handle = reinterpret_cast<device::bang::cambricon::Handle *>(handle_);

    auto result = CausalSoftmaxInfo::create(y_desc, x_desc);
    CHECK_RESULT(result);

    auto y_shape = y_desc->shape();
    if (y_desc->ndim() < 2 || y_shape[y_desc->ndim() - 1] < y_shape[y_desc->ndim() - 2]) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // cnnlMaskedSoftmax only support 4D or 5D tensors
    int ndim_ = std::max(static_cast<int>(y_desc->ndim()), 4);
    std::vector<int> dims(ndim_, 1);
    for (uint64_t i = 0; i < y_desc->ndim(); i++) {
        dims[ndim_ - 1 - i] = static_cast<int>(y_shape[y_desc->ndim() - i - 1]);
    }

    // Create tensor descriptors
    cnnlTensorDescriptor_t yDesc, maskDesc;
    CHECK_BANG(cnnlCreateTensorDescriptor(&yDesc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&maskDesc));

    // Set tensor descriptors
    CHECK_BANG(cnnlSetTensorDescriptor(yDesc, CNNL_LAYOUT_ARRAY,
                                       device::bang::getCnnlDtype(y_desc->dtype()),
                                       dims.size(), dims.data()));
    CHECK_BANG(cnnlSetTensorDescriptor(maskDesc, CNNL_LAYOUT_ARRAY, CNNL_DTYPE_BOOL,
                                       dims.size(), dims.data()));

    // Calculate workspace size (for mask)
    size_t wsSize = sizeof(bool) * dims[0] * dims[1] * dims[2] * dims[3];

    *desc_ptr = new Descriptor(
        new Opaque{
            yDesc,
            maskDesc,
            static_cast<device::bang::Handle *>(handle)->internal(),
            std::move(dims)},
        result.take(),
        wsSize,
        handle->device,     // Changed from device() to device
        handle->device_id); // Changed from device_id() to device_id

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y, const void *x, void *stream) const {
    // Create and fill mask matrix (upper triangular)
    bool mask_matrix[_opaque->dims[0]][_opaque->dims[1]][_opaque->dims[2]][_opaque->dims[3]];
    for (int i = 0; i < _opaque->dims[0]; ++i) {
        for (int j = 0; j < _opaque->dims[1]; ++j) {
            for (int m = 0; m < _opaque->dims[2]; ++m) {
                for (int n = 0; n < _opaque->dims[3]; ++n) {
                    if (n - m > _opaque->dims[3] - _opaque->dims[2]) { // Upper triangular mask
                        mask_matrix[i][j][m][n] = true;
                    } else {
                        mask_matrix[i][j][m][n] = false;
                    }
                }
            }
        }
    }

    // Copy mask to device
    size_t mask_size = sizeof(bool) * _opaque->dims[0] * _opaque->dims[1] * _opaque->dims[2] * _opaque->dims[3];
    cnrtMemcpyAsync(workspace, mask_matrix, mask_size,
                    reinterpret_cast<cnrtQueue_t>(stream),
                    cnrtMemcpyHostToDev);

    // Perform masked softmax
    CHECK_STATUS(_opaque->internal->useCnnl(
        reinterpret_cast<cnrtQueue_t>(stream),
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlMaskedSoftmax(handle, CNNL_MASKED_SOFTMAX_MASKED_FILL,
                                         -1, 1.0, _opaque->yDesc, x,
                                         _opaque->maskDesc, workspace,
                                         _opaque->yDesc, y));
            return INFINI_STATUS_SUCCESS;
        }));

    cnrtQueueSync(reinterpret_cast<cnrtQueue_t>(stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::causal_softmax::cnnl
