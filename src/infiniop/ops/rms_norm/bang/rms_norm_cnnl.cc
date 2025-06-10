#include "rms_norm_cnnl.h"
#include "../../../devices/bang/bang_handle.h"
#include "../../../devices/bang/common_bang.h"

namespace op::rms_norm::cnnl {

struct Descriptor::Opaque {
    cnnlTensorDescriptor_t yDesc;
    cnnlTensorDescriptor_t xDesc;
    cnnlTensorDescriptor_t wDesc;
    cnnlFuseNormDescriptor_t opDesc;
    std::shared_ptr<device::bang::Handle::Internal> internal;

    ~Opaque() {
        cnnlDestroyTensorDescriptor(yDesc);
        cnnlDestroyTensorDescriptor(xDesc);
        cnnlDestroyTensorDescriptor(wDesc);
        cnnlDestroyFuseNormDescriptor(opDesc);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    float epsilon) {
    auto handle = reinterpret_cast<device::bang::cambricon::Handle *>(handle_);

    auto result = RMSNormInfo::create(y_desc, x_desc, w_desc, epsilon);
    CHECK_RESULT(result);
    auto info = result.take();

    // Create tensor descriptors
    cnnlTensorDescriptor_t yDesc, xDesc, wDesc;
    CHECK_BANG(cnnlCreateTensorDescriptor(&yDesc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&xDesc));
    CHECK_BANG(cnnlCreateTensorDescriptor(&wDesc));

    // Set tensor descriptors
    CHECK_STATUS(device::bang::setCnnlTensor(yDesc, y_desc));
    CHECK_STATUS(device::bang::setCnnlTensor(xDesc, x_desc));
    CHECK_STATUS(device::bang::setCnnlTensor(wDesc, w_desc));

    // Create and set RMSNorm descriptor
    cnnlFuseNormDescriptor_t opDesc;
    CHECK_BANG(cnnlCreateFuseNormDescriptor(&opDesc));
    CHECK_BANG(cnnlSetFuseNormDescriptor(opDesc, epsilon, 1.0, true,
                                         false, false, false, false,
                                         device::bang::getCnnlDtype(y_desc->dtype()),
                                         CNNL_TRANSFORMER_RMSNORM));

    // Get workspace size
    size_t wsSize = 0;
    CHECK_STATUS(handle->internal()->useCnnl(nullptr, [&](cnnlHandle_t cnnlHandle) {
        CHECK_BANG(cnnlGetFuseNormWorkspaceSize(cnnlHandle, opDesc, xDesc, &wsSize));
        return INFINI_STATUS_SUCCESS;
    }));

    *desc_ptr = new Descriptor(
        new Opaque{
            yDesc,
            xDesc,
            wDesc,
            opDesc,
            static_cast<device::bang::Handle *>(handle)->internal()},
        info,
        wsSize,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *y, const void *x, const void *w,
                                     void *stream) const {
    CHECK_STATUS(_opaque->internal->useCnnl(
        reinterpret_cast<cnrtQueue_t>(stream),
        [&](cnnlHandle_t handle) {
            CHECK_BANG(cnnlFuseNorm(handle,
                                    _opaque->opDesc,
                                    _opaque->xDesc, x,
                                    _opaque->wDesc, w,
                                    nullptr, nullptr,
                                    nullptr, nullptr,
                                    nullptr, nullptr,
                                    workspace, workspace_size,
                                    _opaque->yDesc, y,
                                    nullptr, nullptr));
            return INFINI_STATUS_SUCCESS;
        }));
    cnrtQueueSync(reinterpret_cast<cnrtQueue_t>(stream));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rms_norm::cnnl
