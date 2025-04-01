#include "swiglu_kunlun.h"
#include "../../../devices/kunlun/kunlun_handle.h"
#include <memory>
#include <stdint.h>
void swiglu_f32(unsigned long long c_data_size,
                unsigned long long ndim,
                bool contiguous,
                bool broadcasted, const unsigned long long *c_shape, const long long *c_strides, const unsigned long long *a_shape, const long long *a_strides,
                const unsigned long long *b_shape, const long long *b_strides, void *c, const void *a, const void *b, XPUStream stream);

namespace op::swiglu::kunlun {

struct Descriptor::Opaque {
    std::shared_ptr<device::kunlun::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t up_desc,
    infiniopTensorDescriptor_t gate_desc) {

    auto handle = reinterpret_cast<device::kunlun::Handle *>(handle_);
    auto dtype = out_desc->dtype();
    const auto &out_shape = out_desc->shape();
    const auto &up_shape = up_desc->shape();
    const auto &gate_shape = gate_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(out_shape, up_shape, gate_shape);

    op::binary::BinaryInfo info;
    CHECK_STATUS(op::binary::createBinaryInfo(info, out_desc, up_desc, gate_desc));

    // Create descriptor
    *desc_ptr = new Descriptor(
        dtype,
        std::move(info),
        new Descriptor::Opaque{static_cast<device::kunlun::Handle *>(handle)->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *c,
    const void *a,
    const void *b,
    void *stream) const {
    unsigned long long c_data_size = _info.c_data_size;
    unsigned long long ndim = _info.ndim;
    bool contiguous = _info.contiguous;
    bool broadcasted = _info.broadcasted;

    char *tmp = (char *)malloc(3 * ndim * (sizeof(unsigned long long) + sizeof(long long))); // 昆仑芯涉及的int64,uint64等数据类型必须全部用long long取代
    char *tmp_stride = tmp + 3 * ndim * sizeof(unsigned long long);
    unsigned long long *c_shape = (unsigned long long *)tmp;
    unsigned long long *a_shape = c_shape + ndim;
    unsigned long long *b_shape = a_shape + ndim;

    long long *c_strides = (long long *)tmp_stride;
    long long *a_strides = c_strides + ndim;
    long long *b_strides = a_strides + ndim;
    for (unsigned long long i = 0; i < ndim; i++) {
        c_strides[i] = _info.c_strides.data()[i];
        a_strides[i] = _info.a_strides.data()[i];
        b_strides[i] = _info.b_strides.data()[i];
        c_shape[i] = _info.c_shape.data()[i];
        a_shape[i] = _info.a_shape.data()[i];
        b_shape[i] = _info.b_shape.data()[i];
    }

    switch (_dtype) {
    case INFINI_DTYPE_F32:

        swiglu_f32(c_data_size,
                   ndim,
                   contiguous,
                   broadcasted, c_shape, c_strides, a_shape, a_strides,
                   b_shape, b_strides, c, a, b, reinterpret_cast<kunlunStream_t>(stream));
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    free(tmp);

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::swiglu::kunlun
