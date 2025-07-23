#include "ops.hpp"
#include "utils.hpp"
#include <infinirt.h>
#include <iomanip>
#include <iostream>

namespace infiniop_test::reduce_mean {
struct Test::Attributes {
    std::shared_ptr<Tensor> x;
    std::shared_ptr<Tensor> y;
    std::shared_ptr<Tensor> ans;
    size_t dim;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol, bool equal_nan) {
    auto test = std::shared_ptr<Test>(new Test(rtol, atol, equal_nan));
    test->_attributes = new Attributes();

    if (attributes.find("dim") == attributes.end()
        || tensors.find("x") == tensors.end()
        || tensors.find("ans") == tensors.end()
        || tensors.find("y") == tensors.end()) {
        throw std::runtime_error("Invalid Test: Missing attributes or tensors");
    }

    test->_attributes->dim = size_t(*reinterpret_cast<uint64_t *>(attributes["dim"].data()));
    test->_attributes->ans = tensors["ans"];
    test->_attributes->x = tensors["x"];
    test->_attributes->y = tensors["y"];

    return test;
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {

    infiniopReduceMeanDescriptor_t op_desc;
    CHECK_OR(infiniopCreateReduceMeanDescriptor(handle, &op_desc,
                                             _attributes->y->desc(),
                                             _attributes->x->desc(),
                                             _attributes->dim),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to create ReduceMean descriptor"));

    auto x = _attributes->x->to(device, device_id);
    auto y = _attributes->y->to(device, device_id);

    size_t workspace_size;
    CHECK_OR(infiniopGetReduceMeanWorkspaceSize(op_desc, &workspace_size),
             return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size"));
    void *workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_OR(infinirtMalloc(&workspace, workspace_size),
                 return TEST_FAILED(OP_CREATION_FAILED, "Failed to allocate workspace"));
    }

    CHECK_OR(infiniopReduceMean(op_desc,
                             workspace, workspace_size,
                             y->data(),
                             x->data(),
                             nullptr),
             return TEST_FAILED(OP_EXECUTION_FAILED, "ReduceMean execution failed"));

    try {
        allClose(y, _attributes->ans, _rtol, _atol, _equal_nan);
    } catch (const std::exception &e) {
        return TEST_FAILED(RESULT_INCORRECT, e.what());
    }

    double elapsed_time = 0.;

    elapsed_time = benchmark(
        [=]() {
            infiniopReduceMean(op_desc,
                            workspace, workspace_size,
                            y->data(),
                            x->data(),
                            nullptr);
        },
        warm_ups, iterations);

    if (workspace != nullptr) {
        infinirtFree(workspace);
    }

    return TEST_PASSED(elapsed_time);
}

std::vector<std::string> Test::attribute_names() {
    return {"dim"};
}

std::vector<std::string> Test::tensor_names() {
    return {"x", "ans", "y"};
}

std::vector<std::string> Test::output_names() {
    return {"y"};
}

std::string Test::toString() const {
    std::ostringstream oss;
    oss << op_name() << std::endl;
    oss << "- x: " << _attributes->x->info() << std::endl;
    oss << "- y: " << _attributes->y->info() << std::endl;
    oss << "- dim=" << _attributes->dim << std::endl;
    oss << std::scientific << std::setprecision(2);
    oss << "- rtol=" << _rtol << ", atol=" << _atol << ", equal_nan=" << _equal_nan << std::endl;
    return oss.str();
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::reduce_mean
