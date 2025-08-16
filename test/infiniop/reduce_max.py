import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# 与 Reduce_Mean 保持一致的用例： (y_shape, x_shape, y_stride, x_stride, dim)
_TEST_CASES_ = [
    ((), (), None, None, 0),
    ((1, ), (32, ), None, None, 0),
    ((1, 4), (1, 4), None, None, 0),
    ((1, 1), (1, 4), None, None, 1),
    ((16, 1), (16, 2048), None, None, 1),
    ((1, 16), (2048, 16), None, None, 0),
    ((16, 1), (16, 2048), (4096, 1), (4096, 1), 1),
    ((1, 2048), (16, 2048), (4096, 1), (4096, 1), 0),
    ((4, 4, 1), (4, 4, 2048), None, None, 2),
    ((1, 4, 4), (2048, 4, 4), None, None, 0),
    ((4, 1, 4), (4, 2048, 4), (45056, 5632, 1), (32768, 8, 1), 1),
]

# x 的测试数据类型
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TEST_CASES = _TEST_CASES_

# 容差：对 Max 来说理论上可设为 0，但保持与 Mean 一致以复用工具函数
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def reduce_max(x, dim):
    # PyTorch 参考实现（保留维度）
    # return torch.amax(x, dim=dim, keepdim=True)
    return torch.max(x, dim=dim, keepdim=True).values


def test(
    handle,
    device,
    y_shape,
    x_shape,
    y_stride,
    x_stride,
    dim,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Reduce_Max on {InfiniDeviceNames[device]} with y_shape:{y_shape} x_shape:{x_shape}"
        f" y_stride:{y_stride} x_stride:{x_stride} dim:{dim} dtype:{InfiniDtypeNames[dtype]}"
    )

    # 构造输入与 PyTorch 参考结果
    x = TestTensor(x_shape, x_stride, dtype, device)
    ans = reduce_max(x.torch_tensor(), dim)

    # 目标输出张量
    y = TestTensor(y_shape, y_stride, dtype, device)

    if sync is not None:
        sync()

    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateReduceMaxDescriptor(
            handle, ctypes.byref(descriptor), y.descriptor, x.descriptor, ctypes.c_size_t(dim)
        )
    )

    # 使 desc 中的 shape/stride 失效，确保 kernel 不直接使用
    x.destroy_desc()
    y.destroy_desc()

    # 工作区
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetReduceMaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_reduce_max():
        check_error(
            LIBINFINIOP.infiniopReduceMax(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                None,
            )
        )

    # 调用库实现
    lib_reduce_max()

    if sync is not None:
        sync()

    # 校验
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    # 性能分析（可选）
    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: reduce_max(x.torch_tensor(), dim),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib",
            lambda: lib_reduce_max(),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )

    # 销毁描述符
    check_error(LIBINFINIOP.infiniopDestroyReduceMaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # 测试选项
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # 执行测试
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
