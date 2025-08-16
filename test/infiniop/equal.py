import ctypes
from ctypes import c_uint64
from enum import Enum, auto

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceEnum,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [
    InfiniDtype.BOOL,
    InfiniDtype.I8,
    InfiniDtype.I16,
    InfiniDtype.I32,
    InfiniDtype.I64,
    InfiniDtype.BF16,
    InfiniDtype.F16,
    InfiniDtype.F32,
    InfiniDtype.F64,
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.BOOL: {"atol": 0, "rtol": 0},
    InfiniDtype.I8: {"atol": 0, "rtol": 0},
    InfiniDtype.I16: {"atol": 0, "rtol": 0},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
    InfiniDtype.F64: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def eq(c, a, b):
    torch.eq(a, b, out=c)


def test(
    handle,
    device,
    shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    a = TestTensor(shape, a_stride, dtype, device)
    b = TestTensor(shape, b_stride, dtype, device)
    if inplace == Inplace.INPLACE_A:
        if a_stride != c_stride:
            return
        c = a
    elif inplace == Inplace.INPLACE_B:
        if c_stride != b_stride:
            return
        c = b
    else:
        c = TestTensor(shape, c_stride, InfiniDtype.BOOL, device, mode="ones")

    if c.is_broadcast():
        return

    print(
        f"Testing Equal on {InfiniDeviceNames[device]} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    eq(c.torch_tensor(), a.torch_tensor(), b.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateEqualDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a, b, c]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetEqualWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_equal():
        check_error(
            LIBINFINIOP.infiniopEqual(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),
                a.data(),
                b.data(),
                None,
            )
        )

    lib_equal()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.equal(c.actual_tensor(), c.torch_tensor())

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: eq(c.torch_tensor(), a.torch_tensor(), b.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_equal(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyEqualDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        if device == InfiniDeviceEnum.ILUVATAR:
            _TENSOR_DTYPES = [
                InfiniDtype.BOOL,
                InfiniDtype.I8,
                InfiniDtype.I16,
                InfiniDtype.I32,
                InfiniDtype.I64,
                InfiniDtype.BF16,
                InfiniDtype.F16,
                InfiniDtype.F32,
            ]
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")


# import ctypes
# from ctypes import c_uint64
# from enum import Enum, auto
# import numpy as np

# import torch
# from libinfiniop import (
#     LIBINFINIOP,
#     InfiniDeviceEnum,
#     InfiniDeviceNames,
#     InfiniDtype,
#     InfiniDtypeNames,
#     TestTensor,
#     TestWorkspace,
#     check_error,
#     debug,
#     get_args,
#     get_test_devices,
#     get_tolerance,
#     infiniopOperatorDescriptor_t,
#     profile_operation,
#     test_operator,
# )

# class Inplace(Enum):
#     OUT_OF_PLACE = auto()
#     INPLACE_A = auto()
#     INPLACE_B = auto()

# def eq(c, a, b):
#     torch.eq(a, b, out=c)

# def simple_test():
#     """一个最简单的测试案例"""
#     print("=== SIMPLE EQUAL TEST ===")
    
#     # 创建最简单的测试数据
#     shape = (2, 2)
#     device = InfiniDeviceEnum.CUDA  # 假设是CUDA设备
#     dtype = InfiniDtype.BOOL
    
#     # 创建简单的测试张量
#     # 让 a 和 b 完全相同，这样所有位置都应该返回 True
#     a_data = torch.zeros(shape, dtype=torch.bool).cuda()  # 全是 False
#     b_data = torch.zeros(shape, dtype=torch.bool).cuda()  # 全是 False
    
#     print("Input a:")
#     print(a_data)
#     print("Input b:")
#     print(b_data)
    
#     # PyTorch 参考结果
#     expected_result = torch.eq(a_data, b_data)
#     print("Expected result (PyTorch):")
#     print(expected_result)
    
#     # 现在测试你的库
#     a = TestTensor(shape, None, dtype, device)
#     b = TestTensor(shape, None, dtype, device) 
#     c = TestTensor(shape, None, InfiniDtype.BOOL, device, mode="zeros")  # 改为zeros初始化
    
#     # 将我们的测试数据复制到TestTensor中
#     a.torch_tensor().copy_(a_data)
#     b.torch_tensor().copy_(b_data)
    
#     print("TestTensor a:")
#     print(a.torch_tensor())
#     print("TestTensor b:")
#     print(b.torch_tensor())
#     print("TestTensor c (before):")
#     print(c.torch_tensor())
    
#     # 创建描述符
#     handle = None  # 你需要从某个地方获取handle
#     descriptor = infiniopOperatorDescriptor_t()
#     check_error(
#         LIBINFINIOP.infiniopCreateEqualDescriptor(
#             handle,
#             ctypes.byref(descriptor),
#             c.descriptor,
#             a.descriptor,
#             b.descriptor,
#         )
#     )
    
#     # 销毁描述符中的形状信息（按照你的代码）
#     for tensor in [a, b, c]:
#         tensor.destroy_desc()
    
#     # 获取工作空间大小
#     workspace_size = c_uint64(0)
#     check_error(
#         LIBINFINIOP.infiniopGetEqualWorkspaceSize(
#             descriptor, ctypes.byref(workspace_size)
#         )
#     )
#     workspace = TestWorkspace(workspace_size.value, c.device)
    
#     print(f"Workspace size: {workspace_size.value}")
    
#     # 调用你的equal函数
#     check_error(
#         LIBINFINIOP.infiniopEqual(
#             descriptor,
#             workspace.data(),
#             workspace.size(),
#             c.data(),
#             a.data(),
#             b.data(),
#             None,
#         )
#     )
    
#     print("TestTensor c (after library call):")
#     print(c.actual_tensor())
    
#     # 比较结果
#     if torch.equal(c.actual_tensor(), expected_result):
#         print("✅ TEST PASSED!")
#     else:
#         print("❌ TEST FAILED!")
#         print("Expected:")
#         print(expected_result)
#         print("Got:")
#         print(c.actual_tensor())
        
#         # 详细比较
#         diff = expected_result.int() - c.actual_tensor().int()
#         print("Difference (expected - actual):")
#         print(diff)
    
#     # 清理
#     check_error(LIBINFINIOP.infiniopDestroyEqualDescriptor(descriptor))

# def debug_tensor_creation():
#     """调试张量创建过程"""
#     print("=== TENSOR CREATION DEBUG ===")
    
#     shape = (2, 2)
#     device = InfiniDeviceEnum.CUDA
#     dtype = InfiniDtype.BOOL
    
#     # 测试不同的初始化模式
#     for mode in ["zeros", "ones", "random"]:
#         print(f"\nTesting mode: {mode}")
#         try:
#             tensor = TestTensor(shape, None, dtype, device, mode=mode)
#             print(f"  Created tensor:")
#             print(f"  {tensor.torch_tensor()}")
#             print(f"  Actual tensor:")
#             print(f"  {tensor.actual_tensor()}")
#         except Exception as e:
#             print(f"  Error creating tensor with mode {mode}: {e}")

# if __name__ == "__main__":
#     print("Starting detailed debug...")
    
#     # 先测试张量创建
#     debug_tensor_creation()
    
#     # 然后运行简单测试
#     # 注意：你需要确保有正确的设备句柄
#     # simple_test()
    
#     print("Debug completed.")