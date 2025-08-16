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
from enum import Enum, auto

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

# Equal算子不支持inplace操作，所以只有OUT_OF_PLACE
class Inplace(Enum):
    OUT_OF_PLACE = auto()

# Equal算子只支持OUT_OF_PLACE操作
_INPLACE = [Inplace.OUT_OF_PLACE]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing - equal算子支持所有合法类型作为输入
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]  # 先测试基本的浮点类型

# Tolerance map for different data types - bool输出类型不需要tolerance，但保留用于调试
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
    InfiniDtype.I8: {"atol": 0, "rtol": 0},
    InfiniDtype.I16: {"atol": 0, "rtol": 0},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def equal(c, a, b):
    """PyTorch参考实现：element-wise相等比较，对应torch.eq而不是torch.equal"""
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
    # 创建输入张量a和b
    a = TestTensor(shape, a_stride, dtype, device)
    b = TestTensor(shape, b_stride, dtype, device)
    
    # equal算子不支持inplace操作，输出c始终是新的bool类型张量
    if inplace != Inplace.OUT_OF_PLACE:
        return  # Skip unsupported inplace operations
    
    # 输出张量c必须是bool类型
    # 由于InfiniDtype.BOOL可能不被支持，我们需要检查并使用替代方案
    try:
        c = TestTensor(shape, c_stride, InfiniDtype.BOOL, device, mode="zeros")
    except (ValueError, AttributeError):
        # 如果BOOL类型不支持，尝试使用I8或其他方式
        print("Warning: BOOL dtype not supported, using I8 as fallback")
        c = TestTensor(shape, c_stride, InfiniDtype.I8, device, mode="zeros")
        # 需要手动创建torch bool张量用于参考
        c_torch_bool = torch.zeros(shape, dtype=torch.bool, device=c.torch_tensor().device)

    if c.is_broadcast():
        return

    print(
        f"Testing Equal on {InfiniDeviceNames[device]} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # PyTorch参考计算
    if 'c_torch_bool' in locals():
        # 使用bool类型的张量进行参考计算
        torch.eq(a.torch_tensor(), b.torch_tensor(), out=c_torch_bool)
        reference_result = c_torch_bool
    else:
        torch.eq(a.torch_tensor(), b.torch_tensor(), out=c.torch_tensor())
        reference_result = c.torch_tensor()

    if sync is not None:
        sync()

    # 创建equal算子描述符
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

    # 获取工作空间大小
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

    # 执行库函数
    lib_equal()

    # 验证结果 - bool类型的精确比较，不需要tolerance
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    
    # 检查结果是否匹配（可能需要类型转换）
    actual_result = c.actual_tensor()
    if 'c_torch_bool' in locals():
        # 如果使用了fallback类型，需要转换为bool进行比较
        if actual_result.dtype != torch.bool:
            actual_result = actual_result.bool()
    
    # 添加详细的调试信息
    print(f"  Expected result dtype: {reference_result.dtype}, shape: {reference_result.shape}")
    print(f"  Actual result dtype: {actual_result.dtype}, shape: {actual_result.shape}")
    print(f"  Input a sample: {a.torch_tensor().flatten()[:5]}")
    print(f"  Input b sample: {b.torch_tensor().flatten()[:5]}")
    print(f"  Expected sample: {reference_result.flatten()[:10]}")
    print(f"  Actual sample: {actual_result.flatten()[:10]}")
    
    # 检查有多少元素不匹配
    mismatch_mask = actual_result != reference_result
    num_mismatches = mismatch_mask.sum().item()
    total_elements = actual_result.numel()
    print(f"  Mismatches: {num_mismatches}/{total_elements} ({100*num_mismatches/total_elements:.2f}%)")
    
    if num_mismatches > 0:
        print(f"  First few mismatches:")
        mismatch_indices = torch.where(mismatch_mask.flatten())[0][:5]
        for idx in mismatch_indices:
            idx = idx.item()
            print(f"    Index {idx}: expected {reference_result.flatten()[idx]}, got {actual_result.flatten()[idx]}")
            print(f"      a[{idx}] = {a.torch_tensor().flatten()[idx]}, b[{idx}] = {b.torch_tensor().flatten()[idx]}")
    
    if DEBUG:
        debug(actual_result, reference_result, atol=atol, rtol=rtol)
    
    assert torch.equal(actual_result, reference_result), "Equal operation results do not match exactly"

    # Profiling workflow
    if PROFILE:
        # fmt: off
        if 'c_torch_bool' in locals():
            profile_operation("PyTorch", lambda: torch.eq(a.torch_tensor(), b.torch_tensor(), out=c_torch_bool), device, NUM_PRERUN, NUM_ITERATIONS)
        else:
            profile_operation("PyTorch", lambda: torch.eq(a.torch_tensor(), b.torch_tensor(), out=c.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_equal(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    # 清理描述符
    check_error(LIBINFINIOP.infiniopDestroyEqualDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")