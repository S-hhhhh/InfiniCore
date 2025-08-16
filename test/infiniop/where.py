#!/usr/bin/env python3

import torch
import ctypes
from ctypes import c_uint64
from enum import Enum, auto

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
# 用例格式： (shape, a_stride, b_stride, c_stride, cond_stride)
_TEST_CASES_ = [
    # 基本形状
    ((10,), None, None, None, None),
    ((5, 10), None, None, None, None),
    ((2, 3, 4), None, None, None, None),

    # 边界/较大形状
    ((1, 1), None, None, None, None),
    ((7, 13), None, None, None, None),
    ((3, 5, 7), None, None, None, None),
    ((32, 256, 112, 112), None, None, None, None),

    # 如需覆盖非常规 stride，请确保内核支持后再放开
    # ((5, 10), (10, 2), None, (10, 2), None),
    # ((4, 6, 8), None, (48, 8, 1), (48, 8, 1), None),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-6},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()  # 输出 c 复用 a
    INPLACE_B = auto()  # 输出 c 复用 b

_INPLACE = [Inplace.OUT_OF_PLACE, Inplace.INPLACE_A, Inplace.INPLACE_B]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def where_ref(c_torch, a_torch, b_torch, cond_bool_torch):
    """
    PyTorch 参考实现：c = torch.where(condition, a, b)
    写入到 c_torch（与 TestTensor.c 绑定）
    """
    out = torch.where(cond_bool_torch, a_torch, b_torch)
    with torch.no_grad():
        c_torch.copy_(out)


def _pick_condition_dtype():
    """
    选择框架中的布尔 dtype。若枚举名不同请在此调整：
      - 优先用 InfiniDtype.BOOL
      - 若没有，可改为实际可用的布尔枚举（如 BOOL8）
    """
    if hasattr(InfiniDtype, "BOOL"):
        return InfiniDtype.BOOL
    # 如果项目里布尔枚举叫 BOOL8 / U1 等，请解除下面的注释并替换
    # elif hasattr(InfiniDtype, "BOOL8"):
    #     return InfiniDtype.BOOL8
    # 否则抛出异常，提示去 utils.to_torch_dtype 补映射
    raise RuntimeError(
        "No boolean dtype found in InfiniDtype. "
        "Please add a BOOL-like enum (e.g., BOOL / BOOL8) and map it to torch.bool in to_torch_dtype."
    )


def test(
    handle,
    device,
    shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    cond_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F32,
    sync=None,
):
    # a, b, c 的 dtype 必须一致
    a = TestTensor(shape, a_stride, dtype, device, mode="random", scale=4.0, bias=-2.0)
    b = TestTensor(shape, b_stride, dtype, device, mode="random", scale=4.0, bias=-2.0)

    # condition: 布尔张量
    # cond_bool = (torch.rand(shape) > 0.5)
    # cond_dtype = _pick_condition_dtype()
    # condition = TestTensor(
    #     shape,
    #     cond_bool.stride() if cond_stride is None else cond_stride,
    #     cond_dtype,
    #     device,
    #     mode="manual",
    #     set_tensor=cond_bool.to(torch.bool),
    # )
        # 生成布尔条件（用于 PyTorch 参考）
    cond_bool = (torch.rand(shape) > 0.5)

    # 用与 a/b/c 相同的数据类型，构造 0/1 蒙版，避免 BOOL dtype 的映射问题
    if dtype == InfiniDtype.F16:
        cond_mask = cond_bool.to(torch.float16)
    elif dtype == InfiniDtype.BF16:
        cond_mask = cond_bool.to(torch.bfloat16)
    else:
        cond_mask = cond_bool.to(torch.float32)

    condition = TestTensor(
        shape,
        cond_mask.stride() if cond_stride is None else cond_stride,
        dtype,                 # 重点：用 a/b/c 的 dtype，而不是 BOOL
        device,
        mode="manual",
        set_tensor=cond_mask,
    )


    # 输出 c：根据 inplace 复用 a 或 b 的缓冲区；stride 不匹配则跳过该 case
    if inplace == Inplace.INPLACE_A:
        if a_stride != c_stride:
            return
        c = a
    elif inplace == Inplace.INPLACE_B:
        if b_stride != c_stride:
            return
        c = b
    else:
        c = TestTensor(shape, c_stride, dtype, device)

    if c.is_broadcast():
        return

    print(
        f"Testing Where on {InfiniDeviceNames[device]} with shape:{shape} "
        f"a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} cond_stride:{cond_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # PyTorch 参考（写入 c.torch_tensor）
    where_ref(c.torch_tensor(), a.torch_tensor(), b.torch_tensor(), cond_bool)

    if sync is not None:
        sync()

    # ===== 描述子创建（顺序：c, condition, a, b）=====
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,            # output first
            condition.descriptor,    # condition second (BOOL)
            a.descriptor,
            b.descriptor,
        )
    )

    # Workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetWhereWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_where():
        # ===== 执行顺序也为：c, condition, a, b =====
        check_error(
            LIBINFINIOP.infiniopWhere(
                descriptor,
                workspace.data() if workspace is not None else None,
                workspace_size.value,
                c.data(),             # output first
                condition.data(),     # condition second
                a.data(),
                b.data(),
                None,
            )
        )

    # 运行库实现
    lib_where()

    # 校验
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG and c.actual_tensor().numel() > 0:
        max_diff = (c.actual_tensor() - c.torch_tensor()).abs().max().item()
        print("max |diff| =", max_diff)
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling（可选）
    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: where_ref(c.torch_tensor(), a.torch_tensor(), b.torch_tensor(), cond_bool),
            device, NUM_PRERUN, NUM_ITERATIONS
        )
        profile_operation(
            "    lib",
            lambda: lib_where(),
            device, NUM_PRERUN, NUM_ITERATIONS
        )

    # 释放描述子
    check_error(LIBINFINIOP.infiniopDestroyWhereDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mWhere test passed!\033[0m")
