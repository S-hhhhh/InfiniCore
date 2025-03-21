# Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np
import torch
import torch.nn as nn
import unittest

import ctypes
from ctypes import POINTER, Structure, c_int32, c_size_t, c_uint64, c_void_p, c_float
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    create_handle,
    InfiniDeviceEnum,
    InfiniDtype,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    create_workspace,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
)


class MatmulQuantizeDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopMatmulQuantizeDescriptor_t = POINTER(MatmulQuantizeDescriptor)


lib = open_lib()

lib.infiniopCreateMatmulQuantizeDescriptor.restype = c_int32
lib.infiniopCreateMatmulQuantizeDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopMatmulQuantizeDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
]

lib.infiniopGetMatmulQuantizeWorkspaceSize.restype = c_int32
lib.infiniopGetMatmulQuantizeWorkspaceSize.argtypes = [
    infiniopMatmulQuantizeDescriptor_t,
    POINTER(c_size_t),
]

lib.infiniopMatmulQuantize.restype = c_int32
lib.infiniopMatmulQuantize.argtypes = [
    infiniopMatmulQuantizeDescriptor_t,
    c_void_p,
    c_uint64,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]

lib.infiniopDestroyMatmulQuantizeDescriptor.restype = c_int32
lib.infiniopDestroyMatmulQuantizeDescriptor.argtypes = [
    infiniopMatmulQuantizeDescriptor_t,
]
lib.infinirtSetDevice(InfiniDeviceEnum.NVIDIA, ctypes.c_int(0))
handle = create_handle(lib)


DEV = torch.device("cuda:0")


def marlin_matmul(A, B, C, s):
    a_tensor, b_tensor, c_tensor, s_tensor = (
        to_tensor(A, lib),
        to_tensor(B, lib),
        to_tensor(C, lib),
        to_tensor(s, lib),
    )
    b_desc = infiniopTensorDescriptor_t()
    lib.infiniopCreateTensorDescriptor(
        ctypes.byref(b_desc),
        2,
        (ctypes.c_size_t * 2)(*[A.shape[1], C.shape[1]]),
        (ctypes.c_int64 * 2)(*[C.shape[1], 1]),
        InfiniDtype.I4,
    )
    b_tensor.descriptor = b_desc
    descriptor = infiniopMatmulQuantizeDescriptor_t()
    check_error(
        lib.infiniopCreateMatmulQuantizeDescriptor(
            handle,
            ctypes.byref(descriptor),
            c_tensor.descriptor,
            a_tensor.descriptor,
            b_tensor.descriptor,
            s_tensor.descriptor,
        )
    )
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetMatmulQuantizeWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = create_workspace(workspace_size.value, DEV)
    lib.infiniopMatmulQuantize(
        descriptor,
        workspace.data_ptr(),
        workspace_size,
        C.data_ptr(),
        A.data_ptr(),
        B.data_ptr(),
        s.data_ptr(),
        None,
    )


# Precompute permutations for Marlin weight and scale shuffling
def _get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm, scale_perm_single


_perm, _scale_perm, _scale_perm_single = _get_perms()


class MarlinLayer(nn.Module):
    """PyTorch compatible Marlin layer; 4-bit (symmetric grouped) linear layer without bias."""

    def __init__(self, infeatures, outfeatures, groupsize=-1):
        """Create an empty Marlin layer.
        @infeatures: number of input features (must be divisible by 128)
        @outfeatures: number of output features (must be divisible by 256)
        @groupsize: quantization groupsize (must be -1 or 128)
        """
        super().__init__()
        if groupsize not in [-1, 128]:
            raise ValueError("Only groupsize -1 and 128 are supported.")
        if infeatures % 128 != 0 or outfeatures % 256 != 0:
            raise ValueError(
                "`infeatures` must be divisible by 128 and `outfeatures` by 256."
            )
        if groupsize == -1:
            groupsize = infeatures
        if infeatures % groupsize != 0:
            raise ValueError("`infeatures` must be divisible by `groupsize`.")
        self.k = infeatures
        self.n = outfeatures
        self.groupsize = groupsize
        self.register_buffer(
            "B", torch.empty((self.k // 16, self.n * 16 // 8), dtype=torch.int)
        )
        self.register_buffer(
            "s", torch.empty((self.k // groupsize, self.n), dtype=torch.half)
        )

    def forward(self, A):
        C = torch.empty(
            A.shape[:-1] + (self.s.shape[1],), dtype=A.dtype, device=A.device
        )
        marlin_matmul(
            A.view((-1, A.shape[-1])),
            self.B,
            C.view((-1, C.shape[-1])),
            self.s,
        )
        return C

    def pack(self, linear, scales):
        """Pack a fake-quantized linear layer into this actual Marlin representation.
        @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
        @scales: corresponding quantization scales of shape `(infeatures, groups)`
        """
        if linear.weight.dtype != torch.half:
            raise ValueError("Only `torch.half` weights are supported.")
        tile = 16
        maxq = 2**4 - 1
        s = scales.t()
        w = linear.weight.data.t()
        if self.groupsize != self.k:
            w = w.reshape((-1, self.groupsize, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.groupsize, -1))
            s = s.reshape((1, -1))
        w = torch.round(w / s).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)
        if self.groupsize != self.k:
            w = w.reshape((self.groupsize, -1, self.n))
            w = w.permute(1, 0, 2)
            w = w.reshape((self.k, self.n)).contiguous()
            s = s.reshape((-1, len(_scale_perm)))[:, _scale_perm]
        else:
            s = s.reshape((-1, len(_scale_perm_single)))[:, _scale_perm_single]
        s = s.reshape((-1, self.n)).contiguous()
        w = w.reshape((self.k // tile, tile, self.n // tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((self.k // tile, self.n * tile))
        res = w
        res = res.reshape((-1, _perm.numel()))[:, _perm].reshape(res.shape)
        q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
        res = res.cpu().numpy().astype(np.uint32)
        for i in range(8):
            q |= res[:, i::8] << 4 * i
        q = torch.from_numpy(q.astype(np.int32)).to(w.device)
        self.B[:, :] = q.to(self.B.device)
        self.s[:, :] = s.to(self.s.device)


def replace_linear(module, name_filter=lambda n: True, groupsize=-1, name=""):
    """Recursively replace all `torch.nn.Linear` layers by empty Marlin layers.
    @module: top-level module in which to perform the replacement
    @name_filter: lambda indicating if a layer should be replaced
    @groupsize: marlin groupsize
    @name: root-level name
    """
    if isinstance(module, MarlinLayer):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + "." + attr if name != "" else attr
        if isinstance(tmp, nn.Linear) and name_filter(name1):
            setattr(
                module,
                attr,
                MarlinLayer(tmp.in_features, tmp.out_features, groupsize=groupsize),
            )
    for name1, child in module.named_children():
        replace_linear(
            child,
            name_filter,
            groupsize=groupsize,
            name=name + "." + name1 if name != "" else name1,
        )


def gen_quant4(m, n, groupsize=-1):
    tile = 16
    maxq = 2**4 - 1
    w = torch.randn((m, n), dtype=torch.half, device=DEV)
    if groupsize != -1:
        w = w.reshape((-1, groupsize, n))
        w = w.permute(1, 0, 2)
        w = w.reshape((groupsize, -1))
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    ref = (w - (maxq + 1) // 2).half() * s
    if groupsize != -1:

        def reshape(w):
            w = w.reshape((groupsize, -1, n))
            w = w.permute(1, 0, 2)
            w = w.reshape((m, n)).contiguous()
            return w

        ref = reshape(ref)
        w = reshape(w)
    s = s.reshape((-1, n)).contiguous()
    linear = nn.Linear(m, n)
    linear.weight.data = ref.t()
    # Workaround to test some special cases that are forbidden by the API
    layer = MarlinLayer(256, 256, groupsize=groupsize)
    if groupsize == -1:
        groupsize = m
    layer.k = m
    layer.n = n
    layer.groupsize = groupsize
    layer.B = torch.empty((m // 16, n * 16 // 8), dtype=torch.int, device=DEV)
    layer.s = torch.empty((m // groupsize, n), dtype=torch.half, device=DEV)
    layer.pack(linear, s.t())
    q = layer.B
    s = layer.s
    return ref, q, s


class Test(unittest.TestCase):

    def run_problem(self, m, n, k, groupsize=-1):
        print("% 5d % 6d % 6d % 4d" % (m, n, k, groupsize))
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B, s = gen_quant4(k, n, groupsize=groupsize)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        C_ref = torch.matmul(A, B_ref)
        marlin_matmul(A, B, C, s)
        torch.cuda.synchronize()
        self.assertLess(
            torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.002
        )

    def test_tiles(self):
        print()
        for m in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 118, 128, 152, 768, 1024]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                self.run_problem(m, 2 * 256, 1024)

    # def test_k_stages_divisibility(self):
    #     print()
    #     for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
    #         self.run_problem(16, 2 * 256, k, 64, 256)

    # def test_very_few_stages(self):
    #     print()
    #     for k in [64, 128, 192]:
    #         self.run_problem(16, 2 * 256, k, 64, 256)

    def test_llama_shapes(self):
        print()
        MODELS = {
            " 7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
            "13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
            "33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
            "70B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
        }
        for _, layers in MODELS.items():
            for layer in layers:
                for thread_k, thread_n in [(128, 128)]:
                    for batch in [1, 16]:
                        self.run_problem(batch, layer[1], layer[0])

    # def test_groups(self):
    #     print()
    #     for m in [16]:
    #         for groupsize in [128]:
    #             for n, k in [(256, 512), (256, 1024), (256 * 128, 1024)]:
    #                 for thread_shape in [(128, 128), (64, 256)]:
    #                     self.run_problem(m, n, k, *thread_shape, groupsize)


if __name__ == "__main__":
    unittest.main()
