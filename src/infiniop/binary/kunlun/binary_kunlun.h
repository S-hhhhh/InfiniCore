#ifndef __INFINIOP_BINARY_KUNLUN_H__
#define __INFINIOP_BINARY_KUNLUN_H__

#include "../../devices/kunlun/kunlun_common.h"
#include <iostream>
namespace op::kunlun_common {

namespace binary_op {

void host2device(const unsigned long long *c_shape, const long long *c_strides, const unsigned long long *a_shape, const long long *a_strides,
                 const unsigned long long *b_shape, const long long *b_strides,
                 unsigned long long *xpu_c_shape, long long *xpu_c_strides, unsigned long long *xpu_a_shape, long long *xpu_a_strides,
                 unsigned long long *xpu_b_shape, long long *xpu_b_strides,
                 unsigned long long ndim);

// Perform binary computation when inputs and the output can have different dtypes
template <typename Tc, typename Ta, typename Tb, typename BinaryOp, typename... Args>
__global__ void calculate(unsigned long long c_data_size,
                          unsigned long long ndim,
                          bool contiguous,
                          bool broadcasted, Tc *c_, const Ta *a_, const Tb *b_,
                          unsigned long long *xpu_c_shape, long long *xpu_c_strides, unsigned long long *xpu_a_shape, long long *xpu_a_strides,
                          unsigned long long *xpu_b_shape, long long *xpu_b_strides,
                          Args &&...args) {

    unsigned long long data_size = c_data_size;
    int cid = core_id();
    int ncores = core_num();
    if (cid >= ncores) {
        return;
    }
    int thread_id = ncores * cluster_id() + cid;
    int nthreads = ncores * cluster_num();

    constexpr int buf_size = 512; // 保证所有内存加起来不超过16kB
    int task_size = buf_size * nthreads;

    __local__ Ta a_local[buf_size];
    __local__ Tb b_local[buf_size];
    __local__ Tc c_local[buf_size];

    int remain = data_size % task_size;
    int repeat = (data_size - remain) / task_size;

    int remain_task = remain % nthreads;
    int step_easy = (remain - remain_task) / nthreads;
    int step_hard = step_easy + 1;
    int step = (thread_id < remain_task ? step_hard : step_easy);
    int ind_start = repeat * task_size + (thread_id < remain_task ? thread_id * step_hard : remain_task * step_hard + (thread_id - remain_task) * step_easy);

    for (int r = 0; r < repeat + (step > 0 ? 1 : 0); r++) {
        int read_len = (r < repeat ? buf_size : step);
        int start = (r < repeat ? r * task_size + thread_id * buf_size : ind_start);
        if (contiguous) {
            GM2LM(a_ + start, a_local, read_len * sizeof(Ta));
            GM2LM(b_ + start, b_local, read_len * sizeof(Tb));

            for (int i = 0; i < read_len; i++) {
                c_local[i] = BinaryOp{}(a_local[i], b_local[i], std::forward<Args>(args)...);
            }
            mfence();

            LM2GM(c_local, c_ + start, read_len * sizeof(Tc));
        } else {
            for (int i = 0; i < read_len; i++) {
                int i_index = i + start;
                int a_index = broadcasted ? op::kunlun_common::indexToReducedOffset(i_index, ndim, xpu_c_strides, xpu_a_strides) : op::kunlun_common::indexToOffset(i_index, ndim, xpu_a_shape, xpu_a_strides);
                int b_index = broadcasted ? op::kunlun_common::indexToReducedOffset(i_index, ndim, xpu_c_strides, xpu_b_strides) : op::kunlun_common::indexToOffset(i_index, ndim, xpu_b_shape, xpu_b_strides);
                int c_index = op::kunlun_common::indexToOffset(i_index, ndim, xpu_c_shape, xpu_c_strides);

                GM2LM(a_ + a_index, a_local + i, 1 * sizeof(Ta));
                GM2LM(b_ + b_index, b_local + i, 1 * sizeof(Tb));
                c_local[i] = BinaryOp{}(a_local[i], b_local[i], std::forward<Args>(args)...);
                mfence();

                LM2GM(c_local + i, c_ + c_index, 1 * sizeof(Tc));
            }
        }
    }
}

// Perform binary computation when all inputs and the output share the same dtype
template <typename Tdata, typename BinaryOp, typename... Args>
__global__ void calculate(unsigned long long c_data_size,
                          unsigned long long ndim,
                          bool contiguous,
                          bool broadcasted, Tdata *c_, const Tdata *a_, const Tdata *b_,
                          unsigned long long *xpu_c_shape, long long *xpu_c_strides, unsigned long long *xpu_a_shape, long long *xpu_a_strides,
                          unsigned long long *xpu_b_shape, long long *xpu_b_strides,
                          Args &&...args) {

    unsigned long long data_size = c_data_size;

    int cid = core_id();
    int ncores = core_num();
    if (cid >= ncores) {
        return;
    }
    int thread_id = ncores * cluster_id() + cid;
    int nthreads = ncores * cluster_num();

    constexpr int buf_size = 512; // 保证所有内存加起来不超过16kB
    int task_size = buf_size * nthreads;

    __local__ Tdata a_local[buf_size];
    __local__ Tdata b_local[buf_size];
    __local__ Tdata c_local[buf_size];

    int remain = data_size % task_size;
    int repeat = (data_size - remain) / task_size;

    int remain_task = remain % nthreads;
    int step_easy = (remain - remain_task) / nthreads;
    int step_hard = step_easy + 1;
    int step = (thread_id < remain_task ? step_hard : step_easy);
    int ind_start = repeat * task_size + (thread_id < remain_task ? thread_id * step_hard : remain_task * step_hard + (thread_id - remain_task) * step_easy);

    for (int r = 0; r < repeat + (step > 0 ? 1 : 0); r++) {
        int read_len = (r < repeat ? buf_size : step);
        int start = (r < repeat ? r * task_size + thread_id * buf_size : ind_start);
        if (contiguous) {
            GM2LM(a_ + start, a_local, read_len * sizeof(Tdata));
            GM2LM(b_ + start, b_local, read_len * sizeof(Tdata));

            for (int i = 0; i < read_len; i++) {

                c_local[i] = BinaryOp{}(a_local[i], b_local[i], std::forward<Args>(args)...);
            }
            mfence();

            LM2GM(c_local, c_ + start, read_len * sizeof(Tdata));
        } else {
            for (int i = 0; i < read_len; i++) {
                int i_index = i + start;
                int a_index = broadcasted ? op::kunlun_common::indexToReducedOffset(i_index, ndim, xpu_c_strides, xpu_a_strides) : op::kunlun_common::indexToOffset(i_index, ndim, xpu_a_shape, xpu_a_strides);
                int b_index = broadcasted ? op::kunlun_common::indexToReducedOffset(i_index, ndim, xpu_c_strides, xpu_b_strides) : op::kunlun_common::indexToOffset(i_index, ndim, xpu_b_shape, xpu_b_strides);
                int c_index = op::kunlun_common::indexToOffset(i_index, ndim, xpu_c_shape, xpu_c_strides);

                GM2LM(a_ + a_index, a_local + i, 1 * sizeof(Tdata));
                GM2LM(b_ + b_index, b_local + i, 1 * sizeof(Tdata));
                c_local[i] = BinaryOp{}(a_local[i], b_local[i], std::forward<Args>(args)...);
                mfence();
                LM2GM(c_local + i, c_ + c_index, 1 * sizeof(Tdata));
            }
        }
    }
}
template <typename Tdata, typename BinaryOp, typename... Args>
void launch_calculate(unsigned long long c_data_size,
                      unsigned long long ndim,
                      bool contiguous,
                      bool broadcasted, const unsigned long long *c_shape, const long long *c_strides, const unsigned long long *a_shape, const long long *a_strides,
                      const unsigned long long *b_shape, const long long *b_strides, void *c, const void *a, const void *b, XPUStream stream,
                      Args... args) {

    char *workspace;
    int ret = 0;
    ret = xpu_malloc((void **)&workspace, ndim * (3 * sizeof(unsigned long long) + 3 * sizeof(long)));
    assert(ret == 0);
    char *tmp_strides = workspace + 3 * ndim * sizeof(unsigned long long);
    unsigned long long *xpu_c_shape = (unsigned long long *)workspace;
    unsigned long long *xpu_a_shape = xpu_c_shape + ndim;
    unsigned long long *xpu_b_shape = xpu_a_shape + ndim;
    long long *xpu_c_strides = (long long *)tmp_strides;
    long long *xpu_a_strides = xpu_c_strides + ndim;
    long long *xpu_b_strides = xpu_a_strides + ndim;

    host2device(c_shape, c_strides, a_shape, a_strides,
                b_shape, b_strides, xpu_c_shape, xpu_c_strides, xpu_a_shape, xpu_a_strides,
                xpu_b_shape, xpu_b_strides, ndim);

    calculate<Tdata, BinaryOp><<<8, 64, stream>>>(c_data_size,
                                                  ndim,
                                                  contiguous,
                                                  broadcasted, (Tdata *)c, (Tdata *)a, (Tdata *)b,
                                                  xpu_c_shape, xpu_c_strides,
                                                  xpu_a_shape, xpu_a_strides,
                                                  xpu_b_shape, xpu_b_strides,
                                                  std::forward<Args>(args)...);
    xpu_free(workspace);
}

template <typename Tc, typename Ta, typename Tb, typename BinaryOp, typename... Args>
void launch_calculate(unsigned long long c_data_size,
                      unsigned long long ndim,
                      bool contiguous,
                      bool broadcasted, const unsigned long long *c_shape, const long long *c_strides, const unsigned long long *a_shape, const long long *a_strides,
                      const unsigned long long *b_shape, const long long *b_strides, void *c, const void *a, const void *b, XPUStream stream,
                      Args... args) {

    char *workspace;
    int ret = 0;
    ret = xpu_malloc((void **)&workspace, ndim * 3 * (sizeof(unsigned long long) + sizeof(long long)));
    assert(ret == 0);
    char *tmp_strides = workspace + 3 * ndim * sizeof(unsigned long long);
    unsigned long long *xpu_c_shape = (unsigned long long *)workspace;
    unsigned long long *xpu_a_shape = xpu_c_shape + ndim;
    unsigned long long *xpu_b_shape = xpu_a_shape + ndim;
    long long *xpu_c_strides = (long long *)tmp_strides;
    long long *xpu_a_strides = xpu_c_strides + ndim;
    long long *xpu_b_strides = xpu_a_strides + ndim;
    host2device(c_shape, c_strides, a_shape, a_strides,
                b_shape, b_strides, xpu_c_shape, xpu_c_strides, xpu_a_shape, xpu_a_strides,
                xpu_b_shape, xpu_b_strides, ndim);
    calculate<Tc, Ta, Tb, BinaryOp><<<8, 64, stream>>>(c_data_size,
                                                       ndim,
                                                       contiguous,
                                                       broadcasted, (Tc *)c, (Ta *)a, (Tb *)b,
                                                       xpu_c_shape, xpu_c_strides,
                                                       xpu_a_shape, xpu_a_strides,
                                                       xpu_b_shape, xpu_b_strides,
                                                       std::forward<Args>(args)...);
    xpu_free(workspace);
}

} // namespace binary_op
} // namespace op::kunlun_common

#endif // __INFINIOP_BINARY_KUNLUN_H__
