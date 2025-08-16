#ifndef __REDUCE_MAX_KERNEL_CUH__
#define __REDUCE_MAX_KERNEL_CUH__

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void ReduceMaxKernel(
    Tdata *output_, const Tdata *input_,
    size_t batch, size_t height, size_t width,
    ptrdiff_t output_stride_b, ptrdiff_t output_stride_h,
    ptrdiff_t input_stride_b, ptrdiff_t input_stride_h, ptrdiff_t input_stride_w) {

    Tdata *output = output_ + blockIdx.x * output_stride_b + blockIdx.y * output_stride_h;
    const Tdata *input = input_ + blockIdx.x * input_stride_b + blockIdx.y * input_stride_h;

    // [Reduce] Find the max of each row using block reduce
    typedef cub::BlockReduce<Tcompute, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    
    // Initialize with negative infinity for max reduction
    Tcompute thread_max = static_cast<Tcompute>(-INFINITY);
    
    // Each thread processes elements with stride
    for (size_t i = threadIdx.x; i < width; i += blockDim.x) {
        Tcompute val = static_cast<Tcompute>(input[i * input_stride_w]);
        thread_max = max(thread_max, val);
    }
    
    // Perform block-wide reduction
    Tcompute block_max = BlockReduce(temp_storage).Reduce(thread_max, cub::Max());
    
    // Write result
    if (threadIdx.x == 0) {
        *output = static_cast<Tdata>(block_max);
    }
}

#endif // __REDUCE_MAX_KERNEL_CUH__
