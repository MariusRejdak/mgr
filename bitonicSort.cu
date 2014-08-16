/*
 * bitonicSort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "cuda_utils.h"


__global__ static void CUDA_BitonicSort(Element* __restrict__ values,
                                        const int32_t j,
                                        const int32_t k)
{
    const int32_t idx = TDIM * BID + TID;
    const int32_t ixj = idx^j;

    if (ixj > idx) {
        const Element v_idx = values[idx];
        const Element v_ixj = values[ixj];

        if ((idx&k) ? (v_idx.k < v_ixj.k) : (v_idx.k > v_ixj.k)) {
            values[idx] = v_ixj;
            values[ixj] = v_idx;
        }
    }
}

__host__ void inline BitonicSort(Element* d_mem, const int32_t N)
{
    kdim v = get_kdim(N);

    for (int32_t k = 2; k <= N; k <<= 1) {
        for (int32_t j = k >> 1; j > 0; j >>= 1) {
            CUDA_BitonicSort<<<v.dim_blocks, v.num_threads>>>(d_mem, j, k);
            cudaDeviceSynchronize();
        }
    }
}

// program main
int main(int argc, char** argv)
{
    void *h_mem, *d_mem;

    h_mem = malloc(MAX_SIZE);
    assert(h_mem != NULL);
    gpuErrchk(cudaMalloc(&d_mem, MAX_SIZE));

    srand(time(NULL));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        init_values((Element*) h_mem, N);

        copy_to_device_time(d_mem, h_mem, size);
        cudaDeviceSynchronize();

        BitonicSort((Element*) d_mem, N);
        gpuErrchk( cudaPeekAtLastError() );

        copy_to_host_time(h_mem, d_mem, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((Element*) h_mem, N, false) ? "true":"false");
    }

    cudaFree(d_mem);
    free(h_mem);

    return 0;
}
