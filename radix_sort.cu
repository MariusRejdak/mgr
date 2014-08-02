/*
 * bitonic_sort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "cuda_utils.h"


__global__ static void CUDA_RadixPrefixSum(int* __restrict__ values,
                                           uint* __restrict__ values_masks,
                                           uint* __restrict__ aux,
                                           const int mask)
{
    const uint idx = TDIM * BID + TID;
    uint tmp_in0 = (values[idx*2] & mask) ? 0 : 1;
    uint tmp_in1 = (values[idx*2 + 1] & mask) ? 0 : 1;

    extern __shared__ uint shared[];

    shared[TID] = tmp_in0 + tmp_in1;
    __syncthreads();

    for (uint i = 1; i < TDIM; i <<= 1)
    {
        const uint x = (i<<1)-1;
        if (TID >= i && (TID & x) == x)
        {
            shared[TID] += shared[TID - i];
        }
        __syncthreads();
    }

    if (TID == 0)
        shared[TDIM - 1] = 0;
    __syncthreads();

    for (uint i = TDIM>>1; i >= 1; i >>= 1)
    {
        uint x = (i<<1)-1;
        if (TID >= i && (TID & x) == x)
        {
            uint swp_tmp = shared[TID - i];
            shared[TID - i] = shared[TID];
            shared[TID] += swp_tmp;
        }
        __syncthreads();
    }

    values_masks[idx*2] = shared[TID];
    values_masks[idx*2 + 1] = shared[TID] + tmp_in0;

    __syncthreads();

    if (TID == TDIM-1)
        aux[BID] = tmp_in0 + shared[TID] + tmp_in1;
}

__global__ static void CUDA_RadixSort(int* __restrict__ values,
                                      int* __restrict__ values_sorted,
                                      uint* __restrict__ values_masks_psum,
                                      const int mask)
{
    const uint idx = TDIM * BID + TID;
    const uint bdim = TDIM * BDIM;
    const uint current = values[idx];
    const uint new_idx = values_masks_psum[idx];

    if (current & mask)
        values_sorted[idx + (values_masks_psum[bdim-1] + ((values[bdim-1] & mask) ? 0 : 1)) - new_idx] = current;
    else
        values_sorted[new_idx] = current;
}

__host__ void RadixPrefixSum(int* d_mem_values, uint* d_mem_masks,
                    const uint N, const int mask)
{
    uint *d_mem_aux;
    kdim v = get_kdim(N);

    gpuErrchk( cudaMalloc(&d_mem_aux, v.num_blocks * sizeof(uint)) );
    CUDA_RadixPrefixSum<<<v.dim_blocks, v.num_threads>>1, v.num_threads*sizeof(uint)>>>(d_mem_values, d_mem_masks, d_mem_aux, mask);
    cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );

    if (v.num_blocks > 1) {
        SumScan_Inclusive(d_mem_aux, v.num_blocks);
        CUDA_SumScanUpdate<<<v.dim_blocks, v.num_threads>>>(d_mem_masks, d_mem_aux);
        cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );
    }

    cudaFree(d_mem_aux);
}

__host__ void inline RadixSort(int* d_mem_values,
                         int* d_mem_sorted,
                         const uint N)
{
    int *d_v, *d_s;
    uint *d_m;
    kdim v = get_kdim(N);

    gpuErrchk( cudaMalloc(&d_m, N * sizeof(uint)) );

    for (short bit = 0; bit < sizeof(int)*8; ++bit)
    {
        int mask = 1 << bit;

        if (bit % 2) {
            d_v = d_mem_values;
            d_s = d_mem_sorted;
        }
        else {
            d_v = d_mem_sorted;
            d_s = d_mem_values;
        }

        RadixPrefixSum(d_v, d_m, N, mask);

        CUDA_RadixSort<<<v.dim_blocks, v.num_threads>>>(d_v, d_s, d_m, mask);
        cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );
    }

    cudaFree(d_m);
}

// program main
int main(int argc, char** argv) {
    void *h_mem, *d_mem_values, *d_mem_sorted;
    size_t min_size = 1024UL; //1kB
    size_t max_size = 1024UL*1024UL*256UL; //256MB

    h_mem = malloc(max_size);
    assert(h_mem != NULL);
    gpuErrchk( cudaMalloc(&d_mem_values, max_size) );
    gpuErrchk( cudaMalloc(&d_mem_sorted, max_size) );

    srand(time(NULL));

    for(size_t size = min_size; size <= max_size; size <<= 1) {
        size_t N = size/sizeof(int);
        init_values_int((int*) h_mem, N);

        copy_to_device_time(d_mem_values, h_mem, size);
        cudaDeviceSynchronize();

        RadixSort((int*) d_mem_values, (int*) d_mem_sorted, N);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        copy_to_host_time(h_mem, d_mem_sorted, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
    }

    cudaFree(d_mem_values);
    cudaFree(d_mem_sorted);
    free(h_mem);

    return 0;
}
