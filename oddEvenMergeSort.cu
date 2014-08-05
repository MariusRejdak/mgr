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


__global__ static void CUDA_OddEvenMergeSortShared(int* __restrict__ values,
                                                   uint N)
{
    const uint idx = TDIM * BID * 2 + TID;
    extern __shared__ int s_v[];

    s_v[TID] = values[idx];
    s_v[TID+TDIM] = values[idx+TDIM];
    __syncthreads();

    for (uint size = 2; size <= N; size <<= 1)
    {
        uint stride = size / 2;
        uint offset = threadIdx.x & (stride - 1);

        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            if(s_v[pos] > s_v[pos + stride]) {
                int tmp = s_v[pos];
                s_v[pos] = s_v[pos + stride];
                s_v[pos + stride] = tmp;
            }
            stride >>= 1;
        }

        for (; stride > 0; stride >>= 1)
        {
            __syncthreads();
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            if (offset >= stride)
                if(s_v[pos - stride] > s_v[pos]) {
                    int tmp = s_v[pos - stride];
                    s_v[pos - stride] = s_v[pos];
                    s_v[pos] = tmp;
                }
        }
    }

    __syncthreads();

    values[idx] = s_v[TID];
    values[idx+TDIM] = s_v[TID+TDIM];
}

__global__ static void CUDA_OddEvenMergeSortGlobal(int* __restrict__ values,
                                                   uint size,
                                                   uint stride)
{
    const uint idx = TDIM * BID + TID;

    //Odd-even merge
    uint pos = 2 * idx - (idx & (stride - 1));

    if (stride < size / 2)
    {
        uint offset = idx & ((size / 2) - 1);

        if (offset >= stride)
        {
            int keyA = values[pos - stride];
            int keyB = values[pos +      0];

            if(keyA > keyB) {
                values[pos - stride] = keyB;
                values[pos +      0] = keyA;
            }
        }
    }
    else
    {
        int keyA = values[pos +      0];
        int keyB = values[pos + stride];

        if(keyA > keyB) {
            values[pos +      0] = keyB;
            values[pos + stride] = keyA;
        }
    }
}

__host__ void inline OddEvenMergeSort(int** d_mem_values,
                                      const uint N)
{
    kdim v = get_kdim(N>>1);

    if (v.num_blocks == 1)
    {
        CUDA_OddEvenMergeSortShared<<<v.dim_blocks, v.num_threads, N*sizeof(int)>>>(*d_mem_values, N);
        cudaDeviceSynchronize();
    }
    else {
        CUDA_OddEvenMergeSortShared<<<v.dim_blocks, v.num_threads, v.num_threads*2*sizeof(int)>>>(*d_mem_values, v.num_threads*2);
        cudaDeviceSynchronize();

        for (uint size = 2 * v.num_threads; size <= N; size <<= 1)
        {
            for (uint stride = size / 2; stride > 0; stride >>= 1)
            {
                CUDA_OddEvenMergeSortGlobal<<<v.dim_blocks, v.num_threads>>>(*d_mem_values, size, stride);
                cudaDeviceSynchronize();
            }
        }
    }
}

// program main
int main(int argc, char** argv)
{
    void *h_mem, *d_mem_values; //, *d_mem_sorted;
    size_t min_size = 1024UL; //1kB
    size_t max_size = 1024UL*1024UL*256UL; //256MB

    h_mem = malloc(max_size);
    assert(h_mem != NULL);
    gpuErrchk( cudaMalloc(&d_mem_values, max_size) );

    srand(time(NULL));

    for(size_t size = min_size; size <= max_size; size <<= 1) {
        size_t N = size/sizeof(int);
        init_values_int((int*) h_mem, N);

        copy_to_device_time(d_mem_values, h_mem, size);
        cudaDeviceSynchronize();

        OddEvenMergeSort((int**) &d_mem_values, N);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        copy_to_host_time(h_mem, d_mem_values, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
    }

    cudaFree(d_mem_values);
    free(h_mem);

    return 0;
}
