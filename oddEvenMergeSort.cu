/*
 * oddEvenMergeSort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "cuda_utils.h"


__global__ static void CUDA_OddEvenMergeSortShared(Element* __restrict__ values,
                                                   const int32_t N)
{
    const int32_t idx = TDIM * BID * 2 + TID;
    extern __shared__ Element s_v[];

    s_v[TID] = values[idx];
    s_v[TID+TDIM] = values[idx+TDIM];
    __syncthreads();

    for (int32_t size = 2; size <= N; size <<= 1)
    {
        int32_t stride = size / 2;
        int32_t offset = threadIdx.x & (stride - 1);

        {
            __syncthreads();
            int32_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            if(s_v[pos].k > s_v[pos + stride].k) {
                Element tmp = s_v[pos];
                s_v[pos] = s_v[pos + stride];
                s_v[pos + stride] = tmp;
            }
            stride >>= 1;
        }

        for (; stride > 0; stride >>= 1)
        {
            __syncthreads();
            int32_t pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));

            if (offset >= stride)
                if(s_v[pos - stride].k > s_v[pos].k) {
                    Element tmp = s_v[pos - stride];
                    s_v[pos - stride] = s_v[pos];
                    s_v[pos] = tmp;
                }
        }
    }

    __syncthreads();

    values[idx] = s_v[TID];
    values[idx+TDIM] = s_v[TID+TDIM];
}

__global__ static void CUDA_OddEvenMergeSortGlobal(Element* __restrict__ values,
                                                   const int32_t size,
                                                   const int32_t stride)
{
    const int32_t idx = TDIM * BID + TID;

    //Odd-even merge
    int32_t pos = 2 * idx - (idx & (stride - 1));

    if (stride < size / 2)
    {
        int32_t offset = idx & ((size / 2) - 1);

        if (offset >= stride)
        {
            Element elA = values[pos - stride];
            Element elB = values[pos +      0];

            if(elA.k > elB.k) {
                values[pos - stride] = elB;
                values[pos +      0] = elA;
            }
        }
    }
    else
    {
        Element elA = values[pos +      0];
        Element elB = values[pos + stride];

        if(elA.k > elB.k) {
            values[pos +      0] = elB;
            values[pos + stride] = elA;
        }
    }
}

__host__ void inline OddEvenMergeSort(Element** d_mem_values,
                                      const int32_t N)
{
    kdim v = get_kdim(N>>1);

    if (v.num_blocks == 1)
    {
        CUDA_OddEvenMergeSortShared<<<v.dim_blocks, v.num_threads, N*sizeof(Element)>>>(*d_mem_values, N);
        cudaDeviceSynchronize();
    }
    else {
        CUDA_OddEvenMergeSortShared<<<v.dim_blocks, v.num_threads, v.num_threads*2*sizeof(Element)>>>(*d_mem_values, v.num_threads*2);
        cudaDeviceSynchronize();

        for (int32_t size = 2 * v.num_threads; size <= N; size <<= 1)
        {
            for (int32_t stride = size / 2; stride > 0; stride >>= 1)
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

    h_mem = malloc(MAX_SIZE);
    assert(h_mem != NULL);
    gpuErrchk( cudaMalloc(&d_mem_values, MAX_SIZE) );

    srand(time(NULL));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        init_values((Element*) h_mem, N);

        copy_to_device_time(d_mem_values, h_mem, size);
        cudaDeviceSynchronize();

        OddEvenMergeSort((Element**) &d_mem_values, N);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        copy_to_host_time(h_mem, d_mem_values, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((Element*) h_mem, N, false) ? "true":"false");
    }

    cudaFree(d_mem_values);
    free(h_mem);

    return 0;
}
