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


__global__ static void CUDA_MergeSortShared(int* __restrict__ values,
                                            int* __restrict__ values_sorted,
                                            const uint dstMergeSize)
{
    extern __shared__ int shared_values[];
    const uint srcMergeSize = dstMergeSize>>1;
    int* shared_a = shared_values;
    int* shared_b = shared_values+srcMergeSize;

    values_sorted += TDIM * BID;

    shared_values[TID] = values[TDIM * BID + TID];
    __syncthreads();

    if (TID == 0) {
        uint i = 0;
        uint a = 0;
        uint b = 0;

        while (i < dstMergeSize) {
            if (b >= srcMergeSize || (a < srcMergeSize && shared_a[a] < shared_b[b]))
                values_sorted[i++] = shared_a[a++];
            else
                values_sorted[i++] = shared_b[b++];
        }
    }
}

__global__ static void CUDA_MergeSortGlobal(int* __restrict__ values,
                                            int* __restrict__ values_sorted,
                                            const uint dstMergeSize)
{
    const uint srcMergeSize = dstMergeSize>>1;
    int *values_a = values + (dstMergeSize * BID);
    int *values_b = values + (dstMergeSize * BID) + srcMergeSize;
    values_sorted += dstMergeSize * BID;

    __shared__ bool merged;
    __shared__ bool shared_need_a;
    __shared__ bool shared_need_b;
    __shared__ uint shared_a_i;
    __shared__ uint shared_b_i;
    __shared__ int shared_a[MAX_THREADS];
    __shared__ int shared_b[MAX_THREADS];

    uint a_i, b_i, i;

    if (TID == 0) {
        a_i = 0;
        b_i = 0;
        i = 0;
        merged = false;
        shared_a_i = 0;
        shared_b_i = 0;
        shared_need_a = true;
        shared_need_b = true;
    }

    __syncthreads();

    while (!merged) {
        if (shared_need_a && shared_a_i+TID < srcMergeSize) {
            shared_a[TID] = values_a[TID+shared_a_i];
        }
        if (shared_need_b && shared_b_i+TID < srcMergeSize) {
            shared_b[TID] = values_b[TID+shared_b_i];
        }

        __syncthreads();
        if (TID == 0) {
            if (shared_need_a) {
                shared_need_a = false;
                shared_a_i += TDIM;
                a_i = 0;
            }
            if (shared_need_b) {
                shared_need_b = false;
                shared_b_i += TDIM;
                b_i = 0;
            }

            while (a_i < TDIM && b_i < TDIM) {
                if (shared_a[a_i] < shared_b[b_i]) {
                    values_sorted[i++] = shared_a[a_i++];
                }
                else {
                    values_sorted[i++] = shared_b[b_i++];
                }
            }

            if (shared_a_i >= srcMergeSize) {
                while (b_i < TDIM) {
                    values_sorted[i++] = shared_b[b_i++];
                }
            }
            if (shared_b_i >= srcMergeSize) {
                while (a_i < TDIM) {
                    values_sorted[i++] = shared_a[a_i++];
                }
            }

            if (a_i >= TDIM && shared_a_i < srcMergeSize) {
                shared_need_a = true;
            }
            if (b_i >= TDIM && shared_b_i < srcMergeSize) {
                shared_need_b = true;
            }

            if (i >= dstMergeSize)
                merged = true;
        }
        __syncthreads();
    }
}

__host__ void inline MergeSort(int** d_mem_values,
                               int** d_mem_sorted,
                               const uint N)
{
    for (uint i = 2; i <= N; i <<= 1) {

        if (i <= MAX_THREADS) {
            kdim v = get_kdim_nt(N, i);
            CUDA_MergeSortShared<<<v.dim_blocks, v.num_threads, v.num_threads*sizeof(int)>>>(*d_mem_values, *d_mem_sorted, i);
        }
        else {
            kdim v = get_kdim_b(N/i);
            CUDA_MergeSortGlobal<<<v.dim_blocks, v.num_threads>>>(*d_mem_values, *d_mem_sorted, i);
        }

        swap((void**)d_mem_values, (void**)d_mem_sorted);

        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
    }

    swap((void**)d_mem_values, (void**)d_mem_sorted);
}

// program main
int main(int argc, char** argv)
{
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

        MergeSort((int**) &d_mem_values, (int**) &d_mem_sorted, N);
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
