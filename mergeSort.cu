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
                                            const uint dstMergeSize)
{
    extern __shared__ int shared_values[];
    const uint idx = TDIM * BID + TID;
    const uint srcMergeSize = dstMergeSize>>1;
    int* shared_a = shared_values;
    int* shared_b = shared_values+srcMergeSize;
    int* shared_out = shared_values+dstMergeSize;

    shared_values[TID] = values[idx];
    __syncthreads();

    if (TID == 0) {
        uint i = 0;
        uint a = 0;
        uint b = 0;
        int v_a = shared_a[a];
        int v_b = shared_b[b];

        while (i < srcMergeSize) {
            if (b >= srcMergeSize || (a < srcMergeSize && v_a < v_b)) {
                shared_out[i++] = v_a;
                v_a = shared_a[++a];
            } else {
                shared_out[i++] = v_b;
                v_b = shared_b[++b];
            }
        }
    } else if (TID == TDIM-1) {
        uint i = dstMergeSize-1;
        uint a = srcMergeSize-1;
        uint b = srcMergeSize-1;
        int v_a = shared_a[a];
        int v_b = shared_b[b];

        while (i >= srcMergeSize) {
            if (b > srcMergeSize || (a < srcMergeSize && v_a >= v_b)) {
                shared_out[i--] = v_a;
                v_a = a > 0 ? shared_a[--a] : 0;
            } else {
                shared_out[i--] = v_b;
                v_b = b > 0 ? shared_b[--b] : 0;
            }
        }
    }

    __syncthreads();
    values[idx] = shared_out[TID];
}

__global__ static void CUDA_MergeSortGlobal(int* __restrict__ values,
                                            int* __restrict__ values_sorted,
                                            const uint iteration,
                                            const uint N)
{
    const uint lb_idx = TDIM * BID * 2;
    const uint idx = lb_idx + TID;

    const uint srcMergeSize = 1 << iteration; //2^iteration
    const uint dstMergeSize = srcMergeSize << 1; //2^(iteration+1)

    const uint lb_num = srcMergeSize / TDIM;

    const uint srcMergeIdA = (idx >> iteration) << iteration; // start, end = srcMergeIdA+srcMergeSize-1
    const uint srcMergeIdB = srcMergeIdA + srcMergeSize; // start, end = srcMergeIdB+srcMergeSize-1


}

__host__ void inline MergeSort(int** d_mem_values,
                               int** d_mem_sorted,
                               const uint N)
{
    for (uint i = 0; (1 << i) < N; ++i) {

        if ((2 << i) <= MAX_THREADS) {
            kdim v = get_kdim_nt(N, (2 << i));
            CUDA_MergeSortShared<<<v.dim_blocks, v.num_threads, v.num_threads*sizeof(int) << 1>>>(*d_mem_values, (2 << i));
        }
        else {
            //kdim v = get_kdim_b(N/i);
            CUDA_MergeSortGlobal<<<v.dim_blocks, v.num_threads>>>(*d_mem_values, *d_mem_sorted, i, N);
            //swap((void**)d_mem_values, (void**)d_mem_sorted);
        }

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
