/*
 * mergeSort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "cuda_utils.h"

__global__ static void CUDA_MergeSortSmall(Element* __restrict__ values,
                                           const int32_t iteration)
{
    extern __shared__ Element shared_values[];
    const int32_t idx = TDIM * BID + TID;
    const int32_t srcMergeSize = 1 << iteration;
    Element* shared_a = shared_values;
    Element* shared_b = shared_values + srcMergeSize;
    Element* shared_out = shared_values + (srcMergeSize << 1);

    shared_values[TID] = values[idx];
    __syncthreads();

    if (TID == 0) {
        int32_t a = 0;
        int32_t b = 0;
        Element v_a = shared_a[a];
        Element v_b = shared_b[b];

        while (a + b < srcMergeSize) {
            if (b >= srcMergeSize || (a < srcMergeSize && v_a.k < v_b.k)) {
                shared_out[a + b] = v_a;
                v_a = shared_a[++a];
            } else {
                shared_out[a + b] = v_b;
                v_b = shared_b[++b];
            }
        }
    } else if (TID == TDIM-1) {
        int32_t a = srcMergeSize - 1;
        int32_t b = srcMergeSize - 1;
        Element v_a = shared_a[a];
        Element v_b = shared_b[b];

        while (a + b + 1 >= srcMergeSize) {
            if (b < 0 || (a >= 0 && v_a.k >= v_b.k)) {
                shared_out[a + b + 1] = v_a;
                v_a = --a > 0 ? shared_a[a] : v_a;
            } else {
                shared_out[a + b + 1] = v_b;
                v_b = --b > 0 ? shared_b[b] : v_b;
            }
        }
    }

    __syncthreads();
    values[idx] = shared_out[TID];
}


__global__ static void CUDA_MergeSortShared(Element* __restrict__ values,
                                            const int32_t iteration)
{
    extern __shared__ Element shared_values[];
    const int32_t idx = TDIM * BID + TID;
    Element* shared_a = shared_values;
    Element* shared_b = shared_values + (1 << iteration);
    Element* shared_out = shared_values + (2 << iteration);

    shared_a[TID] = values[idx];
    shared_b[TID] = values[idx + (1 << iteration)];
    __syncthreads();

    int32_t a = TID & ~(32 - 1);
    int32_t a_end = a + 32;
    int32_t b = a;
    int32_t b_end = a_end;

    if (a > 0) {
        const Key a_min = shared_a[a].k;
        while (b > 0 && a_min <= shared_b[b].k) b -= 32;
        while (b < TDIM-1 && a_min > shared_b[b].k) ++b;
    }
    if (a_end < TDIM) {
        const Key a_next_min = shared_a[a_end].k;
        while (b_end < TDIM && a_next_min > shared_b[b_end-1].k) b_end += 32;
        while (b_end > 0 && a_next_min <= shared_b[b_end-1].k) --b_end;
    }

    __syncthreads();

    if ((TID & 32 - 1) == 0) {
        Element v_a = shared_a[a];
        Element v_b = shared_b[b];

        while (a < a_end || b < b_end) {
            if (b >= b_end || (a < a_end && v_a.k < v_b.k)) {
                shared_out[a + b] = v_a;
                v_a = shared_a[++a];
            } else {
                shared_out[a + b] = v_b;
                v_b = shared_b[++b];
            }
        }
    }

    __syncthreads();
    values[idx << 1] = shared_out[TID << 1];
    values[(idx << 1) + 1] = shared_out[(TID << 1) + 1];
}

__global__ static void CUDA_MergeSortGlobal(Element* __restrict__ values,
                                            Element* __restrict__ values_sorted,
                                            const int32_t iteration,
                                            const int32_t N)
{
    /*extern __shared__ Element shared_values[];
    const int32_t idx = TDIM * BID + TID;

    values += idx*/
}

__host__ void inline MergeSort(Element** d_mem_values,
                               Element** d_mem_sorted,
                               const int32_t N)
{
    for (int32_t i = 0; (1 << i) < N; ++i) {

        if (i <= 6) {
            kdim v = get_kdim_nt(N, (2 << i));
            CUDA_MergeSortSmall<<<v.dim_blocks, v.num_threads, v.num_threads*sizeof(Element) << 1>>>(*d_mem_values, i);
        } else if ((1 << i) <= MAX_THREADS) {
            kdim v = get_kdim_nt(N/2, (1 << i));
            CUDA_MergeSortShared<<<v.dim_blocks, v.num_threads, v.num_threads*sizeof(Element) << 2>>>(*d_mem_values, i);
        }
        else {
            /*kdim v = get_kdim_b(N/i);
            CUDA_MergeSortGlobal<<<v.dim_blocks, v.num_threads>>>(*d_mem_values, *d_mem_sorted, i, N);
            swap((void**)d_mem_values, (void**)d_mem_sorted);*/
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

    h_mem = malloc(MAX_SIZE);
    assert(h_mem != NULL);
    gpuErrchk( cudaMalloc(&d_mem_values, MAX_SIZE) );
    gpuErrchk( cudaMalloc(&d_mem_sorted, MAX_SIZE) );

    srand(time(NULL));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        init_values((Element*) h_mem, N);

        copy_to_device_time(d_mem_values, h_mem, size);
        cudaDeviceSynchronize();

        MergeSort((Element**) &d_mem_values, (Element**) &d_mem_sorted, N);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        copy_to_host_time(h_mem, d_mem_sorted, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((Element*) h_mem, N, false) ? "true":"false");
    }

    cudaFree(d_mem_values);
    cudaFree(d_mem_sorted);
    free(h_mem);

    return 0;
}
