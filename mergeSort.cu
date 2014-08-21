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

#define MERGE_SIZE 32
#define MERGE_SIZE_G 256

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
                if (++a < srcMergeSize) v_a = shared_a[a];
            } else {
                shared_out[a + b] = v_b;
                if (++b < srcMergeSize) v_b = shared_b[b];
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
                if (--a > 0) v_a = shared_a[a];
            } else {
                shared_out[a + b + 1] = v_b;
                if (--b > 0) v_b = shared_b[b];
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
    const int32_t srcMergeSize = 1 << iteration;
    int32_t idx = TDIM * BID + TID;
    Element* shared_a = shared_values;
    Element* shared_b = shared_values + srcMergeSize;
    Element* shared_out = shared_values + (srcMergeSize << 1);

    values += (idx & ~(srcMergeSize - 1)) << 1;
    idx &= srcMergeSize - 1;

    shared_a[TID] = values[idx];
    shared_b[TID] = values[idx + srcMergeSize];
    __syncthreads();

    if ((TID & MERGE_SIZE - 1) == 0) {
        int32_t a = TID & ~(MERGE_SIZE - 1);
        int32_t a_end = a + MERGE_SIZE;
        int32_t b = a;
        int32_t b_end = a_end;

        if (a > 0) {
            const Key a_min = shared_a[a].k;
            while (b > 0 && a_min <= shared_b[b].k) b -= MERGE_SIZE;
            while (b < TDIM-1 && a_min > shared_b[b].k) ++b;
        }
        if (a_end < TDIM) {
            const Key a_next_min = shared_a[a_end].k;
            while (b_end < TDIM && a_next_min > shared_b[b_end-1].k) b_end += MERGE_SIZE;
            while (b_end > 0 && a_next_min <= shared_b[b_end-1].k) --b_end;
        }

        Element v_a = shared_a[a];
        Element v_b = shared_b[b];

        if (a < a_end && b < b_end) {
            while (true) {
                if (v_a.k < v_b.k) {
                    shared_out[a + b] = v_a;
                    if (++a < a_end)
                        v_a = shared_a[a];
                    else
                        break;
                }
                else {
                    shared_out[a + b] = v_b;
                    if (++b < b_end)
                        v_b = shared_b[b];
                    else
                        break;
                }
            }
        }

        if (a < a_end) {
            while (true) {
                shared_out[a + b] = v_a;
                if (++a < a_end)
                    v_a = shared_a[a];
                else
                    break;
            }
        } else {
            while (true) {
                shared_out[a + b] = v_b;
                if (++b < b_end)
                    v_b = shared_b[b];
                else
                    break;
            }
        }
    }

    __syncthreads();
    values[idx] = shared_out[TID];
    values[idx + srcMergeSize] = shared_out[TID + srcMergeSize];
}

__global__ static void CUDA_MergeSortGlobal(Element* __restrict__ values,
                                            Element* __restrict__ values_sorted,
                                            const int32_t iteration,
                                            const int32_t N)
{
    const int32_t srcMergeSize = 1 << iteration;
    int32_t idx = TDIM * BID + TID;
    {
        const int32_t offset = (idx & ~(srcMergeSize - 1)) << 1;
        values += offset;
        values_sorted += offset;
    }
    idx &= srcMergeSize - 1;

    Element* values_b = values + srcMergeSize;

    int32_t a = idx & ~(MERGE_SIZE_G - 1);
    int32_t a_end = a + MERGE_SIZE_G;
    int32_t b = a;
    int32_t b_end = a_end;

    if (TID == 0) {
        if (a > 0) {
            const Key a_min = values[a].k;
            while (b > 0 && a_min <= values_b[b].k) b -= MERGE_SIZE_G;
            while (b < srcMergeSize-1 && a_min > values_b[b].k) ++b;
        }
        if (a_end < srcMergeSize) {
            const Key a_next_min = values[a_end].k;
            while (b_end < srcMergeSize && a_next_min > values_b[b_end-1].k) b_end += MERGE_SIZE_G;
            while (b_end > 0 && a_next_min <= values_b[b_end-1].k) --b_end;
        }

        Element v_a = values[a];
        Element v_b = values_b[b];

        if (a < a_end && b < b_end)
            while (true) {
                if (v_a.k < v_b.k) {
                    values_sorted[a + b] = v_a;
                    if (++a < a_end)
                        v_a = values[a];
                    else
                        break;
                }
                else {
                    values_sorted[a + b] = v_b;
                    if (++b < b_end)
                        v_b = values_b[b];
                    else
                        break;
                }
            }

        if (a < a_end) {
            while (true) {
                values_sorted[a + b] = v_a;
                if (++a < a_end)
                    v_a = values[a];
                else
                    break;
            }
        } else {
            while (true) {
                values_sorted[a + b] = v_b;
                if (++b < b_end)
                    v_b = values_b[b];
                else
                    break;
            }
        }
    }
}

__host__ void inline MergeSort(Element** d_mem_values,
                               Element** d_mem_sorted,
                               const int32_t N)
{
    for (int32_t i = 0; (1 << i) < N; ++i) {
        if (i <= 5) {
            kdim v = get_kdim_nt(N, (2 << i));
            CUDA_MergeSortSmall<<<v.dim_blocks, v.num_threads, v.num_threads*sizeof(Element) << 1>>>(*d_mem_values, i);
        } else if ((1 << i) <= MAX_THREADS) {
            kdim v = get_kdim_nt(N/2, (1 << i));
            CUDA_MergeSortShared<<<v.dim_blocks, v.num_threads, v.num_threads*sizeof(Element) << 2>>>(*d_mem_values, i);
        }
        else {
            kdim v = get_kdim_nt(N/2, MERGE_SIZE_G);
            CUDA_MergeSortGlobal<<<v.dim_blocks, v.num_threads>>>(*d_mem_values, *d_mem_sorted, i, N);
            swap((void**)d_mem_values, (void**)d_mem_sorted);
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
        //print_int_array((Element*) h_mem, N);
    }

    cudaFree(d_mem_values);
    cudaFree(d_mem_sorted);
    free(h_mem);

    return 0;
}
