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
#define MERGE_SIZE_G 1024

__global__ static void CUDA_MergeSortSmall(Element* __restrict__ values,
                                           Element* __restrict__ values_sorted,
                                           const int32_t iteration)
{
    const int32_t srcMergeSize = 1 << iteration;
    const int32_t dstMergeSize = srcMergeSize << 1;
    const int32_t idx = BID * dstMergeSize;

    values += idx;
    Element* __restrict__ values_a = values;
    Element* __restrict__ values_b = values + srcMergeSize;
    values_sorted += idx;

    int32_t a = 0;
    int32_t b = 0;
    Element v_a = values_a[a];
    Element v_b = values_b[b];

    while (a + b < dstMergeSize) {
        if (b >= srcMergeSize || (a < srcMergeSize && v_a.k < v_b.k)) {
            values_sorted[a + b] = v_a;
            if (++a < srcMergeSize) v_a = values_a[a];
        } else {
            values_sorted[a + b] = v_b;
            if (++b < srcMergeSize) v_b = values_b[b];
        }
    }
}


__global__ static void CUDA_MergeSortShared(Element* __restrict__ values,
                                            Element* __restrict__ values_sorted,
                                            const int32_t iteration,
                                            const int32_t merge_size)
{
    extern __shared__ Element shared_values[];
    const int32_t srcMergeSize = 1 << iteration;
    int32_t idx = (TDIM * BID + TID) * merge_size;

    {
        int32_t offset = (idx & ~(srcMergeSize - 1)) << 1;
        values_sorted += offset;
        values += offset;
        idx &= srcMergeSize - 1;
    }

    Element* __restrict__ shared_a = values + idx - TID*merge_size;
    Element* __restrict__ shared_b = shared_a + srcMergeSize;
    Element* __restrict__ shared_out = values_sorted + idx - TID*merge_size;

    int32_t a = TID*merge_size;
    int32_t a_end = a + merge_size;
    int32_t b = a;
    int32_t b_end = a_end;

    if (a > 0) {
        const Key a_min = shared_a[a].k;
        while (b > 0 && a_min <= shared_b[b].k) b -= merge_size;
        while (b < (TDIM*merge_size)-1 && a_min > shared_b[b].k) ++b;
    }
    if (a_end < TDIM*merge_size) {
        const Key a_next_min = shared_a[a_end].k;
        while (b_end < (TDIM*merge_size) && a_next_min > shared_b[b_end-1].k) b_end += merge_size;
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

__global__ static void CUDA_MergeSortGlobal(Element* __restrict__ values,
                                            Element* __restrict__ values_sorted,
                                            const int32_t iteration,
                                            const int32_t merge_size)
{
    const int32_t srcMergeSize = 1 << iteration;
    int32_t idx = BID * merge_size;
    {
        const int32_t offset = (idx & ~(srcMergeSize - 1)) << 1;
        values += offset;
        values_sorted += offset;
    }
    idx &= srcMergeSize - 1;

    Element* values_b = values + srcMergeSize;

    int32_t a = idx;
    int32_t a_end = a + merge_size;
    int32_t b = a;
    int32_t b_end = a_end;

    if (a > 0) {
        const Key a_min = values[a].k;
        while (b > 0 && a_min <= values_b[b].k) b -= merge_size;
        while (b < srcMergeSize-1 && a_min > values_b[b].k) ++b;
    }
    if (a_end < srcMergeSize) {
        const Key a_next_min = values[a_end].k;
        while (b_end < srcMergeSize && a_next_min > values_b[b_end-1].k) b_end += merge_size;
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

__host__ void inline MergeSort(Element** d_mem_values,
                               Element** d_mem_sorted,
                               const int32_t N)
{
    for (int32_t i = 0; (1 << i) < N; ++i) {
        if (i <= 5) {
            kdim v = get_kdim_b(N >> (i+1), 1);
            CUDA_MergeSortSmall<<<v.dim_blocks, v.num_threads>>>(*d_mem_values, *d_mem_sorted, i);
            swap((void**)d_mem_values, (void**)d_mem_sorted);
        } else if ((1 << i) <= MAX_THREADS) {
            kdim v = get_kdim_nt(N/2, (1 << i));
            CUDA_MergeSortShared<<<v.dim_blocks, v.num_threads / MERGE_SIZE>>>(*d_mem_values, *d_mem_sorted, i, MERGE_SIZE);
            swap((void**)d_mem_values, (void**)d_mem_sorted);
        }
        else {
            kdim v = get_kdim_b(N/MERGE_SIZE_G/2, 1);
            CUDA_MergeSortGlobal<<<v.dim_blocks, v.num_threads>>>(*d_mem_values, *d_mem_sorted, i, MERGE_SIZE_G);
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

    printf("Merge sort\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(Element));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        clock_t t1, t2, t_sum = 0;

        for (int i = 0; i < 100; ++i) {
            init_values((Element*) h_mem, N);

            copy_to_device_time(d_mem_values, h_mem, size);
            cudaDeviceSynchronize();

            t1 = clock();
            MergeSort((Element**) &d_mem_values, (Element**) &d_mem_sorted, N);
            cudaDeviceSynchronize();
            t2 = clock();
            t_sum += t2 - t1;
            gpuErrchk( cudaPeekAtLastError() );

            copy_to_host_time(h_mem, d_mem_sorted, size);
            cudaDeviceSynchronize();

            assert(is_int_array_sorted((Element*) h_mem, N, false));
        }
        t_sum /= 100;

        printf("%ld,%ld\n", N, t_sum);
    }

    cudaFree(d_mem_values);
    cudaFree(d_mem_sorted);
    free(h_mem);

    return 0;
}
