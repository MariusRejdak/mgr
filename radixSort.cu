/*
 * radixSort.cu
 * Author: Marius Rejdak
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "cuda_utils.h"


__global__ static void CUDA_RadixPrefixSum(Element* __restrict__ values,
                                           int32_t* __restrict__ values_masks,
                                           int32_t* __restrict__ aux,
                                           const Key mask)
{
    const int32_t idx = (TDIM * BID + TID) << 1;
    const int32_t tmp_in0 = (values[idx].k & mask) ? 0 : 1;
    const int32_t tmp_in1 = (values[idx + 1].k & mask) ? 0 : 1;

    extern __shared__ int32_t shared_int32[];

    shared_int32[TID] = tmp_in0 + tmp_in1;
    __syncthreads();

    for (int32_t i = 1; i < TDIM; i <<= 1) {
        const int32_t x = (i<<1)-1;
        if (TID >= i && (TID & x) == x) {
            shared_int32[TID] += shared_int32[TID - i];
        }
        __syncthreads();
    }

    if (TID == 0)
        shared_int32[TDIM - 1] = 0;
    __syncthreads();

    for (int32_t i = TDIM>>1; i >= 1; i >>= 1) {
        int32_t x = (i<<1)-1;
        if (TID >= i && (TID & x) == x) {
            int32_t temp = shared_int32[TID - i];
            shared_int32[TID - i] = shared_int32[TID];
            shared_int32[TID] += temp;
        }
        __syncthreads();
    }

    values_masks[idx] = shared_int32[TID];
    values_masks[idx + 1] = shared_int32[TID] + tmp_in0;

    if (TID == TDIM-1)
        aux[BID] = tmp_in0 + shared_int32[TID] + tmp_in1;
}

__global__ static void CUDA_RadixSort(Element* __restrict__ values,
                                      Element* __restrict__ values_sorted,
                                      int32_t* __restrict__ values_masks_psum,
                                      const Key mask)
{
    const int32_t idx = TDIM * BID + TID;
    const int32_t bdim = TDIM * BDIM;
    const Element current = values[idx];
    const int32_t new_idx = values_masks_psum[idx];

    if (current.k & mask)
        values_sorted[idx + (values_masks_psum[bdim-1] + ((values[bdim-1].k & mask) ? 0 : 1)) - new_idx] = current;
    else
        values_sorted[new_idx] = current;
}

__host__ void RadixPrefixSum(Element* d_mem_values, int32_t* d_mem_masks,
                             const int32_t N, const Key mask)
{
    int32_t *d_mem_aux;
    kdim v = get_kdim(N);

    gpuErrchk( cudaMalloc(&d_mem_aux, v.num_blocks * sizeof(int32_t)) );
    CUDA_RadixPrefixSum<<<v.dim_blocks, v.num_threads>>1, v.num_threads*sizeof(int32_t)>>>(d_mem_values, d_mem_masks, d_mem_aux, mask);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );

    if (v.num_blocks > 1) {
        SumScan_Inclusive(d_mem_aux, v.num_blocks);
        CUDA_SumScanUpdate<<<v.dim_blocks, v.num_threads>>>(d_mem_masks, d_mem_aux);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
    }

    cudaFree(d_mem_aux);
}

__host__ void inline RadixSort(Element* d_mem_values,
                               Element* d_mem_sorted,
                               const int32_t N)
{
    Element *d_v, *d_s;
    int32_t *d_m;
    kdim v = get_kdim(N);

    gpuErrchk( cudaMalloc(&d_m, N * sizeof(int32_t)) );

    for (int16_t bit = 0; bit < sizeof(Key)*8; ++bit) {
        Key mask = 1 << bit;

        if (bit % 2) {
            d_v = d_mem_values;
            d_s = d_mem_sorted;
        } else {
            d_v = d_mem_sorted;
            d_s = d_mem_values;
        }

        RadixPrefixSum(d_v, d_m, N, mask);

        CUDA_RadixSort<<<v.dim_blocks, v.num_threads>>>(d_v, d_s, d_m, mask);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );
    }

    cudaFree(d_m);
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

    printf("Radix sort\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(Element));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        clock_t t1, t2, t_sum = 0;

        for (int i = 0; i < NUM_PASSES; ++i) {
            init_values((Element*) h_mem, N);

            copy_to_device_time(d_mem_values, h_mem, size);
            cudaDeviceSynchronize();

            t1 = clock();
            RadixSort((Element*) d_mem_values, (Element*) d_mem_sorted, N);
            cudaDeviceSynchronize();
            t2 = clock();
            t_sum += t2 - t1;
            gpuErrchk( cudaPeekAtLastError() );

            copy_to_host_time(h_mem, d_mem_sorted, size);
            cudaDeviceSynchronize();

            assert(is_int_array_sorted((Element*) h_mem, N, false));
        }
        t_sum /= NUM_PASSES;

        printf("%ld,%ld\n", N, t_sum);
    }

    cudaFree(d_mem_values);
    cudaFree(d_mem_sorted);
    free(h_mem);

    return 0;
}
