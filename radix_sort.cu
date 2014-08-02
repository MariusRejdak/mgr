/*
 * bitonic_sort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"


__global__ static void CUDA_RadixMask(int* __restrict__ values,
                                      uint* __restrict__ values_masks,
                                      const int mask)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x

    register const uint idx = TDIM * (gridDim.x * blockIdx.y + blockIdx.x) + TID;

    values_masks[idx] = (values[idx] & mask) ? 0 : 1;

    #undef TID
    #undef TDIM
}

__global__ void CUDA_SumScan(uint* __restrict__ values,
                             uint* __restrict__ aux,
                             bool inclusive)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x
    #define BID (gridDim.x * blockIdx.y + blockIdx.x)
    #define BDIM (gridDim.x * gridDim.y)

    const uint idx = TDIM * BID + TID;
    uint tmp_in0 = values[idx*2];
    uint tmp_in1 = values[idx*2 + 1];

    __shared__ uint shared[MAX_THREADS];

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

    values[idx*2] = shared[TID] + (inclusive ? tmp_in0 : 0);
    values[idx*2 + 1] = shared[TID] + tmp_in0 + (inclusive ? tmp_in1 : 0);

    __syncthreads();

    if (TID == TDIM-1 && aux)
        aux[BID] = tmp_in0 + shared[TID] + tmp_in1;

    #undef TID
    #undef TDIM
    #undef BID
    #undef BDIM
}

__global__ void CUDA_SumScanUpdate(uint* __restrict__ values,
                                   uint* __restrict__ aux)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x

    const uint bid = gridDim.x * blockIdx.y + blockIdx.x;
    const uint idx = TDIM * bid + TID;
    __shared__ uint op;

    if (TID == 0 && bid > 0)
        op = aux[bid - 1];
    __syncthreads();

    if (bid > 0) {
        atomicAdd(values+idx, op);
    }

    #undef TID
    #undef TDIM
}

__global__ static void CUDA_RadixSort(int* __restrict__ values,
                                      int* __restrict__ values_sorted,
                                      uint* __restrict__ values_masks_psum,
                                      const int mask)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x
    #define BID (gridDim.x * blockIdx.y + blockIdx.x)
    #define BDIM (gridDim.x * gridDim.y * TDIM)

    const uint idx = TDIM * BID + TID;
    const uint current = values[idx];
    const uint current_masked = (current & mask) ? 0 : 1;
    uint new_id;

    __shared__ uint max_psum;

    if (TID == 0)
    {
        max_psum = values_masks_psum[BDIM-1] + ((values[BDIM-1] & mask) ? 0 : 1);
    }

    __syncthreads();

    new_id = current_masked ? values_masks_psum[idx] : (idx + max_psum - values_masks_psum[idx]);

    values_sorted[new_id] = current;

    #undef TID
    #undef TDIM
}

void SumScan_C(uint* d_mem_values,
               const uint N, bool inclusive)
{
    uint *d_mem_aux;
    kdim v = get_kdim(N);

    gpuErrchk( cudaMalloc(&d_mem_aux, v.num_blocks * sizeof(uint)) );
    CUDA_SumScan<<<v.dim_blocks, v.num_threads/2>>>(d_mem_values, d_mem_aux, inclusive);
    cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );

    if (v.num_blocks > 1) {
        SumScan_C(d_mem_aux, v.num_blocks, true);
        CUDA_SumScanUpdate<<<v.dim_blocks, v.num_threads>>>(d_mem_values, d_mem_aux);
        cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );
    }

    cudaFree(d_mem_aux);
}

void inline Radix_Sort_C(int* d_mem_values,
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

        CUDA_RadixMask<<<v.dim_blocks, v.num_threads>>>(d_v, d_m, mask);
        cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );

        SumScan_C(d_m, N, false);

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

        Radix_Sort_C((int*) d_mem_values, (int*) d_mem_sorted, N);
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
