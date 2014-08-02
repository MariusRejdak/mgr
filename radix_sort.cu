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

__global__ void CUDA_Sum(uint* __restrict__ values,
                         uint* __restrict__ aux)
{
    #define tid threadIdx.x
    #define tdim blockDim.x
    #define bid (gridDim.x * blockIdx.y + blockIdx.x)
    #define bdim (gridDim.x * gridDim.y)

    const uint idx = tdim * bid + tid;
    uint tmp_in0 = values[idx*2];
    uint tmp_in1 = values[idx*2 + 1];

    __shared__ uint shared[MAX_THREADS];

    shared[tid] = tmp_in0 + tmp_in1;
    __syncthreads();

    for (uint i = 1; i < tdim; i <<= 1)
    {
        const uint x = (i<<1)-1;
        if (tid >= i && (tid & x) == x)
        {
            shared[tid] += shared[tid - i];
        }
        __syncthreads();
    }

    if (tid == 0)
        shared[tdim - 1] = 0;
    __syncthreads();

    for (uint i = tdim>>1; i >= 1; i >>= 1)
    {
        uint x = (i<<1)-1;
        if (tid >= i && (tid & x) == x)
        {
            uint swp_tmp = shared[tid - i];
            shared[tid - i] = shared[tid];
            shared[tid] += swp_tmp;
        }
        __syncthreads();
    }

    values[idx*2] = tmp_in0 + shared[tid];
    values[idx*2 + 1] = tmp_in0 + tmp_in1 + shared[tid];

    __syncthreads();

    if (tid == tdim-1 && aux)
        aux[bid] = tmp_in0 + shared[tid] + tmp_in1;

    #undef tid
    #undef tdim
    #undef bid
    #undef bdim
}

__global__ void CUDA_PrefixSum(uint* __restrict__ values,
                               uint* __restrict__ aux)
{
    #define tid threadIdx.x
    #define tdim blockDim.x
    #define bid (gridDim.x * blockIdx.y + blockIdx.x)
    #define bdim (gridDim.x * gridDim.y)

    const uint idx = tdim * bid + tid;
    uint tmp_in0 = values[idx*2];
    uint tmp_in1 = values[idx*2 + 1];

    __shared__ uint shared[MAX_THREADS];

    shared[tid] = tmp_in0 + tmp_in1;
    __syncthreads();

    for (uint i = 1; i < tdim; i <<= 1)
    {
        const uint x = (i<<1)-1;
        if (tid >= i && (tid & x) == x)
        {
            shared[tid] += shared[tid - i];
        }
        __syncthreads();
    }

    if (tid == 0)
        shared[tdim - 1] = 0;
    __syncthreads();

    for (uint i = tdim>>1; i >= 1; i >>= 1)
    {
        uint x = (i<<1)-1;
        if (tid >= i && (tid & x) == x)
        {
            uint swp_tmp = shared[tid - i];
            shared[tid - i] = shared[tid];
            shared[tid] += swp_tmp;
        }
        __syncthreads();
    }

    values[idx*2] = shared[tid];
    values[idx*2 + 1] = tmp_in0 + shared[tid];

    __syncthreads();

    if (tid == tdim-1 && aux)
        aux[bid] = tmp_in0 + shared[tid] + tmp_in1;

    #undef tid
    #undef tdim
    #undef bid
    #undef bdim
}

__global__ void CUDA_PrefixSumUpdate(uint* __restrict__ values,
                                     uint* __restrict__ aux)
{
    #define tid threadIdx.x
    #define tdim blockDim.x

    const uint bid = gridDim.x * blockIdx.y + blockIdx.x;
    const uint idx = tdim * bid + tid;
    __shared__ uint op;

    if (tid == 0 && bid > 0)
        op = aux[bid - 1];
    __syncthreads();

    if (bid > 0) {
        atomicAdd(values+idx*2, op);
        atomicAdd(values+idx*2+1, op);
    }

    #undef tid
    #undef tdim
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
    const uint current_masked = current & mask ? 0 : 1;

    __shared__ uint max_psum;

    if (TID == 0)
        max_psum = values_masks_psum[BDIM-1] + (values[BDIM-1] & mask ? 0 : 1);
    __syncthreads();

    values_sorted[current_masked ? values_masks_psum[idx] : (idx + max_psum - values_masks_psum[idx])] = current;

    #undef TID
    #undef TDIM
}

void Sum_C(uint* d_mem_values,
                 const uint N)
{
    uint *d_mem_aux;
    kdim v = get_kdim(N>>1);

    //printf("blocks: x=%ld y=%ld, threads: %ld\n", v.dim_blocks.x, v.dim_blocks.y, v.num_threads);

    gpuErrchk( cudaMalloc(&d_mem_aux, v.num_blocks * sizeof(uint)) );
    CUDA_Sum<<<v.dim_blocks, v.num_threads>>>(d_mem_values, d_mem_aux);
    cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );

    if (v.num_blocks > 1) {
        Sum_C(d_mem_aux, v.num_blocks);
        CUDA_PrefixSumUpdate<<<v.dim_blocks, v.num_threads>>>(d_mem_values, d_mem_aux);
        cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );
    }

    cudaFree(d_mem_aux);
}

void PrefixSum_C(uint* d_mem_values,
                 const uint N)
{
    uint *d_mem_aux;
    kdim v = get_kdim(N>>1);

    //printf("blocks: x=%ld y=%ld, threads: %ld\n", v.dim_blocks.x, v.dim_blocks.y, v.num_threads);

    gpuErrchk( cudaMalloc(&d_mem_aux, v.num_blocks * sizeof(uint)) );
    CUDA_PrefixSum<<<v.dim_blocks, v.num_threads>>>(d_mem_values, d_mem_aux);
    cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );

    if (v.num_blocks > 1) {
        Sum_C(d_mem_aux, v.num_blocks);
        CUDA_PrefixSumUpdate<<<v.dim_blocks, v.num_threads>>>(d_mem_values, d_mem_aux);
        cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );
    }

    cudaFree(d_mem_aux);
}


void inline Radix_Sort_C(int* d_mem_values,
                         int* d_mem_sorted,
                         const uint N)
{
    uint *d_mem_masks;
    kdim v = get_kdim(N);

    gpuErrchk( cudaMalloc(&d_mem_masks, N * sizeof(uint)) );

    //printf("blocks: x=%ld y=%ld, threads: %ld\n", v.dim_blocks.x, v.dim_blocks.y, v.num_threads);

    for (short bit = 0; bit < sizeof(int)*8; ++bit)
    {
        int mask = 1 << bit;

        /*int *h_mem0 = (int*)malloc(N * sizeof(int));
        int *h_mem0x = (int*)malloc(N * sizeof(int));
        uint *h_mem1 = (uint*)malloc(N * sizeof(uint));
        uint *h_mem2 = (uint*)malloc(N * sizeof(uint));*/

        CUDA_RadixMask<<<v.dim_blocks, v.num_threads>>>(d_mem_values, d_mem_masks, mask);
        cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );

        //copy_to_host_time(h_mem0, d_mem_values, N * sizeof(int));
        //copy_to_host_time(h_mem1, d_mem_masks, N * sizeof(uint));

        PrefixSum_C(d_mem_masks, N);

        //copy_to_host_time(h_mem2, d_mem_masks, N * sizeof(uint));

        CUDA_RadixSort<<<v.dim_blocks, v.num_threads>>>(d_mem_values, d_mem_sorted, d_mem_masks, mask);
        cudaDeviceSynchronize(); gpuErrchk( cudaPeekAtLastError() );

        int *tmp = d_mem_values;
        d_mem_values = d_mem_sorted;
        d_mem_sorted = tmp;

        //copy_to_host_time(h_mem0x, d_mem_values, N * sizeof(int));


        //for (size_t i = 0; i < N; ++i)
        //{
            /*uint current = h_mem1[i];
            uint max_psum = h_mem2[N-1];

            uint n1 = h_mem2[i] - current;
            uint n0 = i + max_psum - h_mem2[i];
            printf("i:%d %d %d %d %d\n", i, h_mem0[i], h_mem0x[i], current, current ? n1 : n0);*/
            //printf("i:%d %d %d\n", i, h_mem1[i], h_mem2[i]);
        //}

        //free(h_mem1);
        //free(h_mem2);
        //exit(0);
    }

    cudaFree(d_mem_masks);
}

// program main
int main(int argc, char** argv) {
    void *h_mem, *d_mem_values, *d_mem_sorted;
    size_t min_size = 1024UL; //1kB
    size_t max_size = 1024UL*1024UL*256; //256MB

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
        gpuErrchk( cudaPeekAtLastError() );

        copy_to_host_time(h_mem, d_mem_sorted, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
        //print_int_array((int*) h_mem, N);
    }

    cudaFree(d_mem_values);
    cudaFree(d_mem_sorted);
    free(h_mem);

    return 0;
}
