/*
 * bitonic_sort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

typedef unsigned int uint;
typedef unsigned short ushort;

#define MAX_THREADS 512UL
#define MAX_DIM 32768UL

__global__ static void Radix_Sum(int* __restrict__ values,
                                 uint* __restrict__ values_sum0,
                                 uint* __restrict__ values_sum1,
                                 const int mask, const size_t N)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x

    register const uint bid = gridDim.x * blockIdx.y + blockIdx.x;
    register const uint idx = TDIM * bid + TID;

    __shared__ ushort masks[MAX_THREADS];


    if (idx < N)
        masks[TID] = values[idx] & mask;
    __syncthreads();

    if (idx < N)
    {
        if (TID == TDIM-1 || idx == N-1)
        {
            register uint count0 = 0;
            register uint count1 = 0;

            for (register uint i = 0; i <= TID; ++i)
            {
                if (masks[i])
                    count1 += 1;
                else
                    count0 += 1;
            }

            values_sum0[bid] = count0;
            values_sum1[bid] = count1;
        }
    }

    #undef TID
    #undef TDIM
}

__global__ static void Sum_PreScan(uint* __restrict__ values_sum,
                                    uint* __restrict__ blocks_sum,
                                    const uint N)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x
    register const uint bid = gridDim.x * blockIdx.y + blockIdx.x;
    register const uint idx = TDIM * bid + TID;

    __shared__ uint sums[MAX_THREADS];

    if (idx < N)
        sums[TID] = values_sum[idx];
    __syncthreads();

    if (idx < N)
    {
        if (TID == TDIM-1 || idx == N-1)
        {
            register uint count = 0;
            for (register uint i = 0; i <= TID; ++i)
                count += sums[i];

            blocks_sum[bid] = count;
        }

    }

    #undef TID
    #undef TDIM
}

__global__ static void Sum_Scan(uint* __restrict__ values_sum,
                                const uint N)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x

    __shared__ uint sums[MAX_THREADS];

    if (TID < N)
        sums[TID] = values_sum[TID];
    __syncthreads();

    if (TID < N)
    {
        register uint count = 0;

        for (register uint i = 0; i <= TID; ++i)
            count += sums[i];

        __syncthreads();

        values_sum[TID] = count;
    }

    #undef TID
    #undef TDIM
}

__global__ static void Sum_PostScan(uint* __restrict__ values_sum,
                                    uint* __restrict__ blocks_sum,
                                    const uint N)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x
    register const uint bid = gridDim.x * blockIdx.y + blockIdx.x;
    register const uint idx = TDIM * bid + TID;
    register uint count;

    __shared__ uint sums[MAX_THREADS];

    if (idx < N)
        sums[TID] = values_sum[idx];
    __syncthreads();

    if (idx < N)
    {
        if (TID < TDIM/2)
        {
            count = bid > 0 ? blocks_sum[bid-1] : 0;
            for (register uint i = 0; i <= TID; ++i)
                count += sums[i];
        }
        else
        {
            count = blocks_sum[bid];
            for (register uint i = TID+1; i < TDIM; ++i)
                count -= sums[i];
        }
    }
    __syncthreads();

    if (idx < N)
        values_sum[idx] = count;

    #undef TID
    #undef TDIM
}

__global__ static void Radix_Sort(int* __restrict__ values,
                                  int* __restrict__ values_sorted,
                                  uint* __restrict__ values_sum0,
                                  uint* __restrict__ values_sum1,
                                  const int mask, const uint N)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x
    #define BDIM gridDim.x * blockDim.y

    register const uint bid = gridDim.x * blockIdx.y + blockIdx.x;
    register const uint idx = TDIM * bid + TID;

    __shared__ ushort masks[MAX_THREADS];

    if (idx < N)
        masks[TID] = values[idx] & mask;
    __syncthreads();

    if (idx < N)
    {
        register uint count;
        register uint current;
        current = masks[TID];

        count = current ? values_sum0[BDIM-1] : 0;

        if (TID < TDIM/2)
        {
            count += bid > 0 ? values_sum0[bid-1] : 0;
            for (register uint i = 0; i < TID; ++i)
                count += current ? (masks[i] ? 1 : 0) : (masks[i] ? 0 : 1);
        }
        else
        {
            count += values_sum0[bid];
            for (register uint i = TID; i < TDIM; ++i)
                count -= current ? (masks[i] ? 1 : 0) : (masks[i] ? 0 : 1);
        }

        values_sorted[idx] = count;
    }

    #undef TID
    #undef TDIM
    #undef BDIM
}

void Radix_Sort_C(int* d_mem_values,
                  int* d_mem_sorted,
                  const int mask, const uint N)
{
    uint *d_mem_sum0, *d_mem_sum1;
    gpuErrchk( cudaMalloc(&d_mem_sum0, N*sizeof(uint)/MAX_THREADS) );
    gpuErrchk( cudaMalloc(&d_mem_sum1, N*sizeof(uint)/MAX_THREADS) );

    if (N <= MAX_THREADS) {
        Radix_Sum<<<1, N>>>(d_mem_values, d_mem_sum0,  d_mem_sum1, mask, N);
        cudaDeviceSynchronize();

        Radix_Sort<<<1, N>>>(d_mem_values, d_mem_sorted, d_mem_sum0, d_mem_sum1, mask, N);
        cudaDeviceSynchronize();
    }
    else
    {
        uint *d_mem_blocks_sum0, *d_mem_blocks_sum1;
        dim3 blocks(1,1,1);
        dim3 blocks2(1,1,1);
        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        gpuErrchk( cudaMalloc(&d_mem_blocks_sum0, N*sizeof(uint)/MAX_THREADS/MAX_THREADS) );
        gpuErrchk( cudaMalloc(&d_mem_blocks_sum1, N*sizeof(uint)/MAX_THREADS/MAX_THREADS) );

        if(N <= MAX_DIM*MAX_THREADS)
        {
            blocks.x = N/MAX_THREADS + 1;
            blocks2.x = N/MAX_THREADS/MAX_THREADS + 1;
        }
        else
        {
            blocks.x = MAX_DIM;
            blocks.y = N/MAX_THREADS/MAX_DIM;
            blocks2.x = N/MAX_THREADS/MAX_THREADS;
        }

        Radix_Sum<<<blocks, MAX_THREADS>>>(d_mem_values, d_mem_sum0, d_mem_sum1, mask, N);
        cudaDeviceSynchronize();

        if(N < MAX_THREADS*MAX_THREADS)
        {
            Sum_Scan<<<1, blocks.x, 0, stream1>>>(d_mem_sum0, blocks.x);
            Sum_Scan<<<1, blocks.x, 0, stream2>>>(d_mem_sum1, blocks.x);
            cudaDeviceSynchronize();
        }
        else
        {
            Sum_PreScan<<<blocks2, MAX_THREADS, 0, stream1>>>(d_mem_sum0, d_mem_blocks_sum0, blocks.x*blocks.y);
            Sum_PreScan<<<blocks2, MAX_THREADS, 0, stream2>>>(d_mem_sum1, d_mem_blocks_sum1, blocks.x*blocks.y);
            cudaDeviceSynchronize();

            Sum_Scan<<<1, blocks2.x, 0, stream1>>>(d_mem_blocks_sum0, blocks2.x);
            Sum_Scan<<<1, blocks2.x, 0, stream2>>>(d_mem_blocks_sum1, blocks2.x);
            cudaDeviceSynchronize();

            Sum_PostScan<<<blocks2, MAX_THREADS, 0, stream1>>>(d_mem_sum0, d_mem_blocks_sum0, blocks.x*blocks.y);
            Sum_PostScan<<<blocks2, MAX_THREADS, 0, stream2>>>(d_mem_sum1, d_mem_blocks_sum1, blocks.x*blocks.y);
            cudaDeviceSynchronize();
        }

        Radix_Sort<<<blocks, MAX_THREADS>>>(d_mem_values, d_mem_sorted, d_mem_sum0, d_mem_sum1, mask, N);
        cudaDeviceSynchronize();

        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
        cudaFree(d_mem_blocks_sum0);
        cudaFree(d_mem_blocks_sum1);
    }
    cudaFree(d_mem_sum0);
    cudaFree(d_mem_sum1);
}

// program main
int main(int argc, char** argv) {
    void *h_mem, *d_mem_values, *d_mem_sorted;

    size_t min_size = 1024UL;
    size_t max_size = 1024UL*1024UL*256UL;

    h_mem = malloc(max_size);
    assert(h_mem != NULL);
    gpuErrchk( cudaMalloc(&d_mem_values, max_size) );
    gpuErrchk( cudaMalloc(&d_mem_sorted, max_size) );

    //srand(time(NULL));

    for(size_t size = min_size; size <= max_size; size <<= 1) {
        size_t N = size/sizeof(int);
        init_values_int((int*) h_mem, N);

        copy_to_device_time(d_mem_values, h_mem, size);
        cudaDeviceSynchronize();

        for (short bit = 0; bit < sizeof(int)*8; ++bit)
        {
            Radix_Sort_C((int*) d_mem_values, (int*) d_mem_sorted, 1 << bit, N);
            gpuErrchk( cudaPeekAtLastError() );
        }

        copy_to_host_time(h_mem, d_mem_sorted, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
    }

    cudaFree(d_mem_values);
    cudaFree(d_mem_sorted);
    free(h_mem);

    return 0;
}
