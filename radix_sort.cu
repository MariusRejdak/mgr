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

#define MAX_THREADS 1024UL
#define MAX_DIM 65535UL

__global__ static void Radix_Sum(int* values,
                                  uint* values_sum0,
                                  uint* values_sum1,
                                  const int mask, const size_t N)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x

    register const uint bid = gridDim.x * blockIdx.y + blockIdx.x;
    register const uint idx = TDIM * bid + TID;

    __shared__ ushort masks[MAX_THREADS];


    if (idx < N)
    {
        masks[TID] = values[idx] & mask;
        __syncthreads();

        if (TID == TDIM-1 || idx == N-1)
        {
            register uint count0 = 0;
            register uint count1 = 0;

            for (register uint i = 0; i <= TID; ++i)
            {
                if (masks[i])
                {
                    count1 += 1;
                }
                else
                {
                    count0 += 1;
                }
            }

            values_sum0[bid] = count0;
            values_sum1[bid] = count1;
        }
    }

    #undef TID
    #undef TDIM
}

__global__ static void Sum_PreScan(uint* values_sum,
                                    uint* blocks_sum,
                                    const uint N)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x
    register const uint bid = gridDim.x * blockIdx.y + blockIdx.x;
    register const uint idx = TDIM * bid + TID;

    __shared__ uint sums[MAX_THREADS];

    if (idx < N)
    {
        sums[TID] = values_sum[idx];
        __syncthreads();

        if (TID == TDIM-1 || idx == N-1)
        {
            register uint count = 0;
            for (register uint i = 0; i <= TID; ++i)
            {
                count += sums[i];
            }
            blocks_sum[bid] = count;
        }

    }

    #undef TID
    #undef TDIM
}

__global__ static void Sum_Scan(uint* values_sum,
                                 const uint N)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x

    __shared__ uint sums[MAX_THREADS];

    if (TID < N)
    {
        register uint count = 0;
        sums[TID] = values_sum[TID];
        __syncthreads();

        for (register uint i = 0; i <= TID; ++i)
        {
            count += sums[i];
        }
        __syncthreads();

        values_sum[TID] = count;
    }

    #undef TID
    #undef TDIM
}

__global__ static void Sum_PostScan(uint* values_sum,
                                     uint* blocks_sum,
                                     const uint N)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x
    register const uint bid = gridDim.x * blockIdx.y + blockIdx.x;
    register const uint idx = TDIM * bid + TID;
    register uint count;

    __shared__ uint sums[MAX_THREADS];

    if (idx < N)
    {
        sums[TID] = values_sum[TID];
        __syncthreads();

        if (TID < TDIM/2)
        {
            count = bid > 0 ? blocks_sum[bid-1] : 0;
            for (register uint i = 0; i <= TID; ++i)
            {
                count += sums[i];
            }
        }
        else
        {
            count = blocks_sum[bid];
            for (register uint i = TID+1; i < TDIM; ++i)
            {
                count -= sums[i];
            }
        }
        __syncthreads();

        values_sum[idx] = count;
    }

    #undef TID
    #undef TDIM
}

__global__ static void Radix_Sort(int* values,
                                  int* values_sorted,
                                  uint* values_sum0,
                                  uint* values_sum1,
                                  const int mask, const uint N)
{
    #define TID threadIdx.x
    #define TDIM blockDim.x
    #define BDIM gridDim.x * blockDim.y

    register const uint bid = gridDim.x * blockIdx.y + blockIdx.x;
    register const uint idx = TDIM * bid + TID;

    __shared__ ushort masks[MAX_THREADS];


    if (idx < N)
    {
        register uint count;
        register uint current;
        masks[TID] = values[idx] & mask;
        __syncthreads();
        current = masks[TID];

        if (current)
        {
            count = values_sum0[BDIM-1] + (bid > 0 ? values_sum1[bid-1] : 0);
        }
        else
        {
            count = bid > 0 ? values_sum0[bid-1] : 0;
        }

        for (register uint i = 0; i < TID; ++i)
        {
            count += current ? (masks[i] ? 1 : 0) : (masks[i] ? 0 : 1);
        }

        values_sorted[idx] = count;
    }

    #undef TID
    #undef TDIM
    #undef BDIM
}

// program main
int main(int argc, char** argv) {
    void *h_mem, *d_mem_values, *d_mem_sorted, *d_mem_psum0, *d_mem_psum1, *d_mem_blocks_psum;

    size_t min_size = 1024UL;
    size_t max_size = 1024UL*1024UL*256UL;
    //size_t min_size = 1024UL*1024UL*64UL;
    //size_t max_size = 1024UL*1024UL*64UL;

    h_mem = malloc(max_size);
    assert(h_mem != NULL);
    gpuErrchk( cudaMalloc(&d_mem_values, max_size) );
    gpuErrchk( cudaMalloc(&d_mem_sorted, max_size) );
    gpuErrchk( cudaMalloc(&d_mem_psum0, max_size/MAX_THREADS) );
    gpuErrchk( cudaMalloc(&d_mem_psum1, max_size/MAX_THREADS) );
    gpuErrchk( cudaMalloc(&d_mem_blocks_psum, max_size/MAX_THREADS/MAX_THREADS) );

    //srand(time(NULL));

    for(size_t size = min_size; size <= max_size; size <<= 1) {
        size_t N = size/sizeof(int);
        init_values_int((int*) h_mem, N);

        copy_to_device_time(d_mem_values, h_mem, size);
        cudaDeviceSynchronize();

        for (short bit = 0; bit < sizeof(int)*8; ++bit)
        {
            int mask = 1 << bit;

            if (N <= MAX_THREADS) {
                Radix_Sum<<<1, N>>>((int*) d_mem_values, (uint*) d_mem_psum0, (uint*) d_mem_psum1, mask, N); cudaDeviceSynchronize();
                Radix_Sort<<<1, N>>>((int*) d_mem_values, (int*) d_mem_sorted, (uint*) d_mem_psum0, (uint*) d_mem_psum1, mask, N); cudaDeviceSynchronize();
            }
            else if(N < MAX_DIM*MAX_THREADS) {
                dim3 blocks(N/MAX_THREADS + 1);

                Radix_Sum<<<blocks, MAX_THREADS>>>((int*) d_mem_values, (uint*) d_mem_psum0, (uint*) d_mem_psum1, mask, N);
                cudaDeviceSynchronize();

                if(N < MAX_THREADS*MAX_THREADS) {
                    Sum_Scan<<<1, blocks.x>>>((uint*) d_mem_psum0, blocks.x);
                    Sum_Scan<<<1, blocks.x>>>((uint*) d_mem_psum1, blocks.x);
                }
                else {
                    dim3 blocks2 (N/MAX_THREADS/MAX_THREADS + 1);
                    //printf("%ld %ld\n", blocks2.x, blocks.x);

                    Sum_PreScan<<<blocks2, MAX_THREADS>>>((uint*) d_mem_psum0, (uint*) d_mem_blocks_psum, blocks.x); cudaDeviceSynchronize();
                    Sum_Scan<<<1, blocks2.x>>>((uint*) d_mem_blocks_psum, blocks2.x); cudaDeviceSynchronize();
                    Sum_PostScan<<<blocks2, MAX_THREADS>>>((uint*) d_mem_psum0, (uint*) d_mem_blocks_psum, blocks.x); cudaDeviceSynchronize();

                    Sum_PreScan<<<blocks2, MAX_THREADS>>>((uint*) d_mem_psum1, (uint*) d_mem_blocks_psum, blocks.x); cudaDeviceSynchronize();
                    Sum_Scan<<<1, blocks2.x>>>((uint*) d_mem_blocks_psum, blocks2.x); cudaDeviceSynchronize();
                    Sum_PostScan<<<blocks2, MAX_THREADS>>>((uint*) d_mem_psum1, (uint*) d_mem_blocks_psum, blocks.x); cudaDeviceSynchronize();
                }

                Radix_Sort<<<blocks, MAX_THREADS>>>((int*) d_mem_values, (int*) d_mem_sorted, (uint*) d_mem_psum0, (uint*) d_mem_psum1, mask, N); cudaDeviceSynchronize();
            }
            else {
                dim3 blocks(MAX_DIM, N/MAX_THREADS/MAX_DIM);
                dim3 blocks2(N/MAX_THREADS/MAX_THREADS/MAX_DIM);
                //printf("nope %ld %ld\n", blocks.y, blocks2.x);

                /*Radix_Sum<<<blocks, MAX_THREADS>>>((int*) d_mem_values, (uint*) d_mem_psum0, (uint*) d_mem_psum1, mask, N);

                Sum_PreScan<<<blocks2, MAX_THREADS>>>((uint*) d_mem_psum0, (uint*) d_mem_blocks_psum, blocks.x*blocks.y+1);
                Sum_Scan<<<1, blocks2.x>>>((uint*) d_mem_blocks_psum, blocks2.x);
                Sum_PostScan<<<blocks2, MAX_THREADS>>>((uint*) d_mem_psum0, (uint*) d_mem_blocks_psum, blocks.x*blocks.y+1);

                Sum_PreScan<<<blocks2, MAX_THREADS>>>((uint*) d_mem_psum1, (uint*) d_mem_blocks_psum, blocks.x*blocks.y+1);
                Sum_Scan<<<1, blocks2.x>>>((uint*) d_mem_blocks_psum, blocks2.x);
                Sum_PostScan<<<blocks2, MAX_THREADS>>>((uint*) d_mem_psum1, (uint*) d_mem_blocks_psum, blocks.x*blocks.y+1);

                Radix_Sort<<<blocks, MAX_THREADS>>>((int*) d_mem_values, (int*) d_mem_sorted, (uint*) d_mem_psum0, (uint*) d_mem_psum1, mask, N);*/
                //printf("nope\n");
            }

            gpuErrchk( cudaPeekAtLastError() );
        }

        copy_to_host_time(h_mem, d_mem_sorted, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
    }

    cudaFree(d_mem_values);
    cudaFree(d_mem_sorted);
    cudaFree(d_mem_blocks_psum);
    cudaFree(d_mem_psum0);
    cudaFree(d_mem_psum1);
    free(h_mem);

    return 0;
}
