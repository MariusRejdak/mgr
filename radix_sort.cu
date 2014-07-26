/*
 * bitonic_sort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define MAX_THREADS 320UL
#define MAX_DIM 65535UL

__global__ static void Radix_Sort(int* values, int* values_sorted, short bit, size_t N) {
    const size_t idx = gridDim.x * blockDim.x * blockIdx.y
                     + blockDim.x * blockIdx.x
                     + threadIdx.x;

    if (idx < N) {
        volatile size_t count;
        int mask = 1 << bit;

        if(values[idx] & mask) {
            count = idx+1;
            for (volatile size_t i = idx+1; i < N; ++i)
            {
                count += (values[i] & mask) ? 1 : 0;
            }
        }
        else {
            count = 0;
            for (volatile size_t i = 0; i < idx; ++i)
            {
                count += (values[i] & mask) ? 0 : 1;
            }
        }

        values_sorted[count] = values[idx];
    }
}

// program main
int main(int argc, char** argv) {
    void *h_mem, *d_mem_values, *d_mem_sorted;
    size_t min_size = 1024UL; //1kB
    size_t max_size = 1024UL*256UL; //1MB

    h_mem = malloc(max_size);
    assert(h_mem != NULL);
    assert(cudaMalloc(&d_mem_values, max_size) == cudaSuccess);
    assert(cudaMalloc(&d_mem_sorted, max_size) == cudaSuccess);

    //srand(time(NULL));

    for(size_t size = min_size; size <= max_size; size <<= 1) {
        size_t N = size/sizeof(int);
        size_t count = 0;
        init_values_int_sorted((int*) h_mem, N, true);

        copy_to_device_time(d_mem_values, h_mem, size);
        cudaDeviceSynchronize();

        for (short bit = 0; bit < sizeof(int)*8; bit+=2)
        {
            if (N <= MAX_THREADS) {
                Radix_Sort<<<1, N>>>((int*) d_mem_values, (int*) d_mem_sorted, bit, N);
                cudaDeviceSynchronize();
                Radix_Sort<<<1, N>>>((int*) d_mem_sorted, (int*) d_mem_values, bit+1, N);
                count += 2;
            }
            else if(N <= MAX_DIM*MAX_THREADS) {
                dim3 blocks(N/MAX_THREADS + 1);
                Radix_Sort<<<blocks, MAX_THREADS>>>((int*) d_mem_values, (int*) d_mem_sorted, bit, N);
                cudaDeviceSynchronize();
                Radix_Sort<<<blocks, MAX_THREADS>>>((int*) d_mem_sorted, (int*) d_mem_values, bit+1, N);
                count += 2;
            }
            else {
                dim3 blocks(MAX_DIM, N/MAX_THREADS/MAX_DIM + 1);
                Radix_Sort<<<blocks, MAX_THREADS>>>((int*) d_mem_values, (int*) d_mem_sorted, bit, N);
                cudaDeviceSynchronize();
                Radix_Sort<<<blocks, MAX_THREADS>>>((int*) d_mem_sorted, (int*) d_mem_values, bit+1, N);
                count += 2;
            }
            cudaDeviceSynchronize();
        }

        copy_to_host_time(h_mem, d_mem_sorted, size);
        cudaDeviceSynchronize();

        printf("after %ld %s %ld\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false", count);
    }

    cudaFree(d_mem_values);
    cudaFree(d_mem_sorted);
    free(h_mem);

    return 0;
}
