/*
 * bitonic_sort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#define MAX_THREADS 1024UL
#define MAX_DIM 65535UL

__global__ static void Counting_Sort(int* values, int* values_sorted, size_t N) {
    const size_t idx = gridDim.x * blockDim.x * blockIdx.y
                     + blockDim.x * blockIdx.x
                     + threadIdx.x;

    if (idx < N) {
        volatile size_t count = 0;
        int current = values[idx];

        for (volatile size_t i = 0; i < idx; ++i)
        {
            count += (current >= values[i]) ? 1 : 0;
        }
        for (volatile size_t i = idx; i < N; ++i)
        {
            count += (current > values[i]) ? 1 : 0;
        }
        values_sorted[count] = current;
    }
}

// program main
int main(int argc, char** argv) {
    void *h_mem, *d_mem_values, *d_mem_sorted;
    size_t min_size = 1024UL; //1kB
    size_t max_size = 1024UL*1024UL*256UL; //512MB

    h_mem = malloc(max_size);
    assert(h_mem != NULL);
    assert(cudaMalloc(&d_mem_values, max_size) == cudaSuccess);
    assert(cudaMalloc(&d_mem_sorted, max_size) == cudaSuccess);

    //srand(time(NULL));

    for(size_t size = min_size; size <= max_size; size <<= 1) {
        size_t N = size/sizeof(int);
        init_values_int_sorted((int*) h_mem, N, true);

        copy_to_device_time(d_mem_values, h_mem, size);
        cudaDeviceSynchronize();

        if (N <= MAX_THREADS) {
            Counting_Sort<<<1, N>>>((int*) d_mem_values, (int*) d_mem_sorted, N);
        }
        else if(N <= MAX_DIM*MAX_THREADS) {
            dim3 blocks(N/MAX_THREADS + 1);
            Counting_Sort<<<blocks, MAX_THREADS>>>((int*) d_mem_values, (int*) d_mem_sorted, N);
        }
        else {
            dim3 blocks(MAX_DIM, N/MAX_THREADS/MAX_DIM + 1);
            Counting_Sort<<<blocks, MAX_THREADS>>>((int*) d_mem_values, (int*) d_mem_sorted, N);
        }
        cudaDeviceSynchronize();

        copy_to_host_time(h_mem, d_mem_sorted, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
    }

    cudaFree(d_mem_values);
    cudaFree(d_mem_sorted);
    free(h_mem);

    return 0;
}
