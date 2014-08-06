/*
 * bitonic_sort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

// Kernel function
__global__ static void Quicksort(int* __restrict__ values,
                                 const uint N)
{
#define MAX_LEVELS	300

    int pivot, L, R;
    int idx =  threadIdx.x + blockIdx.x * blockDim.x;
    int start[MAX_LEVELS];
    int end[MAX_LEVELS];

    start[idx] = idx;
    end[idx] = N - 1;
    while (idx >= 0) {
        L = start[idx];
        R = end[idx];
        if (L < R) {
            pivot = values[L];
            while (L < R) {
                while (values[R] >= pivot && L < R)
                    R--;
                if(L < R)
                    values[L++] = values[R];
                while (values[L] < pivot && L < R)
                    L++;
                if (L < R)
                    values[R--] = values[L];
            }
            values[L] = pivot;
            start[idx + 1] = L + 1;
            end[idx + 1] = end[idx];
            end[idx++] = L;
            if (end[idx] - start[idx] > end[idx - 1] - start[idx - 1]) {
                // swap start[idx] and start[idx-1]
                int tmp = start[idx];
                start[idx] = start[idx - 1];
                start[idx - 1] = tmp;

                // swap end[idx] and end[idx-1]
                tmp = end[idx];
                end[idx] = end[idx - 1];
                end[idx - 1] = tmp;
            }

        } else
            idx--;
    }
}

// program main
int main(int argc, char** argv)
{
    void *h_mem, *d_mem;
    size_t min_size = 1024UL; //1kB
    size_t max_size = 1024UL*1024UL*256UL; //256MB

    h_mem = malloc(max_size);
    assert(h_mem != NULL);
    gpuErrchk(cudaMalloc(&d_mem, max_size));

    srand(time(NULL));

    for(size_t size = min_size; size <= max_size; size <<= 1) {
        size_t N = size/sizeof(int);
        init_values_int((int*) h_mem, N);

        copy_to_device_time(d_mem, h_mem, size);
        cudaDeviceSynchronize();

        Quicksort_C((int*) d_mem, N);
        gpuErrchk( cudaPeekAtLastError() );

        copy_to_host_time(h_mem, d_mem, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
    }

    cudaFree(d_mem);
    free(h_mem);

    return 0;
}
