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

__global__ static void Bitonic_Sort(int* values, size_t j, size_t k, size_t N) {
	const size_t idx = gridDim.x * blockDim.x * blockIdx.y
					 + blockDim.x * blockIdx.x
					 + threadIdx.x;

	if (idx < N) {
		size_t ixj = idx^j;
		if (ixj > idx) {
			if ((idx&k) == 0 && values[idx] > values[ixj]) {
				int tmp = values[idx];
				values[idx] = values[ixj];
				values[ixj] = tmp;
			}
			if ((idx&k) != 0 && values[idx] < values[ixj]) {
				int tmp = values[idx];
				values[idx] = values[ixj];
				values[ixj] = tmp;
			}
		}
	}
}

// program main
int main(int argc, char** argv) {
	void *h_mem, *d_mem;
	size_t min_size = 1024UL; //1kB
	size_t max_size = 1024UL*1024UL*512UL; //512MB

	h_mem = malloc(max_size);
	assert(h_mem != NULL);
	gpuErrchk(cudaMalloc(&d_mem, max_size));

	//srand(time(NULL));

	for(size_t size = min_size; size <= max_size; size <<= 1) {
		size_t N = size/sizeof(int);
		init_values_int((int*) h_mem, N);

		copy_to_device_time(d_mem, h_mem, size);
		cudaDeviceSynchronize();

		for (size_t k = 2; k <= N; k <<= 1) {
			for (size_t j = k >> 1; j > 0; j >>= 1) {
				if (N <= MAX_THREADS) {
					Bitonic_Sort<<<1, N>>>((int*) d_mem, j, k, N);
				}
				else if(N <= MAX_DIM*MAX_THREADS) {
					dim3 blocks(N/MAX_THREADS + 1);
					Bitonic_Sort<<<blocks, MAX_THREADS>>>((int*) d_mem, j, k, N);
				}
				else {
					dim3 blocks(MAX_DIM, N/MAX_THREADS/MAX_DIM + 1);
					Bitonic_Sort<<<blocks, MAX_THREADS>>>((int*) d_mem, j, k, N);
				}
				cudaDeviceSynchronize();
			}
		}

		copy_to_host_time(h_mem, d_mem, size);
		cudaDeviceSynchronize();

		printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
	}

	cudaFree(d_mem);
	free(h_mem);

	return 0;
}
