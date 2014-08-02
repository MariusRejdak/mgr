/*
 * bitonic_sort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "cuda_utils.h"


__global__ static void CUDA_BitonicSort(int* __restrict__ values,
                                        const uint j, const uint k)
{
	const uint idx = TDIM * BID + TID;
	const uint ixj = idx^j;

	if (ixj > idx) {
		const uint v_idx = values[idx];
		const uint v_ixj = values[ixj];

		if ((idx&k) ? (v_idx < v_ixj) : (v_idx > v_ixj))
		{
			values[idx] = v_ixj;
			values[ixj] = v_idx;
		}
	}
}

__host__ void inline BitonicSort(int* d_mem, const uint N)
{
	kdim v = get_kdim(N);

	for (uint k = 2; k <= N; k <<= 1) {
		for (uint j = k >> 1; j > 0; j >>= 1) {
			CUDA_BitonicSort<<<v.dim_blocks, v.num_threads>>>(d_mem, j, k);
			cudaDeviceSynchronize();
		}
	}
}

// program main
int main(int argc, char** argv) {
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

		BitonicSort((int*) d_mem, N);
		gpuErrchk( cudaPeekAtLastError() );

		copy_to_host_time(h_mem, d_mem, size);
		cudaDeviceSynchronize();

		printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
	}

	cudaFree(d_mem);
	free(h_mem);

	return 0;
}
