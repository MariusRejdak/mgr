/*
 *
 * radix_sort.cu
 *
 */

 #include <time.h>
 #include <stdio.h>
 #include <stdlib.h>

 #define MAX_THREADS	128
 #define N		513

 int* r_values;
 int* d_values;
 int* t_values;

 int* d_split;
 int* d_e;
 int* d_f;
 int* d_t;


 // Kernel function
 __global__ static void Radix_sort(int* values, int* temp, int loop, int* split, int* e, int* f, int* t) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int remainder[N], quotient[N];
	int f_count, totalFalses;

	if (idx < N) {
		// split based on least significant bit
		quotient[idx] = values[idx];
		for (int x = 0; x < loop + 1; ++x) {
			remainder[idx] = quotient[idx] % 10;
			quotient[idx] = quotient[idx] / 10;
		}

		// set e[idx] = 0 in each 1 input and e[idx] = 1 in each 0 input
		if (remainder[idx] == 1) {
			split[idx] = 1;
			e[idx] = 0;
		}
		else {
			split[idx] = 0;
			e[idx] = 1;
		}
	}
	__syncthreads();

	if (idx < N) {
		// scan the 1s
		f_count = 0;
		for (int x = 0; x < N; ++x) {
			f[x] = f_count;
			if (e[x] == 1)
				f_count++;
		}

		// calculate totalFalses
		totalFalses = e[N-1] + f[N-1];

		if (split[idx] == 1) {
			// t = idx - f + totalFalses
			t[idx] = idx - f[idx] + totalFalses;
		}
		else if (split[idx] == 0) {
			// t = f[idx]
			t[idx] = f[idx];
		}

		// Scatter input using t as scatter address
		temp[t[idx]] = values[idx];
	}
	__syncthreads();

	// copy new arrangement back to values
	if (idx < N) {
		values[idx] = temp[idx];
	}
}

 // program main
 int main(int argc, char** argv) {
	printf("./radix_sort starting with %d numbers...\n", N);
	size_t size = N * sizeof(int);

	// allocate host memory
	r_values = (int*)malloc(size);

	// allocate device memory
	cudaMalloc((void**)&d_values, size);
	cudaMalloc((void**)&t_values, size);
	cudaMalloc((void**)&d_split, size);
	cudaMalloc((void**)&d_e, size);
	cudaMalloc((void**)&d_f, size);
	cudaMalloc((void**)&d_t, size);

	/* Types of data sets to be sorted:
	 *	1. Normal distribution
	 *	2. Gaussian distribution
	 *	3. Bucket distribution
	 * 	4. Sorted distribution
	 *	5. Zero distribution
	 */

	for (int i = 0; i < 5; ++i) {
		// Initialize data set
		Init(r_values, i);

		// copy data to device
		cudaMemcpy(d_values, r_values, size, cudaMemcpyHostToDevice);

		printf("Beginning kernel execution...\n");

		cudaThreadSynchronize();

		// execute kernel
		for (int j = 0; j < 8; ++j) {
			Radix_sort <<< 1, N >>> (d_values, t_values, j, d_split, d_e, d_f, d_t);
		}

		cudaThreadSynchronize();

		// copy data back to host
		cudaMemcpy(r_values, t_values, size, cudaMemcpyDeviceToHost);


	// free memory
	cudaFree(d_values);
	cudaFree(t_values);
	cudaFree(d_split);
	cudaFree(d_e);
	cudaFree(d_f);
	cudaFree(d_t);
	free(r_values);

	cudaThreadExit();
 }
