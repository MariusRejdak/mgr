/*
 * cuda_utils.h
 * Author: Marius Rejdak
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <math.h>
#include "utils.h"

#define TID threadIdx.x
#define TDIM blockDim.x
#define BID (gridDim.x * blockIdx.y + blockIdx.x)
#define BDIM (gridDim.x * gridDim.y)

#define MAX_THREADS 512
#define MAX_DIM 32768

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert (%s:%d): %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}

typedef struct kdim {
    size_t num_threads;
    size_t num_blocks;
    dim3 dim_blocks;
} kdim;

kdim get_kdim_b(size_t blocks, size_t threads) {
    kdim v;
    v.num_threads = threads;
    v.dim_blocks.y = 1;
    v.dim_blocks.z = 1;

    if (blocks <= MAX_DIM) {
        v.dim_blocks.x = blocks;
        v.num_blocks = blocks;
    } else {
        v.dim_blocks.x = MAX_DIM;
        v.dim_blocks.y = blocks / MAX_DIM;
        v.num_blocks = v.dim_blocks.x * v.dim_blocks.y;
    }

    return v;
}

kdim get_kdim_nt(size_t n, size_t max_threads)
{
    kdim v;
    v.dim_blocks.x = 1;
    v.dim_blocks.y = 1;
    v.dim_blocks.z = 1;
    v.num_blocks = 1;

    if (n <= max_threads) {
        v.num_threads = n;
    } else if (n <= MAX_DIM * max_threads) {
        v.num_threads = max_threads;
        v.dim_blocks.x = n / max_threads;
        v.num_blocks = v.dim_blocks.x;
    } else {
        v.num_threads = max_threads;
        v.dim_blocks.x = MAX_DIM;
        v.dim_blocks.y = n / MAX_DIM / max_threads;
        v.num_blocks = v.dim_blocks.x * v.dim_blocks.y;
    }

    return v;
}

kdim inline get_kdim(size_t n)
{
    return get_kdim_nt(n, MAX_THREADS);
}

__host__ clock_t copy_to_device_time(void *dst, const void *src, size_t size)
{
    clock_t t1, t2;

    t1 = clock();
    gpuErrchk(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
    t2 = clock();

    return t2 - t1;
}

__host__ clock_t copy_to_host_time(void *dst, const void *src, size_t size)
{
    clock_t t1, t2;

    t1 = clock();
    gpuErrchk(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    t2 = clock();

    return t2 - t1;
}

__global__ void CUDA_SumScan_Inclusive(int32_t* __restrict__ values,
                                       int32_t* __restrict__ aux)
{
    const int32_t idx = (TDIM * BID + TID) << 1;
    const int32_t tmp_in0 = values[idx];
    const int32_t tmp_in1 = values[idx + 1];

    extern __shared__ int32_t shared_int32[];

    shared_int32[TID] = tmp_in0 + tmp_in1;
    __syncthreads();

    for (int32_t i = 1; i < TDIM; i <<= 1) {
        const int32_t x = (i<<1)-1;
        if (TID >= i && (TID & x) == x) {
            shared_int32[TID] += shared_int32[TID - i];
        }
        __syncthreads();
    }

    if (TID == 0)
        shared_int32[TDIM - 1] = 0;
    __syncthreads();

    for (int32_t i = TDIM>>1; i >= 1; i >>= 1) {
        int32_t x = (i<<1)-1;
        if (TID >= i && (TID & x) == x) {
            int32_t temp = shared_int32[TID - i];
            shared_int32[TID - i] = shared_int32[TID];
            shared_int32[TID] += temp;
        }
        __syncthreads();
    }

    values[idx] = shared_int32[TID] + tmp_in0;
    values[idx + 1] = shared_int32[TID] + tmp_in0 + tmp_in1;

    if (TID == TDIM-1 && aux)
        aux[BID] = tmp_in0 + shared_int32[TID] + tmp_in1;
}

__global__ void CUDA_SumScan_Exclusive(int32_t* __restrict__ values,
                                       int32_t* __restrict__ aux)
{
    const int32_t idx = (TDIM * BID + TID) << 1;
    const int32_t tmp_in0 = values[idx];
    const int32_t tmp_in1 = values[idx + 1];

    extern __shared__ int32_t shared_int32[];

    shared_int32[TID] = tmp_in0 + tmp_in1;
    __syncthreads();

    for (int32_t i = 1; i < TDIM; i <<= 1) {
        const int32_t x = (i<<1)-1;
        if (TID >= i && (TID & x) == x) {
            shared_int32[TID] += shared_int32[TID - i];
        }
        __syncthreads();
    }

    if (TID == 0)
        shared_int32[TDIM - 1] = 0;
    __syncthreads();

    for (int32_t i = TDIM>>1; i >= 1; i >>= 1) {
        int32_t x = (i<<1)-1;
        if (TID >= i && (TID & x) == x) {
            int32_t temp = shared_int32[TID - i];
            shared_int32[TID - i] = shared_int32[TID];
            shared_int32[TID] += temp;
        }
        __syncthreads();
    }

    values[idx] = shared_int32[TID];
    values[idx + 1] = shared_int32[TID] + tmp_in0;

    if (TID == TDIM-1 && aux)
        aux[BID] = tmp_in0 + shared_int32[TID] + tmp_in1;
}

__global__ void CUDA_SumScanUpdate(int32_t* __restrict__ values,
                                   int32_t* __restrict__ aux)
{
    const int32_t bid = BID;

    if (bid > 0)
        values[TDIM * bid + TID] += aux[bid - 1];
}

__host__ void SumScan_Inclusive(int32_t* d_mem_values, const int32_t N)
{
    int32_t *d_mem_aux;
    kdim v = get_kdim(N);

    if (v.num_blocks > 1) {
        gpuErrchk( cudaMalloc(&d_mem_aux, v.num_blocks * sizeof(int32_t)) );
        CUDA_SumScan_Inclusive<<<v.dim_blocks, v.num_threads>>1, v.num_threads*sizeof(int32_t)>>>(d_mem_values, d_mem_aux);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        SumScan_Inclusive(d_mem_aux, v.num_blocks);
        CUDA_SumScanUpdate<<<v.dim_blocks, v.num_threads>>>(d_mem_values, d_mem_aux);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        cudaFree(d_mem_aux);
    } else {
        CUDA_SumScan_Inclusive<<<v.dim_blocks, v.num_threads>>1, v.num_threads*sizeof(int32_t)>>>(d_mem_values, 0);
    }
}

__host__ void SumScan_Exclusive(int32_t* d_mem_values, const int32_t N)
{
    int32_t *d_mem_aux;
    kdim v = get_kdim(N);

    if (v.num_blocks > 1) {
        gpuErrchk( cudaMalloc(&d_mem_aux, v.num_blocks * sizeof(int32_t)) );
        CUDA_SumScan_Exclusive<<<v.dim_blocks, v.num_threads/2>>>(d_mem_values, d_mem_aux);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        SumScan_Inclusive(d_mem_aux, v.num_blocks);
        CUDA_SumScanUpdate<<<v.dim_blocks, v.num_threads>>>(d_mem_values, d_mem_aux);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        cudaFree(d_mem_aux);
    } else {
        CUDA_SumScan_Exclusive<<<v.dim_blocks, v.num_threads/2>>>(d_mem_values, 0);
    }
}

#endif /* CUDA_UTILS_H */
