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
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert (%s:%d): %s\n", file, line, cudaGetErrorString(code));
        if (abort) exit(code);
    }
}

typedef struct kdim {
    uint num_threads;
    uint num_blocks;
    dim3 dim_blocks;
} kdim;


kdim get_kdim(size_t n)
{
    kdim v;
    v.dim_blocks.x = 1;
    v.dim_blocks.y = 1;
    v.dim_blocks.z = 1;
    v.num_blocks = 1;

    if (n <= MAX_THREADS) {
        v.num_threads = n;
    } else if (n <= MAX_DIM * MAX_THREADS) {
        v.num_threads = MAX_THREADS;
        v.dim_blocks.x = n / MAX_THREADS;
        v.num_blocks = v.dim_blocks.x;
    } else {
        v.num_threads = MAX_THREADS;
        v.dim_blocks.x = MAX_DIM;
        v.dim_blocks.y = n / MAX_DIM / MAX_THREADS;
        v.num_blocks = v.dim_blocks.x * v.dim_blocks.y;
    }

    return v;
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

__global__ void CUDA_SumScan_Inclusive(uint* __restrict__ values,
                                       uint* __restrict__ aux)
{
    const uint idx = (TDIM * BID + TID) << 1;
    const uint tmp_in0 = values[idx];
    const uint tmp_in1 = values[idx + 1];

    extern __shared__ uint shared[];

    shared[TID] = tmp_in0 + tmp_in1;
    __syncthreads();

    for (uint i = 1; i < TDIM; i <<= 1) {
        const uint x = (i<<1)-1;
        if (TID >= i && (TID & x) == x) {
            shared[TID] += shared[TID - i];
        }
        __syncthreads();
    }

    if (TID == 0)
        shared[TDIM - 1] = 0;
    __syncthreads();

    for (uint i = TDIM>>1; i >= 1; i >>= 1) {
        uint x = (i<<1)-1;
        if (TID >= i && (TID & x) == x) {
            uint swp_tmp = shared[TID - i];
            shared[TID - i] = shared[TID];
            shared[TID] += swp_tmp;
        }
        __syncthreads();
    }

    values[idx] = shared[TID] + tmp_in0;
    values[idx + 1] = shared[TID] + tmp_in0 + tmp_in1;

    __syncthreads();

    if (TID == TDIM-1 && aux)
        aux[BID] = tmp_in0 + shared[TID] + tmp_in1;
}

__global__ void CUDA_SumScan_Exclusive(uint* __restrict__ values,
                                       uint* __restrict__ aux)
{
    const uint idx = (TDIM * BID + TID) << 1;
    const uint tmp_in0 = values[idx];
    const uint tmp_in1 = values[idx + 1];

    extern __shared__ uint shared[];

    shared[TID] = tmp_in0 + tmp_in1;
    __syncthreads();

    for (uint i = 1; i < TDIM; i <<= 1) {
        const uint x = (i<<1)-1;
        if (TID >= i && (TID & x) == x) {
            shared[TID] += shared[TID - i];
        }
        __syncthreads();
    }

    if (TID == 0)
        shared[TDIM - 1] = 0;
    __syncthreads();

    for (uint i = TDIM>>1; i >= 1; i >>= 1) {
        uint x = (i<<1)-1;
        if (TID >= i && (TID & x) == x) {
            uint swp_tmp = shared[TID - i];
            shared[TID - i] = shared[TID];
            shared[TID] += swp_tmp;
        }
        __syncthreads();
    }

    values[idx] = shared[TID];
    values[idx + 1] = shared[TID] + tmp_in0;

    __syncthreads();

    if (TID == TDIM-1 && aux)
        aux[BID] = tmp_in0 + shared[TID] + tmp_in1;
}

__global__ void CUDA_SumScanUpdate(uint* __restrict__ values,
                                   uint* __restrict__ aux)
{
    const uint bid = gridDim.x * blockIdx.y + blockIdx.x;

    if (bid > 0)
        values[TDIM * bid + TID] += aux[bid - 1];
}

__host__ void SumScan_Inclusive(uint* d_mem_values, const uint N)
{
    uint *d_mem_aux;
    kdim v = get_kdim(N);

    if (v.num_blocks > 1) {
        gpuErrchk( cudaMalloc(&d_mem_aux, v.num_blocks * sizeof(uint)) );
        CUDA_SumScan_Inclusive<<<v.dim_blocks, v.num_threads>>1, v.num_threads*sizeof(uint)>>>(d_mem_values, d_mem_aux);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        SumScan_Inclusive(d_mem_aux, v.num_blocks);
        CUDA_SumScanUpdate<<<v.dim_blocks, v.num_threads>>>(d_mem_values, d_mem_aux);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        cudaFree(d_mem_aux);
    } else {
        CUDA_SumScan_Inclusive<<<v.dim_blocks, v.num_threads>>1, v.num_threads*sizeof(uint)>>>(d_mem_values, 0);
    }
}

__host__ void SumScan_Exclusive(uint* d_mem_values, const uint N)
{
    uint *d_mem_aux;
    kdim v = get_kdim(N);

    if (v.num_blocks > 1) {
        gpuErrchk( cudaMalloc(&d_mem_aux, v.num_blocks * sizeof(uint)) );
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
