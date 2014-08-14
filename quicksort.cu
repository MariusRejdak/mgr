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

typedef struct QuickSort_args
{
    uint left;
    uint right;
} QuickSort_args;

__global__ static void CUDA_QuickSortGlobal(int* __restrict__ values,
                                            const uint left, const uint right,
                                            QuickSort_args* __restrict__ args_out)
{
    uint i = left;
    uint lt = left;
    uint gt = right;
    const int v = values[i];
    int v_i = v;

    while (i <= gt) {
        if (v_i < v) {
            values[i] = values[lt];
            values[lt++] = v_i;
            v_i = values[++i];

        } else if (v_i > v) {

            values[i] = values[gt];
            values[gt--] = v_i;
        } else {
            v_i = values[++i];
        }
    }

    args_out[0].left = left;
    args_out[0].right = lt > 0 ? lt - 1 : 0;
    args_out[1].left = gt + 1;
    args_out[1].right = right;
}

__global__ static void CUDA_QuickSortShared(int* __restrict__ values,
                                            const uint left, const uint right,
                                            QuickSort_args* __restrict__ args_out)
{
    uint i, lt, gt, v_i;
    extern __shared__ uint shared[];

    shared[TID+left] = values[TID+left];
    __syncthreads();

    if (TID == 0) {
        i = left;
        lt = left;
        gt = right;
        const int v = shared[i];
        v_i = v;

        while (i <= gt) {
            if (v_i < v) {
                shared[i] = shared[lt];
                shared[lt++] = v_i;
                v_i = shared[++i];

            } else if (v_i > v) {

                shared[i] = shared[gt];
                shared[gt--] = v_i;
            } else {
                v_i = shared[++i];
            }
        }
    }

    __syncthreads();
    values[TID+left] = shared[TID+left];

    if (TID == 0) {
        args_out[0].right = lt > 0 ? lt - 1 : 0;
        args_out[1].left = gt + 1;

    } else if (TID == TDIM - 1) {
        args_out[0].left = left;
        args_out[1].right = right;
    }
}

__host__ void inline QuickSort(int* d_mem_values,
                               const uint N)
{
    QuickSort_args *h_args_out, *d_args_out;
    cudaStream_t *streams;

    h_args_out = (QuickSort_args*) malloc(sizeof(QuickSort_args));
    h_args_out->left = 0;
    h_args_out->right = N-1;

    uint i = 1;
    uint sorted = 0;
    while(sorted != i) {
        sorted = 0;
        streams = (cudaStream_t*) malloc(sizeof(cudaStream_t) * i);
        gpuErrchk( cudaMalloc(&d_args_out, sizeof(QuickSort_args) * (i<<1)) );

        for (int j = 0; j < i; ++j) {
            kdim v = get_kdim(h_args_out[j].right - h_args_out[j].left + 1);
            cudaStreamCreate(streams+j);
            if (h_args_out[j].left < h_args_out[j].right) {
                if (N <= MAX_THREADS) {
                    CUDA_QuickSortShared<<<1, v.num_threads, v.num_threads*sizeof(int), streams[j]>>>(d_mem_values, h_args_out[j].left, h_args_out[j].right, d_args_out+(j<<1));
                } else {
                    CUDA_QuickSortGlobal<<<1, 1, 0, streams[j]>>>(d_mem_values, h_args_out[j].left, h_args_out[j].right, d_args_out+(j<<1));
                }
            }
        }


        QuickSort_args *h_args_out_new = (QuickSort_args*) malloc(sizeof(QuickSort_args) * (i<<1));

        for (int j = 0; j < i; ++j) {
            if (h_args_out[j].left < h_args_out[j].right) {
                cudaMemcpyAsync (h_args_out_new+(j<<1), d_args_out+(j<<1), sizeof(QuickSort_args) << 1, cudaMemcpyDeviceToHost, streams[j]);
            } else {
                h_args_out_new[(j<<1)].left = h_args_out[j].left;
                h_args_out_new[(j<<1)].right = h_args_out[j].right;
                h_args_out_new[(j<<1) + 1].left = h_args_out[j].left;
                h_args_out_new[(j<<1) + 1].right = h_args_out[j].right;
                ++sorted;
            }
        }

        free(h_args_out);
        h_args_out = h_args_out_new;

        cudaDeviceSynchronize();
        cudaFree(d_args_out);
        for (int j = 0; j < i; ++j)
            cudaStreamDestroy(streams[j]);
    }
    free(h_args_out);
}

// program main
int main(int argc, char** argv)
{
    void *h_mem, *d_mem_values;
    size_t min_size = 1024UL; //1kB
    size_t max_size = 1024UL*1024UL*256UL; //256MB

    h_mem = malloc(max_size);
    assert(h_mem != NULL);
    gpuErrchk( cudaMalloc(&d_mem_values, max_size) );

    srand(time(NULL));

    for(size_t size = min_size; size <= max_size; size <<= 1) {
        size_t N = size/sizeof(int);
        init_values_int((int*) h_mem, N);

        copy_to_device_time(d_mem_values, h_mem, size);
        cudaDeviceSynchronize();

        QuickSort((int*) d_mem_values, N);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        copy_to_host_time(h_mem, d_mem_values, size);
        cudaDeviceSynchronize();

        printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
    }

    cudaFree(d_mem_values);
    free(h_mem);

    return 0;
}
