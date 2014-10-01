/*
 * bitonicSort.cu
 * Author: Marius Rejdak
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "cuda_utils.h"

/*
 * Sortowanie bitoniczne w pamięci globalnej
 */
__global__ static void CUDA_BitonicSort_Global(
    Element* __restrict__ values, //Słowo kluczowe __restrict__ wyłącza optymalizacje kompilatora dla tej zmiennej
    const int32_t j,
    const int32_t k)
{
    // Aby zachować synchronizację kolejne kernele są wywoływane synchronicznie
    const int32_t idx = TDIM * BID + TID;
    const int32_t ixj = idx^j;

    if (ixj > idx) {
        // Zmienne lokalne (optymalizowane przez kompilator do rejestrów)
        // dla ograniczenia ilości odwołań do pamięci globalnej
        const Element v_idx = values[idx];
        const Element v_ixj = values[ixj];

        // Porównanie i zamiana
        if ((idx&k) ? (v_idx.k < v_ixj.k) : (v_idx.k > v_ixj.k)) {
            values[idx] = v_ixj;
            values[ixj] = v_idx;
        }
    }
}

/*
 * Sortowanie bitoniczne przeprowadzane w pamięci współdzielonej
 */
__global__ static void CUDA_BitonicSort_Shared(Element* __restrict__ values)
{
    extern __shared__ Element shared_values[];

    // Kopiowanie do pamięci współdzielonej
    shared_values[TID] = values[TID];

    // Oczekiwanie na zakończenie kopiowania
    __syncthreads();

    // Pętla sortująca
    for (int32_t k = 2; k <= TDIM; k <<= 1) {
        for (int32_t j = k >> 1; j > 0; j >>= 1) {
            const int32_t ixj = TID^j;

            if (ixj > TID) {
                // Zmienne lokalne (optymalizowane przez kompilator do rejestrów)
                // dla ograniczenia ilości odwołań do pamięci współdzielonej
                const Element v_idx = shared_values[TID];
                const Element v_ixj = shared_values[ixj];

                // Porównanie i zamiana
                if ((TID&k) ? (v_idx.k < v_ixj.k) : (v_idx.k > v_ixj.k)) {
                    shared_values[TID] = v_ixj;
                    shared_values[ixj] = v_idx;
                }
            }

            // Synchronizacja globalna
            __syncthreads();
        }
    }

    values[TID] = shared_values[TID];
}

/*
 * Wywołania funkcji kernel
 */
__host__ void inline BitonicSort(Element* d_mem, const int32_t N)
{
    kdim v = get_kdim(N);

    if (v.num_blocks == 1) {
        // Sortowanie można przeprowadzić w całości w pamięci współdzielonej
        CUDA_BitonicSort_Shared<<<v.dim_blocks, v.num_threads, v.num_threads * sizeof(Element)>>>(d_mem);
    } else {
        // Sortowanie należy przeprowadzić w pamięci globalnej
        for (int32_t k = 2; k <= N; k <<= 1) {
            for (int32_t j = k >> 1; j > 0; j >>= 1) {
                CUDA_BitonicSort_Global<<<v.dim_blocks, v.num_threads>>>(d_mem, j, k);
                cudaDeviceSynchronize();
            }
        }
    }
}

// program main
int main(int argc, char** argv)
{
    void *h_mem, *d_mem;

    h_mem = malloc(MAX_SIZE);
    assert(h_mem != NULL);
    gpuErrchk(cudaMalloc(&d_mem, MAX_SIZE));

    srand(time(NULL));

    printf("Bitonic sort\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(Element));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        clock_t t1, t2, t_sum = 0;

        for (int i = 0; i < NUM_PASSES; ++i) {
            init_values((Element*) h_mem, N);

            copy_to_device_time(d_mem, h_mem, size);
            cudaDeviceSynchronize();

            t1 = clock();
            BitonicSort((Element*) d_mem, N);
            cudaDeviceSynchronize();
            t2 = clock();
            t_sum += t2 - t1;
            gpuErrchk( cudaPeekAtLastError() );

            copy_to_host_time(h_mem, d_mem, size);
            cudaDeviceSynchronize();

            assert(is_int_array_sorted((Element*) h_mem, N, false));
        }
        t_sum /= NUM_PASSES;

        printf("%ld,%ld\n", N, t_sum);
    }

    cudaFree(d_mem);
    free(h_mem);

    return 0;
}
