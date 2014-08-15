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
#include "gpuqsortlib/gpuqsort.cu"

// program main
int main(int argc, char** argv)
{
    void *h_mem;
    size_t min_size = 1024UL; //1kB
    size_t max_size = 1024UL*1024UL*256UL; //256MB

    h_mem = malloc(max_size);
    assert(h_mem != NULL);

    srand(time(NULL));

    for(size_t size = min_size; size <= max_size; size <<= 1) {
        size_t N = size/sizeof(int);
        init_values_int((int*) h_mem, N);

        gpuqsort((uint*) h_mem, N);
        cudaDeviceSynchronize();
        gpuErrchk( cudaPeekAtLastError() );

        printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
        print_int_array((int*) h_mem, N);
    }

    free(h_mem);

    return 0;
}
