/*
 * quickSort.cu
 * Author: Marius Rejdak
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

    h_mem = malloc(MAX_SIZE);
    assert(h_mem != NULL);

    srand(time(NULL));

    printf("Quick sort\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(Element));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        clock_t t_sum = 0;

        for (int i = 0; i < NUM_PASSES; ++i) {
            clock_t t;
            init_values((Element*) h_mem, N);

            gpuqsort((unsigned int*) h_mem, N, &t);
            gpuErrchk( cudaPeekAtLastError() );

            assert(is_int_array_sorted((Element*) h_mem, N, false));
            t_sum += t;
        }
        t_sum /= NUM_PASSES;

        printf("%ld,%ld\n", N, t_sum);
    }

    free(h_mem);

    return 0;
}
