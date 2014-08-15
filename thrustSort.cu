/*
 * thrust_radix_sort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#include <thrust/uninitialized_copy.h>
#include <thrust/device_malloc.h>
#include <thrust/sort.h>

int my_compare (const void * a, const void * b)
{
    int _a = *(int*)a;
    int _b = *(int*)b;
    if(_a < _b) return -1;
    else if(_a == _b) return 0;
    else return 1;
}

// program main
int main(int argc, char** argv)
{
    int *h_mem;
    size_t min_size = 1024UL; //1kB
    size_t max_size = 1024UL*1024UL*256UL; //256MB

    h_mem = (int*)malloc(max_size);
    assert(h_mem != NULL);
    thrust::device_ptr<int> d_mem = thrust::device_malloc<int>(max_size/sizeof(int));

    srand(time(NULL));

    for(size_t size = min_size; size <= max_size; size <<= 1) {
        size_t N = size/sizeof(int);
        init_values_int((int*) h_mem, N);

        //thrust::uninitialized_copy(h_mem, h_mem+N, d_mem);

        //thrust::sort(d_mem, d_mem+N);
        //thrust::stable_sort(d_mem, d_mem+N);

        //thrust::copy(d_mem, d_mem+N, h_mem);
        qsort(h_mem, N, sizeof(int), my_compare);

        printf("after %ld %s\n", N, is_int_array_sorted((int*) h_mem, N, false) ? "true":"false");
    }

    free(h_mem);

    return 0;
}
