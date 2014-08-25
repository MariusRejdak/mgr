/*
 * thrustSort.cu
 *
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

#include <thrust/uninitialized_copy.h>
#include <thrust/device_malloc.h>
#include <thrust/system/cuda/detail/detail/stable_radix_sort.h>
#include <thrust/execution_policy.h>

/*int my_compare (const void *a, const void *b)
{
    Key _a = ((Element*)a)->k;
    Key _b = ((Element*)b)->k;
    if(_a < _b) return -1;
    else if(_a == _b) return 0;
    else return 1;
}*/

template<typename T>
struct my_less {
    __host__ __device__
    bool operator()(const T &x, const T &y) const {
        return x < y;
    }
};

// program main
int main(int argc, char** argv)
{
    void *h_mem;

    h_mem = malloc(MAX_SIZE);
    assert(h_mem != NULL);
    thrust::device_ptr<Key> d_mem = thrust::device_malloc<Key>(MAX_SIZE/sizeof(Key));

    srand(time(NULL));

    printf("Radix sort (thrust)\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(Element));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        clock_t t_sum = 0;

        for (int i = 0; i < 100; ++i) {
            clock_t t1;
            init_values((Element*) h_mem, N);

            thrust::uninitialized_copy((Key*)h_mem, (Key*)h_mem+N, d_mem);

            //Radix sort
            t1 = clock();
            thrust::stable_sort(d_mem, d_mem+N);
            t_sum += clock() - t1;

            thrust::copy(d_mem, d_mem+N, (Key*)h_mem);

            assert(is_int_array_sorted((Element*) h_mem, N, false));
        }
        t_sum /= 100;

        printf("%ld,%ld\n", N, t_sum);
    }
    printf("\n");

    printf("Merge sort (thrust)\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(Element));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        clock_t t_sum = 0;

        for (int i = 0; i < 100; ++i) {
            clock_t t1;
            init_values((Element*) h_mem, N);

            thrust::uninitialized_copy((Key*)h_mem, (Key*)h_mem+N, d_mem);

            //Merge sort
            t1 = clock();
            thrust::stable_sort(d_mem, d_mem+N, my_less<Key>());
            t_sum += clock() - t1;

            thrust::copy(d_mem, d_mem+N, (Key*)h_mem);

            assert(is_int_array_sorted((Element*) h_mem, N, false));
        }
        t_sum /= 100;

        printf("%ld,%ld\n", N, t_sum);
    }

    free(h_mem);

    return 0;
}
