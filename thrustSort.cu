/*
 * thrustSort.cu
 * Author: Marius Rejdak
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
    thrust::device_ptr<Key> d_mem_keys = thrust::device_malloc<Key>(MAX_SIZE/2/sizeof(Key));
    thrust::device_ptr<Key> d_mem_values = thrust::device_malloc<Key>(MAX_SIZE/2/sizeof(Key));

    srand(time(NULL));

    printf("Radix sort (thrust)\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(Element));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/2/sizeof(Key);
        clock_t t_sum = 0;

        for (int i = 0; i < NUM_PASSES; ++i) {
            clock_t t1;
            init_values((Element*) h_mem, N);

            thrust::uninitialized_copy((Key*)h_mem, (Key*)h_mem+N, d_mem_keys);
            thrust::uninitialized_copy((Key*)h_mem, (Key*)h_mem+N, d_mem_values);

            //Radix sort
            t1 = clock();
            thrust::stable_sort_by_key(d_mem_keys, d_mem_keys + N, d_mem_values);
            t_sum += clock() - t1;

            thrust::copy(d_mem_keys, d_mem_keys+N, (Key*)h_mem);

            assert(is_int_array_sorted((Element*) h_mem, N, false));
        }
        t_sum /= NUM_PASSES;

        printf("%ld,%ld\n", N, t_sum);
    }
    printf("\n");

    printf("Merge sort (thrust)\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(Element));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        clock_t t_sum = 0;

        for (int i = 0; i < NUM_PASSES; ++i) {
            clock_t t1;
            init_values((Element*) h_mem, N);

            thrust::uninitialized_copy((Key*)h_mem, (Key*)h_mem+N, d_mem_keys);
            thrust::uninitialized_copy((Key*)h_mem, (Key*)h_mem+N, d_mem_values);

            //Merge sort
            t1 = clock();
            thrust::stable_sort_by_key(d_mem_keys, d_mem_keys + N, d_mem_values, my_less<Key>());
            t_sum += clock() - t1;

            thrust::copy(d_mem_keys, d_mem_keys+N, (Key*)h_mem);

            assert(is_int_array_sorted((Element*) h_mem, N, false));
        }
        t_sum /= NUM_PASSES;

        printf("%ld,%ld\n", N, t_sum);
    }

    free(h_mem);

    return 0;
}
