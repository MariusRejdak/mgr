/*
 * copy_times.cu
 * Author: Marius Rejdak
 */

#include "../utils.h"
#include "../cuda_utils.h"
#include <limits.h>

int main(int argc, char const *argv[])
{
    void *h_mem, *d_mem;

    h_mem = malloc(MAX_SIZE);
    assert(h_mem != NULL);
    assert(cudaMalloc(&d_mem, MAX_SIZE) == cudaSuccess);

    printf("Copy times\n");
    printf("%s,%s,%s,%ld\n", "size", "time_to_device", "time_to_host", CLOCKS_PER_SEC);

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        clock_t host_to_device = 0;
        clock_t device_to_host = 0;
        for (int i = 0; i < 100; ++i) {
            host_to_device += copy_to_device_time(d_mem, h_mem, size);
            device_to_host += copy_to_host_time(h_mem, d_mem, size);
        }
        host_to_device /= 100;
        device_to_host /= 100;

        printf("%ld,%ld,%ld\n", size, host_to_device, device_to_host);
    }

    cudaFree(d_mem);
    free(h_mem);

    return 0;
}
