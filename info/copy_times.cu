#include "utils.h"
#include <limits.h>

int main(int argc, char const *argv[])
{
    void *h_mem, *d_mem;
    int32_t min_size = 1024; //1kB
    int32_t max_size = 1024*1024*512; //512MB

    h_mem = malloc(max_size);
    assert(h_mem != NULL);
    assert(cudaMalloc(&d_mem, max_size) == cudaSuccess);

    printf("%s,%s\n", "size", "time", "type");

    for(int32_t size = min_size; size <= max_size; size <<= 1) {
        printf("%ld,%ld,%s\n", size, copy_to_device_time(d_mem, h_mem, size), "h->d");
        printf("%ld,%ld,%s\n", size, copy_to_host_time(h_mem, d_mem, size), "d->h");
    }

    cudaFree(d_mem);
    free(h_mem);

    return 0;
}
