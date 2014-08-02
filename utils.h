#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#define MAX_THREADS 512UL
#define MAX_DIM 32768UL
typedef unsigned int uint;
typedef unsigned short ushort;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert (%s:%d): %s\n", file, line, cudaGetErrorString(code));
      if (abort) exit(code);
   }
}

typedef struct kdim
{
    uint num_threads;
    uint num_blocks;
    dim3 dim_blocks;
} kdim;

struct counted_value {
    int v;
    int c;
};

kdim get_kdim(uint n) {
    kdim v;
    v.dim_blocks.x = 1;
    v.dim_blocks.y = 1;
    v.dim_blocks.z = 1;
    v.num_blocks = 1;

    if (n <= MAX_THREADS)
    {
        v.num_threads = n;
    }
    else if (n <= MAX_DIM * MAX_THREADS)
    {
        v.num_threads = MAX_THREADS;
        v.dim_blocks.x = n / MAX_THREADS;
        v.num_blocks = v.dim_blocks.x;
    }
    else
    {
        v.num_threads = MAX_THREADS;
        v.dim_blocks.x = MAX_DIM;
        v.dim_blocks.y = n / MAX_DIM / MAX_THREADS;
        v.num_blocks = v.dim_blocks.x * v.dim_blocks.y;
    }

    return v;
}

void init_values_int(int *values, size_t length) {
    for (size_t i = 0; i < length; ++i)
    {
        values[i] = rand();
    }
}

void init_values_int_sorted(int *values, size_t length, bool reverse) {
    for (size_t i = 0; i < length; ++i)
    {
        values[i] = !reverse ? i : length-i-1;
    }
}

void init_values_long(long *values, size_t length) {
    for (size_t i = 0; i < length; ++i)
    {
        values[i] = (long)rand() * (long)rand();
    }
}

void init_values_long_sorted(long *values, size_t length, bool reverse) {
    for (size_t i = 0; i < length; ++i)
    {
        values[i] = !reverse ? i : length-i-1;
    }
}

void init_values_counted(struct counted_value *values, int length, int max_value) {
    int counter = 0;

    for (int i = 0; i < length; ++i)
    {
        values[i].v = rand() % (max_value+1);
        values[i].c = counter++;
    }
}

/*void init_values_custom(void* values, int length, int item_size, int max_value) {
    for (int i = 0; i < length; ++i)
    {
        int *ptr = (int*)(values+(item_size*i));
        *ptr = rand() % (max_value+1);
    }
}*/

bool is_int_array_sorted(int *values, size_t length, bool reverse) {
    bool sorted = true;
    for (size_t i = 0; i < length-1; ++i)
    {
        sorted = sorted && (!reverse ? (values[i] <= values[i+1]) : (values[i] >= values[i+1]));
    }
    return sorted;
}

bool is_long_array_sorted(long *values, size_t length, bool reverse) {
    bool sorted = true;
    for (size_t i = 0; i < length-1; ++i)
    {
        sorted = sorted && (!reverse ? (values[i] <= values[i+1]) : (values[i] >= values[i+1]));
    }
    return sorted;
}

bool is_counted_array_sorted(struct counted_value *values, int length, bool reverse) {
    bool sorted = true;
    for (int i = 0; i < length-1; ++i)
    {
        sorted = sorted && (!reverse ? (values[i].v <= values[i+1].v) : (values[i].v >= values[i+1].v));
    }
    return sorted;
}

bool is_counted_array_sorted_stable(struct counted_value *values, int length, bool reverse) {
    bool stable = true;
    for (int i = 0; i < length-1; ++i)
    {
        if(values[i].v == values[i+1].v) {
            stable = stable && (!reverse ? (values[i].c <= values[i+1].c) : (values[i].c >= values[i+1].c));
        }
    }
    return stable;
}

/*bool is_custom_array_sorted(void* values, int length, int item_size, bool reverse) {
    bool sorted = true;
    for (int i = 0; i < length-1; ++i)
    {
        int *ptr_l = (int*)(values+(item_size*i));
        int *ptr_r = (int*)(values+(item_size*(i+1)));

        sorted = sorted && (!reverse ? (ptr_l <= ptr_r) : (ptr_l >= ptr_r));
    }
    return sorted;
}*/

clock_t copy_to_device_time(void *dst, const void *src, size_t size) {
    clock_t t1, t2;

    t1 = clock();
    gpuErrchk(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    t2 = clock();

    return t2 - t1;
}

clock_t copy_to_host_time(void *dst, const void *src, size_t size) {
    clock_t t1, t2;

    t1 = clock();
    gpuErrchk(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    t2 = clock();

    return t2 - t1;
}

void print_int_array(const int *a, size_t size) {
    for (size_t i = 0; i < size; ++i)
    {
        printf("%d\n", a[i]);
    }
}
