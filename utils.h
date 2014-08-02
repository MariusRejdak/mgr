#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

typedef unsigned int uint;
typedef unsigned short ushort;

struct counted_value {
    int v;
    int c;
};

void init_values_int(int *values, size_t length)
{
    for (size_t i = 0; i < length; ++i) {
        values[i] = rand();
    }
}

void init_values_int_sorted(int *values, size_t length, bool reverse)
{
    for (size_t i = 0; i < length; ++i) {
        values[i] = !reverse ? i : length-i-1;
    }
}

void init_values_long(long *values, size_t length)
{
    for (size_t i = 0; i < length; ++i) {
        values[i] = (long)rand() * (long)rand();
    }
}

void init_values_long_sorted(long *values, size_t length, bool reverse)
{
    for (size_t i = 0; i < length; ++i) {
        values[i] = !reverse ? i : length-i-1;
    }
}

void init_values_counted(struct counted_value *values, int length, int max_value)
{
    int counter = 0;

    for (int i = 0; i < length; ++i) {
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

bool is_int_array_sorted(int *values, size_t length, bool reverse)
{
    for (size_t i = 0; i+1 < length-1; ++i) {
        if (!reverse ? (values[i] > values[i+1]) : (values[i] < values[i+1])) {
            return false;
        }
    }
    return true;
}

bool is_long_array_sorted(long *values, size_t length, bool reverse)
{
    bool sorted = true;
    for (size_t i = 0; i < length-1; ++i) {
        sorted = sorted && (!reverse ? (values[i] <= values[i+1]) : (values[i] >= values[i+1]));
    }
    return sorted;
}

bool is_counted_array_sorted(struct counted_value *values, int length, bool reverse)
{
    bool sorted = true;
    for (int i = 0; i < length-1; ++i) {
        sorted = sorted && (!reverse ? (values[i].v <= values[i+1].v) : (values[i].v >= values[i+1].v));
    }
    return sorted;
}

bool is_counted_array_sorted_stable(struct counted_value *values, int length, bool reverse)
{
    bool stable = true;
    for (int i = 0; i < length-1; ++i) {
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

void print_int_array(const int *a, size_t size)
{
    for (size_t i = 0; i < size; ++i) {
        printf("%d\n", a[i]);
    }
}

#endif /* UTILS_H */
