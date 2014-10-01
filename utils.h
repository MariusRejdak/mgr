/*
 * utils.h
 * Author: Marius Rejdak
 */

#ifndef UTILS_H
#define UTILS_H

#include <stdbool.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

typedef struct Element_key32 {
    int32_t k;
} Element_key32;

typedef struct Element_key64 {
    int64_t k;
} Element_key64;

typedef struct Element_pair32 {
    int32_t k;
    int32_t v;
} Element_pair32;

typedef struct Element_pair64 {
    int64_t k;
    int64_t v;
} Element_pair64;

typedef struct Element_pair64p {
    int64_t k;
    int64_t v1;
    int64_t v2;
    int64_t v3;
} Element_pair64p;

//typedef Element_key32 Element;
//typedef int32_t Key;

// Wybrane porcje danych do testów
// nie dotyczy thrustSort.cu
typedef Element_key64 Element;
typedef int64_t Key;

// Rozmiary do testów
#define MIN_SIZE 1024UL //1kB
#define MAX_SIZE 1024UL*1024UL*256UL //256MB

// Ilość powtórzeń
#define NUM_PASSES 100

void swap(void **lhs, void **rhs)
{
    void *tmp = *lhs;
    *lhs = *rhs;
    *rhs = tmp;
}

void init_values(Element *values, int32_t length)
{
    for (int32_t i = 0; i < length; ++i) {
        values[i].k = rand();
    }
}

void init_values_sorted(Element *values, int32_t length, bool reverse)
{
    for (int32_t i = 0; i < length; ++i) {
        values[i].k = !reverse ? i : length-i-1;
    }
}

bool is_int_array_sorted(Element *values, int32_t length, bool reverse)
{
    for (int32_t i = 0; i+1 < length-1; ++i) {
        if (!reverse ? (values[i].k > values[i+1].k) : (values[i].k < values[i+1].k)) {
            return false;
        }
    }
    return true;
}

void print_int_array(const Element *a, int32_t size)
{
    for (int32_t i = 0; i < size; ++i) {
        printf("%d\n", a[i].k);
    }
}

#endif /* UTILS_H */
