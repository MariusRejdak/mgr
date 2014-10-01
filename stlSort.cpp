/*
 * thrustSort.cu
 * Author: Marius Rejdak
 */

#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

extern "C" {
    #include "utils.h"
}

using namespace std;

int my_compare (const Element &a, const Element &b)
{
    return a.k < b.k;
}

// program main
int main(int argc, char** argv)
{
    void *h_mem;

    h_mem = malloc(MAX_SIZE);
    assert(h_mem != NULL);

    srand(time(NULL));

    printf("CPU STL sort\n");
    printf("%s,%s,%ld,%ld\n", "size", "time", CLOCKS_PER_SEC, sizeof(Element));

    for(int32_t size = MIN_SIZE; size <= MAX_SIZE; size <<= 1) {
        int32_t N = size/sizeof(Element);
        clock_t t_sum = 0;

        for (int i = 0; i < NUM_PASSES; ++i) {
            clock_t t1;
            init_values((Element*) h_mem, N);

            t1 = clock();
            sort((Element*)h_mem, (Element*)h_mem+N, my_compare);
            t_sum += clock() - t1;

            assert(is_int_array_sorted((Element*) h_mem, N, false));
        }
        t_sum /= NUM_PASSES;

        printf("%ld,%ld\n", N, t_sum);
    }

    free(h_mem);

    return 0;
}
