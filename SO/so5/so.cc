#include <algorithm>
#include <iostream>
#include <omp.h>
#include <random>
#include <cstdlib>
using namespace std;

typedef unsigned long long data_t;

#define MIN_LEN 2200


void quickSortRecursive(data_t *left, data_t *right) {
    if (right - left < 2) {
        return;
	}
    else if (right - left  < MIN_LEN) {
        return std::sort(left, right);
    }
    else {
        //reference https://www.geeksforgeeks.org/quicksort-using-random-pivoting/
        srand(time(NULL));
        data_t pivot = *(left + rand() % (std::distance(left, right) - 1));
        //reference from https://en.cppreference.com/w/cpp/algorithm/partition
        data_t *middle1 = std::partition(left, right, [pivot](data_t em) -> bool { return em < pivot; });
        data_t *middle2 = std::partition(middle1, right, [pivot](data_t em) -> bool { return !(pivot < em); });

        #pragma omp taskgroup
        {
            #pragma omp task 
            quickSortRecursive(left, middle1);

            #pragma omp task 
            quickSortRecursive(middle2,right);
        }
    } 
}

void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of quicksort.
    data_t *end = data + n;
    #pragma omp parallel shared (data, end, n)
    #pragma omp single nowait
    {
        quickSortRecursive(data, end); 
    }
}
