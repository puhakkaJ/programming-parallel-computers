#include <algorithm>
#include <iostream>
#include <omp.h>
using namespace std;

typedef unsigned long long data_t;

// reference from http://www1.chapman.edu/~radenski/research/papers/mergesort-pdpta11.pdf
void mergeSortRecursive(data_t *v, int left, int right, int threads) {
    if (right - left < 2) {
        return;
	}
    else if (threads == 1) {
        sort(v + left, v + right);
    }
    else {
        int mid = (left + right) / 2; 
        int mid1 = (left + mid) / 2; 
        int mid2 = (mid + right) / 2; 
        #pragma omp taskgroup
        {
            #pragma omp task
            mergeSortRecursive(v, left, mid1, threads/2);
                    
            #pragma omp task
            mergeSortRecursive(v, mid1, mid, threads/2);

            #pragma omp task
            mergeSortRecursive(v, mid, mid2, threads - threads/2);
                    
            #pragma omp task
            mergeSortRecursive(v, mid2, right, threads - threads/2);
        }
        #pragma omp taskgroup
        {
            #pragma omp task
            inplace_merge(v + left, v + mid1, v + mid);   
            #pragma omp task 
            inplace_merge(v + mid, v + mid2, v + right);   
            
        }
        inplace_merge(v + left, v + mid, v + right);  
    } 
}

// reference from http://www1.chapman.edu/~radenski/research/papers/mergesort-pdpta11.pdf
void psort(int n, data_t *data) {
    // FIXME: Implement a more efficient parallel sorting algorithm for the CPU,
    // using the basic idea of merge sort.
    #pragma omp parallel
    #pragma omp single
    {
        mergeSortRecursive(data, 0, n, omp_get_max_threads()); 
    }
}