#include<cmath>
#include <iostream>
#include <vector>
using namespace std;

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

// straigth copy from course material
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

constexpr float param_zero = 0.0;
constexpr double4_t d4_init {
    param_zero, param_zero, param_zero, param_zero
};

// straigth copy from course material
static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

static inline double sum_all4(double4_t res) {
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum += res[i];
    }
    return sum;
}

void correlate(const int ny, int nx, const float *data, float *result) {
    // elements per vector
    constexpr int nb = 4;
    // vectors per input row
    const int na = (nx + nb - 1) / nb;

    // input data, padded, converted to vectors
    double4_t* vd = double4_alloc(ny*na);
    std::vector<double> data_norm(nx*ny);

    for (int y = ny; y--;) {
        double sum = 0.0, mean = 0.0,pow_sum = 0.0, sqrt_sum_sqrt = 0.0;

        for (int x = nx; x--; ) {
            sum += data[x + y*nx];
        }

        mean = sum / nx;

        for (int x = nx; x--; ) {
            double normalized = data[x + y*nx] - mean;
            data_norm[x + y*nx] = normalized;
            pow_sum += normalized*normalized;
        }

        sqrt_sum_sqrt = sqrt(pow_sum);

        for (int x = nx; x--; ) {
            data_norm[x + y*nx] /= sqrt_sum_sqrt;
        }

        // NEW
        for (int ka = 0; ka < na; ka++) {
            for (int kb = 0; kb < nb; kb++) {
                int i = ka*nb + kb;
                vd[ka+ na*y][kb] = i < nx ? data_norm[nx*y + i] : 0.0;
            }
        }
    }

    for (int i = ny; i--; ) {
        for (int j = i+1; j--;){
            double4_t res = d4_init;
            for (int k = 0; k< na;k++){
                res += vd[k + i*na] * vd[k + j*na];
            }

            result[i + j*ny] = sum_all4(res);
        }
    }
    std::free(vd);
}

