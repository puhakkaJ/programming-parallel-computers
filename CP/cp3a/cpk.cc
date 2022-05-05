#include <cmath>
#include <iostream>
#include <vector>
#include <new>
#include <cstdlib>
#include <x86intrin.h>
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
static double4_t* double8_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

static inline double sum_all4(double4_t vv) {
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum += vv[i];
    }
    return sum;
}

static inline double4_t swap2(double4_t x) { return _mm256_permute2f128_pd(x, x, 0b00000001); }
// static inline double4_t swap2(double4_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 0b00000101); }

void correlate(const int ny, int nx, const float *data, float *result) {
    // block size nd
    constexpr int nd = 4;
    // how many blocks of rows
    int nc = (ny + nd - 1) / nd;
    // number of rows after padding
    int ncd = nc * nd;

    // elements per vector
    constexpr int nb = 4;
    // vectors per input row
    const int na = (nx + nb - 1) / nb;

    // input data, padded, converted to vectors
    double4_t* vd = double8_alloc(nx*na);
    std::vector<double> data_norm(nx*ny);

    #pragma omp parallel for schedule(static,1)
    for (int y = 0; y < ny; y++) {
        double sum = 0.0;
        double mean = 0.0;
        double pow_sum = 0.0;
        double sqrt_sum_sqrt = 0.0;

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

        for (int x = 0; x < nx; x++ ) {
            data_norm[x + y*nx] /= sqrt_sum_sqrt;
        }

        // NEW
        for (int ka = 0; ka < na; ka++) {
            for (int kb = 0; kb < nx; kb++) {
                for (int jb = 0; jb < nb; ++jb) {
                    int i = ka*nb + jb;
                    vd[ka+ na*y][kb] = i < nx ? data_norm[nx*y + i] : 0.0;
                }
            }
        }
    }

   
    // see Chapter 2: Case study
    #pragma omp parallel for schedule(static,1)
    for (int ia=0; ia<na; ia++) {
        for (int ja=ia; ja<na; ja++) {
            // init
            double4_t vv000 = {0.0}, vv001 = {0.0}, vv010 = {0.0},vv011 = {0.0};
            
            for(int k=0; k<nx; k++) {
                
                double4_t a000 = vd[nx*ia + k];
                double4_t b000 = vd[nx*ja + k];
                double4_t a100 = swap2(a000);
                double4_t b001 = swap1(b000);
                vv000 += a000 * b000;
                vv001 += a000 * b001;
                vv010 += a100 * b000;
                vv011 += a100 * b001;
            }

            vv010 = swap2(vv010);
            vv011 = swap2(vv011);
            double4_t vv[4] = { vv000, vv001, vv010, vv011};
             for(int tmp=0; tmp<4; tmp++){
                printf("dot[%d]: %lf, %lf, %lf, %lf\n", tmp, vv[tmp][0],vv[tmp][1], vv[tmp][2], vv[tmp][3]);
             }
            
            for (int ib = 0; ib < nb; ib++) {
                for (int jb = 0; jb < nb;jb++) {
                    int i = ib + ia*nb;
                    int j = jb + ja*nb;
                    if (i<=j && j < ny && i < ny) {
                        result[ny*i + j] = (double)vv[jb^ib][ib];
                    }
                }
            }
        }
    }
    std::free(vd);
}