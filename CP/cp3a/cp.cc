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

static inline double sum_all4(double4_t vv) {
    double sum = 0.0;
    for (int i = 0; i < 4; ++i) {
        sum += vv[i];
    }
    return sum;
}

void correlate(const int ny, int nx, const float *data, float *result) {
    // block size nd
    constexpr int nd = 8;
    // how many blocks of rows
    int nc = (ny + nd - 1) / nd;
    // number of rows after padding
    int ncd = nc * nd;

    // elements per vector
    constexpr int nb = 4;
    // vectors per input row
    const int na = (nx + nb - 1) / nb;

    // input data, padded, converted to vectors
    double4_t* vd = double4_alloc(ncd*na);
    std::vector<double> data_norm(nx*ny);

    #pragma omp parallel for schedule(dynamic,1)
    for (int y = 0; y < ny; ++y) {
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

        for (int x = nx; x--; ) {
            data_norm[x + y*nx] /= sqrt_sum_sqrt;
        }

        // NEW
        for (int ka = 0; ka < na; ++ka) {
            for (int kb = 0; kb < nb; kb++) {
                int i = ka*nb + kb;
                vd[ka+ na*y][kb] = i < nx ? data_norm[nx*y + i] : 0.0;
            }
        }
    }

   
    // see Chapter 2: Case study
    #pragma omp parallel for schedule(dynamic,1)
    for (int row0 = 0; row0 < ncd-7; row0 += 8) {
        int row1 = row0+1;
        int row2 = row0+2;
        int row3 = row0+3;
        int row4 = row0+4;
        int row5 = row0+5;
        int row6 = row0+6;
        int row7 = row0+7;
        for (int row8 = row0; row8 < ncd-7; row8 += 8){
            int row9 = row8+1;
            int row10 = row8+2;
            int row11 = row8+3;
            int row12 = row8+4;
            int row13 = row8+5;
            int row14 = row8+6;
            int row15 = row8+6;
            int row16 = row8+7;
            
            double4_t vv[nd][nd];

            // init
            for (int id = 0; id < nd; ++id) {
                for (int jd = 0; jd < nd; ++jd) {
                    vv[id][jd] = d4_init;
                }
            }
        
            for (int ka = 0; ka < na; ++ka) {
                double4_t x0 = vd[ka + row0 * na];
                double4_t x1 = vd[ka + row1 * na];
                double4_t x2 = vd[ka + row2 * na];
                double4_t x3 = vd[ka + row3 * na];
                double4_t x4 = vd[ka + row4 * na];
                double4_t x5 = vd[ka + row5 * na];
                double4_t x6 = vd[ka + row6 * na];
                double4_t x7 = vd[ka + row7 * na];

                double4_t y0 = vd[ka + row8 * na];
                double4_t y1 = vd[ka + row9 * na];
                double4_t y2 = vd[ka + row10 * na];
                double4_t y3 = vd[ka + row11 * na];
                double4_t y4 = vd[ka + row12 * na];
                double4_t y5 = vd[ka + row13 * na];
                double4_t y6 = vd[ka + row14 * na];
                double4_t y7 = vd[ka + row16 * na];

                vv[0][0] += x0 * y0;
                vv[0][1] += x0 * y1;
                vv[0][2] += x0 * y2;
                vv[0][3] += x0 * y3;
                vv[0][4] += x0 * y4;
                vv[0][5] += x0 * y5;
                vv[0][6] += x0 * y6;
                vv[0][7] += x0 * y7;

                vv[1][0] += x1 * y0;
                vv[1][1] += x1 * y1;
                vv[1][2] += x1 * y2;
                vv[1][3] += x1 * y3;
                vv[1][4] += x1 * y4;
                vv[1][5] += x1 * y5;
                vv[1][6] += x1 * y6;
                vv[1][7] += x1 * y7;

                vv[2][0] += x2 * y0;
                vv[2][1] += x2 * y1;
                vv[2][2] += x2 * y2;
                vv[2][3] += x2 * y3;
                vv[2][4] += x2 * y4;
                vv[2][5] += x2 * y5;
                vv[2][6] += x2 * y6;
                vv[2][7] += x2 * y7;

                vv[3][0] += x3 * y0;
                vv[3][1] += x3 * y1;
                vv[3][2] += x3 * y2;
                vv[3][3] += x3 * y3;
                vv[3][4] += x3 * y4;
                vv[3][5] += x3 * y5;
                vv[3][6] += x3 * y6;
                vv[3][7] += x3 * y7;

                vv[4][0] += x4 * y0;
                vv[4][1] += x4 * y1;
                vv[4][2] += x4 * y2;
                vv[4][3] += x4 * y3;
                vv[4][4] += x4 * y4;
                vv[4][5] += x4 * y5;
                vv[4][6] += x4 * y6;
                vv[4][7] += x4 * y7;

                vv[5][0] += x5 * y0;
                vv[5][1] += x5 * y1;
                vv[5][2] += x5 * y2;
                vv[5][3] += x5 * y3;
                vv[5][4] += x5 * y4;
                vv[5][5] += x5 * y5;
                vv[5][6] += x5 * y6;
                vv[5][7] += x5 * y7;

                vv[6][0] += x6 * y0;
                vv[6][1] += x6 * y1;
                vv[6][2] += x6 * y2;
                vv[6][3] += x6 * y3;
                vv[6][4] += x6 * y4;
                vv[6][5] += x6 * y5;
                vv[6][6] += x6 * y6;
                vv[6][7] += x6 * y7;

                vv[7][0] += x7 * y0;
                vv[7][1] += x7 * y1;
                vv[7][2] += x7 * y2;
                vv[7][3] += x7 * y3;
                vv[7][4] += x7 * y4;
                vv[7][5] += x7 * y5;
                vv[7][6] += x7 * y6;
                vv[7][7] += x7 * y7;
            }

            for (int id = 0; id < nd; ++id) {
                for (int jd = 0; jd < nd; ++jd) {
                    double sum = 0.0;
                    int ir = (row0+id);
                    int jr = (row8+jd);
                    if (ir <= jr && ir < ny && jr < ny) {
                        sum = sum_all4(vv[id][jd]);
                        result[ny*ir + jr] = sum;
                    }   
                }
            }
        }
    }
    std::free(vd);
}