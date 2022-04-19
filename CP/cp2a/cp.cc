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
void correlate(const int ny, int nx, const float *data, float *result) {
    // otherwise same as cp1 but dividing the matrix multiplication part in chunks
    std::vector<double> data_norm(nx*ny);
    // dividing rows to chuncks
    const int chunks = 6;
    const int out = nx % chunks;
    const int full_groups = nx / chunks;

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
    }

    // matrix multiplication done in chunks
    for (int i = 0; i < ny; i++ ) {
        for (int j = i; j < ny; j++){
            double group_sum[chunks] = {0};
            double out_sum = 0.0;
            for (int full = 0; full < full_groups; full++) {
                for (int chunk = 0; chunk < chunks; chunk++) {
                    group_sum[chunk] += data_norm[full*chunks + chunk + i*nx] * data_norm[full*chunks + chunk + j*nx];
                }
            }
            int offset = nx - out;
            for (int ind = 0; ind < out; ind++) {
                out_sum += data_norm[offset + ind + i*nx] * data_norm[offset + ind + j*nx];
            }

            for (int chunk = 0; chunk < chunks; chunk++) {
                out_sum += group_sum[chunk];
            }
            
            result[j + i*ny] = out_sum;
        }
    }
}
