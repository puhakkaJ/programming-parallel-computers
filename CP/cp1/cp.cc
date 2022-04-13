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
    }

    for (int i = ny; i--; ) {
        for (int j = i+1; j--;){
            double res = 0.0;
            for (int k = nx; k--;){
                res += data_norm[k + i*nx] * data_norm[k + j*nx];
            }
            result[i + j*ny] = res;
        }
    }
}
