#include <limits>
#include <cstdio>
#include <iostream>
#include <chrono>
#include <ctime>
#include <omp.h>

using namespace std;


// straight copy from course material
typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

// straight copy from course material
static double4_t* double4_alloc(std::size_t n) {
    void* tmp = 0;
    if (posix_memalign(&tmp, sizeof(double4_t), sizeof(double4_t) * n)) {
        throw std::bad_alloc();
    }
    return (double4_t*)tmp;
}

constexpr double param_zero = 0.0;
constexpr double4_t d4_init {
    param_zero, param_zero, param_zero, param_zero
};

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

static inline double sum_all3(double4_t vv) {
    double sum = 0.0;
    sum += vv[0];
    sum += vv[1];
    sum += vv[2];

    return sum;
}

Result findResult(int ny, int nx, double4_t* sums, Result result){
    int nx1 = nx + 1;
    int ny1 = ny + 1;
    double4_t s_all = sums[(nx+1)*(ny+1) - 1];
    double max = 0.0;

    #pragma omp parallel
    {
        double current_max = -1.0;
        int current_x0 = 0, current_y0 = 0, current_x1 = 1, current_y1 = 1;

        #pragma omp for schedule(dynamic, 1)
        for (int height = 1; height < ny1; ++height) {
            for (int width = 1; width < nx1; ++width) {
                double size_x = height * width;
                double size_y = nx * ny - size_x;
                double sizex_div = 1.0 / size_x;
                double sizey_div = 1.0 / size_y;

                for (int y0 = 0; y0 < ny1 - height; ++y0) {
                    for (int x0 = 0; x0 < nx1 - width; ++x0) {
                        int y1 = y0 + height;
                        int x1 = x0 + width;
                        double4_t in_sum = sums[y1*nx1 + x1] - sums[y1*nx1 + x0] - sums[y0*nx1 + x1] + sums[y0*nx1 + x0];
                        double4_t out_sum = s_all - in_sum;
                        double4_t in_sum2 = in_sum * in_sum * sizex_div;
                        double4_t out_sum2 = out_sum * out_sum * sizey_div;

                        double potential_max = sum_all3(in_sum2 + out_sum2);

                        if (potential_max > current_max) {
                            current_max = potential_max;
                            current_x0 = x0;
                            current_y0 = y0;
                            current_x1 = x1;
                            current_y1 = y1;
                        }
                    }
                }
            }
        }

        #pragma omp critical
        {
            if (current_max > max) {
                max = current_max;
                result.x0 = current_x0;
                result.y0 = current_y0;
                result.x1 = current_x1;
                result.y1 = current_y1;
            }
        }
    }

    double4_t inner = sums[result.y1*nx1 + result.x1] - sums[result.y1*nx1 + result.x0] - sums[result.y0*nx1 + result.x1] + sums[result.y0*nx1 + result.x0];
    double4_t outer = s_all - inner;
    int size_in = (result.x1 - result.x0) * (result.y1 - result.y0);
    int size_out = nx * ny - size_in;

    result.inner[0] = inner[0]/size_in;
    result.inner[1] = inner[1]/size_in;
    result.inner[2] = inner[2]/size_in;
    result.outer[0] = outer[0]/size_out;
    result.outer[1] = outer[1]/size_out;
    result.outer[2] = outer[2]/size_out;

    return result;
}


/*
maxis is maxe function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
	Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    int pf = 20; 
    int nx1 = nx + 1;
    int ny1 = ny + 1;
    double4_t* sums = double4_alloc((nx+1)*(ny+1));
	double4_t* vdata = double4_alloc(ny*nx);

    //prework
    #pragma omp parallel for schedule(static, 1)
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            __builtin_prefetch(&data[0 + 3*x + 3*nx*y + 3*pf]);
            __builtin_prefetch(&data[1 + 3*x + 3*nx*y + 3*pf]);
            __builtin_prefetch(&data[2 + 3*x + 3*nx*y + 3*pf]);
            double red = data[0 + 3*x + 3*nx*y];
            double green = data[1 + 3*x + 3*nx*y];
            double blue = data[2 + 3*x + 3*nx*y];
            vdata[x + nx*y][0] = red;
            vdata[x + nx*y][1] = green;
            vdata[x + nx*y][2] = blue;
            vdata[x + nx*y][3] = 0.0;
        }
    }
    
    //init
    #pragma omp parallel for schedule(static, 1)
    for (int y = 0; y < ny1; ++y) {
        for (int x = 0; x < nx1; ++x) {
            sums[x + nx1*y] = d4_init;
        }
    }

    //rectangle color sums
    for (int y = 1; y < ny1; ++y) {
        for (int x = 1; x < nx1; ++x) {
            //__builtin_prefetch(&sums[(x-1) + pf + nx1*(y-1)]);
            //__builtin_prefetch(&sums[(x) + pf + nx1*(y-1)]);
            __builtin_prefetch(&sums[(x-1) + pf + nx1*y]);
            __builtin_prefetch(&vdata[(x-1)+ pf + nx*(y-1)]);
            sums[x + nx1*y] = (sums[(x) + nx1*(y-1)] + sums[(x-1) + nx1*y] - sums[(x-1) + nx1*(y-1)]) + vdata[(x-1)+ nx*(y-1)];
        }
    }

    result = findResult(ny, nx, sums, result);

    free(vdata);
    free(sums);

    return result;
}