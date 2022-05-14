#include <cstdio>
#include <iostream>
#include <chrono>
#include <vector>
#include <ctime>
#include <omp.h>

using namespace std;


struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};

Result findResult(int ny, int nx, std::vector<float> sums, Result result){
    int nx1 = nx + 1;
    int ny1 = ny + 1;
    float s_all = sums[nx1*ny1 - 1];
    float max = 0.0;
    float inner = 0;
    float outer = 0;

    #pragma omp parallel
    {
        float current_max = -1.0, in = 0.0, out = 0.0;
        int current_x0 = 0, current_y0 = 0, current_x1 = 1, current_y1 = 1;

        #pragma omp for schedule(dynamic, 1)
        for (int height = 1; height < ny1; ++height) {
            for (int width = 1; width < nx1; ++width) {
                float size_x = height * width;
                float size_y = nx * ny - size_x;
                float sizex_div = 1.0 / size_x;
                float sizey_div = 1.0 / size_y;

                for (int y0 = 0; y0 < ny1 - height; ++y0) {
                    for (int x0 = 0; x0 < nx1 - width; ++x0) {
                        int y1 = y0 + height;
                        int x1 = x0 + width;
                        float in_sum = sums[y1*nx1 + x1] - sums[y1*nx1 + x0] - sums[y0*nx1 + x1] + sums[y0*nx1 + x0];
                        float out_sum = s_all - in_sum;
                        float in_sum2 = in_sum * in_sum * sizex_div;
                        float out_sum2 = out_sum * out_sum * sizey_div;

                        float potential_max = in_sum2 + out_sum2;

                        if (potential_max > current_max) {
                            current_max = potential_max;
                            current_x0 = x0;
                            current_y0 = y0;
                            current_x1 = x1;
                            current_y1 = y1;
                            in = in_sum / size_x;
                            out = out_sum / size_y;
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
                inner = in;
                outer = out;
            }
        }
    }


    result.inner[0] = inner;
    result.inner[1] = inner;
    result.inner[2] = inner;
    result.outer[0] = outer;
    result.outer[1] = outer;
    result.outer[2] = outer;

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
    std::vector<float> sums(nx1 * ny1, 0);

   

    //rectangle color sums
    for (int y = 1; y < ny1; ++y) {
        for (int x = 1; x < nx1; ++x) {
            //__builtin_prefetch(&sums[(x-1) + pf + nx1*(y-1)]);
            //__builtin_prefetch(&sums[(x) + pf + nx1*(y-1)]);
            __builtin_prefetch(&sums[(x-1) + pf + nx1*y]);
            __builtin_prefetch(&data[3*((x-1)+ pf + nx*(y-1))]);
            sums[x + nx1*y] = (sums[(x) + nx1*(y-1)] + sums[(x-1) + nx1*y] - sums[(x-1) + nx1*(y-1)]) + data[3*((x-1)+ nx*(y-1))];
        }
    }

    result = findResult(ny, nx, sums, result);

    //free(vdata);
    //free(sums);

    return result;
}
