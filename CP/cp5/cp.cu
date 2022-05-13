#include "cp.h"
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define BASE_VALF 0.0


//straight copy from course material
static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}

//straight copy from course material
static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}

//straight copy from course material
static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)


//straight copy from course material v2
__global__ void mykernel(const float* t_data, float* result, int ny, int nx, int nny, int nnx) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    //init result
    for (int ib = 0; ib < 8; ib++) {
        for (int jb = 0; jb < 8; jb++) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64+ jb * 8 + ja;
            if (i < ny && j < ny) {
                result[j + i*ny] = 0.0;
            }
        }
    }

    if (ic > jc) {return;}

    float v[8][8];
    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            v[ib][jb] = BASE_VALF;
        }
    }

    for (int k = 0; k < nx; ++k) {
        float x[8];
        float y[8];
        for (int ib = 0; ib < 8; ++ib) {
            int i = ic * 64 + ib * 8 + ia;
            x[ib] = t_data[nny*k + i];
        }
        for (int jb = 0; jb < 8; ++jb) {
            int j = jc * 64 + jb * 8 + ja;
            y[jb] = t_data[nny*k + j];
        }
        for (int ib = 0; ib < 8; ++ib) {
            for (int jb = 0; jb < 8; ++jb) {
                v[ib][jb] += x[ib] * y[jb];
            }
        }
    }

    for (int ib = 0; ib < 8; ++ib) {
        for (int jb = 0; jb < 8; ++jb) {
            int i = ic * 64 + ib * 8 + ia;
            int j = jc * 64 + jb * 8 + ja;
            if (i < ny && j < ny) {
                result[ny*i + j] = v[ib][jb];
            }
        }
    }
}

//straight copy from course material v2
__global__ void myppkernel(float* t_data, const float* norm_data, int ny, int nx, int nny, int nnx) {
    int ja = threadIdx.x;
    int i = blockIdx.y;

    for (int jb = 0; jb < nnx; jb += 64) {
        int j = jb + ja;
        float v = (i < ny && j < nx) ? norm_data[nx*i + j] : BASE_VALF;
        t_data[nny*j + i] = v;
    }
}

//straight copy from course material v3
void step(float* result, const float* d, int ny, int nx, int nny, int nnx) {
    // Allocate memory & copy data to GPU
    float* rGPU = NULL; 
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    float* norm_dGPU = NULL; //new for normalized data
    CHECK(cudaMalloc((void**)&norm_dGPU, nx * ny * sizeof(float)));
    float* tGPU = NULL; //new for transpose
    CHECK(cudaMalloc((void**)&tGPU, nny * nnx * sizeof(float)));
    CHECK(cudaMemcpy(norm_dGPU, d, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    
    // Run kernel  
    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nny);
        myppkernel<<<dimGrid, dimBlock>>>(tGPU, norm_dGPU, ny, nx, nny, nnx);
        CHECK(cudaGetLastError());
    }

    // Run kernel
    {
        dim3 dimBlock(8, 8); 
        dim3 dimGrid(nny / 64, nny / 64); 
        mykernel<<<dimGrid, dimBlock>>>(tGPU, rGPU, ny, nx, nny, nnx);
        CHECK(cudaGetLastError());
    }

    // Copy data back to CPU & release memory 
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(norm_dGPU));
    CHECK(cudaFree(tGPU));
    CHECK(cudaFree(rGPU));
}


/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    int nnx = roundup(nx, 64);
    int nny = roundup(ny, 64);

    //copy from CPU
    float* data_norm = (float*) malloc(ny * nx * sizeof(float));

    for (int y = ny; y--;) {
        double sum = 0.0, mean = 0.0, pow_sum = 0.0, sqrt_sum_sqrt = 0.0;

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


    step(result, data_norm, ny, nx, nny, nnx);
}
