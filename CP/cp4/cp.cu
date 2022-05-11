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
static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)

__global__ void mykernel(const float* data, float* result, int ny, int nx) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= ny || j >= ny)
        return;
    float v = BASE_VALF;
    for (int k = 0; k < nx; ++k) {
        float x = data[nx*j + k];
        float y = data[nx*i + k];
        float z = x * y;
        v += z;
    }
    result[ny*i + j] = v;
}

//straight copy from course material
void step(float* result, const float* d, int ny, int nx) {
    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, ny * nx * sizeof(float)));
    float* rGPU = NULL; 
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, d, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(dGPU, rGPU, ny, nx);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
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

    step(result, data_norm, ny, nx);
}