#include "cp.h"
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include<vector>
using namespace std;

inline void check(cudaError_t err, const char* context) 
{
    if (err != cudaSuccess) 
    {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(x) check(x, #x)

inline int static divup(int a, int b) 
{
    return (a + b - 1)/b;
}

inline int static roundup(int a, int b) 
{
    return divup(a, b) * b;
}

__global__ void mykernel(int nx, int ny, const float *data, float *result) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= ny || y >= ny || x < y) {    
        return;
    }

    float sum = 0.0;
    float v = HUGE_VALF; 

    for (int k = 0; k < nx; ++k) {
        sum += data[k + y*nx] * data[x*nx + k];
    }

    result[x + ny * y] = sum;
}

/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float* data, float *result) 
{
    //preprocess data into ProcessedData
    float * ProcessedData = (float * )calloc(nx*ny , sizeof(float));

    for( int y = 0; y < ny; y++ )
    {
        float s = 0.0;
        
        for( int x = 0; x < nx; x++ )
        {
            s += data[ x + y * nx ];
        }
        s /= (float)nx;

        float t = 0.0;

        for( int x = 0; x < nx; x++ )
        {
            ProcessedData[ x + y*nx ] = (double)data[ x + y*nx ] - s;
            t +=  ProcessedData[ x + y * nx ] * ProcessedData[ x + y * nx ];
        }

        t = sqrt( t );

        for( int x = 0; x < nx; x ++ )
        {
            ProcessedData[ x + y * nx ] = ProcessedData[ x + y * nx ] / t;
        }
    }    

    //Allocate memory and copy data to GPU
    float * dGPU = NULL;//data for GPU
    CHECK(cudaMalloc((void**)&dGPU, nx * ny * sizeof(float)));

    float * rGPU = NULL;//result for GPU
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));

    CHECK(cudaMemcpy(dGPU, ProcessedData, nx * ny * sizeof(float), cudaMemcpyHostToDevice));
    
    // Run kernel
    dim3 dimBlock(8, 8);
    dim3 dimGrid(roundup(ny, dimBlock.x), roundup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>( nx, ny, dGPU, rGPU);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost ));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
	std::free(ProcessedData);
}
