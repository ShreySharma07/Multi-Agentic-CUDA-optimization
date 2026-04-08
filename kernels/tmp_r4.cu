#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>

__global__ void vectorAdd(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int numElements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    #pragma unroll 4
    for (int i = idx; i < numElements; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_Ref = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = (float)rand() / (float)RAND_MAX;
        h_B[i] = (float)rand() / (float)RAND_MAX;
        h_Ref[i] = h_A[i] + h_B[i];
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        float diff = std::abs(h_C[i] - h_Ref[i]);
        if (diff > 1e-3f * std::max(1.0f, std::max(std::abs(h_C[i]), std::abs(h_Ref[i])))) {
            printf("ERROR at index %d: GPU=%f CPU=%f DIFF=%f\n", i, h_C[i], h_Ref[i], diff);
            success = false;
            break;
        }
    }

    if (success) printf("SUCCESS\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_Ref);

    return 0;
}