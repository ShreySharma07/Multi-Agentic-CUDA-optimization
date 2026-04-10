#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int numElements) {
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (i + 3 < numElements) {
        float a0 = A[i], a1 = A[i+1], a2 = A[i+2], a3 = A[i+3];
        float b0 = B[i], b1 = B[i+1], b2 = B[i+2], b3 = B[i+3];
        C[i] = a0 + b0;
        C[i+1] = a1 + b1;
        C[i+2] = a2 + b2;
        C[i+3] = a3 + b3;
    } else {
        for (int j = i; j < numElements; ++j) {
            C[j] = A[j] + B[j];
        }
    }
}

int main() {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements / 4 + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    std::cout << "Vector addition completed successfully!" << std::endl;
    return 0;
}