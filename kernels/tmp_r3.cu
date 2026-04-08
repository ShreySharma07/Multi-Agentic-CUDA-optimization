#include <iostream>
#include <cuda_runtime.h>
#include <cmath>

__global__ void vectorAdd(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int numElements) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    int stride = blockDim.x * gridDim.x * 8;

    for (int j = i; j < (numElements & ~7); j += stride) {
        float4 a1 = reinterpret_cast<const float4*>(A + j)[0];
        float4 a2 = reinterpret_cast<const float4*>(A + j + 4)[0];
        float4 b1 = reinterpret_cast<const float4*>(B + j)[0];
        float4 b2 = reinterpret_cast<const float4*>(B + j + 4)[0];
        
        float4 c1, c2;
        c1.x = a1.x + b1.x; c1.y = a1.y + b1.y; c1.z = a1.z + b1.z; c1.w = a1.w + b1.w;
        c2.x = a2.x + b2.x; c2.y = a2.y + b2.y; c2.z = a2.z + b2.z; c2.w = a2.w + b2.w;
        
        reinterpret_cast<float4*>(C + j)[0] = c1;
        reinterpret_cast<float4*>(C + j + 4)[0] = c2;
    }

    for (int j = (numElements & ~7) + threadIdx.x + blockIdx.x * blockDim.x; j < numElements; j += blockDim.x * gridDim.x) {
        C[j] = A[j] + B[j];
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
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
        h_Ref[i] = h_A[i] + h_B[i];
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = 128; 
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < numElements; ++i) {
        float diff = std::abs(h_C[i] - h_Ref[i]);
        if (diff > 1e-3 * std::max(1.0f, std::max(std::abs(h_C[i]), std::abs(h_Ref[i])))) {
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