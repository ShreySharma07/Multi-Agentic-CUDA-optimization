#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

#define B 8
#define I 256
#define J 512
#define L 256
#define K 768

__global__ void einsum_4d_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    int b = blockIdx.z / (I / blockDim.y);
    int i = (blockIdx.z % (I / blockDim.y)) * blockDim.y + threadIdx.y;
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Adjusted logic to map dimensions correctly
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // Use a simpler mapping for clarity and performance
    int global_b = blockIdx.z;
    int global_i = blockIdx.y;
    int global_j = blockIdx.x;
    int global_k = threadIdx.x;

    if (global_b < B && global_i < I && global_j < J && global_k < K) {
        float sum = 0.0f;
        const float* row_A = &A[((global_b * I + global_i) * J + global_j) * L];
        for (int l = 0; l < L; ++l) {
            sum += row_A[l] * B[l * K + global_k];
        }
        C[((global_b * I + global_i) * J + global_j) * K + global_k] = sum;
    }
}

int main() {
    size_t sizeA = (size_t)B * I * J * L * sizeof(float);
    size_t sizeB = (size_t)L * K * sizeof(float);
    size_t sizeC = (size_t)B * I * J * K * sizeof(float);

    std::vector<float> h_A(B * I * J * L), h_B(L * K), h_C(B * I * J * K);
    for (auto& f : h_A) f = static_cast<float>(rand()) / RAND_MAX;
    for (auto& f : h_B) f = static_cast<float>(rand()) / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid(1, I, B * J); // Simplified for demonstration

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Optimized kernel launch configuration
    einsum_4d_kernel<<<dim3(3, I, B), 256>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "GPU Time: " << milliseconds << " ms" << std::endl;

    cudaMemcpy(h_C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                for (int k = 0; k < K; ++k) {
                    float cpu_val = 0.0f;
                    for (int l = 0; l < L; ++l) {
                        cpu_val += h_A[((b * I + i) * J + j) * L + l] * h_B[l * K + k];
                    }
                    float gpu_val = h_C[((b * I + i) * J + j) * K + k];
                    if (std::abs(gpu_val - cpu_val) > 1e-3 * std::max(1.0f, std::max(std::abs(gpu_val), std::abs(cpu_val)))) {
                        success = false;
                        break;
                    }
                }
                if (!success) break;
            }
            if (!success) break;
        }
        if (!success) break;
    }

    if (success) std::cout << "SUCCESS" << std::endl;
    else std::cout << "FAILURE" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}