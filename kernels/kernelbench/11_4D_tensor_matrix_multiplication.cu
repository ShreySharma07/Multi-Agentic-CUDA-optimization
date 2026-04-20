#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define B 8
#define I 256
#define J 512
#define L 256
#define K 768

__global__ void einsum_4d_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int i_b = blockIdx.z;

    int i = i_b / B;
    int b = i_b % B;

    if (k < K && j < J) {
        float sum = 0.0f;
        int a_offset = ((b * I + i) * J + j) * L;
        for (int l = 0; l < L; ++l) {
            sum += A[a_offset + l] * B[l * K + k];
        }
        C[((b * I + i) * J + j) * K + k] = sum;
    }
}

int main() {
    size_t sizeA = (size_t)B * I * J * L * sizeof(float);
    size_t sizeB = (size_t)L * K * sizeof(float);
    size_t sizeC = (size_t)B * I * J * K * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    float *h_C_ref = (float*)malloc(sizeC);

    for (size_t i = 0; i < (size_t)B * I * J * L; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)L * K; ++i) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, (J + block.y - 1) / block.y, B * I);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    einsum_4d_kernel<<<grid, block>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < I; ++i) {
            for (int j = 0; j < J; ++j) {
                for (int k = 0; k < K; ++k) {
                    double sum = 0.0;
                    for (int l = 0; l < L; ++l) {
                        sum += (double)h_A[(((b * I + i) * J + j) * L + l)] * (double)h_B[l * K + k];
                    }
                    h_C_ref[(((b * I + i) * J + j) * K + k)] = (float)sum;
                }
            }
        }
    }

    bool success = true;
    for (size_t i = 0; i < (size_t)B * I * J * K; ++i) {
        if (std::abs(h_C[i] - h_C_ref[i]) > 1e-4f * std::max(1.0f, std::max(std::abs(h_C[i]), std::abs(h_C_ref[i])))) {
            success = false;
            break;
        }
    }

    if (success) printf("SUCCESS\n"); else printf("FAILURE\n");
    printf("GPU Time: %f\n", ms);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}