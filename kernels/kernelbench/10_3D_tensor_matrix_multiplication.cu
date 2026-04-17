#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 16
#define M 1024
#define K 2048
#define L 768

__global__ void matmul_3d_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.z;

    if (l < L && m < M && n < N) {
        float sum = 0.0f;
        int a_offset = n * M * K + m * K;
        int b_offset = l;
        for (int k = 0; k < K; ++k) {
            sum += A[a_offset + k] * B[k * L + b_offset];
        }
        C[n * M * L + m * L + l] = sum;
    }
}

int main() {
    size_t size_A = (size_t)N * M * K * sizeof(float);
    size_t size_B = (size_t)K * L * sizeof(float);
    size_t size_C = (size_t)N * M * L * sizeof(float);

    float *h_A = (float*)malloc(size_A);
    float *h_B = (float*)malloc(size_B);
    float *h_C = (float*)malloc(size_C);
    float *h_C_ref = (float*)malloc(size_C);

    for (size_t i = 0; i < (size_t)N * M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)K * L; ++i) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((L + block.x - 1) / block.x, (M + block.y - 1) / block.y, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_3d_kernel<<<grid, block>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    for (int n = 0; n < N; ++n) {
        for (int m = 0; m < M; ++m) {
            for (int l = 0; l < L; ++l) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += h_A[n * M * K + m * K + k] * h_B[k * L + l];
                }
                h_C_ref[n * M * L + m * L + l] = sum;
            }
        }
    }

    bool success = true;
    for (size_t i = 0; i < (size_t)N * M * L; ++i) {
        float diff = h_C[i] - h_C_ref[i];
        if (diff < 0) diff = -diff;
        float tol = 1e-3f * (1.0f + (h_C[i] > h_C_ref[i] ? h_C[i] : h_C_ref[i]));
        if (diff > tol) {
            success = false;
            break;
        }
    }

    if (success) printf("SUCCESS\n");
    else printf("FAILURE\n");
    printf("GPU Time: %f\n", milliseconds);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
    return 0;
}