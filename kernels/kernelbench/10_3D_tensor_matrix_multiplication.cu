#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_K 16
#define TILE_L 16
#define TILE_M 16

__global__ void matmul_3d_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N, int M, int K, int L) {
    int n = blockIdx.z;
    int m = blockIdx.y * TILE_M + threadIdx.y;
    int l = blockIdx.x * TILE_L + threadIdx.x;

    if (n < N && m < M && l < L) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[(n * M + m) * K + k] * B[k * L + l];
        }
        C[(n * M + m) * L + l] = sum;
    }
}

int main() {
    const int N = 16, M = 128, K = 256, L = 128;
    size_t size_A = (size_t)N * M * K * sizeof(float);
    size_t size_B = (size_t)K * L * sizeof(float);
    size_t size_C = (size_t)N * M * L * sizeof(float);

    float *h_A = new float[N * M * K];
    float *h_B = new float[K * L];
    float *h_C = new float[N * M * L];
    float *h_C_gpu = new float[N * M * L];

    for (int i = 0; i < N * M * K; i++) h_A[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K * L; i++) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 block(TILE_L, TILE_M);
    dim3 grid((L + TILE_L - 1) / TILE_L, (M + TILE_M - 1) / TILE_M, N);

    matmul_3d_kernel<<<grid, block>>>(d_A, d_B, d_C, N, M, K, L);

    cudaMemcpy(h_C_gpu, d_C, size_C, cudaMemcpyDeviceToHost);

    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            for (int l = 0; l < L; l++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    sum += h_A[(n * M + m) * K + k] * h_B[k * L + l];
                }
                h_C[(n * M + m) * L + l] = sum;
            }
        }
    }

    bool success = true;
    for (int i = 0; i < N * M * L; i++) {
        if (std::abs(h_C_gpu[i] - h_C[i]) > 1e-3f * std::max(1.0f, std::max(std::abs(h_C_gpu[i]), std::abs(h_C[i])))) {
            success = false;
            break;
        }
    }

    if (success) std::cout << "SUCCESS" << std::endl;
    else std::cout << "FAILURE" << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_gpu;
    return 0;
}