#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_SIZE 16

__global__ void matmul_3d_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N, int M, int K, int L) {
    extern __shared__ float shared_mem[];
    float* A_tile = &shared_mem[0];
    float* B_tile = &shared_mem[TILE_SIZE * TILE_SIZE];

    int n = blockIdx.z;
    int m_base = blockIdx.y * (TILE_SIZE * 2);
    int l_base = blockIdx.x * (TILE_SIZE * 2);

    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;

    for (int k_block = 0; k_block < K; k_block += TILE_SIZE) {
        // Load tile A (4 segments of TILE_SIZE x TILE_SIZE)
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int m = m_base + i * TILE_SIZE + threadIdx.y;
                int k = k_block + threadIdx.x;
                if (m < M && k < K)
                    A_tile[(i * TILE_SIZE + threadIdx.y) * TILE_SIZE + threadIdx.x] = A[(n * M + m) * K + k];
                else
                    A_tile[(i * TILE_SIZE + threadIdx.y) * TILE_SIZE + threadIdx.x] = 0.0f;
            }
        }
        // Load tile B (similar)
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                int k = k_block + threadIdx.y;
                int l = l_base + j * TILE_SIZE + threadIdx.x;
                if (k < K && l < L)
                    B_tile[(i * TILE_SIZE + threadIdx.y) * TILE_SIZE + threadIdx.x] = B[k * L + l];
                else
                    B_tile[(i * TILE_SIZE + threadIdx.y) * TILE_SIZE + threadIdx.x] = 0.0f;
            }
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k) {
            float b00 = B_tile[k * TILE_SIZE + threadIdx.x];
            float b01 = B_tile[k * TILE_SIZE + TILE_SIZE + threadIdx.x];
            sum00 += A_tile[threadIdx.y * TILE_SIZE + k] * b00;
            sum01 += A_tile[threadIdx.y * TILE_SIZE + k] * b01;
            sum10 += A_tile[(threadIdx.y + TILE_SIZE) * TILE_SIZE + k] * b00;
            sum11 += A_tile[(threadIdx.y + TILE_SIZE) * TILE_SIZE + k] * b01;
        }
        __syncthreads();
    }

    if (n < N) {
        int m = m_base + threadIdx.y;
        int l = l_base + threadIdx.x;
        if (m < M && l < L) C[(n * M + m) * L + l] = sum00;
        if (m < M && l + TILE_SIZE < L) C[(n * M + m) * L + l + TILE_SIZE] = sum01;
        if (m + TILE_SIZE < M && l < L) C[(n * M + m + TILE_SIZE) * L + l] = sum10;
        if (m + TILE_SIZE < M && l + TILE_SIZE < L) C[(n * M + m + TILE_SIZE) * L + l + TILE_SIZE] = sum11;
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

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((L + 2 * TILE_SIZE - 1) / (2 * TILE_SIZE), (M + 2 * TILE_SIZE - 1) / (2 * TILE_SIZE), N);

    size_t shared_size = 4 * TILE_SIZE * TILE_SIZE * sizeof(float);
    matmul_3d_kernel<<<grid, block, shared_size>>>(d_A, d_B, d_C, N, M, K, L);

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
            success = false; break;
        }
    }
    std::cout << (success ? "SUCCESS" : "FAILURE") << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_gpu;
    return 0;
}