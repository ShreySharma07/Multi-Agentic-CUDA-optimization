#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_SIZE 32

__global__ void matmul_3d_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N, int M, int K, int L) {
    extern __shared__ float shared_mem[];
    float* A_tile = &shared_mem[0];
    float* B_tile = &shared_mem[TILE_SIZE * TILE_SIZE];

    int n = blockIdx.z;
    int m = blockIdx.y * TILE_SIZE + threadIdx.y;
    int l = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int k_block = 0; k_block < K; k_block += TILE_SIZE) {
        if (m < M && (k_block + threadIdx.x) < K) {
            A_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = A[(n * M + m) * K + (k_block + threadIdx.x)];
        } else {
            A_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }

        if (l < L && (k_block + threadIdx.y) < K) {
            B_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = B[(k_block + threadIdx.y) * L + l];
        } else {
            B_tile[threadIdx.y * TILE_SIZE + threadIdx.x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += A_tile[threadIdx.y * TILE_SIZE + k] * B_tile[k * TILE_SIZE + threadIdx.x];
        }
        __syncthreads();
    }

    if (n < N && m < M && l < L) {
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

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((L + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, N);

    size_t shared_size = 2 * TILE_SIZE * TILE_SIZE * sizeof(float);
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
        float diff = h_C_gpu[i] - h_C[i];
        if (diff < 0) diff = -diff;
        float ref = h_C[i] > 0 ? h_C[i] : -h_C[i];
        if (ref < 1.0f) ref = 1.0f;
        if (diff > 1e-3f * ref) {
            success = false; break;
        }
    }
    std::cout << (success ? "SUCCESS" : "FAILURE") << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    delete[] h_A; delete[] h_B; delete[] h_C; delete[] h_C_gpu;
    return 0;
}