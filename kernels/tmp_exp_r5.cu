#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define N 4096
#define TILE_SIZE 32

__global__ void matmul_symmetric_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    for (int k_tile = 0; k_tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
        int k = k_tile * TILE_SIZE;
        
        if (row < N && (k + threadIdx.x) < N)
            tileA[threadIdx.y][threadIdx.x] = A[row * N + (k + threadIdx.x)];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (k + threadIdx.y) < N)
            tileB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum = fmaf(tileA[threadIdx.y][i], tileB[i][threadIdx.x], sum);
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    size_t size = (size_t)N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_C_ref = (float*)malloc(size);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            float valA = (h_A[i * N + j] + h_A[j * N + i]) * 0.5f;
            h_A[i * N + j] = h_A[j * N + i] = valA;
            float valB = (h_B[i * N + j] + h_B[j * N + i]) * 0.5f;
            h_B[i * N + j] = h_B[j * N + i] = valB;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_symmetric_kernel<<<grid, block>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += (double)h_A[i * N + k] * (double)h_B[k * N + j];
            }
            h_C_ref[i * N + j] = (float)sum;
        }
    }

    bool success = true;
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        if (std::abs(h_C[i] - h_C_ref[i]) > 1e-3f * std::max(1.0f, std::max(std::abs(h_C[i]), std::abs(h_C_ref[i])))) {
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