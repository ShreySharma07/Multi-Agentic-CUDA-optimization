#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>

#define N 16
#define M 1024
#define K 2048
#define L 768
#define TILE_SIZE 16

__global__ void matmul_3d_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int l = blockIdx.x * TILE_SIZE + threadIdx.x;
    int m = blockIdx.y * TILE_SIZE + threadIdx.y;
    int n = blockIdx.z;

    float sum = 0.0f;
    for (int k_tile = 0; k_tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++k_tile) {
        if (m < M && k_tile * TILE_SIZE + threadIdx.x < K)
            tileA[threadIdx.y][threadIdx.x] = A[n * M * K + m * K + k_tile * TILE_SIZE + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        if (l < L && k_tile * TILE_SIZE + threadIdx.y < K)
            tileB[threadIdx.y][threadIdx.x] = B[(k_tile * TILE_SIZE + threadIdx.y) * L + l];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (l < L && m < M) {
        C[n * M * L + m * L + l] = sum;
    }
}

int main() {
    size_t sizeA = (size_t)N * M * K * sizeof(float);
    size_t sizeB = (size_t)K * L * sizeof(float);
    size_t sizeC = (size_t)N * M * L * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    float *h_C_ref = (float*)malloc(sizeC);

    for (size_t i = 0; i < (size_t)N * M * K; ++i) h_A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)K * L; ++i) h_B[i] = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((L + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    matmul_3d_kernel<<<grid, block>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

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