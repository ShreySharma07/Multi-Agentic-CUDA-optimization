#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N 16
#define M 1024
#define K 2048
#define L 768
#define TS 16

__global__ void matmul_3d_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float sA[TS][TS];
    __shared__ float sB[TS][TS];

    int l = blockIdx.x * TS + threadIdx.x;
    int m = blockIdx.y * TS + threadIdx.y;
    int n = blockIdx.z;

    float sum = 0.0f;
    int a_base = n * M * K + m * K;
    int b_base = l;

    for (int k_tile = 0; k_tile < K; k_tile += TS) {
        sA[threadIdx.y][threadIdx.x] = (m < M && (k_tile + threadIdx.x) < K) ? A[a_base + k_tile + threadIdx.x] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = ((k_tile + threadIdx.y) < K && l < L) ? B[(k_tile + threadIdx.y) * L + b_base] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TS; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (l < L && m < M) {
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

    dim3 block(TS, TS);
    dim3 grid((L + TS - 1) / TS, (M + TS - 1) / TS, N);

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
        float val_max = (h_C[i] > h_C_ref[i] ? h_C[i] : h_C_ref[i]);
        if (val_max < 1.0f) val_max = 1.0f;
        if (diff > 1e-3f * val_max) {
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