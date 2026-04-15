#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <cstdlib>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, double* global_sum, size_t total_elements, size_t N) {
    double thread_sum = 0.0;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    for (size_t i = idx; i < total_elements; i += stride) {
        size_t col = i % N;
        float p = predictions[i];
        float t = targets[col];
        float val = 1.0f - p * t;
        if (val > 0.0f) {
            thread_sum += (double)val;
        }
    }

    __shared__ double sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(global_sum, sdata[0]);
    }
}

int main() {
    const size_t N = 32768;
    const size_t total_elements = N * N;

    float* h_preds = (float*)malloc(total_elements * sizeof(float));
    float* h_targets = (float*)malloc(N * sizeof(float));

    if (!h_preds || !h_targets) {
        if (h_preds) free(h_preds);
        if (h_targets) free(h_targets);
        return 1;
    }

    for (size_t i = 0; i < total_elements; ++i) {
        h_preds[i] = (float)(i % 1024) / 1024.0f;
    }
    for (size_t i = 0; i < N; ++i) {
        h_targets[i] = (i % 2 == 0) ? 1.0f : -1.0f;
    }

    float *d_preds, *d_targets;
    double *d_sum;
    cudaMalloc(&d_preds, total_elements * sizeof(float));
    cudaMalloc(&d_targets, N * sizeof(float));
    cudaMalloc(&d_sum, sizeof(double));

    cudaMemcpy(d_preds, h_preds, total_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum, 0, sizeof(double));

    int blockSize = 256;
    int gridSize = 2048;
    hinge_loss_kernel<<<gridSize, blockSize>>>(d_preds, d_targets, d_sum, total_elements, N);

    double h_sum_gpu_db;
    cudaMemcpy(&h_sum_gpu_db, d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    float gpu_result = (float)(h_sum_gpu_db / (double)total_elements);

    double h_sum_cpu = 0.0;
    for (size_t i = 0; i < total_elements; ++i) {
        float val = 1.0f - h_preds[i] * h_targets[i % N];
        if (val > 0.0f) h_sum_cpu += (double)val;
    }
    float cpu_result = (float)(h_sum_cpu / (double)total_elements);

    float diff = fabsf(gpu_result - cpu_result);
    float max_val = std::max({1.0f, fabsf(gpu_result), fabsf(cpu_result)});
    if (diff <= 1e-3f * max_val) {
        printf("SUCCESS\n");
    } else {
        printf("FAILURE: GPU=%f CPU=%f DIFF=%f\n", gpu_result, cpu_result, diff);
    }

    free(h_preds);
    free(h_targets);
    cudaFree(d_preds);
    cudaFree(d_targets);
    cudaFree(d_sum);

    return 0;
}