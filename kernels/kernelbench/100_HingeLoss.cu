#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* partial_sums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    while (i < n) {
        float val = 1.0f - predictions[i] * targets[i];
        sum += (val > 0.0f) ? val : 0.0f;
        i += blockDim.x * gridDim.x;
    }

    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 32768;
    size_t size = N * sizeof(float);

    float *h_pred = new float[N];
    float *h_targ = new float[N];
    for (int i = 0; i < N; i++) {
        h_pred[i] = (float)rand() / RAND_MAX;
        h_targ[i] = (rand() % 2 == 0) ? -1.0f : 1.0f;
    }

    float *d_pred, *d_targ, *d_partial;
    cudaMalloc(&d_pred, size);
    cudaMalloc(&d_targ, size);
    int grid_size = 128;
    cudaMalloc(&d_partial, grid_size * sizeof(float));

    cudaMemcpy(d_pred, h_pred, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targ, h_targ, size, cudaMemcpyHostToDevice);

    hinge_loss_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_pred, d_targ, d_partial, N);

    std::vector<float> h_partial(grid_size);
    cudaMemcpy(h_partial.data(), d_partial, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    float gpu_res = 0.0f;
    for (float val : h_partial) gpu_res += val;
    gpu_res /= N;

    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        cpu_sum += std::max(0.0f, 1.0f - h_pred[i] * h_targ[i]);
    }
    float cpu_res = cpu_sum / N;

    if (std::abs(gpu_res - cpu_res) <= 1e-3 * std::max(1.0f, std::max(std::abs(gpu_res), std::abs(cpu_res)))) {
        std::cout << "SUCCESS" << std::endl;
    } else {
        printf("FAILURE: GPU=%f CPU=%f\n", gpu_res, cpu_res);
    }

    cudaFree(d_pred); cudaFree(d_targ); cudaFree(d_partial);
    delete[] h_pred; delete[] h_targ;
    return 0;
}