#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void hinge_loss_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* __restrict__ partial_sums, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS_PER_THREAD;

    float sum = 0.0f;
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        if (i + k < n) {
            float p = predictions[i + k];
            float t = targets[i + k];
            float val = 1.0f - p * t;
            sum += (val > 0.0f) ? val : 0.0f;
        }
    }

    sum = warpReduceSum(sum);

    if (tid % warpSize == 0) {
        sdata[tid / warpSize] = sum;
    }
    __syncthreads();

    if (tid < (BLOCK_SIZE / warpSize)) {
        sum = sdata[tid];
        sum = warpReduceSum(sum);
        if (tid == 0) partial_sums[blockIdx.x] = sum;
    }
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
    
    int grid_size = (N + (BLOCK_SIZE * ITEMS_PER_THREAD) - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    cudaMalloc(&d_partial, grid_size * sizeof(float));

    cudaMemcpy(d_pred, h_pred, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targ, h_targ, size, cudaMemcpyHostToDevice);

    hinge_loss_kernel<<<grid_size, BLOCK_SIZE, (BLOCK_SIZE / 32) * sizeof(float)>>>(d_pred, d_targ, d_partial, N);

    std::vector<float> h_partial(grid_size);
    cudaMemcpy(h_partial.data(), d_partial, grid_size * sizeof(float), cudaMemcpyDeviceToHost);

    float gpu_sum = 0.0f;
    for (int i = 0; i < grid_size; ++i) gpu_sum += h_partial[i];
    float gpu_res = gpu_sum / (float)N;

    float cpu_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float val = 1.0f - h_pred[i] * h_targ[i];
        cpu_sum += (val > 0.0f) ? val : 0.0f;
    }
    float cpu_res = cpu_sum / (float)N;

    float diff = gpu_res - cpu_res;
    if (diff < 0) diff = -diff;
    float max_val = gpu_res;
    if (cpu_res > max_val) max_val = cpu_res;
    if (max_val < 1.0f) max_val = 1.0f;

    if (diff <= 1e-3f * max_val) {
        std::cout << "SUCCESS" << std::endl;
    } else {
        printf("FAILURE: GPU=%f CPU=%f\n", gpu_res, cpu_res);
    }

    cudaFree(d_pred); cudaFree(d_targ); cudaFree(d_partial);
    delete[] h_pred; delete[] h_targ;
    return 0;
}