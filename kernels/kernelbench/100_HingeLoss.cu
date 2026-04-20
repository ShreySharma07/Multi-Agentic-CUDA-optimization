#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void hinge_loss_kernel(const float* predictions, const float* targets, float* partial_sums, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float val = 0.0f;
    if (idx < n) {
        float p = predictions[idx];
        float t = targets[idx];
        float loss = 1.0f - (p * t);
        val = (loss > 0.0f) ? loss : 0.0f;
    }

    sdata[tid] = val;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 32768;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;

    size_t size = N * sizeof(float);
    float *h_pred = (float*)malloc(size);
    float *h_targ = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_pred[i] = (float)rand() / RAND_MAX;
        h_targ[i] = (rand() % 2 == 0) ? -1.0f : 1.0f;
    }

    float *d_pred, *d_targ, *d_partial;
    cudaMalloc(&d_pred, size);
    cudaMalloc(&d_targ, size);
    cudaMalloc(&d_partial, blocks * sizeof(float));

    cudaMemcpy(d_pred, h_pred, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_targ, h_targ, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    hinge_loss_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_pred, d_targ, d_partial, N);
    
    std::vector<float> h_partial(blocks);
    cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float gpu_sum = 0.0f;
    for(float f : h_partial) gpu_sum += f;
    float gpu_loss = gpu_sum / N;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    float cpu_sum = 0.0f;
    for (int i = 0; i < N; ++i) {
        float val = 1.0f - (h_pred[i] * h_targ[i]);
        cpu_sum += (val > 0.0f) ? val : 0.0f;
    }
    float cpu_loss = cpu_sum / N;

    if (std::abs(gpu_loss - cpu_loss) <= 1e-3 * std::max(1.0f, std::max(std::abs(gpu_loss), std::abs(cpu_loss)))) {
        printf("SUCCESS\n");
    } else {
        printf("FAILURE\n");
    }
    printf("GPU Time: %f\n", milliseconds);

    cudaFree(d_pred);
    cudaFree(d_targ);
    cudaFree(d_partial);
    free(h_pred);
    free(h_targ);

    return 0;
}