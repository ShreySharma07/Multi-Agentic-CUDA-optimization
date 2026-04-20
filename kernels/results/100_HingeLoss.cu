#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void hinge_loss_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, float* __restrict__ partial_sums, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int pos = idx + i * blockDim.x;
        if (pos < n) {
            float p = predictions[pos];
            float t = targets[pos];
            float diff = 1.0f - (p * t);
            sum += fmaxf(0.0f, diff);
        }
    }

    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        volatile float* vdata = sdata;
        vdata[tid] += vdata[tid + 32];
        vdata[tid] += vdata[tid + 16];
        vdata[tid] += vdata[tid + 8];
        vdata[tid] += vdata[tid + 4];
        vdata[tid] += vdata[tid + 2];
        vdata[tid] += vdata[tid + 1];
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

int main() {
    const int N = 32768;
    const int threads = 256;
    const int items_per_thread = 4;
    const int blocks = (N + (threads * items_per_thread) - 1) / (threads * items_per_thread);

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

    double cpu_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        float val = 1.0f - (h_pred[i] * h_targ[i]);
        if (val > 0.0f) cpu_sum += (double)val;
    }
    float cpu_loss = (float)(cpu_sum / N);

    float max_val = std::max(1.0f, std::max(std::abs(gpu_loss), std::abs(cpu_loss)));
    if (std::abs(gpu_loss - cpu_loss) <= 1e-3f * max_val) {
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