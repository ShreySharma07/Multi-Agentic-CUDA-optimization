#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <cstdlib>

__global__ void hinge_loss_kernel(const float* __restrict__ predictions, const float* __restrict__ targets, double* global_sum, size_t total_elements, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;

    const float4* preds4 = (const float4*)predictions;
    const float4* targets4 = (const float4*)targets;
    size_t total_elements4 = total_elements / 4;
    size_t N4 = N / 4;

    size_t col_idx = idx % N4;
    float4 t = targets4[col_idx];

    // Using float accumulators for maximum FP32 throughput on sm_86.
    // Loop unrolling to increase Instruction Level Parallelism (ILP).
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;

    #pragma unroll 4
    for (size_t i = idx; i < total_elements4; i += stride) {
        float4 p = preds4[i];
        
        // fmaf: fused multiply-add for compute efficiency
        // fmaxf: branchless compute instruction
        acc0 += fmaxf(0.0f, fmaf(-p.x, t.x, 1.0f));
        acc1 += fmaxf(0.0f, fmaf(-p.y, t.y, 1.0f));
        acc2 += fmaxf(0.0f, fmaf(-p.z, t.z, 1.0f));
        acc3 += fmaxf(0.0f, fmaf(-p.w, t.w, 1.0f));
    }

    // Accumulate everything into double before the reduction to maintain precision
    double thread_sum = (double)acc0 + (double)acc1 + (double)acc2 + (double)acc3;

    __shared__ double sdata[256];
    unsigned int tid = threadIdx.x;
    sdata[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        double val = sdata[tid];
        val += sdata[tid + 32];
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        if (tid == 0) {
            atomicAdd(global_sum, val);
        }
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