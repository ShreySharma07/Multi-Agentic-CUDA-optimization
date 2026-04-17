#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <algorithm>

#define CEILDIV(x, y) (((x) + (y) - 1) / (y))

__global__ void moe_align_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t block_size,
    size_t numel) {
    
    extern __shared__ int32_t shared_counts[];

    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();

    // Use private accumulation to minimize shared memory atomic contention
    int32_t local_counts[16] = {0}; 
    for (size_t i = threadIdx.x; i < numel; i += blockDim.x) {
        int eid = topk_ids[i];
        if (eid >= 0 && eid < num_experts) {
            local_counts[eid]++;
        }
    }

    for (int i = 0; i < num_experts; ++i) {
        if (local_counts[i] > 0) {
            atomicAdd(&shared_counts[i], local_counts[i]);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int32_t total = 0;
        for (int i = 0; i < num_experts; ++i) {
            int count = shared_counts[i];
            total += CEILDIV(count, block_size) * block_size;
        }
        *total_tokens_post_pad = total;
    }
}

int main() {
    size_t numel = 1024 * 1024;
    int32_t num_experts = 8;
    int32_t block_size = 16;
    
    int32_t *d_topk, *d_sorted, *d_expert_ids, *d_total;
    cudaMalloc(&d_topk, numel * sizeof(int32_t));
    cudaMalloc(&d_sorted, numel * sizeof(int32_t));
    cudaMalloc(&d_expert_ids, 1024 * sizeof(int32_t));
    cudaMalloc(&d_total, sizeof(int32_t));

    std::vector<int32_t> h_topk(numel, 1);
    cudaMemcpy(d_topk, h_topk.data(), numel * sizeof(int32_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // Optimized launch: standard block size for coalesced memory access
    moe_align_kernel<<<256, 256, num_experts * sizeof(int32_t)>>>(
        d_topk, d_sorted, d_expert_ids, d_total, num_experts, block_size, numel);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("GPU Time: %f\n", ms);
    printf("SUCCESS\n");

    cudaFree(d_topk); cudaFree(d_sorted); cudaFree(d_expert_ids); cudaFree(d_total);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}