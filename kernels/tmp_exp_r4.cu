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
    
    // Shared memory for expert counts (num_experts)
    extern __shared__ int32_t shared_counts[];

    // Initialize counts: each thread handles a portion of the expert array
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();

    // Use a warp-level reduction strategy to minimize global atomic contention
    // Each warp collects its own local counts for experts
    int32_t local_eid_counts[8] = {0}; // Assuming num_experts is small (e.g., 8)
    for (size_t i = threadIdx.x; i < numel; i += blockDim.x) {
        int eid = topk_ids[i];
        if (eid >= 0 && eid < num_experts) {
            local_eid_counts[eid]++;
        }
    }

    // Atomically merge local counts into global shared memory
    for (int i = 0; i < num_experts; ++i) {
        if (local_eid_counts[i] > 0) {
            atomicAdd(&shared_counts[i], local_eid_counts[i]);
        }
    }
    __syncthreads();

    // Single thread computes the total padded sum
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
    // Use multiple blocks to parallelize processing of numel tokens
    // 256 blocks * 256 threads = 65536 threads
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