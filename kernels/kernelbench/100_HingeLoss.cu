#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel to compute element-wise clamped hinge loss components
__global__ void hinge_loss_element_kernel(const float* predictions, const float* targets, float* element_losses, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = 1.0f - predictions[idx] * targets[idx];
        element_losses[idx] = fmaxf(0.0f, val);
    }
}

// Kernel to perform parallel reduction sum
__global__ void reduce_sum_kernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;

    sdata[tid] = 0.0f; // Initialize shared memory for reduction

    // Sum elements in global memory into shared memory
    while (i < N) {
        sdata[tid] += input[i];
        if (i + blockDim.x < N) {
            sdata[tid] += input[i + blockDim.x];
        }
        i += gridSize;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block sum to global memory
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}


float calculate_cpu_hinge_loss(const std::vector<float>& predictions, const std::vector<float>& targets, int N) {
    float total_loss = 0.0f;
    for (int i = 0; i < N; ++i) {
        total_loss += std::fmax(0.0f, 1.0f - predictions[i] * targets[i]);
    }
    return total_loss / N;
}

int main() {
    const int batch_size = 32768;
    const int N = batch_size; // Total number of elements

    // Host memory allocation and initialization
    std::vector<float> h_predictions(N);
    std::vector<float> h_targets(N);

    // Initialize host data
    // PyTorch reference: torch.rand(batch_size), torch.randint(0, 2, (batch_size,)).float() * 2 - 1
    srand(42); // For reproducible results
    for (int i = 0; i < N; ++i) {
        h_predictions[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX); // 0.0f to 1.0f
        h_targets[i] = (rand() % 2 == 0) ? 1.0f : -1.0f; // -1.0f or 1.0f
    }

    // Device memory allocation
    float *d_predictions, *d_targets, *d_element_losses, *d_total_sum;
    CUDA_CHECK(cudaMalloc((void**)&d_predictions, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_targets, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_element_losses, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_total_sum, sizeof(float)));

    // Initialize d_total_sum to 0.0f
    CUDA_CHECK(cudaMemset(d_total_sum, 0, sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_predictions, h_predictions.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_targets, h_targets.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start));

    // Launch element-wise kernel
    hinge_loss_element_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_targets, d_element_losses, N);
    CUDA_CHECK(cudaGetLastError());

    // Launch reduction kernel
    // The shared memory size needed for reduce_sum_kernel is threadsPerBlock * sizeof(float)
    reduce_sum_kernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_element_losses, d_total_sum, N);
    CUDA_CHECK(cudaGetLastError());

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy total sum from device to host
    float h_total_sum_gpu;
    CUDA_CHECK(cudaMemcpy(&h_total_sum_gpu, d_total_sum, sizeof(float), cudaMemcpyDeviceToHost));

    // Calculate final mean on GPU
    float gpu_result = h_total_sum_gpu / N;

    // Calculate CPU reference result
    float cpu_result = calculate_cpu_hinge_loss(h_predictions, h_targets, N);

    // Validate GPU output against CPU reference
    bool success = false;
    float tolerance = 1e-3f * std::max({1.0f, std::fabs(gpu_result), std::fabs(cpu_result)});
    if (std::fabs(gpu_result - cpu_result) <= tolerance) {
        success = true;
    }

    if (success) {
        printf("SUCCESS\n");
    } else {
        printf("FAILURE\n");
        printf("CPU Result: %f, GPU Result: %f\n", cpu_result, gpu_result);
    }
    printf("GPU Time: %f\n", milliseconds);

    // Free device memory
    CUDA_CHECK(cudaFree(d_predictions));
    CUDA_CHECK(cudaFree(d_targets));
    CUDA_CHECK(cudaFree(d_element_losses));
    CUDA_CHECK(cudaFree(d_total_sum));

    // Destroy events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}