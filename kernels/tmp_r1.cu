#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <cuda_runtime.h>
#include <algorithm> // For std::max

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error at %s:%d: %s %s\n", file, line, func, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// CUDA kernel for vector addition with restrict keyword
__global__ void vectorAdd(const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 5000000;
    size_t size = numElements * sizeof(float);

    // Host memory allocation
    std::vector<float> h_A(numElements);
    std::vector<float> h_B(numElements);
    std::vector<float> h_C_gpu(numElements);
    std::vector<float> h_C_cpu(numElements);

    // Initialize host arrays and perform CPU reference calculation
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
        h_C_cpu[i] = h_A[i] + h_B[i]; // CPU reference
    }

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // Event for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Launch the kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

    CHECK_CUDA_ERROR(cudaEventRecord(start));
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
    CHECK_CUDA_ERROR(cudaGetLastError()); // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaEventRecord(stop));

    // Wait for GPU to finish and measure time
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_C_gpu.data(), d_C, size, cudaMemcpyDeviceToHost));

    // Validate GPU output against CPU reference
    bool success = true;
    float tolerance = 1e-3;
    for (int i = 0; i < numElements; ++i) {
        float diff = std::abs(h_C_gpu[i] - h_C_cpu[i]);
        float max_val = std::max({1.0f, std::abs(h_C_gpu[i]), std::abs(h_C_cpu[i])});
        if (diff > tolerance * max_val) {
            success = false;
            break;
        }
    }

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));

    // Destroy events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    if (success) {
        printf("SUCCESS\n");
    } else {
        printf("FAILURE\n");
    }
    printf("GPU Time: %f\n", milliseconds);

    return 0;
}