#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib> // For rand()

// CUDA kernel for vector addition
// Added __restrict__ keyword to inform the compiler that pointers do not alias,
// which can enable more aggressive optimizations related to memory access.
__global__ void vectorAdd(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int numElements = 50000;
    size_t size = numElements * sizeof(float);

    // Allocate host memory using pinned memory for faster transfers
    float *h_A = nullptr;
    float *h_B = nullptr;
    float *h_C = nullptr;
    cudaMallocHost((void **)&h_A, size);
    cudaMallocHost((void **)&h_B, size);
    cudaMallocHost((void **)&h_C, size);

    // Initialize host arrays
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Create a CUDA stream for asynchronous operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy data from host to device asynchronously
    cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream);

    // Launch the kernel asynchronously on the created stream
    // Using 1024 threads per block for potentially better occupancy on Ampere
    int threadsPerBlock = 1024;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, numElements);

    // Copy result back to host asynchronously
    cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream);

    // Wait for all operations in the stream to complete
    cudaStreamSynchronize(stream);

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    // Destroy the CUDA stream
    cudaStreamDestroy(stream);

    std::cout << "Vector addition completed successfully!" << std::endl;
    return 0;
}