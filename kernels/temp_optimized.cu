#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>
#include <chrono>
#include <math.h>

/**
 * Optimized Sigmoid Activation Kernel
 * Strategy: One Warp per Neuron + Vectorized Loads + Shuffle Reduction
 */
__global__ void SigmoidActivationOptimized(
    const float* __restrict__ inputs,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    int numNeurons,
    int inputSize,
    float* __restrict__ output) 
{
    // Each warp (32 threads) handles one neuron
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int laneId = threadIdx.x % 32;

    if (warpId < numNeurons) {
        float sum = 0.0f;
        
        // Offset to the start of this neuron's weight row
        const float* weightRow = weights + (warpId * inputSize);
        
        // Cast to float4 for 128-bit vectorized loads
        // inputSize 1024 / 4 = 256 float4 elements
        const float4* weights4 = reinterpret_cast<const float4*>(weightRow);
        const float4* inputs4  = reinterpret_cast<const float4*>(inputs);
        int numElements4 = inputSize / 4;
        
        // Coalesced parallel dot product within the warp
        for (int i = laneId; i < numElements4; i += 32) {
            float4 w4 = weights4[i];
            float4 i4 = inputs4[i];
            
            sum += w4.x * i4.x;
            sum += w4.y * i4.y;
            sum += w4.z * i4.z;
            sum += w4.w * i4.w;
        }

        // Warp Reduction: sum partial results from all 32 threads in the warp
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // Lane 0 writes the final sigmoid result
        if (laneId == 0) {
            sum += bias[warpId];
            // 1.0f / (1.0f + expf(-z)) using high-performance intrinsics
            output[warpId] = __fdividef(1.0f, 1.0f + __expf(-sum));
        }
    }
}

// --- Host Helper Functions ---

void initRandom(float* data, float min, float max, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    for (int i = 0; i < size; i++) data[i] = dist(gen);
}

void serialSigmoid(float* inputs, float* weights, float* bias, int numNeurons, int inputSize, float* output) {
    for (int j = 0; j < numNeurons; j++) {
        float z = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            z += inputs[i] * weights[j * inputSize + i];
        }
        z += bias[j];
        output[j] = 1.0f / (1.0f + expf(-z));
    }
}

bool verifyResults(float* gpu, float* cpu, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(gpu[i] - cpu[i]);
        if (err > 1e-3) {
            printf("Error at neuron %d: GPU=%f, CPU=%f\n", i, gpu[i], cpu[i]);
            return false;
        }
        if (err > max_err) max_err = err;
    }
    printf("Verification SUCCESS! Max difference: %f\n", max_err);
    return true;
}

int main() {
    const int numNeurons = 10000;
    const int inputSize = 1024; // Power of 2, multiple of 4 and 32

    float *inputs, *weights, *bias, *gpu_output, *cpu_output;

    // Allocate Unified Memory
    cudaMallocManaged(&inputs, inputSize * sizeof(float));
    cudaMallocManaged(&weights, numNeurons * inputSize * sizeof(float));
    cudaMallocManaged(&bias, numNeurons * sizeof(float));
    cudaMallocManaged(&gpu_output, numNeurons * sizeof(float));
    cpu_output = (float*)malloc(numNeurons * sizeof(float));

    // Initialize data
    initRandom(inputs, 0.0f, 1.0f, inputSize);
    initRandom(weights, -0.1f, 0.1f, numNeurons * inputSize);
    initRandom(bias, -0.1f, 0.1f, numNeurons);

    // Configuration
    const int threadsPerBlock = 128; // 4 warps per block
    const int warpsPerBlock = threadsPerBlock / 32;
    const int blocksPerGrid = (numNeurons + warpsPerBlock - 1) / warpsPerBlock;

    // --- Warm-up Run ---
    // This migrates Unified Memory to GPU and avoids first-run latency in timing
    SigmoidActivationOptimized<<<blocksPerGrid, threadsPerBlock>>>(
        inputs, weights, bias, numNeurons, inputSize, gpu_output
    );
    cudaDeviceSynchronize();

    // --- Timed Run ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    SigmoidActivationOptimized<<<blocksPerGrid, threadsPerBlock>>>(
        inputs, weights, bias, numNeurons, inputSize, gpu_output
    );
    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // --- CPU Benchmark ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    serialSigmoid(inputs, weights, bias, numNeurons, inputSize, cpu_output);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_ms = cpu_end - cpu_start;

    // Results
    if (verifyResults(gpu_output, cpu_output, numNeurons)) {
        printf("GPU Time: %f ms\n", ms);
        printf("CPU Time: %f ms\n", cpu_ms.count());
        printf("Speedup: %fx\n", cpu_ms.count() / ms);
    }

    // Cleanup
    cudaFree(inputs);
    cudaFree(weights);
    cudaFree(bias);
    cudaFree(gpu_output);
    free(cpu_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}