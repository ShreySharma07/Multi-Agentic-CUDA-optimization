#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cuda/cmath>
#include <memory.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>
#include <random>
#include <chrono>
#include <cuda_runtime.h>

__global__ void k(){}

__global__ void SigmoidActivation(float* inputs, float* weights, float* bias, int numNeurons, int inputSize, float* output){
    int neuronIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (neuronIdx < numNeurons){
        float z = 0.0f;

        for (int i=0; i<inputSize; i++){
            z += inputs[i] * weights[neuronIdx * inputSize + i];
        }

        z += bias[neuronIdx];

        output[neuronIdx] = 1.0f / (1.0f + expf(-z));
    }
}

void initRandom(float* inputs, float min, float max, int inputSize){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min, max);
    for(int i = 0; i<inputSize; i++){
        inputs[i] = dist(gen);
    }
}

void serialSigmoid(float* inputs, float* weights, float* bias, int numNeurons, int inputSize, float* output){
    for (int j = 0; j < numNeurons; j++) {
        float z = 0.0f;
        for (int i = 0; i < inputSize; i++) {
            z += inputs[i] * weights[j * inputSize + i];
        }
        z += bias[j];
        output[j] = 1.0f / (1.0f + exp(-z)); // Standard CPU exp
    }
}

bool verifyresults(float* gout, float* cout, int n){
    float epsilon = 1e-4;
    for(int i = 0; i<n ; i++){
        if(fabs(gout[i] - cout[i]) > epsilon){
            printf("Error at neuron %d: GPU=%f, CPU=%f\n", i, gout[i], cout[i]);
            return false;
        }
    }
    return true;
}

void UnifiedMemory(int numNeurons, int inputSize){
    float* inputs = nullptr;
    float* weights = nullptr;
    float* bias = nullptr;
    // float* output = nullptr;
    float* cpu_output = nullptr;
    float* gpu_output = nullptr;

    cudaMallocManaged(&inputs, inputSize*sizeof(float));
    cudaMallocManaged(&weights, inputSize*numNeurons*sizeof(float));
    cudaMallocManaged(&bias, numNeurons*sizeof(float));
    // cudaMallocManaged(&output, inputSize*sizeof(float));
    cudaMallocManaged(&gpu_output, numNeurons * sizeof(float));
    cpu_output = (float*)malloc(numNeurons * sizeof(float));

    // 1. Initialize ALL weights (numNeurons * inputSize)
    initRandom(weights, -0.5f, 0.5f, numNeurons * inputSize); 

    // 2. Initialize ALL biases (numNeurons)
    initRandom(bias, 0.0f, 0.01f, numNeurons);

    // 3. Initialize ALL inputs (inputSize)
    initRandom(inputs, 0.0f, 1.0f, inputSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threads = 256;
    int block = (numNeurons + threads - 1) / threads;
    cudaEventRecord(start);
    SigmoidActivation<<<block, threads>>>(inputs, weights, bias,numNeurons, inputSize, gpu_output);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    serialSigmoid(inputs, weights, bias, numNeurons, inputSize, cpu_output);
    auto cpu_stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> cpu_ms = cpu_stop - cpu_start;

    if (verifyresults(gpu_output, cpu_output, numNeurons)) {
        printf("SUCCESS: GPU and CPU results match for %d neurons!\n", numNeurons);
        printf("GPU Time: %f ms\n", gpu_ms);
        printf("CPU Time: %f ms\n", cpu_ms.count());
        printf("Speedup: %fx\n", cpu_ms.count() / gpu_ms);
    } else {
        printf("FAILURE: Results do not match.\n");
    }

    cudaFree(inputs);
    cudaFree(weights);
    cudaFree(bias);
    // cudaFree(output);
    free(cpu_output);
    cudaFree(gpu_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
UnifiedMemory(10000, 1024);
return 0;
}