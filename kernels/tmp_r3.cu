#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

__global__ void SigmoidActivation(const float* __restrict__ inputs, const float* __restrict__ weights, const float* __restrict__ bias, int numNeurons, int inputSize, float* __restrict__ output) {
    int neuronIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuronIdx >= numNeurons) return;

    const float* weightRow = weights + (size_t)neuronIdx * inputSize;
    float z = 0.0f;
    
    int i = 0;
    #pragma unroll 4
    for (; i <= inputSize - 4; i += 4) {
        float4 w = reinterpret_cast<const float4*>(weightRow + i)[0];
        float4 in = reinterpret_cast<const float4*>(inputs + i)[0];
        z += w.x * in.x + w.y * in.y + w.z * in.z + w.w * in.w;
    }
    for (; i < inputSize; i++) {
        z += weightRow[i] * inputs[i];
    }

    z += bias[neuronIdx];
    output[neuronIdx] = 1.0f / (1.0f + expf(-z));
}

void serialSigmoid(float* inputs, float* weights, float* bias, int numNeurons, int inputSize, float* output){
    for (int j = 0; j < numNeurons; j++) {
        double z = 0.0;
        for (int i = 0; i < inputSize; i++) {
            z += (double)inputs[i] * (double)weights[j * inputSize + i];
        }
        z += (double)bias[j];
        output[j] = 1.0f / (1.0f + exp(-z));
    }
}

int main(){
    int numNeurons = 10000;
    int inputSize = 1024;
    size_t sizeWeights = (size_t)numNeurons * inputSize * sizeof(float);
    size_t sizeInputs = (size_t)inputSize * sizeof(float);
    size_t sizeBias = (size_t)numNeurons * sizeof(float);
    size_t sizeOutput = (size_t)numNeurons * sizeof(float);

    float *h_inputs = (float*)malloc(sizeInputs);
    float *h_weights = (float*)malloc(sizeWeights);
    float *h_bias = (float*)malloc(sizeBias);
    float *h_output = (float*)malloc(sizeOutput);
    float *h_cpu = (float*)malloc(sizeOutput);

    for(int i=0; i<inputSize; i++) h_inputs[i] = (float)rand()/RAND_MAX;
    for(int i=0; i<numNeurons*inputSize; i++) h_weights[i] = ((float)rand()/RAND_MAX) - 0.5f;
    for(int i=0; i<numNeurons; i++) h_bias[i] = (float)rand()/RAND_MAX * 0.01f;

    float *d_inputs, *d_weights, *d_bias, *d_output;
    cudaMalloc(&d_inputs, sizeInputs);
    cudaMalloc(&d_weights, sizeWeights);
    cudaMalloc(&d_bias, sizeBias);
    cudaMalloc(&d_output, sizeOutput);

    cudaMemcpy(d_inputs, h_inputs, sizeInputs, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, sizeWeights, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, sizeBias, cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks = (numNeurons + threads - 1) / threads;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    SigmoidActivation<<<blocks, threads>>>(d_inputs, d_weights, d_bias, numNeurons, inputSize, d_output);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float ms; cudaEventElapsedTime(&ms, start, stop);
    cudaMemcpy(h_output, d_output, sizeOutput, cudaMemcpyDeviceToHost);

    serialSigmoid(h_inputs, h_weights, h_bias, numNeurons, inputSize, h_cpu);

    bool success = true;
    for(int i = 0; i < numNeurons; i++){
        if(std::abs(h_output[i] - h_cpu[i]) > 1e-3f * std::max(1.0f, std::max(std::abs(h_output[i]), std::abs(h_cpu[i])))){
            printf("Error at neuron %d: GPU=%f, CPU=%f\n", i, h_output[i], h_cpu[i]);
            success = false; break;
        }
    }
    if(success) printf("SUCCESS: GPU Time: %f ms\n", ms);

    cudaFree(d_inputs); cudaFree(d_weights); cudaFree(d_bias); cudaFree(d_output);
    free(h_inputs); free(h_weights); free(h_bias); free(h_output); free(h_cpu);
    return 0;
}