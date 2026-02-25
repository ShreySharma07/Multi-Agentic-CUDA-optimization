#include <iostream>
#include <ctime>
#include <cstdlib>
#include <cuda/cmath>
#include <memory.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <math.h>

__global__ void SigmoidActivation(float* inputs, float* weights, float* bias, int numNeurons, int inputSize, float* output){
    int neuronIdx = threadIdx.x + blockIdx.x * blockDim.x;

    if (neuronIdx < numNeurons){
        z = 0.0f;

        for (int i=0; i<inputSize; i++){
            z += inputs[i] * weights[neuronIdx * numNeurons + i];
        }

        z += bias[neuronIdx];

        output = 1.0f / (1.0f + expf(-z));
    }
}

void initArray(float* inputs, float* weights, float* bias, int inputSize){
    for(int i = 0; i<inputSize; i++){
        
    }
}