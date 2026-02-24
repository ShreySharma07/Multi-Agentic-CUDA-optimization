#include <iostream>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda/cmath>
#include <cstdlib>
#include <ctime>
#include <memory.h>


__global__ void vecMUL(float* A, float* B, float* C, int vectorLength){
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < vectorLength){
        C[workIndex] = A[workIndex] * B[workIndex];
    }
}

void initArray(float* A, int length){
    std::srand(std::time({}));
    for(int i = 0; i<length; i++){
        A[i] = (float)rand() / (float)RAND_MAX;
    }
}

void serialvecMUL(float* A, float* B, float* C, int length){
    for(int i=0; i<length; i++){
        C[i] = A[i] * B[i];
    }
}

bool comparisonResult(float* A, float* B, int length){
    float epsilon = 0.00001f;
    for(int i=0; i<length; i++){
        if(std::abs(A[i] - B[i]) > epsilon){
            printf("Index %d Mismatch %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

void unifiedmemory(int vectorLength){
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* compare_result = (float*)malloc(vectorLength*sizeof(float));

    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    initArray(A, vectorLength);
    initArray(B, vectorLength);
    
    int Threads = 256;
    int block = cuda::ceil_div(vectorLength, Threads);
    vecMUL<<<block, Threads>>>(A, B, C, vectorLength);

    cudaDeviceSynchronize();

    serialvecMUL(A, B, compare_result, vectorLength);

    if(comparisonResult(C, compare_result, vectorLength)){
        printf("Both GPU and CPU output the same result \n");
    }
    else{
        printf("GPU and CPU have different result");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(compare_result);
}

int main(int argc, char** argv){
    int vectorLength = 1024;
    if (argc >= 2){
        vectorLength = std::atoi(argv[1]);
    }
    unifiedmemory(vectorLength);
    return 0;
}