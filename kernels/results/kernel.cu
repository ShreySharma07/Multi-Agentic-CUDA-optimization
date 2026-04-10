#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__ void vectorMul(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int vectorLength){
    int workIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (workIndex < vectorLength){
        C[workIndex] = A[workIndex] * B[workIndex];
    }
}

void initArray(float* data, int n) {
    for(int i = 0; i < n; ++i) data[i] = (float)rand() / (float)RAND_MAX;
}

void serialvecmul(float* A, float* B, float* C, int n) {
    for(int i = 0; i < n; ++i) C[i] = A[i] * B[i];
}

bool vectorApproximatelyEqual(float* gpu, float* cpu, int n) {
    for(int i = 0; i < n; ++i) {
        float diff = fabsf(gpu[i] - cpu[i]);
        float max_val = fmaxf(1.0f, fmaxf(fabsf(gpu[i]), fabsf(cpu[i])));
        if (diff > 1e-3f * max_val) {
            printf("ERROR at index %d: GPU=%f CPU=%f DIFF=%f\n", i, gpu[i], cpu[i], diff);
            return false;
        }
    }
    return true;
}

int main(){
    int vectorLength = 1 << 20;
    size_t size = vectorLength * sizeof(float);
    float *A, *B, *C, *comparisonResult;

    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);
    comparisonResult = (float*)malloc(size);

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    int threads = 256;
    int blocks = (vectorLength + threads - 1) / threads;
    
    vectorMul<<<blocks, threads>>>(A, B, C, vectorLength);

    cudaDeviceSynchronize();

    serialvecmul(A, B, comparisonResult, vectorLength);

    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength)){
        printf("SUCCESS\n");
    }
    else{
        printf("FAILURE\n");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

    return 0;
}