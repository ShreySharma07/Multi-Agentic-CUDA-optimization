#include <iostream>

using namespace std;

//kernel definition
__global__ void vectorMul(float* A, float* B, float* C, int vectorLength){
            int workIndex = threadIdx.x + blockIdx.x * blockDim.x
            if (workIndex < vectorLength){
                C[workIndex] = A[workIndex] * B[workIndex];
            }
}

void unifiedmemory(int vectorLength){
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    //Unified Memory Management
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&c, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    int Threads = 256;
    // int blocks = (vectorLength - Threads - 1)/Threads;
    int blocks = cuda::ceil_div(vectorLength, Threads);
    vectorMul<<<blocks, Threads>>>(devA, devB, devC, vectorLength);

    cudaDeviceSynchronize();

    serialvecmul(A, B, comparisonResult, vectorLength);

    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength)){
        printf("the result of both CPU and GPU Matched!");
    }
    else{
        printf("The results of both the CPU and GPU do not match");
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}


int main(){
        // vectorMul<<<1, 256>>>(A, B,C);
        // dim3 grid(16, 16);
        // dim3 block(8, 8);
        int Threads = 256;
        // int blocks = (vectorLength - Threads - 1)/Threads;
        int blocks = cuda::ceil_div(vectorLength, Threads);
        vectorMul<<<blocks, Threads>>>(devA, devB, devC, vectorLength);

}