#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

__global__ void vecMUL(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int vectorLength){
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (tid + 3 < vectorLength){
        float4 a_vec = reinterpret_cast<const float4*>(A)[(blockIdx.x * blockDim.x + threadIdx.x)];
        float4 b_vec = reinterpret_cast<const float4*>(B)[(blockIdx.x * blockDim.x + threadIdx.x)];
        float4 res;
        res.x = a_vec.x * b_vec.x;
        res.y = a_vec.y * b_vec.y;
        res.z = a_vec.z * b_vec.z;
        res.w = a_vec.w * b_vec.w;
        reinterpret_cast<float4*>(C)[(blockIdx.x * blockDim.x + threadIdx.x)] = res;
    } else {
        for (int i = tid; i < tid + 4 && i < vectorLength; i++) {
            C[i] = A[i] * B[i];
        }
    }
}

void initArray(float* A, int length){
    for(int i = 0; i < length; i++){
        A[i] = (float)rand() / (float)RAND_MAX;
    }
}

void serialvecMUL(float* A, float* B, float* C, int length){
    for(int i = 0; i < length; i++){
        C[i] = A[i] * B[i];
    }
}

bool comparisonResult(float* gpu, float* cpu, int length){
    for(int i = 0; i < length; i++){
        float diff = fabsf(gpu[i] - cpu[i]);
        float max_val = fmaxf(1.0f, fmaxf(fabsf(gpu[i]), fabsf(cpu[i])));
        if(diff > 1e-3f * max_val){
            printf("ERROR at index %d: GPU=%f CPU=%f DIFF=%f\n", i, gpu[i], cpu[i], diff);
            return false;
        }
    }
    printf("SUCCESS\n");
    return true;
}

int main(int argc, char** argv){
    int vectorLength = 1048576;
    if (argc >= 2) vectorLength = atoi(argv[1]);
    
    size_t size = vectorLength * sizeof(float);
    float *A = (float*)malloc(size);
    float *B = (float*)malloc(size);
    float *C_cpu = (float*)malloc(size);
    float *C_gpu = (float*)malloc(size);

    initArray(A, vectorLength);
    initArray(B, vectorLength);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (vectorLength / 4 + threads - 1) / threads;
    vecMUL<<<blocks, threads>>>(d_A, d_B, d_C, vectorLength);

    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    serialvecMUL(A, B, C_cpu, vectorLength);
    comparisonResult(C_gpu, C_cpu, vectorLength);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(A); free(B); free(C_cpu); free(C_gpu);
    return 0;
}