#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    const int b = 8, i = 256, j = 512, l = 256, k = 768;
    const size_t A_size = (size_t)b * i * j * l;
    const size_t B_size = (size_t)l * k;
    const size_t C_size = (size_t)b * i * j * k;

    std::vector<float> hA(A_size), hB(B_size), hC(C_size);
    for(auto& x : hA) x = static_cast<float>(rand()) / RAND_MAX;
    for(auto& x : hB) x = static_cast<float>(rand()) / RAND_MAX;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, A_size * sizeof(float));
    cudaMalloc(&dB, B_size * sizeof(float));
    cudaMalloc(&dC, C_size * sizeof(float));

    cudaMemcpy(dA, hA.data(), A_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), B_size * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // The operation "bijl, lk -> bijk" can be viewed as (b*i*j, l) x (l, k) -> (b*i*j, k)
    // Cublas performs matmul: C = A * B.
    // Matrix A is (b*i*j) rows and l columns.
    // Matrix B is l rows and k columns.
    // cublasSgemm uses column-major.
    // For Row-Major B[l, k] and A[b*i*j, l], we use:
    // C(Row-Major) = A(Row-Major) * B(Row-Major)
    // cublasSgemm(..., transA=N, transB=N, N=k, M=b*i*j, K=l, alpha, B(k, l), lda=k, A(b*i*j, l), ldb=l, beta, C(b*i*j, k), ldc=k)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, k, b * i * j, l, &alpha, dB, k, dA, l, &beta, dC, k);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaMemcpy(hC.data(), dC, C_size * sizeof(float), cudaMemcpyDeviceToHost);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    bool success = true;
    for(int n = 0; n < b * i * j; ++n) {
        for(int col = 0; col < k; ++col) {
            float sum = 0.0f;
            for(int row = 0; row < l; ++row) {
                sum += hA[n * l + row] * hB[row * k + col];
            }
            if(fabsf(hC[n * k + col] - sum) > 1e-3f * fmaxf(1.0f, fmaxf(fabsf(hC[n * k + col]), fabsf(sum)))) {
                success = false;
                break;
            }
        }
        if(!success) break;
    }

    if(success) printf("SUCCESS\n"); else printf("FAILURE\n");
    printf("GPU Time: %f\n", ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(handle);
    return 0;
}