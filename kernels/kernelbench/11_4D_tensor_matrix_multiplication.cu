#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_DIM 16

__global__ void einsum_4d_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int b, int i, int j, int l, int k) {
    int b_idx = blockIdx.z;
    int k_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    int ij_idx = blockIdx.y * TILE_DIM + threadIdx.y;
    int i_idx = ij_idx / j;
    int j_idx = ij_idx % j;

    if (b_idx < b && i_idx < i && j_idx < j && k_idx < k) {
        float sum = 0.0f;
        int offset_A = ((b_idx * i + i_idx) * j + j_idx) * l;
        for (int l_idx = 0; l_idx < l; ++l_idx) {
            sum += A[offset_A + l_idx] * B[l_idx * k + k_idx];
        }
        C[((b_idx * i + i_idx) * j + j_idx) * k + k_idx] = sum;
    }
}

int main() {
    const int b = 2, i = 16, j = 32, l = 64, k = 32;
    size_t size_A = (size_t)b * i * j * l * sizeof(float);
    size_t size_B = (size_t)l * k * sizeof(float);
    size_t size_C = (size_t)b * i * j * k * sizeof(float);

    std::vector<float> h_A(b * i * j * l);
    std::vector<float> h_B(l * k);
    std::vector<float> h_C(b * i * j * k);
    std::vector<float> h_C_gpu(b * i * j * k);

    for (auto& v : h_A) v = (float)rand() / RAND_MAX;
    for (auto& v : h_B) v = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((k + TILE_DIM - 1) / TILE_DIM, (i * j + TILE_DIM - 1) / TILE_DIM, b);

    einsum_4d_kernel<<<grid, block>>>(d_A, d_B, d_C, b, i, j, l, k);

    cudaMemcpy(h_C_gpu.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    for (int b_idx = 0; b_idx < b; ++b_idx) {
        for (int i_idx = 0; i_idx < i; ++i_idx) {
            for (int j_idx = 0; j_idx < j; ++j_idx) {
                for (int k_idx = 0; k_idx < k; ++k_idx) {
                    float sum = 0.0f;
                    for (int l_idx = 0; l_idx < l; ++l_idx) {
                        sum += h_A[((b_idx * i + i_idx) * j + j_idx) * l + l_idx] * h_B[l_idx * k + k_idx];
                    }
                    h_C[((b_idx * i + i_idx) * j + j_idx) * k + k_idx] = sum;
                }
            }
        }
    }

    bool success = true;
    for (size_t idx = 0; idx < h_C.size(); ++idx) {
        float diff = std::abs(h_C_gpu[idx] - h_C[idx]);
        float ref = std::abs(h_C[idx]);
        if (diff > 1e-3f * std::max(1.0f, ref)) {
            success = false;
            break;
        }
    }

    std::cout << (success ? "SUCCESS" : "FAILURE") << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}