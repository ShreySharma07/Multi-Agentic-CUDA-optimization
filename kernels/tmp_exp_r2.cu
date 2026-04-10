#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define TILE_DIM 32

__global__ void einsum_4d_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int b, int i, int j, int l, int k) {
    extern __shared__ float s_B[];

    int b_idx = blockIdx.z;
    int ij_base = blockIdx.y * TILE_DIM;
    int k_base = blockIdx.x * TILE_DIM;

    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;

    float sum = 0.0f;

    for (int l_block = 0; l_block < l; l_block += TILE_DIM) {
        if (l_block + tid_y < l && k_base + tid_x < k) {
            s_B[tid_y * TILE_DIM + tid_x] = B[(l_block + tid_y) * k + (k_base + tid_x)];
        } else {
            s_B[tid_y * TILE_DIM + tid_x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll
        for (int l_sub = 0; l_sub < TILE_DIM; ++l_sub) {
            int l_idx = l_block + l_sub;
            if (l_idx < l) {
                int ij_idx = ij_base + tid_y;
                if (ij_idx < i * j) {
                    float a_val = A[(((b_idx * i + (ij_idx / j)) * j + (ij_idx % j)) * l + l_idx)];
                    sum += a_val * s_B[l_sub * TILE_DIM + tid_x];
                }
            }
        }
        __syncthreads();
    }

    int ij_out = ij_base + tid_y;
    int i_out = ij_out / j;
    int j_out = ij_out % j;
    int k_out = k_base + tid_x;

    if (b_idx < b && i_out < i && j_out < j && k_out < k) {
        C[(((b_idx * i + i_out) * j + j_out) * k + k_out)] = sum;
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

    einsum_4d_kernel<<<grid, block, TILE_DIM * TILE_DIM * sizeof(float)>>>(d_A, d_B, d_C, b, i, j, l, k);

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
        float diff = h_C_gpu[idx] - h_C[idx];
        if (diff < 0) diff = -diff;
        float ref = h_C[idx] < 0 ? -h_C[idx] : h_C[idx];
        if (diff > 1e-3f * (ref > 1.0f ? ref : 1.0f)) {
            success = false;
            break;
        }
    }

    std::cout << (success ? "SUCCESS" : "FAILURE") << std::endl;

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}