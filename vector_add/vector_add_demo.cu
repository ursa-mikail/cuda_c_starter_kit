#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024  // Vector size
#define THREADS_PER_BLOCK 256

// CUDA Kernel: Add two vectors
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    __shared__ float sA[THREADS_PER_BLOCK];
    __shared__ float sB[THREADS_PER_BLOCK];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Load data into shared memory
        sA[threadIdx.x] = A[idx];
        sB[threadIdx.x] = B[idx];

        __syncthreads();  // Ensure all threads have loaded data

        // Compute addition
        C[idx] = sA[threadIdx.x] + sB[threadIdx.x];
    }
}

int main() {
    int size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vectorAdd<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify result
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Error at index %d\n", i);
            return -1;
        }
    }
    printf("Vector addition successful!\n");

    // Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
