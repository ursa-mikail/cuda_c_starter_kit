#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <omp.h>
#define N 1024
#define TILE_SIZE 32
#define LUT_SIZE 256  // Small lookup table in constant memory

// Constant memory example: lookup table
__constant__ float constLUT[LUT_SIZE];
__constant__ int constN;  // Storing matrix size as constant

// Global memory (input/output matrices)
__global__ void matMulGlobal(const float *A, const float *B, float *C, int n) {
    // Registers / local variables: sum for each thread
    float sum = 0.0f;  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        for (int k = 0; k < n; k++) {
            // Use global memory loads
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum; // Each thread writes in parallel
    }
}

// Shared memory + constant memory version
__global__ void matMulSharedConst(const float *A, const float *B, float *C) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    float sum = 0.0f;  
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Copy constant memory to a register first
    int n = constN;  
    
    for (int tile = 0; tile < n / TILE_SIZE; tile++) {
        tileA[threadIdx.y][threadIdx.x] = A[row * n + tile * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(tile * TILE_SIZE + threadIdx.y) * n + col];
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x] * constLUT[k % LUT_SIZE];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n)
        C[row * n + col] = sum;
}

int main() {
    int size = N * N * sizeof(float);
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    
    // Initialize matrices with random values in parallel
    #pragma omp parallel for
    for (int i = 0; i < N*N; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }
    
    // Initialize constant lookup table
    float h_LUT[LUT_SIZE];
    for (int i = 0; i < LUT_SIZE; i++) h_LUT[i] = (float)(i + 1) / LUT_SIZE;
    
    cudaMemcpyToSymbol(constLUT, h_LUT, LUT_SIZE * sizeof(float));
    
    // FIX: Create a variable to hold N since we can't take address of a #define
    int host_N = N;
    cudaMemcpyToSymbol(constN, &host_N, sizeof(int));
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(N / TILE_SIZE, N / TILE_SIZE);
    
    float ms;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Global memory kernel
    cudaEventRecord(start);
    matMulGlobal<<<blocks, threads>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Global memory kernel: %f ms\n", ms);
    
    // Shared + constant memory kernel
    cudaEventRecord(start);
    matMulSharedConst<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf("Shared + constant memory kernel: %f ms\n", ms);
    
    // Copy back and verify
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    
    double checksum = 0.0;
    #pragma omp parallel for reduction(+:checksum)
    for (int i = 0; i < N*N; i++) checksum += h_C[i];
    printf("Checksum: %f\n", checksum);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A); 
    free(h_B); 
    free(h_C);
    
    return 0;
}