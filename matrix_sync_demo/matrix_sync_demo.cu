#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_WIDTH 16
#define MAX_SIZE 1024

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Simple matrix multiplication kernel (no optimization)
__global__ void matmul_naive(float *A, float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Optimized kernel using shared memory and __syncthreads()
__global__ void matmul_shared(float *A, float *B, float *C, int N) {
    // Shared memory tiles
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Calculate global thread position
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (N + TILE_WIDTH - 1) / TILE_WIDTH; tile++) {
        // Load tiles into shared memory
        // Each thread loads one element from A and B
        if (row < N && (tile * TILE_WIDTH + tx) < N) {
            As[ty][tx] = A[row * N + tile * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && (tile * TILE_WIDTH + ty) < N) {
            Bs[ty][tx] = B[(tile * TILE_WIDTH + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // CRITICAL: Wait for all threads in block to finish loading
        // This ensures all threads see the complete tiles before computation
        __syncthreads();
        
        // Compute partial dot product using shared memory
        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        // CRITICAL: Wait for all threads to finish computation before 
        // loading next tile (prevents data races)
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Kernel demonstrating warp divergence issues
__global__ void demonstrate_warp_divergence(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // This creates warp divergence - some threads take if path, others take else
        if (threadIdx.x % 2 == 0) {
            // Even threads do expensive computation
            for (int i = 0; i < 100; i++) {
                data[idx] = data[idx] * 1.001f + 0.001f;
            }
        } else {
            // Odd threads do simple computation
            data[idx] = data[idx] + 1.0f;
        }
    }
    
    // Synchronize to measure total time including divergence
    __syncthreads();
}

// Kernel optimized to avoid warp divergence
__global__ void demonstrate_warp_optimized(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < N) {
        // All threads do the same amount of work to avoid divergence
        if ((idx / 32) % 2 == 0) { // Work assignment per warp, not per thread
            // All threads in this warp do expensive computation
            for (int i = 0; i < 100; i++) {
                data[idx] = data[idx] * 1.001f + 0.001f;
            }
        } else {
            // All threads in this warp do simple computation
            data[idx] = data[idx] + 1.0f;
        }
    }
    
    __syncthreads();
}

// Utility functions
void fill_matrix(float *matrix, int N) {
    for (int i = 0; i < N * N; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

void print_matrix_sample(float *matrix, int N, const char* name) {
    printf("%s (showing 4x4 corner):\n", name);
    for (int i = 0; i < 4 && i < N; i++) {
        for (int j = 0; j < 4 && j < N; j++) {
            printf("%.3f ", matrix[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

double get_time_diff(clock_t start, clock_t end) {
    return ((double)(end - start)) / CLOCKS_PER_SEC * 1000.0; // milliseconds
}

int main(int argc, char *argv[]) {
    // Parse command line arguments
    int N = 512;  // Default matrix size
    int num_gpus = 1;
    
    if (argc > 1) {
        num_gpus = atoi(argv[1]);
        if (num_gpus < 1) num_gpus = 1;
    }
    
    if (argc > 2) {
        N = atoi(argv[2]);
        if (N < 32) N = 32;
        if (N > MAX_SIZE) N = MAX_SIZE;
    }
    
    printf("=== CUDA Thread Synchronization Analysis ===\n");
    printf("Matrix size: %dx%d\n", N, N);
    printf("Using %d GPU(s)\n\n", num_gpus);
    
    // Check available GPUs
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < num_gpus) {
        printf("Warning: Requested %d GPUs but only %d available.\n", num_gpus, device_count);
        num_gpus = device_count;
    }
    
    size_t matrix_size = N * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = (float*)malloc(matrix_size);
    float *h_B = (float*)malloc(matrix_size);
    float *h_C_naive = (float*)malloc(matrix_size);
    float *h_C_shared = (float*)malloc(matrix_size);
    float *h_divergence_data = (float*)malloc(N * sizeof(float));
    
    if (!h_A || !h_B || !h_C_naive || !h_C_shared || !h_divergence_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Initialize matrices
    srand(42); // For reproducible results
    fill_matrix(h_A, N);
    fill_matrix(h_B, N);
    
    // Initialize divergence test data
    for (int i = 0; i < N; i++) {
        h_divergence_data[i] = (float)i;
    }
    
    print_matrix_sample(h_A, N, "Matrix A");
    print_matrix_sample(h_B, N, "Matrix B");
    
    // Run tests on each GPU
    for (int gpu = 0; gpu < num_gpus; gpu++) {
        printf("=== GPU %d Results ===\n", gpu);
        CUDA_CHECK(cudaSetDevice(gpu));
        
        // Get GPU properties
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, gpu));
        printf("Device: %s\n", prop.name);
        printf("Warp size: %d\n", prop.warpSize);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Shared memory per block: %d KB\n", prop.sharedMemPerBlock / 1024);
        
        // Allocate device memory
        float *d_A, *d_B, *d_C, *d_divergence_data;
        CUDA_CHECK(cudaMalloc(&d_A, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_B, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_C, matrix_size));
        CUDA_CHECK(cudaMalloc(&d_divergence_data, N * sizeof(float)));
        
        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_A, h_A, matrix_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, matrix_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_divergence_data, h_divergence_data, N * sizeof(float), cudaMemcpyHostToDevice));
        
        // Configure grid and block dimensions
        dim3 blockDim(16, 16);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
        
        // Test 1: Naive matrix multiplication
        printf("\n--- Test 1: Naive Matrix Multiplication ---\n");
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        matmul_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float naive_time;
        CUDA_CHECK(cudaEventElapsedTime(&naive_time, start, stop));
        printf("Naive kernel time: %.2f ms\n", naive_time);
        
        CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, matrix_size, cudaMemcpyDeviceToHost));
        print_matrix_sample(h_C_naive, N, "Naive Result C");
        
        // Test 2: Shared memory with __syncthreads()
        printf("--- Test 2: Shared Memory with __syncthreads() ---\n");
        CUDA_CHECK(cudaEventRecord(start));
        matmul_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float shared_time;
        CUDA_CHECK(cudaEventElapsedTime(&shared_time, start, stop));
        printf("Shared memory kernel time: %.2f ms\n", shared_time);
        printf("Speedup: %.2fx\n", naive_time / shared_time);
        
        CUDA_CHECK(cudaMemcpy(h_C_shared, d_C, matrix_size, cudaMemcpyDeviceToHost));
        print_matrix_sample(h_C_shared, N, "Shared Memory Result C");
        
        // Verify results match
        bool results_match = true;
        float max_diff = 0.0f;
        for (int i = 0; i < N * N; i++) {
            float diff = fabs(h_C_naive[i] - h_C_shared[i]);
            max_diff = fmax(max_diff, diff);
            if (diff > 1e-3) {
                results_match = false;
            }
        }
        printf("Results match: %s (max diff: %.6f)\n", results_match ? "YES" : "NO", max_diff);
        
        // Test 3: Warp divergence demonstration
        printf("\n--- Test 3: Warp Divergence Analysis ---\n");
        int divergence_threads = 256;
        int divergence_blocks = (N + divergence_threads - 1) / divergence_threads;
        
        // Reset data
        CUDA_CHECK(cudaMemcpy(d_divergence_data, h_divergence_data, N * sizeof(float), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start));
        demonstrate_warp_divergence<<<divergence_blocks, divergence_threads>>>(d_divergence_data, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float divergent_time;
        CUDA_CHECK(cudaEventElapsedTime(&divergent_time, start, stop));
        printf("Warp divergent kernel time: %.2f ms\n", divergent_time);
        
        // Reset data
        CUDA_CHECK(cudaMemcpy(d_divergence_data, h_divergence_data, N * sizeof(float), cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaEventRecord(start));
        demonstrate_warp_optimized<<<divergence_blocks, divergence_threads>>>(d_divergence_data, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float optimized_time;
        CUDA_CHECK(cudaEventElapsedTime(&optimized_time, start, stop));
        printf("Warp optimized kernel time: %.2f ms\n", optimized_time);
        printf("Divergence penalty: %.2fx slower\n", divergent_time / optimized_time);
        
        // Cleanup GPU resources
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaFree(d_divergence_data));
        
        printf("\n");
    }
    
    printf("=== Key Synchronization Insights ===\n");
    printf("1. __syncthreads() ensures all threads in a block reach the same point\n");
    printf("2. Shared memory + sync enables data reuse and reduces global memory access\n");
    printf("3. Warp divergence occurs when threads in a warp take different code paths\n");
    printf("4. Proper thread scheduling can hide memory latency\n");
    printf("5. SIMD execution within warps means divergence hurts performance\n");
    
    // Cleanup host memory
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_shared);
    free(h_divergence_data);
    
    return 0;
}

/*
Compilation and Usage:
make clean && make && ./matrix_sync_demo [num_gpus] [matrix_size]

Examples:
./matrix_sync_demo 1 512          # 1 GPU, 512x512 matrices
./matrix_sync_demo 2 1024         # 2 GPUs, 1024x1024 matrices
*/
