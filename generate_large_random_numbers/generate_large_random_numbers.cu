#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <gmp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define BITS 4000
#define LIMBS (BITS/32)

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel to generate random numbers
__global__ void generate_random_numbers(uint32_t *output, int num_numbers, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_numbers) {
        // Initialize random state
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        // Generate random 32-bit words for this thread's number
        uint32_t *my_output = output + idx * LIMBS;
        for (int i = 0; i < LIMBS; i++) {
            my_output[i] = curand(&state);
        }
        
        // Set the highest bit to ensure the number has full bit length
        my_output[LIMBS-1] |= 0x80000000;
        
        // Make the number odd by setting the lowest bit
        my_output[0] |= 1;
    }
}

void print_random_number(uint32_t *limbs) {
    mpz_t number;
    mpz_init(number);
    
    // Import the limbs into GMP
    mpz_import(number, LIMBS, -1, sizeof(uint32_t), 0, 0, limbs);
    
    // Print the number
    gmp_printf("%Zd\n", number);
    
    // Clean up
    mpz_clear(number);
}

int main(int argc, char *argv[]) {
    // Default values
    int num_gpus = 1;
    int num_numbers = 1; // Number of random numbers to generate
    
    // Parse command line arguments
    if (argc > 1) {
        num_gpus = atoi(argv[1]);
        if (num_gpus < 1) num_gpus = 1;
    }
    
    if (argc > 2) {
        num_numbers = atoi(argv[2]);
        if (num_numbers < 1) num_numbers = 1;
    }
    
    printf("Using %d GPU(s) to generate %d random numbers of %d bits each\n", 
           num_gpus, num_numbers, BITS);
    
    // Check available GPUs
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < num_gpus) {
        printf("Warning: Requested %d GPUs but only %d available. Using %d GPUs.\n", 
               num_gpus, device_count, device_count);
        num_gpus = device_count;
    }
    
    // Calculate numbers per GPU
    int numbers_per_gpu = (num_numbers + num_gpus - 1) / num_gpus;
    
    // Allocate host memory for all random numbers
    uint32_t *host_numbers = (uint32_t*)malloc(num_numbers * LIMBS * sizeof(uint32_t));
    if (!host_numbers) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }
    
    // Seed for random number generation
    unsigned long long seed = time(NULL);
    
    // Generate numbers on each GPU
    int numbers_generated = 0;
    
    for (int gpu = 0; gpu < num_gpus && numbers_generated < num_numbers; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        
        // Calculate how many numbers to generate on this GPU
        int gpu_numbers = numbers_per_gpu;
        if (numbers_generated + gpu_numbers > num_numbers) {
            gpu_numbers = num_numbers - numbers_generated;
        }
        
        // Allocate device memory
        uint32_t *dev_numbers;
        CUDA_CHECK(cudaMalloc(&dev_numbers, gpu_numbers * LIMBS * sizeof(uint32_t)));
        
        // Configure kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (gpu_numbers + threadsPerBlock - 1) / threadsPerBlock;
        
        // Launch kernel
        generate_random_numbers<<<blocksPerGrid, threadsPerBlock>>>(dev_numbers, gpu_numbers, seed + gpu);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(host_numbers + numbers_generated * LIMBS, 
                              dev_numbers, 
                              gpu_numbers * LIMBS * sizeof(uint32_t), 
                              cudaMemcpyDeviceToHost));
        
        // Free device memory
        CUDA_CHECK(cudaFree(dev_numbers));
        
        numbers_generated += gpu_numbers;
    }
    
    // Print all generated numbers
    printf("\nGenerated %d-bit random numbers:\n", BITS);
    for (int i = 0; i < num_numbers; i++) {
        printf("%d: ", i+1);
        print_random_number(host_numbers + i * LIMBS);
    }
    
    // Clean up
    free(host_numbers);
    
    return 0;
}

/* 
make // with Makefile

# Basic usage (1 GPU, 1 number)
./gen_large_rand

# Specify number of GPUs (e.g., 2 GPUs)
./gen_large_rand 2

# Specify number of GPUs and random numbers (e.g., 2 GPUs, 5 numbers)
./gen_large_rand 2 5
*/