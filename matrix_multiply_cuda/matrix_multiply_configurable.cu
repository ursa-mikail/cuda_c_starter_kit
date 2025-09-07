#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <omp.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>
#include "matrix_config.h"

// CUDA error checking macro
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel to generate random matrix elements
__global__ void generate_random_matrix(float *matrix, int rows, int cols, 
                                      float min_val, float max_val, unsigned long long seed) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        
        // Initialize random state for this thread
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        // Generate random float in specified range
        float range = max_val - min_val;
        matrix[idx] = min_val + curand_uniform(&state) * range;
    }
}

// Matrix multiplication kernel with configurable dimensions
__global__ void matrix_multiply_kernel(float *d_M, float *d_N, float *d_P, 
                                     int M_rows, int M_cols, int N_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M_rows && col < N_cols) {
        float product_val = 0.0f;
        
        int k = 0;
        while (k < M_cols) {
            product_val += d_M[row * M_cols + k] * d_N[k * N_cols + col];
            k++;
        }
        
        d_P[row * N_cols + col] = product_val;
    }
}

// Configuration management functions
MatrixConfig load_config_from_file(const char* filename) {
    FILE *file = fopen(filename, "r");
    MatrixConfig config = DEFAULT_CONFIG;
    
    if (!file) {
        printf("Config file %s not found, using defaults\n", filename);
        return config;
    }
    
    // Check return values to suppress warnings
    if (fscanf(file, "M_rows=%d\n", &config.M_rows) != 1) config.M_rows = DEFAULT_CONFIG.M_rows;
    if (fscanf(file, "M_cols=%d\n", &config.M_cols) != 1) config.M_cols = DEFAULT_CONFIG.M_cols;
    if (fscanf(file, "N_rows=%d\n", &config.N_rows) != 1) config.N_rows = DEFAULT_CONFIG.N_rows;
    if (fscanf(file, "N_cols=%d\n", &config.N_cols) != 1) config.N_cols = DEFAULT_CONFIG.N_cols;
    if (fscanf(file, "min_value=%f\n", &config.min_value) != 1) config.min_value = DEFAULT_CONFIG.min_value;
    if (fscanf(file, "max_value=%f\n", &config.max_value) != 1) config.max_value = DEFAULT_CONFIG.max_value;
    if (fscanf(file, "block_size=%d\n", &config.block_size) != 1) config.block_size = DEFAULT_CONFIG.block_size;
    if (fscanf(file, "num_tasks=%d\n", &config.num_tasks) != 1) config.num_tasks = DEFAULT_CONFIG.num_tasks;
    if (fscanf(file, "num_gpus=%d\n", &config.num_gpus) != 1) config.num_gpus = DEFAULT_CONFIG.num_gpus;
    
    // Update dependent dimensions
    config.P_rows = config.M_rows;
    config.P_cols = config.N_cols;
    
    fclose(file);
    printf("Configuration loaded from %s\n", filename);
    return config;
}

void save_config_to_file(const MatrixConfig* config, const char* filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Cannot create config file %s\n", filename);
        return;
    }
    
    fprintf(file, "M_rows=%d\n", config->M_rows);
    fprintf(file, "M_cols=%d\n", config->M_cols);
    fprintf(file, "N_rows=%d\n", config->N_rows);
    fprintf(file, "N_cols=%d\n", config->N_cols);
    fprintf(file, "min_value=%.2f\n", config->min_value);
    fprintf(file, "max_value=%.2f\n", config->max_value);
    fprintf(file, "block_size=%d\n", config->block_size);
    fprintf(file, "num_tasks=%d\n", config->num_tasks);
    fprintf(file, "num_gpus=%d\n", config->num_gpus);
    
    fclose(file);
    printf("Configuration saved to %s\n", filename);
}

MatrixConfig randomize_config_dimensions(int min_dim, int max_dim, int num_tasks, int num_gpus) {
    MatrixConfig config = DEFAULT_CONFIG;
    
    srand(time(NULL));
    
    config.M_rows = min_dim + rand() % (max_dim - min_dim + 1);
    config.M_cols = min_dim + rand() % (max_dim - min_dim + 1);
    config.N_rows = config.M_cols; // Must match for multiplication
    config.N_cols = min_dim + rand() % (max_dim - min_dim + 1);
    
    config.P_rows = config.M_rows;
    config.P_cols = config.N_cols;
    
    config.num_tasks = num_tasks;
    config.num_gpus = num_gpus;
    
    // Adjust block size based on matrix dimensions
    if (config.M_rows < 128 || config.N_cols < 128) {
        config.block_size = 8;
    } else if (config.M_rows < 512 || config.N_cols < 512) {
        config.block_size = 16;
    } else {
        config.block_size = 32;
    }
    
    printf("Generated random configuration:\n");
    print_config(&config);
    
    return config;
}

void print_config(const MatrixConfig* config) {
    printf("Matrix Configuration:\n");
    printf("  Matrix M: %d x %d\n", config->M_rows, config->M_cols);
    printf("  Matrix N: %d x %d\n", config->N_rows, config->N_cols);
    printf("  Result P: %d x %d\n", config->P_rows, config->P_cols);
    printf("  Value range: [%.2f, %.2f]\n", config->min_value, config->max_value);
    printf("  CUDA block size: %d x %d\n", config->block_size, config->block_size);
    printf("  Tasks: %d, GPUs: %d\n", config->num_tasks, config->num_gpus);
    printf("  File prefix: %s\n", config->file_prefix);
    
    // Memory estimation
    size_t M_size = config->M_rows * config->M_cols * sizeof(float);
    size_t N_size = config->N_rows * config->N_cols * sizeof(float);
    size_t P_size = config->P_rows * config->P_cols * sizeof(float);
    size_t total_per_task = M_size + N_size + P_size;
    
    printf("  Memory per task: %.2f MB\n", total_per_task / (1024.0 * 1024.0));
    printf("  Total memory (all tasks): %.2f MB\n", 
           (total_per_task * config->num_tasks) / (1024.0 * 1024.0));
}

int validate_config(const MatrixConfig* config) {
    if (config->M_cols != config->N_rows) {
        fprintf(stderr, "Error: M_cols (%d) must equal N_rows (%d) for multiplication\n", 
                config->M_cols, config->N_rows);
        return 0;
    }
    
    if (config->M_rows <= 0 || config->M_cols <= 0 || config->N_cols <= 0) {
        fprintf(stderr, "Error: All matrix dimensions must be positive\n");
        return 0;
    }
    
    if (config->num_tasks <= 0 || config->num_gpus <= 0) {
        fprintf(stderr, "Error: num_tasks and num_gpus must be positive\n");
        return 0;
    }
    
    if (config->block_size <= 0 || config->block_size > 32) {
        fprintf(stderr, "Error: block_size must be between 1 and 32\n");
        return 0;
    }
    
    if (config->min_value >= config->max_value) {
        fprintf(stderr, "Error: min_value must be less than max_value\n");
        return 0;
    }
    
    return 1;
}

// Function to write matrix to file
void write_matrix_to_file(const char *filename, float *matrix, int rows, int cols) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error: Cannot create file %s\n", filename);
        return;
    }
    
    // Write dimensions first
    fprintf(file, "%d %d\n", rows, cols);
    
    // Write matrix elements
    int row = 0;
    while (row < rows) {
        int col = 0;
        while (col < cols) {
            fprintf(file, "%.6f", matrix[row * cols + col]);
            if (col < cols - 1) fprintf(file, " ");
            col++;
        }
        fprintf(file, "\n");
        row++;
    }
    
    fclose(file);
    printf("Matrix written to %s (%dx%d)\n", filename, rows, cols);
}

// Function to read matrix from file
int read_matrix_from_file(const char *filename, float **matrix, int *rows, int *cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        return 0;
    }
    
    // Read dimensions
    if (fscanf(file, "%d %d", rows, cols) != 2) {
        fclose(file);
        return 0;
    }
    
    // Allocate memory
    *matrix = (float*)malloc((*rows) * (*cols) * sizeof(float));
    if (!*matrix) {
        fclose(file);
        return 0;
    }
    
    // Read matrix elements
    int row = 0;
    while (row < *rows) {
        int col = 0;
        while (col < *cols) {
            if (fscanf(file, "%f", &((*matrix)[row * (*cols) + col])) != 1) {
                free(*matrix);
                fclose(file);
                return 0;
            }
            col++;
        }
        row++;
    }
    
    fclose(file);
    return 1;
}

// Function to generate or load matrix with configuration
void generate_or_load_matrix(const char *filename, float **matrix, int *rows, int *cols, 
                            int desired_rows, int desired_cols, int gpu_id, 
                            const MatrixConfig *config, unsigned long long seed) {
    
    // Try to read existing matrix
    if (read_matrix_from_file(filename, matrix, rows, cols)) {
        if (*rows == desired_rows && *cols == desired_cols) {
            printf("Matrix loaded: %s (%dx%d)\n", filename, *rows, *cols);
            return;
        } else {
            printf("Matrix %s dimensions mismatch (%dx%d vs %dx%d), regenerating...\n", 
                   filename, *rows, *cols, desired_rows, desired_cols);
            free(*matrix);
        }
    }
    
    // Generate new matrix
    *rows = desired_rows;
    *cols = desired_cols;
    *matrix = (float*)malloc((*rows) * (*cols) * sizeof(float));
    
    // Set GPU device
    CUDA_CHECK(cudaSetDevice(gpu_id));
    
    // Allocate device memory
    float *d_matrix;
    CUDA_CHECK(cudaMalloc(&d_matrix, (*rows) * (*cols) * sizeof(float)));
    
    // Configure kernel launch parameters
    dim3 blockSize(config->block_size, config->block_size);
    dim3 gridSize((*cols + config->block_size - 1) / config->block_size, 
                  (*rows + config->block_size - 1) / config->block_size);
    
    // Generate random matrix on GPU
    generate_random_matrix<<<gridSize, blockSize>>>(
        d_matrix, *rows, *cols, config->min_value, config->max_value, seed);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(*matrix, d_matrix, (*rows) * (*cols) * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Write to file
    write_matrix_to_file(filename, *matrix, *rows, *cols);
    
    // Clean up GPU memory
    CUDA_CHECK(cudaFree(d_matrix));
}

// Function to perform matrix multiplication with configuration
void multiply_matrices_cuda(float *h_M, float *h_N, float *h_P, 
                           const MatrixConfig *config, int gpu_id) {
    
    printf("GPU %d: Multiplying (%dx%d) x (%dx%d) = (%dx%d)\n", 
           gpu_id, config->M_rows, config->M_cols, 
           config->N_rows, config->N_cols, 
           config->P_rows, config->P_cols);
    
    // Set GPU device
    CUDA_CHECK(cudaSetDevice(gpu_id));
    
    // Allocate device memory
    float *d_M, *d_N, *d_P;
    CUDA_CHECK(cudaMalloc(&d_M, config->M_rows * config->M_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_N, config->N_rows * config->N_cols * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_P, config->P_rows * config->P_cols * sizeof(float)));
    
    // Copy matrices to device
    CUDA_CHECK(cudaMemcpy(d_M, h_M, config->M_rows * config->M_cols * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_N, h_N, config->N_rows * config->N_cols * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters
    dim3 blockSize(config->block_size, config->block_size);
    dim3 gridSize((config->N_cols + config->block_size - 1) / config->block_size, 
                  (config->M_rows + config->block_size - 1) / config->block_size);
    
    // Launch matrix multiplication kernel
    matrix_multiply_kernel<<<gridSize, blockSize>>>(
        d_M, d_N, d_P, config->M_rows, config->M_cols, config->N_cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_P, d_P, config->P_rows * config->P_cols * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Clean up device memory
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_N));
    CUDA_CHECK(cudaFree(d_P));
}

// Process function for matrix multiplication task with configuration
void process_matrix_multiplication(int task_id, int gpu_id, const MatrixConfig *config, 
                                  unsigned long long seed) {
    printf("Process %d: GPU %d handling task %d\n", getpid(), gpu_id, task_id);
    
    char filename_M[256], filename_N[256], filename_P[256];
    snprintf(filename_M, sizeof(filename_M), "%s_M_%d.txt", config->file_prefix, task_id);
    snprintf(filename_N, sizeof(filename_N), "%s_N_%d.txt", config->file_prefix, task_id);
    snprintf(filename_P, sizeof(filename_P), "%s_P_%d.txt", config->file_prefix, task_id);
    
    // Generate or load matrices M and N
    float *h_M, *h_N, *h_P;
    int actual_M_rows, actual_M_cols, actual_N_rows, actual_N_cols;
    
    generate_or_load_matrix(filename_M, &h_M, &actual_M_rows, &actual_M_cols, 
                           config->M_rows, config->M_cols, gpu_id, config, seed + task_id);
    
    generate_or_load_matrix(filename_N, &h_N, &actual_N_rows, &actual_N_cols, 
                           config->N_rows, config->N_cols, gpu_id, config, seed + task_id + 1000);
    
    // Allocate result matrix
    h_P = (float*)malloc(config->P_rows * config->P_cols * sizeof(float));
    
    // Perform matrix multiplication
    multiply_matrices_cuda(h_M, h_N, h_P, config, gpu_id);
    
    // Write result matrix to file
    write_matrix_to_file(filename_P, h_P, config->P_rows, config->P_cols);
    
    // Clean up memory
    free(h_M);
    free(h_N);
    free(h_P);
    
    printf("Process %d: Completed task %d\n", getpid(), task_id);
}

// Command line argument parsing
MatrixConfig parse_config_from_args(int argc, char* argv[]) {
    MatrixConfig config = DEFAULT_CONFIG;
    
    if (argc > 1) config.num_tasks = atoi(argv[1]);
    if (argc > 2) config.num_gpus = atoi(argv[2]);
    if (argc > 3) config.M_rows = atoi(argv[3]);
    if (argc > 4) config.M_cols = atoi(argv[4]);
    if (argc > 5) config.N_cols = atoi(argv[5]);
    
    // Update dependent dimensions
    config.N_rows = config.M_cols;
    config.P_rows = config.M_rows;
    config.P_cols = config.N_cols;
    
    return config;
}

int main(int argc, char *argv[]) {
    MatrixConfig config;
    
    // Parse command line arguments for configuration selection
    if (argc > 1 && strcmp(argv[1], "--config") == 0 && argc > 2) {
        config = load_config_from_file(argv[2]);
    } else if (argc > 1 && strcmp(argv[1], "--random") == 0 && argc > 5) {
        int min_dim = atoi(argv[2]);
        int max_dim = atoi(argv[3]);
        int num_tasks = atoi(argv[4]);
        int num_gpus = atoi(argv[5]);
        config = randomize_config_dimensions(min_dim, max_dim, num_tasks, num_gpus);
    } else if (argc > 1 && strcmp(argv[1], "--preset") == 0 && argc > 2) {
        if (strcmp(argv[2], "small") == 0) config = SMALL_CONFIG;
        else if (strcmp(argv[2], "medium") == 0) config = MEDIUM_CONFIG;
        else if (strcmp(argv[2], "large") == 0) config = LARGE_CONFIG;
        else if (strcmp(argv[2], "rectangular") == 0) config = RECTANGULAR_CONFIG;
        else if (strcmp(argv[2], "stress") == 0) config = STRESS_CONFIG;
        else {
            printf("Unknown preset: %s\n", argv[2]);
            printf("Available presets: small, medium, large, rectangular, stress\n");
            config = DEFAULT_CONFIG;
        }
    } else {
        config = parse_config_from_args(argc, argv);
    }
    
    // Validate configuration
    if (!validate_config(&config)) {
        fprintf(stderr, "Invalid configuration. Exiting.\n");
        return 1;
    }
    
    // Print configuration
    print_config(&config);
    
    // Save current configuration
    save_config_to_file(&config, "current_config.txt");
    
    // Check available GPUs
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (device_count < config.num_gpus) {
        printf("Warning: Requested %d GPUs but only %d available. Using %d GPUs.\n", 
               config.num_gpus, device_count, device_count);
        config.num_gpus = device_count;
    }
    
    unsigned long long seed = time(NULL);
    
    // Fork processes for parallel matrix multiplication
    pid_t *child_pids = (pid_t*)malloc(config.num_tasks * sizeof(pid_t));
    if (!child_pids) {
        fprintf(stderr, "Memory allocation failed for child_pids\n");
        return 1;
    }
    
    int task_id = 0;
    while (task_id < config.num_tasks) {
        child_pids[task_id] = fork();
        
        if (child_pids[task_id] == 0) {
            // Child process
            int gpu_id = task_id % config.num_gpus;
            process_matrix_multiplication(task_id, gpu_id, &config, seed);
            exit(0);
        } else if (child_pids[task_id] < 0) {
            fprintf(stderr, "Fork failed for task %d\n", task_id);
            exit(1);
        }
        task_id++;
    }
    
    // Wait for all children to complete
    task_id = 0;
    while (task_id < config.num_tasks) {
        int status;
        waitpid(child_pids[task_id], &status, 0);
        printf("Task %d completed\n", task_id);
        task_id++;
    }
    
    free(child_pids);
    printf("\nAll matrix multiplication tasks completed successfully!\n");
    printf("Generated files: %s_*.txt\n", config.file_prefix);
    
    return 0;
}

/*
Compilation:
nvcc -O3 -Xcompiler -fopenmp matrix_multiply_configurable.cu -o matrix_multiply_configurable

Usage examples:
./matrix_multiply_configurable                                    # Use default config
./matrix_multiply_configurable 8 2 512 1024 256                  # 8 tasks, 2 GPUs, custom dimensions
./matrix_multiply_configurable --preset small                     # Use predefined small config
./matrix_multiply_configurable --preset large                     # Use predefined large config
./matrix_multiply_configurable --config my_config.txt             # Load from config file
./matrix_multiply_configurable --random 64 1024 16 4              # Random dimensions 64-1024, 16 tasks, 4 GPUs
*/