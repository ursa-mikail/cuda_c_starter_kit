# Parallel CUDA Matrix Multiplication with File I/O (matrix_multiply_with_file_io)

This project performs parallel matrix multiplication using CUDA across multiple GPUs. It generates random matrices, stores them in files, reads them back, and performs matrix multiplication with results written to output files.

## Features

- **Multi-GPU Parallel Processing**: Distributes matrix multiplication tasks across multiple GPUs
- **Multi-Process Architecture**: Uses fork() for true parallel processing
- **File-Based Matrix Storage**: Persistent matrix storage and retrieval
- **Configurable Matrix Sizes**: Support for various matrix dimensions
- **CUDA-Optimized Kernels**: Efficient GPU-based random generation and matrix multiplication
- **Row-Major Memory Layout**: Standard C-style matrix addressing

## Matrix Addressing

This implementation uses **row-major layout** where element (x,y) is addressed as:
```
index = x * width + y
```

### Thread Mapping
Each CUDA thread is mapped to a matrix element using:
```cpp
row = blockIdx.y * blockDim.y + threadIdx.y;
col = blockIdx.x * blockDim.x + threadIdx.x;
```

### Boundary Checking
Extra threads are prevented from doing work using:
```cpp
if(row < height && col < width) {
    // do work
}
```

## Matrix Multiplication Algorithm

For matrices M(I×J) × N(J×K) = P(I×K), each element is calculated as:
```
P[x,y] = Σ(k=0 to J-1) M[x,k] * N[k,y]
```

The CUDA kernel implements:
```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;

if(row < M_rows && col < N_cols) {
    float product_val = 0.0f;
    int k = 0;
    while (k < M_cols) {
        product_val += d_M[row * M_cols + k] * d_N[k * N_cols + col];
        k++;
    }
    d_P[row * N_cols + col] = product_val;
}
```

## Prerequisites

- CUDA-capable GPU(s)
- CUDA Toolkit (nvcc compiler)
- Linux/Unix system with fork() support
- Optional: Python with NumPy (for validation)

### Installation

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install nvidia-cuda-toolkit build-essential

# CentOS/RHEL
sudo yum install cuda-toolkit gcc-c++

# Check dependencies
make check-deps
```

## Building

```bash
# Build the project
make

# Check dependencies first
make check-deps

# Clean and rebuild
make clean && make
```

## Usage

### Basic Usage

```bash
# Default: 4 tasks, 1 GPU, 256×256 matrices
./matrix_multiply_parallel

# Custom configuration: 8 tasks, 2 GPUs, 512×512 matrices
./matrix_multiply_parallel 8 2 512 512 512

# Large scale: 16 tasks, 4 GPUs, 1024×1024 matrices
./matrix_multiply_parallel 16 4 1024 1024 1024
```

### Parameters

1. **num_tasks**: Number of parallel matrix multiplication tasks (default: 4)
2. **num_gpus**: Number of GPUs to utilize (default: 1)
3. **M_rows**: Matrix M row count (default: 256)
4. **M_cols**: Matrix M column count = Matrix N row count (default: 256)
5. **N_cols**: Matrix N column count (default: 256)

### File Format

Generated files follow the pattern:
- `matrix_M_0.txt`, `matrix_M_1.txt`, ... (Input matrix M)
- `matrix_N_0.txt`, `matrix_N_1.txt`, ... (Input matrix N)
- `matrix_P_0.txt`, `matrix_P_1.txt`, ... (Result matrix P = M × N)

File format:
```
rows cols
element_0_0 element_0_1 ... element_0_cols-1
element_1_0 element_1_1 ... element_1_cols-1
...
element_rows-1_0 ... element_rows-1_cols-1
```

## Testing

```bash
# Quick test with small matrices
make test

# Medium scale test
make test-medium

# Large scale test
make test-large

# Performance benchmark
make benchmark

# Create sample matrices
make create-samples

# View results
make show-results

# Validate correctness (requires Python/NumPy)
make validate-result
```

## Remote Execution with SSH

### Basic SSH Commands

```bash
# Quick test run
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && make test" \
    "cuda_matrix_test.log"

# Medium scale run
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && ./matrix_multiply_parallel 8 2 512 512 512" \
    "cuda_matrix_medium.log"

# Large scale benchmark
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && make benchmark" \
    "cuda_matrix_benchmark.log"

# Performance analysis
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && make perf-analysis" \
    "cuda_matrix_performance.log"
```

### Specific Configuration Examples

```bash
# 16 tasks across 4 GPUs with 1024×1024 matrices
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && ./matrix_multiply_configurable 16 4 1024 1024 1024" \
    "cuda_matrix_large_scale.log"

# Memory-intensive test with different matrix shapes
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && ./matrix_multiply_configurable 8 2 512 1024 256" \
    "cuda_matrix_rectangular.log"

# Multi-step analysis
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && make gpu-info && make test-medium && make show-results" \
    "cuda_matrix_analysis.log"
```

```bash
# Large scale test - CORRECTED command
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && ./matrix_multiply_configurable 16 4 1024 1024 1024" \
    "cuda_matrix_large_scale.log"

# Use preset configurations (more reliable)
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && ./matrix_multiply_configurable --preset large" \
    "cuda_matrix_large.log"

# Stress testing configuration
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && ./matrix_multiply_configurable --preset stress" \
    "cuda_matrix_stress.log"

# Random configuration testing
ssh_run_and_collect "matrix_multiply_cuda" \
    "make clean && make && ./matrix_multiply_configurable --random 512 2048 32 8" \
    "cuda_matrix_random.log"
```

## Performance Characteristics

### Scaling Patterns

**GPU Scaling:**
- 1 GPU: Sequential task processing
- 2 GPUs: ~1.8x speedup (with sufficient tasks)
- 4 GPUs: ~3.2x speedup (with sufficient parallelizable work)

**Matrix Size Impact:**
- 64×64: Memory-bound, limited GPU utilization
- 256×256: Good balance of computation and memory
- 512×512: Compute-bound, excellent GPU utilization
- 1024×1024: Memory-intensive, may hit GPU memory limits

**Task Parallelism:**
- Tasks < GPUs: Underutilized hardware
- Tasks ≈ GPUs: Good load balancing
- Tasks >> GPUs: Excellent task distribution

### Typical Performance

```
Matrix Size
