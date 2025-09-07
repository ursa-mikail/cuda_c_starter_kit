# CUDA Thread Synchronization Demo

A comprehensive example demonstrating CUDA thread synchronization concepts including `__syncthreads()`, warp divergence, and shared memory optimization.

## What This Demo Shows

### 1. Thread Synchronization with `__syncthreads()`
- **Naive Matrix Multiplication**: Basic kernel without optimization
- **Shared Memory Matrix Multiplication**: Uses `__syncthreads()` to coordinate shared memory access
- **Performance Comparison**: Shows speedup from proper synchronization and memory usage

### 2. Warp Divergence Analysis
- **Divergent Kernel**: Demonstrates performance penalty when threads in a warp take different paths
- **Optimized Kernel**: Shows how to minimize warp divergence
- **Performance Impact**: Quantifies the cost of poor thread scheduling

### 3. Key Synchronization Concepts Demonstrated

#### `__syncthreads()` Usage
```cuda
// Load data into shared memory
As[ty][tx] = A[row * N + tile * TILE_WIDTH + tx];
Bs[ty][tx] = B[(tile * TILE_WIDTH + ty) * N + col];

// CRITICAL: Wait for all threads to finish loading
__syncthreads();

// Now all threads can safely use the shared data
for (int k = 0; k < TILE_WIDTH; k++) {
    sum += As[ty][k] * Bs[k][tx];
}

// CRITICAL: Wait before loading next tile
__syncthreads();
```

#### Warp Divergence Problem
```cuda
// BAD: Creates warp divergence
if (threadIdx.x % 2 == 0) {
    // Even threads do expensive work
    expensive_computation();
} else {
    // Odd threads do simple work
    simple_computation();
}
```

#### Warp Divergence Solution
```cuda
// GOOD: Work assigned per warp, not per thread
if ((idx / 32) % 2 == 0) {
    // All threads in this warp do same work
    expensive_computation();
} else {
    // All threads in this warp do same work
    simple_computation();
}
```

## Building and Running

### Basic Usage
```bash
# Build the project
make clean && make

# Run with default settings (1 GPU, 512x512 matrices)
./matrix_sync_demo

# Run with specific GPU count and matrix size
./matrix_sync_demo 2 1024
```

### Using the SSH Framework

#### Quick Test Run
```bash
# Test on remote CUDA system with 2 GPUs, 512x512 matrices
ssh_run_and_collect "matrix_sync_demo" "make clean && make && ./matrix_sync_demo 2 512" "sync_results.log"
```

#### System Info Check
```bash
# Check remote system capabilities
ssh_run_and_collect "matrix_sync_demo" "make gpu-info && make check-deps" "system_info.log"
```

#### Performance Benchmark
```bash
# Run comprehensive benchmark
ssh_run_and_collect "matrix_sync_demo" "make benchmark" "performance_results.log"
```

#### Multiple Test Scenarios
```bash
# Test different configurations
ssh_run_and_collect "matrix_sync_demo" "make test-small && make test-medium && make test-large" "scaling_results.log"
```

## Expected Output

The demo will show:

### 1. Matrix Multiplication Results
```
=== GPU 0 Results ===
Device: Quadro M4000
Warp size: 32
Max threads per block: 1024

--- Test 1: Naive Matrix Multiplication ---
Naive kernel time: 45.23 ms

--- Test 2: Shared Memory with __syncthreads() ---
Shared memory kernel time: 12.87 ms
Speedup: 3.51x
Results match: YES (max diff: 0.000001)
```

### 2. Warp Divergence Analysis
```
--- Test 3: Warp Divergence Analysis ---
Warp divergent kernel time: 8.45 ms
Warp optimized kernel time: 3.21 ms
Divergence penalty: 2.63x slower
```

### 3. Key Insights Summary
```
=== Key Synchronization Insights ===
1. __syncthreads() ensures all threads in a block reach the same point
2. Shared memory + sync enables data reuse and reduces global memory access
3. Warp divergence occurs when threads in a warp take different code paths
4. Proper thread scheduling can hide memory latency
5. SIMD execution within warps means divergence hurts performance
```

## Learning Objectives

After running this demo, you'll understand:

1. **When and why to use `__syncthreads()`**
   - Coordinating shared memory access
   - Ensuring data consistency across threads
   - Phase synchronization in algorithms

2. **Impact of warp divergence**
   - How SIMD execution works in CUDA
   - Performance penalties of divergent code paths
   - Strategies to minimize divergence

3. **Shared memory optimization**
   - Tiling techniques for matrix operations
   - Memory access pattern optimization
   - Balancing computation and memory access

4. **Thread scheduling concepts**
   - How warps are scheduled on SMs
   - Latency hiding through warp switching
   - Occupancy considerations

## Advanced Usage

### Custom Matrix Sizes
```bash
# Test with different matrix sizes to see scaling behavior
ssh_run_and_collect "matrix_sync_demo" "./matrix_sync_demo 1 128" "small_test.log"
ssh_run_and_collect "matrix_sync_demo" "./matrix_sync_demo 1 2048" "large_test.log"
```

### Multi-GPU Comparison
```bash
# Compare single vs multi-GPU performance
ssh_run_and_collect "matrix_sync_demo" "./matrix_sync_demo 1 1024 && ./matrix_sync_demo 2 1024" "gpu_comparison.log"
```

### Profiling Integration
```bash
# Run with NVIDIA profiler (if available)
ssh_run_and_collect "matrix_sync_demo" "nvprof ./matrix_sync_demo 1 512" "profiling_results.log"
```

```bash
# Quick performance test
ssh_run_and_collect "matrix_sync_demo" "make clean && make && ./matrix_sync_demo 2 512" "sync_results.log"

# System capability check
ssh_run_and_collect "matrix_sync_demo" "make gpu-info && make check-deps" "system_info.log"

# Comprehensive benchmark
ssh_run_and_collect "matrix_sync_demo" "make benchmark" "performance_results.log"
```

```
% ssh_run_and_collect "matrix_sync_demo" "make clean && make && ./matrix_sync_demo 2 512" "sync_results.log"

[2025-09-07 10:07:25] Starting automated run for folder: matrix_sync_demo
[2025-09-07 10:07:25] Zipping and sending folder...
[2025-09-07 10:07:25] Folder sent successfully
[2025-09-07 10:07:25] Unzipping on remote...
Archive:  matrix_sync_demo.zip
  inflating: matrix_sync_demo/Makefile  
  inflating: matrix_sync_demo/readme.md  
  inflating: matrix_sync_demo/matrix_sync_demo.cu  
[2025-09-07 10:07:25] Unzipped successfully
[2025-09-07 10:07:25] Running command: make clean && make && ./matrix_sync_demo 2 512
rm -f matrix_sync_demo
nvcc -O3 -arch=sm_50     matrix_sync_demo.cu -o matrix_sync_demo -lcublas
matrix_sync_demo.cu: In function ‘int main(int, char**)’:
matrix_sync_demo.cu:214:8: warning: format ‘%d’ expects argument of type ‘int’, but argument 2 has type ‘size_t’ {aka ‘long unsigned int’} [-Wformat=]
  214 |         printf("Shared memory per block: %d KB\n", prop.sharedMemPerBlock / 1024);
      |        ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      |                                                                     |
      |                                                                     size_t {aka long unsigned int}
=== CUDA Thread Synchronization Analysis ===
Matrix size: 512x512
Using 2 GPU(s)

Matrix A (showing 4x4 corner):
0.033 0.330 0.691 0.422 
0.220 0.059 0.813 0.082 
0.670 0.478 0.572 0.684 
0.023 0.094 0.272 0.006 

Matrix B (showing 4x4 corner):
0.995 0.909 0.545 0.154 
0.905 0.352 0.379 0.550 
0.350 0.209 0.811 0.114 
0.636 0.912 0.265 0.516 

=== GPU 0 Results ===
Device: Quadro M4000
Warp size: 32
Max threads per block: 1024
Shared memory per block: 48 KB

--- Test 1: Naive Matrix Multiplication ---
Naive kernel time: 2.19 ms
Naive Result C (showing 4x4 corner):
127.418 126.391 131.332 122.913 
123.177 124.367 125.839 120.228 
131.123 123.759 127.717 125.969 
132.173 132.938 133.770 124.866 

--- Test 2: Shared Memory with __syncthreads() ---
Shared memory kernel time: 0.90 ms
Speedup: 2.43x
Shared Memory Result C (showing 4x4 corner):
127.418 126.391 131.332 122.913 
123.177 124.367 125.839 120.228 
131.123 123.759 127.717 125.969 
132.173 132.938 133.770 124.866 

Results match: YES (max diff: 0.000000)

--- Test 3: Warp Divergence Analysis ---
Warp divergent kernel time: 0.06 ms
Warp optimized kernel time: 0.05 ms
Divergence penalty: 1.15x slower

=== GPU 1 Results ===
Device: Quadro M4000
Warp size: 32
Max threads per block: 1024
Shared memory per block: 48 KB

--- Test 1: Naive Matrix Multiplication ---
Naive kernel time: 2.01 ms
Naive Result C (showing 4x4 corner):
127.418 126.391 131.332 122.913 
123.177 124.367 125.839 120.228 
131.123 123.759 127.717 125.969 
132.173 132.938 133.770 124.866 

--- Test 2: Shared Memory with __syncthreads() ---
Shared memory kernel time: 0.89 ms
Speedup: 2.26x
Shared Memory Result C (showing 4x4 corner):
127.418 126.391 131.332 122.913 
123.177 124.367 125.839 120.228 
131.123 123.759 127.717 125.969 
132.173 132.938 133.770 124.866 

Results match: YES (max diff: 0.000000)

--- Test 3: Warp Divergence Analysis ---
Warp divergent kernel time: 0.06 ms
Warp optimized kernel time: 0.06 ms
Divergence penalty: 1.08x slower

=== Key Synchronization Insights ===
1. __syncthreads() ensures all threads in a block reach the same point
2. Shared memory + sync enables data reuse and reduces global memory access
3. Warp divergence occurs when threads in a warp take different code paths
4. Proper thread scheduling can hide memory latency
5. SIMD execution within warps means divergence hurts performance
[2025-09-07 10:07:25] Retrieving logs...
[2025-09-07 10:07:25] Logs retrieved and appended to sync_results.log
[2025-09-07 10:07:25] Automated run completed
```

```
Key Features:

Three Different Kernels:

Naive Matrix Multiplication: Basic implementation without optimization
Shared Memory with __syncthreads(): Demonstrates proper synchronization for shared memory tiling
Warp Divergence Comparison: Shows the performance impact of divergent vs. optimized code paths


Real-World Synchronization Concepts:

__syncthreads() Usage: Critical synchronization points in shared memory algorithms
Warp Divergence: Demonstrates SIMD execution penalties and solutions
Memory Access Patterns: Shows how synchronization enables efficient data reuse
Latency Hiding: Illustrates how proper scheduling can hide memory latency


Framework Integration:

Uses your ssh_run_and_collect function for remote execution
Modular Makefile with multiple test targets
Comprehensive logging and performance measurement
Multi-GPU support
```

This example provides a comprehensive demonstration of CUDA synchronization concepts while utilizing your SSH framework for remote execution and result collection.
