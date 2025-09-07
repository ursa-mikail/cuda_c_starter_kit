# CUDA Matrix Multiply: Global vs Shared Memory

This project compares the performance of matrix multiplication 
using **global memory** vs **shared memory tiling** in CUDA.

📌 A Classic Demo: Matrix Multiplication (Naïve vs Optimized with Shared Memory)

Matrix multiplication is perfect for this because:

Each thread needs to repeatedly access rows and columns of matrices.

If we use only global memory, threads will fetch the same elements redundantly → slow.

If we use shared memory, each block can cache a tile of data → much faster.

If we use constant memory for frequently reused constants (like matrix dimensions), we save global memory bandwidth.

Registers automatically store thread-private scalars.

🔍 

Naïve kernel: Every multiply–add requires fetching from global memory. → Memory bound.

Shared memory kernel: Each tile of data is loaded once into fast on-chip memory and reused by all threads in the block. → Huge speedup.

On a modern GPU, you’ll see 10×–30× faster performance with the shared memory version at N=1024.

```
Global memory is slow unless optimized.

Shared memory accelerates access when data is reused by threads.

Registers/local variables are fastest but limited in use.

Constant memory shines when many threads read the same data.

Registers / local variables → per-thread temporary sum in kernels

Local arrays → per-thread arrays in kernel (not heavily used here)

Shared memory → tiling in optimized matrix multiplication

Global memory → input/output matrices

Constant memory → small lookup table (or matrix dimensions)

- Measure performance differences between global memory and constant memory for small frequently accessed data.

Registers / Local variables: float sum per thread in kernel

Local arrays: could add per-thread arrays if needed (float temp[16];)

Shared memory: tileA and tileB for tiling

Global memory: input matrices A and B

Constant memory: constLUT lookup table & constN matrix size

⚡ Performance Notes

Global kernel: slow due to repeated global memory reads.

Shared + constant kernel: faster due to:
	- Shared memory tiling (reuses loaded data across threads in a block)
	- Constant memory (very fast when all threads read the same small table)
```

## Files
- `matmul_compare.cu`: CUDA source code
- `Makefile`: Build automation
- `scripts/run_local.sh`: Example local run script

## Run
```bash
ssh_run_cuda "matrix_mutiply_comparing_global_vs_shared_memory_usage" 4 1024 "cuda_results.log"
```

```
Global memory kernel: 18.086559 ms
Shared + constant memory kernel: 6.750976 ms
Checksum: 17301326.230399
```

