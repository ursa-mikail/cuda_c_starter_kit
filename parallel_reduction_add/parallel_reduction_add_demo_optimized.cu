#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduceSum(int *input, int *output, int N) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load input into shared memory
    if (i < N) {
        sdata[tid] = input[i] + ((i + blockDim.x < N) ? input[i + blockDim.x] : 0);
    } else {
        sdata[tid] = 0;
    }
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block result
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(int);
    int *h_in = (int*)malloc(size);

    for (int i = 0; i < N; i++) h_in[i] = 1;  // sum should be N

    int *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads * 2 - 1) / (threads * 2);
    cudaMalloc(&d_out, blocks * sizeof(int));

    reduceSum<<<blocks, threads, threads * sizeof(int)>>>(d_in, d_out, N);

    // Copy results back and finalize on CPU
    int *h_out = (int*)malloc(blocks * sizeof(int));
    cudaMemcpy(h_out, d_out, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    int total = 0;
    for (int i = 0; i < blocks; i++) total += h_out[i];
    printf("Sum = %d\n", total);

    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    return 0;
}

/*
🔹 Why is This Optimized?

No shared memory inside warps (shuffle directly moves registers).

Fewer __syncthreads() (only once when warps deposit results).

Better memory efficiency → avoids bank conflicts in shared memory.

🔹 Performance Takeaway

Shared-memory reduction: good intro, portable.

Warp-shuffle reduction: faster, fewer syncs, used in real libraries (cuBLAS, Thrust).

You can extend this with loop unrolling and loading multiple elements per thread for even more throughput.
*/