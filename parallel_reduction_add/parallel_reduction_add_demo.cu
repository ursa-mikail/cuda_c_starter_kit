#include <stdio.h>
#include <cuda.h>

// --------------------------------------
// Shared-memory reduction
// --------------------------------------
__global__ void reduceSum(int *g_idata, int *g_odata, int n) {
    extern __shared__ int sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int sum = 0;
    if (i < n) sum = g_idata[i];
    if (i + blockDim.x < n) sum += g_idata[i + blockDim.x];
    sdata[tid] = sum;
    __syncthreads();

    // reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// --------------------------------------
// Warp-shuffle optimized reduction
// --------------------------------------
__inline__ __device__ int warpReduceSum(int val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduceSumOptimized(int *g_idata, int *g_odata, int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int sum = 0;
    if (i < n) sum = g_idata[i];
    if (i + blockDim.x < n) sum += g_idata[i + blockDim.x];

    // warp-level reduction
    sum = warpReduceSum(sum);

    // first thread of each warp writes to shared mem
    __shared__ int shared[32];  // supports up to 1024 threads (32 warps)
    if (tid % warpSize == 0) shared[tid / warpSize] = sum;
    __syncthreads();

    // first warp reduces shared results
    sum = (tid < blockDim.x / warpSize) ? shared[tid] : 0;
    if (tid < warpSize) {
        sum = warpReduceSum(sum);
    }

    if (tid == 0) g_odata[blockIdx.x] = sum;
}

// --------------------------------------
// Host code: compare both versions
// --------------------------------------
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

    int *h_out = (int*)malloc(blocks * sizeof(int));

    // ------------------------------
    // Shared-memory timing
    // ------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    reduceSum<<<blocks, threads, threads * sizeof(int)>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms1;
    cudaEventElapsedTime(&ms1, start, stop);

    cudaMemcpy(h_out, d_out, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    int total1 = 0; for (int i = 0; i < blocks; i++) total1 += h_out[i];
    printf("Shared-Memory: Sum = %d, Time = %.4f ms\n", total1, ms1);

    // ------------------------------
    // Warp-shuffle timing
    // ------------------------------
    cudaEventRecord(start);
    reduceSumOptimized<<<blocks, threads>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms2;
    cudaEventElapsedTime(&ms2, start, stop);

    cudaMemcpy(h_out, d_out, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    int total2 = 0; for (int i = 0; i < blocks; i++) total2 += h_out[i];
    printf("Warp-Shuffle  : Sum = %d, Time = %.4f ms\n", total2, ms2);

    // cleanup
    cudaFree(d_in); cudaFree(d_out);
    free(h_in); free(h_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
