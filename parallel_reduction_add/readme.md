# Parallel Reduction (Efficient Tool Example)

This example computes the sum of an array using GPU threads. It introduces shared memory and reducing work efficiently. This is a building block for many ML and scientific workloads.


## What This Demo Shows
```
Uses shared memory to reduce global memory reads

Demonstrates synchronization (__syncthreads())

Introduces parallel reduction, a pattern reused in ML, graphics, and simulations
```

```
🔹 Naive vs Parallel
Naive CPU reduction:
int sum = 0;
for (int i = 0; i < N; i++) {
    sum += data[i];
}

Parallel GPU reduction idea:

Break the array into chunks.

Each thread block computes a partial sum.

Store each block’s result.

Reduce again until only 1 result remains.

🔹 How Parallel Reduction Works (Step by Step)

Let’s say we want to sum 8 numbers:

Index: 0   1   2   3   4   5   6   7
Data : 2   4   6   8   1   3   5   7

Step 1 – Load into shared memory

Each thread loads one element into shared memory (fast memory shared by threads in the block).

sdata = [2, 4, 6, 8, 1, 3, 5, 7]

Step 2 – Pairwise Reduction

Now threads cooperate:

First round: each thread adds element i and i+stride where stride = 4.

[2+1, 4+3, 6+5, 8+7] → [3, 7, 11, 15]

Step 3 – Halve stride

Next round, stride = 2.

[3+11, 7+15] → [14, 22]

Step 4 – Final reduction

Stride = 1.

[14+22] = [36]


✅ Done: the sum is 36.

Each round halves the active threads → O(log N) steps instead of O(N).

[2, 4, 6, 8, 1, 3, 5, 7]
   ↘   ↘   ↘   ↘
   [6, 14, 4, 12]
       ↘     ↘
       [20, 16]
           ↘
           [36]


```

## Run
```bash
ssh_run_cuda_multi "parallel_reduction_add" 4 1024 "cuda_results.log"
```



