# Naive Matmul — W6 Analysis

## Kernel

```cuda
__global__ void matmul(float *A, float *B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < N && col < N){
        float sum = 0.0f;
        for(int k = 0; k < N; k++){
            sum += A[row*N + k] * B[k*N + col];
        }
        C[row*N + col] = sum;
    }
}
```

## Benchmark (1024×1024, RTX 4070 Laptop)

| Implementation | Time |
|---------------|------|
| Naive CUDA kernel | 2.758 ms |
| torch.matmul (cuBLAS) | 0.227 ms |
| **Gap** | **12x** |

## Analysis

The kernel I wrote uses a very naive way of matrix multiplication. The biggest issue is that in order to find the sum:

```cuda
sum += A[row*N + k] * B[k*N + col];
```

it has to access the elements in array A and array B every time for each individual thread. Every single thread is reading from global memory on every loop iteration. This is the exact issue: each thread does N reads from global memory, and neighboring threads redundantly load the same data. This is a memory bound problem.

On a 1024×1024 matrix, the naive kernel takes 2.758ms vs torch.matmul's 0.227ms — a 12x gap entirely explained by redundant global memory traffic. With tile size 16, shared memory tiling reduces global memory reads by 16x.

The exact fix is shared memory tiling. Load a tile of A and a tile of B into shared memory once, then all threads in the block reuse that tile. This will reduce global memory reads significantly. This is what W8 implements.
