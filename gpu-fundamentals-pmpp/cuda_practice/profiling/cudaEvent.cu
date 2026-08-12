


#include <cuda.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

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

int main(){

    // 1. Create two events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int N = 1024;
    int size = N * N;

    float *h_A = new float[size];
    float *h_B = new float[size];
    float *h_C = new float[size];

    for(int i = 0; i < size; i++){
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;

    if(cudaMalloc(&d_A, sizeof(float) * size) != cudaSuccess){
        cout << "Cuda Mem for d_A not allocated" << endl;
        return 0;
    }
    if(cudaMalloc(&d_B, sizeof(float) * size) != cudaSuccess){
        cout << "Cuda Mem for d_B not allocated" << endl;
        return 0;
    }
    if(cudaMalloc(&d_C, sizeof(float) * size) != cudaSuccess){
        cout << "Cuda Mem for d_C not allocated" << endl;
        return 0;
    }

    if(cudaMemcpy(d_A, h_A, sizeof(float) * size, cudaMemcpyHostToDevice) != cudaSuccess){
        cout << "Failed to copy memory from d_A to h_A" << endl;
        return 0;
    }
    if(cudaMemcpy(d_B, h_B, sizeof(float) * size, cudaMemcpyHostToDevice) != cudaSuccess){
        cout << "Failed to copy memory from d_B to h_B" << endl;
        return 0;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N+15)/16, (N+15)/16);

    // 2. Record start
    cudaEventRecord(start);

    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 4. Record stop
    cudaEventRecord(stop);

    
    // 5. Wait for stop to finish, then get elapsed time in ms
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    printf("Kernel time: %.3f ms\n", ms);

    // 6. Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // copy back from d_C to h_C
    cudaMemcpy(h_C, d_C, sizeof(float) * size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}