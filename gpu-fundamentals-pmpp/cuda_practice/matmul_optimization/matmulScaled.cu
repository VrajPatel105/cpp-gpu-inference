#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>

using namespace std;

__global__ void matmul(float *A, float *B, float *C, int N){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < N && col < N){
        float sum = 0.0f;
        for(int k = 0; k < N; k++){
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(){
    int N = 1024;
    size_t size = N * N;
    size_t bytes = size * sizeof(float);

    float *h_A = new float[size];
    float *h_B = new float[size];
    float *h_C = new float[size];

    srand(time(NULL));

    for(size_t i = 0; i < size; i++){
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    float *d_A, *d_B, *d_C;

    if(cudaMalloc(&d_A, bytes) != cudaSuccess){
        cout << "Cuda Mem for d_A not allocated" << endl;
        return 0;
    }
    if(cudaMalloc(&d_B, bytes) != cudaSuccess){
        cout << "Cuda Mem for d_B not allocated" << endl;
        return 0;
    }
    if(cudaMalloc(&d_C, bytes) != cudaSuccess){
        cout << "Cuda Mem for d_C not allocated" << endl;
        return 0;
    }

    if(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice) != cudaSuccess){
        cout << "Failed to copy memory from h_A to d_A" << endl;
        return 0;
    }
    if(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice) != cudaSuccess){
        cout << "Failed to copy memory from h_B to d_B" << endl;
        return 0;
    }

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cout << "Kernel time: " << ms << " ms" << endl;
    cout << "C[0][0] = " << h_C[0] << endl;
    cout << "C[N-1][N-1] = " << h_C[N * N - 1] << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

// output : 

// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ nvcc matmul_optimization/matmulScaled.
// cu -o build/naive_matmul_scaled
// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ ./build/naive_matmul_scaled

// Kernel time: 2.49312 ms
// C[0][0] = 1024
// C[N-1][N-1] = 1024