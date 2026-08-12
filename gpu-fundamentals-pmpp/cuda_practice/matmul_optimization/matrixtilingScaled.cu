#include <iostream>
#include <cuda.h>
using namespace std;

#define TILE_WIDTH 16


__global__ void optimizedMatmul(float* A, float* B, float *P, int N){

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // blocks scratchpad
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];


    float Pvalue = 0.0f; // the running total

    for(int ph = 0; ph < N / TILE_WIDTH; ++ph){

        Mds[ty][tx] = A[row*N + (ph*TILE_WIDTH + tx)];
        Nds[ty][tx] = B[(ph*TILE_WIDTH + ty)*N + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }
    P[row*N + col] = Pvalue;

}


int main(){
    int N = 1024;
    size_t size = N * N;
    size_t bytes = size * sizeof(float);

    float *h_A = new float[size];
    float *h_B = new float[size];
    float *h_C = new float[size];

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

    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks(N/TILE_WIDTH, N/TILE_WIDTH);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    optimizedMatmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
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

// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ nvcc matmul_optimization/matrixtilingScaled.cu -o build/tiled_scaled_matrix
// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ ./build/tiled_scaled_matrix

// Kernel time: 2.10192 ms
// C[0][0] = 1024
// C[N-1][N-1] = 1024