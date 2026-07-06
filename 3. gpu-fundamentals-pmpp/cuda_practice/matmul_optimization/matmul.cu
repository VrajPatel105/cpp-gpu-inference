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

    int N = 4;
    float arrA[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, *h_A = arrA;
    float arrB[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, *h_B = arrB;
    float arrC[16], *h_C = arrC; 

    float *d_A, *d_B, *d_C;

    if(cudaMalloc(&d_A, sizeof(float) * 16) != cudaSuccess){
        cout << "Cuda Mem for d_A not allocated" << endl;
        return 0;
    }
    if(cudaMalloc(&d_B, sizeof(float) * 16) != cudaSuccess){
        cout << "Cuda Mem for d_B not allocated" << endl;
        return 0;
    }
    if(cudaMalloc(&d_C, sizeof(float) * 16) != cudaSuccess){
        cout << "Cuda Mem for d_C not allocated" << endl;
        return 0;
    }

    if(cudaMemcpy(d_A, h_A, sizeof(float) * 16, cudaMemcpyHostToDevice) != cudaSuccess){
        cout << "Failed to copy memory from d_A to h_A" << endl;
        return 0;
    }
    if(cudaMemcpy(d_B, h_B, sizeof(float) * 16, cudaMemcpyHostToDevice) != cudaSuccess){
        cout << "Failed to copy memory from d_B to h_B" << endl;
        return 0;
    }

    dim3 threadsPerBlock(16,16);
    dim3 numBlocks((N+15)/16, (N+15)/16);
    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    cudaMemcpy(h_C, d_C, sizeof(float) * 16, cudaMemcpyDeviceToHost);

    // print
    cout << "Pasting the output " << endl;

    for(int i = 0; i<16; i++){
        cout << h_C[i] << endl;
    }

    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            cout << h_C[i*N + j] << "\t";
        }
        cout << endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
