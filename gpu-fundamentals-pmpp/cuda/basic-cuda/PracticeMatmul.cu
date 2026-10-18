#include <cuda.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

__global__ void practiceMatmulKernel(float *A, float *B, float *C, int N){

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(row < N && col < N){
        float sum = 0.0f;
        for(int k = 0; k<N; k++){
            sum += A[row * N + k] * B[k * N + col];
        }
    }

}


int main(){

    int N = 3;

    float arrA[9] = {1,2,3,4,5,6,7,8,9}, *h_A = arrA;
    float arrB[9] = {1,2,3,4,5,6,7,8,9}, *h_B = arrB;
    float arrC[9], *h_C = arrC; 

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, sizeof(float) * 9);
    cudaMalloc(&d_B, sizeof(float) * 9);
    cudaMalloc(&d_C, sizeof(float) * 9);

    cudaMemcpy(d_A, h_A, sizeof(float) * 9, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * 9, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(9,9);
    dim3 numBlocks((N+9)/9, (N+9)/9);
    practiceMatmulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, sizeof(float) * 9, cudaMemcpyDeviceToHost);

    for(int i = 0; i<9; i++){
        cout << h_C[i] << endl;
    }

}