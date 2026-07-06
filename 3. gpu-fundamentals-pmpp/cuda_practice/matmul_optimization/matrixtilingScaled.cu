#include <iostream>
#include <cuda.h>
using namespace std;

#define TILE_WIDTH 2


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

    int N = 4;
    float arrA[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, *h_A = arrA;
    float arrB[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, *h_B = arrB;
    float arrC[16], *h_C = arrC; 
    
    float *d_A, *d_B, *d_C;

    if(cudaMalloc(&d_A, sizeof(float) * 16) != cudaSuccess){
        cout << "Cuda memory not allocated" << endl;
        return 0;
    }
    if(cudaMalloc(&d_B, sizeof(float) * 16) != cudaSuccess){
        cout << "Cuda memory not allocated" << endl;
        return 0;
    }
    if(cudaMalloc(&d_C, sizeof(float) * 16) != cudaSuccess){
        cout << "Cuda memory not allocated" << endl;
        return 0;
    }

    cudaMemcpy(d_A, h_A, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(float) * 16, cudaMemcpyHostToDevice);
       
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(N / TILE_WIDTH, N / TILE_WIDTH);
    optimizedMatmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, sizeof(float) * 16, cudaMemcpyDeviceToHost);

    for(int i = 0; i<16; i++){
        cout << "index i  : " << i << " --> " << h_C[i] << endl;
    }


}