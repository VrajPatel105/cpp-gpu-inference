// this file is for understanding corner turning (PMPP ch-6 : Performance Considerations)
#include <iostream>
#include <cuda.h>
using namespace std;

#define TILE_WIDTH 2 

__global__ void CornerTurning(float* A, float* B, float* P, int N){ 

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    float Pvalue = 0.0f;

    for(int ph = 0; ph < N / TILE_WIDTH; ++ph){

        Mds[ty][tx] = A[row*N + (ph*TILE_WIDTH + tx)];
        Nds[tx][ty] = B[(bx*TILE_WIDTH + ty)*N + (ph*TILE_WIDTH + tx)]; 
        __syncthreads(); 

        for(int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads(); 
    }
    P[row*N + col] = Pvalue;
}



int main(){

    int N = 4; 
    float arrA[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, *h_A = arrA;
    float arrB[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16}, *h_B = arrB;
    float resultArr[16], *h_result = resultArr;

    float *d_A, *d_B, *d_result;

    // allocate the memory
    cudaMalloc(&d_A, sizeof(float) * 16);
    cudaMalloc(&d_B, sizeof(float) * 16);
    cudaMalloc(&d_result, sizeof(float) * 16);

    // transfer the memory from host to device
    cudaMemcpy(d_A, h_A, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * 16, cudaMemcpyHostToDevice);

    // define the num blocks and threads
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(N / TILE_WIDTH, N / TILE_WIDTH);
    CornerTurning<<<gridDim, blockDim>>>(d_A, d_B, d_result, N);

    // transferring the resuling mem from d_result to h_result (device to host)
    cudaMemcpy(h_result, d_result, sizeof(float) * 16, cudaMemcpyDeviceToHost);

    // pritint the final output
    for(int i = 0; i < 16; ++i){
        cout << "i = " << i << " -> " << h_result[i] << endl;
    }

    return 0;

}