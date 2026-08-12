// This code covers implementation of thread coarsening 

// NOTE : I also have a handtrace for thread coarsening covered. please read the readme. 

#include <iostream>
#include <cuda.h>
using namespace std;

#define TILE_WIDTH 2
#define COARSE_FACTOR 2 // the coarse factor

__global__ void coarsenedMatrixTiling(float* A, float* B, float* P, int N){

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    // base column: where this block's *first* strip starts.
    // colStart var -> keeps track of what the block's first strip was
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    // defining the shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    
    // a Pavalue array that will store per thread output. This means, this is simply a register for that thread.
    float Pvalue[COARSE_FACTOR];
    for (int c = 0; c < COARSE_FACTOR; ++c) Pvalue[c] = 0.0f; // filling the entire value with 0 floating point

    for (int ph = 0; ph < N / TILE_WIDTH; ++ph) { // main phase loop

        // A's tile does NOT depend on which output column we're on, (this is purely design based. we can also go for row to keep them fixed isntead)
        // in general, row is what is being reused over and over again and col is something that we need to fetch it fresh.
        // so it loads exactly once per phase, same as the non-coarsened kernel.
        Mds[ty][tx] = A[row*N + (ph*TILE_WIDTH + tx)];

        // B's tile DOES depend on the column, so we loop over the columns
        // this thread owns, reloading Nds fresh for each one.

        // Since Nds is changing values every iteration (because we have to change cols),
        // we have to get fresh values — and to get those fresh values we run this loop,
        // which simply overwrites the same Nds buffer every time.
        // (Reuse over time, not growth: Nds never gets bigger, its contents just get replaced per strip.)

        for (int c = 0; c < COARSE_FACTOR; ++c) {

            int col = colStart + c * TILE_WIDTH;

            // corner-turned load,
            // just using this iteration's col instead of a fixed one.
            Nds[tx][ty] = B[col*N + (ph*TILE_WIDTH + tx)];

            __syncthreads(); // Mds and this c's Nds must be fully loaded before anyone computes

            for (int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }

            __syncthreads(); // everyone must finish reading Nds before the next c overwrites it
        }
    }

    // write out all COARSE_FACTOR results this thread computed
    // moving the finished results out of the thread's private registers into the actual output matrix in global memory
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c * TILE_WIDTH;
        P[row*N + col] = Pvalue[c];
    }
}

int main(){

    int N = 4;
    float arrA[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, *h_A = arrA;
    float arrB[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16}, *h_B = arrB;
    float resultArr[16], *h_result = resultArr;

    float *d_A, *d_B, *d_result;

    cudaMalloc(&d_A, sizeof(float) * 16);
    cudaMalloc(&d_B, sizeof(float) * 16);
    cudaMalloc(&d_result, sizeof(float) * 16);

    cudaMemcpy(d_A, h_A, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * 16, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    // gridDim.x shrinks by COARSE_FACTOR: each block now covers COARSE_FACTOR
    // times as many output columns as before.
    dim3 gridDim(N / (TILE_WIDTH * COARSE_FACTOR), N / TILE_WIDTH);
    coarsenedMatrixTiling<<<gridDim, blockDim>>>(d_A, d_B, d_result, N);

    cudaMemcpy(h_result, d_result, sizeof(float) * 16, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 16; ++i){
        cout << "i = " << i << " -> " << h_result[i] << endl;
    }

    return 0;
}