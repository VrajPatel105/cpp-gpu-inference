// What do I produce? → I own one P element. (this is the seed — nothing else makes sense without it)
// Where am I in the grid? → bx, by, tx, ty
// So which P element is mine? → Row, Col
// Where's my running total, and what scratchpad does my block share? → Pvalue, Mds, Nds
// Which tile am I on right now? → the ph loop
// Inside this phase: which one M and one N cell do I load — then wait? → lines 19, 20, __syncthreads
// Tile's full — accumulate my slice of the dot product → the k loop
// Wait so nobody clobbers the tile → __syncthreads
// All phases done — write my answer home → P[...] = Pvalue


// Above are the questions that claude has provided me and I will answer them on my own and then try to convert it into cuda code. 


#include <iostream>
#include <cuda.h>
using namespace std;

#define TILE_WIDTH 2


// writing the main kernel

__global__ void optimizedMatmul(float* A, float* B, float *P, int N){

    // defining the variables for easy naming
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // declaing the block's scratchpads and the tiling
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // blocks scratchpad
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // Mds and Nds are shared tiles
    // pValue is the accumulator
    // ph walks the phases

    float Pvalue = 0.0f; // the running total

    for(int ph = 0; ph < N / TILE_WIDTH; ++ph){ // which tile am I on currently
        // load, sync, multiply, sync 
        // 1. load my one M element and my one N element into the shared tile
        Mds[ty][tx] = A[row*N + (ph*TILE_WIDTH + tx)];
        Nds[ty][tx] = B[(ph*TILE_WIDTH + ty)*N + col];
        __syncthreads();
        // 3. k-loop: Pvalue += Mds[ty][k] * Nds[k][tx];  ---> the dot product from the tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }
    P[row*N + col] = Pvalue;;

}


// writing the main 

int main(){

    // lets first define the arrays
    int N = 4;
    float arrA[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, *h_A = arrA;
    float arrB[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, *h_B = arrB;
    float arrC[16], *h_C = arrC; // this is the resulting array.
    
    float *d_A, *d_B, *d_C;

    // allocating the memory

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

    cudaMemcpy(d_A, h_A, sizeof(float) * 16, cudaMemcpyHostToDevice); // i am not writing the if statement to validate failure for now 
    cudaMemcpy(d_B, h_B, sizeof(float) * 16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, sizeof(float) * 16, cudaMemcpyHostToDevice);

    
    // define the blocks and threads and then caLL the kernel
    
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(N / TILE_WIDTH, N / TILE_WIDTH);
    optimizedMatmul<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);


}