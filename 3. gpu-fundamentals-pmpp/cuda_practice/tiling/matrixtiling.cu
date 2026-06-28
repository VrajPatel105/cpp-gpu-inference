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



// writing the main kernel


// writing the main 

int main(){

    // lets first define the arrays
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


}