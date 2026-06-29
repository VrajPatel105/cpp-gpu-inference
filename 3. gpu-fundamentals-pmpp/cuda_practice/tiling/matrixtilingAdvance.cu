// This problem is given by claude. I am going to rederive this whole kernel without any reference of past code.


// //Cold re-derivation roadmap (blank file, no notes)

// The seed. What is one thread responsible for producing? (Everything below depends on this one sentence — write it first.)

// --> one thread is responsibe to hold the product of two numbers. One from matrix A and one from matrix B from same i'th index pos. 

// Identity. What two CUDA built-ins identify a thread, and what's the rule for which family (.x / .y) maps to row vs col? State the rule before you use it.

// --> in cuda, col is horizontal -> x and row is vertical -> y

// Ownership. From the badge, which P element is mine? Write row and col — each is badge × TILE_WIDTH + offset. Get the pairing right without looking.

// -->     int row = by * TILE_WIDTH + ty;
//         int col = bx * TILE_WIDTH + tx;

// Storage. Three declarations. Which get __shared__ and why? Which one is a private register and why? Tie each to its scope.

// --> the __shared__ are the tiles that goes to the block which is accessible to all the threads. 
// the Pvalue will be private register that will be resposible for each private's thread's value

// Phases. Write the loop bound. Then say out loud: a phase is a chunk of ____ (time / space?), and what does one phase accomplish for a single thread?
// Load. Write both load indices flattened. The test: which one carries the per-phase jump on the column, which on the row, and why? (M slides →, N slides ↓.)
// Barrier #1. Why here? What bug does it stop? (Your mailbox / box3 story.)
// Inner k-loop. Write the accumulate line. Why is it a loop and not one statement?
// Barrier #2, then write home. Why a second barrier? What's the final write, flattened?

// Verification: trace thread (row=2, col=1), hand-compute its element, run, check h_C[9].
// Twists (after the clean version works)

// Easy — kill the symmetry. Right now A == B, so a row/col swap bug could hide. Make B different from A and re-verify by hand. If your output still matches, your row/col indexing is genuinely correct, not accidentally.
// Medium — break the square. The book (and your kernel) assumes N×N. Real matmul is A[I×K] · B[K×J] = P[I×J]. You've been using one N for three different dimensions. Which of your indices actually need I, K, J instead? This forces you to see the three dimensions you've been collapsing into one.
// Hard — break the seed. What if each thread computed two P elements instead of one? (This is real — it's called thread coarsening / register tiling, a genuine optimization PMPP hits later.) What changes in the launch, the loop, the accumulators? It deliberately violates "one thread, one element," which is exactly why wrestling with it deepens the rule.


// This will also contain me Adding comments with my understanding and the answres that i got from claude 


#include <iostream>
#include <cuda.h>
using namespace std;

#define TILE_WIDTH 2 // defining the TILE_WIDTH as a constant. We do this to make sure to avoid confusion when initializing the blockDim and gridDim


__global__ void advanceMatrixTiling(float* A, float* B, float* P, int N){ // main kernel start

    // these are just for easy naming and avoiding confusion
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // row and col in cuda is simply : col is horizontal -> x and row is vertical -> y. 
    // This is just initializing the vars. badge X TILE_WIDTH + offset 
    // Question : What is badge here? is it simply the jumps needed to skip a big chunk? meainng skipping the rows (since it's flat indexed array internally?)
    // Answer : 
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    // ths is defining the shared tiles. One for Matrix A and one for Matrix B. 
    // this size is TILE_WIDTH x TILE_WIDTH -> 2x2 meaning at once, 2x2 block will be loaded
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    // private register value for each thread. this is defined outside because each Pvalue hold the value for the output matrix i'th position.
    // We cannot place it inside the loop, if we do, then the entire product of all the numbers will be accumulated, where as, here we only want
    // the product of that specific i'th index 
    float Pvalue = 0.0f;

    // this is the main phase loop. for each different phases. Meaning if theres' one job being done, after that job is executed by all the threads, we syncthreads() and then the next phase starts where the thread reads the other values loaded by other threads from the shared mem
    // How are the values decided? why is it N / TILE_WIDTH? how do we make sure that we understand this concept and apply it anywhere else?
    // Answer : 
    for(int ph = 0; ph < N / TILE_WIDTH; ++ph){

        // This is simply loading those 2x2 tiles values from each matrix A and B and storing it in shared memory for their corresponding shared 
        // memory. i.e, Mds is shared memory for array A and Nds is shared memroy for values from array B 
        // Here, each thread loads their specific value from the global mem to shared mem and then we syncthreads() to make sure that other threads
        // don't try to read other thread's values when the task is done quickly for one thread and the other thread is still working. 
        // Question : How is the flat index decided? how can we make sure that we get this correct no matter the values of diff arrays? 
        // Answer : 
        Mds[ty][tx] = A[row*N + (ph*TILE_WIDTH + tx)];
        Nds[ty][tx] = B[(ph*TILE_WIDTH + ty)*N + col];
        __syncthreads(); // just syncing the threads
        
        // this for loop is what calculates the main Pvalue.
        // Question : what is is actuall'y calculating visually? it would be better if it's visually shown. Cuz this is where the main confusion is
        for(int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads(); // syncing threads
    }
    P[row*N + col] = Pvalue; // writing the final pvalue in the resulting array mem. 
}


// the main part is really easy and straightforward. No need for commenting here : ) 

int main(){

    int N = 4; 
    float arrA[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}, *h_A = arrA;
    float arrB[16] = {1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1}, *h_B = arrB;
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
    advanceMatrixTiling<<<gridDim, blockDim>>>(d_A, d_B, d_result, N);

    // transferring the resuling mem from d_result to h_result (device to host)
    cudaMemcpy(h_result, d_result, sizeof(float) * 16, cudaMemcpyDeviceToHost);

    // pritint the final output
    for(int i = 0; i < 16; ++i){
        cout << "i = " << i << " -> " << h_result[i] << endl;
    }

    return 0;

}

// output : 
// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ nvcc tiling/matrixtilingAdvance.cu -o build/matrixtilingadvance
// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ ./build/matrixtilingadvance
// i = 0 -> 1
// i = 1 -> 2
// i = 2 -> 3
// i = 3 -> 4
// i = 4 -> 5
// i = 5 -> 6
// i = 6 -> 7
// i = 7 -> 8
// i = 8 -> 9
// i = 9 -> 10
// i = 10 -> 11
// i = 11 -> 12
// i = 12 -> 13
// i = 13 -> 14
// i = 14 -> 15
// i = 15 -> 16