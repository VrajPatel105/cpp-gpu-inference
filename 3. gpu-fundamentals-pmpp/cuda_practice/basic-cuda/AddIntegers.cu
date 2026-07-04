/*
Steps that we will be following for this code file : 

1. Cerate two integers for host
2. Allocate memory for copies of them on the device
3. Copy the integers to the device's memory
4. Call the kernel to add them together
5. Copy the result back to host memory
6. Print out the result the GPU computed
7. Free the device's memory we allocated

*/

#include <iostream>
#include <cuda.h>

using namespace std;

// writing the kernel
__global__ void AddInt(int* a, int* b){
    a[0] += b[0];
}

int main(){

    int a = 5;
    int b = 10;

    int *d_a, *d_b; // device pointers for a and b 

    cudaMalloc(&d_a, sizeof(int));
    cudaMalloc(&d_b, sizeof(int)); // we should also have a if block here and check cudaSuccess == True

    cudaMemcpy(d_a, &a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b, sizeof(int), cudaMemcpyHostToDevice);

    AddInt<<<1,1>>>(d_a, d_b); // syntax : kernel<<< blocks, threads>>>(Parameters)
    
    cudaMemcpy(&a, d_a, sizeof(int), cudaMemcpyDeviceToHost);

    cout << "The answer is : " << a << endl;

    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}


// the output for the code : 

// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ nvcc AddIntegers.cu -o build/AddIntegers
// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ ./build/AddIntegers
// The answer is : 15