/*
I am writing this custom kernel that basically returns the square of every element in the array.
This is me writing it without watching or referencing of any video, just from only what i have learned from my own.
*/

/*
Custom Square Element Kernel:
1. Define the kernel
2. initialize the two arrays / or can also take one array and write directly to it. on host
3. mem copy to device
4. kernel does it's job
5. copy the array back to host from device
6. print before and after. (if taking one array, then print it before kernel call)
*/

#include <cuda.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

// kernel

__global__ void squareKernel(int *a, int *b, int len){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    b[id] = a[id] * a[id];
}

int main(){

    int len = 10;

    int arr[] = {1,2,3,4,5,6,7,8,9,10};

    int *h_arr = arr;
    int *h_result_arr = new int[len];

    int *d_arr, *d_result_arr;

    cudaMalloc(&d_arr, sizeof(int) * len);
    cudaMalloc(&d_result_arr, sizeof(int) * len);

    cudaMemcpy(d_arr, h_arr, len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result_arr, h_result_arr, len * sizeof(int), cudaMemcpyHostToDevice);

    squareKernel<<<1,len>>>(d_arr, d_result_arr, len);

    cudaMemcpy(h_result_arr, d_result_arr, len * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i<len; i++){
        cout << h_result_arr[i] << endl;
    }

    return 0;
}

// output : 
// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp$ cd cuda_practice/
// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ ls
// AddIntegers.cu  README.md  build  gpu-memory-README.md  hello.cu  images  main.cu  pc_stats.txt  squareElement.cu
// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ nvcc squareElement.cu -o build/square
// vrajpatel@Vraj:/mnt/c/My Projects/Deep Learning Projects/cpp-gpu-inference/3. gpu-fundamentals-pmpp/cuda_practice$ ./build/square
// 1
// 4
// 9
// 16
// 25
// 36
// 49
// 64
// 81
// 100