#include <iostream>
#include <cuda.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

// lets create the kernel

__global__ void AddInts(int *a, int *b, int count){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < count){
        a[id] += b[id];
    }
}

int main(){

    srand(time(NULL));
    int count = 100;
    int *h_a = new int[count];
    int *h_b = new int[count];

    for(int i = 0; i < count; i++){
        h_a[i] = rand() % 1000; 
        h_b[i] = rand() % 1000; 
    }

    cout << "Prior to the addition : " << endl;
    for(int i = 0; i < 5; i++){
        cout << h_a[i] << " " << h_b[i] << endl;
    }

    // making the device copies of the above arrays
    int *d_a, *d_b;

    if(cudaMalloc(&d_a, sizeof(int) * count) != cudaSuccess){
        cout << "d_a : Nopppee";
        return 0;
    }
    if(cudaMalloc(&d_b, sizeof(int) * count) != cudaSuccess){
        cout << "d_b : Nopppee";
        cudaFree(d_a);
        return 0;
    }

    if(cudaMemcpy(d_a, h_a, sizeof(int) * count, cudaMemcpyHostToDevice) != cudaSuccess){
        cout << "Could not copy" << endl;
        cudaFree(d_a);
        cudaFree(d_b);
        return 0;
    }
    if(cudaMemcpy(d_b, h_b, sizeof(int) * count, cudaMemcpyHostToDevice) != cudaSuccess){
        cout << "Could not copy" << endl;
        cudaFree(d_a);
        cudaFree(d_b);
        return 0;
    }

    // now the kernel part
    AddInts<<<count/256+1, 256>>>(d_a, d_b, count);

    if(cudaMemcpy(h_a, d_a, sizeof(int) * count, cudaMemcpyDeviceToHost) != cudaSuccess){
        
        delete[] h_a;
        delete[] h_b;
        cudaFree(d_a);
        cudaFree(d_b);
        cout << "Nope, errrrrrorrr" << endl;
        return 0;
    }

    // if success, then lets print
    for(int i = 0; i<5; i++){
        cout << "Addition for elements at index " << i << " : " << h_a[i] << endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
    return 0;
}

// output : 
// Prior to the addition : 
// 789 311
// 226 409
// 406 561
// 189 400
// 429 452
// Addition for elements at index 0 : 1100
// Addition for elements at index 1 : 635
// Addition for elements at index 2 : 967
// Addition for elements at index 3 : 589
// Addition for elements at index 4 : 881