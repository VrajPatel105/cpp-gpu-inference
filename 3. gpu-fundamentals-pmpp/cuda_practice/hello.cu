#include <stdio.h>

__global__ void helloFromGPU() {
    printf("Hello from GPU! Thread %d in Block %d\n", 
           threadIdx.x, blockIdx.x);
}

int main() {
    // Launch 2 blocks of 4 threads each
    helloFromGPU<<<2, 4>>>();
    cudaDeviceSynchronize();
    printf("Hello from CPU!\n");
    cudaDeviceReset();  // forces printf buffer flus
    return 0;
}