#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaDeviceProp prop;
    int dev = 0;

    cudaGetDeviceProperties(&prop, dev);

    std::cout << "Device name: " << prop.name << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
    std::cout << "Registers per block: " << prop.regsPerBlock << "\n";
    std::cout << "Warp size: " << prop.warpSize << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max threads dim: "
              << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", "
              << prop.maxThreadsDim[2] << "\n";
    std::cout << "Max grid size: "
              << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", "
              << prop.maxGridSize[2] << "\n";

    return 0;
}

// Device name: NVIDIA GeForce RTX 4070 Laptop GPU
// Global memory: 8187 MB
// Shared memory per block: 49152 bytes
// Registers per block: 65536
// Warp size: 32
// Max threads per block: 1024
// Max threads dim: 1024, 1024, 64
// Max grid size: 2147483647, 65535, 65535